from collections.abc import Sequence
import dataclasses
import logging
import pathlib
import time
from typing import Any, Literal, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


@dataclasses.dataclass(frozen=True)
class AccelFeedbackConfig:
    """Opt-in action-layer response to a high Accel score.

    Accel remains a detached readout.  This configuration only decides what a
    caller does *after* the readout crosses a threshold; it never contributes
    to the FM loss or gradient path.
    """

    strategy: Literal["none", "replan", "shorten", "more_denoise", "backup", "adaptive"] = "none"
    threshold: float | None = None
    max_replans: int = 1
    shorten_to: int | None = None
    extra_num_steps: int = 10
    backup_policy: BasePolicy | None = None
    # ``adaptive`` (Accel-guided adaptive replanning) turns the score into an
    # execution horizon instead of a binary gate: ``<= threshold_low`` commits
    # ``n_low`` actions, ``<= threshold_high`` commits ``n_mid``, anything above
    # commits ``n_high`` and re-observes almost immediately.  See
    # :func:`select_execution_steps`.
    threshold_low: float | None = None
    threshold_high: float | None = None
    n_low: int = 5
    n_mid: int = 3
    n_high: int = 1

    def __post_init__(self) -> None:
        if self.strategy not in {"none", "replan", "shorten", "more_denoise", "backup", "adaptive"}:
            raise ValueError(f"unknown Accel feedback strategy: {self.strategy!r}")
        if self.max_replans < 1:
            raise ValueError("max_replans must be positive")
        if self.strategy == "shorten" and (self.shorten_to is None or self.shorten_to < 1):
            raise ValueError("shorten strategy requires shorten_to >= 1")
        if self.strategy == "more_denoise" and self.extra_num_steps < 1:
            raise ValueError("extra_num_steps must be positive")
        if self.strategy == "backup" and self.backup_policy is None:
            raise ValueError("backup strategy requires backup_policy")
        if self.strategy == "adaptive":
            if self.threshold_low is None or self.threshold_high is None:
                raise ValueError("adaptive strategy requires both threshold_low and threshold_high")
            if not self.threshold_low < self.threshold_high:
                raise ValueError("adaptive strategy requires threshold_low < threshold_high")
            if min(self.n_low, self.n_mid, self.n_high) < 1:
                raise ValueError("adaptive strategy requires every horizon to be >= 1")


def _full_horizon_table(
    denoise_path: np.ndarray,
    *,
    action_dim: int,
    horizon: int,
    fixed_std: np.ndarray | None,
) -> np.ndarray | None:
    """The whole-chunk prefix table, or ``None`` when the scale cannot score it.

    A reference scale frozen on an ``n_exec``-wide window has ``n_exec *
    action_dim`` entries, but the whole chunk flattens to ``horizon *
    action_dim``. Handing the window scale to the wider computation raises, so a
    run configured with ``n_exec < action_horizon`` and a frozen scale simply
    goes without the wider table rather than failing every request. Nothing
    downstream silently substitutes a different scale: the field is absent.
    """

    if fixed_std is not None and int(np.asarray(fixed_std).shape[0]) != horizon * action_dim:
        return None
    # Imported here for the same reason as in infer_with_accel: the uncertainty
    # package stays out of the policy's import graph unless it is actually used.
    from openpi.uncertainty.accel import score_prefix_table

    table = score_prefix_table(denoise_path, action_dim=action_dim, fixed_std=fixed_std)
    return np.asarray(table[0], dtype=np.float32)


def _batch_observation(observations: dict, *, is_pytorch: bool, device: str) -> dict:
    """Convert an already-batched observation dict into the model's expected form.

    Same conversion ``infer``/``infer_with_accel`` perform, so a replayed
    observation reaches the sampler through the identical path.
    """

    def convert(value):
        if isinstance(value, dict):
            return {name: convert(array) for name, array in value.items()}
        if is_pytorch:
            return torch.from_numpy(np.asarray(value)).to(device)
        return jnp.asarray(value)

    return {key: convert(value) for key, value in observations.items()}


def select_execution_steps(score: float, feedback: AccelFeedbackConfig) -> int:
    """Map an Accel score to an execution horizon (Accel-guided replanning).

    Implements the plain-Python rule directly::

        if accel <= threshold_low:
            execute_steps = n_low
        elif accel <= threshold_high:
            execute_steps = n_mid
        else:
            execute_steps = n_high

    A low-curvature denoising path is treated as a confident chunk and commits
    the full window; a high-curvature path commits a single action and forces an
    immediate re-observation.  This is a pure function of a score the sampler
    already produced, so no extra Action Expert forward is performed.
    """
    if score <= feedback.threshold_low:  # type: ignore[operator]
        return int(feedback.n_low)
    if score <= feedback.threshold_high:  # type: ignore[operator]
        return int(feedback.n_mid)
    return int(feedback.n_high)


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
            self._sample_actions_with_trajectory = getattr(model, "sample_actions_with_trajectory", None)
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            # The trajectory sampler has a shape-dependent ``num_steps`` output,
            # so keep that argument static to avoid recompiling a dynamic-shaped
            # JAX result.  The regular sampler above is intentionally untouched.
            trajectory_method = getattr(type(model), "sample_actions_with_trajectory", None)
            self._sample_actions_with_trajectory = (
                nnx_utils.module_jit(model.sample_actions_with_trajectory, static_argnames=("num_steps",))
                if trajectory_method is not None
                and trajectory_method is not _model.BaseModel.sample_actions_with_trajectory
                else None
            )
            # JAX typed keys do not have boolean truth-value semantics; use an
            # explicit None check so callers can reproduce an inference stream.
            self._rng = jax.random.key(0) if rng is None else rng

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    def infer_with_accel(
        self,
        obs: dict,
        *,
        noise: np.ndarray | None = None,
        num_steps: int | None = None,
        action_dim: int | None = None,
        n_exec: int | None = None,
        mode: str | None = None,
        fixed_std: np.ndarray | None = None,
        accel_monitor: Any | None = None,
        capture_context: bool = False,
    ) -> dict:
        """Infer actions and return an Accel uncertainty readout.

        This opt-in API is the OpenPI counterpart of ``fm-geometry``'s
        recording + ``AccelDetector`` path.  It runs the same π₀.₅ FM sampler,
        but retains ``x_t``/``v_t`` so :func:`openpi.uncertainty.accel.score_chunks`
        can score the model-space denoising path.  The normal :meth:`infer`
        path remains unchanged and incurs no trajectory allocation.

        ``mode`` defaults to :data:`openpi.uncertainty.accel.DEFAULT_ACCEL_MODE`.
        Mind the depth convention: ``accel_prefix:<j>`` scores the first
        ``j + 2`` denoise steps, so the default ``accel_prefix:3`` reads
        ``accel_5``, not ``accel_3``.

        ``n_exec`` is the executed window and defaults to the model's full
        action horizon.  The executed-window modes (``accel_exec``,
        ``accel_prefix:<j>``) require it: without a window they silently widen
        to the whole chunk, which is a different statistic.  ``fixed_std``
        should be a reference scale of length ``n_exec * action_dim`` for those
        modes and ``action_horizon * action_dim`` for the whole-horizon ones.

        For padded model outputs (for example, OpenPI LIBERO uses a
        32-dimensional FM head for a 7-dimensional environment action), pass the
        real ``action_dim`` so padding does not dilute the score; the model
        dimension is only a fallback, and it is the wrong scale for a
        calibration fitted on the real one.  Leaving ``fixed_std`` unset uses
        the per-call standardisation from ``fm-geometry``, which self-normalises
        each chunk and does not reproduce the validated AUROC; for comparable
        online scores, pass a fixed scale (see
        :class:`openpi.uncertainty.accel.ReferenceStd`).
        """
        if self._sample_actions_with_trajectory is None:
            raise NotImplementedError(
                "this policy's model does not expose sample_actions_with_trajectory; "
                "Accel is currently available for Pi0/π₀.₅ only"
            )

        from openpi.uncertainty.accel import DEFAULT_ACCEL_MODE
        from openpi.uncertainty.accel import parse_accel_mode
        from openpi.uncertainty.accel import prefix_index_for_mode
        from openpi.uncertainty.accel import score_chunks
        from openpi.uncertainty.accel import score_prefix_table

        mode = DEFAULT_ACCEL_MODE if mode is None else mode
        # Resolve the executed window before scoring.  The executed-window modes
        # require it, and defaulting to the full horizon keeps a bare
        # infer_with_accel() call working while still passing an explicit window
        # down to score_chunks.
        horizon = getattr(self._model, "action_horizon", None)
        if n_exec is None:
            n_exec = int(horizon) if horizon is not None else None
        elif horizon is not None and n_exec > int(horizon):
            raise ValueError(f"n_exec={n_exec} exceeds the model's action_horizon={horizon}")

        # Keep preprocessing and output transforms identical to infer().
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        # Captured here -- after the transforms, before batching -- so images are
        # still in the dtype the transform emitted and the offline rebuild can
        # feed ``Observation.from_dict`` exactly what the live path fed it. No
        # extra forward pass: this is a copy of arrays that already exist.
        captured_context = None
        if capture_context:
            from openpi.uncertainty.context import capture_context as _capture

            captured_context = _capture(inputs)
        if not self._is_pytorch_model:
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_device = jax.random.split(self._rng)
        else:
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_device = self._pytorch_device

        sample_kwargs = dict(self._sample_kwargs)
        if num_steps is not None:
            sample_kwargs["num_steps"] = int(num_steps)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)
            if noise.ndim == 2:
                noise = noise[None, ...]
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        trajectory = self._sample_actions_with_trajectory(sample_rng_or_device, observation, **sample_kwargs)
        model_time = time.monotonic() - start_time
        actions = trajectory["actions"]

        if self._is_pytorch_model:
            outputs = {
                "state": inputs["state"],
                "actions": actions,
            }
            trajectory_np = {
                key: np.asarray(value.detach().cpu()) if isinstance(value, torch.Tensor) else np.asarray(value)
                for key, value in trajectory.items()
            }
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = {"state": inputs["state"], "actions": actions}
            trajectory_np = {key: np.asarray(value) for key, value in trajectory.items()}
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        # Accel is measured before output transforms, matching fm-geometry's
        # model-space convention.  The user-facing actions still go through the
        # exact same unnormalisation/data transforms as regular inference.
        outputs = self._output_transform(outputs)
        if n_exec is not None and n_exec > trajectory_np["x_t"].shape[2]:
            # numpy would silently accept the oversized slice and score a
            # shorter window than the caller asked for.
            raise ValueError(f"n_exec={n_exec} exceeds the sampled action horizon {trajectory_np['x_t'].shape[2]}")
        if action_dim is None:
            action_dim = getattr(self._model, "action_dim", None)
            if action_dim is None:
                action_dim = actions.shape[-1]
            logging.warning(
                "infer_with_accel: action_dim not given, falling back to the model dimension %d. If the FM head "
                "is padded (LIBERO scores 7 of 32 dims), the zero-pad dimensions dilute the norm and the score "
                "will not be on a calibrated scale.",
                int(action_dim),
            )
        denoise_path = trajectory_np["x_t"]
        # For a prefix mode we compute the (batch, T-1) table once and read the
        # requested column out of it, so the served scalar and the recorded table
        # are the same statistic by construction. Recomputing later instead would
        # risk scoring against a reference scale that had just been re-frozen,
        # and would double the cost of every request.
        prefix_base, _prefix_depth = parse_accel_mode(mode)
        if prefix_base in {"accel_prefix", "accel_prefix_full"}:
            table = score_prefix_table(
                denoise_path,
                action_dim=int(action_dim),
                n_exec=None if prefix_base == "accel_prefix_full" else n_exec,
                fixed_std=fixed_std,
            )
            scores = table[:, prefix_index_for_mode(mode, table.shape[1])]
            outputs["accel_prefix_table"] = np.asarray(table[0], dtype=np.float32)
            # The whole-horizon table alongside the windowed one: the two differ
            # (fewer flattened coordinates) and conflating them offline would be
            # a silent scale swap, so both go into the record. They are the same
            # computation when the window already covers the chunk, so the second
            # pass is skipped rather than paid on every one of ~130k chunks.
            horizon = int(denoise_path.shape[2])
            if prefix_base == "accel_prefix_full" or n_exec == horizon:
                # Same window, so the second pass would be a duplicate of the
                # first rather than a different statistic.
                outputs["accel_prefix_table_full"] = outputs["accel_prefix_table"]
            else:
                outputs["accel_prefix_table_full"] = _full_horizon_table(
                    denoise_path,
                    action_dim=int(action_dim),
                    horizon=horizon,
                    fixed_std=fixed_std,
                )
        else:
            scores = score_chunks(
                denoise_path,
                action_dim=int(action_dim),
                n_exec=n_exec,
                mode=mode,
                fixed_std=fixed_std,
            )
            outputs["accel_prefix_table"] = None
            outputs["accel_prefix_table_full"] = None
        outputs["accel"] = float(scores[0]) if scores.shape[0] == 1 else scores
        if accel_monitor is not None:
            if scores.shape[0] != 1:
                raise ValueError("accel_monitor is stateful and requires a single-observation inference call")
            outputs["accel_monitor"] = accel_monitor.update(float(scores[0]))
        outputs["fm_trajectory"] = {
            "x_t": trajectory_np["x_t"][0] if trajectory_np["x_t"].shape[0] == 1 else trajectory_np["x_t"],
            "v_t": trajectory_np["v_t"][0] if trajectory_np["v_t"].shape[0] == 1 else trajectory_np["v_t"],
            "time": trajectory_np["time"],
        }
        # Large, so the server pops it alongside fm_trajectory rather than sending
        # it to the rollout client. Absent (not None) when not requested, so a
        # consumer cannot mistake "not captured" for "captured and empty".
        if captured_context is not None:
            outputs["fm_context"] = captured_context
        outputs["policy_timing"] = {"infer_ms": model_time * 1000}
        return outputs

    def infer_with_accel_feedback(
        self,
        obs: dict,
        *,
        feedback: AccelFeedbackConfig,
        accel_monitor: Any | None = None,
        **kwargs: Any,
    ) -> dict:
        """Infer with a detached Accel gate and an explicit response strategy.

        The first call always follows the normal π₀.₅ FM sampler and produces
        the candidate action chunk.  If ``accel_monitor`` reports an alarm or
        the returned score exceeds ``feedback.threshold``, the selected
        application-layer strategy is applied:

        * ``none``: keep the first chunk;
        * ``replan``: resample the same observation up to ``max_replans`` times;
        * ``shorten``: return only the first ``shorten_to`` actions;
        * ``more_denoise``: resample with ``extra_num_steps`` additional FM
          steps;
        * ``backup``: delegate action generation to an injected policy.

        No strategy creates a target, reward, or gradient for π₀.₅.  The
        returned ``feedback`` field records the decision while the action
        remains the only field consumed by existing rollout clients.
        """
        result = self.infer_with_accel(obs, accel_monitor=accel_monitor, **kwargs)
        monitor_info = result.get("accel_monitor")
        score = float(monitor_info["score"] if monitor_info is not None else result["accel"])
        if feedback.strategy not in {"none", "adaptive"} and monitor_info is None and feedback.threshold is None:
            raise ValueError("feedback requires an AccelMonitor or an explicit threshold")
        alarm = (
            bool(monitor_info["alarm"])
            if monitor_info is not None
            else (feedback.threshold is not None and score > feedback.threshold)
        )
        decision: dict[str, Any] = {
            "strategy": feedback.strategy,
            "triggered": alarm,
            "score": score,
            "attempts": 0,
            "execute_steps": None,
        }
        if monitor_info is not None:
            # Surfaced so a caller can compute lead/onset offline without
            # re-deriving the CUSUM from the scores.
            decision["cusum"] = float(monitor_info["cusum"])
            decision["index"] = int(monitor_info["index"])
            decision["onset"] = int(monitor_info["onset"])

        if feedback.strategy == "adaptive":
            # Accel-guided adaptive replanning.  The score selects the
            # commitment length instead of acting as a binary alarm, so this
            # branch runs on every observation rather than only after an alarm.
            # The trajectory scored here is the one the sampler already
            # produced for this observation: no extra Action Expert forward.
            steps = select_execution_steps(score, feedback)
            steps = max(1, min(steps, result["actions"].shape[0]))
            result["actions"] = result["actions"][:steps]
            decision["execute_steps"] = steps
            # ``triggered`` means "committed less than the full window", which
            # keeps the existing rollout clients' semantics for a shortened
            # commitment while making the baseline branch (n_low) a no-op.
            decision["triggered"] = steps < feedback.n_low
            result["feedback"] = decision
            return result

        if not alarm or feedback.strategy == "none":
            result["feedback"] = decision
            return result

        if feedback.strategy == "replan":
            for attempt in range(1, feedback.max_replans + 1):
                # Do not update a stateful monitor for replacement samples;
                # one observation should contribute one causal decision.
                result = self.infer_with_accel(obs, **kwargs)
                decision["attempts"] = attempt
                if feedback.threshold is None or float(result["accel"]) <= feedback.threshold:
                    decision["triggered"] = False
                    break
        elif feedback.strategy == "shorten":
            result["actions"] = result["actions"][: feedback.shorten_to]
            decision["attempts"] = 1
            decision["execute_steps"] = len(result["actions"])
        elif feedback.strategy == "more_denoise":
            base_steps = kwargs.get("num_steps", self._sample_kwargs.get("num_steps", 10))
            resample_kwargs = dict(kwargs)
            resample_kwargs["num_steps"] = int(base_steps) + feedback.extra_num_steps
            result = self.infer_with_accel(obs, **resample_kwargs)
            decision["attempts"] = 1
        elif feedback.strategy == "backup":
            result = feedback.backup_policy.infer(obs)  # type: ignore[union-attr]
            decision["attempts"] = 1
            result["accel"] = score

        result["feedback"] = decision
        return result

    def sample_trajectory(
        self,
        observations: dict,
        *,
        noise: np.ndarray | None = None,
        num_steps: int | None = None,
    ) -> dict:
        """Run the FM trajectory sampler on an already-transformed observation batch.

        The entry point the offline resample stage uses. It takes the *batched*
        dict that :func:`openpi.uncertainty.context.load_context_dict` rebuilds
        from a capture -- past the input transforms, exactly where
        :meth:`infer_with_accel` hands the observation to the model -- so a
        replay runs the identical code path and the fidelity gate can prove the
        reconstruction was faithful rather than assume it.

        ``noise`` is ``(B, action_horizon, model_action_dim)``; pass ``B = k`` to
        draw ``k`` independent candidates in one batched pass, which reuses the
        prefix cache instead of paying for it ``k`` times.
        """

        if self._sample_actions_with_trajectory is None:
            raise NotImplementedError(
                "this policy's model does not expose sample_actions_with_trajectory; "
                "the resample ground truth is available for Pi0/pi0.5 only"
            )
        inputs = _batch_observation(observations, is_pytorch=self._is_pytorch_model, device=self._pytorch_device)
        sample_kwargs = dict(self._sample_kwargs)
        if num_steps is not None:
            sample_kwargs["num_steps"] = int(num_steps)
        if noise is not None:
            if self._is_pytorch_model:
                noise = torch.from_numpy(np.asarray(noise)).to(self._pytorch_device)
            else:
                noise = jnp.asarray(noise)
            if noise.ndim == 2:
                noise = noise[None, ...]
            sample_kwargs["noise"] = noise

        # On the JAX path, deliberately not consuming self._rng: a replay supplies
        # its own noise, and advancing the live stream would make an offline
        # analysis perturb the policy it is analysing. On the PyTorch path the
        # sampler takes a device instead of a key.
        sample_rng_or_device = self._pytorch_device if self._is_pytorch_model else jax.random.split(self._rng)[1]
        observation = _model.Observation.from_dict(inputs)
        return self._sample_actions_with_trajectory(sample_rng_or_device, observation, **sample_kwargs)

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results

    def infer_with_accel(self, obs: dict, **kwargs: Any) -> dict:
        """Reject Accel recording explicitly rather than looking like a missing policy method.

        The record path flattens ``inputs``/``outputs`` into a single ``np.save``
        array.  An Accel result additionally carries a full ``(T+1, horizon,
        action_dim)`` denoising trajectory, which is both too large to record
        that way and never consumed by a rollout client.
        """

        del obs, kwargs
        raise NotImplementedError(
            "PolicyRecorder does not support Accel inference: recording a denoising trajectory alongside the "
            "observation is not implemented. Run --record and --accel in separate processes, or use "
            "--accel-record for the Accel side."
        )
