"""The resample ground truth: replay a decision and measure how far it scatters.

``accel`` is a claim about the *posterior* over action chunks -- that the bend in
the denoising path equals how uncertain the policy is about where to go. The only
way to check a claim about a posterior is to sample it: fix the observation the
policy actually saw, redraw the starting noise, and see whether the resulting
chunks land on top of each other or fan out. That fan-out is ``D_resample``.

Two things make this honest rather than circular:

* **The fidelity gate.** A replay is only evidence if the observation was
  reconstructed exactly, so ``reproduce`` feeds the *recorded* noise back through
  the *same* sampler and requires the *recorded* chunk to come out. Resampling at
  a drifted observation would measure a posterior the policy never faced, and
  nothing downstream could detect it.
* **The spread is a max over pairs, not a variance.** Multimodality is the thing
  being measured, and a variance averages two well-separated modes into something
  that looks confident. The maximum pairwise distance is the statistic that does
  not.

This module holds no policy: the sampler arrives as a callable, so the gate and
the spread are testable against a stub and the analysis can run on a machine that
never loads weights.
"""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

EPS = 1e-12
# Same tolerance fm-geometry uses: tight enough that a drifted observation fails,
# loose enough to absorb bfloat16 replay noise on a different attention path.
REPRODUCE_TOL = 5e-2


def pooled_dim_scale(samples: Any, *, eps: float = EPS, clip_pct: float = 1.0) -> np.ndarray:
    """Per-action-dimension winsorized std, pooled over the whole run.

    One fixed vector for the run, so distances are comparable across episodes
    rather than being re-scaled per episode -- which is what lets the pooled rank
    correlation mean anything. Winsorizing (clipping the extreme ``clip_pct``
    percentiles before taking the std) keeps the occasional catastrophically
    diverged rollout from dominating a plain std; IQR is not used because it
    reweights heavy-tailed dims that are perfectly healthy.
    """

    array = np.asarray(samples, dtype=np.float32)
    flat = array.reshape(-1, array.shape[-1])
    low, high = np.percentile(flat, [clip_pct, 100.0 - clip_pct], axis=0)
    std = np.clip(flat, low, high).std(axis=0)
    # A near-constant dimension carries no scale information; leave it unscaled
    # rather than dividing by ~0.
    return np.where(std < eps, 1.0, std).astype(np.float32)


def pairwise_spread(
    candidates: Any,
    scale: np.ndarray | None = None,
) -> tuple[float, float]:
    """``(max, mean)`` over candidate pairs of the chunk-mean L2 distance.

    ``candidates`` is ``(k, chunk, act)``. A pair's distance is the mean over
    aligned action positions of the per-position L2, with each action dimension
    divided by ``scale`` first so a large-scale dimension cannot dominate. The
    **max** is the headline: the two candidates that disagree most are the
    evidence of multimodality, and averaging over pairs would dilute exactly the
    signal being measured.
    """

    chunks = np.asarray(candidates, dtype=np.float32)
    if chunks.ndim != 3:
        raise ValueError(f"candidates must be (k, chunk, act); got {chunks.shape}")
    k = chunks.shape[0]
    if k < 2:
        return 0.0, 0.0
    diff = chunks[:, None] - chunks[None, :]
    if scale is not None:
        diff = diff / np.asarray(scale, dtype=np.float32)
    per_action = np.linalg.norm(diff, axis=-1)
    upper = np.triu_indices(k, 1)
    per_pair = per_action[upper[0], upper[1], :].mean(axis=-1)
    return float(per_pair.max()), float(per_pair.mean())


def repeat_observation(observation: dict, k: int) -> dict:
    """Tile a B=1 observation to B=``k``.

    The sampler takes its batch size from the observation, not from the noise, so
    a ``k``-candidate pass needs the observation repeated. Broadcast views are
    enough -- the model never writes to them -- so this costs no memory worth
    mentioning. Note the prefix is recomputed inside that batched call rather than
    shared across candidates; sharing it would need the key/value cache hoisted
    out of the sampler, which is a different (and separately verifiable) change.
    """

    out: dict[str, Any] = {}
    for key, value in observation.items():
        if isinstance(value, dict):
            out[key] = {name: np.broadcast_to(array, (k, *array.shape[1:])) for name, array in value.items()}
        else:
            array = np.asarray(value)
            out[key] = np.broadcast_to(array, (k, *array.shape[1:]))
    return out


@dataclasses.dataclass
class ResampleSession:
    """Replays recorded decisions to produce the divergence ground truth.

    ``sample_trajectory`` is called as
    ``sample_trajectory(observation, noise=(B, action_horizon, model_action_dim),
    num_steps=T)`` and must return the denoise path, ``(B, T+1, action_horizon,
    model_action_dim)``. ``Policy.sample_trajectory`` is the intended
    implementation; a stub is enough for tests.
    """

    sample_trajectory: Callable[..., Any]
    action_dim: int
    n_exec: int
    num_steps: int
    action_horizon: int
    model_action_dim: int
    reproduce_tol: float = REPRODUCE_TOL

    @classmethod
    def for_policy(cls, policy: Any, *, action_dim: int, n_exec: int, num_steps: int) -> ResampleSession:
        """Build a session bound to a loaded policy."""

        horizon = int(getattr(policy._model, "action_horizon", 0))  # noqa: SLF001
        model_dim = int(getattr(policy._model, "action_dim", 0))  # noqa: SLF001
        if not horizon or not model_dim:
            raise ValueError("the policy's model does not report action_horizon / action_dim")
        return cls(
            sample_trajectory=policy.sample_trajectory,
            action_dim=int(action_dim),
            n_exec=int(n_exec),
            num_steps=int(num_steps),
            action_horizon=horizon,
            model_action_dim=model_dim,
        )

    def _noise_shape(self) -> tuple[int, int]:
        return self.action_horizon, self.model_action_dim

    def reproduce(self, observation: dict, noise: Any, chunk: Any) -> dict:
        """Replay with the recorded noise; return how far the chunk moved.

        ``max_abs_err`` near zero is the only evidence that the reconstruction
        was faithful. Anything large means the observation drifted and every
        divergence computed from it is measuring a posterior the policy never
        faced.
        """

        recorded = np.asarray(noise)
        if recorded.shape != self._noise_shape():
            raise ValueError(f"recorded noise has shape {recorded.shape}, expected {self._noise_shape()}")
        # B=1 deliberately: the recording was produced at B=1, and a batched
        # replay can differ in the last bits, which would blur the one comparison
        # whose whole job is to be exact.
        trajectory = self.sample_trajectory(observation, noise=recorded[None, ...], num_steps=self.num_steps)
        path = np.asarray(trajectory["x_t"] if isinstance(trajectory, dict) else trajectory)
        replayed = path[0, -1]
        expected = np.asarray(chunk)
        if replayed.shape != expected.shape:
            raise ValueError(f"replayed chunk has shape {replayed.shape}, expected {expected.shape}")
        difference = np.abs(replayed - expected)
        return {
            "chunk": replayed,
            "max_abs_err": float(difference.max()),
            "mean_abs_err": float(difference.mean()),
            "ok": bool(difference.max() <= self.reproduce_tol),
        }

    def resample(self, observation: dict, k: int, *, seed: int) -> np.ndarray:
        """Draw ``k`` fresh chunks at this observation; returns ``(k, chunk, act)``.

        The candidates go through in ONE batched pass, which is much faster than
        ``k`` calls through the same weights -- but it is a real difference from
        the recording (B=k rather than B=1), which is why the fidelity gate is
        left at B=1. A few last bits of batch-shape noise are irrelevant to how
        far candidates scatter, and fatal to a test whose whole job is exactness.
        """

        if int(k) < 2:
            raise ValueError(f"k must be >= 2 to measure a spread; got {k}")
        horizon, dim = self._noise_shape()
        noise = np.random.default_rng(int(seed)).normal(size=(int(k), horizon, dim)).astype(np.float32)
        trajectory = self.sample_trajectory(
            repeat_observation(observation, int(k)), noise=noise, num_steps=self.num_steps
        )
        path = np.asarray(trajectory["x_t"] if isinstance(trajectory, dict) else trajectory)
        if path.shape[0] != int(k):
            raise ValueError(f"sampler returned {path.shape[0]} trajectories for k={k}")
        return path[:, -1, : self.n_exec, : self.action_dim].astype(np.float32)


__all__ = [
    "REPRODUCE_TOL",
    "ResampleSession",
    "pairwise_spread",
    "pooled_dim_scale",
]
