"""Denoising acceleration (``accel``) for OpenPI flow-matching policies.

This is the numpy-only part of the ``fm-geometry`` implementation, adapted to
OpenPI's trajectory convention.  A trajectory is represented as
``(batch, T + 1, action_horizon, action_dim)`` and starts at Gaussian noise at
``t=1`` and ends at the denoised action at ``t=0``.  The estimator reads the
curvature of that path; it does not perform another model forward pass.

The public functions deliberately accept numpy-like arrays (including JAX
arrays after materialisation) and return numpy arrays.  This keeps the metric
usable from evaluation scripts, logs, and CPU-only post-processing.
"""

from __future__ import annotations

import dataclasses
import logging
import pathlib
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

EPS = 1e-12

# ``accel_prefix:<j>`` indexes the ``(T-1,)`` table returned by
# :func:`whole_chunk_prefix_accel`: column ``j`` scores the first ``j + 2``
# denoise steps (see :func:`prefix_depth`), so the ``j`` in the name is NOT the
# depth.  ``accel_prefix:3`` reads a depth of p = 5.
#
# Upstream ``fm-geometry`` ships ``accel_prefix:7`` (p = 9) as its detector
# default, but the depth with measured evidence behind it is the curvature/signal
# argmax at ``p* / T = 5 / 10`` (rho 0.541 -> 0.792) over a T = 10 sampler, which
# is ``accel_prefix:3``.  Scores from different depths are different statistics,
# so a calibration artifact only transfers at the depth it was fitted on.
DEFAULT_ACCEL_MODE = "accel_prefix:3"


@dataclasses.dataclass(frozen=True)
class AccelConfig:
    """The score identity: everything that determines an Accel value.

    Two runs produce the same score for the same trajectory only when ``mode``,
    ``action_dim``, ``n_exec`` and ``num_steps`` all agree.  ``num_steps``
    matters even for a fixed ``mode`` because a different Euler schedule is a
    different denoise path.  ``fixed_std`` is excluded from the identity: it is
    an external reference scale checked separately (by length and digest), since
    it is an array rather than a scalar knob.

    ``action_dim`` is the model-space dimension to score.  ``n_exec`` limits
    the chunk to the actions that will actually be executed; ``accel_exec`` and
    ``accel_prefix`` require it, and the full horizon is only used by the
    ``accel`` / ``accel_prefix_full`` variants.
    """

    action_dim: int
    n_exec: int | None = None
    mode: str = DEFAULT_ACCEL_MODE
    num_steps: int = 10
    fixed_std: np.ndarray | None = None

    def to_artifact(self) -> dict[str, Any]:
        """Return the JSON-serialisable provenance of this score identity."""

        return {
            "mode": self.mode,
            "action_dim": int(self.action_dim),
            "n_exec": None if self.n_exec is None else int(self.n_exec),
            "num_steps": int(self.num_steps),
            "n_prefix_layers": prefix_layers(int(self.num_steps)),
        }

    @classmethod
    def from_artifact(cls, payload: dict[str, Any]) -> AccelConfig:
        """Rebuild the score identity from :meth:`to_artifact` output."""

        return cls(
            action_dim=int(payload["action_dim"]),
            n_exec=None if payload.get("n_exec") is None else int(payload["n_exec"]),
            mode=str(payload["mode"]),
            num_steps=int(payload.get("num_steps", 10)),
        )

    def mismatches(self, other: AccelConfig) -> list[str]:
        """Return the score-identity fields on which ``other`` disagrees."""

        mine, theirs = self.to_artifact(), other.to_artifact()
        return [
            f"{key}: calibrated {mine[key]!r} != live {theirs[key]!r}"
            for key in ("mode", "action_dim", "n_exec", "num_steps")
            if mine[key] != theirs[key]
        ]


def prefix_layers(num_steps: int) -> int:
    """Number of denoise depths ``whole_chunk_prefix_accel`` returns for a sampler.

    A ``T = num_steps`` Euler path has ``T`` velocity increments, so the growing
    prefixes it can score are of length ``2 .. T`` -- that is ``T - 1`` columns.
    """

    return max(1, int(num_steps) - 1)


def parse_accel_mode(mode: str) -> tuple[str, int | None]:
    """Split ``"accel_prefix:3"`` into ``("accel_prefix", 3)``."""

    base, separator, column = mode.partition(":")
    if not separator:
        return base, None
    return base, int(column)


def prefix_depth(index: int, n_prefixes: int) -> int:
    """Denoise depth ``p`` scored by column ``index`` of the prefix table.

    ``p = index + 2`` after wrapping, which is the ``accel_p`` formula
    ``p * sum_{t=1}^{p-1} ||v_t - v_{t-1}|| / sum_{t=0}^{p-1} ||v_t||``.  Column
    ``-1`` is therefore the full path (``p = T``), matching upstream's
    ``accel_prefix:-1 == accel_exec`` equivalence.
    """

    if not -n_prefixes <= index < n_prefixes:
        raise IndexError(f"prefix depth {index} is out of range for {n_prefixes} available depths")
    return int(index % n_prefixes) + 2


def resolve_accel_mode(mode: str, num_steps: int) -> str:
    """Downgrade a too-deep prefix to the last available one, with a log line.

    Mirrors ``fm-geometry``'s ``_accel_mode``: a shallow-denoise model (fewer
    Euler steps than the requested depth) cannot index the requested prefix, and
    silently scoring the whole chunk instead would swap in an undocumented
    signal.  Only ``accel_prefix`` is rewritten -- ``accel_prefix_full:<j>`` is
    scored against the whole horizon, so its depth is independent of ``n_exec``
    and upstream's broader ``startswith(("accel_exec", "accel_prefix"))`` test
    would wrongly rewrite it.
    """

    base, index = parse_accel_mode(mode)
    if base != "accel_prefix" or index is None:
        return mode
    available = prefix_layers(num_steps)
    if not -available <= index < available:
        logger.info(
            "accel mode %r needs denoise depth %d but this sampler has only %d prefix layer(s) "
            "(num_steps=%d); falling back to 'accel_prefix:-1' (last layer == accel_exec).",
            mode,
            index,
            available,
            num_steps,
        )
        return "accel_prefix:-1"
    return mode


def _resolve_std(flat: np.ndarray, fixed_std: np.ndarray | None) -> np.ndarray:
    """Return the per-flattened-coordinate scale used before path norms."""

    if fixed_std is None:
        return flat.std(axis=(0, 1))
    scale = np.asarray(fixed_std, dtype=np.float32)
    if scale.ndim != 1 or scale.shape[0] != flat.shape[-1]:
        raise ValueError(
            f"fixed_std shape {scale.shape} does not match flattened dimension {flat.shape[-1]} (window * action_dim)"
        )
    if not np.all(np.isfinite(scale)) or np.any(scale < 0):
        raise ValueError("fixed_std must contain finite non-negative values")
    return scale


def _accel_per_step(xz: np.ndarray) -> np.ndarray:
    """Per-denoise-step curvature contributions for a z-scored path."""

    steps = np.diff(xz, axis=-2)  # proportional to v_t when dt is constant
    step_length = np.linalg.norm(steps, axis=-1)
    mean_step = step_length.mean(axis=-1)
    jerk = np.linalg.norm(np.diff(steps, axis=-2), axis=-1)
    return jerk / (mean_step[..., None] + EPS)


def _accel(xz: np.ndarray) -> np.ndarray:
    """Sum the normalised changes of the denoising tangent."""

    return _accel_per_step(xz).sum(axis=-1)


def _prefix_accel(xz: np.ndarray) -> np.ndarray:
    """Compute Accel for every growing noise-to-action denoise prefix.

    Returns ``(batch, T-1)``: column ``j`` is ``accel_p`` for ``p = j + 2``.
    """

    steps = np.diff(xz, axis=-2)
    step_length = np.linalg.norm(steps, axis=-1)
    jerk = np.linalg.norm(np.diff(steps, axis=-2), axis=-1)
    cumulative_jerk = np.cumsum(jerk, axis=-1)
    cumulative_step = np.cumsum(step_length, axis=-1)
    counts = np.arange(2, step_length.shape[-1] + 1, dtype=np.float32)
    mean_step = cumulative_step[..., 1:] / counts
    return cumulative_jerk / (mean_step + EPS)


def _flatten_path(x_t: Any, action_dim: int, positions: slice) -> np.ndarray:
    """Select real action dimensions and flatten horizon positions into one point."""

    arr = np.asarray(x_t, dtype=np.float32)
    if arr.ndim != 4:
        raise ValueError(f"x_t must have shape (batch, T+1, action_horizon, action_dim); got {arr.shape}")
    if action_dim <= 0 or action_dim > arr.shape[-1]:
        raise ValueError(f"action_dim={action_dim} is invalid for trajectory dim {arr.shape[-1]}")
    selected = arr[:, :, positions, :action_dim]
    return selected.reshape(selected.shape[0], selected.shape[1], -1)


def whole_chunk_curvature(
    x_t: Any,
    action_dim: int,
    fixed_std: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Return Accel and two companion curvature measures for each batch item.

    ``x_t`` has shape ``(batch, T+1, action_horizon, model_action_dim)``.
    All action positions are treated as one flattened high-dimensional path.
    """

    flat = _flatten_path(x_t, action_dim, slice(None))
    xz = flat / (_resolve_std(flat, fixed_std) + EPS)
    steps = np.diff(xz, axis=1)
    step_length = np.linalg.norm(steps, axis=-1)
    path = step_length.sum(axis=1)
    chord = np.linalg.norm(xz[:, -1] - xz[:, 0], axis=-1)
    unit_steps = steps / (step_length[..., None] + EPS)
    cosine = (unit_steps[:, :-1] * unit_steps[:, 1:]).sum(axis=-1)
    return {
        "accel": _accel(xz).astype(np.float32),
        "turn_total": (1.0 - cosine).sum(axis=1).astype(np.float32),
        "arc_chord": (path - chord).astype(np.float32),
    }


def whole_chunk_prefix_accel(
    x_t: Any,
    action_dim: int,
    fixed_std: np.ndarray | None = None,
    *,
    n_exec: int | None = None,
) -> np.ndarray:
    """Return ``(batch, T-1)`` Accel values at every denoise depth.

    When ``n_exec`` is supplied, only the first ``n_exec`` action positions
    participate in the flattened trajectory.
    """

    positions = slice(0, n_exec) if n_exec is not None else slice(None)
    flat = _flatten_path(x_t, action_dim, positions)
    xz = flat / (_resolve_std(flat, fixed_std) + EPS)
    return _prefix_accel(xz).astype(np.float32)


def _resolve_prefix_index(index: int, n_prefixes: int) -> int:
    if not -n_prefixes <= index < n_prefixes:
        raise IndexError(f"prefix depth {index} is out of range for {n_prefixes} available depths")
    return index % n_prefixes


def prefix_index_for_mode(mode: str, n_prefixes: int) -> int:
    """Resolve which column of a prefix table ``mode`` names.

    The single place an out-of-range depth is downgraded, so the policy (which
    keeps the whole table) and :func:`score_chunks` (which keeps one column)
    can never disagree about which statistic a mode means. Columns are
    ``j -> p = j + 2`` denoise steps, negative counts from the end, and a
    request deeper than the sampler has goes to the last layer rather than
    silently rescoring a different statistic.
    """

    if n_prefixes < 1:
        raise ValueError(f"n_prefixes must be >= 1; got {n_prefixes}")
    base, column = parse_accel_mode(mode)
    if base not in {"accel_prefix", "accel_prefix_full"}:
        raise ValueError(f"mode {mode!r} does not index a prefix table")
    if column is None:
        raise ValueError(f"{base!r} requires a denoise depth, for example {base}:7")
    index = int(column)
    if not -n_prefixes <= index < n_prefixes:
        logger.warning(
            "accel mode %r asks for prefix column %d (depth %d) but only %d prefix layer(s) are "
            "available; scoring the last layer (depth %d) instead. Use resolve_accel_mode() to "
            "downgrade before scoring -- a depth swap silently changes the score's scale.",
            mode,
            index,
            index + 2,
            n_prefixes,
            n_prefixes + 1,
        )
        index = -1
    return _resolve_prefix_index(index, n_prefixes)


def score_prefix_table(
    x_t: Any,
    action_dim: int,
    *,
    n_exec: int | None = None,
    fixed_std: np.ndarray | None = None,
) -> np.ndarray:
    """Return the full ``(batch, T-1)`` Accel table instead of one column.

    Column ``j`` is ``accel_p`` at depth ``p = j + 2`` (see
    :func:`prefix_depth`), so one recording can be rescored at every depth
    offline. This is the same computation :func:`score_chunks` performs for a
    prefix mode -- it just keeps the whole row -- which is why the two share
    :func:`prefix_index_for_mode`. ``n_exec=None`` scores the whole action
    horizon, the same convention every other Accel helper uses.
    """

    return whole_chunk_prefix_accel(x_t, action_dim, fixed_std, n_exec=n_exec)


def score_chunks(
    x_t: Any,
    action_dim: int,
    *,
    n_exec: int | None = None,
    mode: str = DEFAULT_ACCEL_MODE,
    fixed_std: np.ndarray | None = None,
) -> np.ndarray:
    """Score a batch of denoising paths with a named Accel variant.

    Accepted modes mirror ``fm-geometry``:

    * ``accel``: whole horizon, full denoise path;
    * ``accel_exec``: executed action window, full denoise path;
    * ``accel_prefix:<j>``: executed window and denoise prefix ``j+2``;
    * ``accel_prefix_full:<j>``: full horizon and denoise prefix ``j+2``;
    * ``window:<n>``: first ``n`` action positions, full denoise path.

    ``accel_prefix:<j>`` is a denoise-depth prefix, not an action-position
    prefix: the depth it reads is ``j + 2``, so ``accel_prefix:3`` scores
    ``accel_5`` and ``accel_prefix:-1`` equals ``accel_exec``.  Call
    :func:`resolve_accel_mode` first when ``num_steps`` is configurable; an
    out-of-range depth that survives to here is logged and downgraded to the
    last available prefix rather than silently rescored on a different window.

    ``accel_exec`` and ``accel_prefix`` describe the *executed* window, so they
    require ``n_exec``.  Without it the window would silently widen to the whole
    chunk, which is an undocumented statistic: the same trap upstream guards
    against in ``fmaccel/detectors/accel.py``.  ``accel_prefix_full`` is exempt
    because it is defined over the whole horizon.
    """

    arr = np.asarray(x_t, dtype=np.float32)
    if arr.ndim != 4:
        raise ValueError(f"x_t must have shape (batch, T+1, action_horizon, action_dim); got {arr.shape}")
    base, separator, column = mode.partition(":")
    exec_slice = slice(0, n_exec) if n_exec is not None else slice(None)

    if n_exec is None and base in {"accel_exec", "accel_prefix"}:
        raise ValueError(
            f"accel mode {mode!r} is an executed-window variant but n_exec is None; pass n_exec so the "
            f"executed window is well-defined instead of silently scoring the whole chunk."
        )

    if base == "accel":
        return whole_chunk_curvature(arr, action_dim, fixed_std)["accel"]
    if base == "accel_exec":
        return _curvature_window(arr, action_dim, exec_slice, fixed_std)
    if base in {"accel_prefix", "accel_prefix_full"}:
        if not separator:
            raise ValueError(f"{base!r} requires a denoise depth, for example {base}:7")
        prefix = score_prefix_table(
            arr,
            action_dim,
            n_exec=None if base == "accel_prefix_full" else n_exec,
            fixed_std=fixed_std,
        )
        return prefix[:, prefix_index_for_mode(mode, prefix.shape[1])]
    if base == "window":
        if not separator:
            raise ValueError("window mode requires a length, for example window:8")
        n = int(column)
        if n <= 0 or n > arr.shape[2]:
            raise ValueError(f"window length {n} is invalid for horizon {arr.shape[2]}")
        return _curvature_window(arr, action_dim, slice(0, n), fixed_std)
    raise ValueError(f"unknown Accel mode {mode!r}")


def _curvature_window(
    x_t: np.ndarray,
    action_dim: int,
    positions: slice,
    fixed_std: np.ndarray | None,
) -> np.ndarray:
    flat = _flatten_path(x_t, action_dim, positions)
    xz = flat / (_resolve_std(flat, fixed_std) + EPS)
    return _accel(xz).astype(np.float32)


@dataclasses.dataclass
class ReferenceStd:
    """Accumulate the run-pooled per-coordinate scale a fixed ``fixed_std`` needs.

    With ``fixed_std=None`` every call self-normalises a single chunk against
    itself, so the score loses magnitude and is no longer the statistic the
    detector was validated on (upstream measured correlation ~0.30 between the
    two scales and ~0.08 AUROC apart).  This accumulator pools the flattened
    paths of every chunk seen so far and freezes one scale, reproducing
    ``fm-geometry``'s ``_accel_std`` exactly -- including pooling the denoise
    axis, so the length is ``window * action_dim``.

    ``update`` takes the same ``(batch, T+1, action_horizon, action_dim)`` array
    as :func:`score_chunks` and applies the same ``_flatten_path`` windowing, so
    the frozen vector matches the score's flattened dimension by construction.
    """

    action_dim: int
    n_exec: int | None = None

    def __post_init__(self) -> None:
        self._parts: list[np.ndarray] = []

    @property
    def positions(self) -> slice:
        return slice(0, self.n_exec) if self.n_exec is not None else slice(None)

    @property
    def n_chunks(self) -> int:
        """Number of trajectories accumulated so far."""

        return sum(part.shape[0] for part in self._parts)

    def update(self, x_t: Any) -> None:
        """Accumulate a batch of denoise paths."""

        flat = _flatten_path(x_t, self.action_dim, self.positions)
        if self._parts and flat.shape[-1] != self._parts[0].shape[-1]:
            raise ValueError(
                f"flattened dimension changed from {self._parts[0].shape[-1]} to {flat.shape[-1]}; "
                f"action_dim/n_exec must stay fixed while accumulating a reference scale"
            )
        self._parts.append(flat)

    def freeze(self) -> np.ndarray:
        """Return the run-pooled per-coordinate std, shape ``(window * action_dim,)``."""

        if not self._parts:
            raise ValueError("no chunks accumulated; call update() before freeze()")
        flat = np.concatenate(self._parts, axis=0)
        # float32 in, float32 out, exactly like fm-geometry's _accel_std. The
        # buffer is dropped so a long-running server does not retain the run.
        std = flat.std(axis=(0, 1)).astype(np.float32)
        self.clear()
        return std

    def clear(self) -> None:
        """Drop the accumulated paths, keeping the configured window."""

        self._parts = []


def save_fixed_std(path: pathlib.Path | str, std: np.ndarray) -> None:
    """Write a reference scale as an npz under ``std``.

    The key and container match ``fm-geometry``'s reference-std npz, so an
    artifact written here can be consumed by the upstream detector and vice
    versa.
    """

    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, std=np.asarray(std, dtype=np.float32))


def load_fixed_std(path: pathlib.Path | str, expected_len: int | None = None) -> np.ndarray:
    """Load a reference scale written by :func:`save_fixed_std` or by fm-geometry.

    Unlike upstream, a file without a ``std`` key is rejected rather than
    silently read from its first key: a wrong-key vector is a wrong scale, and
    nothing downstream can tell.
    """

    target = pathlib.Path(path)
    if not target.exists():
        raise FileNotFoundError(f"reference-std artifact not found: {target}")
    with np.load(target) as handle:
        if "std" not in handle:
            raise ValueError(f"{target} has keys {sorted(handle.keys())}; a reference-std npz must contain 'std'")
        std = np.asarray(handle["std"], dtype=np.float32)
    if std.ndim != 1:
        raise ValueError(f"reference std in {target} must be 1-D; got shape {std.shape}")
    if expected_len is not None and std.shape[0] != expected_len:
        raise ValueError(
            f"reference std in {target} has {std.shape[0]} entries but this window needs {expected_len} "
            f"(window * action_dim)"
        )
    if not np.all(np.isfinite(std)) or np.any(std < 0):
        raise ValueError(f"reference std in {target} must contain finite non-negative values")
    return std


def action_level_accel(x_t: Any, action_dim: int) -> np.ndarray:
    """Return Accel separately for each action position: ``(batch, horizon)``."""

    arr = np.asarray(x_t, dtype=np.float32)
    if arr.ndim != 4:
        raise ValueError(f"x_t must have shape (batch, T+1, horizon, dim); got {arr.shape}")
    if action_dim <= 0 or action_dim > arr.shape[-1]:
        raise ValueError(f"action_dim={action_dim} is invalid for trajectory dim {arr.shape[-1]}")
    paths = np.transpose(arr[..., :action_dim], (0, 2, 1, 3))
    flat = paths.reshape(arr.shape[0] * arr.shape[2], arr.shape[1], action_dim)
    xz = flat / (flat.std(axis=(0, 1)) + EPS)
    return _accel(xz).reshape(arr.shape[0], arr.shape[2]).astype(np.float32)


__all__ = [
    "DEFAULT_ACCEL_MODE",
    "AccelConfig",
    "ReferenceStd",
    "action_level_accel",
    "load_fixed_std",
    "parse_accel_mode",
    "prefix_depth",
    "prefix_index_for_mode",
    "prefix_layers",
    "resolve_accel_mode",
    "save_fixed_std",
    "score_chunks",
    "score_prefix_table",
    "whole_chunk_curvature",
    "whole_chunk_prefix_accel",
]
