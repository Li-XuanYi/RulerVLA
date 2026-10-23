"""Observation capture, so a recorded decision can be re-sampled offline.

The free Accel score is read off a denoising path that already exists. Its ground
truth is not free: ``D_resample`` re-runs the *same* decision with fresh noise and
measures how far the resulting chunks scatter, which is only meaningful if the
observation is the one the policy actually saw. A drifted observation measures a
posterior the policy never faced, and nothing downstream can tell.

Two facts make an exact reconstruction possible, and both are load-bearing:

* ``Observation.from_dict`` converts ``uint8`` images to ``[-1, 1]`` float32
  itself, so images can be stored as ``uint8`` -- a quarter of the bytes, and
  lossless, because the conversion is deterministic.
* The sampler applies ``preprocess_observation(train=False)``, which at 224x224
  is a no-op besides filling a default mask. Replaying the captured fields
  through the same entry point therefore runs the identical code path.

This module stays numpy-only and policy-free: capturing, writing and reading a
context never needs a model loaded, which is what lets the resample stage run on
a different machine from the one that recorded.
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# The fields ``Observation.from_dict`` reads, and therefore the ones a faithful
# reconstruction needs. ``actions`` and the pi0-fast token masks also ride along
# in the inputs dict but the sampler never reads them at inference time.
REQUIRED_KEYS = ("image", "image_mask", "state")
OPTIONAL_KEYS = ("tokenized_prompt", "tokenized_prompt_mask")

# Nested dicts (images, masks per camera) are flattened with this separator
# because npz keys are flat strings. Checked for collisions on the way out.
SEPARATOR = "__"
META_KEY = "__meta__"
# The recorded noise and endpoint live alongside the observation, not in the
# per-chunk jsonl: they are only meaningful together with it.
NOISE_KEY = "noise"
CHUNK_KEY = "chunk"

# TrajectoryRecorder's arrays, also npz-only.
PATH_KEY = "path"  # x_t: (T+1, action_horizon, action_dim)
VELOCITY_KEY = "velocity"  # v_t: (T, action_horizon, action_dim)
OBS_INDEX_KEY = "obs_index"  # 0-based within the episode


def _flatten(context: dict) -> dict[str, np.ndarray]:
    """Flatten a captured context into npz-keyed arrays."""

    flat: dict[str, np.ndarray] = {}
    for key, value in context.items():
        if isinstance(value, dict):
            for name, array in value.items():
                if SEPARATOR in name:
                    raise ValueError(f"image key {name!r} contains the separator {SEPARATOR!r}")
                flat[f"{key}{SEPARATOR}{name}"] = np.asarray(array)
        else:
            flat[key] = np.asarray(value)
    return flat


def _nest(flat: dict[str, np.ndarray]) -> dict:
    """Inverse of :func:`_flatten`."""

    nested: dict[str, Any] = {}
    for key, array in flat.items():
        if SEPARATOR in key:
            head, _, tail = key.partition(SEPARATOR)
            nested.setdefault(head, {})[tail] = array
        else:
            nested[key] = array
    return nested


def capture_context(inputs: dict) -> dict:
    """Extract the reconstructable conditioning from a pre-batched inputs dict.

    Called on the dict produced by the policy's input transforms, *before* the
    batch axis is added, so images are still in the dtype the transform emitted
    and nothing has been converted behind our back. Values are copied to numpy;
    a captured context holds no reference to live device memory.
    """

    missing = [key for key in REQUIRED_KEYS if key not in inputs]
    if missing:
        raise ValueError(f"cannot capture context: inputs is missing {missing}; got {sorted(inputs)}")
    context: dict[str, Any] = {}
    for key in (*REQUIRED_KEYS, *OPTIONAL_KEYS):
        if key not in inputs:
            continue
        value = inputs[key]
        if isinstance(value, dict):
            context[key] = {name: np.array(np.asarray(array)) for name, array in value.items()}
        else:
            context[key] = np.array(np.asarray(value))
    return context


class ContextRecorder:
    """Buffers captured contexts and writes them one npz per episode.

    Per episode rather than per decision because the arrays are large (a
    two-camera LIBERO observation is ~287 KB) and a file per decision would make
    the output directory unusable. Buffered because a rollout client asks for a
    chunk every few hundred milliseconds and a synchronous write on that path
    would show up in the control loop.

    ``stride`` records every ``stride``-th decision. Context is the expensive
    thing to store here, so this is the knob that trades sample size for disk.
    """

    def __init__(
        self,
        directory: pathlib.Path | str,
        *,
        stride: int = 1,
        max_buffered: int = 512,
    ) -> None:
        if int(stride) < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        self.directory = pathlib.Path(directory)
        self.stride = int(stride)
        self.max_buffered = int(max_buffered)
        self._episode_id: str | None = None
        self._rows: list[tuple[int, dict]] = []
        self._npz_path: pathlib.Path | None = None
        self.closed = False

    # ---- lifecycle -------------------------------------------------------
    def start_episode(self, episode_id: str | None) -> None:
        """Flush the finished episode and begin recording the next."""

        self.flush()
        self._episode_id = None if episode_id is None else str(episode_id)
        self._rows = []
        if self._episode_id is None:
            logger.warning("ContextRecorder: episode has no id; context for it will be dropped")
            return
        self._npz_path = self.directory / f"{safe_name(self._episode_id)}.npz"

    def add(self, chunk_index: int, context: dict | None, *, trajectory: Any | None = None) -> None:
        """Record one decision's context, subject to the stride.

        ``trajectory`` is the recorded denoise path ``(T+1, action_horizon,
        action_dim)``. Its first entry is the noise that was used and its last is
        the chunk that was produced, and both are needed to replay the decision
        faithfully -- so the noise is stored rather than redrawn, and the pair
        lives with the observation it belongs to rather than inflating the
        per-chunk jsonl with a few hundred floats per decision.
        """

        if context is None or self._npz_path is None or self.closed:
            return
        if int(chunk_index) % self.stride:
            return
        row = dict(context)
        if trajectory is not None:
            path = np.asarray(trajectory)
            if path.ndim == 4:
                # The runtime hands over the batched form; a capture is always one
                # decision, so drop the batch axis rather than storing it.
                if path.shape[0] != 1:
                    raise ValueError(f"a capture holds one decision but the trajectory has batch {path.shape[0]}")
                path = path[0]
            if path.ndim != 3:
                raise ValueError(
                    f"trajectory must be (T+1, action_horizon, action_dim); got {np.asarray(trajectory).shape}"
                )
            # ``x_t`` is built as [noise, ...Euler iterates...], so index 0 is the
            # noise the sampler was handed and index -1 is the executed endpoint.
            row[NOISE_KEY] = np.array(path[0])
            row[CHUNK_KEY] = np.array(path[-1])
        self._rows.append((int(chunk_index), row))
        if len(self._rows) >= self.max_buffered:
            self.flush()

    def flush(self) -> None:
        """Write what has been buffered for the current episode.

        Idempotent and merge-based: a second flush for the same episode (a client
        reconnecting mid-episode, or an explicit shutdown after a boundary) adds
        its rows to the file instead of truncating the ones already on disk.
        """

        if self._npz_path is None or not self._rows:
            return
        _write_rows(self._npz_path, self._rows)
        self._rows = []

    def close(self) -> None:
        """Flush the last episode and stop recording; safe to call more than once."""

        if self.closed:
            return
        self.flush()
        self.closed = True

def _write_rows(path: pathlib.Path, rows: list[tuple[int, dict]]) -> None:
    """Merge ``rows`` into the episode's npz and write it.

    Shared by both recorders, which persist the same shape of row keyed by
    chunk_index. Idempotent and merge-based: a second flush for the same episode
    (a client reconnecting mid-episode, or an explicit shutdown after a boundary)
    adds its rows to the file instead of truncating the ones already on disk.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    # Keyed by chunk_index so a re-flush of the same episode merges rather
    # than overwrites, and so a row recorded twice is stored once.
    merged: dict[int, dict[str, np.ndarray]] = dict(_read_rows(path)) if path.exists() else {}
    for chunk_index, context in rows:
        merged[int(chunk_index)] = _flatten(context)

    chunk_indices = sorted(merged)
    columns: dict[str, list[np.ndarray]] = {}
    for chunk_index in chunk_indices:
        for key, array in merged[chunk_index].items():
            columns.setdefault(key, []).append(array)
    payload: dict[str, np.ndarray] = {}
    for key, arrays in columns.items():
        try:
            payload[key] = np.stack(arrays)
        except ValueError as exc:  # shapes differ within one episode
            raise ValueError(
                f"context field {key!r} changed shape inside one episode "
                f"({[array.shape for array in arrays][:3]}); the observation pipeline is not stable "
                f"across decisions"
            ) from exc
    meta = {
        "chunk_index": chunk_indices,
        "dtypes": {key: str(array.dtype) for key, array in payload.items()},
        "shapes": {key: list(array.shape[1:]) for key, array in payload.items()},
        "n_rows": len(chunk_indices),
    }
    payload[META_KEY] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    # Compressed: these are natural images, which shrink substantially, and
    # the file is written once per episode rather than on the control path.
    np.savez_compressed(path, **payload)


def _single_decision(array: Any, name: str) -> np.ndarray:
    """Drop a batch axis of one and enforce the sampler's array rank."""

    arr = np.asarray(array)
    if arr.ndim == 4:
        if arr.shape[0] != 1:
            raise ValueError(f"{name} holds one decision but has batch {arr.shape[0]}")
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"{name} must be (steps, action_horizon, action_dim); got {arr.shape}")
    return arr


class TrajectoryRecorder:
    """Buffers denoise trajectories and writes them one npz per episode.

    The Accel score is a lossy readout of a denoising path the sampler already
    computed. Keeping the path itself makes every later statistic over it -- a
    different depth, a different normalisation, a geometry term that has not been
    invented yet -- an offline recomputation over recorded arrays instead of
    another rollout, which is the difference between re-scoring twenty thousand
    episodes in minutes and re-running them over days.

    ``x_t`` is ``[noise, ...Euler iterates...]``, so the stored path already
    determines both endpoints: index 0 is the noise that was sampled and index -1
    is the action chunk that was executed (its first ``replan_steps`` entries are
    what a rollout client actually applies). ``v_t`` is stored alongside even
    though a constant Euler step makes it recoverable by differencing: it is the
    quantity downstream geometry is defined on, and differencing adjacent
    quantised iterates would put the quantisation error of ``x`` straight into it.

    Storage is float32, not float16. Accel is a ratio of norms over the
    *differences* of adjacent iterates, and those differences are much smaller
    than the iterates themselves, so a coarse dtype's relative error on ``x``
    lands almost undiluted on the difference: a float16 round-trip reproduced the
    live score to only ~1e-2 relative (median 1.1%, max 3.4% over 90 chunks),
    which is enough to move a chunk across a threshold. float32 removes it.

    Observations are deliberately *not* stored here. That is ContextRecorder's
    job, and they are two orders of magnitude larger.
    """

    def __init__(
        self,
        directory: pathlib.Path | str,
        *,
        dtype: str = "float32",
        max_buffered: int = 512,
    ) -> None:
        self.directory = pathlib.Path(directory)
        self.dtype = np.dtype(dtype)
        self.max_buffered = int(max_buffered)
        self._episode_id: str | None = None
        self._rows: list[tuple[int, dict]] = []
        self._npz_path: pathlib.Path | None = None
        self._next_obs_index = 0
        self.closed = False

    # ---- lifecycle -------------------------------------------------------
    def start_episode(self, episode_id: str | None) -> None:
        """Flush the finished episode and begin recording the next."""

        self.flush()
        self._episode_id = None if episode_id is None else str(episode_id)
        self._rows = []
        self._next_obs_index = 0
        if self._episode_id is None:
            logger.warning("TrajectoryRecorder: episode has no id; its trajectories will be dropped")
            return
        self._npz_path = self.directory / f"{safe_name(self._episode_id)}.npz"

    def add(
        self,
        chunk_index: int,
        path: Any,
        velocity: Any | None = None,
        *,
        obs_index: int | None = None,
    ) -> None:
        """Record one decision's denoise trajectory.

        Sizes are checked here rather than at analysis time: a run that recorded
        a mismatched velocity array would otherwise be discovered only after the
        rollout finished, with the compute already spent.
        """

        if self._npz_path is None or self.closed:
            return
        path_array = _single_decision(path, "x_t")
        row: dict[str, np.ndarray] = {PATH_KEY: path_array.astype(self.dtype, copy=False)}
        if velocity is not None:
            velocity_array = _single_decision(velocity, "v_t")
            # v_t holds one velocity per Euler step, so it is x_t minus the final
            # iterate: (T, H, D) against x_t's (T+1, H, D). Checked rather than
            # assumed, because a mismatched pair would only surface at analysis
            # time, after the rollout compute had been spent.
            expected = (path_array.shape[0] - 1,) + path_array.shape[1:]
            if velocity_array.shape != expected:
                raise ValueError(
                    f"v_t has shape {velocity_array.shape} but x_t {path_array.shape} implies {expected}; "
                    f"the two describe different denoise schedules"
                )
            row[VELOCITY_KEY] = velocity_array.astype(self.dtype, copy=False)
        if obs_index is not None:
            # An explicit index declares where in the episode this decision sits,
            # so the ones after it continue from there rather than from the count,
            # which would leave the episode's indices out of order.
            self._next_obs_index = int(obs_index)
        row[OBS_INDEX_KEY] = np.int32(self._next_obs_index)
        self._next_obs_index += 1

        self._rows.append((int(chunk_index), row))
        if len(self._rows) >= self.max_buffered:
            self.flush()

    def flush(self) -> None:
        """Write what has been buffered for the current episode."""

        if self._npz_path is None or not self._rows:
            return
        _write_rows(self._npz_path, self._rows)
        self._rows = []

    def close(self) -> None:
        """Flush the last episode and stop recording; safe to call more than once."""

        if self.closed:
            return
        self.flush()
        self.closed = True


def _read_rows(path: pathlib.Path) -> dict[int, dict[str, np.ndarray]]:
    """Existing rows of a context file, keyed by chunk index."""

    with np.load(path) as handle:
        if META_KEY not in handle:
            raise ValueError(f"{path} is not a context file (no {META_KEY!r} entry)")
        meta = json.loads(bytes(np.asarray(handle[META_KEY])).decode())
        rows = {}
        for position, chunk_index in enumerate(meta["chunk_index"]):
            rows[int(chunk_index)] = {key: np.asarray(handle[key][position]) for key in meta["dtypes"]}
    return rows


def load_replay(path: pathlib.Path | str, chunk_index: int) -> tuple[dict, np.ndarray, np.ndarray]:
    """Load one decision's observation plus the noise to replay and chunk to check.

    Returns ``(observation, noise, chunk)``. Raises if the recording predates the
    trajectory capture, because a fidelity gate needs both endpoints and
    fabricating either would defeat the point of having one.
    """

    path = pathlib.Path(path)
    with np.load(path) as handle:
        meta = json.loads(bytes(np.asarray(handle[META_KEY])).decode())
        positions = [index for index, value in enumerate(meta["chunk_index"]) if int(value) == int(chunk_index)]
        if not positions:
            raise KeyError(
                f"{path} has no context for chunk_index={chunk_index}; recorded chunks are "
                f"{meta['chunk_index'][:8]}{'...' if len(meta['chunk_index']) > 8 else ''}"
            )
        row = positions[0]
        for key in (NOISE_KEY, CHUNK_KEY):
            if key not in meta["dtypes"]:
                raise ValueError(
                    f"{path} has no {key!r}; it was recorded without the denoise trajectory, so a replay "
                    f"cannot be checked against it. Re-record with the trajectory capture enabled."
                )
        arrays = {key: np.asarray(handle[key][row]) for key in meta["dtypes"]}
    noise = arrays.pop(NOISE_KEY)
    chunk = arrays.pop(CHUNK_KEY)
    return add_batch_axis(_nest(arrays)), noise, chunk


def safe_name(episode_id: str) -> str:
    """Map an episode id onto a filename component."""

    safe = "".join(character if character.isalnum() or character in "-_." else "_" for character in episode_id)
    if not safe or safe.startswith("."):
        raise ValueError(f"episode id {episode_id!r} has no usable characters for a filename")
    return safe


def read_context_meta(path: pathlib.Path | str) -> dict:
    """Read the index a recorder wrote, without materialising the arrays."""

    with np.load(path) as handle:
        if META_KEY not in handle:
            raise ValueError(f"{path} is not a context file (no {META_KEY!r} entry)")
        return json.loads(bytes(np.asarray(handle[META_KEY])).decode())


def load_context_dict(path: pathlib.Path | str, chunk_index: int) -> dict:
    """Rebuild one decision's context as the dict ``Observation.from_dict`` wants.

    The batch axis is added here, matching what ``Policy.infer`` does before the
    model sees the inputs, so the caller can hand the result straight to the
    sampler and get the same code path the recording took. Dtypes are preserved
    exactly as captured -- notably ``uint8`` images, which ``from_dict`` converts
    itself rather than us converting on capture and freezing the choice in.
    """

    path = pathlib.Path(path)
    with np.load(path) as handle:
        meta = json.loads(bytes(np.asarray(handle[META_KEY])).decode())
        positions = [index for index, value in enumerate(meta["chunk_index"]) if int(value) == int(chunk_index)]
        if not positions:
            raise KeyError(
                f"{path} has no context for chunk_index={chunk_index}; recorded chunks are "
                f"{meta['chunk_index'][:8]}{'...' if len(meta['chunk_index']) > 8 else ''}"
            )
        row = positions[0]
        arrays = {key: np.asarray(handle[key][row]) for key in meta["dtypes"]}
    return add_batch_axis(_nest(arrays))


def add_batch_axis(context: dict) -> dict:
    """Add the leading batch axis, matching the policy's pre-model batching."""

    out: dict[str, Any] = {}
    for key, value in context.items():
        if isinstance(value, dict):
            out[key] = {name: array[None, ...] for name, array in value.items()}
        else:
            out[key] = value[None, ...]
    return out


__all__ = [
    "CHUNK_KEY",
    "META_KEY",
    "NOISE_KEY",
    "OPTIONAL_KEYS",
    "REQUIRED_KEYS",
    "SEPARATOR",
    "ContextRecorder",
    "add_batch_axis",
    "capture_context",
    "load_context_dict",
    "load_replay",
    "read_context_meta",
    "safe_name",
]
