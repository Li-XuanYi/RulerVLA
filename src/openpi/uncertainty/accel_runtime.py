"""Serving-side state for the Accel readout: reference scale, calibration, logs.

``fm-geometry`` records a run and fits its detector afterwards, in one process.
A served policy cannot: the score has to be comparable *across* calls from the
very first request, which means the per-coordinate reference scale cannot be
recomputed per chunk, and a threshold fitted offline has to be shown applicable
to the live configuration before it is trusted.

This module is that seam, deliberately kept out of the WebSocket handler so it
is testable without an event loop (``pytest-asyncio`` is not a dependency) and
so the artifact a calibration script writes and the artifact the server checks
come from one implementation.  It is numpy-only, like the rest of
``openpi.uncertainty``: the feedback knobs travel as a plain dict because
materialising them would pull the policy layer's torch/flax imports in here.

Two rules it enforces, both silent-corruption bugs otherwise:

* a loaded calibration must match the live score identity, or requests are
  refused rather than scored against a meaningless ``h``;
* ``fixed_std="auto"`` cannot be combined with a loaded calibration, because a
  run-pooled scale is a property of *that* run's chunks and a later run's will
  not reproduce it.
"""

from __future__ import annotations

from collections.abc import Iterable
import dataclasses
import json
import logging
import pathlib
import re
import sys
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from openpi.uncertainty.accel import DEFAULT_ACCEL_MODE
from openpi.uncertainty.accel import AccelConfig
from openpi.uncertainty.accel import ReferenceStd
from openpi.uncertainty.accel import load_fixed_std
from openpi.uncertainty.accel import parse_accel_mode
from openpi.uncertainty.accel import resolve_accel_mode
from openpi.uncertainty.accel import save_fixed_std
from openpi.uncertainty.calibration import AccelCalibration
from openpi.uncertainty.calibration import digest_std
from openpi.uncertainty.cusum import AccelMonitor

if TYPE_CHECKING:
    from openpi.uncertainty.context import ContextRecorder
    from openpi.uncertainty.context import TrajectoryRecorder

logger = logging.getLogger(__name__)

WHOLE_HORIZON_MODES = frozenset({"accel", "accel_prefix_full"})
REFERENCE_STD_FILENAME = "accel_fixed_std.npz"
# A client that never announces an episode boundary lets the CUSUM walk across
# episodes; warn once this many chunks have gone by without one.
STALE_BOUNDARY_CHUNKS = 200


@dataclasses.dataclass(frozen=True)
class AccelRuntimeConfig:
    """Runtime knobs for the Accel readout, as they arrive from the CLI."""

    mode: str = DEFAULT_ACCEL_MODE
    action_dim: int = 7
    n_exec: int | None = 10
    num_steps: int = 10
    fpr: float = 0.1
    k_sigma: float = 0.25
    # "none" (self-normalised; exploratory only), "auto" (pool this run and
    # freeze), or a path to a reference-std npz.
    fixed_std: str = "none"
    warmup_chunks: int = 200
    calibration: str | None = None
    record: str | None = None
    # Directory for per-episode observation captures. Empty means no capture --
    # the score is unaffected either way, so this is purely additive.
    context_dir: str | None = None
    # Store every ``context_stride``-th decision. Observations dominate the
    # on-disk cost, so this is the knob that trades rho sample size for disk.
    context_stride: int = 1
    # Directory for per-episode denoise trajectories (the full x_t and v_t
    # arrays, not just the scalar scores). Empty disables capture. Kept separate
    # from context_dir because the two answer different questions -- a context
    # replays a decision, a trajectory re-scores it -- and a run may want either
    # without the other.
    traj_dir: str | None = None
    # Precision for the stored trajectories. float32, not float16: Accel is a
    # ratio of norms over the *differences* of adjacent iterates, and those
    # differences are far smaller than x itself, so float16's ~1e-3 relative
    # quantisation lands almost entirely on the difference. Measured on a
    # 90-chunk round-trip, float16 reproduced the live score to only ~1e-2
    # relative (median 1.1%, max 3.4%) -- enough to change a threshold's answer.
    traj_dtype: str = "float32"
    feedback: dict[str, Any] | None = None

    def score_config(self) -> AccelConfig:
        """The score identity these knobs describe (excluding the scale array)."""

        return AccelConfig(
            action_dim=int(self.action_dim),
            n_exec=None if self.n_exec is None else int(self.n_exec),
            mode=self.mode,
            num_steps=int(self.num_steps),
        )


class AccelRuntime:
    """Owns the reference scale, the monitor, and the per-chunk record.

    One instance serves one rollout client: the monitor is a single causal state
    machine, so two clients sharing it would interleave their episodes and each
    would look like drift to the other.  Parallel evaluation against one server
    needs the monitor keyed per worker (as the RoboTwin harness does); that is
    out of scope here, and the record carries a chunk index so the split can be
    made offline.
    """

    def __init__(
        self,
        config: AccelRuntimeConfig,
        *,
        action_horizon: int | None = None,
        model_action_dim: int | None = None,
    ) -> None:
        self.calibration: AccelCalibration | None = None
        self._monitor: AccelMonitor | None = None
        self._chunks_seen = 0
        self._chunks_without_boundary = 0
        self._boundary_warned = False
        # Episode identity. Counting here rather than in the reader means an
        # episode whose scores are all unusable still occupies its slot: without
        # that, the Nth stream would pair with the (N-k)th outcome and every
        # threshold fitted downstream would be meaningless.
        self._episode_index = 0
        self._episode_id: str | None = None
        # Observation capture runs alongside scoring and can never change it: the
        # recorder is written to after the score is already decided, and nothing
        # about the reference scale or the calibration consults it.
        self._context_recorder: ContextRecorder | None = None
        if config.context_dir:
            from openpi.uncertainty.context import ContextRecorder

            self._context_recorder = ContextRecorder(config.context_dir, stride=config.context_stride)
        # The trajectory recorder is independent of the context recorder: it
        # stores the denoise path the score was read off, which is what makes the
        # score recomputable later. Like capture, it is written to after the score
        # is decided and cannot change it.
        self._traj_recorder: TrajectoryRecorder | None = None
        if config.traj_dir:
            from openpi.uncertainty.context import TrajectoryRecorder

            self._traj_recorder = TrajectoryRecorder(config.traj_dir, dtype=config.traj_dtype)

        # Nothing named a reference scale but the calibration artifact did: adopt
        # it, so serving a fitted calibration is `--accel-calibration X.json`
        # rather than restating the npz path it already records.
        if config.calibration is not None and config.fixed_std in {"none", ""}:
            artifact = AccelCalibration.load(pathlib.Path(config.calibration))
            if artifact.fixed_std_path is not None:
                logger.info("Accel: adopting reference scale %s from the calibration artifact", artifact.fixed_std_path)
                config = dataclasses.replace(config, fixed_std=artifact.fixed_std_path)
        self.config = config

        self._validate_window(action_horizon, model_action_dim)
        base, index = parse_accel_mode(config.mode)
        if base in {"accel_exec", "accel_prefix"} and config.n_exec is None:
            raise ValueError(f"accel mode {config.mode!r} is an executed-window variant and requires n_exec")
        self.mode = resolve_accel_mode(config.mode, config.num_steps)
        if self.mode != config.mode:
            # Resolving at startup, not per call, is what keeps a loaded `h` from
            # being applied to a score computed at a different denoise depth.
            logger.warning("Accel mode %r downgraded to %r for num_steps=%d", config.mode, self.mode, config.num_steps)

        self._horizon = action_horizon
        self._reference = ReferenceStd(
            action_dim=config.action_dim,
            n_exec=None if self._whole_horizon() else config.n_exec,
        )
        if config.fixed_std == "auto" and config.calibration is not None:
            raise ValueError(
                "fixed_std='auto' pools the reference scale from this run's chunks, so the resulting vector "
                "cannot reproduce a calibration fitted on another run (the digest can never match). Pass the "
                "reference-std npz the calibration names, or drop --accel-calibration."
            )
        if config.fixed_std == "auto" and config.warmup_chunks <= 0:
            raise ValueError("fixed_std='auto' needs a positive warmup_chunks to pool the reference scale")
        self._fixed_std = (
            None
            if config.fixed_std in {"none", "auto", ""}
            else load_fixed_std(config.fixed_std, expected_len=self.flattened_length)
        )

        if config.calibration is not None:
            self._load_calibration(pathlib.Path(config.calibration), self._fixed_std)
        self._validate_feedback()

    def _validate_window(self, action_horizon: int | None, model_action_dim: int | None) -> None:
        config = self.config
        if config.action_dim < 1:
            raise ValueError(f"action_dim must be >= 1, got {config.action_dim}")
        if model_action_dim is not None and config.action_dim > model_action_dim:
            raise ValueError(f"action_dim={config.action_dim} exceeds the model's action_dim={model_action_dim}")
        if config.n_exec is None:
            return
        if config.n_exec < 1:
            raise ValueError(f"n_exec must be >= 1, got {config.n_exec}")
        if action_horizon is not None and config.n_exec > action_horizon:
            # numpy accepts an oversized slice without complaint and would score
            # a shorter window than the mode's name claims.
            raise ValueError(
                f"n_exec={config.n_exec} exceeds the model's action_horizon={action_horizon}; the executed "
                f"window would be shorter than requested and every score would be computed on a window the "
                f"calibration never saw"
            )

    def _validate_feedback(self) -> None:
        """Fail at startup when the response strategy has no source of alarms."""

        if self.config.feedback is None:
            return
        strategy = str(self.config.feedback.get("strategy", "none"))
        # ``adaptive`` is exempt: it maps the score to a horizon on every request
        # instead of gating on an alarm.  Without this check the failure surfaces
        # as a ValueError on the first request of a run rather than at startup.
        if (
            strategy not in {"none", "adaptive"}
            and self.calibration is None
            and self.config.feedback.get("threshold") is None
        ):
            raise ValueError(
                f"Accel feedback strategy {strategy!r} needs a source of alarms: pass --accel-calibration "
                f"(which supplies the CUSUM threshold) or an explicit --accel-threshold."
            )

    def _whole_horizon(self) -> bool:
        return parse_accel_mode(self.mode)[0] in WHOLE_HORIZON_MODES

    @property
    def window(self) -> int:
        """Number of action positions the flattened path covers."""

        if self._whole_horizon():
            if self._horizon is None:
                raise ValueError("whole-horizon Accel modes need the model's action_horizon to size the window")
            return int(self._horizon)
        if self.config.n_exec is None:
            raise ValueError(f"accel mode {self.mode!r} needs n_exec to size the window")
        return int(self.config.n_exec)

    @property
    def flattened_length(self) -> int:
        return self.window * self.config.action_dim

    @property
    def monitor(self) -> AccelMonitor | None:
        return self._monitor

    @property
    def fixed_std(self) -> np.ndarray | None:
        return self._fixed_std

    @property
    def calibrating(self) -> bool:
        """Whether the reference scale is still being pooled from live chunks."""

        return self.config.fixed_std == "auto" and self._fixed_std is None

    @property
    def capturing_context(self) -> bool:
        """Whether requests should carry the observation back for capture.

        Kept out of :meth:`call_kwargs`: that method describes the *score
        identity*, and capture cannot change a score.
        """

        return self._context_recorder is not None

    def _load_calibration(self, path: pathlib.Path, fixed_std: np.ndarray | None) -> None:
        calibration = AccelCalibration.load(path)
        problems = calibration.mismatches(self.config.score_config(), fixed_std=fixed_std)
        if problems:
            raise ValueError(
                f"Accel calibration at {path} does not apply to this configuration:\n  - "
                + "\n  - ".join(problems)
                + "\nRe-fit with scripts/calibrate_accel.py using the live mode/action_dim/n_exec/num_steps, "
                "or fix the server flags."
            )
        if calibration.fixed_std_length is not None and calibration.fixed_std_length != self.flattened_length:
            raise ValueError(
                f"calibration expects a reference scale of {calibration.fixed_std_length} entries but this "
                f"window needs {self.flattened_length}"
            )
        self.calibration = calibration
        self._monitor = AccelMonitor.from_calibration(calibration)
        logger.info(
            "Accel CUSUM calibrated: mu0=%.4f sigma=%.4f k=%.4f h=%.4f protocol=%s conformal=%s M=%d",
            calibration.mu0,
            calibration.sigma,
            calibration.k,
            calibration.h,
            calibration.protocol,
            calibration.conformal,
            calibration.n_calib,
        )

    def call_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for the policy's Accel inference entry points."""

        return {
            "mode": self.mode,
            "n_exec": self.config.n_exec,
            "action_dim": self.config.action_dim,
            "num_steps": self.config.num_steps,
            "fixed_std": self._fixed_std,
        }

    def feedback_kwargs(self) -> dict[str, Any] | None:
        """Return a copy of the feedback knobs, or ``None`` for monitor-only."""

        return None if self.config.feedback is None else dict(self.config.feedback)

    def reset_connection(self) -> None:
        """Start serving a client connection without opening an episode.

        Distinct from :meth:`start_episode` because it must not consume an
        episode number: the monitor still has to be cleared so a previous
        client's drift does not carry over, but no rollout has begun yet.
        """

        self._chunks_without_boundary = 0
        if self._monitor is not None:
            self._monitor.reset()
        self._episode_id = None
        self._log({"event": "connection_boundary"})

    def start_episode(self, episode_id: str | None = None) -> None:
        """Close the finished episode and open the next one.

        ``episode_id`` comes from the rollout client and is what the calibration
        script joins on. It is echoed into every record of the episode so a stream
        can never be paired with another episode's outcome, including when some
        episodes yield no usable scores at all. Left ``None``, the server's own
        monotonic counter identifies the episode instead.
        """

        self._chunks_without_boundary = 0
        if self._monitor is not None:
            self._monitor.reset()
        self._episode_index += 1
        self._episode_id = None if episode_id is None else str(episode_id)
        if self._context_recorder is not None:
            self._context_recorder.start_episode(self._episode_id)
        if self._traj_recorder is not None:
            self._traj_recorder.start_episode(self._episode_id)
        self._log(
            {
                "event": "episode_boundary",
                "episode_index": int(self._episode_index),
                "episode_id": self._episode_id,
            }
        )

    def flush_context(self) -> None:
        """Persist buffered captures without ending the run.

        Called when a connection closes: a client that reconnects mid-episode
        must not lose the decisions it already made, and the recorder merges on
        write, so flushing early is free.
        """

        if self._context_recorder is not None:
            self._context_recorder.flush()
        if self._traj_recorder is not None:
            self._traj_recorder.flush()

    def close(self) -> None:
        """Flush anything still buffered; safe to call more than once."""

        if self._context_recorder is not None:
            self._context_recorder.close()
        if self._traj_recorder is not None:
            self._traj_recorder.close()

    def observe(
        self,
        x_t: Any,
        *,
        v_t: Any | None = None,
        score: float | None = None,
        monitor_info: dict[str, Any] | None = None,
        prefix_table: Any | None = None,
        prefix_table_full: Any | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Pool the reference scale and record one request's outcome.

        This does *not* advance the monitor: the policy owns that, so one
        observation yields exactly one causal decision, taken on the first
        trajectory of the request rather than on a resample.  ``score`` and
        ``monitor_info`` are the values the policy already computed, echoed here
        so the record describes the decision the caller actually got.

        Returns the recorded dict, or ``None`` when the request could not be
        scored because the sampler's denoise depth disagrees with the
        configuration.
        """

        batch = np.asarray(x_t)
        if batch.ndim == 3:  # a single observation arrives batch-squeezed
            batch = batch[None, ...]
        if batch.shape[0] != 1:
            raise ValueError("the Accel monitor is causal and requires one observation per call")

        if batch.shape[1] - 1 != self.config.num_steps:
            # The Euler schedule is part of the score's identity: a different
            # depth is a different statistic and every downstream threshold
            # would be meaningless. Fail on the FIRST request rather than
            # latching a flag -- a run that silently stopped recording would
            # otherwise be discovered only after the rollout finished.
            raise ValueError(
                f"the live trajectory has T={batch.shape[1] - 1} the configuration says num_steps="
                f"{self.config.num_steps}; a different Euler schedule is a different statistic, so "
                f"these scores cannot be compared with a calibrated threshold"
            )

        # Captured before the freeze below, because the score this request was
        # computed with is whatever scale was live when call_kwargs() was read.
        # The chunk that trips the freeze was still scored self-normalised, so it
        # must be tagged not-ready and kept out of the calibration stream.
        was_calibrating = self.calibrating
        if was_calibrating:
            self._reference.update(batch)
            if self._reference.n_chunks >= self.config.warmup_chunks:
                pooled = self._reference.n_chunks
                self._fixed_std = self._reference.freeze()
                self._on_frozen(pooled)

        self._chunks_seen += 1
        self._chunks_without_boundary += 1
        if self._context_recorder is not None:
            # Recorded against the same chunk_index the jsonl carries, so the
            # resample stage can join a captured observation to the denoise path
            # and the score that were taken with it. The trajectory rides along
            # because its endpoints are the noise to replay and the chunk to
            # check the replay against.
            self._context_recorder.add(self._chunks_seen - 1, context, trajectory=x_t)
        if self._traj_recorder is not None:
            # Keyed by the chunk_index the jsonl carries, so an offline re-score
            # joins to exactly the score the server recorded for this decision.
            self._traj_recorder.add(self._chunks_seen - 1, batch, v_t)
        if self._chunks_without_boundary > STALE_BOUNDARY_CHUNKS and not self._boundary_warned:
            self._boundary_warned = True
            logger.warning(
                "Accel has seen %d chunks on one connection with no episode boundary; the CUSUM will keep "
                "accumulating across episodes and its specificity will degrade. Rollout clients must send "
                "'episode_start: True' on the first request of each episode.",
                self._chunks_without_boundary,
            )

        record: dict[str, Any] = {
            "event": "chunk",
            "chunk_index": self._chunks_seen - 1,
            # Identity first: everything about how this score is joined to an
            # outcome depends on these two, and they survive even when the
            # score itself has to be dropped.
            "episode_index": int(self._episode_index),
            "episode_id": self._episode_id,
            "score": None if score is None else float(score),
            "std_ready": not was_calibrating,
            "calibrated": self.calibration is not None,
            "mode": self.mode,
            "n_exec": self.config.n_exec,
            "action_dim": self.config.action_dim,
            "num_steps": self.config.num_steps,
        }
        # Every denoise depth at once, so the recorded run can be re-scored
        # offline instead of re-recorded when the depth changes. ~T-1 floats per
        # chunk, which is why it is affordable at ~130k chunks per run.
        if prefix_table is not None:
            record["prefix_table"] = [float(value) for value in np.asarray(prefix_table).ravel()]
        if prefix_table_full is not None:
            record["prefix_table_full"] = [float(value) for value in np.asarray(prefix_table_full).ravel()]
        for key, cast in (("cusum", float), ("alarm", bool), ("onset", int), ("index", int)):
            if monitor_info is not None and key in monitor_info:
                record[key] = cast(monitor_info[key])
        self._log(record)
        return record

    def _on_frozen(self, pooled_chunks: int) -> None:
        """Report a freshly frozen reference scale and persist it when recording.

        ``pooled_chunks`` is passed in because ``freeze()`` drops the buffer it
        was measured over.
        """

        logger.info(
            "Accel reference scale frozen from %d chunks (window=%d, flattened=%d). Scores recorded before "
            "this point were self-normalised per chunk and are NOT comparable with later ones.",
            pooled_chunks,
            self.window,
            self.flattened_length,
        )
        path = self.reference_std_path()
        if path is not None:
            save_fixed_std(path, self._fixed_std)
        self._log(
            {
                "event": "reference_std",
                "kind": self.config.fixed_std,
                "path": None if path is None else str(path),
                "sha256": None if self._fixed_std is None else digest_std(self._fixed_std),
                "length": self.flattened_length,
                "n_chunks": pooled_chunks,
                "window": self.window,
                "action_dim": self.config.action_dim,
                "mode": self.mode,
                "num_steps": self.config.num_steps,
            }
        )

    def reference_std_path(self) -> pathlib.Path | None:
        """Where this runtime's reference scale lives, if it lives in a file."""

        source = self.config.fixed_std
        if source in {"", "none"}:
            return None
        if source == "auto":
            if self.config.record is None:
                return None
            directory = pathlib.Path(self.config.record)
            return (directory.parent if directory.suffix else directory) / REFERENCE_STD_FILENAME
        return pathlib.Path(source)

    def provenance(self) -> dict[str, Any]:
        """Everything a calibration artifact needs to describe these scores."""

        path = self.reference_std_path()
        return {
            "score": self.config.score_config(),
            "fixed_std_kind": self.config.fixed_std if self.config.fixed_std in {"none", "auto"} else "npz",
            "fixed_std_path": None if path is None else str(path),
            "fixed_std_sha256": None if self._fixed_std is None else digest_std(self._fixed_std),
            "fixed_std_length": None if self._fixed_std is None else int(self._fixed_std.shape[0]),
        }

    def _log(self, record: dict[str, Any]) -> None:
        if self.config.record is None:
            return
        try:
            with open(self.config.record, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(dict(record, time=time.time())) + "\n")
        except OSError:
            # A full or read-only disk must not take down serving.
            pass


# ------------------------------------------------------- reading a record
SUCCESS_PATTERN = re.compile(r"Success:\s*(True|False)")


# The readers live beside the writer: a record's join semantics (episode
# identity, which chunks are usable, what a truncated line means) are the
# same question from both ends, and splitting them across modules is how the
# two drift apart.
def load_records(path: pathlib.Path) -> list[dict]:
    """Read the jsonl, skipping a partially written final line.

    A run interrupted mid-write otherwise fails centuries later during analysis
    with a JSONDecodeError that reads like corrupt data rather than a truncated
    flush; the skipped line is reported so nothing is hidden.
    """

    records: list[dict] = []
    skipped = 0
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            skipped += 1
            print(f"WARNING: {path}:{number} is not valid JSON (truncated write?); skipping.", file=sys.stderr)
    if skipped:
        print(f"WARNING: skipped {skipped} unreadable line(s) in {path}.", file=sys.stderr)
    return records


def load_outcomes(path: pathlib.Path) -> tuple[dict[str, bool], list[bool]]:
    """Return ``({episode_id: success}, [success in file order])``.

    The ordered list is only a fallback for rollout logs that predate per-episode
    ids, and using it is announced: it is the pairing that used to be wrong by
    default.
    """

    text = path.read_text()
    if path.suffix == ".jsonl":
        by_id: dict[str, bool] = {}
        ordered: list[bool] = []
        for number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"WARNING: {path}:{number} is not valid JSON; skipping.", file=sys.stderr)
                continue
            success = bool(record["success"])
            ordered.append(success)
            episode_id = record.get("episode_id")
            if episode_id is not None:
                by_id[str(episode_id)] = success
        return by_id, ordered
    return {}, [match == "True" for match in SUCCESS_PATTERN.findall(text)]


def group_episodes(records: list[dict]) -> list[tuple[str | int, list[dict]]]:
    """Group chunk records into ``(episode_key, chunks)`` in encounter order.

    Mirrors the runtime reader's slot discipline: every boundary opens an episode
    even when none of its chunks are usable, so a recorded episode can never be
    quietly swapped for its neighbour.
    """

    buckets: dict[str | int, list[dict]] = {}
    order: list[str | int] = []
    key: str | int | None = None
    for record in records:
        event = record.get("event")
        if event == "episode_boundary":
            key = record.get("episode_id") or record.get("episode_index")
            if key is None:
                continue
            key = str(key)
            if key not in buckets:
                buckets[key] = []
                order.append(key)
            continue
        if event != "chunk":
            continue
        # Chunks before the boundary-less head of a file belong to an episode the
        # reader never saw opened; fall back to whatever identity they carry.
        chunk_key = record.get("episode_id") or record.get("episode_index")
        chunk_key = str(chunk_key) if chunk_key is not None else key
        if chunk_key is None:
            raise ValueError(
                "a chunk record carries neither 'episode_id' nor 'episode_index'; re-record with the "
                "current server or pass --legacy-positional"
            )
        if chunk_key not in buckets:
            buckets[chunk_key] = []
            order.append(chunk_key)
        key = chunk_key
        buckets[chunk_key].append(record)
    return [(k, buckets[k]) for k in order]


def _episode_key(record: dict[str, Any]) -> str | int | None:
    """The identity a record carries for its episode, or ``None`` if it has none.

    The client-supplied id wins because it survives a restart and is the value
    the outcome file writes; the server's own counter is the fallback so a
    client that sends no id is still identifiable.
    """

    episode_id = record.get("episode_id")
    if episode_id is not None:
        return str(episode_id)
    index = record.get("episode_index")
    return None if index is None else int(index)


def _open_episode(buckets: dict, order: list, key) -> None:
    if key not in buckets:
        buckets[key] = {"scores": [], "alarms": []}
        order.append(key)


def _episode_sequence(records, *, legacy_positional: bool):
    """Group records into ``(key, scores, alarms)`` triples in encounter order."""

    buckets: dict = {}
    order: list = []
    for record in records:
        event = record.get("event")
        if event not in {"chunk", "episode_boundary", "connection_boundary"}:
            continue

        if legacy_positional:
            # Records written before identity existed fall back to counting
            # boundaries. Empty episodes are still kept, so position continues
            # to line up with the outcome file -- that is what the slot is for.
            if event == "chunk":
                if not order:
                    _open_episode(buckets, order, 0)
                key = order[-1]
            else:
                key = len(order)
                _open_episode(buckets, order, key)
        elif event == "episode_boundary":
            # Opened even when the episode turns out to have no usable scores:
            # dropping it would shift every later episode against the outcome
            # file, which is exactly the pairing bug this reader fixes.
            key = _episode_key(record)
            if key is None:
                continue
            _open_episode(buckets, order, key)
            continue
        elif event == "chunk":
            # "connection_boundary" carries no identity by design -- a client
            # connecting is not an episode starting -- so it falls through here.
            key = _episode_key(record)
            if key is None:
                raise ValueError(
                    "a chunk record carries neither 'episode_id' nor 'episode_index'; streams cannot be "
                    "joined to outcomes without episode identity. Re-record with this version, or pass "
                    "legacy_positional=True to fall back to positional pairing."
                )
            _open_episode(buckets, order, key)
        else:
            continue

        if event != "chunk":
            continue
        # Chunks recorded before the reference scale froze were self-normalised
        # per chunk, so including them would fit `h` to a scale that no longer
        # exists. The episode survives; only the chunk is dropped.
        if record.get("std_ready") is not False and record.get("score") is not None:
            buckets[key]["scores"].append(float(record["score"]))
            buckets[key]["alarms"].append(bool(record.get("alarm")))
    return [(key, buckets[key]["scores"], buckets[key]["alarms"]) for key in order]


def streams_from_records(
    records: Iterable[dict[str, Any]],
    *,
    legacy_positional: bool = False,
) -> dict[str | int, list[float]]:
    """Group recorded chunk scores into one stream per episode, keyed by identity.

    Keys are the client's ``episode_id`` when there is one and the server's
    ``episode_index`` otherwise, **in encounter order**, so a caller can join
    them against an outcome file without relying on position. Episodes with no
    usable scores are present with an empty list rather than merged into their
    neighbour.

    Chunks recorded before the reference scale froze are still excluded -- see
    the note in :func:`_episode_sequence` -- but they no longer remove the
    episode they belong to.
    """

    return {key: scores for key, scores, _ in _episode_sequence(records, legacy_positional=legacy_positional)}


def episodes_from_records(
    records: Iterable[dict[str, Any]],
    *,
    legacy_positional: bool = False,
) -> list[dict[str, Any]]:
    """Return one entry per episode, with its identity, scores and firing decisions."""

    return [
        {"episode": key, "scores": scores, "alarms": fired}
        for key, scores, fired in _episode_sequence(records, legacy_positional=legacy_positional)
    ]


__all__ = [
    "REFERENCE_STD_FILENAME",
    "STALE_BOUNDARY_CHUNKS",
    "WHOLE_HORIZON_MODES",
    "AccelRuntime",
    "AccelRuntimeConfig",
    "episodes_from_records",
    "streams_from_records",
]
