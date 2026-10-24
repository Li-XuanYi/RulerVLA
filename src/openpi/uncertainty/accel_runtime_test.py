import json

import numpy as np
import pytest

from openpi.uncertainty.accel import AccelConfig
from openpi.uncertainty.accel import load_fixed_std
from openpi.uncertainty.accel import save_fixed_std
from openpi.uncertainty.accel_runtime import REFERENCE_STD_FILENAME
from openpi.uncertainty.accel_runtime import AccelRuntime
from openpi.uncertainty.accel_runtime import AccelRuntimeConfig
from openpi.uncertainty.accel_runtime import episodes_from_records
from openpi.uncertainty.accel_runtime import streams_from_records
from openpi.uncertainty.calibration import AccelCalibration
from openpi.uncertainty.calibration import digest_std

ACTION_DIM = 2
N_EXEC = 10
NUM_STEPS = 10
WINDOW = N_EXEC * ACTION_DIM


def _write_std(tmp_path):
    std = np.linspace(0.5, 1.5, WINDOW, dtype=np.float32)
    path = tmp_path / "accel_fixed_std.npz"
    save_fixed_std(path, std)
    return path, std


def _write_calibration(tmp_path, std_path, std, **overrides):
    payload = {
        "mu0": 1.8,
        "sigma": 0.4,
        "k": 0.1,
        "h": 2.7,
        "score": AccelConfig(action_dim=ACTION_DIM, n_exec=N_EXEC, mode="accel_prefix:3", num_steps=NUM_STEPS),
        "protocol": "held-out",
        "conformal": True,
        "n_calib": 30,
        "fixed_std_kind": "npz",
        "fixed_std_path": str(std_path),
        "fixed_std_sha256": digest_std(std),
        "fixed_std_length": WINDOW,
    }
    payload.update(overrides)
    path = tmp_path / "calibration.json"
    AccelCalibration(**payload).save(path)
    return path


def _config(**overrides) -> AccelRuntimeConfig:
    payload = {
        "mode": "accel_prefix:3",
        "action_dim": ACTION_DIM,
        "n_exec": N_EXEC,
        "num_steps": NUM_STEPS,
        "fixed_std": "none",
    }
    payload.update(overrides)
    return AccelRuntimeConfig(**payload)


def _chunk(scale: float = 1.0, seed: int = 0, steps: int = NUM_STEPS) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(1, steps + 1, N_EXEC, ACTION_DIM)) * scale).astype(np.float32)


def test_call_kwargs_describe_the_score_identity():
    runtime = AccelRuntime(_config(), action_horizon=N_EXEC, model_action_dim=32)
    assert runtime.call_kwargs() == {
        "mode": "accel_prefix:3",
        "n_exec": N_EXEC,
        "action_dim": ACTION_DIM,
        "num_steps": NUM_STEPS,
        "fixed_std": None,
    }
    assert runtime.flattened_length == WINDOW
    assert runtime.monitor is None
    assert runtime.feedback_kwargs() is None


def test_rejects_a_window_wider_than_the_sampler():
    with pytest.raises(ValueError, match="exceeds the model's action_horizon"):
        AccelRuntime(_config(n_exec=20), action_horizon=10, model_action_dim=32)
    with pytest.raises(ValueError, match="exceeds the model's action_dim"):
        AccelRuntime(_config(action_dim=64), action_horizon=N_EXEC, model_action_dim=32)
    with pytest.raises(ValueError, match="n_exec must be >= 1"):
        AccelRuntime(_config(n_exec=0), action_horizon=N_EXEC)
    # An executed-window mode with no window silently widens to the whole chunk.
    with pytest.raises(ValueError, match="requires n_exec"):
        AccelRuntime(_config(n_exec=None), action_horizon=N_EXEC)


def test_rejects_auto_scale_alongside_a_calibration(tmp_path):
    std_path, std = _write_std(tmp_path)
    calibration = _write_calibration(tmp_path, std_path, std)
    with pytest.raises(ValueError, match="cannot reproduce a calibration fitted on another run"):
        AccelRuntime(_config(fixed_std="auto", calibration=str(calibration)), action_horizon=N_EXEC)


def test_rejects_a_calibration_from_another_configuration(tmp_path):
    std_path, std = _write_std(tmp_path)
    calibration = _write_calibration(
        tmp_path,
        std_path,
        std,
        score=AccelConfig(action_dim=ACTION_DIM, n_exec=N_EXEC, mode="accel_prefix:7", num_steps=NUM_STEPS),
    )
    with pytest.raises(ValueError, match="does not apply to this configuration"):
        AccelRuntime(_config(fixed_std=str(std_path), calibration=str(calibration)), action_horizon=N_EXEC)


def test_rejects_a_tampered_reference_scale(tmp_path):
    std_path, std = _write_std(tmp_path)
    calibration = _write_calibration(tmp_path, std_path, std)
    std_path.unlink()
    tampered = std.copy()
    tampered[0] += 0.5
    save_fixed_std(std_path, tampered)
    with pytest.raises(ValueError, match="digest mismatch"):
        AccelRuntime(_config(fixed_std=str(std_path), calibration=str(calibration)), action_horizon=N_EXEC)


def test_adopts_the_reference_scale_named_by_the_artifact(tmp_path):
    std_path, std = _write_std(tmp_path)
    calibration = _write_calibration(tmp_path, std_path, std)
    # No fixed_std on the CLI: the artifact already records which scale it used.
    runtime = AccelRuntime(_config(calibration=str(calibration)), action_horizon=N_EXEC)
    np.testing.assert_array_equal(runtime.fixed_std, std)
    assert runtime.monitor is not None
    assert runtime.monitor.threshold == pytest.approx(2.7)


def test_strategy_without_a_source_of_alarms_fails_at_startup(tmp_path):
    with pytest.raises(ValueError, match="needs a source of alarms"):
        AccelRuntime(_config(feedback={"strategy": "shorten", "shorten_to": 2}), action_horizon=N_EXEC)
    # An explicit threshold is an acceptable source.
    runtime = AccelRuntime(
        _config(feedback={"strategy": "shorten", "threshold": 2.0, "shorten_to": 2}), action_horizon=N_EXEC
    )
    assert runtime.feedback_kwargs() == {"strategy": "shorten", "threshold": 2.0, "shorten_to": 2}
    # adaptive gates nothing, so it needs no alarm source at all.
    AccelRuntime(_config(feedback={"strategy": "adaptive"}), action_horizon=N_EXEC)


def test_warmup_pools_then_freezes_and_persists_the_scale(tmp_path):
    record = tmp_path / "run.jsonl"
    runtime = AccelRuntime(_config(fixed_std="auto", warmup_chunks=3, record=str(record)), action_horizon=N_EXEC)
    assert runtime.calibrating
    assert runtime.call_kwargs()["fixed_std"] is None

    records = [runtime.observe(_chunk(seed=index), score=1.0) for index in range(2)]
    assert runtime.calibrating
    assert all(record["std_ready"] is False for record in records)

    # The third chunk trips the freeze; its own score was still self-normalised,
    # so it is tagged not-ready and kept out of the calibration stream.
    third = runtime.observe(_chunk(seed=2), score=1.0)
    assert third["std_ready"] is False
    assert not runtime.calibrating
    assert runtime.fixed_std is not None
    assert runtime.flattened_length == WINDOW

    frozen = load_fixed_std(tmp_path / REFERENCE_STD_FILENAME, expected_len=WINDOW)
    np.testing.assert_array_equal(runtime.fixed_std, frozen)
    # The score in call_kwargs is the frozen vector from now on.
    np.testing.assert_array_equal(runtime.call_kwargs()["fixed_std"], frozen)

    fourth = runtime.observe(_chunk(seed=3), score=1.0)
    assert fourth["std_ready"] is True

    lines = [json.loads(line) for line in record.read_text().splitlines()]
    events = [line["event"] for line in lines]
    assert events.count("chunk") == 4
    reference_std = next(line for line in lines if line["event"] == "reference_std")
    assert reference_std["n_chunks"] == 3
    assert reference_std["length"] == WINDOW


def test_warmup_scores_are_excluded_from_the_stream(tmp_path):
    runtime = AccelRuntime(_config(fixed_std="auto", warmup_chunks=2), action_horizon=N_EXEC)
    runtime.observe(_chunk(seed=0), score=1.0)  # self-normalised
    runtime.observe(_chunk(seed=1), score=2.0)  # trips the freeze, still self-normalised
    runtime.observe(_chunk(seed=2), score=3.0)  # first score on the frozen scale
    runtime.start_episode()

    info = runtime.observe(_chunk(seed=3), score=4.0)
    assert info["score"] == 4.0

    streams = streams_from_records(
        [
            {"event": "chunk", "score": 1.0, "std_ready": False, "episode_index": 1},
            {"event": "chunk", "score": 2.0, "std_ready": False, "episode_index": 1},
            {"event": "chunk", "score": 3.0, "std_ready": True, "episode_index": 1},
            {"event": "episode_boundary", "episode_index": 2},
            {"event": "chunk", "score": 4.0, "std_ready": True, "episode_index": 2},
        ]
    )
    assert streams == {1: [3.0], 2: [4.0]}
    episodes = episodes_from_records(
        [
            {"event": "chunk", "score": 1.0, "std_ready": True, "alarm": False, "episode_index": 1},
            {"event": "chunk", "score": 2.0, "std_ready": True, "alarm": True, "episode_index": 1},
            {"event": "episode_boundary", "episode_index": 2},
            {"event": "chunk", "score": 3.0, "std_ready": True, "alarm": False, "episode_index": 2},
        ]
    )
    assert [episode["scores"] for episode in episodes] == [[1.0, 2.0], [3.0]]
    assert [episode["alarms"] for episode in episodes] == [[False, True], [False]]


def test_an_episode_without_usable_scores_keeps_its_slot():
    # The bug this reader replaces: an episode whose every chunk predated the
    # frozen reference scale vanished, so stream N paired with outcome N-k.
    records = [
        {"event": "episode_boundary", "episode_index": 1, "episode_id": "t0-e0"},
        {"event": "chunk", "score": 1.0, "std_ready": False, "episode_index": 1, "episode_id": "t0-e0"},
        {"event": "episode_boundary", "episode_index": 2, "episode_id": "t0-e1"},
        {"event": "chunk", "score": 2.0, "std_ready": True, "episode_index": 2, "episode_id": "t0-e1"},
        {"event": "chunk", "score": 3.0, "std_ready": True, "episode_index": 2, "episode_id": "t0-e1"},
    ]
    # Keyed by the client's id, not by position, and the empty episode survives.
    assert list(streams_from_records(records)) == ["t0-e0", "t0-e1"]
    assert streams_from_records(records) == {"t0-e0": [], "t0-e1": [2.0, 3.0]}
    # Without an id the server counter identifies it just as unambiguously.
    anonymous = [{**r, "episode_id": None} for r in records]
    assert streams_from_records(anonymous) == {1: [], 2: [2.0, 3.0]}


def test_records_without_identity_are_refused_not_guessed():
    with pytest.raises(ValueError, match="carries neither"):
        streams_from_records([{"event": "chunk", "score": 1.0, "std_ready": True}])
    # Position remains available, explicitly, and keeps empty slots aligned.
    assert streams_from_records(
        [
            {"event": "episode_boundary"},
            {"event": "chunk", "score": 1.0, "std_ready": False},
            {"event": "episode_boundary"},
            {"event": "chunk", "score": 2.0, "std_ready": True},
        ],
        legacy_positional=True,
    ) == {0: [], 1: [2.0]}


def test_connection_boundary_does_not_consume_an_episode(tmp_path):
    runtime = AccelRuntime(_config(record=str(tmp_path / "run.jsonl")), action_horizon=N_EXEC)
    # The handler resets on every new connection; doing that through
    # start_episode would number a phantom episode and shift the whole run.
    runtime.reset_connection()
    runtime.start_episode("t0-e0")
    runtime.observe(_chunk(seed=0), score=1.0)
    runtime.start_episode("t0-e1")
    runtime.observe(_chunk(seed=1), score=2.0)

    lines = [json.loads(line) for line in (tmp_path / "run.jsonl").read_text().splitlines()]
    boundaries = [line for line in lines if line["event"] == "episode_boundary"]
    assert [line["episode_id"] for line in boundaries] == ["t0-e0", "t0-e1"]
    assert streams_from_records(lines) == {"t0-e0": [1.0], "t0-e1": [2.0]}


def test_observe_records_the_prefix_tables(tmp_path):
    runtime = AccelRuntime(_config(record=str(tmp_path / "run.jsonl")), action_horizon=N_EXEC)
    runtime.start_episode("t0-e0")
    table = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    record = runtime.observe(_chunk(), score=1.0, prefix_table=table, prefix_table_full=table)
    assert record["episode_id"] == "t0-e0"
    assert record["episode_index"] == 1
    assert record["prefix_table"] == table
    assert record["prefix_table_full"] == table


def test_observe_records_the_monitor_decision(tmp_path):
    std_path, std = _write_std(tmp_path)
    calibration = _write_calibration(tmp_path, std_path, std, h=0.001)
    runtime = AccelRuntime(_config(fixed_std=str(std_path), calibration=str(calibration)), action_horizon=N_EXEC)

    first = runtime.observe(
        _chunk(seed=0), score=1.8, monitor_info={"score": 1.8, "cusum": 0.0, "alarm": False, "onset": 0, "index": 0}
    )
    assert first["calibrated"] is True
    assert first["cusum"] == 0.0
    assert first["index"] == 0
    assert first["mode"] == "accel_prefix:3"
    assert first["num_steps"] == NUM_STEPS

    # A monitor block without an alarm key (the feedback dict shape) is recorded
    # with whatever it does carry instead of raising.
    partial = runtime.observe(_chunk(seed=1), score=9.0, monitor_info={"score": 9.0, "onset": 1, "index": 1})
    assert partial["onset"] == 1
    assert "alarm" not in partial


def test_observe_refuses_a_depth_mismatch_on_the_first_request(tmp_path):
    std_path, std = _write_std(tmp_path)
    calibration = _write_calibration(tmp_path, std_path, std)
    runtime = AccelRuntime(_config(fixed_std=str(std_path), calibration=str(calibration)), action_horizon=N_EXEC)
    assert runtime.monitor is not None

    # A different Euler schedule is a different statistic, so a calibrated
    # threshold cannot apply. This must raise on the first request rather than
    # latching a flag: a 5-hour rollout that silently stopped recording would
    # otherwise only be noticed after it finished.
    with pytest.raises(ValueError, match="different statistic"):
        runtime.observe(_chunk(steps=20), score=1.0)
    with pytest.raises(ValueError, match="different statistic"):
        runtime.observe(_chunk(steps=20), score=1.0)


def test_start_episode_resets_the_monitor_but_keeps_the_scale(tmp_path):
    std_path, std = _write_std(tmp_path)
    calibration = _write_calibration(tmp_path, std_path, std, h=0.001)
    runtime = AccelRuntime(_config(fixed_std=str(std_path), calibration=str(calibration)), action_horizon=N_EXEC)
    monitor = runtime.monitor
    monitor.update(5.0)
    assert monitor.state > 0
    runtime.observe(
        _chunk(seed=0), score=1.8, monitor_info={"score": 1.8, "cusum": 0.0, "alarm": False, "onset": 0, "index": 0}
    )
    runtime.start_episode()
    assert monitor.state == 0.0
    assert monitor.index == 0
    assert monitor.alarm_index is None
    np.testing.assert_array_equal(runtime.fixed_std, std)
    assert runtime.monitor is monitor


def test_context_capture_is_additive_and_never_touches_the_score(tmp_path):
    # The whole point of the capture path is that it is a bystander: a run with
    # it on must produce byte-identical scores and reference-scale behaviour to
    # one with it off, or the rho ground truth would describe a different policy
    # than the detection results did.
    base = AccelRuntime(_config(fixed_std="auto", warmup_chunks=2), action_horizon=N_EXEC)
    recording = AccelRuntime(
        _config(fixed_std="auto", warmup_chunks=2, context_dir=str(tmp_path / "ctx")),
        action_horizon=N_EXEC,
    )
    assert recording.capturing_context
    assert not base.capturing_context

    chunks = [_chunk(seed=index) for index in range(3)]
    for index, chunk in enumerate(chunks):
        plain = base.observe(chunk, score=float(index))
        captured = recording.observe(
            chunk,
            score=float(index),
            context={"image": {"cam": np.zeros((1, 4, 4, 3), np.uint8)}},
        )
        assert plain["score"] == captured["score"]
        assert plain["std_ready"] == captured["std_ready"]
    # Same reference scale from the same chunks: capture changed nothing.
    np.testing.assert_array_equal(base.fixed_std, recording.fixed_std)


def test_captured_context_lands_in_a_per_episode_file(tmp_path):
    runtime = AccelRuntime(_config(context_dir=str(tmp_path / "ctx"), context_stride=2), action_horizon=N_EXEC)
    runtime.start_episode("t0-e0")
    for chunk_index in range(4):
        runtime.observe(_chunk(seed=chunk_index), score=1.0, context=_context(chunk_index))
    runtime.flush_context()

    import json as _json

    from openpi.uncertainty.context import META_KEY

    path = tmp_path / "ctx" / "t0-e0.npz"
    with np.load(path) as handle:
        meta = _json.loads(bytes(np.asarray(handle[META_KEY])).decode())
    # Stride 2 over indices 0..3 keeps 0 and 2, and the chunk_index recorded in
    # the context file is the same one the jsonl carries, so a resample can join
    # an observation to the denoise path taken with it.
    assert meta["chunk_index"] == [0, 2]


def _context(seed: int) -> dict:
    return {"image": {"cam": np.full((4, 4, 3), seed % 256, dtype=np.uint8)}}


def test_context_is_optional_and_its_absence_changes_nothing(tmp_path):
    runtime = AccelRuntime(_config(context_dir=str(tmp_path / "ctx")), action_horizon=N_EXEC)
    runtime.start_episode("t0-e0")
    record = runtime.observe(_chunk(), score=1.5)  # no context supplied
    assert record["score"] == 1.5
    runtime.close()
    assert list((tmp_path / "ctx").glob("*.npz")) == []


def test_observe_rejects_a_batch():
    runtime = AccelRuntime(_config(), action_horizon=N_EXEC)
    with pytest.raises(ValueError, match="requires one observation per call"):
        runtime.observe(np.zeros((2, NUM_STEPS + 1, N_EXEC, ACTION_DIM), dtype=np.float32))


def test_provenance_describes_where_the_scale_came_from(tmp_path):
    std_path, std = _write_std(tmp_path)
    calibration = _write_calibration(tmp_path, std_path, std)
    runtime = AccelRuntime(_config(calibration=str(calibration)), action_horizon=N_EXEC)
    provenance = runtime.provenance()
    assert provenance["score"] == runtime.config.score_config()
    assert provenance["fixed_std_kind"] == "npz"
    assert provenance["fixed_std_sha256"] == digest_std(std)
    assert provenance["fixed_std_length"] == WINDOW

    # "auto" records where the frozen artifact will land.
    auto = AccelRuntime(
        _config(fixed_std="auto", warmup_chunks=1, record=str(tmp_path / "run.jsonl")), action_horizon=N_EXEC
    )
    auto.observe(_chunk(), score=1.0)
    assert auto.provenance()["fixed_std_kind"] == "auto"
    assert auto.reference_std_path() == tmp_path / REFERENCE_STD_FILENAME
