import json

import numpy as np
import pytest

from openpi.models import model as _model
from openpi.uncertainty import context as ctx


def _observation_inputs(seed: int = 0, *, height: int = 8, width: int = 8) -> dict:
    """A stand-in for the dict the policy's input transforms produce."""

    rng = np.random.default_rng(seed)
    images = {
        "base_0_rgb": rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8),
        "left_wrist_0_rgb": rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8),
    }
    return {
        "image": images,
        "image_mask": {name: np.ones((), dtype=bool) for name in images},
        "state": rng.normal(size=(8,)).astype(np.float32),
        "tokenized_prompt": rng.integers(0, 100, size=(12,), dtype=np.int32),
        "tokenized_prompt_mask": np.ones((12,), dtype=bool),
        # Rides along in the real inputs dict and must not be captured.
        "actions": rng.normal(size=(10, 7)).astype(np.float32),
    }


def test_capture_keeps_only_what_the_observation_needs():
    captured = ctx.capture_context(_observation_inputs())
    assert set(captured) == {"image", "image_mask", "state", "tokenized_prompt", "tokenized_prompt_mask"}
    # uint8 is preserved rather than converted: from_dict owns the conversion, so
    # converting here would freeze a choice that the live path makes elsewhere.
    assert captured["image"]["base_0_rgb"].dtype == np.uint8
    assert captured["state"].dtype == np.float32
    # Nothing may alias live device/array memory.
    source = _observation_inputs()
    captured = ctx.capture_context(source)
    source["image"]["base_0_rgb"][...] = 0
    assert captured["image"]["base_0_rgb"].any()


def test_capture_refuses_inputs_it_cannot_reconstruct():
    broken = _observation_inputs()
    del broken["image_mask"]
    with pytest.raises(ValueError, match="missing \\['image_mask'\\]"):
        ctx.capture_context(broken)


def test_recorder_round_trips_a_context_bitwise(tmp_path):
    recorder = ctx.ContextRecorder(tmp_path, stride=1)
    recorder.start_episode("t0-e0")
    for chunk_index in range(3):
        recorder.add(chunk_index, ctx.capture_context(_observation_inputs(seed=chunk_index)))
    recorder.flush()

    for chunk_index in range(3):
        loaded = ctx.load_context_dict(tmp_path / "t0-e0.npz", chunk_index)
        original = ctx.capture_context(_observation_inputs(seed=chunk_index))
        for key in ("state", "tokenized_prompt", "tokenized_prompt_mask"):
            np.testing.assert_array_equal(loaded[key][0], original[key])
        for name in original["image"]:
            np.testing.assert_array_equal(loaded["image"][name][0], original["image"][name])
        # The batch axis matches what the policy adds before the model sees it.
        assert loaded["state"].shape == (1, 8)
        assert loaded["image"]["base_0_rgb"].shape == (1, 8, 8, 3)


def test_round_trip_survives_observation_from_dict(tmp_path):
    # The invariant that matters: captured -> stored -> loaded -> Observation has
    # to equal the live path, or the resample is conditioned on a different
    # observation and the ground truth silently measures the wrong posterior.
    recorder = ctx.ContextRecorder(tmp_path)
    recorder.start_episode("t0-e0")
    recorder.add(0, ctx.capture_context(_observation_inputs(seed=7)))
    recorder.flush()

    loaded = ctx.load_context_dict(tmp_path / "t0-e0.npz", 0)
    # The live path: transform output, then the batch axis the policy adds.
    live = _observation_inputs(seed=7)
    live_batched = {
        key: {k: np.asarray(v)[None, ...] for k, v in value.items()}
        if isinstance(value, dict)
        else np.asarray(value)[None, ...]
        for key, value in live.items()
    }

    rebuilt = _model.Observation.from_dict(loaded)
    expected = _model.Observation.from_dict(live_batched)
    for name in expected.images:
        np.testing.assert_array_equal(rebuilt.images[name], expected.images[name])
        np.testing.assert_array_equal(rebuilt.image_masks[name], expected.image_masks[name])
    np.testing.assert_array_equal(rebuilt.state, expected.state)
    np.testing.assert_array_equal(rebuilt.tokenized_prompt, expected.tokenized_prompt)


def test_stride_records_a_subset_and_keeps_the_index(tmp_path):
    recorder = ctx.ContextRecorder(tmp_path, stride=3)
    recorder.start_episode("t0-e0")
    for chunk_index in range(9):
        recorder.add(chunk_index, ctx.capture_context(_observation_inputs(seed=chunk_index)))
    recorder.flush()

    meta = ctx.read_context_meta(tmp_path / "t0-e0.npz")
    assert meta["chunk_index"] == [0, 3, 6]
    # A skipped decision must be refused loudly rather than silently returning a
    # neighbour's observation.
    with pytest.raises(KeyError, match="no context for chunk_index=1"):
        ctx.load_context_dict(tmp_path / "t0-e0.npz", 1)


def test_episode_boundary_flushes_and_close_is_idempotent(tmp_path):
    recorder = ctx.ContextRecorder(tmp_path, stride=1)
    recorder.start_episode("t0-e0")
    recorder.add(0, ctx.capture_context(_observation_inputs()))
    # Starting the next episode must not lose the previous one's buffered rows.
    recorder.start_episode("t0-e1")
    recorder.add(0, ctx.capture_context(_observation_inputs(seed=1)))
    recorder.close()
    recorder.close()

    assert ctx.read_context_meta(tmp_path / "t0-e0.npz")["n_rows"] == 1
    assert ctx.read_context_meta(tmp_path / "t0-e1.npz")["n_rows"] == 1
    # After close, further captures are dropped rather than written into a file
    # whose index was already persisted.
    recorder.add(1, ctx.capture_context(_observation_inputs(seed=2)))
    assert ctx.read_context_meta(tmp_path / "t0-e1.npz")["n_rows"] == 1


def test_missing_episode_id_drops_context_instead_of_misnaming_it(tmp_path):
    recorder = ctx.ContextRecorder(tmp_path)
    recorder.start_episode(None)
    recorder.add(0, ctx.capture_context(_observation_inputs()))
    recorder.close()
    assert list(tmp_path.glob("*.npz")) == []


def test_a_field_that_changes_shape_inside_one_episode_is_refused(tmp_path):
    recorder = ctx.ContextRecorder(tmp_path)
    recorder.start_episode("t0-e0")
    recorder.add(0, ctx.capture_context(_observation_inputs(height=8)))
    recorder.add(1, ctx.capture_context(_observation_inputs(height=16)))
    with pytest.raises(ValueError, match="changed shape inside one episode"):
        recorder.flush()


def test_replay_loads_the_noise_and_chunk_alongside_the_observation(tmp_path):
    trajectory = np.random.default_rng(5).normal(size=(11, 10, 7)).astype(np.float32)
    recorder = ctx.ContextRecorder(tmp_path)
    recorder.start_episode("t0-e0")
    recorder.add(0, ctx.capture_context(_observation_inputs(seed=1)), trajectory=trajectory)
    recorder.flush()

    observation, noise, chunk = ctx.load_replay(tmp_path / "t0-e0.npz", 0)
    # x_t is [noise, ...Euler iterates...]: the two endpoints are exactly what a
    # replay needs, so they must survive bit for bit.
    np.testing.assert_array_equal(noise, trajectory[0])
    np.testing.assert_array_equal(chunk, trajectory[-1])
    assert observation["state"].shape == (1, 8)
    # The denoise path itself is not stored: only its endpoints are needed.
    assert "noise" not in observation
    assert "chunk" not in observation


def test_replay_refuses_a_capture_without_the_trajectory(tmp_path):
    recorder = ctx.ContextRecorder(tmp_path)
    recorder.start_episode("t0-e0")
    recorder.add(0, ctx.capture_context(_observation_inputs()))
    recorder.flush()
    # A fidelity gate with a fabricated noise would prove nothing, so a capture
    # that predates trajectory recording is refused rather than approximated.
    with pytest.raises(ValueError, match="recorded without the denoise trajectory"):
        ctx.load_replay(tmp_path / "t0-e0.npz", 0)


def test_a_batched_trajectory_is_accepted_and_its_batch_axis_dropped(tmp_path):
    recorder = ctx.ContextRecorder(tmp_path)
    recorder.start_episode("t0-e0")
    trajectory = np.random.default_rng(6).normal(size=(1, 11, 10, 7)).astype(np.float32)
    recorder.add(0, ctx.capture_context(_observation_inputs()), trajectory=trajectory)
    recorder.flush()
    _, noise, chunk = ctx.load_replay(tmp_path / "t0-e0.npz", 0)
    assert noise.shape == (10, 7)
    np.testing.assert_array_equal(chunk, trajectory[0, -1])


def test_meta_is_json_serialisable_and_describes_the_arrays(tmp_path):
    recorder = ctx.ContextRecorder(tmp_path)
    recorder.start_episode("t0-e0")
    recorder.add(0, ctx.capture_context(_observation_inputs()))
    recorder.flush()
    meta = ctx.read_context_meta(tmp_path / "t0-e0.npz")
    json.dumps(meta)  # must survive a round trip through json
    assert meta["dtypes"]["image__base_0_rgb"] == "uint8"
    assert meta["shapes"]["image__base_0_rgb"] == [8, 8, 3]
    assert meta["shapes"]["state"] == [8]


# ---- TrajectoryRecorder --------------------------------------------------


def _denoise_path(steps: int = 10, horizon: int = 10, dim: int = 7, seed: int = 0) -> np.ndarray:
    """A path shaped like a real sampler's: O(1) iterates reached by small steps.

    The smallness is the point. Accel is a ratio over the differences of adjacent
    iterates, and a path whose steps were O(1) would hide a precision problem that
    a real path -- 10 Euler steps from N(0, 1) noise down to an O(1) action --
    has in full.
    """

    rng = np.random.default_rng(seed)
    start = rng.normal(size=(horizon, dim)).astype(np.float32)
    increments = (rng.normal(size=(steps, horizon, dim)) * 0.1).astype(np.float32)
    return np.concatenate([start[None], start[None] + np.cumsum(increments, axis=0)]).astype(np.float32)


def test_trajectory_keeps_the_path_and_velocity_bitwise(tmp_path):
    path = _denoise_path(seed=1)
    velocity = np.diff(path, axis=0).astype(np.float32)
    recorder = ctx.TrajectoryRecorder(tmp_path)
    recorder.start_episode("t0-e0")
    recorder.add(3, path, velocity)
    recorder.flush()

    with np.load(tmp_path / "t0-e0.npz", allow_pickle=False) as handle:
        meta = json.loads(bytes(np.asarray(handle[ctx.META_KEY])).decode())
        np.testing.assert_array_equal(handle[ctx.PATH_KEY][0], path)
        np.testing.assert_array_equal(handle[ctx.VELOCITY_KEY][0], velocity)
        assert handle[ctx.OBS_INDEX_KEY].tolist() == [0]
    # The chunk_index is the join key to the per-chunk score record, so it has to
    # come back as the number the server logged, not as a row position.
    assert meta["chunk_index"] == [3]
    assert meta["dtypes"][ctx.PATH_KEY] == "float32"


def test_the_default_dtype_is_the_one_that_survives_a_difference(tmp_path):
    """Pins *why* the default is float32, so it is not optimised back to float16.

    A coarse dtype's relative error on the iterates lands on the difference
    almost undiluted, because the difference is an order of magnitude smaller.
    Measured on a real 90-chunk round trip, float16 reproduced the live score to
    only ~1e-2 relative -- enough to move a chunk across a threshold.
    """

    path = _denoise_path(seed=2)
    for dtype, survives in (("float32", True), ("float16", False)):
        recorder = ctx.TrajectoryRecorder(tmp_path / dtype, dtype=dtype)
        recorder.start_episode("t0-e0")
        recorder.add(0, path)
        recorder.flush()
        with np.load(tmp_path / dtype / "t0-e0.npz", allow_pickle=False) as handle:
            stored = np.asarray(handle[ctx.PATH_KEY][0])
        assert np.array_equal(np.diff(stored, axis=0), np.diff(path, axis=0)) is survives, dtype


def test_a_velocity_that_does_not_match_the_path_is_refused(tmp_path):
    recorder = ctx.TrajectoryRecorder(tmp_path)
    recorder.start_episode("t0-e0")
    path = _denoise_path(steps=10)  # (11, H, D), so v_t must be (10, H, D)
    # Checked at record time rather than discovered at analysis time, when the
    # rollout compute that produced the mismatch has already been spent.
    with pytest.raises(ValueError, match="different denoise schedules"):
        recorder.add(0, path, np.zeros((9, 10, 7), dtype=np.float32))


def test_a_batched_pair_is_accepted_and_its_batch_axis_dropped(tmp_path):
    path = _denoise_path(seed=3)
    velocity = np.diff(path, axis=0).astype(np.float32)
    recorder = ctx.TrajectoryRecorder(tmp_path)
    recorder.start_episode("t0-e0")
    recorder.add(0, path[None], velocity[None])
    recorder.flush()

    with np.load(tmp_path / "t0-e0.npz", allow_pickle=False) as handle:
        assert handle[ctx.PATH_KEY].shape == (1, 11, 10, 7)
        np.testing.assert_array_equal(handle[ctx.PATH_KEY][0], path)
        np.testing.assert_array_equal(handle[ctx.VELOCITY_KEY][0], velocity)


def test_obs_index_counts_decisions_and_restarts_with_the_episode(tmp_path):
    recorder = ctx.TrajectoryRecorder(tmp_path)
    recorder.start_episode("t0-e0")
    for chunk_index in (0, 5, 6):
        recorder.add(chunk_index, _denoise_path(seed=chunk_index))
    recorder.start_episode("t0-e1")
    recorder.add(0, _denoise_path(seed=9))
    recorder.close()

    with np.load(tmp_path / "t0-e0.npz", allow_pickle=False) as handle:
        # Position within the episode, not the chunk index: a run that skipped
        # decisions must not make the gap look like a shorter episode.
        assert handle[ctx.OBS_INDEX_KEY].tolist() == [0, 1, 2]
    with np.load(tmp_path / "t0-e1.npz", allow_pickle=False) as handle:
        assert handle[ctx.OBS_INDEX_KEY].tolist() == [0]


def test_an_explicit_obs_index_overrides_only_that_decision(tmp_path):
    recorder = ctx.TrajectoryRecorder(tmp_path)
    recorder.start_episode("t0-e0")
    recorder.add(0, _denoise_path(seed=4), obs_index=7)
    recorder.add(1, _denoise_path(seed=5))
    recorder.flush()

    with np.load(tmp_path / "t0-e0.npz", allow_pickle=False) as handle:
        assert handle[ctx.OBS_INDEX_KEY].tolist() == [7, 8]


def test_a_decision_without_an_episode_id_is_dropped(tmp_path):
    recorder = ctx.TrajectoryRecorder(tmp_path)
    recorder.start_episode(None)
    recorder.add(0, _denoise_path())
    recorder.close()
    assert list(tmp_path.glob("*.npz")) == []


def test_buffered_decisions_merge_into_one_file_as_the_buffer_fills(tmp_path):
    recorder = ctx.TrajectoryRecorder(tmp_path, max_buffered=2)
    recorder.start_episode("t0-e0")
    for chunk_index in range(3):
        recorder.add(chunk_index, _denoise_path(seed=chunk_index))

    # Two decisions filled the buffer; the third is still held. The second flush
    # must merge into the first's file rather than replace it, or a long episode
    # would keep only its last buffer's worth of decisions.
    assert ctx.read_context_meta(tmp_path / "t0-e0.npz")["n_rows"] == 2
    recorder.close()
    assert ctx.read_context_meta(tmp_path / "t0-e0.npz")["n_rows"] == 3

