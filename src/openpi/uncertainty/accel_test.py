import logging

import numpy as np
import pytest

from openpi.uncertainty import accel
from openpi.uncertainty import cusum


def test_straight_flow_has_zero_accel():
    # One action position, constant velocity over four Euler intervals.
    path = np.arange(5, dtype=np.float32)[None, :, None, None]
    scores = accel.score_chunks(path, action_dim=1, mode="accel")
    np.testing.assert_allclose(scores, 0.0, atol=1e-6)


def test_curved_flow_has_positive_accel_and_prefix_shape():
    # The increments change from 1 to 2, hence the denoising tangent bends.
    path = np.array([0.0, 1.0, 3.0, 6.0, 10.0], dtype=np.float32)[None, :, None, None]
    scores = accel.score_chunks(path, action_dim=1, mode="accel")
    prefixes = accel.whole_chunk_prefix_accel(path, action_dim=1)
    assert scores.shape == (1,)
    assert prefixes.shape == (1, 3)
    assert scores[0] > 0
    np.testing.assert_allclose(prefixes[:, -1], scores, atol=1e-6)


def test_executed_window_and_fixed_scale_are_checked():
    path = np.zeros((2, 5, 3, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="fixed_std shape"):
        accel.score_chunks(
            path,
            action_dim=2,
            n_exec=2,
            mode="accel_exec",
            fixed_std=np.ones(2, dtype=np.float32),
        )


def test_exec_window_modes_require_n_exec():
    path = np.zeros((2, 5, 3, 2), dtype=np.float32)
    for mode in ("accel_exec", "accel_prefix:3"):
        with pytest.raises(ValueError, match="requires|executed-window"):
            accel.score_chunks(path, action_dim=2, mode=mode)
    # accel_prefix_full is scored over the whole horizon, so n_exec is not part
    # of its definition. Upstream's startswith(("accel_exec", "accel_prefix"))
    # test would wrongly flag it; this is the regression guard for that.
    score = accel.score_chunks(path, action_dim=2, mode="accel_prefix_full:3")
    assert score.shape == (2,)


def test_default_mode_reads_depth_five():
    assert accel.DEFAULT_ACCEL_MODE == "accel_prefix:3"
    assert accel.prefix_depth(3, 9) == 5
    assert accel.prefix_depth(-1, 9) == 10
    assert accel.prefix_depth(0, 9) == 2
    assert accel.prefix_layers(10) == 9
    with pytest.raises(IndexError):
        accel.prefix_depth(9, 9)


def test_prefix_table_matches_score_chunks_at_every_column():
    # The policy keeps the whole table and derives the same scalar from it, so a
    # divergence here would mean the recorded stream and the served score are
    # two different statistics.
    path = np.random.default_rng(3).normal(size=(2, 5, 3, 2)).astype(np.float32)
    table = accel.score_prefix_table(path, action_dim=2, n_exec=2)
    assert table.shape == (2, 3)
    for column in (-3, -2, -1, 0, 1, 2):
        np.testing.assert_array_equal(
            accel.score_chunks(path, action_dim=2, n_exec=2, mode=f"accel_prefix:{column}"),
            table[:, column],
        )


def test_prefix_table_full_horizon_ignores_the_executed_window():
    path = np.random.default_rng(4).normal(size=(1, 5, 3, 2)).astype(np.float32)
    full = accel.score_prefix_table(path, action_dim=2)
    windowed = accel.score_prefix_table(path, action_dim=2, n_exec=2)
    # Same shape (both are (batch, T-1)) but different statistics: the windowed
    # table drops the third action position before flattening. Conflating them
    # is the silent error this pair of tables exists to prevent.
    assert full.shape == windowed.shape
    assert not np.allclose(full, windowed)
    np.testing.assert_array_equal(
        full[:, -1],
        accel.score_chunks(path, action_dim=2, mode="accel_prefix_full:-1"),
    )


def test_prefix_index_for_mode_downgrades_instead_of_rescoring():
    assert accel.prefix_index_for_mode("accel_prefix:3", 9) == 3
    assert accel.prefix_index_for_mode("accel_prefix:-1", 9) == 8
    # Column 7 does not exist for a 3-layer table; it must land on the last one.
    with pytest.raises(ValueError, match="does not index a prefix table"):
        accel.prefix_index_for_mode("accel_exec", 9)
    with pytest.raises(ValueError, match="requires a denoise depth"):
        accel.prefix_index_for_mode("accel_prefix", 9)


def test_resolve_accel_mode_downgrades_shallow_samplers():
    # num_steps=4 leaves 3 prefix layers (p=2..4), so depth 3 cannot be indexed.
    assert accel.resolve_accel_mode("accel_prefix:3", 4) == "accel_prefix:-1"
    assert accel.resolve_accel_mode("accel_prefix:3", 10) == "accel_prefix:3"
    # Only accel_prefix is rewritten; the full-horizon variant is depth-safe.
    assert accel.resolve_accel_mode("accel_prefix_full:3", 4) == "accel_prefix_full:3"
    assert accel.resolve_accel_mode("accel_exec", 4) == "accel_exec"


def test_out_of_range_prefix_logs_and_scores_the_last_layer(caplog):
    path = np.random.default_rng(0).normal(size=(1, 5, 3, 2)).astype(np.float32)
    with caplog.at_level(logging.WARNING, logger="openpi.uncertainty.accel"):
        deep = accel.score_chunks(path, action_dim=2, n_exec=2, mode="accel_prefix:7")
    assert "only 3 prefix layer(s)" in caplog.text
    # The fallback must be the *same column* the explicit deepest mode reads, so
    # this is exact, not approximate: a downgrade that perturbed the score would
    # mean the warning was hiding a different statistic.
    last = accel.score_chunks(path, action_dim=2, n_exec=2, mode="accel_prefix:-1")
    np.testing.assert_array_equal(deep, last)
    # The last prefix layer and accel_exec are the same statistic (p = T) reached
    # through different code paths -- a cumulative-ratio table versus a direct
    # sum-of-norms -- so they agree to float32 rounding, not bitwise.
    exec_score = accel.score_chunks(path, action_dim=2, n_exec=2, mode="accel_exec")
    np.testing.assert_allclose(last, exec_score, rtol=1e-6)


def test_reference_std_matches_run_pooled_std():
    rng = np.random.default_rng(1)
    chunks = [rng.normal(size=(1, 5, 3, 2)).astype(np.float32) for _ in range(4)]
    reference = accel.ReferenceStd(action_dim=2, n_exec=2)
    for chunk in chunks:
        reference.update(chunk)
    assert reference.n_chunks == 4

    # Independently re-derive the windowed, run-pooled scale fm-geometry uses.
    flats = [chunk[:, :, :2, :2].reshape(chunk.shape[0], chunk.shape[1], -1) for chunk in chunks]
    expected = np.concatenate(flats, axis=0).std(axis=(0, 1)).astype(np.float32)
    np.testing.assert_array_equal(reference.freeze(), expected)
    # freeze() drops the buffer so a long-running server does not accumulate it.
    with pytest.raises(ValueError, match="no chunks accumulated"):
        reference.freeze()


def test_reference_std_rejects_a_changing_window():
    reference = accel.ReferenceStd(action_dim=2, n_exec=2)
    reference.update(np.zeros((1, 5, 3, 2), dtype=np.float32))
    # Reconfiguring the window mid-accumulation would pool two different
    # flattened coordinate spaces into one reference scale.
    reference.n_exec = 3
    with pytest.raises(ValueError, match="flattened dimension changed"):
        reference.update(np.zeros((1, 5, 3, 2), dtype=np.float32))


def test_fixed_std_npz_round_trip(tmp_path):
    path = tmp_path / "accel_fixed_std.npz"
    std = np.linspace(0.5, 1.5, 6, dtype=np.float32)
    accel.save_fixed_std(path, std)
    np.testing.assert_array_equal(accel.load_fixed_std(path), std)
    np.testing.assert_array_equal(accel.load_fixed_std(path, expected_len=6), std)

    with pytest.raises(ValueError, match="but this window needs 4"):
        accel.load_fixed_std(path, expected_len=4)
    with pytest.raises(FileNotFoundError):
        accel.load_fixed_std(tmp_path / "missing.npz")

    # A wrong key must fail loudly: silently reading the first key would return a
    # wrong-scale vector that nothing downstream can detect.
    wrong_key = tmp_path / "wrong_key.npz"
    np.savez_compressed(wrong_key, scale=std)
    with pytest.raises(ValueError, match="must contain 'std'"):
        accel.load_fixed_std(wrong_key)


def test_causal_accel_monitor_and_conformal_height():
    monitor = cusum.AccelMonitor.from_success_streams([[0.1, 0.1], [0.2, 0.1]], fpr=0.1)
    assert monitor.threshold >= 0.0
    first = monitor.update(0.1)
    assert first["alarm"] is False
    alarm = monitor.update(10.0)
    assert alarm["cusum"] > first["cusum"]
    assert alarm["alarm"] is True

    index, onset = cusum.cusum_alarm([0.1, 10.0], threshold=0.5, mean0=0.1, slack=0.0)
    assert index == 1
    assert onset == 0
