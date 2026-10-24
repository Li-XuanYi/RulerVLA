import numpy as np
import pytest

from openpi.uncertainty import cusum


def _success_streams(count: int = 10, length: int = 12, seed: int = 0) -> list[list[float]]:
    rng = np.random.default_rng(seed)
    return [list(rng.normal(1.0, 0.1, size=length)) for _ in range(count)]


def test_in_sample_uses_quantile_and_held_out_uses_the_order_statistic():
    streams = _success_streams()
    in_sample = cusum.calibrate(streams, fpr=0.1, conformal=False)
    peaks = np.array([cusum.cusum_peak(stream, in_sample["mu0"], in_sample["k"]) for stream in streams])

    # In-sample is upstream's legacy shortcut: the interpolated upper quantile.
    assert in_sample["h"] == pytest.approx(float(np.quantile(peaks, 0.9)))
    assert in_sample["calibration"] == "in-sample"
    assert in_sample["conformal"] is False

    # With M=10 and fpr=0.1 the conformal rank is ceil(11 * 0.9) = 10, i.e. the
    # maximum, which is strictly above the interpolated quantile. Mixing the two
    # up is the difference between a guarantee and an optimistic threshold.
    conformal_h = cusum.conformal_height(peaks, 0.1)
    assert conformal_h == pytest.approx(float(peaks.max()))
    assert in_sample["h"] < conformal_h


def test_held_out_protocol_defaults_to_conformal():
    scored = _success_streams(seed=1)
    calib = _success_streams(seed=2)
    fitted = cusum.calibrate(scored, calib_streams=calib, fpr=0.1, k_sigma=0.25)
    assert fitted["calibration"] == "held-out"
    assert fitted["conformal"] is True
    assert fitted["n_calib"] == len(calib)

    # mu0/sigma/h must come from the calibration set, not from the scored set.
    pooled = np.concatenate([np.asarray(stream) for stream in calib])
    assert fitted["mu0"] == pytest.approx(float(pooled.mean()))
    assert fitted["sigma"] == pytest.approx(float(pooled.std()))
    assert fitted["k"] == pytest.approx(0.25 * fitted["sigma"])


def test_from_success_streams_keeps_the_positional_signature_and_accepts_held_out():
    legacy = cusum.AccelMonitor.from_success_streams([[0.1, 0.1], [0.2, 0.1]], fpr=0.1)
    assert legacy.threshold >= 0.0
    assert legacy.mean0 == pytest.approx(0.125)

    held_out = cusum.AccelMonitor.from_success_streams(
        [[1.0, 1.0], [1.1, 0.9]], calib_streams=[[1.0, 1.0], [1.0, 1.0], [1.1, 1.0]], fpr=0.2
    )
    # Calibrated on the held-out set, so its mean is that set's mean
    # ((1 + 1 + 1 + 1 + 1.1 + 1) / 6), not the monitored streams'.
    assert held_out.mean0 == pytest.approx(6.1 / 6)
    assert held_out.k_sigma == 0.25


def test_auroc_is_fifty_fifty_for_no_signal_and_one_for_separation():
    rng = np.random.default_rng(0)
    # Identical distributions: every pair is a coin flip, so AUROC is 0.5 not 0.
    pooled = list(rng.normal(1.0, 0.1, size=160))
    score, pairs = cusum.auroc(pooled[:80], pooled[80:])
    assert pairs == 80 * 80
    assert abs(score - 0.5) < 0.1

    separated = [9.0] * 10
    perfect, _ = cusum.auroc(separated, [1.0] * 10)
    assert perfect == 1.0
    # An empty group is missing data, not a score: nan with zero pairs.
    empty, pairs = cusum.auroc([], [1.0])
    assert np.isnan(empty)
    assert pairs == 0


def test_tpr_at_fpr_and_cohen_d_use_the_success_reference():
    succ = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    # Half the failures sit above the success distribution's top decile.
    fail = [1.0, 2.0, 3.0, 4.0, 5.0, 99.0, 98.0, 97.0, 96.0, 95.0]
    assert cusum.tpr_at_fpr(fail, succ, 0.1) == pytest.approx(0.5)
    assert cusum.cohen_d(fail, succ) > 0
    assert np.isnan(cusum.cohen_d([1.0], succ))  # <2 samples has no variance

    # Ties are averaged, not dropped: a failure tied with every success counts
    # as half a win per pair rather than as a win or a loss.
    tied, pairs = cusum.auroc([2.0] * 4, [2.0] * 4)
    assert pairs == 16
    assert tied == pytest.approx(0.5)


def test_early_window_is_survival_conditioned_and_reports_its_support():
    # Failures run long (they time out); successes are short. Comparing the
    # first k chunks only among episodes that reached k is what keeps the metric
    # from measuring episode length instead of the score.
    episodes = [
        {"success": True, "scores": [1.0, 1.0, 1.0, 1.0]},
        {"success": True, "scores": [1.1, 1.0, 0.9, 1.0]},
        {"success": True, "scores": [0.9, 1.1, 1.0, 1.0]},
        {"success": False, "scores": [5.0, 6.0, 7.0, 8.0]},
        {"success": False, "scores": [4.0, 5.0, 6.0, 7.0]},
    ]
    rows = cusum.early_window_summary(episodes, early_ks=(1, 2, 4), min_survivors=1)
    assert [row["k"] for row in rows] == [1, 2, 4]
    assert all(row["auroc"] == 1.0 for row in rows)
    # Support shrinks as k grows past the short successes' length -- reported,
    # so a reader can see the estimate rests on fewer episodes.
    assert [row["n_surv"] for row in rows] == [5, 5, 5]

    # Below the survivor floor the row is dropped rather than reported as noise.
    assert cusum.early_window_summary(episodes, early_ks=(1,), min_survivors=99) == []


def test_evaluate_streams_reports_tpr_lead_and_specificity():
    successes = _success_streams(count=10, length=12, seed=3)
    # Failures drift upward late in the episode.
    failures = []
    for index in range(6):
        rng = np.random.default_rng(100 + index)
        stream = list(rng.normal(1.0, 0.1, size=8)) + [12.0] * 4
        failures.append(stream)

    metrics = cusum.evaluate_streams(successes[:5], failures, calib_streams=successes[5:], fpr=0.1, k_sigma=0.25)
    assert metrics["tpr"] == 1.0
    assert metrics["calibration"] == "held-out"
    assert metrics["conformal"] is True
    assert metrics["n_fail"] == 6
    assert metrics["n_succ"] == 5
    # The spike starts at index 8 of 12, and it clears h on its own chunk, so the
    # alarm fires there: lead = 12 - 1 - 8 chunks still executable, alarm at 8/12
    # of the episode.
    assert metrics["median_lead"] == pytest.approx(3.0)
    assert metrics["median_alarm_frac"] == pytest.approx(8 / 12)
    assert 0.0 <= metrics["succ_fire_rate"] <= 1.0
    assert metrics["h"] > 0


def test_evaluate_streams_without_a_class_returns_nan_metrics():
    metrics = cusum.evaluate_streams(_success_streams(), [], fpr=0.1)
    assert np.isnan(metrics["tpr"])
    assert metrics["n_fail"] == 0


def test_monitor_tracks_onset_index_and_alarm_position():
    monitor = cusum.AccelMonitor(mean0=1.0, std0=0.0, threshold=1.0)
    for _ in range(3):
        info = monitor.update(1.0)
        assert info["alarm"] is False
        assert info["cusum"] == pytest.approx(0.0)
    assert info["index"] == 2
    # Every zero state resets the change-point estimate to the current chunk.
    assert info["onset"] == 2

    spike = monitor.update(5.0)
    assert spike["index"] == 3
    assert spike["onset"] == 2  # the last chunk before the sum left zero
    assert spike["cusum"] == pytest.approx(4.0)
    assert spike["alarm"] is True
    assert monitor.alarm_index == 3
    # A later alarm does not move the first alarm's position.
    monitor.update(50.0)
    assert monitor.alarm_index == 3


def test_monitor_reset_clears_state_but_keeps_calibration():
    monitor = cusum.AccelMonitor(mean0=1.0, std0=0.5, threshold=2.0, k_sigma=0.25)
    monitor.update(10.0)
    assert monitor.state > 0
    monitor.reset()
    assert monitor.state == 0.0
    assert monitor.index == 0
    assert monitor.onset == 0
    assert monitor.alarm_index is None
    # The fitted reference survives the episode boundary.
    assert monitor.mean0 == 1.0
    assert monitor.std0 == 0.5
    assert monitor.threshold == 2.0
    assert monitor.slack == pytest.approx(0.125)


def test_monitor_online_agrees_with_the_offline_cusum():
    stream = [*np.random.default_rng(7).normal(1.0, 0.1, size=10), 9.0, 9.0]
    monitor = cusum.AccelMonitor(mean0=1.0, std0=0.1, threshold=0.5, k_sigma=0.25)
    online_alarms = [monitor.update(value)["alarm"] for value in stream]
    alarm_index, onset = cusum.cusum_alarm(stream, threshold=0.5, mean0=1.0, slack=0.1 * 0.25)
    assert online_alarms.index(True) == alarm_index
    assert monitor.alarm_index == alarm_index
    assert onset is not None


def test_score_frozen_reports_realized_rates_without_fitting():
    # mu0=0, k=0 makes the state a running sum, so h=1 is cleared on the first
    # chunk of a 9.0 stream and never by a 0.0 one.
    scored = cusum.score_frozen([[0.0, 0.0], [0.0, 0.0]], [[9.0, 9.0]], mu0=0.0, k=0.0, h=1.0)
    assert scored["n_succ"] == 2
    assert scored["n_succ_fire"] == 0
    assert scored["succ_fire_rate"] == 0.0
    assert scored["tpr"] == 1.0
    # The alarm lands on chunk 0 of 2, so one chunk was still executable.
    assert scored["median_lead"] == pytest.approx(1.0)

    # Nothing is derived here, so an empty class is nan with a count beside it
    # rather than a silently zero rate.
    empty = cusum.score_frozen([[0.0], [0.0]], [], mu0=0.0, k=0.0, h=1.0)
    assert np.isnan(empty["tpr"])
    assert empty["n_fail"] == 0


def test_score_frozen_reproduces_evaluate_streams_at_the_fitted_point():
    successes = _success_streams(count=10, length=12, seed=5)
    failures = [[5.0] * 12 for _ in range(4)]
    metrics = cusum.evaluate_streams(successes[:5], failures, calib_streams=successes[5:], fpr=0.1)
    frozen = cusum.score_frozen(successes[:5], failures, mu0=metrics["mu0"], k=metrics["k"], h=metrics["h"])

    # evaluate_streams is fit-then-score; the scoring half must be the same code,
    # or the two drift and a reported TPR depends on which entry point ran.
    assert frozen["n_succ"] == metrics["n_succ"]
    assert frozen["n_fail_fire"] == metrics["n_fail_fire"]
    assert frozen["succ_fire_rate"] == pytest.approx(metrics["succ_fire_rate"])
    assert frozen["tpr"] == pytest.approx(metrics["tpr"])
    assert frozen["median_lead"] == pytest.approx(metrics["median_lead"])
