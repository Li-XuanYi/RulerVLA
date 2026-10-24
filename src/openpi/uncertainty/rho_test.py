import numpy as np
import pytest

from openpi.uncertainty import rho as _rho


def test_spearman_matches_a_known_ordering():
    # Perfectly monotone but non-linear: Spearman is 1 where Pearson is not.
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = x**3
    assert _rho.spearman(x, y) == pytest.approx(1.0)
    assert _rho.spearman(x, -y) == pytest.approx(-1.0)
    # No signal at all.
    assert abs(_rho.spearman(x, [3.0, 1.0, 5.0, 2.0, 4.0])) < 1.0


def test_spearman_averages_ties_instead_of_breaking_them_arbitrarily():
    # Two decisions with an identical divergence are genuinely tied; ranking them
    # by position would invent an order the data does not have.
    x = np.array([1.0, 2.0, 2.0, 4.0])
    y = np.array([10.0, 20.0, 20.0, 40.0])
    assert _rho.spearman(x, y) == pytest.approx(1.0)


def test_spearman_refuses_degenerate_input():
    assert np.isnan(_rho.spearman([1.0, 2.0], [1.0, 2.0]))  # too few points
    assert np.isnan(_rho.spearman([1.0] * 5, [1.0, 2.0, 3.0, 4.0, 5.0]))  # a constant


def test_prefix_curve_finds_the_depth_that_correlates_best():
    rng = np.random.default_rng(0)
    n, depth = 400, 5
    divergence = rng.normal(size=n)
    tables = rng.normal(scale=0.1, size=(n, depth))
    # Only column 2 carries the signal, so the argmax must land there.
    tables[:, 2] = divergence + rng.normal(scale=0.3, size=n)

    curve = _rho.prefix_rho_curve(tables, divergence)
    column, best = curve.best
    assert column == 2
    assert best > 0.7
    # The pre-registered depth can be read directly, whatever the peak says.
    assert curve.at(2) == pytest.approx(best)
    assert abs(curve.at(0)) < 0.2


def test_prefix_curve_checks_its_inputs():
    with pytest.raises(ValueError, match="must be \\(n, T-1\\)"):
        _rho.prefix_rho_curve(np.zeros((4, 2, 1)), np.zeros(4))
    with pytest.raises(ValueError, match="prefix rows for"):
        _rho.prefix_rho_curve(np.zeros((4, 2)), np.zeros(3))


def test_held_out_best_separates_a_real_peak_from_a_lucky_one():
    rng = np.random.default_rng(1)
    n_episodes, per_episode, depth = 40, 10, 6
    divergence = rng.normal(size=n_episodes * per_episode)
    groups = [f"e{index // per_episode}" for index in range(n_episodes * per_episode)]

    # A genuine signal at column 1, reproducible on held-out episodes.
    tables = rng.normal(scale=0.1, size=(divergence.size, depth))
    tables[:, 1] = divergence + rng.normal(scale=0.3, size=divergence.size)

    out = _rho.held_out_best(tables, divergence, groups, seed=0)
    assert out["n_episodes"] == n_episodes
    assert len(out["splits"]) == 2
    # The depth found on one half still holds up on the other.
    assert out["rho_best_held_out"] > 0.6
    for split in out["splits"]:
        assert split["chosen_column"] == 1
        assert split["rho_on_held_out_half"] > 0.6


def test_held_out_best_collapses_when_the_peak_was_pure_selection():
    rng = np.random.default_rng(2)
    n_episodes, per_episode, depth = 60, 8, 8
    divergence = rng.normal(size=n_episodes * per_episode)
    groups = [f"e{index // per_episode}" for index in range(n_episodes * per_episode)]
    # Every column is pure noise, so the in-sample peak is a fluke and the
    # held-out score must fall back towards zero.
    tables = rng.normal(size=(divergence.size, depth))

    curve = _rho.prefix_rho_curve(tables, divergence)
    _, in_sample = curve.best
    out = _rho.held_out_best(tables, divergence, groups, seed=0)
    assert in_sample > 0.05  # the fluke exists
    assert out["rho_best_held_out"] < in_sample / 2  # and does not survive


def test_held_out_best_needs_enough_episodes_to_split():
    out = _rho.held_out_best(np.zeros((6, 3)), np.zeros(6), ["a", "a", "b", "b", "c", "c"])
    assert "error" in out
