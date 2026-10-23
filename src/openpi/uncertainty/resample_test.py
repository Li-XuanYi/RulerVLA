import numpy as np
import pytest

from openpi.uncertainty import resample as rs

ACTION_DIM = 3
N_EXEC = 4
HORIZON = 6
MODEL_DIM = 5
STEPS = 3


def _step_function(endpoint, *, noise_scale: float = 0.0):
    """A sampler stub whose final path entry is ``endpoint(noise)``.

    Mimics ``x_t``'s layout: index 0 is the noise that was passed in and index -1
    is the produced chunk, which is what both the fidelity gate and the spread
    read.
    """

    calls = []

    def sample(observation, *, noise, num_steps):
        noise = np.asarray(noise)
        calls.append({"observation": observation, "shape": noise.shape, "num_steps": num_steps})
        batch = noise.shape[0]
        final = endpoint(noise, batch)[:, None, :, :]
        # Constant along the path: only the endpoints matter here.
        path = np.repeat(final, num_steps + 1, axis=1)
        path[:, 0] = noise
        return {"x_t": path, "v_t": np.zeros_like(path[:, 1:]), "time": np.zeros(num_steps)}

    sample.calls = calls
    return sample


def _session(sample, **overrides):
    payload = {
        "sample_trajectory": sample,
        "action_dim": ACTION_DIM,
        "n_exec": N_EXEC,
        "num_steps": STEPS,
        "action_horizon": HORIZON,
        "model_action_dim": MODEL_DIM,
    }
    payload.update(overrides)
    return rs.ResampleSession(**payload)


def test_reproduce_passes_when_the_replay_returns_the_recorded_chunk():
    chunk = np.random.default_rng(0).normal(size=(HORIZON, MODEL_DIM)).astype(np.float32)
    sample = _step_function(lambda noise, batch: np.repeat(chunk[None], batch, axis=0))
    session = _session(sample)

    out = session.reproduce({}, np.zeros((HORIZON, MODEL_DIM), np.float32), chunk)
    assert out["max_abs_err"] == pytest.approx(0.0, abs=1e-7)
    assert out["ok"] is True
    # The gate must replay at B=1: the recording was B=1, and a batched replay
    # would blur the one comparison whose job is exactness.
    assert sample.calls[-1]["shape"] == (1, HORIZON, MODEL_DIM)
    assert sample.calls[-1]["num_steps"] == STEPS


def test_reproduce_fails_loudly_on_a_drifted_observation():
    recorded = np.zeros((HORIZON, MODEL_DIM), np.float32)
    # The replayed policy now produces something else entirely, which is what a
    # drifted observation looks like from the outside.
    sample = _step_function(lambda noise, batch: np.full((batch, HORIZON, MODEL_DIM), 1.0, np.float32))
    session = _session(sample)

    out = session.reproduce({}, np.zeros((HORIZON, MODEL_DIM), np.float32), recorded)
    assert out["max_abs_err"] == pytest.approx(1.0)
    assert out["ok"] is False


def test_reproduce_rejects_a_mismatched_noise_shape():
    session = _session(_step_function(lambda noise, batch: np.zeros((batch, HORIZON, MODEL_DIM), np.float32)))
    with pytest.raises(ValueError, match="recorded noise has shape"):
        session.reproduce({}, np.zeros((HORIZON + 1, MODEL_DIM), np.float32), np.zeros((HORIZON, MODEL_DIM)))


def test_resample_draws_k_candidates_in_one_batched_pass():
    sample = _step_function(lambda noise, batch: noise[:, :HORIZON, :MODEL_DIM])
    session = _session(sample)

    candidates = session.resample({}, 8, seed=0)
    # Only the executed window and the real action dims are returned: scoring the
    # positions that will never be executed adds noise to the spread.
    assert candidates.shape == (8, N_EXEC, ACTION_DIM)
    # One call, not k: the prefix cache is built once and the denoise batched.
    assert len(sample.calls) == 1
    assert sample.calls[0]["shape"] == (8, HORIZON, MODEL_DIM)
    assert len({tuple(row.ravel()) for row in candidates}) == 8


def test_resample_is_reproducible_from_the_seed():
    sample = _step_function(lambda noise, batch: noise[:, :HORIZON, :MODEL_DIM])
    session = _session(sample)
    first = session.resample({}, 4, seed=7)
    second = session.resample({}, 4, seed=7)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, session.resample({}, 4, seed=8))


def test_resample_needs_at_least_two_candidates():
    session = _session(_step_function(lambda noise, batch: noise[:, :HORIZON, :MODEL_DIM]))
    with pytest.raises(ValueError, match="k must be >= 2"):
        session.resample({}, 1, seed=0)


def test_pairwise_spread_is_the_max_over_pairs_not_an_average():
    # Two tight clusters far apart: a variance would report something that looks
    # confident, the max-pair distance reports the separation.
    tight = np.zeros((4, 2, 2), np.float32)
    far = np.full((4, 2, 2), 10.0, np.float32)
    candidates = np.concatenate([tight, far], axis=0)
    maximum, mean = rs.pairwise_spread(candidates)
    assert maximum == pytest.approx(np.sqrt(2) * 10.0, rel=1e-5)
    assert mean < maximum
    # A single candidate has no pairs, so no spread -- not an error.
    assert rs.pairwise_spread(np.zeros((1, 2, 2), np.float32)) == (0.0, 0.0)


def test_pairwise_spread_standardizes_before_measuring():
    # Dimension 0 varies 100x more than dimension 1. Unstandardized, the distance
    # is entirely dimension 0 and the second dimension's disagreement vanishes.
    candidates = np.zeros((2, 1, 2), np.float32)
    candidates[1, 0, 0] = 100.0
    candidates[1, 0, 1] = 1.0
    raw, _ = rs.pairwise_spread(candidates)
    scaled, _ = rs.pairwise_spread(candidates, scale=np.array([100.0, 1.0], np.float32))
    assert raw == pytest.approx(100.005, rel=1e-4)
    # Both dimensions now contribute comparably instead of one dominating.
    assert scaled == pytest.approx(np.sqrt(2), rel=1e-4)


def test_pooled_dim_scale_winsorizes_a_diverged_rollout():
    clean = np.random.default_rng(0).normal(size=(500, 3)).astype(np.float32)
    diverged = clean.copy()
    # One catastrophically blown-up sample, as a broken rollout produces.
    diverged[0] = 1e6
    plain = np.concatenate([clean, diverged[:1]]).std(axis=0)
    robust = rs.pooled_dim_scale(np.concatenate([clean, diverged[:1]]))
    # A plain std is dominated by the outlier; the winsorized one is not.
    assert robust.max() < plain.max() / 100
    np.testing.assert_allclose(robust, clean.std(axis=0), rtol=0.2)


def test_pooled_dim_scale_leaves_a_constant_dimension_unscaled():
    samples = np.zeros((10, 3), np.float32)
    samples[:, 0] = np.linspace(0, 1, 10)
    scale = rs.pooled_dim_scale(samples)
    assert scale[0] > 0
    # Dividing by ~0 would make an informationless dimension dominate everything.
    assert scale[1] == 1.0
    assert scale[2] == 1.0


def test_for_policy_requires_a_model_that_reports_its_shapes():
    class _Model:
        action_horizon = HORIZON
        action_dim = MODEL_DIM

    class _Policy:
        _model = _Model()

        def sample_trajectory(self, *args, **kwargs):  # pragma: no cover - never called
            raise AssertionError

    session = rs.ResampleSession.for_policy(_Policy(), action_dim=ACTION_DIM, n_exec=N_EXEC, num_steps=STEPS)
    assert session.action_horizon == HORIZON
    assert session.model_action_dim == MODEL_DIM

    class _Broken:
        _model = object()

    with pytest.raises(ValueError, match="does not report action_horizon"):
        rs.ResampleSession.for_policy(_Broken(), action_dim=ACTION_DIM, n_exec=N_EXEC, num_steps=STEPS)
