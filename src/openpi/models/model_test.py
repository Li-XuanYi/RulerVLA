from flax import nnx
import jax
import numpy as np
import pytest

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import pi0_fast
from openpi.shared import download
from openpi.shared import nnx_utils
from openpi.uncertainty import accel


def test_pi05_accel_trajectory_smoke():
    """The π₀.₅ diagnostic sampler preserves shapes and produces an Accel score."""
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=4,
        action_horizon=3,
        max_token_len=8,
    )
    model = config.create(key)
    obs = config.fake_obs(batch_size=1)
    noise = jax.random.normal(jax.random.key(1), (1, config.action_horizon, config.action_dim))
    regular_sampler = nnx_utils.module_jit(model.sample_actions)
    regular_actions = regular_sampler(key, obs, num_steps=4, noise=noise)
    sampler = nnx_utils.module_jit(model.sample_actions_with_trajectory, static_argnames=("num_steps",))
    trajectory = sampler(key, obs, num_steps=4, noise=noise)

    assert trajectory["actions"].shape == (1, 3, 4)
    assert trajectory["x_t"].shape == (1, 5, 3, 4)
    assert trajectory["v_t"].shape == (1, 4, 3, 4)
    assert trajectory["time"].shape == (4,)
    np.testing.assert_allclose(trajectory["actions"], regular_actions, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(trajectory["x_t"][:, -1], trajectory["actions"], rtol=1e-5, atol=1e-5)
    # T=4 leaves 3 prefix layers (p=2..4), so depth index 1 is the deepest that
    # fits; the default accel_prefix:3 needs num_steps >= 5.
    score = accel.score_chunks(np.asarray(trajectory["x_t"]), action_dim=4, n_exec=3, mode="accel_prefix:1")
    assert score.shape == (1,)
    assert np.isfinite(score).all()


def test_pi0_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi0_lora_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi0_fast_model():
    key = jax.random.key(0)
    config = pi0_fast.Pi0FASTConfig()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size,)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs)
    assert actions.shape == (batch_size, 256)


def test_pi0_fast_lora_model():
    key = jax.random.key(0)
    config = pi0_fast.Pi0FASTConfig(paligemma_variant="gemma_2b_lora")
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size,)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs)
    assert actions.shape == (batch_size, 256)

    lora_filter = nnx_utils.PathRegex(".*lora.*")
    model_state = nnx.state(model)

    lora_state_elems = list(model_state.filter(lora_filter))
    assert len(lora_state_elems) > 0


@pytest.mark.manual
def test_model_restore():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config()

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    model = config.load(
        _model.restore_params(download.maybe_download("gs://openpi-assets/checkpoints/pi0_base/params"))
    )

    loss = model.compute_loss(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = model.sample_actions(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)
