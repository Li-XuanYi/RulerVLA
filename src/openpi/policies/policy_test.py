import jax
import numpy as np
from openpi_client import action_chunk_broker
import pytest

from openpi.models import pi0_config
from openpi.policies import aloha_policy
from openpi.policies import policy_config as _policy_config
from openpi.policies.policy import AccelFeedbackConfig
from openpi.policies.policy import Policy
from openpi.policies.policy import select_execution_steps
from openpi.training import config as _config
from openpi.uncertainty import AccelMonitor


def test_infer_with_accel_pi05_dummy():
    config = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=4,
        action_horizon=3,
        max_token_len=8,
    )
    policy = Policy(config.create(jax.random.key(0)))
    batched = config.fake_obs().to_dict()
    example = jax.tree.map(lambda value: np.asarray(value[0]), batched)

    monitor = AccelMonitor(mean0=0.0, std0=0.0, threshold=1e9)
    # num_steps=4 leaves 3 prefix layers (p=2..4), so column 1 is the deepest
    # that is in range; the default accel_prefix:3 needs num_steps >= 5.
    result = policy.infer_with_accel(example, num_steps=4, mode="accel_prefix:1", accel_monitor=monitor)

    assert result["actions"].shape == (3, 4)
    assert np.isscalar(result["accel"])
    assert result["accel_monitor"]["alarm"] is False
    assert result["fm_trajectory"]["x_t"].shape == (5, 3, 4)
    assert result["fm_trajectory"]["v_t"].shape == (4, 3, 4)


def test_accel_executed_window_defaults_to_the_model_horizon():
    config = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=4,
        action_horizon=3,
        max_token_len=8,
    )
    policy = Policy(config.create(jax.random.key(0)))
    example = jax.tree.map(lambda value: np.asarray(value[0]), config.fake_obs().to_dict())

    # A reference scale of 12 entries only fits an executed window of 3 * 4, so
    # accepting it is what proves n_exec resolved to the model's horizon rather
    # than staying None (which would be an error) or silently widening.
    result = policy.infer_with_accel(
        example, num_steps=4, mode="accel_prefix:1", fixed_std=np.ones(12, dtype=np.float32)
    )
    assert np.isscalar(result["accel"])

    with pytest.raises(ValueError, match="fixed_std shape"):
        policy.infer_with_accel(
            example, num_steps=4, mode="accel_prefix:1", fixed_std=np.ones(8, dtype=np.float32)
        )
    with pytest.raises(ValueError, match="exceeds the model's action_horizon"):
        policy.infer_with_accel(example, num_steps=4, mode="accel_prefix:1", n_exec=5)


def test_accel_feedback_shortens_only_after_alarm():
    config = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=4,
        action_horizon=3,
        max_token_len=8,
    )
    policy = Policy(config.create(jax.random.key(0)))
    batched = config.fake_obs().to_dict()
    example = jax.tree.map(lambda value: np.asarray(value[0]), batched)

    result = policy.infer_with_accel_feedback(
        example,
        num_steps=4,
        feedback=AccelFeedbackConfig(strategy="shorten", threshold=-1.0, shorten_to=1),
    )

    assert result["actions"].shape == (1, 4)
    assert result["feedback"]["triggered"] is True
    assert result["feedback"]["strategy"] == "shorten"


def test_accel_feedback_replans_until_score_is_below_threshold(monkeypatch):
    policy = object.__new__(Policy)
    policy._sample_kwargs = {"num_steps": 4}  # noqa: SLF001
    scores = iter([10.0, 3.0, 1.0])

    def fake_infer_with_accel(obs, **kwargs):
        del obs, kwargs
        score = next(scores)
        return {"actions": np.full((3, 4), score), "accel": score}

    monkeypatch.setattr(policy, "infer_with_accel", fake_infer_with_accel)
    result = policy.infer_with_accel_feedback(
        {}, feedback=AccelFeedbackConfig(strategy="replan", threshold=2.0, max_replans=3)
    )

    assert result["accel"] == 1.0
    assert result["feedback"]["attempts"] == 2
    assert result["feedback"]["triggered"] is False


def test_accel_feedback_more_denoise_adds_steps(monkeypatch):
    policy = object.__new__(Policy)
    policy._sample_kwargs = {"num_steps": 4}  # noqa: SLF001
    observed_steps = []

    def fake_infer_with_accel(obs, **kwargs):
        del obs
        observed_steps.append(kwargs.get("num_steps", 4))
        return {"actions": np.zeros((3, 4)), "accel": 10.0}

    monkeypatch.setattr(policy, "infer_with_accel", fake_infer_with_accel)
    result = policy.infer_with_accel_feedback(
        {},
        num_steps=4,
        feedback=AccelFeedbackConfig(strategy="more_denoise", threshold=2.0, extra_num_steps=5),
    )

    assert observed_steps == [4, 9]
    assert result["feedback"]["attempts"] == 1


def test_accel_feedback_uses_backup_policy_after_alarm(monkeypatch):
    policy = object.__new__(Policy)
    policy._sample_kwargs = {}  # noqa: SLF001

    def fake_infer_with_accel(obs, **kwargs):
        del obs, kwargs
        return {"actions": np.ones((3, 4)), "accel": 10.0}

    class BackupPolicy:
        def infer(self, obs):
            del obs
            return {"actions": np.zeros((2, 4))}

    monkeypatch.setattr(policy, "infer_with_accel", fake_infer_with_accel)
    result = policy.infer_with_accel_feedback(
        {}, feedback=AccelFeedbackConfig(strategy="backup", threshold=2.0, backup_policy=BackupPolicy())
    )

    np.testing.assert_array_equal(result["actions"], np.zeros((2, 4)))
    assert result["accel"] == 10.0
    assert result["feedback"]["strategy"] == "backup"


def test_adaptive_execution_steps_rule():
    feedback = AccelFeedbackConfig(
        strategy="adaptive", threshold_low=1.0, threshold_high=2.0, n_low=5, n_mid=3, n_high=1
    )
    # Boundaries are inclusive on the lower band, matching ``<=`` in the rule.
    assert select_execution_steps(0.5, feedback) == 5
    assert select_execution_steps(1.0, feedback) == 5
    assert select_execution_steps(1.5, feedback) == 3
    assert select_execution_steps(2.0, feedback) == 3
    assert select_execution_steps(2.5, feedback) == 1

    with pytest.raises(ValueError, match="threshold_low < threshold_high"):
        AccelFeedbackConfig(strategy="adaptive", threshold_low=2.0, threshold_high=1.0)
    with pytest.raises(ValueError, match="threshold_low and threshold_high"):
        AccelFeedbackConfig(strategy="adaptive", threshold_low=1.0)


def test_adaptive_feedback_shortens_actions_per_score(monkeypatch):
    policy = object.__new__(Policy)
    policy._sample_kwargs = {"num_steps": 4}  # noqa: SLF001
    scores = iter([0.5, 1.5, 2.5])

    def fake_infer_with_accel(obs, **kwargs):
        del obs, kwargs
        score = next(scores)
        return {"actions": np.full((6, 4), score), "accel": score}

    monkeypatch.setattr(policy, "infer_with_accel", fake_infer_with_accel)
    feedback = AccelFeedbackConfig(
        strategy="adaptive", threshold_low=1.0, threshold_high=2.0, n_low=5, n_mid=3, n_high=1
    )

    low = policy.infer_with_accel_feedback({}, feedback=feedback)
    mid = policy.infer_with_accel_feedback({}, feedback=feedback)
    high = policy.infer_with_accel_feedback({}, feedback=feedback)

    assert low["actions"].shape == (5, 4)
    assert mid["actions"].shape == (3, 4)
    assert high["actions"].shape == (1, 4)
    assert [low["feedback"]["execute_steps"], mid["feedback"]["execute_steps"], high["feedback"]["execute_steps"]] == [
        5,
        3,
        1,
    ]
    # Keeping the full window is not an alarm; committing fewer actions is.
    assert low["feedback"]["triggered"] is False
    assert mid["feedback"]["triggered"] is True
    assert high["feedback"]["triggered"] is True


def test_adaptive_feedback_clamps_horizon_to_chunk_length(monkeypatch):
    policy = object.__new__(Policy)
    policy._sample_kwargs = {}  # noqa: SLF001

    def fake_infer_with_accel(obs, **kwargs):
        del obs, kwargs
        return {"actions": np.zeros((2, 4)), "accel": 0.0}

    monkeypatch.setattr(policy, "infer_with_accel", fake_infer_with_accel)
    result = policy.infer_with_accel_feedback(
        {}, feedback=AccelFeedbackConfig(strategy="adaptive", threshold_low=1.0, threshold_high=2.0, n_low=5)
    )

    assert result["actions"].shape == (2, 4)
    assert result["feedback"]["execute_steps"] == 2


def test_accel_path_preserves_actions_for_same_noise():
    config = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=4,
        action_horizon=3,
        max_token_len=8,
    )
    obs = config.fake_obs()
    example = jax.tree.map(lambda value: np.asarray(value[0]), obs.to_dict())
    noise = np.asarray(jax.random.normal(jax.random.key(1), (3, 4)))
    regular_policy = Policy(config.create(jax.random.key(0)), rng=jax.random.key(2), sample_kwargs={"num_steps": 4})
    accel_policy = Policy(config.create(jax.random.key(0)), rng=jax.random.key(2), sample_kwargs={"num_steps": 4})

    regular = regular_policy.infer(example, noise=noise)
    # An explicit in-range depth, because the mode must not change the sampled
    # path: num_steps=4 has no column 3 for the default accel_prefix:3.
    diagnostic = accel_policy.infer_with_accel(example, noise=noise, num_steps=4, mode="accel_prefix:1")
    np.testing.assert_allclose(diagnostic["actions"], regular["actions"], rtol=1e-5, atol=1e-5)


@pytest.mark.manual
def test_infer():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    example = aloha_policy.make_aloha_example()
    result = policy.infer(example)

    assert result["actions"].shape == (config.model.action_horizon, 14)


@pytest.mark.manual
def test_broker():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    broker = action_chunk_broker.ActionChunkBroker(
        policy,
        # Only execute the first half of the chunk.
        action_horizon=config.model.action_horizon // 2,
    )

    example = aloha_policy.make_aloha_example()
    for _ in range(config.model.action_horizon):
        outputs = broker.infer(example)
        assert outputs["actions"].shape == (14,)
