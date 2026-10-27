from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from rulervla.config import load_metric_method_config
from rulervla.training.metric_protocol import load_formal_metric_run, require_arm_ruler
from rulervla.training.metric_resources import load_run_config


ROOT = Path(__file__).resolve().parents[1]
FAMILY = ROOT / "experiments/metric_cue_v1"


@pytest.mark.parametrize("arm", ["M0", "M1", "M2", "M3"])
def test_formal_arms_load_current_positive_ruler(arm):
    protocol = json.loads((FAMILY / "protocol.json").read_text())
    _, current, _ = load_metric_method_config(ROOT / "configs/metric_ruler.yaml")
    _, actual, echo = load_metric_method_config(ROOT / protocol["arms"][arm])
    mode = {"M0": "blank", "M1": "direction", "M2": "metric", "M3": "metric"}[arm]
    assert actual == replace(current, enabled=arm != "M0", mode=mode)
    assert echo.enabled == (arm == "M3")
    run = load_run_config(ROOT / "configs/metric_workflow.yaml")
    require_arm_ruler(ROOT, arm, run, protocol)


def test_legacy_workflow_selects_current_metric_input():
    from rulervla.adapters.metric_pipeline import MetricPipeline

    current = load_run_config(ROOT / "configs/metric_workflow.yaml")
    legacy = load_run_config(ROOT / "configs/rulervla_metric_workflow.yaml")
    expected = MetricPipeline.from_files(current["cue_config"], current["norm_stats_dir"])
    actual = MetricPipeline.from_files(legacy["cue_config"], legacy["norm_stats_dir"])
    assert actual.fingerprint() == expected.fingerprint()


def test_protocol_and_release_match_saved_sources():
    protocol_bytes = (FAMILY / "protocol.json").read_bytes()
    protocol = json.loads(protocol_bytes)
    release = json.loads((FAMILY / "release.json").read_text())
    plan = ROOT / "docs/metric_cue/RESEARCH_PLAN.md"
    assert protocol["research_plan_sha256"] == hashlib.sha256(plan.read_bytes()).hexdigest()
    assert release["protocol_sha256"] == hashlib.sha256(protocol_bytes).hexdigest()
    assert not release["training_released"] and not release["evaluation_released"]


@pytest.mark.parametrize("arm", ["M0", "M1", "M2", "M3"])
def test_formal_training_stops_at_closed_gates_after_config_loading(arm):
    run = load_run_config(ROOT / "configs/metric_workflow.yaml")
    with pytest.raises(PermissionError, match="gates remain closed"):
        load_formal_metric_run(ROOT, arm, 42, run)


def test_batch8_real_inputs_and_full_model_gradient_shapes():
    import flax.nnx as nnx
    import jax

    from rulervla.adapters.metric_pipeline import MetricPipeline
    from rulervla.training.main_data import SpatialDataset, update_indices
    from rulervla.training.metric_workflow import MetricTrainer, make_training_config, prepare_batch

    run = load_run_config(ROOT / "configs/metric_workflow.yaml")
    cache = ROOT / "outputs/metric_cue_pilot/g2_engineering/segmented_data_cache"
    dataset = SpatialDataset(
        cache / Path(run["data"]["rgbd_root"]).relative_to("/"),
        cache / Path(run["data"]["geometry_root"]).relative_to("/"),
        path_mappings={"/home/Dataset2": str(cache / "home/Dataset2")},
    )
    protocol = json.loads((FAMILY / "protocol.json").read_text())
    training = dict(protocol["training"], seed=42, batch_size=8, gradient_accumulation=8)
    candidate = dict(run, training=training, resources=dict(run["resources"], gpu_indices=[2, 5]))
    config = make_training_config(candidate)
    assert config.batch_size == 8 and config.num_train_steps == 30000
    assert training["micro_batch_size"] * training["gradient_accumulation"] == 8
    assert len(dataset) == 52970
    order = update_indices(len(dataset), 42, 29999, batch_size=8)
    np.testing.assert_array_equal(order, update_indices(len(dataset), 42, 29999, batch_size=8))
    assert order.shape == (8,) and np.all((0 <= order) & (order < len(dataset)))
    pipeline = MetricPipeline.from_files(run["cue_config"], run["norm_stats_dir"])
    batch = prepare_batch(dataset, list(range(20, 28)), pipeline)
    assert batch.actions.shape == (8, 10, 32)
    assert len(batch.identities) == 8
    graph, params = nnx.split(nnx.eval_shape(config.model.create, jax.random.key(42)))
    trainer = MetricTrainer(config, graph)
    observations = jax.tree.map(lambda value: value.reshape((8, 1) + value.shape[1:]), batch.observation)
    actions = batch.actions.reshape((8, 1, 10, 32))
    gradients, metrics = jax.eval_shape(
        trainer._accumulated_backward, params, observations, actions, np.int32(0),
    )
    expected = params.filter(config.trainable_filter)
    assert [leaf.shape for leaf in jax.tree.leaves(gradients)] == [
        leaf.shape for leaf in jax.tree.leaves(expected)
    ]
    assert metrics["cue_grad_l2"].shape == () and metrics["gradients_finite"].shape == ()
