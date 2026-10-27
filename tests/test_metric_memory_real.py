import json
from pathlib import Path

import flax.nnx as nnx
import jax
import pytest

from rulervla.training import metric_workflow
from rulervla.training.metric_resources import load_run_config


ROOT = Path(__file__).resolve().parents[1]


def test_full_model_restore_template_has_no_array_buffers():
    assert hasattr(metric_workflow, "abstract_training_state")
    run = load_run_config(ROOT / "configs/metric_workflow_pilot.yaml")
    protocol = json.loads((ROOT / "experiments/metric_cue_v1/protocol.json").read_text())
    run["training"] = dict(protocol["training"], seed=42)
    config = metric_workflow.make_training_config(run)
    cpu = jax.devices("cpu")[0]
    assert hasattr(metric_workflow, "training_state_template")
    template = metric_workflow.training_state_template(config, cpu)
    shaped_params = nnx.state(nnx.eval_shape(config.model.create, jax.random.key(42)))
    for actual, expected in zip(jax.tree.leaves(template.params),
                                jax.tree.leaves(shaped_params), strict=True):
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
    for actual in jax.tree.leaves(template):
        assert isinstance(actual, jax.ShapeDtypeStruct)
        assert actual.sharding.device_set == {cpu}
    assert template.ema_decay == .999
    assert metric_workflow.flat_params(template.params)[metric_workflow.STEM].shape == (14, 14, 6, 1152)
    metric_workflow.require_trainable_cue(template.params, config.trainable_filter)
    assert metric_workflow.require_cue_optimizer(template.opt_state) == ["mu", "nu"]


def test_runtime_capacity_counts_only_this_process_gpu_memory():
    from rulervla.training import metric_resources

    assert hasattr(metric_resources, "runtime_capacity")
    run = load_run_config(ROOT / "configs/metric_workflow_pilot.yaml")
    measured = metric_resources.runtime_capacity(run)
    assert measured["pid"] > 0
    assert measured["cpu_rss_gib"] > 0
    for item in measured["gpu_snapshot"]:
        if item["index"] in run["resources"]["gpu_indices"]:
            assert item["own_mib"] == 0
            assert item["budget_available_mib"] == item["free_mib"]
    assert measured["cpu_budget_available_gib"] > measured["cpu_available_gib"]


def test_shape_template_cannot_be_hashed_as_restored_values():
    assert hasattr(metric_workflow, "abstract_training_state")
    run = load_run_config(ROOT / "configs/metric_workflow_pilot.yaml")
    config = metric_workflow.make_training_config(run)
    state = nnx.state(nnx.eval_shape(config.model.create, jax.random.key(42)))
    with pytest.raises((TypeError, ValueError)):
        metric_workflow.state_signatures(state)
