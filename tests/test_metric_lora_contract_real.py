import json
import math
from pathlib import Path

import flax.nnx as nnx
import jax
import numpy as np
import pytest

from rulervla.adapters.metric_pipeline import metric_model_config
from rulervla.training import metric_workflow
from rulervla.training.metric_resources import load_run_config

ROOT = Path(__file__).resolve().parents[1]


def lora_run(arm="M2"):
    run = load_run_config(ROOT / "configs/metric_workflow_node145.yaml")
    protocol = json.loads((ROOT / "experiments/metric_cue_v1/protocol.json").read_text())
    run["training"] = dict(protocol["training"], seed=42)
    run["cue_config"] = str(ROOT / protocol["arms"][arm])
    return run


def test_real_lora_config_freezes_backbones_instead_of_training_the_whole_vision_encoder():
    counts = []
    manifests = []
    for arm in ("M0", "M1", "M2", "M3"):
        config = metric_workflow.make_training_config(lora_run(arm))
        assert config.model.paligemma_variant == "gemma_2b_lora"
        assert config.model.action_expert_variant == "gemma_300m_lora"
        params = nnx.state(nnx.eval_shape(config.model.create, jax.random.key(42)), nnx.Param)
        selected = metric_workflow.flat_params(params.filter(config.trainable_filter))
        counts.append(sum(math.prod(value.shape) for value in selected.values()))
        manifests.append({name: (value.shape, str(value.dtype)) for name, value in selected.items()})
        assert "PaliGemma/img/embedding/kernel" in selected
        assert "action_out_proj/kernel" in selected
        assert not any("encoderblock" in name for name in selected)
        assert all("lora" in name for name in selected if name.startswith("PaliGemma/llm/"))
    assert counts == [53509280] * 4
    assert all(manifest == manifests[0] for manifest in manifests)


def test_complete_model_lora_initialization_is_deterministic_with_zero_b_and_nonzero_a():
    from rulervla.training.metric_lora import initialize_lora_parameters

    config = metric_model_config(finetuning="lora")
    reference = nnx.state(nnx.eval_shape(config.create, jax.random.key(42)), nnx.Param).to_pure_dict()
    first = metric_workflow.flat_params(initialize_lora_parameters(reference, 42))
    second = metric_workflow.flat_params(initialize_lora_parameters(reference, 42))
    names = [name for name in first if "lora" in name]
    assert names
    for name in names:
        a, b = np.asarray(first[name]), np.asarray(second[name])
        np.testing.assert_array_equal(a, b)
        if name.endswith("lora_b"):
            assert np.count_nonzero(a) == 0
        else:
            assert name.endswith("lora_a") and np.count_nonzero(a) > 0
    assert isinstance(first["PaliGemma/img/embedding/kernel"], jax.ShapeDtypeStruct)


def test_lora_optimizer_and_ema_shape_trees_exclude_every_frozen_backbone_leaf():
    config = metric_workflow.make_training_config(lora_run())
    template = metric_workflow.training_state_template(config, jax.devices("cpu")[0])
    expected = set(metric_workflow.flat_params(template.params.filter(config.trainable_filter)))
    assert set(metric_workflow.flat_params(template.ema_params)) == expected
    assert sum(math.prod(x.shape) for x in jax.tree.leaves(template.ema_params)) == 53509280
    for part in jax.tree.leaves(template.opt_state, is_leaf=lambda v: hasattr(v, "mu") and hasattr(v, "nu")):
        if hasattr(part, "mu") and hasattr(part, "nu"):
            assert set(metric_workflow.flat_params(part.mu)) == expected
            assert set(metric_workflow.flat_params(part.nu)) == expected


def test_unknown_metric_model_finetuning_mode_fails():
    with pytest.raises(ValueError, match="finetuning"):
        metric_model_config(finetuning="not_a_recipe")
