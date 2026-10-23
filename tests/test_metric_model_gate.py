"""Reject the zero-initialized but frozen Cue failure mode before model execution."""
import dataclasses
import importlib.util
from pathlib import Path

import flax.nnx as nnx
import jax.numpy as jnp
import pytest

path = Path(__file__).resolve().parents[1] / 'scripts/metric_cue_model_gate.py'
spec = importlib.util.spec_from_file_location('metric_cue_model_gate', path)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def parameters():
    return nnx.State({'PaliGemma': {'img': {'embedding': {
        'kernel': nnx.VariableState(nnx.Param, jnp.ones((2, 2, 6, 4), jnp.float32))}}}})


def test_real_gate_recipe_trains_visual_stem_and_optimizer_moments():
    config = gate.build_gate_config('unused-checkpoint')
    selected = gate.require_trainable_stem(parameters(), config.trainable_filter)
    assert config.model.pi05 and config.model.image_channels == 6
    assert config.model.paligemma_variant == 'gemma_2b'
    assert config.model.action_expert_variant == 'gemma_300m'
    tx = config.optimizer.create(1e-5)
    assert gate.require_optimizer_stem(tx.init(selected)) == ['mu', 'nu']


def test_frozen_visual_encoder_is_rejected_even_if_other_weights_train():
    config = gate.build_gate_config('unused-checkpoint')
    config = dataclasses.replace(config, freeze_filter=nnx.PathContains('img'))
    with pytest.raises(ValueError, match='not trainable'):
        gate.require_trainable_stem(parameters(), config.trainable_filter)


def test_optimizer_missing_visual_stem_is_rejected():
    config = gate.build_gate_config('unused-checkpoint')
    tx = config.optimizer.create(1e-5)
    with pytest.raises(ValueError, match='optimizer'):
        gate.require_optimizer_stem(tx.init(nnx.State({})))
