"""Shared official OpenPI LoRA recipe, also usable by archived full-FT runs."""
import hashlib
import jax
import numpy as np
from openpi.models.pi0_config import Pi0Config


def policy_config(lora=False):
    return Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False,
        paligemma_variant='gemma_2b_lora' if lora else 'gemma_2b',
        action_expert_variant='gemma_300m_lora' if lora else 'gemma_300m')


def frozen_hashes(params, freeze_filter):
    return {jax.tree_util.keystr(path): hashlib.sha256(
        np.asarray(jax.device_get(value)).reshape(-1).view(np.uint8).tobytes()).hexdigest()
        for path, value in jax.tree_util.tree_flatten_with_path(params.filter(freeze_filter))[0]}


def parameter_audit(state, config, lora):
    frozen=state.params.filter(config.freeze_filter)
    trainable=state.params.filter(config.trainable_filter)
    lora_leaves=[value for path,value in jax.tree_util.tree_flatten_with_path(trainable)[0]
                 if 'lora' in jax.tree_util.keystr(path).lower()]
    assert bool(jax.tree.leaves(frozen)) == lora
    assert bool(lora_leaves) == lora
    return dict(trainable_parameter_count=int(sum(np.prod(x.shape) for x in jax.tree.leaves(trainable))),
                frozen_parameter_count=int(sum(np.prod(x.shape) for x in jax.tree.leaves(frozen))),
                lora_parameter_count=int(sum(np.prod(x.shape) for x in lora_leaves)),
                freeze_policy='official Pi0Config.get_freeze_filter; non-LLM modules remain trainable')
