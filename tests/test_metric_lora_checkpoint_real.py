import jax
import pytest

from rulervla.training.metric_workflow import (
    checkpoint_restore_arguments, compact_training_state, make_training_config,
    training_recipe, training_state_template,
)
from rulervla.training.metric_lora import verify_base_reference, validate_delta
from test_metric_lora_contract_real import lora_run


def test_real_complete_model_checkpoint_template_contains_only_trainable_delta():
    config = make_training_config(lora_run())
    state = training_state_template(config, jax.devices('cpu')[0])
    compact = compact_training_state(state, config)
    assert len(jax.tree.leaves(compact.params)) < len(jax.tree.leaves(state.params))
    items, _ = checkpoint_restore_arguments(compact)
    assert len(jax.tree.leaves(items)) < len(jax.tree.leaves(state))
    recipe = training_recipe(config)
    assert recipe['trainable_policy'] == 'lora_plus_six_channel_stem_and_action_time_projections'
    assert recipe['ema_scope'] == 'trainable_parameters_only'
    validate_delta(compact.params, state.params.filter(config.trainable_filter))
    broken = compact.params.to_pure_dict()
    del broken['action_out_proj']
    with pytest.raises(ValueError, match='delta'):
        validate_delta(broken, compact.params)


def test_verified_real_base_rejects_wrong_receipt_before_loading_large_arrays():
    run = lora_run()
    reference = dict(params_path=run['initial_checkpoint'], receipt_path=run['checkpoint_receipt'],
                     receipt_sha256='0' * 64)
    with pytest.raises(ValueError, match='receipt'):
        verify_base_reference(reference)
