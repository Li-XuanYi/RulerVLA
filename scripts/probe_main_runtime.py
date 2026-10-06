"""Actual OpenPI CPU preprocessing/spec gate. No weights, optimizer or rollout."""
import argparse
import dataclasses
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.runtime / 'src'))
    import jax
    import jax.numpy as jnp
    import numpy as np
    from openpi.models import model
    from openpi.models.pi0_config import Pi0Config

    config = Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False)
    spec, actions = config.inputs_spec(batch_size=1)
    keys = ('base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb', 'aux_wrist_cue_rgb')
    assert tuple(spec.images) == keys and tuple(spec.image_masks) == keys
    images = {key: jnp.full((1, 224, 224, 3), value, dtype=jnp.float32)
              for key, value in zip(keys, (-.75, .25, 0, .875))}
    observation = dataclasses.replace(spec, images=images,
        image_masks={key: jnp.ones((1,), bool) for key in keys}, state=jnp.zeros((1, 32)),
        tokenized_prompt=jnp.zeros((1, config.max_token_len), jnp.int32),
        tokenized_prompt_mask=jnp.ones((1, config.max_token_len), bool))
    for train in (False, True):
        result = model.preprocess_observation(jax.random.key(42), observation, train=train)
        assert tuple(result.images) == keys
        for key in keys:
            np.testing.assert_array_equal(result.images[key], images[key])
    report = {'passed': True, 'actual_openpi_preprocessing': True,
        'actual_inputs_spec': True, 'four_slots_retained': True,
        'train_and_inference_values_unchanged': True, 'image_keys': list(keys),
        'action_shape': list(actions.shape), 'full_weight_loader_tested': False,
        'model_forward_backward_tested': False, 'optimizer_updates': 0,
        'formal_training_started': False, 'runtime': str(args.runtime.resolve())}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
