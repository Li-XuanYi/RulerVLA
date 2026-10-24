"""Audit one real frame through the new two-view six-channel input path on CPU."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'configs/rulervla_metric_cue.yaml')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'src'))
    import jax
    from openpi.models import model
    from openpi.models.pi0_config import Pi0Config
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi import transforms
    from rulervla.adapters.metric_cue import MetricViewAdapter, SixChannelInputs
    from rulervla.config import load_metric_method_config
    from rulervla.cues.metric_prompt import StrictPromptTokenizer
    from rulervla.cues.motion_echo import MotionHistory

    if any(device.platform != 'cpu' for device in jax.devices()):
        raise RuntimeError('This smoke is CPU only')
    with np.load(args.sample, allow_pickle=False) as data:
        sample = {key: data[key] for key in data.files}
    sample['task'] = str(sample['task'])
    sample.pop('actions', None)  # The extracted frame has one action, not a training chunk.
    geometry, ruler, echo = load_metric_method_config(args.config)
    history = None
    if echo.enabled:
        history = MotionHistory(True, int(sample['episode_index']), int(sample['frame_index']),
                                int(sample['frame_index']) - echo.history_steps,
                                sample['ee_world_m'], sample['past_ee_world_m'])
    adapter = MetricViewAdapter(geometry, ruler, echo)
    prepared = adapter.prepare(sample, history=history)
    policy = Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False, image_channels=6)
    structured = SixChannelInputs(model_type=policy.model_type)(prepared)
    tokenizer = StrictPromptTokenizer(PaligemmaTokenizer(policy.max_token_len))
    structured = transforms.TokenizePrompt(tokenizer, discrete_state_input=False)(structured)
    structured = transforms.PadStatesAndActions(policy.action_dim)(structured)
    batched = {
        'image': {key: value[None] for key, value in structured['image'].items()},
        'image_mask': {key: np.asarray([value], np.bool_) for key, value in structured['image_mask'].items()},
        'state': structured['state'][None],
        'tokenized_prompt': structured['tokenized_prompt'][None],
        'tokenized_prompt_mask': structured['tokenized_prompt_mask'][None],
    }
    observation = model.Observation.from_dict(batched)
    spec, _ = policy.inputs_spec()
    if list(observation.images) != list(spec.images):
        raise ValueError('Image slot order mismatch')
    if any(observation.images[key].shape != spec.images[key].shape for key in spec.images):
        raise ValueError('Image shape mismatch')
    report = {
        'model_source': str(Path(model.__file__).resolve()),
        'episode_index': int(sample['episode_index']), 'frame_index': int(sample['frame_index']),
        'image_slots': {key: {'shape': list(value.shape), 'dtype': str(value.dtype),
                              'rgb_range': [float(value[..., :3].min()), float(value[..., :3].max())],
                              'cue_range': [float(value[..., 3:].min()), float(value[..., 3:].max())],
                              'mask': bool(observation.image_masks[key][0])}
                        for key, value in observation.images.items()},
        'state_shape': list(observation.state.shape),
        'token_count': int(observation.tokenized_prompt_mask.sum()),
        'token_limit': policy.max_token_len,
        'diagnostics': adapter.last_diagnostics,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
