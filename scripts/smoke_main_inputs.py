#!/usr/bin/env python3
"""Input-only parity on 50 cached frames; no model, optimizer or simulator run."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from rulervla.adapters.main_comparison import MAIN_IMAGE_KEYS, MainComparisonInputs
from rulervla.adapters.openpi import OpenPIAdapter
from rulervla.config import load_config, profile


def main():
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / 'experiments/base_frame_manifest.json'
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == '74029f20cd2dd9ff06568bcac89a7a0f0b07708b485a17d462195117fe5f8706'
    manifest = json.loads(manifest_path.read_text())
    config = load_config(root / 'configs/rulervla_main.yaml')
    output_dir = root / 'outputs/main_comparison'
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, frame in enumerate(manifest['frames']):
        cached = root / frame['cache_path']
        assert hashlib.sha256(cached.read_bytes()).hexdigest() == frame['cache_sha256']
        with np.load(cached, allow_pickle=False) as z:
            sample = {k: z[k] for k in ('agentview_rgb', 'wrist_rgb', 'agentview_depth_m', 'intrinsic',
                                       'camera_to_world', 'ee_world_m', 'state8', 'eef_quaternion_xyzw')}
        sample['task'] = manifest['task']
        before = OpenPIAdapter(profile(config, 'V03')).prepare(sample)
        adapter = OpenPIAdapter(config)
        after = adapter.prepare(sample)
        for key in ('observation/wrist_image', 'observation/spatial_image', 'observation/state'):
            np.testing.assert_array_equal(after[key], before[key])
        assert after['prompt'] == before['prompt']
        assert adapter.last_diagnostics['correction'] is None
        assert adapter.last_diagnostics['reliability'] is None
        assert adapter.last_diagnostics['reference_source'] == 'nominal_ee'
        after['observation/wrist_cue'] = np.zeros((224, 224, 3), np.float32)
        # Only test the input hook; this fixture is not the actual OpenPI runtime.
        def base(data):
            return {'image': {'base_0_rgb': data['observation/image'],
                              'left_wrist_0_rgb': data['observation/wrist_image']},
                    'state': data['observation/state']}
        hooked = MainComparisonInputs(base_transform=base)(after)
        assert tuple(hooked['image']) == MAIN_IMAGE_KEYS and all(hooked['image_mask'].values())
        records.append({'frame_key': frame['frame_key'], 'passed': True,
                        'cue_changed_pixels': int(np.any(after['observation/image'] != before['observation/image'], axis=-1).sum())})
        if index == 0:
            Image.fromarray(np.concatenate([before['observation/image'], after['observation/image'],
                                            after['observation/wrist_image']], axis=1)).save(output_dir / 'nominal_ruler_first_frame.png')
    report = {'passed': True, 'frames': len(records), 'training_started': False,
              'policy_inference': False, 'actual_runtime_forward_tested': False,
              'same_state_wrist_and_unscaled_actioncoord_exact': True,
              'anchor_correction_reliability_used': False, 'records': records}
    (output_dir / 'input_smoke.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}))


if __name__ == '__main__':
    main()
