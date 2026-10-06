#!/usr/bin/env python3
"""Freeze the design and 48 planned runs; never launches a training process."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import yaml

from rulervla.config import load_config, profile


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    folder = root / 'experiments/main_comparison_v1'
    if folder.exists():
        raise FileExistsError('Design is already frozen; use a reviewed new protocol version')
    audit = json.loads(args.audit.read_text(encoding='utf-8'))
    assert audit['metadata_total'] == {'episodes': 1693, 'frames': 273465}
    assert audit['libero_plus']['levels']['2'] == 2202
    assert all(s['task_count'] == 10 for s in audit['suites'].values())
    assert sum(s['episodes'] for s in audit['suites'].values()) == 1693
    methods = ('Baseline', 'AimBot', 'AxisGuide-adapted', 'RulerVLA-main')
    train = {
        'backbone': 'pi0.5', 'paligemma_variant': 'gemma_2b',
        'action_expert_variant': 'gemma_300m', 'finetuning': 'full',
        'lora': False, 'freeze_backbone': False, 'dtype': 'bfloat16',
        'initial_checkpoint': audit['initial_checkpoint']['params'],
        'optimizer_updates': 30000, 'effective_batch': 32,
        'micro_batch': None, 'gradient_accumulation': None, 'data_parallel_replicas': None,
        'memory_configuration_status': 'pending_common_worst_case_smoke',
        'optimizer': {'name': 'AdamW', 'beta1': .9, 'beta2': .95, 'eps': 1e-8,
                      'weight_decay': 1e-10, 'clip_gradient_norm': 1.0},
        'scheduler': {'name': 'warmup_cosine', 'warmup_updates': 1000,
                      'peak_lr': 5e-5, 'end_lr': 5e-6, 'decay_updates': 30000},
        'ema_decay': .999, 'action_horizon': 10, 'action_dim': 32,
        'discrete_state_input': False, 'supervised_libero_components': 7,
        'random_image_augmentation': False, 'extra_delta_transform': False,
        'checkpoint_updates': [10000, 20000, 30000],
        'formal_checkpoint_index': 29999, 'training_seeds': [42, 43, 44],
    }
    design = {
        'schema': 'rulervla_main_comparison_v1',
        'status': 'design_fixed_execution_not_ready',
        'training_started': False, 'run_count': 48, 'methods': list(methods),
        'suites': list(audit['suites']), 'shared_training': train,
        'image_keys': ['base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb', 'aux_wrist_cue_rgb'],
        'all_image_masks_active': True, 'neutral_auxiliary': 0.0,
        'anchor': False, 'correction': False, 'reliability': False,
        'ruler_lengths_m': [.05, .10, .15], 'actioncoord_encoding_radius_m': 3.25,
        'standard_eval': {'tasks_per_suite': 10, 'trials_per_task': 50,
                          'planned_rollouts': 24000, 'initial_state_indices': list(range(50)),
                          'scene_seed_base': 20261006, 'same_scene_seeds_all_methods_and_training_seeds': True},
        'libero_plus_eval': {'difficulty_level': 2, 'conditions_per_training_seed': 2202,
                            'trials_per_condition': 1, 'planned_rollouts': 26424,
                            'same_condition_ids_all_methods': True, 'avg': 'equal_category_macro'},
        'required_pending_gates': ['full_suite_same_state_rgbd_and_state8', 'both_view_calibration',
            'official_renderer_parity', 'four_slot_runtime_retention_and_no_metric_augmentation',
            'full_weight_loader_and_trainable_filter_evidence', 'gradient_accumulation_equivalence',
            'common_memory_smoke', 'checkpoint_save_load_resume', 'eval_scene_and_runtime_lock'],
        'source_sha256': audit.get('source_sha256', {}),
    }
    folder.mkdir(parents=True)
    write_json(folder / 'protocol.json', design)
    write_json(folder / 'preflight_readonly.json', audit)
    main_method = yaml.safe_load((root / 'configs/v05_default.yaml').read_text())
    main_method['method'] = profile(load_config(root / 'configs/v05_default.yaml'), 'RulerVLA-main')['method']
    (root / 'configs/rulervla_main.yaml').write_text(yaml.safe_dump(main_method, sort_keys=False), encoding='utf-8', newline='\n')
    rows = []
    for suite, data in audit['suites'].items():
        manifest = {'suite': suite, 'episodes': data['episodes'], 'frames': data['frames'],
                    'episode_ids': data['episode_ids'], 'task_episode_counts': data['task_episode_counts'],
                    'split': 'all listed official train episodes; evaluation conditions excluded'}
        write_json(folder / 'suite_manifests' / (suite + '.json'), manifest)
        for seed in train['training_seeds']:
            for method in methods:
                run_id = '%s_%s_seed%d' % (suite, method.lower().replace('-', '_'), seed)
                checkpoint = '/home/Dataset2/yqt/rulervla/main_comparison_v1/checkpoints/' + run_id
                run = {'run_id': run_id, 'suite': suite, 'method': method, 'seed': seed,
                       'shared_training': train, 'suite_manifest': 'suite_manifests/' + suite + '.json',
                       'checkpoint_root': checkpoint, 'execution_ready': False, 'status': 'planned'}
                write_json(folder / 'runs' / (run_id + '.json'), run)
                rows.append({k: run[k] for k in ('run_id', 'suite', 'method', 'seed', 'checkpoint_root', 'status')})
    assert len(rows) == 48
    with (folder / 'run_matrix.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    files = [p for p in folder.rglob('*') if p.is_file()]
    files += [root / 'docs/MAIN_COMPARISON_PROTOCOL_V1.md', root / 'configs/rulervla_main.yaml']
    entries = [(p.relative_to(root).as_posix(), hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(files)]
    (folder / 'DESIGN_MANIFEST.sha256').write_text(''.join('%s  %s\n' % (digest, path) for path, digest in entries), encoding='utf-8', newline='\n')
    print(json.dumps({'planned_runs': len(rows), 'training_started': False,
                      'design': str(folder), 'execution_ready': False}))


if __name__ == '__main__':
    main()
