import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def build_plan():
    from rulervla.data.simulator_source import configure_libero_source
    source_root = configure_libero_source(
        config_path=ROOT / 'configs/metric_cue/libero_source.json')
    from libero.libero import benchmark, get_libero_path

    protocol = json.loads((ROOT / 'experiments/metric_cue_v1/protocol.json').read_text())
    evaluation = protocol['evaluation']
    suite = benchmark.get_benchmark_dict()[evaluation['suite']]()
    if suite.get_num_tasks() != evaluation['tasks']:
        raise ValueError('LIBERO Spatial task count differs from the frozen protocol')
    commit = subprocess.check_output(
        ['git', '-C', str(source_root), 'rev-parse', 'HEAD'], text=True).strip()
    seed_base = 42
    tasks = []
    for task_index in range(suite.get_num_tasks()):
        task = suite.get_task(task_index)
        init_file = Path(get_libero_path('init_states')) / task.problem_folder / task.init_states_file
        bddl_file = Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
        states = torch.load(init_file, map_location='cpu', weights_only=False)
        if (not isinstance(states, np.ndarray) or states.shape != (evaluation['initial_states_per_task'], 92)
                or states.dtype != np.float64 or not np.isfinite(states).all()):
            raise ValueError(f'LIBERO initial states differ for task {task_index}')
        scenes = []
        for scene_index in range(len(states)):
            state = np.ascontiguousarray(states[scene_index])
            scenes.append({
                'scene_index': scene_index,
                'environment_seed': seed_base + task_index * len(states) + scene_index,
                'initial_state_sha256': hashlib.sha256(state.tobytes()).hexdigest(),
            })
        tasks.append({
            'suite': evaluation['suite'],
            'task_index': task_index,
            'instruction': task.language,
            'bddl_file': str(bddl_file.resolve()),
            'bddl_file_sha256': file_sha256(bddl_file),
            'init_states_file': str(init_file.resolve()),
            'init_states_file_sha256': file_sha256(init_file),
            'initial_state_shape': list(states.shape[1:]),
            'initial_state_dtype': str(states.dtype),
            'scenes': scenes,
        })
    return {
        'schema': 'rulervla_metric_paired_scene_v1',
        'experiment_id': protocol['experiment_id'],
        'source_root': str(source_root.resolve()),
        'source_commit': commit,
        'task_order_index': 0,
        'environment_seed_base': seed_base,
        'environment_seed_rule': 'base + task_index * initial_states_per_task + scene_index',
        'tasks': tasks,
    }


def main():
    parser = argparse.ArgumentParser(description='冻结并核验 L09 Spatial 配对初始状态')
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    plan = build_plan()
    if args.verify:
        if json.loads(args.out.read_text()) != plan:
            raise ValueError('Paired scene manifest differs from the pinned LIBERO source')
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open('x') as stream:
            json.dump(plan, stream, indent=2, ensure_ascii=False, sort_keys=True)
            stream.write('\n')
    print(json.dumps({
        'state': 'complete', 'manifest': str(args.out.resolve()),
        'sha256': file_sha256(args.out), 'tasks': len(plan['tasks']),
        'scenes': sum(len(task['scenes']) for task in plan['tasks']),
        'source_commit': plan['source_commit'],
    }))


if __name__ == '__main__':
    main()
