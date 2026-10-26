import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import mujoco
import numpy as np
import robosuite
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description='真实 LIBERO Spatial 初始场景与双视角 RGB-D 核验')
    parser.add_argument('--task-index', type=int, default=2)
    parser.add_argument('--scene-index', type=int, default=0)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if os.environ.get('MUJOCO_GL') != 'osmesa' or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('This simulator gate requires CPU OSMesa rendering')

    release = json.loads((ROOT / 'experiments/metric_cue_v1/release.json').read_text())
    plan_path = ROOT / release['paired_scene_manifest']
    if (not release['gate_status']['paired_scene_manifest_lock']
            or file_sha256(plan_path) != release['paired_scene_manifest_sha256']):
        raise ValueError('Paired scene manifest is not locked')
    plan = json.loads(plan_path.read_text())
    task_plan = plan['tasks'][args.task_index]
    scene = task_plan['scenes'][args.scene_index]

    from rulervla.data.simulator_source import configure_libero_source
    source_root = configure_libero_source(config_path=ROOT / 'configs/metric_cue/libero_source.json')
    if source_root.resolve() != Path(plan['source_root']).resolve():
        raise ValueError('LIBERO source differs from paired scenes')
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from prepare_main_spatial_geometry import capture_frame
    from rulervla.data.state_adapter import state8_from_observation

    suite = benchmark.get_benchmark_dict()['libero_spatial']()
    task = suite.get_task(args.task_index)
    if task.language != task_plan['instruction']:
        raise ValueError('Task instruction differs from paired scenes')
    init_file = (Path(get_libero_path('init_states')) / task.problem_folder / task.init_states_file).resolve()
    bddl_file = (Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file).resolve()
    if (file_sha256(init_file) != task_plan['init_states_file_sha256']
            or file_sha256(bddl_file) != task_plan['bddl_file_sha256']):
        raise ValueError('LIBERO scene files differ from paired scenes')
    states = torch.load(init_file, map_location='cpu', weights_only=False)
    if not isinstance(states, np.ndarray) or states.shape != (50, 92):
        raise ValueError('Expected fifty official LIBERO initial states')
    initial_state = np.ascontiguousarray(states[args.scene_index])
    if hashlib.sha256(initial_state.tobytes()).hexdigest() != scene['initial_state_sha256']:
        raise ValueError('Initial state differs from paired scenes')

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_file), camera_heights=256, camera_widths=256,
        camera_depths=True, render_gpu_device_id=-1)
    env.seed(scene['environment_seed'])
    env.reset()
    env.set_init_state(initial_state)
    sim_state = np.asarray(env.sim.get_state().flatten(), dtype=np.float64)
    _, sample = capture_frame(env, sim_state, state8_from_observation)
    for key in ('agentview_rgb', 'wrist_rgb'):
        if sample[key].shape != (256, 256, 3):
            raise ValueError(f'Unexpected camera image shape: {key}')
    for key in ('agentview_depth_m', 'wrist_depth_m'):
        if sample[key].shape != (256, 256) or not np.isfinite(sample[key]).all():
            raise ValueError(f'Unexpected metric depth: {key}')
    if sample['state8'].shape != (8,):
        raise ValueError('Unexpected proprioception shape')
    _, _, done, _ = env.step([0.0] * 6 + [-1.0])
    env.close()

    result = {
        'schema': 'rulervla_metric_simulator_gate_v1',
        'state': 'complete',
        'task_index': args.task_index,
        'scene_index': args.scene_index,
        'source_commit': plan['source_commit'],
        'paired_scene_manifest_sha256': file_sha256(plan_path),
        'mujoco_version': mujoco.__version__,
        'robosuite_version': robosuite.__version__,
        'initial_state_sha256': scene['initial_state_sha256'],
        'control_step_done': bool(done),
        'observations': {
            key: {
                'shape': list(np.asarray(sample[key]).shape),
                'dtype': str(np.asarray(sample[key]).dtype),
                'sha256': hashlib.sha256(np.ascontiguousarray(sample[key]).tobytes()).hexdigest(),
            }
            for key in ('state8', 'agentview_rgb', 'wrist_rgb',
                        'agentview_depth_m', 'wrist_depth_m',
                        'intrinsic', 'camera_to_world',
                        'wrist_intrinsic', 'wrist_camera_to_world')
        },
    }
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'receipt.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'state': 'complete', 'receipt': str(args.out / 'receipt.json')}))


if __name__ == '__main__':
    main()
