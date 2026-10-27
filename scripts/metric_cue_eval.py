import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description='L09 Spatial 配对场景评测')
    parser.add_argument('--scene-plan', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--render-gpu-index', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    protocol_path = ROOT / 'experiments/metric_cue_v1/protocol.json'
    protocol = json.loads(protocol_path.read_text())
    release = json.loads((protocol_path.parent / 'release.json').read_text())
    if release['protocol_sha256'] != file_sha(protocol_path):
        raise ValueError('Evaluation release belongs to another protocol revision')
    from rulervla.training.metric_protocol import require_evaluation_release
    require_evaluation_release(protocol, release)
    from rulervla.training.metric_protocol import validate_data_audit, validate_scene_and_simulator
    validate_data_audit(ROOT, protocol, release)
    validate_scene_and_simulator(ROOT, protocol, release)
    if (release['paired_scene_manifest'] is None or
            args.scene_plan.resolve() != (ROOT / release['paired_scene_manifest']).resolve() or
            file_sha(args.scene_plan) != release['paired_scene_manifest_sha256']):
        raise ValueError('Paired scene manifest differs from the released evaluation plan')
    matrix = json.loads((protocol_path.parent / 'run_matrix.json').read_text())
    if args.run_id not in {entry['run_id'] for entry in matrix['runs']}:
        raise ValueError('Unknown L09 run ID')
    if args.out.name != args.run_id:
        raise ValueError('Evaluation output directory must use the run ID')
    scene_plan = json.loads(args.scene_plan.read_text())
    if (scene_plan['schema'] != 'rulervla_metric_paired_scene_v1'
            or scene_plan['experiment_id'] != protocol['experiment_id']
            or scene_plan['task_order_index'] != 0):
        raise ValueError('Paired scene manifest has an incompatible identity or task order')
    planned = [entry for entry in scene_plan['tasks'] if entry['suite'] == 'libero_spatial']
    if len(planned) != 10 or any(len(entry['scenes']) != 50 for entry in planned):
        raise ValueError('Spatial evaluation needs ten tasks and fifty fixed scenes per task')
    from rulervla.data.simulator_source import configure_libero_source
    source_root = configure_libero_source(
        config_path=ROOT / 'configs/metric_cue/libero_source.json')
    source_commit = subprocess.check_output(
        ['git', '-C', str(source_root), 'rev-parse', 'HEAD'], text=True).strip()
    if (source_root.resolve() != Path(scene_plan['source_root']).resolve()
            or source_commit != scene_plan['source_commit']):
        raise ValueError('LIBERO source differs from the paired scene manifest')
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    from prepare_main_spatial_geometry import capture_frame
    from rulervla.data.state_adapter import state8_from_observation
    import torch
    client = WebsocketClientPolicy(args.host, args.port)
    if client.get_server_metadata()['run_id'] != args.run_id:
        raise ValueError('Connected policy checkpoint belongs to another run')
    suite = benchmark.get_benchmark_dict()['libero_spatial']()
    args.out.mkdir(parents=True, exist_ok=False)
    receipts = []
    for entry in planned:
        task_index = entry['task_index']
        task = suite.get_task(task_index)
        if task.language != entry['instruction']:
            raise ValueError('Task instruction differs from the paired scene plan')
        init_file = (Path(get_libero_path('init_states')) / task.problem_folder /
                     task.init_states_file).resolve()
        if init_file != Path(entry['init_states_file']).resolve():
            raise ValueError('Initial state path differs from the paired scene plan')
        if file_sha(init_file) != entry['init_states_file_sha256']:
            raise ValueError('Initial state file changed')
        states = torch.load(init_file, map_location='cpu', weights_only=False)
        if (not isinstance(states, np.ndarray) or states.shape != (50, 92)
                or states.dtype != np.float64 or not np.isfinite(states).all()):
            raise ValueError('Initial state array differs from the paired scene plan')
        bddl = (Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file).resolve()
        if bddl != Path(entry['bddl_file']).resolve() or file_sha(bddl) != entry['bddl_file_sha256']:
            raise ValueError('BDDL task differs from the paired scene plan')
        if [scene['scene_index'] for scene in entry['scenes']] != list(range(50)):
            raise ValueError('Paired scene indices must cover all fifty initial states')
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256,
                                 camera_widths=256, camera_depths=True,
                                 render_gpu_device_id=args.render_gpu_index)
        for scene in entry['scenes']:
            scene_index = scene['scene_index']
            if (scene['environment_seed'] != scene_plan['environment_seed_base']
                    + task_index * 50 + scene_index):
                raise ValueError('Environment seed differs from the paired scene rule')
            initial_state = np.ascontiguousarray(states[scene_index])
            if hashlib.sha256(initial_state.tobytes()).hexdigest() != scene['initial_state_sha256']:
                raise ValueError('Paired initial state changed')
            episode_index = task_index * 50 + scene_index
            env.seed(scene['environment_seed'])
            env.reset()
            env.set_init_state(initial_state)
            client.infer({'command': 'reset', 'episode_index': episode_index})
            action_plan = deque()
            success = False
            actions_executed = 0
            settling = protocol['evaluation']['settling_control_steps']
            for control_step in range(protocol['evaluation']['maximum_control_steps'] + settling):
                sim_state = np.asarray(env.sim.get_state().flatten(), dtype=np.float64)
                _, sample = capture_frame(env, sim_state, state8_from_observation)
                sample.update(task=task.language, episode_index=episode_index,
                              frame_index=control_step)
                reply = client.infer({'command': 'observe', 'sample': sample})
                if reply['control_step'] != control_step:
                    raise ValueError('Policy acknowledged another control step')
                if control_step < settling:
                    action = [0.0] * 6 + [-1.0]
                else:
                    if not action_plan:
                        predicted = client.infer({'command': 'act'})
                        if (predicted['episode_index'] != episode_index or
                                predicted['control_step'] != control_step):
                            raise ValueError('Policy action refers to another observation')
                        chunk = np.asarray(predicted['actions'])
                        if chunk.shape != (10, 7) or not np.isfinite(chunk).all():
                            raise ValueError('Policy action chunk is invalid')
                        action_plan.extend(chunk[:protocol['evaluation']['replan_control_steps']])
                    action = action_plan.popleft().tolist()
                    actions_executed += 1
                _, _, done, _ = env.step(action)
                if done:
                    success = True
                    break
            receipt = {'run_id': args.run_id, 'task_index': task_index,
                       'scene_index': scene_index, 'episode_index': episode_index,
                       'initial_state_sha256': scene['initial_state_sha256'],
                       'success': success, 'actions_executed': actions_executed,
                       'control_steps': control_step + 1}
            (args.out / ('task_%02d_scene_%02d.json' % (task_index, scene_index))).write_text(
                json.dumps(receipt, indent=2, allow_nan=False) + '\n')
            receipts.append(receipt)
        env.close()
    if len(receipts) != 500:
        raise ValueError('Spatial evaluation did not produce all 500 planned receipts')
    (args.out / 'status.json').write_text(json.dumps({
        'state': 'complete', 'run_id': args.run_id,
        'planned_rollouts': 500, 'completed_rollouts': len(receipts),
        'successes': sum(item['success'] for item in receipts)}, indent=2) + '\n')


if __name__ == '__main__':
    main()
