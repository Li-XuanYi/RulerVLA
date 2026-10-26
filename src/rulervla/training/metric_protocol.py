import hashlib
import json
from pathlib import Path


def validate_data_audit(root, protocol, release):
    status_path = Path(root) / release['data_audit_status']
    status = json.loads(status_path.read_text())
    if (status['state'] != 'complete' or status['episodes'] != protocol['data']['episodes']
            or status['frames'] != protocol['data']['frames']
            or status['manifest_sha256']['rgbd'] != protocol['data']['rgbd_manifest_sha256']
            or status['manifest_sha256']['geometry'] != protocol['data']['geometry_manifest_sha256']):
        raise ValueError('Complete Spatial data audit differs from the protocol')
    digest = hashlib.sha256((status_path.parent / 'episode_receipts.jsonl').read_bytes()).hexdigest()
    if digest != release['data_audit_receipts_sha256']:
        raise ValueError('Spatial data audit receipts changed')


def validate_scene_and_simulator(root, protocol, release):
    root = Path(root)
    plan_path = root / release['paired_scene_manifest']
    plan_bytes = plan_path.read_bytes()
    if hashlib.sha256(plan_bytes).hexdigest() != release['paired_scene_manifest_sha256']:
        raise ValueError('Paired Spatial scene manifest changed')
    plan = json.loads(plan_bytes)
    evaluation = protocol['evaluation']
    tasks = plan['tasks']
    if (plan['schema'] != 'rulervla_metric_paired_scene_v1'
            or plan['experiment_id'] != protocol['experiment_id']
            or plan['task_order_index'] != 0
            or len(tasks) != evaluation['tasks']
            or [task['task_index'] for task in tasks] != list(range(evaluation['tasks']))
            or any(task['suite'] != evaluation['suite']
                   or len(task['scenes']) != evaluation['initial_states_per_task']
                   for task in tasks)):
        raise ValueError('Paired Spatial scene manifest differs from the protocol')
    for task in tasks:
        if [scene['scene_index'] for scene in task['scenes']] != list(range(evaluation['initial_states_per_task'])):
            raise ValueError('Paired Spatial scene indices are incomplete')

    receipt_path = root / release['simulator_runtime_receipt']
    receipt_bytes = receipt_path.read_bytes()
    if hashlib.sha256(receipt_bytes).hexdigest() != release['simulator_runtime_receipt_sha256']:
        raise ValueError('Simulator runtime receipt changed')
    receipt = json.loads(receipt_bytes)
    task = tasks[receipt['task_index']]
    scene = task['scenes'][receipt['scene_index']]
    if (receipt['schema'] != 'rulervla_metric_simulator_gate_v1'
            or receipt['state'] != 'complete'
            or receipt['source_commit'] != plan['source_commit']
            or receipt['paired_scene_manifest_sha256'] != release['paired_scene_manifest_sha256']
            or receipt['initial_state_sha256'] != scene['initial_state_sha256']
            or receipt['control_step_done']):
        raise ValueError('Simulator runtime receipt differs from the locked scene')
    required_shapes = {'state8': [8], 'agentview_rgb': [256, 256, 3],
                       'wrist_rgb': [256, 256, 3], 'agentview_depth_m': [256, 256],
                       'wrist_depth_m': [256, 256], 'intrinsic': [3, 3],
                       'camera_to_world': [4, 4], 'wrist_intrinsic': [3, 3],
                       'wrist_camera_to_world': [4, 4]}
    if any(receipt['observations'][name]['shape'] != shape
           for name, shape in required_shapes.items()):
        raise ValueError('Simulator runtime observation interface differs')


def load_formal_metric_run(root, arm, seed, run):
    root = Path(root).resolve()
    protocol_path = root / 'experiments/metric_cue_v1/protocol.json'
    release_path = root / 'experiments/metric_cue_v1/release.json'
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    release = json.loads(release_path.read_text())
    if protocol['schema'] != 'rulervla_metric_cue_v1' or release['schema'] != 'rulervla_metric_cue_release_v1':
        raise ValueError('Metric Cue protocol or release schema differs')
    if release['protocol_sha256'] != hashlib.sha256(protocol_bytes).hexdigest():
        raise ValueError('Metric Cue release belongs to another protocol revision')
    if arm not in protocol['arms'] or seed not in protocol['training']['seeds']:
        raise ValueError('Unknown formal arm or training seed')
    missing = [gate for gate in protocol['launch_gates'] if not release['gate_status'].get(gate)]
    if not release['training_released'] or missing:
        raise PermissionError('Formal Metric Cue training gates remain closed: ' + ', '.join(missing))
    validate_data_audit(root, protocol, release)
    validate_scene_and_simulator(root, protocol, release)
    receipt_path = release['gpu_smoke_receipt']
    if receipt_path is None:
        raise ValueError('Full model GPU receipt is required')
    receipt = json.loads((root / receipt_path).read_text())
    if (receipt['state'] != 'complete' or not receipt['full_state_restore_bitwise']
            or receipt['optimizer_updates'] < 1):
        raise ValueError('Full model GPU update and checkpoint recovery are unverified')
    training = protocol['training']
    if (training['batch_size'] != training['micro_batch_size'] * training['gradient_accumulation']
            or training['finetuning'] != 'full' or training['image_channels'] != 6
            or training['action_horizon'] != 10
            or training['checkpoint_updates'][-1] != training['num_steps']):
        raise ValueError('Formal Metric Cue training recipe is inconsistent')
    layout = training['gradient_sharding']
    if (layout['strategy'] != 'fsdp' or layout['optimizer_device'] != 'cpu'
            or len(run['resources']['gpu_indices']) != layout['devices']
            or not run['resources']['cpu_optimizer']):
        raise ValueError('Formal Metric Cue GPU and optimizer layout differs from the protocol')
    if (run['data']['rgbd_manifest_sha256'] != protocol['data']['rgbd_manifest_sha256']
            or run['data']['geometry_manifest_sha256'] != protocol['data']['geometry_manifest_sha256']
            or run['norm_stats_sha256'] != protocol['data']['norm_stats_sha256']):
        raise ValueError('Formal Metric Cue data differs from the frozen Spatial data')
    run = dict(run)
    run['training'] = dict(training, seed=seed)
    run['cue_config'] = str(root / protocol['arms'][arm])
    run['formal_run_id'] = f'L09_spatial_{arm}_seed{seed}'
    run['protocol_sha256'] = hashlib.sha256(protocol_bytes).hexdigest()
    return run, protocol
