"""Supplement the frozen Spatial RGB-D with both-view geometry and state8.

Reuse V0.3's exact state adapter and the existing benchmark/seed-0 replay.
Never alter the original RGB-D cache, demonstration, checkpoints or G results.
Compatible with the installed simulator Python 3.8.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def check_rgb(actual, expected, out, episode, frame, view):
    if np.array_equal(actual, expected):
        return None
    from PIL import Image
    difference = actual.astype(np.int16) - expected.astype(np.int16)
    a224 = np.asarray(Image.fromarray(actual).resize((224, 224), Image.BILINEAR))
    b224 = np.asarray(Image.fromarray(expected).resize((224, 224), Image.BILINEAR))
    pixels = np.argwhere(difference != 0)
    evidence = {'episode': episode, 'filtered_frame': frame, 'view': view,
         'mismatched_channels': len(pixels), 'signed_max_abs_delta': int(np.abs(difference).max()),
         'resized_policy_rgb_exact': bool(np.array_equal(a224, b224)),
         'policy_rgb_max_abs_delta': int(np.abs(a224.astype(np.int16)-b224.astype(np.int16)).max()),
         'examples': [{'index': index.tolist(), 'actual': int(actual[tuple(index)]),
                       'expected': int(expected[tuple(index)])} for index in pixels[:10]]}
    # A renderer can differ by one uint8 quantization unit at a raster edge.
    # The actual training RGB remains the original frozen cache for ALL arms.
    # State, action mapping and agent depth must still match exactly below.
    evidence['within_rgb_quantization_bound'] = bool(
        evidence['signed_max_abs_delta'] <= 1 and len(pixels) / difference.size <= 1e-4)
    write_json(out / ('rgb_mismatch_ep%d_f%d_%s.json' % (episode, frame, view)), evidence)
    if not evidence['within_rgb_quantization_bound']:
        raise ValueError('RGB disagreement exceeds the recorded quantization bound')
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v03-root', required=True, type=Path)
    parser.add_argument('--rgbd-root', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--status', required=True, type=Path)
    parser.add_argument('--device-id', type=int, required=True)
    parser.add_argument('--probe', action='store_true', help='first/middle/last from first episode of each task')
    args = parser.parse_args()
    import fcntl
    import h5py
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / '.prepare.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sys.path.insert(0, str(args.v03_root / 'repo/actioncoord_experiments/src'))
    sys.path.insert(1, str(args.v03_root / 'repo/third_party/libero'))
    from actioncoord.data.state_adapter import state8_from_observation
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.camera_utils import get_real_depth_map, get_camera_intrinsic_matrix, get_camera_extrinsic_matrix

    manifest_path = args.rgbd_root / 'regen_manifest.json'
    source_manifest_sha = file_sha(manifest_path)
    rows = sorted(json.loads(manifest_path.read_text())['episodes'], key=lambda row: row['episode_index'])
    assert len(rows) == 432 and sum(row['frames'] for row in rows) == 52970
    suite = benchmark.get_benchmark_dict()['libero_spatial']()
    tasks = {suite.get_task(i).language: suite.get_task(i) for i in range(suite.n_tasks)}
    if args.probe:
        rows = [next(row for row in rows if row['task'] == task) for task in tasks]
    status = {'schema': 'rulervla_main_spatial_geometry_v1', 'state': 'running',
        'pid': os.getpid(), 'gpu_render_device': args.device_id, 'probe': args.probe,
        'expected_episodes': len(rows), 'expected_frames': sum(row['frames'] if not args.probe else min(3, row['frames']) for row in rows),
        'completed_episodes': 0, 'completed_frames': 0, 'current_episode': None,
        'started_at_unix': time.time(), 'source_manifest_sha256': source_manifest_sha,
        'output_dir': str(args.out.resolve()), 'training_started': False, 'policy_rollouts': 0}
    write_json(args.status, status)
    verified_sources, evidence = {}, []
    try:
        for row in rows:
            ep = row['episode_index']
            status['current_episode'] = ep
            write_json(args.status, status)
            destination = args.out / ('episode_%06d_geometry.npz' % ep)
            receipt = destination.with_suffix('.json')
            if destination.exists() or receipt.exists():
                if not destination.is_file() or not receipt.is_file():
                    raise ValueError('Incomplete existing episode; refusing overwrite: ' + str(ep))
                saved = json.loads(receipt.read_text())
                if (saved['source_manifest_sha256'] != source_manifest_sha or saved['probe'] != args.probe
                        or saved['geometry_sha256'] != file_sha(destination)):
                    raise ValueError('Existing geometry provenance/integrity mismatch: ' + str(ep))
                evidence.append(saved)
                status['completed_episodes'] += 1
                status['completed_frames'] += saved['frames']
                write_json(args.status, status)
                continue
            source = row['source']
            if source not in verified_sources:
                verified_sources[source] = file_sha(source)
            if verified_sources[source] != row['source_sha256'] or file_sha(row['npz']) != row['npz_sha256']:
                raise ValueError('Frozen RGB-D/source changed: ' + str(ep))
            task = tasks[row['task']]
            bddl = Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
            env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256,
                camera_widths=256, camera_depths=True, render_gpu_device_id=args.device_id)
            try:
                with np.load(row['npz'], allow_pickle=False) as cached, h5py.File(source, 'r') as h5:
                    group = h5['data'][row['demo']]
                    indices = np.asarray(cached['source_frame_indices'], dtype=np.int64)
                    assert len(indices) == row['frames'] and np.all(np.diff(indices) > 0)
                    assert indices.min() >= 0 and indices.max() < len(group['states'])
                    np.testing.assert_array_equal(cached['actions'], np.asarray(group['actions'][indices], dtype=np.float32))
                    chosen = sorted({0, len(indices)//2, len(indices)-1}) if args.probe else list(range(len(indices)))
                    env.seed(0)
                    env.reset()
                    env.set_init_state(np.asarray(group['states'][indices[0]], dtype=np.float64))
                    payload = {key: [] for key in ('state8', 'eef_quaternion_xyzw', 'wrist_depth_m',
                        'intrinsic', 'camera_to_world', 'wrist_intrinsic', 'wrist_camera_to_world', 'sim_state_sha256')}
                    rgb = cached['agentview_rgb']
                    wrist = cached['wrist_rgb']
                    depth = cached['agentview_depth_m']
                    ee = cached['ee_world_m']
                    rgb_quantization = []
                    for frame in chosen:
                        sim_state = np.asarray(group['states'][indices[frame]], dtype=np.float64)
                        env.sim.set_state_from_flattened(sim_state)
                        env.sim.forward()
                        observation = env.env._get_observations(force_update=True)
                        np.testing.assert_array_equal(env.sim.get_state().flatten(), sim_state)
                        # Match the frozen generator's float64 intermediate exactly.
                        actual_depth = get_real_depth_map(env.sim, np.asarray(observation['agentview_depth'], dtype=np.float64).squeeze())
                        np.testing.assert_array_equal(actual_depth[::-1, ::-1].astype(np.float32), depth[frame])
                        state = state8_from_observation(observation)
                        np.testing.assert_array_equal(state[:3], ee[frame])
                        for image, reference, view in [(observation['agentview_image'][::-1, ::-1], rgb[frame], 'agent'),
                                (observation['robot0_eye_in_hand_image'][::-1, ::-1], wrist[frame], 'wrist')]:
                            quantization = check_rgb(image, reference, args.out, ep, frame, view)
                            if quantization is not None:
                                rgb_quantization.append(quantization)
                        wrist_depth = get_real_depth_map(env.sim, np.asarray(observation['robot0_eye_in_hand_depth'], dtype=np.float64).squeeze())
                        if not np.isfinite(wrist_depth).all() or wrist_depth.min() <= 0:
                            raise ValueError('Invalid wrist optical-Z depth')
                        payload['state8'].append(state)
                        payload['eef_quaternion_xyzw'].append(np.asarray(observation['robot0_eef_quat'], np.float32))
                        payload['wrist_depth_m'].append(np.ascontiguousarray(wrist_depth[::-1, ::-1], dtype=np.float32))
                        for camera, k_key, t_key in [('agentview', 'intrinsic', 'camera_to_world'),
                                ('robot0_eye_in_hand', 'wrist_intrinsic', 'wrist_camera_to_world')]:
                            payload[k_key].append(get_camera_intrinsic_matrix(env.sim, camera, 256, 256))
                            payload[t_key].append(get_camera_extrinsic_matrix(env.sim, camera))
                        payload['sim_state_sha256'].append(hashlib.sha256(sim_state.tobytes()).hexdigest())
                    payload = {key: np.asarray(value) for key, value in payload.items()}
                    payload['filtered_frame_indices'] = np.asarray(chosen, np.int64)
                    payload['source_frame_indices'] = indices[chosen]
                    temporary = destination.with_suffix('.tmp.npz')
                    with temporary.open('xb') as stream:
                        np.savez_compressed(stream, **payload)
                    os.replace(temporary, destination)
                saved = {'episode_index': ep, 'task': row['task'], 'frames': len(chosen), 'probe': args.probe,
                    'rgbd_npz': row['npz'], 'rgbd_sha256': row['npz_sha256'], 'source': source,
                    'source_sha256': row['source_sha256'], 'source_demo': row['demo'],
                    'source_manifest_sha256': source_manifest_sha, 'geometry_npz': str(destination.resolve()),
                    'geometry_sha256': file_sha(destination), 'state_restore_max_abs_error': 0,
                    'all_replayed_rgb_wrist_depth_exact': not bool(rgb_quantization),
                    'agent_depth_exact': True, 'state8_ee_exact': True,
                    'rgb_quantization_differences': rgb_quantization,
                    'frozen_training_rgb_unchanged': True,
                    'rgb_quantization_bound': {'max_uint8_channel_delta': 1, 'max_changed_channel_fraction': 1e-4},
                    'camera_contract': 'robosuite optical camera-to-world, K before V03 -0.5px correction; stored RGB/depth rot180',
                    'policy_ground_truth_target_inputs': False}
                write_json(receipt, saved)
                evidence.append(saved)
                status['completed_episodes'] += 1
                status['completed_frames'] += saved['frames']
                write_json(args.status, status)
                print(json.dumps({'episode': ep, 'frames': saved['frames'], 'state_depth_exact': True,
                    'rgb_exact': not bool(rgb_quantization), 'rgb_quantization_frames': len(rgb_quantization)}), flush=True)
            finally:
                env.close()
        status.update(state='complete', current_episode=None, finished_at_unix=time.time())
        write_json(args.out / 'manifest.json', {'complete': True, 'probe': args.probe, 'episodes': evidence,
            'total_frames': status['completed_frames'], 'source_manifest_sha256': source_manifest_sha})
    except BaseException as error:
        status.update(state='failed', error=repr(error), finished_at_unix=time.time())
        raise
    finally:
        write_json(args.status, status)


if __name__ == '__main__':
    main()
