"""Reuse exact Spatial state replay for the three remaining training suites."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from prepare_main_spatial_geometry import capture_frame, file_sha, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v03-root', required=True, type=Path)
    parser.add_argument('--suite', choices=('libero_object', 'libero_goal', 'libero_10'), required=True)
    parser.add_argument('--coverage', required=True, type=Path)
    parser.add_argument('--suite-manifest', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--status', required=True, type=Path)
    parser.add_argument('--device-id', required=True, type=int)
    args = parser.parse_args()
    import fcntl
    import h5py
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out/'.replay.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sys.path.insert(0, str(args.v03_root/'repo/actioncoord_experiments/src'))
    sys.path.insert(0, str(args.v03_root/'repo/third_party/libero'))
    from actioncoord.data.state_adapter import state8_from_observation
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    coverage = json.loads(args.coverage.read_text())
    suite_manifest = json.loads(args.suite_manifest.read_text())
    assert coverage['complete'] and coverage['suite'] == args.suite
    coverage_sha = file_sha(args.coverage)
    rows = []
    for instruction, task in coverage['tasks'].items():
        for kind in ('modified', 'original_exceptions'):
            if kind not in task:
                continue
            source = task[kind]
            for match in source['matches']:
                rows.append(dict(match, task=instruction, benchmark_stem=task['benchmark_stem'],
                    source=source['source'], source_sha256=source['sha256']))
    rows.sort(key=lambda row: row['episode_index'])
    assert [row['episode_index'] for row in rows] == suite_manifest['episode_ids']
    assert len(rows) == suite_manifest['episodes'] and sum(row['length'] for row in rows) == suite_manifest['frames']
    suite = benchmark.get_benchmark_dict()[args.suite]()
    tasks = {Path(suite.get_task(i).bddl_file).stem: suite.get_task(i) for i in range(suite.n_tasks)}
    status = dict(state='running', pid=os.getpid(), suite=args.suite, device_id=args.device_id,
        completed_episodes=0, completed_frames=0, expected_episodes=len(rows), expected_frames=suite_manifest['frames'],
        current_episode=None, started_at_unix=time.time(), training_started=False, policy_rollouts=0)
    receipts, checked = [], {}
    write_json(args.status, status)
    try:
        for row in rows:
            ep = row['episode_index']
            status['current_episode'] = ep
            write_json(args.status, status)
            destination = args.out/('episode_%06d_rgbd.npz' % ep)
            receipt = destination.with_suffix('.json')
            if destination.exists() or receipt.exists():
                if not destination.is_file() or not receipt.is_file():
                    raise ValueError('Incomplete existing episode: %d' % ep)
                saved = json.loads(receipt.read_text())
                if saved['coverage_sha256'] != coverage_sha or saved['npz_sha256'] != file_sha(destination):
                    raise ValueError('Existing replay integrity mismatch: %d' % ep)
            else:
                if row['source'] not in checked:
                    checked[row['source']] = file_sha(row['source'])
                assert checked[row['source']] == row['source_sha256']
                task = tasks[row['benchmark_stem']]
                bddl = Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
                env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256,
                    camera_depths=True, render_gpu_device_id=args.device_id)
                try:
                    with h5py.File(row['source'], 'r') as source:
                        demo = source['data'][row['demo']]
                        indices = np.asarray(row.get('source_frame_indices', list(range(row['length']))), np.int64)
                        assert len(indices) == row['length'] and np.all(np.diff(indices) > 0)
                        env.seed(0)
                        env.reset()
                        env.set_init_state(np.asarray(demo['states'][indices[0]], np.float64))
                        payload, comparisons = {}, []
                        for frame, source_frame in enumerate(indices):
                            sim_state = np.asarray(demo['states'][source_frame], np.float64)
                            _, capture = capture_frame(env, sim_state, state8_from_observation)
                            for key, value in capture.items():
                                payload.setdefault(key, []).append(value)
                            if frame in {0, len(indices)//2, len(indices)-1}:
                                comparison = {'filtered_frame': frame, 'source_frame': int(source_frame)}
                                for view, key in [('agentview_rgb', 'agentview_rgb'), ('eye_in_hand_rgb', 'wrist_rgb')]:
                                    reference = np.asarray(demo['obs'][view][source_frame])[::-1, ::-1]
                                    actual = capture[key]
                                    if actual.shape != reference.shape:
                                        from PIL import Image
                                        actual = np.asarray(Image.fromarray(actual).resize(
                                            (reference.shape[1], reference.shape[0]), Image.BILINEAR))
                                    difference = actual.astype(np.int16)-reference.astype(np.int16)
                                    comparison[view] = dict(mae=float(np.abs(difference).mean()), max_abs=int(np.abs(difference).max()),
                                        native_reference_shape=list(reference.shape), rendered_training_shape=list(capture[key].shape))
                                comparisons.append(comparison)
                        payload = {key: np.asarray(value) for key, value in payload.items()}
                        payload['actions'] = np.asarray(demo['actions'][indices], np.float32)
                        payload['source_frame_indices'] = indices
                        assert payload['actions'].shape == (row['length'], 7)
                        temporary = destination.with_suffix('.tmp.npz')
                        with temporary.open('xb') as stream:
                            np.savez_compressed(stream, **payload)
                        os.replace(temporary, destination)
                    saved = dict(episode_index=ep, task=row['task'], frames=row['length'],
                        npz=str(destination.resolve()), npz_sha256=file_sha(destination),
                        source=row['source'], source_sha256=row['source_sha256'], demo=row['demo'],
                        coverage_sha256=coverage_sha, source_action_sequence_matched_to_lerobot=True,
                        sim_state_restore_exact=True, rgb_depth_state_same_render=True,
                        demonstration_rgb_comparisons=comparisons,
                        camera_contract='robosuite optical camera-to-world; K before -.5px; RGB/depth policy rot180',
                        target_gt_policy_inputs=False)
                    write_json(receipt, saved)
                finally:
                    env.close()
            receipts.append(saved)
            status['completed_episodes'] += 1
            status['completed_frames'] += saved['frames']
            write_json(args.status, status)
            print(json.dumps(dict(episode=ep, frames=saved['frames'], state='complete')), flush=True)
        write_json(args.out/'manifest.json', dict(complete=True, suite=args.suite, episodes=receipts,
            total_frames=status['completed_frames'], coverage_sha256=coverage_sha))
        status.update(state='complete', current_episode=None, finished_at_unix=time.time())
    except BaseException as error:
        status.update(state='failed', error=repr(error), finished_at_unix=time.time())
        raise
    finally:
        write_json(args.status, status)


if __name__ == '__main__':
    main()
