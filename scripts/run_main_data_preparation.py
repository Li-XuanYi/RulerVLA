"""Detached preparation worker. Stops on error; never launches training/eval."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
V03 = Path('/home/yqt/XuanyiLi/actioncoord/work/v03_sync_rgbd_20261003T164806')
STORE = Path('/home/Dataset2/yqt/rulervla/main_comparison_v1/data_preparation')
RGBD = Path('/home/Dataset2/yqt/openpi-aimbot/spatialcue_depth_source/regenerated_rgbd_v02')
SIM_PYTHON = '/home/yqt/XuanyiLi/openpi/examples/libero/.venv/bin/python'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=('spatial', 'sources'), required=True)
    parser.add_argument('--device-id', type=int, default=3)
    args = parser.parse_args()
    import fcntl
    log_dir = ROOT / 'outputs/main_comparison/logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    lock = (log_dir / (args.worker + '.lock')).open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = ROOT / 'outputs/main_comparison' / (args.worker + '_preparation_status.json')
    status = {'state': 'running', 'worker': args.worker, 'pid': os.getpid(),
        'started_at_unix': time.time(), 'completed_stages': [], 'current_stage': None,
        'formal_training_started': False, 'policy_rollouts': 0}
    def save():
        temporary = status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(status, indent=2) + '\n')
        os.replace(temporary, status_path)
    save()
    if args.worker == 'spatial':
        common = [SIM_PYTHON, str(ROOT / 'scripts/prepare_main_spatial_geometry.py'),
            '--v03-root', str(V03), '--rgbd-root', str(RGBD), '--device-id', str(args.device_id)]
        stages = [('ten_task_replay_probe', common + ['--probe', '--out', str(STORE / 'spatial_probe'),
                    '--status', str(ROOT / 'outputs/main_comparison/spatial_probe_status.json')]),
                  ('spatial_all_432_geometry', common + ['--out', str(STORE / 'spatial_geometry'),
                    '--status', str(ROOT / 'outputs/main_comparison/spatial_geometry_status.json')])]
    else:
        stages = [(suite, [sys.executable, str(ROOT / 'scripts/prepare_main_source_coverage.py'),
            '--suite', suite, '--lerobot', '/home/Dataset2/yqt/openpi/huggingface/lerobot/physical-intelligence/libero',
            '--task-map', str(V03 / 'repo/third_party/libero/libero/libero/benchmark/libero_suite_task_map.py'),
            '--store', str(STORE / 'source_hdf5'), '--out', str(STORE / 'coverage' / suite)])
            for suite in ('libero_object', 'libero_goal', 'libero_10')]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
        MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', MUJOCO_EGL_DEVICE_ID=str(args.device_id),
        LIBERO_CONFIG_PATH='/home/yqt/XuanyiLi/actioncoord/fixes/framework_fix_20261002T151206Z/libero_config')
    try:
        for stage, command in stages:
            status.update(current_stage=stage, command=command)
            with (log_dir / (stage + '.log')).open('a') as stream:
                child = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
                status['child_pid'] = child.pid
                status['log'] = str(stream.name)
                save()
                code = child.wait()
            if code:
                raise RuntimeError('%s failed with exit %d; see %s' % (stage, code, status['log']))
            status['completed_stages'].append(stage)
            save()
        status.update(state='complete', current_stage=None, child_pid=None, finished_at_unix=time.time())
    except BaseException as error:
        status.update(state='failed', error=repr(error), finished_at_unix=time.time())
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
