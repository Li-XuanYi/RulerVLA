"""Detach one common worst-case full-model smoke; never launch formal training."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', default='3,5')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = root / 'outputs/main_comparison/gpu_smoke_20261006/rulervla_main'
    status_file = out/'status.json'
    if status_file.exists():
        status = json.loads(status_file.read_text())
        if status['state'] == 'running':
            try:
                os.kill(status['pid'], 0)
                command = Path('/proc/%d/cmdline' % status['pid']).read_bytes()
                if b'main_gpu_smoke.py' in command:
                    print(json.dumps({'reused_running': True, **status}, indent=2))
                    sys.exit(0)
            except ProcessLookupError:
                pass
        raise RuntimeError('Inspect existing smoke status; refusing overwrite: '+str(status_file))
    selected = [int(value) for value in args.gpus.split(',')]
    if len(selected) != 2 or len(set(selected)) != 2:
        raise ValueError('Two distinct GPUs required')
    query = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    devices = {int(line.split(',')[0]): [int(value) for value in line.split(',')[1:]] for line in query.strip().splitlines()}
    for gpu in selected:
        free, utilization = devices[gpu]
        if free < 50000 or utilization > 10:
            raise RuntimeError('GPU %d is occupied: free=%dMiB utilization=%d%%' % (gpu, free, utilization))
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpus, XLA_PYTHON_CLIENT_PREALLOCATE='false',
        OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1',
        WANDB_MODE='disabled', TF_CPP_MIN_LOG_LEVEL='2')
    v03 = '/home/yqt/XuanyiLi/actioncoord/work/v03_sync_rgbd_20261003T164806'
    out.mkdir(parents=True, exist_ok=False)
    log = out/'train.log'
    command = [sys.executable, str(root/'scripts/main_gpu_smoke.py'),
        '--runtime', str(root/'outputs/main_comparison/runtime_v1'), '--v03-root', v03,
        '--rgbd-root', '/home/Dataset2/yqt/openpi-aimbot/spatialcue_depth_source/regenerated_rgbd_v02',
        '--geometry-root', '/home/Dataset2/yqt/rulervla/main_comparison_v1/data_preparation/spatial_geometry',
        '--out', str(out), '--method', 'RulerVLA-main', '--save-restore',
        '--aimbot-source', '/home/yqt/XuanyiLi/openpi0-aimbot/third_party/AimBot/src',
        '--axisguide-source', '/home/yqt/XuanyiLi/AxisGuide-code/src/lerobot/datasets/visual_cue_utils.py']
    with log.open('a') as stream:
        process = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    receipt = {'pid': process.pid, 'session': process.pid, 'gpus': selected,
        'log': str(log), 'status': str(status_file), 'command': command,
        'stage': 'full_finetuning_smoke', 'formal_training_started': False,
        'micro_batch': 2, 'gradient_accumulation': 16, 'effective_batch': 32,
        'launched_at_unix': time.time()}
    (out/'launch.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))
