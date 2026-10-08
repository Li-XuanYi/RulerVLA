"""Mirror authoritative Camera training status for the existing GPU5 follow-up queue."""
import json
import subprocess
import time
from pathlib import Path
from main_gpu_smoke import write_json

root = Path(__file__).resolve().parents[1]
base = root / 'outputs/small_ablation_v1'
record = base / 'train_records/S22_camera_xyz_only_spatial_seed42/status.json'
path = base / 'camera_gpu5_now_status.json'
unit = 'rulervla-camera-train-gpu5-20261008.service'
previous = path.with_name('camera_gpu5_now_before_systemd_training.json')
if not previous.exists():
    previous.write_text(path.read_text())
while True:
    current = json.loads(record.read_text())
    active = subprocess.check_output(['systemctl', '--user', 'show', unit,
                                     '-p', 'ActiveState', '--value'], text=True).strip()
    current.update(systemd_unit=unit, gpu=5, child_pid=current['pid'],
                   log=str(base/'logs/S22_camera_xyz_only_spatial_seed42.log'))
    if current['state'] == 'running' and active not in ('active', 'activating'):
        current.update(state='failed', error='Training service stopped before completion')
    write_json(path, current)
    if current['state'] != 'running':
        break
    time.sleep(10)
