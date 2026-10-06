"""Launch missing data prerequisites in detached processes; no training."""
import json
import os
from pathlib import Path
import subprocess
import sys


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    logs = root / 'outputs/main_comparison/logs'
    logs.mkdir(parents=True, exist_ok=True)
    launched = []
    for worker in ('spatial', 'sources'):
        status_path = root / 'outputs/main_comparison' / (worker + '_preparation_status.json')
        if status_path.exists():
            status = json.loads(status_path.read_text())
            if status['state'] == 'complete':
                launched.append({'worker': worker, 'reused_complete': True})
                continue
            if status['state'] == 'running':
                try:
                    os.kill(status['pid'], 0)
                    # Require a live command belonging to this specific worker.
                    command = Path('/proc/%d/cmdline' % status['pid']).read_bytes()
                    if b'run_main_data_preparation.py' in command and worker.encode() in command:
                        launched.append({'worker': worker, 'pid': status['pid'], 'reused_running': True})
                        continue
                except ProcessLookupError:
                    pass
            raise RuntimeError('Existing incomplete preparation requires inspection: ' + str(status_path))
        with (logs / (worker + '_queue.log')).open('a') as stream:
            process = subprocess.Popen([sys.executable, str(root / 'scripts/run_main_data_preparation.py'),
                '--worker', worker, '--device-id', '3'], cwd=root, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        launched.append({'worker': worker, 'pid': process.pid, 'session': process.pid,
            'log': str(logs / (worker + '_queue.log')),
            'gpu': '3 (rendering only)' if worker == 'spatial' else 'CPU / download'})
    receipt = {'formal_training_started': False, 'policy_rollouts': 0, 'processes': launched}
    (root / 'outputs/main_comparison/preparation_launch.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))
