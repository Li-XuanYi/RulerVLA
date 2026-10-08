"""User-approved GPU2 switch after Baseline's first verified 5000-update save."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from main_gpu_smoke import write_json

root = Path(__file__).resolve().parents[1]
base = root/'outputs/main_comparison'
switch_path = base/'rulervla_priority_switch_status.json'
owner_path = base/'spatial_training_queue_worker0_status.json'
baseline = 'libero_spatial_baseline_seed42'
ruler = 'libero_spatial_rulervla_main_seed42'
owner = json.loads(owner_path.read_text())
assert owner['current_run'] == baseline and owner['gpu'] == 2
original = list(owner['assigned_runs'])
assert ruler not in original
old_parent = owner['pid']
baseline_pid = owner['child_pid']
switch = dict(state='running', stage='waiting_baseline_verified_5000', pid=os.getpid(),
              baseline_pid=baseline_pid, baseline_original_queue_pid=old_parent,
              target_updates=5000, next_run=ruler, gpu=2)
write_json(switch_path, switch)

def record(run):
    return base/'train_records'/run/'status.json'

def alive(pid):
    p = Path('/proc/%d/stat'%pid)
    return p.exists() and p.read_text().split(') ',1)[1].split()[0] != 'Z'

def stop_parent(pid):
    cmd = Path('/proc/%d/cmdline'%pid).read_bytes()
    assert b'main_spatial_train_queue.py' in cmd
    os.kill(pid, signal.SIGSTOP)

try:
    while True:
        current = json.loads(record(baseline).read_text())
        assert current['pid'] == baseline_pid and alive(baseline_pid), 'Baseline interrupted before save'
        saved = Path(current['checkpoint_root'])/'4999'
        if current.get('latest_checkpoint') == str(saved) and (saved/'V03_COPY_COMPLETE.json').exists():
            assert current['optimizer_updates'] >= 5000 and current['stage'] != 'saving_checkpoint'
            break
        switch['baseline_updates'] = current['optimizer_updates']
        write_json(switch_path, switch)
        time.sleep(5)
    runtime = root/'outputs/main_comparison/runtime_v3'
    v03 = Path('/home/yqt/XuanyiLi/actioncoord/work/v03_sync_rgbd_20261003T164806')
    sys.path[:0] = [str(v03/'repo/actioncoord_experiments/src')]
    from actioncoord.integration.v03_storage import verify_checkpoint
    verify_checkpoint(saved, expected_updates=5000)
    stop_parent(old_parent)
    peer = json.loads((base/'spatial_training_queue_worker1_status.json').read_text())
    assert peer['current_run'] != ruler and peer['state'] == 'running'
    peer_pid = peer['pid']
    stop_parent(peer_pid)  # Its existing AimBot child continues; prevents duplicate Ruler admission.
    cmd = Path('/proc/%d/cmdline'%baseline_pid).read_bytes()
    assert b'main_spatial_train.py' in cmd and baseline.encode() in cmd
    os.kill(baseline_pid, signal.SIGTERM)
    for _ in range(60):
        if not alive(baseline_pid): break
        time.sleep(1)
    assert not alive(baseline_pid), 'Baseline failed to release GPU; no new child launched'
    os.kill(old_parent, signal.SIGTERM)
    os.kill(old_parent, signal.SIGCONT)
    current.update(state='paused', stage='paused_at_verified_checkpoint', latest_checkpoint=str(saved),
                   resume_optimizer_updates=5000, pause_reason='User prioritizes RulerVLA')
    write_json(record(baseline), current)
    write_json(Path(current['checkpoint_root'])/'status.json', current)
    switch.update(stage='rulervla_first_then_baseline_resume', saved_checkpoint=str(saved), peer_queue_paused=peer_pid)
    write_json(switch_path, switch)
    owner.update(pid=os.getpid(), child_pid=None, assigned_runs=[ruler]+original,
                 completed_runs=[], stage='waiting_for_gpu', priority_switch=str(switch_path))
    env=dict(os.environ, CUDA_VISIBLE_DEVICES='2', XLA_PYTHON_CLIENT_PREALLOCATE='false',
             OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', WANDB_MODE='disabled')
    env.pop('JAX_PLATFORMS', None)
    for run in [ruler]+original:
        config_path=root/'experiments/main_comparison_v1/runs'/(run+'.json')
        config=json.loads(config_path.read_text())
        existing=json.loads(record(run).read_text()) if record(run).exists() else None
        if existing and existing['state']=='complete':
            verify_checkpoint(Path(config['checkpoint_root'])/'29999', expected_updates=30000)
            owner['completed_runs'].append(run); continue
        assert not existing or run==baseline and existing['state']=='paused'
        owner.update(current_run=run, child_pid=None, stage='waiting_for_gpu')
        while True:
            mem={x.split(':')[0]:int(x.split()[1]) for x in Path('/proc/meminfo').read_text().splitlines()}
            rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
            free={int(x.split(',')[0]):int(x.split(',')[1]) for x in rows.splitlines()}[2]
            owner.update(host_available_gib=mem['MemAvailable']/1024**2, gpu_free_mib=free)
            write_json(owner_path,owner)
            if free>=60000 and mem['MemAvailable']>=220*1024**2: break
            time.sleep(10)
        command=[sys.executable,str(root/'scripts/main_spatial_train.py'),'--run-config',str(config_path),
                 '--smoke',str(base/'gpu_smoke_20261006/rulervla_main_attempt7/status.json'),
                 '--cue-audit',str(base/'cue_audit_20261006_with_wrist/audit.json')]
        if run==baseline: command.append('--resume')
        log=base/'logs'/(run+'.log')
        with log.open('a') as stream:
            child=subprocess.Popen(command,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
        owner.update(stage='formal_training',child_pid=child.pid,log=str(log),formal_training_started=True)
        write_json(owner_path,owner)
        assert child.wait()==0, 'Training failed: '+run
        verify_checkpoint(Path(config['checkpoint_root'])/'29999',expected_updates=30000)
        owner['completed_runs'].append(run)
        if run==ruler:
            os.kill(peer_pid,signal.SIGCONT)
            switch.update(stage='baseline_resuming',rulervla_completed=True)
            write_json(switch_path,switch)
    owner.update(state='complete',stage='complete',child_pid=None)
    write_json(owner_path,owner)
    switch.update(state='complete',stage='complete')
except BaseException as error:
    switch.update(state='failed',error=repr(error))
    raise
finally:
    write_json(switch_path,switch)
