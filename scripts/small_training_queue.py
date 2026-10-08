"""S22/S23 GPU4/5 admission, per-arm full-state smoke, then frozen training."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from main_gpu_smoke import write_json
from small_training_contract import validate_run, sha


def live(pid):
    if not pid: return False
    try: os.kill(pid,0); return True
    except ProcessLookupError: return False


class ExistingChild:
    """Monitor an existing child without restarting its optimizer state."""
    def __init__(self, pid, record):
        self.pid, self.record = pid, record

    def poll(self):
        if live(self.pid):
            proc = Path('/proc/%d/stat' % self.pid)
            if proc.exists() and proc.read_text().split(') ',1)[1].split()[0] != 'Z':
                return None
        state = json.loads(self.record.read_text())
        return 0 if state['state'] == 'complete' else 1

    def terminate(self):
        import signal
        os.kill(self.pid, signal.SIGTERM)

    def kill(self):
        import signal
        os.kill(self.pid, signal.SIGKILL)

    def wait(self, timeout=None):
        deadline = None if timeout is None else time.monotonic()+timeout
        while self.poll() is None:
            if deadline is not None and time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired('existing training child', timeout)
            time.sleep(1)
        return self.poll()


def resources(root, gpu, profile=None):
    query = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True)
    values = {int(line.split(',')[0]):[int(v) for v in line.split(',')[1:]] for line in query.strip().splitlines()}
    memory = {line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
    stable = True
    for worker in (0,1):
        owner = json.loads((root/'outputs/main_comparison'/('spatial_training_queue_worker%d_status.json'%worker)).read_text())
        if owner['state'] == 'complete': continue
        path = root/'outputs/main_comparison/train_records'/str(owner.get('current_run'))/'status.json'
        if owner['state'] != 'running' or not path.exists(): stable=False; continue
        status = json.loads(path.read_text())
        stable &= live(status['pid']) and status['state']=='running' and status.get('optimizer_updates',0)>=50
    # Wait until any already-admitted main worker is past its high-memory initialization.
    for path in (root/'outputs/main_comparison').glob('remaining_suite_queue_worker*_status.json'):
        owner = json.loads(path.read_text())
        if not live(owner.get('child_pid')): continue
        path = root/'outputs/main_comparison/train_records'/str(owner['current_run'])/'status.json'
        if not path.exists(): stable=False; continue
        status = json.loads(path.read_text())
        stable &= status['state']=='running' and status.get('optimizer_updates',0)>=50
    host_required=profile['required_host_gib'] if profile else 380
    gpu_required=profile['required_gpu_free_mib'] if profile else 70000
    receipt = dict(host_available_gib=round(memory['MemAvailable']/1024**2,1),required_host_gib=host_required,
        gpu_free_mib=values[gpu][0],required_gpu_free_mib=gpu_required,gpu_utilization=values[gpu][1],
        existing_training_initialization_stable=bool(stable),checked_at_unix=time.time())
    ready = stable and memory['MemAvailable']>=host_required*1024**2 and values[gpu][0]>=gpu_required and values[gpu][1]<=20
    return ready,receipt


def memory_profile(root):
    path=root/'outputs/small_ablation_v1/platform_memory_release.json'
    if not path.exists(): return None
    profile=json.loads(path.read_text())
    assert profile['state']=='validated' and profile['allocator']=='platform'
    assert profile['required_host_gib']==220 and profile['required_gpu_free_mib']==54369
    smoke=Path(profile['validated_smoke'])
    assert sha(smoke)==profile['smoke_sha256']
    result=json.loads(smoke.read_text())
    assert result['state']=='complete' and result['full_train_state_restore_bitwise']
    assert result['blocking_cpu_snapshot_unchanged_after_save']
    assert result['optimizer_updates']==2 and result['effective_batch']==32
    return profile


def guarded_wait(child,path,status,record=None,release_admission=None):
    while child.poll() is None:
        memory={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
        status['host_available_gib_live']=round(memory['MemAvailable']/1024**2,1)
        if record is not None and record.exists():
            current=json.loads(record.read_text());status['optimizer_updates']=current.get('optimizer_updates',0)
            if release_admission is not None and status['optimizer_updates']>=50:
                release_admission();release_admission=None;status['stage']='formal_training'
        write_json(path,status)
        if memory['MemAvailable']<32*1024**2:
            child.terminate()
            try: child.wait(timeout=30)
            except subprocess.TimeoutExpired: child.kill();child.wait()
            raise RuntimeError('Stopped only small experiment child to preserve 32GiB host reserve')
        time.sleep(2)
    return child.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker-index',type=int,choices=(0,1),required=True)
    parser.add_argument('--include-camera',action='store_true',help='GPU5: ActionCoord then CameraXYZ then Ruler')
    parser.add_argument('--adopt-running',action='store_true',help='Monitor an already-running matching training child')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    gpu = (4,5)[args.worker_index]
    base = root/'outputs/small_ablation_v1'
    logs = base/'logs'; logs.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock = (logs/('worker%d.lock'%args.worker_index)).open('w')
    fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Same lock as the live large queues; retain it until initialization settles.
    admission = (root/'outputs/main_comparison/logs/remaining_suite_admission.lock').open('w')
    matrix = json.loads((root/'experiments/small_ablation_v1/run_matrix.json').read_text())
    methods = (['CameraXYZ-only'] if args.worker_index==0 else ['ActionCoord-only','Ruler-only'])
    if args.include_camera:
        assert args.worker_index == 1
        methods = ['ActionCoord-only','CameraXYZ-only','Ruler-only']
    by_method = {run['method']:run for run in matrix['runs']}
    runs = [by_method[method] for method in methods]
    path = base/('training_queue_worker%d_status.json'%args.worker_index)
    status = dict(state='running',stage='checking_contract',pid=os.getpid(),session=os.getsid(0),
        gpu=gpu,worker_index=args.worker_index,assigned_runs=[r['run_id'] for r in runs],
        child_pid=None,completed_runs=[],formal_training_started=False,started_at_unix=time.time())
    write_json(path,status)
    held = False
    try:
        for run in runs:
            validate_run(root,run)
            profile=memory_profile(root)
            config = base/'released_runs'/(run['run_id']+'.json')
            write_json(config,run)  # Identity remains exactly the frozen spec.
            smoke_out = base/'gpu_smoke'/run['run_id']
            train_status = base/'train_records'/run['run_id']/'status.json'
            if train_status.exists():
                old=json.loads(train_status.read_text())
                if old['state']=='complete' and (Path(run['checkpoint_root'])/str(run['shared_training']['optimizer_updates']-1)/'V03_COPY_COMPLETE.json').exists():
                    status['completed_runs'].append(run['run_id']); continue
                if args.adopt_running and old['state']=='running' and live(old.get('pid')):
                    assert old['run_id']==run['run_id'] and old['gpu_ids']==str(gpu)
                    commandline=Path('/proc/%d/cmdline'%old['pid']).read_bytes().replace(b'\0',b' ').decode()
                    assert 'small_spatial_train.py' in commandline and str(config) in commandline
                    execution=json.loads((Path(run['checkpoint_root'])/'execution_lock.json').read_text())
                    assert execution['run']==run and execution['effective_batch']==32
                    status.update(current_run=run['run_id'],method=run['method'],stage='formal_training',
                        child_pid=old['pid'],formal_training_started=True,adopted_existing_child=True,
                        target_optimizer_updates=run['shared_training']['optimizer_updates'],
                        checkpoint_root=run['checkpoint_root'],log=str(logs/(run['run_id']+'.log')))
                    write_json(path,status)
                    if guarded_wait(ExistingChild(old['pid'],train_status),path,status,train_status):
                        raise RuntimeError('Adopted training failed: '+run['run_id'])
                    final=json.loads(train_status.read_text())
                    assert final['optimizer_updates']==run['shared_training']['optimizer_updates']
                    assert (Path(final['final_checkpoint'])/'V03_COPY_COMPLETE.json').exists()
                    status['completed_runs'].append(run['run_id']);status['child_pid']=None
                    write_json(path,status)
                    continue
                raise RuntimeError('Existing run needs inspected resume: '+run['run_id'])
            if any(p.name.isdigit() for p in Path(run['checkpoint_root']).glob('*')):
                raise RuntimeError('Existing checkpoints need inspected resume: '+run['run_id'])
            status.update(current_run=run['run_id'],method=run['method'],stage='waiting_for_resources',
                checkpoint_root=run['checkpoint_root'],target_optimizer_updates=run['shared_training']['optimizer_updates'])
            write_json(path,status)
            while True:
                try: fcntl.flock(admission,fcntl.LOCK_EX | fcntl.LOCK_NB); held=True
                except BlockingIOError:
                    status['stage']='waiting_for_admission'; write_json(path,status); time.sleep(20); continue
                ready,receipt = resources(root,gpu,profile)
                status.update(stage='waiting_for_resources',resource_snapshot=receipt); write_json(path,status)
                if ready: break
                fcntl.flock(admission,fcntl.LOCK_UN); held=False; time.sleep(20)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),XLA_PYTHON_CLIENT_PREALLOCATE='false',
                OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',WANDB_MODE='disabled',
                NCCL_P2P_DISABLE='1',NCCL_IB_DISABLE='1',NCCL_SHM_DISABLE='1',NCCL_SOCKET_IFNAME='lo',NCCL_CUMEM_HOST_ENABLE='0')
            env.pop('JAX_PLATFORMS',None)
            if profile:
                env['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'
                status['memory_profile']=str(base/'platform_memory_release.json')
            smoke_status=smoke_out/'status.json'
            if profile and run['method']=='ActionCoord-only':
                smoke_status=Path(profile['validated_smoke'])
            if smoke_status.exists():
                old=json.loads(smoke_status.read_text())
                assert old['state']=='complete' and old['full_train_state_restore_bitwise'] and old['method']==run['method']
            else:
                smoke_out.mkdir(parents=True,exist_ok=True)
                command=[sys.executable,str(root/'scripts/small_gpu_smoke.py'),
                    '--runtime',str(root/'outputs/main_comparison/runtime_v3'),
                    '--v03-root','/home/yqt/XuanyiLi/actioncoord/work/v03_sync_rgbd_20261003T164806',
                    '--rgbd-root','/home/Dataset2/yqt/openpi-aimbot/spatialcue_depth_source/regenerated_rgbd_v02',
                    '--geometry-root','/home/Dataset2/yqt/rulervla/main_comparison_v1/data_preparation/spatial_geometry',
                    '--out',str(smoke_out),'--method',run['method'],'--save-restore','--cpu-optimizer','--fsdp-devices','1',
                    '--aimbot-source','/home/yqt/XuanyiLi/openpi0-aimbot/third_party/AimBot/src',
                    '--axisguide-source','/home/yqt/XuanyiLi/AxisGuide-code/src/lerobot/datasets/visual_cue_utils.py']
                if profile:command.append('--blocking-cpu-save')
                smoke_log=logs/(run['run_id']+'_smoke.log')
                with smoke_log.open('a') as stream:
                    child=subprocess.Popen(command,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
                status.update(stage='per_arm_full_ft_smoke',child_pid=child.pid,log=str(smoke_log));write_json(path,status)
                if guarded_wait(child,path,status): raise RuntimeError('Smoke failed: '+str(smoke_log))
            command=[sys.executable,str(root/'scripts/small_spatial_train.py'),'--run-config',str(config),
                '--smoke',str(smoke_status),'--cue-audit',str(root/'outputs/main_comparison/cue_audit_20261006_with_wrist/audit.json')]
            if profile:command.append('--blocking-cpu-save')
            train_log=logs/(run['run_id']+'.log')
            with train_log.open('a') as stream:
                child=subprocess.Popen(command,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
            status.update(stage='formal_training_initialization',child_pid=child.pid,log=str(train_log),
                formal_training_started=True,smoke_status=str(smoke_status));write_json(path,status)
            def release():
                nonlocal held
                fcntl.flock(admission,fcntl.LOCK_UN);held=False
            code=guarded_wait(child,path,status,train_status,release)
            if held:release()
            if code: raise RuntimeError('Training failed: '+str(train_log))
            final=json.loads(train_status.read_text())
            assert final['state']=='complete' and final['optimizer_updates']==run['shared_training']['optimizer_updates']
            assert (Path(final['final_checkpoint'])/'V03_COPY_COMPLETE.json').exists()
            status['completed_runs'].append(run['run_id']);status['child_pid']=None;write_json(path,status)
        status.update(state='complete',stage='complete',finished_at_unix=time.time())
    except BaseException as error:
        status.update(state='failed',error=repr(error),finished_at_unix=time.time());raise
    finally:
        if held: fcntl.flock(admission,fcntl.LOCK_UN)
        write_json(path,status)


if __name__=='__main__': main()
