"""Split the remaining 36 models between GPU 4 and 5 as resources permit."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from main_gpu_smoke import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker-index',type=int,required=True,choices=(0,1))
    parser.add_argument('--gpu',type=int,required=True,choices=(4,5))
    args=parser.parse_args()
    assert args.gpu==(4,5)[args.worker_index], 'Worker 0 owns GPU 4; worker 1 owns GPU 5'
    root=Path(__file__).resolve().parents[1]
    logs=root/'outputs/main_comparison/logs';logs.mkdir(parents=True,exist_ok=True)
    import fcntl
    suffix='_worker%d'%args.worker_index
    lock=(logs/('remaining_suite_queue%s.lock'%suffix)).open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    admission=(logs/'remaining_suite_admission.lock').open('w')
    path=root/('outputs/main_comparison/remaining_suite_queue%s_status.json'%suffix)
    status=dict(state='running',stage='checking_release',pid=os.getpid(),completed_runs=[],current_run=None,
        child_pid=None,started_at_unix=time.time(),formal_training_started=False,max_additional_concurrent_runs=2,
        worker_index=args.worker_index,worker_count=2,gpu=args.gpu)
    write_json(path,status)
    try:
        suites=('libero_object','libero_goal','libero_10')
        for suite in suites:
            release=root/'outputs/main_comparison/suite_release'/suite
            d=json.loads((release/'release.json').read_text())
            assert d['numerical_preflight_passed'] and not d['visual_review_pending']
            assert json.loads((release/'cue_audit/visual_review.json').read_text())['passed']
        planned=[]
        for config in (root/'experiments/main_comparison_v1/runs').glob('*.json'):
            run=json.loads(config.read_text())
            if run['suite'] in suites:planned.append((config,run))
        methods=['RulerVLA-main','AxisGuide-adapted','Baseline','AimBot']
        planned.sort(key=lambda item:(suites.index(item[1]['suite']),item[1]['seed'],methods.index(item[1]['method'])))
        assert len(planned)==36
        planned=planned[args.worker_index::2]
        assert len(planned)==18
        status['assigned_runs']=[run['run_id'] for _,run in planned]
        for config,run in planned:
            out=Path(run['checkpoint_root'])
            local=root/'outputs/main_comparison/train_records'/run['run_id']/'status.json'
            existing_path=local if local.exists() else out/'status.json'
            if existing_path.exists():
                previous=json.loads(existing_path.read_text())
                if previous['state']=='complete' and (out/'29999/V03_COPY_COMPLETE.json').exists():
                    status['completed_runs'].append(run['run_id']);continue
                raise RuntimeError('Existing non-Spatial run requires inspected full-state resume: '+run['run_id'])
            status.update(stage='waiting_for_spare_gpu_and_memory',current_run=run['run_id']);write_json(path,status)
            # Two existing Spatial workers retain their assignments and GPU processes.
            # Include room for a full CPU snapshot and the existing RAM checkpoint stages.
            min_host_gib=420 if run['suite']=='libero_10' else 380
            release=root/'outputs/main_comparison/suite_release'/run['suite']
            smoke=root/'outputs/main_comparison/gpu_smoke_20261006/rulervla_main_attempt7/status.json'
            command=[sys.executable,str(root/'scripts/main_spatial_train.py'),'--run-config',str(config),
                '--smoke',str(smoke),'--cue-audit',str(release/'cue_audit/audit.json'),'--suite-release',str(release)]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),XLA_PYTHON_CLIENT_PREALLOCATE='false',
                OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',WANDB_MODE='disabled')
            # The lightweight scheduler may run CPU-only; its training child must
            # select the assigned CUDA device, not inherit that platform override.
            env.pop('JAX_PLATFORMS',None)
            log_path=logs/(run['run_id']+'.log')
            while True:
                # Serialize admission, so two workers cannot both consume the same
                # reported host headroom before either model has been initialized.
                try:fcntl.flock(admission,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    time.sleep(20);continue
                try:
                    child=try_launch(root,status,path,args,min_host_gib,command,env,log_path)
                finally:fcntl.flock(admission,fcntl.LOCK_UN)
                if child is not None:break
                time.sleep(20)
            code=child.wait()
            if code:raise RuntimeError('Training failed, inspect '+str(log_path))
            status['completed_runs'].append(run['run_id']);status['child_pid']=None;write_json(path,status)
        status.update(state='complete',stage='complete',current_run=None,finished_at_unix=time.time())
    except BaseException as error:
        status.update(state='failed',error=repr(error),finished_at_unix=time.time());raise
    finally:write_json(path,status)


def try_launch(root,status,path,args,min_host_gib,command,env,log_path):
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    values={int(line.split(',')[0]):[int(v) for v in line.split(',')[1:]] for line in rows.strip().splitlines()}
    memory={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
    owners_stable=True
    for worker in (0,1):
        owner=json.loads((root/'outputs/main_comparison'/('spatial_training_queue_worker%d_status.json'%worker)).read_text())
        if owner['state']=='complete':continue
        current=root/'outputs/main_comparison/train_records'/str(owner.get('current_run'))/'status.json'
        if owner['state']!='running' or not current.exists():owners_stable=False;continue
        live=json.loads(current.read_text())
        try:os.kill(live['pid'],0)
        except ProcessLookupError:owners_stable=False;continue
        if live['state']!='running' or live.get('optimizer_updates',0)<50:owners_stable=False
    peer_path=root/('outputs/main_comparison/remaining_suite_queue_worker%d_status.json'%(1-args.worker_index))
    peer_stable=True
    if peer_path.exists():
        peer=json.loads(peer_path.read_text())
        if peer.get('child_pid'):
            try:os.kill(peer['child_pid'],0)
            except ProcessLookupError:pass
            else:
                current=root/'outputs/main_comparison/train_records'/peer['current_run']/'status.json'
                if not current.exists():peer_stable=False
                else:
                    live=json.loads(current.read_text())
                    peer_stable=live['state']=='running' and live.get('optimizer_updates',0)>=50
    candidates=[args.gpu] if values[args.gpu][0]>=70000 and values[args.gpu][1]<=20 else []
    status.update(host_available_gib=round(memory['MemAvailable']/1024**2,1),required_host_gib=min_host_gib,
        gpu_free_mib={str(args.gpu):values[args.gpu][0]},eligible_gpus=candidates,
        spatial_owners_stable=owners_stable,peer_initialization_complete=peer_stable)
    write_json(path,status)
    if candidates and owners_stable and peer_stable and memory['MemAvailable']>=min_host_gib*1024**2:
        with log_path.open('a') as stream:
            child=subprocess.Popen(command,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
        status.update(stage='formal_training',child_pid=child.pid,log=str(log_path),formal_training_started=True)
        write_json(path,status)
        return child
    return None


if __name__=='__main__':main()
