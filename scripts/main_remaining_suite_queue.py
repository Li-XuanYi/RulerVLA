"""Run the remaining 36 non-Spatial models as spare GPU and host RAM permit."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from main_gpu_smoke import write_json


def main():
    root=Path(__file__).resolve().parents[1]
    logs=root/'outputs/main_comparison/logs';logs.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(logs/'remaining_suite_queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=root/'outputs/main_comparison/remaining_suite_queue_status.json'
    status=dict(state='running',stage='checking_release',pid=os.getpid(),completed_runs=[],current_run=None,
        child_pid=None,started_at_unix=time.time(),formal_training_started=False,max_additional_concurrent_runs=1)
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
            while True:
                rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
                values={int(line.split(',')[0]):[int(v) for v in line.split(',')[1:]] for line in rows.strip().splitlines()}
                memory={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
                candidates=[gpu for gpu in (4,5,0,1) if values[gpu][0]>=70000 and values[gpu][1]<=20]
                status.update(host_available_gib=round(memory['MemAvailable']/1024**2,1),required_host_gib=min_host_gib,
                    gpu_free_mib={str(gpu):values[gpu][0] for gpu in (4,5,0,1)},eligible_gpus=candidates)
                write_json(path,status)
                if candidates and memory['MemAvailable']>=min_host_gib*1024**2:
                    gpu=max(candidates,key=lambda candidate:values[candidate][0]);break
                time.sleep(20)
            release=root/'outputs/main_comparison/suite_release'/run['suite']
            smoke=root/'outputs/main_comparison/gpu_smoke_20261006/rulervla_main_attempt7/status.json'
            command=[sys.executable,str(root/'scripts/main_spatial_train.py'),'--run-config',str(config),
                '--smoke',str(smoke),'--cue-audit',str(release/'cue_audit/audit.json'),'--suite-release',str(release)]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),XLA_PYTHON_CLIENT_PREALLOCATE='false',
                OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',WANDB_MODE='disabled')
            log_path=logs/(run['run_id']+'.log')
            with log_path.open('a') as stream:
                child=subprocess.Popen(command,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
                status.update(stage='formal_training',child_pid=child.pid,gpu=gpu,log=str(log_path),formal_training_started=True);write_json(path,status)
                code=child.wait()
            if code:raise RuntimeError('Training failed, inspect '+str(log_path))
            status['completed_runs'].append(run['run_id']);status['child_pid']=None;write_json(path,status)
        status.update(state='complete',stage='complete',current_run=None,finished_at_unix=time.time())
    except BaseException as error:
        status.update(state='failed',error=repr(error),finished_at_unix=time.time());raise
    finally:write_json(path,status)


if __name__=='__main__':main()
