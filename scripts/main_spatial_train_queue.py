"""Release Spatial's 12 formal runs after the completed common GPU smoke."""
import json
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from main_gpu_smoke import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',type=int,default=4)
    parser.add_argument('--worker-index',type=int,default=0)
    parser.add_argument('--worker-count',type=int,default=1)
    parser.add_argument('--min-host-memory-gib',type=int,default=180)
    args=parser.parse_args()
    assert 0<=args.worker_index<args.worker_count
    root=Path(__file__).resolve().parents[1]
    smoke_path=root/'outputs/main_comparison/gpu_smoke_20261006/rulervla_main_attempt7/status.json'
    audit=root/'outputs/main_comparison/cue_audit_20261006_with_wrist/audit.json'
    suffix='' if args.worker_count==1 else '_worker%d'%args.worker_index
    status_path=root/('outputs/main_comparison/spatial_training_queue%s_status.json'%suffix)
    logs=root/'outputs/main_comparison/logs';logs.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(logs/('spatial_training_queue%s.lock'%suffix)).open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=dict(state='running',stage='waiting_for_common_checkpoint_gate',pid=os.getpid(),completed_runs=[],
        current_run=None,child_pid=None,formal_training_started=False,started_at_unix=time.time())
    status.update(gpu=args.gpu,worker_index=args.worker_index,worker_count=args.worker_count)
    write_json(status_path,status)
    try:
        while True:
            smoke=json.loads(smoke_path.read_text())
            if smoke['state']=='failed':raise RuntimeError('Common GPU smoke failed: '+smoke.get('error',''))
            if smoke['state']=='complete':break
            os.kill(smoke['pid'],0)
            time.sleep(10)
        assert smoke['checkpoint_restored'] and smoke['full_train_state_restore_bitwise']
        assert json.loads(audit.read_text())['passed'] and json.loads((audit.parent/'visual_review.json').read_text())['passed']
        # Training shares one validated placement/microbatch configuration.
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(args.gpu),XLA_PYTHON_CLIENT_PREALLOCATE='false',
            OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',WANDB_MODE='disabled',
            NCCL_P2P_DISABLE='1',NCCL_IB_DISABLE='1',NCCL_SHM_DISABLE='1',NCCL_SOCKET_IFNAME='lo',NCCL_CUMEM_HOST_ENABLE='0')
        run_paths=sorted((root/'experiments/main_comparison_v1/runs').glob('libero_spatial_*.json'),
            key=lambda path:(json.loads(path.read_text())['seed'],
                ['Baseline','AimBot','AxisGuide-adapted','RulerVLA-main'].index(json.loads(path.read_text())['method'])))
        assert len(run_paths)==12
        run_paths=run_paths[args.worker_index::args.worker_count]
        status['assigned_runs']=[json.loads(path.read_text())['run_id'] for path in run_paths]
        for path in run_paths:
            run=json.loads(path.read_text());out=Path(run['checkpoint_root'])
            if (out/'status.json').exists():
                existing=json.loads((out/'status.json').read_text())
                if existing['state']=='complete' and (out/'29999/V03_COPY_COMPLETE.json').is_file():
                    status['completed_runs'].append(run['run_id']);continue
                raise RuntimeError('Existing incomplete run requires inspected full-state resume: '+run['run_id'])
            status.update(stage='waiting_for_gpu',current_run=run['run_id']);write_json(status_path,status)
            while True:
                query=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
                values={int(line.split(',')[0]):[int(v) for v in line.split(',')[1:]] for line in query.strip().splitlines()}
                memory={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
                enough_host=memory['MemAvailable']>=args.min_host_memory_gib*1024*1024
                status.update(gpu_free_mib=values[args.gpu][0],host_available_gib=round(memory['MemAvailable']/1024**2,1))
                write_json(status_path,status)
                if values[args.gpu][0]>=60000 and values[args.gpu][1]<=40 and enough_host:break
                time.sleep(10)
            command=[sys.executable,str(root/'scripts/main_spatial_train.py'),'--run-config',str(path),
                '--smoke',str(smoke_path),'--cue-audit',str(audit)]
            log_path=logs/(run['run_id']+'.log')
            with log_path.open('a') as stream:
                child=subprocess.Popen(command,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
                status.update(stage='formal_training',formal_training_started=True,child_pid=child.pid,log=str(log_path));write_json(status_path,status)
                code=child.wait()
            if code:raise RuntimeError('Training failed with code %d; inspect %s'%(code,log_path))
            status['completed_runs'].append(run['run_id']);status['child_pid']=None;write_json(status_path,status)
        status.update(state='complete',stage='complete',current_run=None,finished_at_unix=time.time())
    except BaseException as error:
        status.update(state='failed',error=repr(error),finished_at_unix=time.time());raise
    finally:write_json(status_path,status)


if __name__=='__main__':main()
