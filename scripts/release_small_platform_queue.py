"""Release allocator placement only after real full-state smoke completes."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from main_gpu_smoke import write_json
from small_training_contract import validate_run, sha


def main():
    root=Path(__file__).resolve().parents[1];base=root/'outputs/small_ablation_v1'
    path=base/'platform_release_status.json'
    import fcntl
    lock=(base/'platform_release.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=dict(state='running',stage='waiting_for_full_state_smoke',pid=os.getpid(),started_at_unix=time.time())
    write_json(path,status)
    try:
        probe_path=base/'allocator_probe_actioncoord_attempt1/probe_status.json'
        while True:
            probe=json.loads(probe_path.read_text())
            if probe['state']=='complete':break
            if probe['state']!='running':raise RuntimeError('Allocator smoke did not pass: '+probe['state'])
            os.kill(probe['pid'],0)
            time.sleep(5)
        smoke_path=base/'allocator_probe_actioncoord_attempt1/smoke/status.json'
        smoke=json.loads(smoke_path.read_text())
        assert smoke['state']=='complete' and smoke['optimizer_updates']==2
        assert smoke['checkpoint_archive_sha_verified'] and smoke['full_train_state_restore_bitwise']
        assert smoke['blocking_cpu_snapshot_unchanged_after_save']
        assert smoke['official_weights_loaded'] and smoke['method']=='ActionCoord-only'
        assert smoke['effective_batch']==32 and smoke['parameter_count']==3353433872
        assert probe['allocator']=='platform' and probe['minimum_host_available_gib']>=32
        while True:
            gate=json.loads((base/'gpu_batch_gate/status.json').read_text())
            if gate['state']=='complete':break
            if gate['state']!='running':raise RuntimeError('GPU input gate failed')
            time.sleep(2)
        matrix=json.loads((root/'experiments/small_ablation_v1/run_matrix.json').read_text())
        for run in matrix['runs']:
            if not run['conditional']:validate_run(root,run)
        profile=dict(state='validated',allocator='platform',required_host_gib=220,required_gpu_free_mib=54369,
            reserve_host_gib=32,validated_smoke=str(smoke_path),smoke_sha256=sha(smoke_path),
            gpu_batch_gate_sha256=sha(base/'gpu_batch_gate/status.json'),probe=probe,
            placement_only='Same single GPU, model, dtype, seed, effective batch, scheduler and optimizer; blocking CPU save',
            released_at_unix=time.time())
        write_json(base/'platform_memory_release.json',profile)
        status['stage']='restarting_idle_small_controllers';write_json(path,status)
        recovery=base/('recovery_platform_'+str(int(time.time())));recovery.mkdir()
        launches=[]
        env=dict(os.environ,JAX_PLATFORMS='cpu',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        for worker in (0,1):
            qpath=base/('training_queue_worker%d_status.json'%worker);queue=json.loads(qpath.read_text())
            assert queue['state']=='running' and queue['child_pid'] is None and not queue['formal_training_started']
            pid=queue['pid'];cmdline=Path('/proc/%d/cmdline'%pid).read_bytes()
            assert b'small_training_queue.py' in cmdline
            assert Path('/proc/%d/cwd'%pid).resolve()==root
            shutil.copy2(qpath,recovery/qpath.name);os.kill(pid,signal.SIGTERM)
            for _ in range(30):
                p=Path('/proc/%d/stat'%pid)
                if not p.exists() or p.read_text().split()[2]=='Z':break
                time.sleep(.2)
            else:raise RuntimeError('Old idle controller did not exit')
            log=base/'logs'/('worker%d.log'%worker)
            with log.open('a') as stream:
                child=subprocess.Popen([sys.executable,str(root/'scripts/small_training_queue.py'),
                    '--worker-index',str(worker)],cwd=root,env=env,stdin=subprocess.DEVNULL,
                    stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            launches.append(dict(worker=worker,pid=child.pid,session=child.pid,gpu=(4,5)[worker],log=str(log)))
        write_json(base/'training_queue_platform_launch.json',launches)
        status.update(state='complete',stage='released',launches=launches,finished_at_unix=time.time())
    except BaseException as error:
        status.update(state='failed',error=repr(error),finished_at_unix=time.time());raise
    finally:write_json(path,status)


if __name__=='__main__':main()
