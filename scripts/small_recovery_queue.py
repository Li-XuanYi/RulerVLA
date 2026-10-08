"""Serial small experiments after the disk/GPU failures; no main-table training."""
import json, os, subprocess, sys, time
from pathlib import Path
from main_gpu_smoke import write_json
from small_training_contract import validate_run

root=Path(__file__).resolve().parents[1]
base=root/'outputs/small_ablation_v1'
path=base/'logs/small_recovery_queue_status.json'
status=dict(state='running',stage='waiting_resources',pid=os.getpid(),gpu=2,completed_runs=[])
env=dict(os.environ,CUDA_VISIBLE_DEVICES='2',XLA_PYTHON_CLIENT_PREALLOCATE='false',
         XLA_PYTHON_CLIENT_ALLOCATOR='platform',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
         PYTHONDONTWRITEBYTECODE='1',WANDB_MODE='disabled')
env.pop('JAX_PLATFORMS',None)
smokes={'CameraXYZ-only':base/'camera_gpu5_solo_smoke_20261008/status.json',
        'ActionCoord-only':base/'allocator_probe_actioncoord_attempt1/smoke/status.json'}
v03='/home/yqt/XuanyiLi/actioncoord/work/v03_sync_rgbd_20261003T164806'
sys.path.insert(0,v03+'/repo/actioncoord_experiments/src')
from actioncoord.integration.v03_storage import verify_checkpoint

def wait_resources():
    while True:
        rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free',
             '--format=csv,noheader,nounits'],text=True)
        frees={int(line.split(',')[0]):int(line.split(',')[1]) for line in rows.splitlines()}
        eligible=[gpu for gpu in (2,4,5) if frees[gpu]>=65000]
        mem={x.split(':')[0]:int(x.split()[1]) for x in Path('/proc/meminfo').read_text().splitlines()}
        status.update(stage='waiting_resources',gpu_free_mib={str(g):frees[g] for g in (2,4,5)},
                      host_available_gib=mem['MemAvailable']/1024**2)
        write_json(path,status)
        if eligible and mem['MemAvailable']>=220*1024**2:
            env['CUDA_VISIBLE_DEVICES']=str(eligible[0]);status['gpu']=eligible[0]
            return
        time.sleep(20)

try:
    matrix=json.loads((root/'experiments/small_ablation_v1/run_matrix.json').read_text())
    for method in ('CameraXYZ-only','ActionCoord-only','Ruler-only'):
        run=next(r for r in matrix['runs'] if r['method']==method and not r['conditional'])
        validate_run(root,run)
        out=Path(run['checkpoint_root']);record=base/'train_records'/run['run_id']
        status.update(current_run=run['run_id'],target_updates=run['shared_training']['optimizer_updates'])
        existing=json.loads((record/'status.json').read_text()) if (record/'status.json').exists() else None
        if existing and existing['state']=='complete':
            verify_checkpoint(out/str(run['shared_training']['optimizer_updates']-1),expected_updates=run['shared_training']['optimizer_updates'])
            status['completed_runs'].append(run['run_id']);continue
        saved=sorted(int(p.name) for p in out.glob('*') if p.name.isdigit())
        resume=bool(saved)
        if existing and not resume:
            assert not Path('/proc/%s'%existing['pid']).exists()
            evidence=base/'train_records/recovery_evidence_20261008';evidence.mkdir(exist_ok=True)
            record.rename(evidence/(run['run_id']+'_records_'+str(int(time.time()))))
            if out.exists():
                assert out.resolve().is_relative_to('/home/Dataset2/yqt/rulervla/small_ablation_v1/checkpoints')
                archive=Path('/home/Dataset2/yqt/rulervla/small_ablation_v1/interrupted_attempts')
                archive.mkdir(exist_ok=True)
                out.rename(archive/(run['run_id']+'_metadata_'+str(int(time.time()))))
        wait_resources()
        smoke=smokes.get(method)
        if smoke is None:
            smoke_out=(base/'logs').resolve().parent/'ruler_only_recovery_smoke_20261008';smoke=smoke_out/'status.json'
            if not smoke.exists():
                command=[sys.executable,str(root/'scripts/small_gpu_smoke.py'),'--runtime',str(base.parent/'main_comparison/runtime_v3'),
                    '--v03-root',v03,'--rgbd-root','/home/Dataset2/yqt/openpi-aimbot/spatialcue_depth_source/regenerated_rgbd_v02',
                    '--geometry-root','/home/Dataset2/yqt/rulervla/main_comparison_v1/data_preparation/spatial_geometry',
                    '--out',str(smoke_out),'--method',method,'--save-restore','--cpu-optimizer','--fsdp-devices','1',
                    '--blocking-cpu-save','--stream-restore-check','--release-init-caches',
                    '--staging-root','/home/Dataset2/yqt/rulervla/small_ablation_v1/staging',
                    '--aimbot-source','/home/yqt/XuanyiLi/openpi0-aimbot/third_party/AimBot/src',
                    '--axisguide-source','/home/yqt/XuanyiLi/AxisGuide-code/src/lerobot/datasets/visual_cue_utils.py']
                with (base/'logs/ruler_only_recovery_smoke.log').open('a') as log:
                    assert subprocess.call(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)==0
        s=json.loads(smoke.read_text());assert s['state']=='complete' and s['method']==method
        config=(base/'logs').resolve().parent/'released_runs'/(run['run_id']+'.json');write_json(config,run)
        command=[sys.executable,str(root/'scripts/small_spatial_train.py'),'--run-config',str(config),'--smoke',str(smoke),
            '--cue-audit',str(base.parent/'main_comparison/cue_audit_20261006_with_wrist/audit.json'),
            '--blocking-cpu-save','--episode-cache-gib','8','--resource-guard',
            '--staging-root','/home/Dataset2/yqt/rulervla/small_ablation_v1/staging']
        if s.get('initialization_caches_released'):command.append('--release-init-caches')
        if resume:command.append('--resume')
        with (base/'logs'/(run['run_id']+'.log')).open('a') as log:
            child=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        status.update(stage='training',child_pid=child.pid,smoke=str(smoke));write_json(path,status)
        assert child.wait()==0,'Training failed: '+run['run_id']
        verify_checkpoint(out/str(run['shared_training']['optimizer_updates']-1),expected_updates=run['shared_training']['optimizer_updates'])
        status['completed_runs'].append(run['run_id']);status['child_pid']=None
    status.update(state='complete',stage='complete')
except BaseException as error:
    status.update(state='failed',error=repr(error));raise
finally:write_json(path,status)
