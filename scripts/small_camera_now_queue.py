"""User-authorized CameraXYZ GPU5 parallel smoke, then fresh-base 10k training."""
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
from main_gpu_smoke import write_json
from small_training_contract import validate_run
from small_training_queue import guarded_wait

root=Path(__file__).resolve().parents[1];base=root/'outputs/small_ablation_v1'
path=base/'camera_gpu5_now_status.json'
status=dict(state='running',stage='gpu_input_gate',pid=os.getpid(),gpu=5,formal_training_started=False)
write_json(path,status)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='5',XLA_PYTHON_CLIENT_PREALLOCATE='false',XLA_PYTHON_CLIENT_ALLOCATOR='platform',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',WANDB_MODE='disabled')
env.pop('JAX_PLATFORMS',None)
lock=(base/'logs/camera_gpu5_now.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
admission=(root/'outputs/main_comparison/logs/remaining_suite_admission.lock').open('w')
fcntl.flock(admission,fcntl.LOCK_EX)
held=True

def run_child(command,log):
    with log.open('a') as stream:
        child=subprocess.Popen(command,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
    status.update(child_pid=child.pid,log=str(log));write_json(path,status)
    return child

def camera_wait(child,record=None):
    global held
    while child.poll() is None:
        memory={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
        status['host_available_gib']=memory['MemAvailable']/1024**2
        if record and record.exists():
            d=json.loads(record.read_text());status['optimizer_updates']=d.get('optimizer_updates',0)
            if held and status['optimizer_updates']>=50:
                fcntl.flock(admission,fcntl.LOCK_UN);held=False
        write_json(path,status)
        if memory['MemAvailable']<48*1024**2:
            child.terminate()
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:child.kill();child.wait()
            raise RuntimeError('Stopped only Camera child to preserve host headroom')
        time.sleep(.25)
    return child.wait()

try:
    gate=base/'gpu_batch_gate'
    gate.rename(base/'gpu_batch_gate_before_camera_solo_20261008')
    child=run_child([sys.executable,str(root/'scripts/small_gpu_batch_gate.py')],base/'logs/camera_gpu5_now_input_gate.log')
    if camera_wait(child):raise RuntimeError('GPU batch gate failed')
    runs=json.loads((root/'experiments/small_ablation_v1/run_matrix.json').read_text())['runs']
    run=next(r for r in runs if r['method']=='CameraXYZ-only');validate_run(root,run)
    train_status=base/'train_records'/run['run_id']/'status.json'
    assert not train_status.exists(), 'Camera training already exists'
    config=base/'released_runs'/(run['run_id']+'.json');write_json(config,run)
    status['stage']='parallel_full_state_smoke';write_json(path,status)
    # Probe launcher owns the same admission lock: run it directly through the common smoke instead.
    smoke_out=base/'camera_gpu5_solo_smoke_20261008'
    command=[sys.executable,str(root/'scripts/small_gpu_smoke.py'),'--runtime',str(root/'outputs/main_comparison/runtime_v3'),
        '--v03-root','/home/yqt/XuanyiLi/actioncoord/work/v03_sync_rgbd_20261003T164806',
        '--rgbd-root','/home/Dataset2/yqt/openpi-aimbot/spatialcue_depth_source/regenerated_rgbd_v02',
        '--geometry-root','/home/Dataset2/yqt/rulervla/main_comparison_v1/data_preparation/spatial_geometry',
        '--out',str(smoke_out),'--method','CameraXYZ-only','--save-restore','--cpu-optimizer','--fsdp-devices','1',
        '--blocking-cpu-save','--stream-restore-check','--release-init-caches','--staging-root','/home/Dataset2/yqt/rulervla/small_ablation_v1/staging','--aimbot-source','/home/yqt/XuanyiLi/openpi0-aimbot/third_party/AimBot/src',
        '--axisguide-source','/home/yqt/XuanyiLi/AxisGuide-code/src/lerobot/datasets/visual_cue_utils.py']
    child=run_child(command,base/'logs/camera_gpu5_parallel_smoke_20261008.log')
    if camera_wait(child):raise RuntimeError('Camera concurrent full-state smoke failed')
    smoke=smoke_out/'status.json';result=json.loads(smoke.read_text())
    assert result['state']=='complete' and result['optimizer_updates']==2 and result['full_train_state_restore_bitwise']
    validate_run(root,run)
    status.update(stage='formal_training',formal_training_started=True,smoke_status=str(smoke))
    child=run_child([sys.executable,str(root/'scripts/small_spatial_train.py'),'--run-config',str(config),'--smoke',str(smoke),
        '--cue-audit',str(root/'outputs/main_comparison/cue_audit_20261006_with_wrist/audit.json'),
        '--blocking-cpu-save','--release-init-caches','--episode-cache-gib','8','--staging-root','/home/Dataset2/yqt/rulervla/small_ablation_v1/staging'],base/'logs'/(run['run_id']+'.log'))
    # Shared admission stays held until the first 50 real optimizer updates.
    if camera_wait(child,train_status):raise RuntimeError('Camera formal training failed')
    final=json.loads(train_status.read_text());assert final['state']=='complete' and final['optimizer_updates']==10000
    status.update(state='complete',stage='complete',final_checkpoint=final['final_checkpoint'])
except BaseException as error:
    status.update(state='failed',error=repr(error));raise
finally:
    write_json(path,status)
    if held:fcntl.flock(admission,fcntl.LOCK_UN)
