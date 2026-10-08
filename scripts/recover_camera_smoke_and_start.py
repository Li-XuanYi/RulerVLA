"""Recover a completed two-update checkpoint after transient NAS copy failure, then start Camera."""
import dataclasses,fcntl,importlib.util,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
from main_gpu_smoke import write_json
from small_training_contract import validate_run

root=Path(__file__).resolve().parents[1];base=root/'outputs/small_ablation_v1'
path=base/'camera_gpu5_now_status.json'
status=dict(state='running',stage='recovering_saved_smoke',pid=os.getpid(),gpu=5,formal_training_started=False)
write_json(path,status)
admission=(root/'outputs/main_comparison/logs/remaining_suite_admission.lock').open('w');fcntl.flock(admission,fcntl.LOCK_EX)
held=True
try:
    runtime=root/'outputs/main_comparison/runtime_v3'
    v03=Path('/home/yqt/XuanyiLi/actioncoord/work/v03_sync_rgbd_20261003T164806')
    sys.path[:0]=[str(root/'src'),str(runtime/'src'),str(runtime/'packages/openpi-client/src'),str(v03/'repo/actioncoord_experiments/src')]
    import jax
    from openpi.models.pi0_config import Pi0Config
    from openpi.training import config,optimizer,weight_loaders,checkpoints
    from openpi.shared import normalize
    from rulervla.training.blocking_snapshot import verify_blocking_snapshot
    from rulervla.training.storage_io import retry_io,publish_with_retry
    from actioncoord.integration.v03_storage import publish_checkpoint,verify_checkpoint
    smoke_out=base/'camera_gpu5_solo_smoke_20261008';smoke_path=smoke_out/'status.json'
    smoke=json.loads(smoke_path.read_text());assert smoke['optimizer_updates']==2 and smoke['official_weights_loaded']
    assert smoke['method']=='CameraXYZ-only' and smoke['effective_batch']==32 and smoke['initialization_caches_released']
    evidence=json.loads((smoke_out/'snapshot.json').read_text())
    source=Path('/home/Dataset2/yqt/rulervla/small_ablation_v1/staging/rulervla_main_smoke_922414')
    assert (source/'1/_CHECKPOINT_METADATA').exists()
    training_config=dataclasses.replace(config.get_config('pi05_libero'),model=Pi0Config(pi05=True,action_horizon=10,discrete_state_input=False),
        batch_size=32,seed=42,num_train_steps=30000,fsdp_devices=1,ema_decay=.999,wandb_enabled=False,
        optimizer=optimizer.AdamW(b1=.9,b2=.95,eps=1e-8,weight_decay=1e-10,clip_gradient_norm=1),
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=1000,peak_lr=5e-5,decay_steps=30000,decay_lr=5e-6),
        weight_loader=weight_loaders.CheckpointWeightLoader(smoke['initial_checkpoint']))
    norm=normalize.load(root/'outputs/main_comparison/gpu_smoke_20261006/shared_spatial_norm')
    class Assets:
        def data_config(self):return config.DataConfig(asset_id='physical-intelligence/libero',norm_stats=norm)
    spec=importlib.util.spec_from_file_location('restore_stock_train',runtime/'scripts/train.py')
    train=importlib.util.module_from_spec(spec);spec.loader.exec_module(train)
    mesh=jax.sharding.Mesh(np.array(jax.devices('cpu')).reshape(1,1),('batch','fsdp'))
    cpu=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec())
    shapes,shards=train.init_train_state(training_config,jax.device_put(jax.random.key(42),cpu),mesh,resume=True)
    shapes=jax.tree.map(lambda shape,shard:jax.ShapeDtypeStruct(shape.shape,shape.dtype,sharding=shard),shapes,shards)
    manager,_=checkpoints.initialize_checkpoint_dir(source,overwrite=False,resume=True,keep_period=None)
    status['stage']='restoring_full_state';write_json(path,status)
    restored=retry_io(lambda:checkpoints.restore_state(manager,shapes,Assets(),step=1))
    jax.block_until_ready(restored);assert int(jax.device_get(restored.step))==2
    verify_blocking_snapshot(restored,evidence)
    manager.close();del restored,manager
    import gc,ctypes
    gc.collect();jax.clear_caches();ctypes.CDLL(None).malloc_trim(0)
    status['stage']='archiving_verified_smoke';write_json(path,status)
    archive=Path('/home/Dataset2/yqt/rulervla/small_ablation_v1/smoke_checkpoints')/smoke_out.name/'1'
    publish_with_retry(source/'1',archive,2,publish_checkpoint,verify_checkpoint)
    retry_io(lambda:verify_checkpoint(archive,expected_updates=2))
    old=smoke_out/'status_before_io_recovery.json';assert not old.exists();old.write_text(json.dumps(smoke,indent=2))
    smoke.update(state='complete',stage='smoke_complete',forward_backward_passed=True,effective_batch_32_verified=True,
        checkpoint_saved=True,checkpoint=str(archive),checkpoint_archive_sha_verified=True,checkpoint_restored=True,
        full_train_state_restore_bitwise=True,blocking_cpu_snapshot_unchanged_after_save=True,
        restore_check='every restored leaf shape/dtype and SHA256 matches original post-update snapshot',
        recovered_after_transient_copy_failure=True,finished_at_unix=time.time())
    smoke.pop('error',None);write_json(smoke_path,smoke)
    run=next(r for r in json.loads((root/'experiments/small_ablation_v1/run_matrix.json').read_text())['runs'] if r['method']=='CameraXYZ-only')
    validate_run(root,run)
    record=base/'train_records'/run['run_id']/'status.json';assert not record.exists()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='5',XLA_PYTHON_CLIENT_PREALLOCATE='false',XLA_PYTHON_CLIENT_ALLOCATOR='platform',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',WANDB_MODE='disabled')
    env.pop('JAX_PLATFORMS',None)
    command=[sys.executable,str(root/'scripts/small_spatial_train.py'),'--run-config',str(base/'released_runs'/(run['run_id']+'.json')),
        '--smoke',str(smoke_path),'--cue-audit',str(root/'outputs/main_comparison/cue_audit_20261006_with_wrist/audit.json'),
        '--blocking-cpu-save','--release-init-caches','--episode-cache-gib','8','--staging-root','/home/Dataset2/yqt/rulervla/small_ablation_v1/staging']
    log=base/'logs'/(run['run_id']+'.log')
    with log.open('a') as f:child=subprocess.Popen(command,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
    status.update(stage='formal_training',child_pid=child.pid,log=str(log),formal_training_started=True,smoke_status=str(smoke_path));write_json(path,status)
    while child.poll() is None:
        memory={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
        status['host_available_gib']=memory['MemAvailable']/1024**2
        if record.exists():
            current=json.loads(record.read_text());status['optimizer_updates']=current.get('optimizer_updates',0)
            if held and status['optimizer_updates']>=50:fcntl.flock(admission,fcntl.LOCK_UN);held=False
        write_json(path,status)
        if memory['MemAvailable']<48*1024**2:
            child.terminate()
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:child.kill();child.wait()
            raise RuntimeError('Stopped Camera child to preserve host headroom')
        time.sleep(.5)
    assert child.wait()==0
    final=json.loads(record.read_text());assert final['state']=='complete' and final['optimizer_updates']==10000
    status.update(state='complete',stage='complete',final_checkpoint=final['final_checkpoint'])
except BaseException as error:
    status.update(state='failed',error=repr(error));raise
finally:
    write_json(path,status)
    if held:fcntl.flock(admission,fcntl.LOCK_UN)
