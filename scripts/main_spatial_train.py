"""30k Spatial full-FT run, released only after common smoke and cue gates."""
import argparse
import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from main_gpu_smoke import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-config', type=Path, required=True)
    parser.add_argument('--smoke', type=Path, required=True)
    parser.add_argument('--cue-audit', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run = json.loads(args.run_config.read_text())
    shared = run['shared_training']
    smoke = json.loads(args.smoke.read_text())
    cue = json.loads(args.cue_audit.read_text())
    review = json.loads((args.cue_audit.parent/'visual_review.json').read_text())
    assert run['suite'] == 'libero_spatial'
    assert smoke['state'] == 'complete' and smoke['forward_backward_passed']
    assert smoke['checkpoint_restored'] and smoke['full_train_state_restore_bitwise']
    assert smoke['optimizer_and_ema_on_cpu'] and smoke['effective_batch'] == 32
    assert cue['passed'] and review['passed']
    assert shared['finetuning'] == 'full' and not shared['lora'] and not shared['freeze_backbone']
    assert shared['optimizer_updates'] == 30000 and shared['effective_batch'] == 32
    assert shared['initial_checkpoint'] == smoke['initial_checkpoint']
    output = Path(run['checkpoint_root'])
    output.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock = (output/'.train.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    runtime = root/'outputs/main_comparison/runtime_v3'
    v03 = Path('/home/yqt/XuanyiLi/actioncoord/work/v03_sync_rgbd_20261003T164806')
    sys.path[:0] = [str(root/'src'), str(runtime/'src'), str(runtime/'packages/openpi-client/src'),
        str(v03/'repo/actioncoord_experiments/src')]
    import jax
    from openpi import transforms
    from openpi.models import model
    from openpi.models.pi0_config import Pi0Config
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.shared import normalize
    from openpi.training import config, optimizer, sharding, weight_loaders, checkpoints
    from rulervla.adapters.main_cues import MainCueAdapter
    from rulervla.adapters.main_comparison import MainComparisonInputs
    from rulervla.config import load_config
    from rulervla.training.accumulation import accumulated_gradients, apply_gradients
    from rulervla.training.main_data import SpatialDataset, update_indices, file_sha
    from actioncoord.integration.v03_checkpoint_snapshot import host_snapshot
    from actioncoord.integration.v03_storage import publish_checkpoint, verify_checkpoint
    for source, expected_sha in cue['official_sources_sha256'].items():
        assert file_sha(source) == expected_sha, 'Official cue renderer changed: ' + source
    runtime_manifest = json.loads((runtime/'RULERVLA_RUNTIME_MANIFEST.json').read_text())
    for relative, expected_sha in runtime_manifest['source_files_sha256'].items():
        actual_sha = runtime_manifest['changes'].get(relative, {}).get('snapshot_sha256', expected_sha)
        assert file_sha(runtime/relative) == actual_sha, 'Frozen runtime changed: ' + relative
    assert jax.device_count() == 1
    status = dict(state='running',stage='loading_data',pid=os.getpid(),run_id=run['run_id'],method=run['method'],
        suite=run['suite'],seed=run['seed'],gpu_ids=os.environ['CUDA_VISIBLE_DEVICES'],optimizer_updates=0,
        target_optimizer_updates=30000,micro_batch=1,gradient_accumulation=32,effective_batch=32,
        optimizer_and_ema_on_cpu=True,full_finetuning=True,lora=False,formal_training_started=True,
        initial_checkpoint=shared['initial_checkpoint'],started_at_unix=time.time(),checkpoint_root=str(output))
    status_path=output/'status.json'
    write_json(status_path,status)
    try:
        rgbd=Path('/home/Dataset2/yqt/openpi-aimbot/spatialcue_depth_source/regenerated_rgbd_v02')
        geometry=Path('/home/Dataset2/yqt/rulervla/main_comparison_v1/data_preparation/spatial_geometry')
        dataset=SpatialDataset(rgbd,geometry)
        norm_dir=args.smoke.parent.parent/'shared_spatial_norm'
        norm_stats=normalize.load(norm_dir)
        assert file_sha(norm_dir/'norm_stats.json')==smoke['norm_stats_sha256']
        policy_config=Pi0Config(pi05=True,action_horizon=10,discrete_state_input=False)
        sched=shared['scheduler'];opt=shared['optimizer']
        training_config=dataclasses.replace(config.get_config('pi05_libero'),model=policy_config,batch_size=32,
            seed=run['seed'],num_train_steps=30000,fsdp_devices=1,ema_decay=shared['ema_decay'],wandb_enabled=False,
            weight_loader=weight_loaders.CheckpointWeightLoader(shared['initial_checkpoint']),
            optimizer=optimizer.AdamW(b1=opt['beta1'],b2=opt['beta2'],eps=opt['eps'],weight_decay=opt['weight_decay'],
                clip_gradient_norm=opt['clip_gradient_norm']),lr_schedule=optimizer.CosineDecaySchedule(
                warmup_steps=sched['warmup_updates'],peak_lr=sched['peak_lr'],decay_steps=sched['decay_updates'],decay_lr=sched['end_lr']))
        cue_adapter=MainCueAdapter(load_config(root/'configs/rulervla_main.yaml'),run['method'],
            aimbot_source='/home/yqt/XuanyiLi/openpi0-aimbot/third_party/AimBot/src',
            axisguide_source='/home/yqt/XuanyiLi/AxisGuide-code/src/lerobot/datasets/visual_cue_utils.py')
        transform=transforms.compose([MainComparisonInputs(model_type=policy_config.model_type),
            transforms.Normalize(norm_stats,use_quantiles=True,strict=True),
            transforms.TokenizePrompt(PaligemmaTokenizer(policy_config.max_token_len),discrete_state_input=False),
            transforms.PadStatesAndActions(32)])
        spec=importlib.util.spec_from_file_location('main_stock_train',runtime/'scripts/train.py')
        train=importlib.util.module_from_spec(spec);spec.loader.exec_module(train)
        gpu_mesh=sharding.make_mesh(1)
        gpu=jax.sharding.NamedSharding(gpu_mesh,jax.sharding.PartitionSpec())
        cpu_mesh=jax.sharding.Mesh(np.array(jax.devices('cpu')).reshape(1,1),('batch','fsdp'))
        cpu=jax.sharding.NamedSharding(cpu_mesh,jax.sharding.PartitionSpec())
        init_rng=jax.device_put(jax.random.key(run['seed']),cpu)
        status['stage']='loading_initial_weights';write_json(status_path,status)
        saved_steps=sorted(int(path.name) for path in output.iterdir() if path.is_dir() and path.name.isdigit())
        if saved_steps and not args.resume:
            raise FileExistsError('Existing checkpoint requires explicit full-state resume')
        stage=Path('/dev/shm')/('rulervla_main_'+run['run_id']+'_'+str(os.getpid()))
        if args.resume:
            if not saved_steps:
                raise FileNotFoundError('No saved full state to resume')
            previous=json.loads((output/'execution_lock.json').read_text())
            assert previous['run']==run and previous['norm_stats_sha256']==smoke['norm_stats_sha256']
            latest=saved_steps[-1];verify_checkpoint(output/str(latest),expected_updates=latest+1)
            import shutil
            stage.mkdir(parents=True,exist_ok=False)
            shutil.copytree(output/str(latest),stage/str(latest))
            state,state_sharding=train.init_train_state(training_config,init_rng,cpu_mesh,resume=True)
            state=jax.tree.map(lambda shape,shard:jax.ShapeDtypeStruct(shape.shape,shape.dtype,sharding=shard),state,state_sharding)
            manager,_=checkpoints.initialize_checkpoint_dir(stage,keep_period=None,overwrite=False,resume=True)
        else:
            state,state_sharding=train.init_train_state(training_config,init_rng,cpu_mesh,resume=False)
            manager,_=checkpoints.initialize_checkpoint_dir(stage,keep_period=None,overwrite=False,resume=False)
        class Assets:
            def data_config(self):
                return config.DataConfig(asset_id='physical-intelligence/libero',norm_stats=norm_stats)
        if args.resume:
            state=checkpoints.restore_state(manager,state,Assets(),step=latest)
        assert not jax.tree.leaves(state.params.filter(training_config.freeze_filter))
        first_update=int(jax.device_get(state.step))
        graph=state.model_def
        param_sharding=sharding.fsdp_sharding(jax.eval_shape(lambda:state.params),gpu_mesh)
        gradient=jax.jit(lambda rng,step,params,batch:accumulated_gradients(training_config,rng,graph,params,step,batch),
            in_shardings=(gpu,gpu,param_sharding,gpu),out_shardings=(param_sharding,gpu))
        update=jax.jit(lambda current,grad:apply_gradients(training_config,current,grad),
            in_shardings=(state_sharding,state_sharding.params),out_shardings=state_sharding,donate_argnums=(0,1))
        execution=dict(run=run,common_smoke_sha256=file_sha(args.smoke),
            cue_audit_sha256=file_sha(args.cue_audit),visual_review=review,runtime_manifest_sha256=file_sha(runtime/'RULERVLA_RUNTIME_MANIFEST.json'),
            micro_batch=1,gradient_accumulation=32,effective_batch=32,optimizer_and_ema_on_cpu=True,
            episodes=432,frames=52970,sampler='uniform shuffled epochs, SeedSequence(training_seed,epoch)',
            norm_stats_sha256=file_sha(norm_dir/'norm_stats.json'),staging_path=str(stage))
        execution['git_commit'] = __import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
        execution['data_manifest_sha256'] = {'rgbd':file_sha(rgbd/'regen_manifest.json'),
            'geometry':file_sha(geometry/'manifest.json')}
        write_json(output/('resume_execution_%d.json'%os.getpid()) if args.resume else output/'execution_lock.json',execution)
        with (output/'metrics.jsonl').open('a',buffering=1) as log,sharding.set_mesh(gpu_mesh):
            for step in range(first_update,30000):
                start=time.time();indices=update_indices(len(dataset),run['seed'],step)
                status.update(stage='preparing_batch',current_optimizer_update=step+1);write_json(status_path,status)
                entries=[transform(cue_adapter.prepare(dataset[int(index)])) for index in indices]
                batch=jax.tree.map(lambda *values:np.stack(values)[:,None],*entries)
                batch=jax.device_put((model.Observation.from_dict(batch),batch['actions']),gpu)
                status['stage']='forward_backward';write_json(status_path,status)
                params=jax.device_put(state.params,param_sharding)
                grads,info=gradient(jax.random.key(run['seed']),jax.device_put(state.step,gpu),params,batch)
                grads_cpu=jax.device_put(grads,state_sharding.params);jax.block_until_ready(grads_cpu)
                metrics={key:float(value) for key,value in jax.device_get(info).items()}
                assert all(np.isfinite(value) for value in metrics.values()) and metrics['grad_norm']>0
                del grads,params
                state=update(state,grads_cpu);jax.block_until_ready(state)
                assert int(jax.device_get(state.step))==step+1
                record=dict(update=step+1,seconds=time.time()-start,sample_indices_sha256=__import__('hashlib').sha256(indices.tobytes()).hexdigest(),**metrics)
                log.write(json.dumps(record)+'\n');print(json.dumps(record),flush=True)
                status.update(stage='training',optimizer_updates=step+1,latest_metrics=record);write_json(status_path,status)
                # Keep all six complete checkpoints; no deletion from Dataset2.
                if (step+1)%5000==0:
                    status['stage']='saving_checkpoint';write_json(status_path,status)
                    # Serialize large CPU snapshots and RAM staging across independent GPU workers.
                    with (root/'outputs/main_comparison/checkpoint_staging.lock').open('w') as save_lock:
                        fcntl.flock(save_lock,fcntl.LOCK_EX)
                        snapshot,evidence=host_snapshot(state)
                        checkpoints.save_state(manager,snapshot,Assets(),step);manager.wait_until_finished()
                        destination=output/str(step)
                        publish_checkpoint(stage/str(step),destination,completed_updates=step+1)
                        verify_checkpoint(destination,expected_updates=step+1)
                        write_json(output/('snapshot_%d.json'%step),evidence)
                        del snapshot
                    status['latest_checkpoint']=str(destination);write_json(status_path,status)
        manager.close();status.update(state='complete',stage='complete',finished_at_unix=time.time(),final_checkpoint=str(output/'29999'))
    except BaseException as error:
        status.update(state='failed',error=repr(error),finished_at_unix=time.time());raise
    finally:
        write_json(status_path,status)


if __name__=='__main__':
    main()
