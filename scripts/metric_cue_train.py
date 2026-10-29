import argparse
import gc
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    parser = argparse.ArgumentParser(description='六通道 Metric Cue 资源核验、完整模型检查与训练')
    parser.add_argument('--config', type=Path, default=ROOT/'configs/metric_workflow_lora.yaml')
    parser.add_argument('--mode', choices=('preflight', 'smoke', 'train'), default='preflight')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--arm', choices=('M0', 'M1', 'M2', 'M3'))
    parser.add_argument('--seed', type=int)
    parser.add_argument('--resume', action='store_true')
    gpu_selection = parser.add_mutually_exclusive_group()
    gpu_selection.add_argument('--gpu-indices', type=int, nargs='+')
    gpu_selection.add_argument('--auto-gpus', action='store_true')
    gpu_selection.add_argument('--gpu-candidates', type=int, nargs='+')
    parser.add_argument('--wait-for-resources', action='store_true')
    args = parser.parse_args()
    if args.wait_for_resources and (args.mode == 'preflight' or args.arm is None):
        raise ValueError('Resource waiting requires an explicit smoke or formal arm')
    if args.wait_for_resources and args.mode == 'smoke' and args.steps != 2:
        raise ValueError('LoRA smoke requires two updates plus checkpoint continuation')
    if args.gpu_candidates is not None and (len(set(args.gpu_candidates)) != len(args.gpu_candidates)
                                           or any(index < 0 for index in args.gpu_candidates)):
        raise ValueError('GPU candidates must be distinct nonnegative integers')
    from rulervla.training.metric_resources import load_run_config, preflight, query_gpus, runtime_capacity
    run = load_run_config(args.config)
    if args.gpu_indices is not None:
        if len(set(args.gpu_indices)) != len(args.gpu_indices) or any(index < 0 for index in args.gpu_indices):
            raise ValueError('GPU indices must be distinct nonnegative integers')
        run['resources']['gpu_indices'] = args.gpu_indices
    protocol = json.loads((ROOT/'experiments/metric_cue_v1/protocol.json').read_text())
    from rulervla.training.metric_protocol import require_lora_recipe
    require_lora_recipe(run, protocol)
    if args.mode == 'train':
        if args.arm is None or args.seed is None or args.steps is not None:
            raise ValueError('Formal training requires --arm and --seed and uses the fixed update budget')
        from rulervla.training.metric_protocol import load_formal_metric_run
        run, protocol = load_formal_metric_run(ROOT, args.arm, args.seed, run)
    elif args.arm is not None:
        from rulervla.training.metric_protocol import require_arm_ruler
        protocol = json.loads((ROOT/'experiments/metric_cue_v1/protocol.json').read_text())
        require_arm_ruler(ROOT, args.arm, run, protocol)
        run['cue_config'] = str(ROOT/protocol['arms'][args.arm])
    if args.resume and args.mode != 'train':
        raise ValueError('Only formal training can resume a full checkpoint')
    if args.mode == 'train' and args.out.name != run['formal_run_id']:
        raise ValueError('Formal output directory must be named ' + run['formal_run_id'])
    args.out.mkdir(parents=True, exist_ok=args.resume)
    if args.resume:
        previous_status = (args.out/'status.json').read_text()
        (args.out/f'status_before_resume_{time.time_ns()}.json').write_text(previous_status)
    attempt = 0
    while True:
        if args.mode == 'train':
            run, protocol = load_formal_metric_run(
                ROOT, args.arm, args.seed, load_run_config(args.config))
            if args.gpu_indices is not None:
                run['resources']['gpu_indices'] = args.gpu_indices
        if args.auto_gpus or args.gpu_candidates is not None:
            gpus = query_gpus()
            if isinstance(gpus, dict):
                raise RuntimeError('GPU selection failed: ' + gpus['error'])
            eligible = [gpu for gpu in gpus if args.gpu_candidates is None
                        or gpu['index'] in args.gpu_candidates]
            selected = sorted(eligible, key=lambda gpu: gpu['free_mib'], reverse=True)[:2]
            if len(selected) != 2:
                raise ValueError('LoRA requires two available GPUs')
            run['resources']['gpu_indices'] = [gpu['index'] for gpu in selected]
        admission = preflight(run)
        attempt += 1
        (args.out/'preflight.json').write_text(json.dumps(admission, indent=2))
        (args.out/'workflow_config.json').write_text(json.dumps(run, indent=2))
        if not args.wait_for_resources or admission['launch_allowed'] or admission['input_blockers']:
            break
        if 'jax' in sys.modules:
            raise RuntimeError('Resource waiting must not initialize JAX')
        waiting = dict(state='waiting_resources', mode=args.mode, attempt=attempt,
                       pid=os.getpid(), gpu_job_started=False, formal_training_started=False,
                       jax_imported=False, blockers=admission['resource_blockers'],
                       selected_gpu_indices=run['resources']['gpu_indices'])
        (args.out/f'preflight_wait_{attempt:04d}.json').write_text(json.dumps(admission, indent=2))
        (args.out/'status.json').write_text(json.dumps(waiting, indent=2))
        print(json.dumps(waiting), flush=True)
        time.sleep(60)
    if args.mode == 'train' and admission['launch_allowed']:
        verified_run, verified_protocol = load_formal_metric_run(
            ROOT, args.arm, args.seed, load_run_config(args.config))
        verified_run['resources']['gpu_indices'] = run['resources']['gpu_indices']
        if verified_run != run or verified_protocol != protocol:
            raise ValueError('Formal recipe changed during input/resource admission')
    if args.mode == 'preflight' or not admission['launch_allowed']:
        result = dict(state='ready' if admission['launch_allowed'] else 'blocked',
                      mode=args.mode, gpu_job_started=False, formal_training_started=False,
                      jax_imported='jax' in sys.modules,
                      blockers=admission['blockers'])
        (args.out/'status.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
        return 0 if args.mode == 'preflight' else 2
    if not run['resources']['gpu_indices'] or not run['resources']['cpu_optimizer']:
        raise ValueError('Training requires GPU gradients and a CPU AdamW optimizer')
    steps = args.steps if args.steps is not None else run['training']['num_steps']
    if steps < 1 or (args.mode == 'smoke' and steps != 2):
        raise ValueError('LoRA smoke requires two updates plus one checkpoint continuation')
    if args.mode == 'smoke' and args.arm is None:
        raise ValueError('LoRA smoke requires an explicit comparison arm')
    if args.mode != 'train' and args.seed is not None:
        raise ValueError('--seed is reserved for formal runs; smoke uses the shared configured seed')
    run['training']['num_steps'] = steps
    (args.out/'workflow_config.json').write_text(json.dumps(run, indent=2))
    memory_sampler = None
    gpu_context_started = False
    gpu_initialization_attempted = False
    def record_failure(error_type, error, traceback):
        sys.__excepthook__(error_type, error, traceback)
        path = args.out/'status.json'
        failed = json.loads(path.read_text()) if path.exists() else {}
        failed.pop('blockers', None)
        failed.pop('attempt', None)
        failed.update(state='failed', error_type=error_type.__name__, error=str(error),
                      pid=os.getpid(), jax_imported='jax' in sys.modules,
                      selected_gpu_indices=run['resources']['gpu_indices'],
                      mode=args.mode, gpu_job_started=gpu_context_started,
                      gpu_initialization_attempted=gpu_initialization_attempted,
                      formal_training_started=args.mode == 'train' and gpu_context_started,
                      peak_cpu_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)
        if memory_sampler is not None:
            failed.update(memory_sampler.finish())
        if gpu_context_started:
            failed['gpu_allocator_stats'] = [device.memory_stats() for device in gpu_devices]
        path.write_text(json.dumps(failed, indent=2, allow_nan=False))
    sys.excepthook = record_failure
    from rulervla.training.metric_resources import configure_lora_gpu_allocator
    allocation_contract = configure_lora_gpu_allocator(run, admission['gpu_snapshot'])
    os.environ['JAX_PLATFORMS'] = 'cuda,cpu'
    cache_dir = ROOT/'build/metric_lora_cache/jax_compilation_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ['JAX_COMPILATION_CACHE_DIR'] = str(cache_dir)
    if args.mode == 'smoke':
        from rulervla.training.metric_resources import SmokeGpuMemory
        memory_sampler = SmokeGpuMemory(os.getpid(), admission['gpu_snapshot'],
                                         run['resources']['gpu_indices'])
    import shutil
    if cache_dir.stat().st_dev != ROOT.stat().st_dev or shutil.disk_usage(cache_dir).free < 8 << 30:
        raise ValueError('LoRA compilation cache requires 8 GiB on the local runtime filesystem')
    import jax
    gpu_initialization_attempted = True
    gpu_devices = jax.devices('gpu')
    gpu_context_started = True
    if len(gpu_devices) != len(run['resources']['gpu_indices']):
        raise ValueError('JAX did not expose every selected GPU')
    from rulervla.training.metric_gpu_allocator import reserve_lora_gpu_pools
    allocation_contract.update(reserve_lora_gpu_pools(
        gpu_devices, allocation_contract['pool_target_mib_per_device']))
    gc.collect()
    (args.out/'gpu_allocator.json').write_text(json.dumps(allocation_contract, indent=2))
    initial_status = dict(state='running', stage='gpu_pool_reserved', pid=os.getpid(),
        jax_imported=True, gpu_initialization_attempted=True, gpu_job_started=True,
        selected_gpu_indices=run['resources']['gpu_indices'], mode=args.mode,
        formal_training_started=args.mode == 'train',
        optimizer_updates=None if args.resume else 0)
    (args.out/'status.json').write_text(json.dumps(initial_status, indent=2))
    print(json.dumps(dict(stage='gpu_pool_reserved', **allocation_contract)), flush=True)
    gpu, cpu = gpu_devices[0], jax.devices('cpu')[0]
    from openpi.training import checkpoints, sharding
    from rulervla.adapters.metric_pipeline import MetricPipeline
    from rulervla.training.main_data import SpatialDataset, update_indices
    from rulervla.training.metric_workflow import (MetricTrainer, abstract_training_state,
        create_training_state, gpu_equivalence, frozen_signatures, flat_params, load_metric_model,
        gpu_prediction,
        load_initial_model, make_training_config, prepare_batch, restore_training_checkpoint,
        save_training_checkpoint, source_provenance, state_signatures, training_state_template)
    pipeline = MetricPipeline.from_files(run['cue_config'], run['norm_stats_dir'])
    dataset = SpatialDataset(run['data']['rgbd_root'], run['data']['geometry_root'],
                             max_cache_bytes=run['data']['max_cache_bytes'], path_mappings=run['path_mappings'])
    config = make_training_config(run)
    provenance = source_provenance(ROOT)
    provenance.update(command=sys.argv, run_config=run,
                      pipeline_sha256=pipeline.fingerprint(), dataset_manifests=dataset.manifest_hashes,
                      checkpoint_receipt=admission['checkpoint_receipt'], resource_admission=admission,
                      gpu_allocator=allocation_contract)
    manifest_path = args.out/'run_manifest.json'
    if args.resume:
        previous = json.loads(manifest_path.read_text())
        for key in ('run_config', 'pipeline_sha256', 'dataset_manifests', 'loaded_source_sha256',
                    'dependency_lock_sha256', 'git_commit'):
            if previous[key] != provenance[key]:
                raise ValueError('Full-state resume identity differs: ' + key)
        provenance = previous
        status = json.loads((args.out/'status.json').read_text())
        status.update(state='running', gpu_job_started=True, formal_training_started=True,
                      gpu_initialization_attempted=True, jax_imported=True, pid=os.getpid(),
                      selected_gpu_indices=run['resources']['gpu_indices'], mode=args.mode)
        for key in ('error', 'error_type', 'last_metrics', 'blockers', 'attempt'):
            status.pop(key, None)
    else:
        manifest_path.write_text(json.dumps(provenance, indent=2))
        status = dict(state='running', gpu_job_started=True, gpu_initialization_attempted=True,
                      jax_imported=True, pid=os.getpid(), selected_gpu_indices=run['resources']['gpu_indices'],
                      formal_training_started=args.mode == 'train',
                      mode=args.mode, optimizer_updates=0, metrics=[] if args.mode == 'smoke' else None)
    def stage(name, **values):
        status.update(stage=name, peak_cpu_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
                      **values)
        (args.out/'status.json').write_text(json.dumps(status, indent=2, allow_nan=False))
        print(json.dumps({'stage':name, **values}), flush=True)
    def require_current_capacity(phase):
        measured = runtime_capacity(run)
        status.setdefault('resource_checks', []).append(dict(phase=phase, **measured))
        if measured['blockers']:
            stage('resource_gate_closed', state='blocked', blockers=measured['blockers'],
                  resources=measured)
            raise RuntimeError('; '.join(measured['blockers']))
    mesh = (sharding.make_mesh(len(run['resources']['gpu_indices']))
            if len(run['resources']['gpu_indices']) > 1 else None)
    # 用户要求：保留启动前准入；启动后不因资源查询或阈值主动终止训练。
    # require_current_capacity('before_state_loading')
    stage('restore_state_template' if args.resume else 'load_real_base_and_six_channel_optimizer')
    with jax.default_device(cpu):
        if args.resume:
            state = training_state_template(config, cpu)
        else:
            state = create_training_state(config, load_initial_model(config), optimizer_device=cpu)
    gc.collect()
    # require_current_capacity('model loading')
    manager, resuming = checkpoints.initialize_checkpoint_dir(
        args.out/'checkpoints', keep_period=10000 if args.mode == 'train' else 1,
        overwrite=False, resume=args.resume)
    if args.resume:
        if not resuming:
            raise FileNotFoundError('No complete full-state checkpoint to resume')
        latest = max(manager.all_steps())
        state = restore_training_checkpoint(manager, state, pipeline, config, step=latest,
                                            expected_manifest=provenance)
        if int(state.step) != latest:
            raise ValueError('Restored optimizer step differs from checkpoint directory')
        stage('restored_training_state', optimizer_updates=int(state.step), resumed_checkpoint=latest)
    first_batch = None
    initial_frozen = None
    if args.mode == 'smoke':
        import flax.nnx as nnx
        import numpy as np
        initial_frozen = frozen_signatures(state, config)
        initial_selected = state.params.filter(config.trainable_filter)
        for name, value in flat_params(initial_selected).items():
            if name.endswith('lora_b') and np.count_nonzero(np.asarray(value)):
                raise ValueError('LoRA B initialization is not zero: ' + name)
        if np.count_nonzero(np.asarray(flat_params(state.params)['PaliGemma/img/embedding/kernel'])[..., 3:, :]):
            raise ValueError('New Cue stem is not zero initialized')
        (args.out/'initial_trainable_sha256.json').write_text(
            json.dumps(state_signatures(initial_selected), indent=2))
        status['trainable_parameters'] = sum(value.size for value in jax.tree.leaves(initial_selected))
        indices = update_indices(len(dataset), config.seed, 0, batch_size=config.batch_size)
        first_batch = prepare_batch(dataset, indices, pipeline)
        stage('gpu_zero_cue_equivalence')
        measured = gpu_equivalence(config, state.params, first_batch, gpu,
                                   run['gpu_equivalence'], mesh=mesh)
        (args.out/'gpu_equivalence.json').write_text(json.dumps(measured, indent=2))
        # require_current_capacity('GPU equivalence')
    trainer = MetricTrainer(config, state.model_def, gradient_device=gpu,
                            optimizer_device=cpu, gradient_mesh=mesh)
    saved = None
    for update in range(int(state.step), steps):
        indices = update_indices(len(dataset), config.seed, update, batch_size=config.batch_size)
        batch = first_batch if update == 0 and first_batch is not None else prepare_batch(dataset, indices, pipeline)
        started = time.monotonic()
        micro_batch_size = protocol['training']['micro_batch_size']
        # require_current_capacity(f'before_update_{update + 1}')
        state, metrics = trainer.step(state, batch, micro_batch_size=micro_batch_size)
        # require_current_capacity(f'after_update_{int(state.step)}')
        if args.mode == 'smoke':
            if metrics['lora_b_grad_l2'] <= 0:
                raise ValueError('Real action loss did not produce nonzero LoRA B gradients')
            if args.arm == 'M0' and metrics['cue_grad_l2'] != 0:
                raise ValueError('Blank M0 Cue produced nonzero Cue kernel gradients')
            if args.arm != 'M0' and metrics['cue_grad_l2'] <= 0:
                raise ValueError('Real action loss did not produce nonzero Cue kernel gradients')
        record = dict(update=int(state.step), seconds=time.monotonic()-started,
                      sample_identities=batch.identities, **metrics)
        if args.mode == 'smoke':
            status['metrics'].append(record)
        else:
            with (args.out/'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(record, allow_nan=False) + '\n')
            status['last_metrics'] = record
        stage('training_updates', optimizer_updates=int(state.step))
        if args.mode == 'train' and int(state.step) in protocol['training']['checkpoint_updates']:
            # require_current_capacity('before_checkpoint_save')
            stage('save_full_training_state')
            saved = save_training_checkpoint(manager, state, pipeline, config, provenance)
    if args.mode == 'smoke':
        # require_current_capacity('before_checkpoint_save')
        stage('save_full_training_state')
        expected = state_signatures(state)
        (args.out/'full_state_expected_sha256.json').write_text(json.dumps(expected, indent=2))
        completed_step = int(state.step)
        saved = save_training_checkpoint(manager, state, pipeline, config, provenance)
        template = abstract_training_state(state, cpu)
        del state
        gc.collect()
        stage('restore_full_training_state')
        state = restore_training_checkpoint(manager, template, pipeline, config, step=completed_step,
                                            expected_manifest=provenance)
        if state_signatures(state) != expected:
            raise ValueError('Full six-channel parameter/optimizer state changed during recovery')
        # require_current_capacity('after_checkpoint_restore')
        (args.out/'full_state_sha256.json').write_text(json.dumps(expected, indent=2))
        if frozen_signatures(state, config) != initial_frozen:
            raise ValueError('Frozen backbone changed during real LoRA updates')
        with jax.default_device(cpu):
            expected_inference = nnx.State(state.params)
            expected_inference.replace_by_pure_dict(state.ema_params.to_pure_dict())
            expected_inference_sha256 = state_signatures(expected_inference)
        del expected_inference, state, initial_selected
        gc.collect()
        jax.clear_caches()
        stage('load_delta_inference_model')
        with jax.default_device(cpu):
            restored_model, restored_pipeline = load_metric_model(saved)
            inferred = nnx.state(restored_model)
        if (state_signatures(inferred) != expected_inference_sha256
                or restored_pipeline.fingerprint() != pipeline.fingerprint()):
            raise ValueError('Delta inference reconstruction differs from the full EMA model')
        measured = gpu_prediction(config, inferred, first_batch, gpu, mesh=mesh)
        (args.out/'delta_inference.json').write_text(json.dumps(measured, indent=2))
        del inferred, restored_model
        gc.collect()
        jax.clear_caches()
        stage('restore_optimizer_after_inference')
        state = restore_training_checkpoint(manager, template, pipeline, config, step=completed_step,
                                            expected_manifest=provenance)
        if state_signatures(state) != expected:
            raise ValueError('Full training state changed during inference verification')
        stage('continue_restored_optimizer')
        indices = update_indices(len(dataset), config.seed, int(state.step), batch_size=config.batch_size)
        batch = prepare_batch(dataset, indices, pipeline)
        started = time.monotonic()
        state, metrics = trainer.step(state, batch, micro_batch_size=1)
        status['metrics'].append(dict(update=int(state.step), seconds=time.monotonic()-started,
                                      sample_identities=batch.identities, **metrics))
        if frozen_signatures(state, config) != initial_frozen:
            raise ValueError('Frozen backbone changed after checkpoint continuation')
        status.update(optimizer_updates=int(state.step), frozen_backbone_unchanged=True,
                      delta_inference_verified=True, restored_optimizer_continued=True,
                      gpu_memory_stats=[device.memory_stats() for device in jax.devices('gpu')])
        saved = save_training_checkpoint(manager, state, pipeline, config, provenance)
    manager.close()
    if memory_sampler is not None:
        status.update(memory_sampler.finish())
    stage('complete', state='complete', full_state_restore_bitwise=args.mode == 'smoke',
          checkpoint=str(saved if saved is not None else args.out/'checkpoints'/str(int(state.step))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
