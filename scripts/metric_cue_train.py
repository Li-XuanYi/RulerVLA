import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    parser = argparse.ArgumentParser(description='六通道 Metric Cue 资源核验、完整模型检查与训练')
    parser.add_argument('--config', type=Path, default=ROOT/'configs/rulervla_metric_workflow.yaml')
    parser.add_argument('--mode', choices=('preflight', 'smoke', 'train'), default='preflight')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--arm', choices=('M0', 'M1', 'M2', 'M3'))
    parser.add_argument('--seed', type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--gpu-indices', type=int, nargs='+')
    args = parser.parse_args()
    from rulervla.training.metric_resources import load_run_config, preflight, query_gpus
    run = load_run_config(args.config)
    if args.gpu_indices is not None:
        if len(set(args.gpu_indices)) != len(args.gpu_indices) or any(index < 0 for index in args.gpu_indices):
            raise ValueError('GPU indices must be distinct nonnegative integers')
        run['resources']['gpu_indices'] = args.gpu_indices
    protocol = None
    if args.mode == 'train':
        if args.arm is None or args.seed is None or args.steps is not None:
            raise ValueError('Formal training requires --arm and --seed and uses the fixed update budget')
        from rulervla.training.metric_protocol import load_formal_metric_run
        run, protocol = load_formal_metric_run(ROOT, args.arm, args.seed, run)
    elif args.arm is not None:
        protocol = json.loads((ROOT/'experiments/metric_cue_v1/protocol.json').read_text())
        run['cue_config'] = str(ROOT/protocol['arms'][args.arm])
    if args.resume and args.mode != 'train':
        raise ValueError('Only formal training can resume a full checkpoint')
    if args.mode == 'train' and args.out.name != run['formal_run_id']:
        raise ValueError('Formal output directory must be named ' + run['formal_run_id'])
    args.out.mkdir(parents=True, exist_ok=args.resume)
    admission = preflight(run)
    (args.out/'preflight.json').write_text(json.dumps(admission, indent=2))
    (args.out/'workflow_config.json').write_text(json.dumps(run, indent=2))
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
    if steps < 1 or (args.mode == 'smoke' and steps > 3):
        raise ValueError('Smoke requires 1..3 updates; larger runs need explicit --mode train')
    run['training']['num_steps'] = steps
    (args.out/'workflow_config.json').write_text(json.dumps(run, indent=2))
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, run['resources']['gpu_indices']))
    os.environ['JAX_PLATFORMS'] = 'cuda,cpu'
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
    cache_dir = ROOT/'outputs/metric_cue_pilot/jax_compilation_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ['JAX_COMPILATION_CACHE_DIR'] = str(cache_dir)
    import jax
    from openpi.training import checkpoints, sharding
    from rulervla.adapters.metric_pipeline import MetricPipeline
    from rulervla.training.main_data import SpatialDataset, update_indices
    from rulervla.training.metric_workflow import (MetricTrainer, create_training_state, gpu_equivalence,
        load_initial_model, make_training_config, prepare_batch, restore_training_checkpoint,
        save_training_checkpoint, source_provenance, state_signatures)
    pipeline = MetricPipeline.from_files(run['cue_config'], run['norm_stats_dir'])
    dataset = SpatialDataset(run['data']['rgbd_root'], run['data']['geometry_root'],
                             max_cache_bytes=2 << 30, path_mappings=run['path_mappings'])
    config = make_training_config(run)
    provenance = source_provenance(ROOT)
    provenance.update(command=sys.argv, run_config=run,
                      pipeline_sha256=pipeline.fingerprint(), dataset_manifests=dataset.manifest_hashes,
                      checkpoint_receipt=admission['checkpoint_receipt'], resource_admission=admission)
    manifest_path = args.out/'run_manifest.json'
    if args.resume:
        previous = json.loads(manifest_path.read_text())
        for key in ('run_config', 'pipeline_sha256', 'dataset_manifests', 'loaded_source_sha256',
                    'dependency_lock_sha256', 'git_commit'):
            if previous[key] != provenance[key]:
                raise ValueError('Full-state resume identity differs: ' + key)
        provenance = previous
        status = json.loads((args.out/'status.json').read_text())
    else:
        manifest_path.write_text(json.dumps(provenance, indent=2))
        status = dict(state='running', gpu_job_started=True, formal_training_started=args.mode == 'train',
                      mode=args.mode, optimizer_updates=0, metrics=[] if args.mode == 'smoke' else None)
    def stage(name, **values):
        status.update(stage=name, **values)
        (args.out/'status.json').write_text(json.dumps(status, indent=2, allow_nan=False))
        print(json.dumps({'stage':name, **values}), flush=True)
    if len(jax.devices('gpu')) != len(run['resources']['gpu_indices']):
        raise ValueError('JAX did not expose every selected GPU')
    gpu, cpu = jax.devices('gpu')[0], jax.devices('cpu')[0]
    mesh = (sharding.make_mesh(len(run['resources']['gpu_indices']))
            if len(run['resources']['gpu_indices']) > 1 else None)
    stage('load_real_base_and_six_channel_optimizer')
    with jax.default_device(cpu):
        state = create_training_state(config, load_initial_model(config), optimizer_device=cpu)
    manager, resuming = checkpoints.initialize_checkpoint_dir(
        args.out/'checkpoints', keep_period=10000 if args.mode == 'train' else 1,
        overwrite=False, resume=args.resume)
    if args.resume:
        if not resuming:
            raise FileNotFoundError('No complete full-state checkpoint to resume')
        latest = max(manager.all_steps())
        template = jax.tree.map(lambda value: jax.numpy.zeros_like(value), state)
        state = restore_training_checkpoint(manager, template, pipeline, config, step=latest,
                                            expected_manifest=provenance)
        if int(state.step) != latest:
            raise ValueError('Restored optimizer step differs from checkpoint directory')
    first_batch = None
    if args.mode == 'smoke':
        indices = update_indices(len(dataset), config.seed, 0, batch_size=config.batch_size)
        first_batch = prepare_batch(dataset, indices, pipeline)
        stage('gpu_zero_cue_equivalence')
        measured = gpu_equivalence(config, state.params, first_batch, gpu,
                                   run['gpu_equivalence'], mesh=mesh)
        (args.out/'gpu_equivalence.json').write_text(json.dumps(measured, indent=2))
    trainer = MetricTrainer(config, state.model_def, gradient_device=gpu,
                            optimizer_device=cpu, gradient_mesh=mesh)
    current_gpus = query_gpus()
    selected = run['resources']['gpu_indices']
    required_free = admission['selected_gpu_min_free_mib']
    if isinstance(current_gpus, dict):
        blockers = ['GPU query unavailable before parameter update']
    else:
        available = {item['index']: item['free_mib'] for item in current_gpus}
        blockers = [f'GPU {index} requires {required_free} MiB free before parameter update'
                    for index in selected if available.get(index, 0) < required_free]
    if blockers:
        stage('resource_gate_closed', state='blocked', blockers=blockers,
              gpu_snapshot=current_gpus)
        manager.close()
        return 2
    saved = None
    for update in range(int(state.step), steps):
        indices = update_indices(len(dataset), config.seed, update, batch_size=config.batch_size)
        batch = first_batch if update == 0 and first_batch is not None else prepare_batch(dataset, indices, pipeline)
        started = time.monotonic()
        micro_batch_size = (protocol['training']['micro_batch_size'] if args.mode == 'train'
                            else 1 if mesh is not None else None)
        state, metrics = trainer.step(state, batch, micro_batch_size=micro_batch_size)
        if args.mode == 'smoke' and metrics['cue_grad_l2'] <= 0:
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
            stage('save_full_training_state')
            saved = save_training_checkpoint(manager, state, pipeline, config, provenance)
    if args.mode == 'smoke':
        stage('save_full_training_state')
        expected = state_signatures(state)
        saved = save_training_checkpoint(manager, state, pipeline, config, provenance)
        stage('restore_full_training_state')
        template = jax.tree.map(lambda value: jax.numpy.zeros_like(value), state)
        restored = restore_training_checkpoint(manager, template, pipeline, config, step=int(state.step),
                                                expected_manifest=provenance)
        if state_signatures(restored) != expected:
            raise ValueError('Full six-channel parameter/optimizer state changed during recovery')
        (args.out/'full_state_sha256.json').write_text(json.dumps(expected, indent=2))
    manager.close()
    stage('complete', state='complete', full_state_restore_bitwise=args.mode == 'smoke',
          checkpoint=str(saved if saved is not None else args.out/'checkpoints'/str(int(state.step))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
