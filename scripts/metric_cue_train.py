"""Metric Ruler six-channel training. Default: admission report only; no GPU job."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/rulervla_metric_workflow.yaml')
    parser.add_argument('--mode', choices=('preflight', 'smoke', 'train'), default='preflight')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--steps', type=int)
    args = parser.parse_args()
    from rulervla.training.metric_resources import load_run_config, preflight
    run = load_run_config(args.config)
    args.out.mkdir(parents=True, exist_ok=False)
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
    if len(run['resources']['gpu_indices']) != 1 or not run['resources']['cpu_optimizer']:
        raise ValueError('This entry uses one GPU for gradients and a CPU AdamW optimizer')
    steps = args.steps if args.steps is not None else run['training']['num_steps']
    if steps < 1 or (args.mode == 'smoke' and steps > 3):
        raise ValueError('Smoke requires 1..3 updates; larger runs need explicit --mode train')
    run['training']['num_steps'] = steps
    (args.out/'workflow_config.json').write_text(json.dumps(run, indent=2))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(run['resources']['gpu_indices'][0])
    os.environ['JAX_PLATFORMS'] = 'cuda,cpu'
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
    import jax
    from openpi.training import checkpoints
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
    (args.out/'run_manifest.json').write_text(json.dumps(provenance, indent=2))
    status = dict(state='running', gpu_job_started=True, formal_training_started=args.mode == 'train',
                  mode=args.mode, optimizer_updates=0, metrics=[])
    def stage(name, **values):
        status.update(stage=name, **values)
        (args.out/'status.json').write_text(json.dumps(status, indent=2, allow_nan=False))
        print(json.dumps({'stage':name, **values}), flush=True)
    try:
        gpu, cpu = jax.devices('gpu')[0], jax.devices('cpu')[0]
        stage('load_real_base_and_fresh_six_channel_optimizer')
        state = create_training_state(config, load_initial_model(config), optimizer_device=cpu)
        indices = update_indices(len(dataset), config.seed, 0, batch_size=config.batch_size)
        first_batch = prepare_batch(dataset, indices, pipeline)
        if args.mode == 'smoke':
            stage('gpu_zero_cue_equivalence')
            measured = gpu_equivalence(config, state.params, first_batch, gpu, run['gpu_equivalence'])
            (args.out/'gpu_equivalence.json').write_text(json.dumps(measured, indent=2))
        trainer = MetricTrainer(config, state.model_def, gradient_device=gpu, optimizer_device=cpu)
        for update in range(steps):
            indices = update_indices(len(dataset), config.seed, update, batch_size=config.batch_size)
            batch = first_batch if update == 0 else prepare_batch(dataset, indices, pipeline)
            started = time.monotonic()
            state, metrics = trainer.step(state, batch)
            if args.mode == 'smoke' and metrics['cue_grad_l2'] <= 0:
                raise ValueError('Real action loss did not produce nonzero Cue kernel gradients')
            status['metrics'].append(dict(update=int(state.step), seconds=time.monotonic()-started,
                                          sample_identities=batch.identities, **metrics))
            stage('training_updates', optimizer_updates=int(state.step))
        stage('save_full_training_state')
        expected = state_signatures(state)
        manager, _ = checkpoints.initialize_checkpoint_dir(args.out/'checkpoints', keep_period=1,
                                                            overwrite=False, resume=False)
        saved = save_training_checkpoint(manager, state, pipeline, config, provenance)
        stage('restore_full_training_state')
        template = jax.tree.map(lambda value: jax.numpy.zeros_like(value), state)
        restored = restore_training_checkpoint(manager, template, pipeline, config, step=int(state.step),
                                                expected_manifest=provenance)
        if state_signatures(restored) != expected:
            raise ValueError('Full six-channel parameter/optimizer state changed during recovery')
        (args.out/'full_state_sha256.json').write_text(json.dumps(expected, indent=2))
        manager.close()
        stage('complete', state='complete', full_state_restore_bitwise=True, checkpoint=str(saved))
    except BaseException as error:
        stage('failed', state='failed', error=repr(error))
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
