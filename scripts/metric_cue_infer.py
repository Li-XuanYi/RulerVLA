"""Prepare a synchronized observation or predict one LIBERO action chunk; no rollout."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--g1-weights-bridge', action='store_true')
    parser.add_argument('--config', type=Path, default=ROOT/'configs/rulervla_metric_workflow.yaml')
    parser.add_argument('--mode', choices=('prepare', 'predict'), default='prepare')
    parser.add_argument('--backend', choices=('cpu', 'gpu'), default='cpu')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'predict' and args.checkpoint is None:
        raise ValueError('Prediction requires a six-channel checkpoint')
    if args.g1_weights_bridge and args.checkpoint is None:
        raise ValueError('G1 weights bridge requires a checkpoint')
    from rulervla.training.metric_resources import load_run_config, inference_preflight
    run = load_run_config(args.config)
    args.out.mkdir(parents=True, exist_ok=False)
    if args.backend == 'gpu':
        admission = inference_preflight(run, args.checkpoint, g1_weights_bridge=args.g1_weights_bridge)
        (args.out/'preflight.json').write_text(json.dumps(admission, indent=2))
        if not admission['launch_allowed']:
            raise RuntimeError('GPU admission blocked: '+str(admission['blockers']))
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str,run['resources']['gpu_indices']))
        os.environ['JAX_PLATFORMS'] = 'cuda,cpu'
        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['JAX_PLATFORMS'] = 'cpu'
    import numpy as np
    from rulervla.adapters.metric_pipeline import MetricPipeline
    from rulervla.training.metric_workflow import load_metric_policy, pipeline_from_checkpoint, source_provenance
    with np.load(args.sample, allow_pickle=False) as archive:
        sample = {key: archive[key] for key in archive.files}
    sample['task'] = str(sample['task'])
    if args.mode == 'predict':
        explicit = (MetricPipeline.from_files(run['cue_config'], run['norm_stats_dir'])
                    if args.g1_weights_bridge else None)
        policy = load_metric_policy(args.checkpoint, explicit_pipeline=explicit, sampling_steps=10)
        pipeline = policy.pipeline
    elif args.checkpoint and not args.g1_weights_bridge:
        pipeline, _ = pipeline_from_checkpoint(args.checkpoint)
    else:
        pipeline = MetricPipeline.from_files(run['cue_config'],run['norm_stats_dir'])
    prepared = pipeline.prepare_inference(sample)
    arrays = {f'image/{key}':value for key,value in prepared['image'].items()}
    arrays.update({f'image_mask/{key}':value for key,value in prepared['image_mask'].items()})
    arrays.update({key:value for key,value in prepared.items() if key not in ('image','image_mask')})
    np.savez_compressed(args.out/'model_input.npz',**arrays)
    result = dict(state='prepared', backend=args.backend, pipeline_sha256=pipeline.fingerprint(),
                  source=source_provenance(ROOT), policy_rollouts=0, robot_actions_executed=0)
    if args.mode == 'predict':
        prediction = policy.infer_prepared(prepared)
        np.savez_compressed(args.out/'prediction.npz',**prediction)
        result.update(state='predicted',actions_shape=list(prediction['actions'].shape),
                      actions_dtype=str(prediction['actions'].dtype),
                      actions_min=float(prediction['actions'].min()),actions_max=float(prediction['actions'].max()),
                      checkpoint=str(args.checkpoint), g1_weights_bridge=args.g1_weights_bridge)
    (args.out/'status.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({key:value for key,value in result.items() if key!='source'}), flush=True)


if __name__ == '__main__':
    main()
