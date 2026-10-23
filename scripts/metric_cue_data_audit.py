"""Audit real contiguous chunks and byte-identical train/inference inputs on CPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/rulervla_metric_workflow.yaml')
    parser.add_argument('--cache-root',type=Path)
    parser.add_argument('--indices',type=int,nargs='+',default=[20,108,109,110])
    parser.add_argument('--reference-g1-frame',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    os.environ['JAX_PLATFORMS']='cpu';os.environ['CUDA_VISIBLE_DEVICES']=''
    import numpy as np
    from rulervla.adapters.metric_pipeline import MetricPipeline
    from rulervla.data.schema import validate_sample
    from rulervla.training.main_data import SpatialDataset
    from rulervla.training.metric_resources import load_run_config
    from rulervla.training.metric_workflow import prepare_batch,source_provenance
    run=load_run_config(args.config)
    mappings=dict(run['path_mappings'])
    roots={name:Path(run['data'][name+'_root']) for name in ('rgbd','geometry')}
    if args.cache_root:
        mappings['/home/Dataset2']=str(args.cache_root/'home/Dataset2')
        roots={name:args.cache_root/path.relative_to('/') for name,path in roots.items()}
    dataset=SpatialDataset(roots['rgbd'],roots['geometry'],max_cache_bytes=1<<30,path_mappings=mappings)
    for name in ('rgbd','geometry'):
        if dataset.manifest_hashes[name]!=run['data'][name+'_manifest_sha256']:
            raise ValueError('Cached manifest does not match the pinned complete training manifest')
    pipeline=MetricPipeline.from_files(run['cue_config'],run['norm_stats_dir'])
    args.out.mkdir(parents=True,exist_ok=False)
    records=[]
    for index in args.indices:
        sample=dataset[index]
        validate_sample(sample)
        identity=dataset.sample_identity(index)
        original_actions=dataset.cache[identity['episode_index']]['actions'][identity['action_frame_indices']]
        np.testing.assert_array_equal(sample['actions'],original_actions)
        trained=pipeline.prepare_train(sample)
        inferred=pipeline.prepare_inference(sample)
        normalized=trained.pop('actions')
        fields={}
        def compare(left,right,prefix=''):
            if left.keys()!=right.keys():raise ValueError('Training/inference input key mismatch')
            for key,a in left.items():
                b=right[key];name=prefix+key
                if isinstance(a,dict):compare(a,b,name+'/');continue
                if a.shape!=b.shape or a.dtype!=b.dtype:raise ValueError('Input shape/dtype differs: '+name)
                np.testing.assert_array_equal(a,b)
                fields[name]=dict(shape=list(a.shape),dtype=str(a.dtype),max_abs_difference=0,
                                  minimum=float(np.min(a)),maximum=float(np.max(a)))
        compare(trained,inferred)
        restored=pipeline.postprocess_actions(normalized)
        inverse_error=float(np.max(np.abs(restored-sample['actions'])))
        np.testing.assert_allclose(restored,sample['actions'],atol=1e-6,rtol=0)
        record=dict(identity=identity,train_inference_fields=fields,
                    raw_action_shape=list(sample['actions'].shape),normalized_action_shape=list(normalized.shape),
                    inverse_action_max_abs_error=inverse_error,
                    action_chunk_sha256=hashlib.sha256(sample['actions'].tobytes()).hexdigest(),
                    equals_repeated_current_action=bool(np.array_equal(sample['actions'],
                        np.repeat(sample['actions'][:1],10,axis=0))))
        if not records:
            np.savez_compressed(args.out/'real_training_sample.npz',**sample,
                                episode_index=np.int64(identity['episode_index']),
                                frame_index=np.int64(identity['frame_index']))
            if args.reference_g1_frame:
                with np.load(args.reference_g1_frame,allow_pickle=False) as old:
                    for key in ('agentview_rgb','wrist_rgb','ee_world_m','state8','intrinsic','camera_to_world',
                                'wrist_intrinsic','wrist_camera_to_world','eef_quaternion_xyzw'):
                        np.testing.assert_array_equal(sample[key],old[key])
                record['g1_current_observation_exact']=True
        records.append(record)
    batch=prepare_batch(dataset,args.indices[:2],pipeline)
    report=dict(state='complete',dataset_frames=len(dataset),manifest_sha256=dataset.manifest_hashes,
                pipeline_sha256=pipeline.fingerprint(),samples=records,
                real_batch=dict(actions_shape=list(batch.actions.shape),actions_dtype=str(batch.actions.dtype),
                    image_shapes={key:list(value.shape) for key,value in batch.observation.images.items()}),
                source=source_provenance(ROOT),cache_root=str(args.cache_root) if args.cache_root else None,
                primary_dataset_mount_repaired=False,gpu_job_started=False)
    (args.out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({key:value for key,value in report.items() if key not in ('samples','source')}),flush=True)


if __name__=='__main__':main()
