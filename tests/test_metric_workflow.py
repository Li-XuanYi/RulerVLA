"""G2 input equality and real action-chunk boundaries, independent of model size."""
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest

from openpi.shared.normalize import NormStats
from rulervla.adapters.metric_pipeline import MetricPipeline
from rulervla.adapters.libero import LiberoAdapter
from rulervla.cues.metric_ruler import MetricCueConfig
from rulervla.training.main_data import SpatialDataset, action_chunk_indices


def stats():
    return {key: NormStats(mean=np.zeros(dim), std=np.ones(dim),
        q01=-np.arange(1,dim+1,dtype=np.float32), q99=np.arange(1,dim+1,dtype=np.float32))
        for key,dim in [('state',8),('actions',7)]}


def pipeline_and_sample(sample, config):
    sample['wrist_intrinsic']=sample['intrinsic'].copy()
    sample['wrist_camera_to_world']=np.eye(4)
    sample['actions']=np.arange(70,dtype=np.float32).reshape(10,7)/100
    pipeline=MetricPipeline(config['geometry'],MetricCueConfig(),stats())
    return pipeline,sample


def assert_equal_tree(a,b):
    assert a.keys()==b.keys()
    for key in a:
        if isinstance(a[key],dict): assert_equal_tree(a[key],b[key])
        else:
            assert a[key].shape==b[key].shape
            assert a[key].dtype==b[key].dtype
            np.testing.assert_array_equal(a[key],b[key])


def test_training_and_inference_use_identical_current_observation(sample,config):
    pipeline,sample=pipeline_and_sample(sample,config)
    original=sample['agentview_rgb'].copy()
    trained=pipeline.prepare_train(sample)
    inferred=pipeline.prepare_inference(sample)
    labels=trained.pop('actions')
    assert_equal_tree(trained,inferred)
    assert labels.shape==(10,32) and labels.dtype==np.float32
    np.testing.assert_array_equal(original,sample['agentview_rgb'])
    assert list(inferred['image'])==['base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb','aux_wrist_cue_rgb']
    assert list(inferred['image_mask'].values())==[True,True,False,False]
    assert all(x.shape==(224,224,6) and x.dtype==np.float32 for x in inferred['image'].values())
    assert inferred['state'].shape==(32,) and inferred['state'].dtype==np.float32
    assert inferred['tokenized_prompt'].dtype==np.int32


def test_future_action_changes_cannot_change_current_model_input(sample,config):
    pipeline,sample=pipeline_and_sample(sample,config)
    before=pipeline.prepare_inference(sample)
    sample['actions']+=123
    after=pipeline.prepare_train(sample)
    after.pop('actions')
    assert_equal_tree(before,after)


def test_quantile_action_inverse_returns_only_libero_seven_dimensions(sample,config):
    pipeline,sample=pipeline_and_sample(sample,config)
    labels=pipeline.prepare_train(sample)['actions']
    labels[:,7:]=987
    restored=pipeline.postprocess_actions(labels)
    assert restored.shape==(10,7) and restored.dtype==np.float32
    # One float32 ULP at the widest quantile range bounds normalization rounding.
    bound=np.finfo(np.float32).eps*np.max(stats()['actions'].q99-stats()['actions'].q01)
    np.testing.assert_allclose(restored,sample['actions'],atol=bound,rtol=0)


def test_repeated_single_action_shape_is_rejected(sample,config):
    pipeline,sample=pipeline_and_sample(sample,config)
    sample['actions']=np.ones(7,np.float32)
    with pytest.raises(ValueError,match='10, 7'):
        pipeline.prepare_train(sample)


def make_dataset():
    dataset=SpatialDataset.__new__(SpatialDataset)
    dataset.rows=[dict(episode_index=10,frames=12,task='task A'),dict(episode_index=11,frames=13,task='task B')]
    dataset.ends=np.array([12,25]);dataset.geometry={10:{},11:{}}
    dataset.cache=OrderedDict();dataset.cache_bytes=0;dataset.max_cache_bytes=None
    dataset._load_episode=lambda row,geo:{'state8':np.zeros((row['frames'],8),np.float32),
        'actions':(np.arange(row['frames']*7,dtype=np.float32).reshape(-1,7)+1000*row['episode_index'])}
    return dataset


def test_true_contiguous_actions_and_repeat_last_never_cross_episode():
    dataset=make_dataset()
    np.testing.assert_array_equal(dataset[1]['actions'],dataset.cache[10]['actions'][1:11])
    expected=dataset.cache[10]['actions'][[10,11,11,11,11,11,11,11,11,11]]
    np.testing.assert_array_equal(dataset[10]['actions'],expected)
    assert dataset[12]['actions'][0,0]==11000
    identity=dataset.sample_identity(10)
    assert identity['episode_index']==10 and identity['frame_index']==10
    assert identity['action_frame_indices']==[10,11,11,11,11,11,11,11,11,11]
    assert identity['action_valid_mask']==[True,True,False,False,False,False,False,False,False,False]


def test_chunk_indices_reject_invalid_identity():
    for frame,length in [(-1,10),(10,10),(0,0)]:
        with pytest.raises((ValueError,IndexError)):
            action_chunk_indices(frame,length)


def test_live_calibrated_observation_matches_dataset_preprocessing(sample,config):
    pipeline,sample=pipeline_and_sample(sample,config)
    raw=dict(agentview_image=sample['agentview_rgb'][::-1,::-1].copy(),
             robot0_eye_in_hand_image=sample['wrist_rgb'][::-1,::-1].copy(),
             robot0_eef_pos=sample['ee_world_m'],robot0_eef_quat=sample['eef_quaternion_xyzw'])
    calibration=dict(depth_m=sample['agentview_depth_m'][::-1,::-1].copy(),
        intrinsic=sample['intrinsic'],camera_to_world=sample['camera_to_world'],
        wrist_depth_m=sample['agentview_depth_m'][::-1,::-1].copy(),
        wrist_intrinsic=sample['wrist_intrinsic'],wrist_camera_to_world=sample['wrist_camera_to_world'],
        state8=sample['state8'],instruction=sample['task'],episode_index=10,control_step=20,calibration_step=20)
    adapter=LiberoAdapter(config['geometry'])
    live=adapter.from_metric_observation(raw,**calibration)
    assert_equal_tree(pipeline.prepare_inference(live),pipeline.prepare_inference(sample))
    calibration['calibration_step']=19
    with pytest.raises(ValueError,match='same control step'):
        adapter.from_metric_observation(raw,**calibration)
