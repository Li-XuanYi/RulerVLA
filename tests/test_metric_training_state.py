"""Small orchestration tests; full pi05 capacity is covered by retained G1 evidence."""
import dataclasses
import json

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.shared.normalize import NormStats
from openpi.training import checkpoints
from rulervla.adapters.metric_pipeline import MetricPipeline
from rulervla.cues.metric_ruler import MetricCueConfig
from rulervla.training.metric_workflow import (MetricTrainer, create_training_state, flat_params,
    make_training_config, pipeline_from_checkpoint, prepare_batch, require_cue_optimizer,
    require_trainable_cue, restore_training_checkpoint, save_training_checkpoint, STEM)


def run_config():
    return dict(initial_checkpoint='unused',training=dict(batch_size=1,num_steps=2,seed=42,
        learning_rate=1e-4,optimizer=dict(b1=.9,b2=.95,eps=1e-8,weight_decay=1e-10,clip_gradient_norm=1.)))


class TinyActionModel(nnx.Module):
    def __init__(self):
        kernel=jnp.concatenate((jnp.full((1,1,3,2),.1),jnp.zeros((1,1,3,2))),axis=2)
        self.PaliGemma=nnx.Dict(img=nnx.Dict(embedding=nnx.Dict(kernel=nnx.Param(kernel))))
        self.action=nnx.Param(jnp.full((2,32),.05))

    def compute_loss(self,rng,observation,actions,train=False):
        del rng,train
        pixels=observation.images['base_0_rgb'].mean(axis=(1,2))
        hidden=pixels@self.PaliGemma.img.embedding.kernel.value[0,0]
        predicted=(hidden@self.action.value)[:,None,:]
        return jnp.square(predicted-actions).mean(axis=-1)


def test_actual_workflow_optimizer_checkpoint_and_preprocessing_restore(tmp_path,sample,config):
    sample['wrist_intrinsic']=sample['intrinsic'].copy();sample['wrist_camera_to_world']=np.eye(4)
    sample['actions']=np.full((10,7),.2,np.float32)
    stats={k:NormStats(mean=np.zeros(n),std=np.ones(n),q01=-np.ones(n),q99=np.ones(n))
           for k,n in [('state',8),('actions',7)]}
    pipeline=MetricPipeline(config['geometry'],MetricCueConfig(),stats)
    class Dataset:
        def __getitem__(self,index): return sample
        def sample_identity(self,index): return dict(episode_index=10,frame_index=index)
    batch=prepare_batch(Dataset(),[0],pipeline)
    recipe=make_training_config(run_config())
    cpu=jax.devices('cpu')[0]
    state=create_training_state(recipe,TinyActionModel(),optimizer_device=cpu)
    require_trainable_cue(state.params,recipe.trainable_filter)
    assert require_cue_optimizer(state.opt_state)==['mu','nu']
    trainer=MetricTrainer(recipe,state.model_def,gradient_device=cpu,optimizer_device=cpu)
    updated,metrics=trainer.step(state,batch)
    assert metrics['cue_grad_l2']>0 and int(updated.step)==1
    assert np.any(np.asarray(flat_params(updated.params)[STEM])[...,3:,:]!=0)
    manager,_=checkpoints.initialize_checkpoint_dir(tmp_path/'checkpoints',keep_period=1,overwrite=False,resume=False)
    directory=save_training_checkpoint(manager,updated,pipeline,recipe,dict(test='small orchestration'))
    restored=restore_training_checkpoint(manager,state,pipeline,recipe,step=1)
    for actual,expected in zip(jax.tree.leaves(restored),jax.tree.leaves(updated),strict=True):
        np.testing.assert_array_equal(actual,expected)
    recovered,_=pipeline_from_checkpoint(directory)
    assert recovered.fingerprint()==pipeline.fingerprint()
    # A normalization edit must not silently change a recovered inference policy.
    stats_file=directory/'assets/physical-intelligence/libero/norm_stats.json'
    changed=json.loads(stats_file.read_text());changed['norm_stats']['actions']['q99'][0]=3
    stats_file.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='assets|statistics'):
        pipeline_from_checkpoint(directory)
    manager.close()


def test_frozen_cue_or_missing_optimizer_state_is_rejected():
    recipe=make_training_config(run_config())
    state=nnx.state(TinyActionModel())
    frozen=dataclasses.replace(recipe,freeze_filter=nnx.PathContains('img'))
    with pytest.raises(ValueError,match='trainable'):
        require_trainable_cue(state,frozen.trainable_filter)
    with pytest.raises(ValueError,match='optimizer'):
        require_cue_optimizer(())
