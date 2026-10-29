import json
import math
from pathlib import Path

import flax.nnx as nnx
import jax
import numpy as np

from rulervla.adapters.metric_pipeline import MetricPipeline
from rulervla.training.main_data import SpatialDataset, update_indices
from rulervla.training.metric_resources import load_run_config
from rulervla.training.metric_workflow import MetricTrainer, make_training_config, prepare_batch

ROOT = Path(__file__).resolve().parents[1]


def test_real_batch8_full_lora_flow_matching_gradient_shapes():
    run = load_run_config(ROOT / 'configs/metric_workflow_lora.yaml')
    protocol = json.loads((ROOT / 'experiments/metric_cue_v1/protocol.json').read_text())
    run['training'] = dict(protocol['training'], seed=42)
    dataset = SpatialDataset(run['data']['rgbd_root'], run['data']['geometry_root'],
        max_cache_bytes=run['data']['max_cache_bytes'], path_mappings=run['path_mappings'])
    pipeline = MetricPipeline.from_files(run['cue_config'], run['norm_stats_dir'])
    batch = prepare_batch(dataset, update_indices(len(dataset), 42, 0, batch_size=8), pipeline)
    config = make_training_config(run)
    graph, params = nnx.split(nnx.eval_shape(config.model.create, jax.random.key(42)))
    trainer = MetricTrainer(config, graph)
    observations = jax.tree.map(lambda value: value.reshape((8, 1) + value.shape[1:]), batch.observation)
    actions = batch.actions.reshape((8, 1, 10, 32))
    gradients, metrics = jax.eval_shape(
        trainer._accumulated_backward, params, observations, actions, np.int32(0))
    expected = params.filter(config.trainable_filter)
    assert jax.tree.structure(gradients) == jax.tree.structure(expected)
    assert sum(math.prod(value.shape) for value in jax.tree.leaves(gradients)) == 53509280
    assert all(metrics[name].shape == () for name in
               ('loss', 'grad_norm', 'cue_grad_l2', 'gradients_finite', 'lora_a_grad_l2', 'lora_b_grad_l2'))
    assert batch.actions.shape == (8, 10, 32) and len(batch.identities) == 8
