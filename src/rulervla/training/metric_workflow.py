"""Six-channel action training and checkpoint-backed inference share one pipeline."""
import dataclasses
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import flax.nnx as nnx
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.models import model as model_lib
from openpi.shared import normalize
from openpi.training import checkpoints, config as config_lib, optimizer, weight_loaders
from openpi.training.utils import TrainState
from rulervla.adapters.metric_pipeline import MetricPipeline, metric_model_config, tokenizer_from_asset
from rulervla.cues.metric_ruler import MetricCueConfig
from rulervla.cues.motion_echo import MotionEchoConfig
from rulervla.training.accumulation import apply_gradients

STEM = 'PaliGemma/img/embedding/kernel'
CHECKPOINT_SCHEMA = 'rulervla_metric_checkpoint_v1'


@dataclasses.dataclass(frozen=True)
class ConstantLearningRate:
    value: float

    def create(self):
        return optax.constant_schedule(self.value)


def make_training_config(run):
    policy = metric_model_config()
    training = run['training']
    return dataclasses.replace(config_lib.get_config('pi05_libero'), model=policy,
        batch_size=training['batch_size'], seed=training['seed'], ema_decay=None,
        freeze_filter=policy.get_freeze_filter(), wandb_enabled=False,
        num_train_steps=training['num_steps'],
        optimizer=optimizer.AdamW(**training['optimizer']),
        lr_schedule=ConstantLearningRate(training['learning_rate']),
        weight_loader=weight_loaders.SixChannelCheckpointWeightLoader(run['initial_checkpoint'],
                                                                       missing_regex='(?!)'))


def flat_params(state):
    if isinstance(state, nnx.State):
        state = state.to_pure_dict()
    return flax.traverse_util.flatten_dict(state, sep='/')


def require_trainable_cue(params, trainable_filter):
    selected = params.filter(trainable_filter)
    kernel = flat_params(selected).get(STEM)
    if kernel is None or kernel.ndim != 4 or kernel.shape[2] != 6:
        raise ValueError('Six-channel Cue patch kernel is not in the actual trainable filter')
    return selected


def require_cue_optimizer(state):
    found = []
    def visit(value):
        if hasattr(value, 'mu') and hasattr(value, 'nu'):
            for name in ('mu', 'nu'):
                kernel = flat_params(getattr(value, name)).get(STEM)
                if kernel is not None and kernel.shape[2] == 6:
                    found.append(name)
        elif isinstance(value, (list, tuple)):
            for part in value:
                visit(part)
    visit(state)
    if sorted(found) != ['mu', 'nu']:
        raise ValueError('Adam optimizer does not manage the six-channel Cue kernel')
    return sorted(found)


def load_initial_model(config):
    reference = nnx.state(nnx.eval_shape(config.model.create, jax.random.key(config.seed))).to_pure_dict()
    params = config.weight_loader.load(reference)
    return config.model.load(params, remove_extra_params=False)


def create_training_state(config, policy, *, optimizer_device=None):
    graph, params = nnx.split(policy)
    if optimizer_device is not None:
        params = jax.device_put(params, optimizer_device)
    selected = require_trainable_cue(params, config.trainable_filter)
    tx = config.optimizer.create(config.lr_schedule.create())
    opt_state = jax.jit(tx.init)(selected)
    require_cue_optimizer(opt_state)
    step = jax.device_put(jnp.asarray(0, jnp.int32), optimizer_device)
    return TrainState(step=step, params=params, model_def=graph, tx=tx,
                      opt_state=opt_state, ema_decay=None)


@dataclasses.dataclass(frozen=True)
class MetricBatch:
    observation: object
    actions: object
    identities: tuple


def prepare_batch(dataset, indices, pipeline):
    values = [pipeline.prepare_train(dataset[int(index)]) for index in indices]
    batched = jax.tree.map(lambda *leaves: np.stack(leaves), *values)
    labels = batched.pop('actions')
    return MetricBatch(model_lib.Observation.from_dict(batched), labels,
                       tuple(dataset.sample_identity(int(index)) for index in indices))


def action_loss(policy, rng, observation, actions):
    """G3 may wrap this objective inside the same NNX differentiated model."""
    return policy.compute_loss(rng, observation, actions, train=True).mean()


class MetricTrainer:
    def __init__(self, config, model_def, *, gradient_device=None, optimizer_device=None,
                 objective=action_loss):
        self.config = config
        self.gradient_device = gradient_device
        self.optimizer_device = optimizer_device
        def backward(params, observation, actions, rng):
            policy = nnx.merge(model_def, params)
            policy.train()
            loss, gradients = nnx.value_and_grad(
                lambda module: objective(module, rng, observation, actions),
                argnums=nnx.DiffState(0, config.trainable_filter))(policy)
            cue = flat_params(gradients)[STEM][..., 3:, :]
            finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(gradients)]))
            return gradients, dict(loss=loss, grad_norm=optax.global_norm(gradients),
                                   cue_grad_l2=jnp.linalg.norm(cue), gradients_finite=finite)
        self._backward = jax.jit(backward)
        self._update = jax.jit(lambda state, gradient: apply_gradients(config, state, gradient))

    def step(self, state, batch):
        rng = jax.random.fold_in(jax.random.key(self.config.seed), state.step)
        params, observation, labels, rng = jax.device_put(
            (state.params, batch.observation, batch.actions, rng), self.gradient_device)
        gradients, metrics = self._backward(params, observation, labels, rng)
        jax.block_until_ready(gradients)
        info = {key: float(value) for key, value in jax.device_get(metrics).items()}
        if not info['gradients_finite'] or not np.isfinite(info['loss']):
            raise ValueError('Nonfinite action training loss or gradients')
        gradients = jax.device_put(gradients, self.optimizer_device)
        updated = self._update(state, gradients)
        jax.block_until_ready(updated)
        require_cue_optimizer(updated.opt_state)
        return updated, info


def training_recipe(config):
    return dict(model=dataclasses.asdict(config.model), optimizer=dataclasses.asdict(config.optimizer),
                learning_rate=dataclasses.asdict(config.lr_schedule), seed=config.seed,
                batch_size=config.batch_size, ema_decay=config.ema_decay,
                trainable_policy='full_finetuning_including_visual_patch_kernel')


def save_training_checkpoint(manager, state, pipeline, config, run_manifest):
    class Assets:
        def data_config(self):
            return SimpleNamespace(norm_stats=pipeline.norm_stats, asset_id='physical-intelligence/libero')
    step = int(state.step)
    checkpoints.save_state(manager, state, Assets(), step)
    manager.wait_until_finished()
    directory = Path(manager.directory)/str(step)
    (directory/'assets/paligemma_tokenizer.model').write_bytes(pipeline.tokenizer_bytes)
    contract = dict(schema=CHECKPOINT_SCHEMA, completed_updates=step,
                    pipeline=pipeline.contract(), pipeline_sha256=pipeline.fingerprint(),
                    training_recipe=training_recipe(config), run_manifest=run_manifest)
    (directory/'metric_workflow.json').write_text(json.dumps(contract, indent=2, allow_nan=False)+'\n')
    return directory


def restore_training_checkpoint(manager, state, pipeline, config, *, step, expected_manifest=None):
    directory = Path(manager.directory)/str(step)
    metadata = json.loads((directory/'metric_workflow.json').read_text())
    if (metadata.get('schema') != CHECKPOINT_SCHEMA or metadata['pipeline_sha256'] != pipeline.fingerprint()
            or metadata['training_recipe'] != training_recipe(config)):
        raise ValueError('Checkpoint pipeline/model/optimizer recipe differs from the current six-channel run')
    if metadata['pipeline']['model']['image_channels'] != 6:
        raise ValueError('Three-channel optimizer state cannot resume a six-channel model')
    if expected_manifest is not None:
        for key in ('dataset_manifests', 'loaded_source_sha256', 'dependency_lock_sha256'):
            if metadata['run_manifest'].get(key) != expected_manifest.get(key):
                raise ValueError('Checkpoint data/source identity differs: '+key)
    restored = checkpoints.restore_state(manager, state, None, step=step)
    require_trainable_cue(restored.params, config.trainable_filter)
    require_cue_optimizer(restored.opt_state)
    return restored


def pipeline_from_checkpoint(directory):
    directory = Path(directory)
    metadata = json.loads((directory/'metric_workflow.json').read_text())
    if metadata.get('schema') != CHECKPOINT_SCHEMA:
        raise ValueError('Unsupported metric checkpoint schema')
    contract = metadata['pipeline']
    if contract.get('version') != MetricPipeline.version:
        raise ValueError('Checkpoint preprocessing version differs; use its original committed runtime')
    if contract['model'] != dataclasses.asdict(metric_model_config()):
        raise ValueError('Checkpoint model differs from the validated pi05 six-channel contract')
    # Norm statistics are saved by OpenPI beside inference params; never use the caller's mutable stats.
    stats = normalize.load(directory/'assets/physical-intelligence/libero')
    tokenizer = tokenizer_from_asset(directory/'assets/paligemma_tokenizer.model', contract['tokenizer']['max_len'])
    pipeline = MetricPipeline(contract['geometry'], MetricCueConfig(**contract['metric_ruler']), stats,
                              echo=MotionEchoConfig(**contract['motion_echo']), tokenizer=tokenizer)
    if pipeline.fingerprint() != metadata['pipeline_sha256']:
        raise ValueError('Checkpoint input assets or normalization statistics changed')
    return pipeline, metadata


class MetricPolicy:
    def __init__(self, policy, pipeline, *, sampling_steps=10, seed=42):
        from openpi.shared.nnx_utils import module_jit
        self.pipeline = pipeline
        self._sample = module_jit(policy.sample_actions, static_argnames=('num_steps',))
        self.sampling_steps = sampling_steps
        self.rng = jax.random.key(seed)

    def prepare(self, sample):
        return self.pipeline.prepare_inference(sample)

    def infer_observation(self, observation, *, noise=None, **synchronized_calibration):
        from rulervla.adapters.libero import LiberoAdapter
        sample = LiberoAdapter(self.pipeline.geometry).from_metric_observation(
            observation, **synchronized_calibration)
        return self.infer(sample, noise=noise)

    def infer(self, sample, *, noise=None):
        inputs = self.prepare(sample)
        batched = jax.tree.map(lambda value: jnp.asarray(value)[None], inputs)
        observation = model_lib.Observation.from_dict(batched)
        self.rng, rng = jax.random.split(self.rng)
        if noise is not None:
            noise = jnp.asarray(noise)
            if noise.shape == (10, 32):
                noise = noise[None]
        normalized = np.asarray(self._sample(rng, observation, num_steps=self.sampling_steps, noise=noise))[0]
        return dict(actions=self.pipeline.postprocess_actions(normalized), normalized_actions=normalized)


def load_metric_policy(directory, *, explicit_pipeline=None, sampling_steps=10):
    directory = Path(directory)
    if (directory/'metric_workflow.json').exists():
        pipeline, _ = pipeline_from_checkpoint(directory)
        if explicit_pipeline is not None and explicit_pipeline.fingerprint() != pipeline.fingerprint():
            raise ValueError('Caller preprocessing differs from checkpoint preprocessing')
    elif explicit_pipeline is not None:
        # Engineering-only bridge for the G1 six-channel artifact, not a full-state resume.
        pipeline = explicit_pipeline
    else:
        raise ValueError('A G2 checkpoint with saved preprocessing is required')
    params = model_lib.restore_params(directory/'params', restore_type=np.ndarray)
    policy = pipeline.model_config.load(params, remove_extra_params=False)
    return MetricPolicy(policy, pipeline, sampling_steps=sampling_steps)


def state_signatures(state):
    result = []
    for path, value in jax.tree_util.tree_flatten_with_path(state)[0]:
        array = np.asarray(jax.device_get(value))
        result.append(dict(path=jax.tree_util.keystr(path), shape=list(array.shape), dtype=str(array.dtype),
                           sha256=hashlib.sha256(array.reshape(-1).view(np.uint8)).hexdigest()))
    return result


def source_provenance(root):
    """Resolve imported code and capture dirty state without altering other work."""
    import openpi.models.model
    import openpi_client.image_tools
    root = Path(root).resolve()
    tracked = set(subprocess.check_output(['git', 'ls-files'], cwd=root, text=True).splitlines())
    changed = set(subprocess.check_output(['git', 'diff', '--name-only', 'HEAD'], cwd=root, text=True).splitlines())
    loaded = {}
    for name, module in list(sys.modules.items()):
        filename = getattr(module, '__file__', None)
        if filename and name.startswith(('rulervla.', 'openpi.', 'openpi_client.')) and filename.endswith('.py'):
            path = Path(filename).resolve()
            if not path.is_relative_to(root/'src'):
                raise ValueError(f'External runtime source imported: {name}: {path}')
            relative = str(path.relative_to(root))
            if relative not in tracked:
                raise ValueError(f'Runtime source is not committed: {relative}')
            if relative in changed:
                raise ValueError(f'Runtime source differs from its Git commit: {relative}')
            loaded[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                git_status=subprocess.check_output(['git', 'status', '--porcelain=v1'], cwd=root, text=True),
                loaded_source_sha256=loaded,
                dependency_lock_sha256=hashlib.sha256((root/'requirements/metric-cue-linux.lock').read_bytes()).hexdigest())


def gpu_equivalence(config, params6, batch, device, tolerances):
    """Measure on the actual GPU; CPU exactness is not an assumed GPU result."""
    import gc
    from openpi.training.weight_loaders import merge_six_channel_params
    observation = batch.observation
    zero = dataclasses.replace(observation, images={
        key: np.concatenate((np.asarray(image)[..., :3], np.zeros_like(np.asarray(image)[..., 3:])), axis=-1)
        for key, image in observation.images.items()})
    rgb = dataclasses.replace(observation, images={key: value[..., :3] for key, value in zero.images.items()})
    noise = np.asarray(jax.random.normal(jax.random.key(77), batch.actions.shape))
    outputs = []
    for channels, obs in ((3, rgb), (6, zero)):
        cfg = dataclasses.replace(config.model, image_channels=channels)
        if channels == 3:
            reference = nnx.state(nnx.eval_shape(cfg.create, jax.random.key(42))).to_pure_dict()
            loaded = model_lib.restore_params(config.weight_loader.params_path, restore_type=np.ndarray)
            loaded = merge_six_channel_params(loaded, reference, missing_regex='(?!)')
            graph, state = nnx.split(cfg.load(loaded, remove_extra_params=False))
            del loaded
        else:
            graph = nnx.split(nnx.eval_shape(cfg.create, jax.random.key(42)))[0]
            state = params6
        def evaluate(params, current):
            policy = nnx.merge(graph, params)
            policy.eval()
            prefix, _, _ = policy.embed_prefix(current)
            prediction = policy.sample_actions(jax.random.key(123), current,
                                               noise=jnp.asarray(noise), num_steps=10)
            return prefix, prediction
        compiled = jax.jit(evaluate)
        values = jax.block_until_ready(compiled(jax.device_put(state, device), jax.device_put(obs, device)))
        outputs.append(tuple(np.asarray(value, np.float32) for value in values))
        del state, values, compiled
        gc.collect()
        jax.clear_caches()
    result = {}
    for index, name in ((0, 'vision'), (1, 'action')):
        old, new = outputs[0][index], outputs[1][index]
        atol, rtol = tolerances[name+'_atol'], tolerances[name+'_rtol']
        delta = new-old
        result[name] = dict(max_abs=float(np.max(np.abs(delta))), rms=float(np.sqrt(np.mean(delta**2))),
                            atol=atol, rtol=rtol)
        np.testing.assert_allclose(new, old, atol=atol, rtol=rtol)
    return result
