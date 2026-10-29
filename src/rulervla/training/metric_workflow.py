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
import orbax.checkpoint as ocp

from openpi_client import image_tools
from openpi.models import model as model_lib
from openpi.shared import array_typing as at, normalize
from openpi.training import checkpoints, config as config_lib, optimizer, sharding, weight_loaders
from openpi.training.utils import TrainState
from rulervla.adapters.metric_pipeline import MetricPipeline, metric_model_config, tokenizer_from_asset
from rulervla.cues.metric_ruler import MetricRulerConfig
from rulervla.cues.motion_echo import ECHO_RENDER_CONTRACT, MetricInferenceHistory, MotionEchoConfig
from rulervla.training.accumulation import accumulated_gradients, apply_gradients
from rulervla.training.metric_lora import (
    base_reference, initialize_lora_parameters, is_lora, lora_freeze_filter,
    parameter_contract, validate_delta, verify_base_reference,
)

STEM = 'PaliGemma/img/embedding/kernel'
CHECKPOINT_SCHEMA = 'rulervla_metric_checkpoint_v1'
LORA_CHECKPOINT_SCHEMA = 'rulervla_metric_lora_checkpoint_v1'


@dataclasses.dataclass(frozen=True)
class ConstantLearningRate:
    value: float

    def create(self):
        return optax.constant_schedule(self.value)


def make_training_config(run):
    training = run['training']
    mode = training.get('finetuning', 'full')
    policy = metric_model_config(finetuning=mode)
    schedule = training.get('scheduler')
    if schedule is None:
        lr_schedule = ConstantLearningRate(training['learning_rate'])
    elif schedule['name'] == 'warmup_cosine':
        lr_schedule = optimizer.CosineDecaySchedule(
            warmup_steps=schedule['warmup_updates'], peak_lr=schedule['peak_lr'],
            decay_steps=schedule['decay_updates'], decay_lr=schedule['end_lr'])
    else:
        raise ValueError('Unknown Metric Cue learning-rate schedule')
    return dataclasses.replace(config_lib.get_config('pi05_libero'), model=policy,
        batch_size=training['batch_size'], seed=training['seed'],
        ema_decay=training.get('ema_decay'),
        freeze_filter=lora_freeze_filter() if mode == 'lora' else policy.get_freeze_filter(),
        wandb_enabled=False,
        num_train_steps=training['num_steps'],
        fsdp_devices=len(run['resources']['gpu_indices']),
        optimizer=optimizer.AdamW(**training['optimizer']),
        lr_schedule=lr_schedule,
        weight_loader=weight_loaders.SixChannelCheckpointWeightLoader(run['initial_checkpoint'],
            missing_regex=r'.*lora_[ab]' if mode == 'lora' else '(?!)'))


def flat_params(state):
    if isinstance(state, nnx.State):
        state = state.to_pure_dict()
    return flax.traverse_util.flatten_dict(state, sep='/')


def rgb_state_from_six(params6):
    state = nnx.State(params6)
    rgb_kernel = flat_params(params6)[STEM][..., :3, :]
    state.replace_by_pure_dict(flax.traverse_util.unflatten_dict(
        {tuple(STEM.split('/')): rgb_kernel}))
    return state


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


def require_training_state_device(state, device):
    if device is None:
        return
    if any(isinstance(value, np.ndarray)
           or (isinstance(value, jax.Array) and value.devices() != {device})
           for value in jax.tree.leaves(state)):
        raise ValueError('Training parameters or optimizer state are on the wrong device')


def load_initial_model(config):
    if is_lora(config):
        base_reference(config)
    reference = nnx.state(nnx.eval_shape(config.model.create, jax.random.key(config.seed))).to_pure_dict()
    params = config.weight_loader.load(reference)
    if is_lora(config):
        params = initialize_lora_parameters(params, seed=config.seed)
    return config.model.load(params, remove_extra_params=False)


def _make_training_state(config, policy, *, optimizer_device=None):
    graph, params = nnx.split(policy)
    if optimizer_device is not None:
        params = jax.device_put(params, optimizer_device)
    selected = require_trainable_cue(params, config.trainable_filter)
    if is_lora(config):
        parameter_contract(selected)
    tx = config.optimizer.create(config.lr_schedule.create())
    opt_state = jax.jit(tx.init, device=optimizer_device)(selected)
    require_cue_optimizer(opt_state)
    step = jax.device_put(jnp.asarray(0, jnp.int32), optimizer_device)
    # EMA与参数使用独立缓冲，允许后续CPU更新复用输入状态。
    ema_params = (jax.tree.map(lambda value: value.copy(), selected)
                  if config.ema_decay is not None else None)
    state = TrainState(step=step, params=params, model_def=graph, tx=tx,
                       opt_state=opt_state, ema_decay=config.ema_decay, ema_params=ema_params)
    return state


def create_training_state(config, policy, *, optimizer_device=None):
    state = _make_training_state(config, policy, optimizer_device=optimizer_device)
    require_training_state_device(state, optimizer_device)
    return state


def abstract_training_state(state, device):
    placement = jax.sharding.SingleDeviceSharding(device)
    return jax.tree.map(lambda value: jax.ShapeDtypeStruct(
        value.shape, value.dtype, sharding=placement), state)


def training_state_template(config, device):
    with jax.default_device(device):
        shaped = jax.eval_shape(lambda: _make_training_state(
            config, config.model.create(jax.random.key(config.seed)), optimizer_device=device))
    return abstract_training_state(shaped, device)


@dataclasses.dataclass(frozen=True)
class MetricBatch:
    observation: object
    actions: object
    identities: tuple


def prepare_batch(dataset, indices, pipeline):
    values = []
    for index in indices:
        index = int(index)
        history = (dataset.get_history(index, history_steps=pipeline.echo.history_steps)
                   if pipeline.echo.enabled else None)
        values.append(pipeline.prepare_train(dataset[index], history=history))
    batched = jax.tree.map(lambda *leaves: np.stack(leaves), *values)
    labels = batched.pop('actions')
    return MetricBatch(model_lib.Observation.from_dict(batched), labels,
                       tuple(dataset.sample_identity(int(index)) for index in indices))


def action_loss(policy, rng, observation, actions):
    """G3 may wrap this objective inside the same NNX differentiated model."""
    return policy.compute_loss(rng, observation, actions, train=True).mean()


class MetricTrainer:
    def __init__(self, config, model_def, *, gradient_device=None, optimizer_device=None,
                 gradient_mesh=None, objective=action_loss):
        self.config = config
        self.gradient_device = gradient_device
        self.optimizer_device = optimizer_device
        self.gradient_mesh = gradient_mesh
        self.parameter_sharding = None
        def backward(params, observation, actions, rng):
            policy = nnx.merge(model_def, params)
            policy.train()
            loss, gradients = nnx.value_and_grad(
                lambda module: objective(module, rng, observation, actions),
                argnums=nnx.DiffState(0, config.trainable_filter))(policy)
            cue = flat_params(gradients)[STEM][..., 3:, :]
            finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(gradients)]))
            info = dict(loss=loss, grad_norm=optax.global_norm(gradients),
                        cue_grad_l2=jnp.linalg.norm(cue), gradients_finite=finite)
            if is_lora(config):
                flat = flat_params(gradients)
                for suffix in ('a', 'b'):
                    info['lora_' + suffix + '_grad_l2'] = optax.global_norm(
                        [value for name, value in flat.items() if name.endswith('lora_' + suffix)])
            return gradients, info
        self._backward = jax.jit(backward)
        def accumulated_backward(params, observation, actions, step):
            gradients, info = accumulated_gradients(config, jax.random.key(config.seed), model_def,
                                                     params, step, (observation, actions))
            cue = flat_params(gradients)[STEM][..., 3:, :]
            info['cue_grad_l2'] = jnp.linalg.norm(cue)
            info['gradients_finite'] = jnp.all(jnp.stack(
                [jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(gradients)]))
            if is_lora(config):
                flat = flat_params(gradients)
                for suffix in ('a', 'b'):
                    info['lora_' + suffix + '_grad_l2'] = optax.global_norm(
                        [value for name, value in flat.items() if name.endswith('lora_' + suffix)])
            return gradients, info
        self._accumulated_backward_fn = accumulated_backward
        self._accumulated_backward = jax.jit(accumulated_backward)
        self._update = jax.jit(lambda state, gradient: apply_gradients(config, state, gradient),
                               device=optimizer_device, donate_argnums=(0, 1))

    def configure_sharding(self, params):
        if self.gradient_mesh is None:
            raise ValueError('Gradient mesh is required for multi-GPU sharding')
        if self.parameter_sharding is not None:
            return
        mesh = self.gradient_mesh
        self.parameter_sharding = sharding.fsdp_sharding(jax.eval_shape(lambda: params), mesh)
        gradient_sharding = self.parameter_sharding.filter(self.config.trainable_filter)
        self.replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        self._accumulated_backward = jax.jit(
            self._accumulated_backward_fn,
            in_shardings=(self.parameter_sharding, self.replicated_sharding,
                          self.replicated_sharding, self.replicated_sharding),
            out_shardings=(gradient_sharding, self.replicated_sharding),
        )

    def step(self, state, batch, *, micro_batch_size=None):
        if micro_batch_size is None:
            rng = jax.random.fold_in(jax.random.key(self.config.seed), state.step)
            params, observation, labels, rng = jax.device_put(
                (state.params, batch.observation, batch.actions, rng), self.gradient_device)
            gradients, metrics = self._backward(params, observation, labels, rng)
        else:
            if (micro_batch_size < 1 or self.config.batch_size % micro_batch_size
                    or batch.actions.shape[0] != self.config.batch_size):
                raise ValueError('Effective batch must equal microbatch times accumulation steps')
            count = self.config.batch_size // micro_batch_size
            def reshape(value):
                return value.reshape((count, micro_batch_size) + value.shape[1:])
            observations = jax.tree.map(reshape, batch.observation)
            labels = reshape(batch.actions)
            if self.gradient_mesh is None:
                params, observations, labels = jax.device_put(
                    (state.params, observations, labels), self.gradient_device)
                gradients, metrics = self._accumulated_backward(
                    params, observations, labels, state.step)
            else:
                self.configure_sharding(state.params)
                params = jax.device_put(state.params, self.parameter_sharding)
                observations = jax.device_put(observations, self.replicated_sharding)
                labels = jax.device_put(labels, self.replicated_sharding)
                step = jax.device_put(state.step, self.replicated_sharding)
                with sharding.set_mesh(self.gradient_mesh):
                    gradients, metrics = self._accumulated_backward(
                        params, observations, labels, step)
                del params
        jax.block_until_ready(gradients)
        info = {key: float(value) for key, value in jax.device_get(metrics).items()}
        if not info['gradients_finite'] or not np.isfinite(info['loss']):
            raise ValueError('Nonfinite action training loss or gradients')
        gradients = jax.device_put(gradients, self.optimizer_device)
        updated = self._update(state, gradients)
        jax.block_until_ready(updated)
        require_training_state_device(updated, self.optimizer_device)
        require_cue_optimizer(updated.opt_state)
        return updated, info


def training_recipe(config):
    return dict(model=dataclasses.asdict(config.model), optimizer=dataclasses.asdict(config.optimizer),
                learning_rate=dataclasses.asdict(config.lr_schedule), seed=config.seed,
                batch_size=config.batch_size, ema_decay=config.ema_decay,
                trainable_policy=('lora_plus_six_channel_stem_and_action_time_projections'
                                  if is_lora(config) else 'full_finetuning_including_visual_patch_kernel'),
                ema_scope='trainable_parameters_only')


def compact_training_state(state, config):
    if not is_lora(config):
        return state
    selected = state.params.filter(config.trainable_filter)
    parameter_contract(selected)
    validate_delta(state.ema_params, selected)
    with at.disable_typechecking():
        return dataclasses.replace(state, params=selected)


def frozen_signatures(state, config):
    return state_signatures(state.params.filter(nnx.Not(config.trainable_filter)))


def save_training_checkpoint(manager, state, pipeline, config, run_manifest):
    class Assets:
        def data_config(self):
            return SimpleNamespace(norm_stats=pipeline.norm_stats, asset_id='physical-intelligence/libero')
    step = int(state.step)
    stored = compact_training_state(state, config)
    checkpoints.save_state(manager, stored, Assets(), step)
    manager.wait_until_finished()
    directory = Path(manager.directory)/str(step)
    (directory/'assets/paligemma_tokenizer.model').write_bytes(pipeline.tokenizer_bytes)
    contract = dict(schema=CHECKPOINT_SCHEMA, completed_updates=step,
                    pipeline=pipeline.contract(), pipeline_sha256=pipeline.fingerprint(),
                    training_recipe=training_recipe(config), run_manifest=run_manifest)
    if is_lora(config):
        contract.update(schema=LORA_CHECKPOINT_SCHEMA, base_reference=base_reference(config, verify_files=False),
                        trainable_parameter_contract=parameter_contract(stored.params),
                        frozen_sha256=frozen_signatures(state, config),
                        delta_state_sha256=state_signatures(stored),
                        inference_sha256=state_signatures(stored.ema_params))
    (directory/'metric_workflow.json').write_text(json.dumps(contract, indent=2, allow_nan=False)+'\n')
    return directory


def checkpoint_restore_arguments(state):
    with at.disable_typechecking():
        train_state, inference_params = checkpoints._split_params(state)
        items = {"train_state": train_state, "params": {"params": inference_params}}
        kwargs = {
            name: {"restore_args": ocp.checkpoint_utils.construct_restore_args(item)}
            for name, item in items.items()
        }
    return items, kwargs


def restore_training_checkpoint(manager, state, pipeline, config, *, step, expected_manifest=None):
    directory = Path(manager.directory)/str(step)
    metadata = json.loads((directory/'metric_workflow.json').read_text())
    schema = LORA_CHECKPOINT_SCHEMA if is_lora(config) else CHECKPOINT_SCHEMA
    if (metadata.get('schema') != schema or metadata['pipeline_sha256'] != pipeline.fingerprint()
            or metadata['training_recipe'] != training_recipe(config)):
        raise ValueError('Checkpoint pipeline/model/optimizer recipe differs from the current six-channel run')
    if metadata['pipeline']['model']['image_channels'] != 6:
        raise ValueError('Three-channel optimizer state cannot resume a six-channel model')
    if expected_manifest is not None:
        for key in ('run_config', 'dataset_manifests', 'loaded_source_sha256',
                    'dependency_lock_sha256'):
            if metadata['run_manifest'].get(key) != expected_manifest.get(key):
                raise ValueError('Checkpoint data/source identity differs: '+key)
    template = compact_training_state(state, config)
    if is_lora(config):
        if (metadata['base_reference'] != base_reference(config)
                or metadata['trainable_parameter_contract'] != parameter_contract(template.params)):
            raise ValueError('LoRA checkpoint base or parameter contract differs')
    with at.disable_typechecking():
        items, restore_kwargs = checkpoint_restore_arguments(template)
        values = manager.restore(step, items=items, restore_kwargs=restore_kwargs)
        restored = checkpoints._merge_params(values["train_state"], values["params"])
    device = jax.tree.leaves(state.params)[0].sharding.device_set
    if len(device) != 1:
        raise ValueError("Checkpoint restore requires one optimizer device")
    if is_lora(config):
        if state_signatures(restored) != metadata['delta_state_sha256']:
            raise ValueError('LoRA delta/AdamW/EMA/step identity changed during restore')
        target = next(iter(device))
        with jax.default_device(target):
            base = jax.device_put(nnx.state(load_initial_model(config)), target)
            validate_delta(restored.params, base.filter(config.trainable_filter))
            base.replace_by_pure_dict(restored.params.to_pure_dict())
        restored = dataclasses.replace(restored, params=base)
        if frozen_signatures(restored, config) != metadata['frozen_sha256']:
            raise ValueError('LoRA frozen backbone differs from the verified base')
    require_training_state_device(restored, next(iter(device)))
    require_trainable_cue(restored.params, config.trainable_filter)
    require_cue_optimizer(restored.opt_state)
    return restored


def pipeline_from_checkpoint(directory):
    directory = Path(directory)
    metadata = json.loads((directory/'metric_workflow.json').read_text())
    if metadata.get('schema') not in (CHECKPOINT_SCHEMA, LORA_CHECKPOINT_SCHEMA):
        raise ValueError('Unsupported metric checkpoint schema')
    contract = metadata['pipeline']
    if contract['model'] != dataclasses.asdict(metric_model_config()):
        raise ValueError('Checkpoint model differs from the validated pi05 six-channel contract')
    ruler = MetricRulerConfig.from_mapping(contract['metric_ruler'])
    if (contract['motion_echo']['enabled']
            and contract.get('motion_echo_rendering') != ECHO_RENDER_CONTRACT):
        raise ValueError('Checkpoint Echo rendering differs from the current input contract')
    # Norm statistics are saved by OpenPI beside inference params; never use the caller's mutable stats.
    stats = normalize.load(directory/'assets/physical-intelligence/libero')
    tokenizer = tokenizer_from_asset(directory/'assets/paligemma_tokenizer.model', contract['tokenizer']['max_len'])
    pipeline = MetricPipeline(contract['geometry'], ruler, stats,
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
        self.seed = seed
        self.rng = jax.random.key(seed)
        self.online_history = MetricInferenceHistory(pipeline.echo) if pipeline.echo.enabled else None

    def reset_episode(self, episode_index):
        self.rng = jax.random.fold_in(jax.random.key(self.seed), int(episode_index))
        if self.online_history is not None:
            self.online_history.reset_episode(episode_index)

    def observe_control_step(self, sample):
        if self.online_history is not None:
            self.online_history.observe_sample(sample)

    def observe_observation(self, observation, **synchronized_calibration):
        from rulervla.adapters.libero import LiberoAdapter
        sample = LiberoAdapter(self.pipeline.geometry).from_metric_observation(
            observation, **synchronized_calibration)
        self.observe_control_step(sample)

    def prepare(self, sample):
        history = self.online_history.for_sample(sample) if self.online_history is not None else None
        return self.pipeline.prepare_inference(sample, history=history)

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


def load_metric_model(directory, *, explicit_pipeline=None, seed=42):
    directory = Path(directory)
    if (directory/'metric_workflow.json').exists():
        pipeline, metadata = pipeline_from_checkpoint(directory)
        if explicit_pipeline is not None and explicit_pipeline.fingerprint() != pipeline.fingerprint():
            raise ValueError('Caller preprocessing differs from checkpoint preprocessing')
    else:
        raise ValueError('A checkpoint with saved preprocessing is required')
    params = model_lib.restore_params(directory/'params', restore_type=np.ndarray)
    model_config = pipeline.model_config
    if metadata['schema'] == LORA_CHECKPOINT_SCHEMA:
        model_config = metric_model_config(finetuning='lora')
        if metadata['training_recipe']['model'] != dataclasses.asdict(model_config):
            raise ValueError('LoRA inference model recipe differs')
        reference = verify_base_reference(metadata['base_reference'])
        shaped = nnx.state(nnx.eval_shape(model_config.create, jax.random.key(seed))).to_pure_dict()
        loader = weight_loaders.SixChannelCheckpointWeightLoader(
            reference['params_path'], missing_regex=r'.*lora_[ab]')
        full = initialize_lora_parameters(loader.load(shaped), seed)
        base = nnx.state(model_config.load(full, remove_extra_params=False))
        selected = base.filter(nnx.Not(lora_freeze_filter()))
        validate_delta(params, selected)
        if parameter_contract(selected) != metadata['trainable_parameter_contract']:
            raise ValueError('LoRA inference parameter contract differs')
        if state_signatures(base.filter(lora_freeze_filter())) != metadata['frozen_sha256']:
            raise ValueError('LoRA inference frozen backbone differs from the checkpoint')
        base.replace_by_pure_dict(params)
        if state_signatures(base.filter(nnx.Not(lora_freeze_filter()))) != metadata['inference_sha256']:
            raise ValueError('LoRA inference EMA delta identity differs')
        params = base.to_pure_dict()
    policy = model_config.load(params, remove_extra_params=False)
    return policy, pipeline


def load_metric_policy(directory, *, explicit_pipeline=None, sampling_steps=10, seed=42):
    policy, pipeline = load_metric_model(directory, explicit_pipeline=explicit_pipeline, seed=seed)
    return MetricPolicy(policy, pipeline, sampling_steps=sampling_steps, seed=seed)


def state_signatures(state):
    result = []
    for path, value in jax.tree_util.tree_flatten_with_path(state)[0]:
        array = np.asarray(jax.device_get(value))
        result.append(dict(path=jax.tree_util.keystr(path), shape=list(array.shape), dtype=str(array.dtype),
                           sha256=hashlib.sha256(array.reshape(-1).view(np.uint8)).hexdigest()))
    return result


def source_provenance(root):
    """Resolve imported code and capture dirty state without altering other work."""
    root = Path(root).resolve()
    tracked = set(subprocess.check_output(['git', 'ls-files'], cwd=root, text=True).splitlines())
    changed = set(subprocess.check_output(['git', 'diff', '--name-only', 'HEAD'], cwd=root, text=True).splitlines())
    loaded = {}
    runtime_modules = dict(sys.modules)
    runtime_modules[model_lib.__name__] = model_lib
    runtime_modules[image_tools.__name__] = image_tools
    for name, module in runtime_modules.items():
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


def gpu_prediction(config, params, batch, device, *, mesh=None):
    observation = jax.tree.map(lambda value: value[:1], batch.observation)
    graph = nnx.split(nnx.eval_shape(config.model.create, jax.random.key(config.seed)))[0]
    def predict(current, inputs):
        policy = nnx.merge(graph, current)
        policy.eval()
        return policy.sample_actions(jax.random.key(123), inputs, num_steps=10)
    if mesh is None:
        values = jax.jit(predict)(jax.device_put(params, device), jax.device_put(observation, device))
    else:
        param_sharding = sharding.fsdp_sharding(jax.eval_shape(lambda: params), mesh)
        replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        compiled = jax.jit(predict, in_shardings=(param_sharding, replicated),
                           out_shardings=replicated)
        with sharding.set_mesh(mesh):
            values = compiled(jax.device_put(params, param_sharding),
                              jax.device_put(observation, replicated))
    result = np.asarray(jax.block_until_ready(values))
    if result.shape != (1, 10, 32) or not np.isfinite(result).all():
        raise ValueError('Restored LoRA inference did not produce finite action chunks')
    return dict(shape=list(result.shape), finite=True,
                sha256=hashlib.sha256(result.tobytes()).hexdigest())


def gpu_equivalence(config, params6, batch, device, tolerances, *, mesh=None):
    """Measure on the actual GPU; CPU exactness is not an assumed GPU result."""
    import gc
    # 等价检查只需要真实样本中的一条，避免把完整有效批次同时送入视觉与动作推理。
    observation = jax.tree.map(lambda value: value[:1], batch.observation)
    zero = dataclasses.replace(observation, images={
        key: np.concatenate((np.asarray(image)[..., :3], np.zeros_like(np.asarray(image)[..., 3:])), axis=-1)
        for key, image in observation.images.items()})
    rgb = dataclasses.replace(observation, images={key: value[..., :3] for key, value in zero.images.items()})
    noise = np.asarray(jax.random.normal(jax.random.key(77), batch.actions[:1].shape))
    outputs = []
    for channels, obs in ((3, rgb), (6, zero)):
        cfg = dataclasses.replace(config.model, image_channels=channels)
        if channels == 3:
            graph = nnx.split(nnx.eval_shape(cfg.create, jax.random.key(42)))[0]
            state = rgb_state_from_six(params6)
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
        if mesh is None:
            compiled = jax.jit(evaluate)
            values = compiled(jax.device_put(state, device), jax.device_put(obs, device))
        else:
            shapes = jax.tree.map(lambda value: jax.ShapeDtypeStruct(value.shape, value.dtype), state)
            param_sharding = sharding.fsdp_sharding(shapes, mesh)
            replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
            compiled = jax.jit(evaluate, in_shardings=(param_sharding, replicated),
                               out_shardings=(replicated, replicated))
            with sharding.set_mesh(mesh):
                values = compiled(jax.device_put(state, param_sharding),
                                  jax.device_put(obs, replicated))
        values = jax.block_until_ready(values)
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
