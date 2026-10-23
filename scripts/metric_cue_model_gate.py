"""G1: real pi05 checkpoint, action-loss gradient, optimizer and recovery gate.

This performs one diagnostic update on one observation. It is not a training run.
"""
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import flax.nnx as nnx
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
STEM = 'PaliGemma/img/embedding/kernel'


def build_gate_config(checkpoint):
    from openpi.models.pi0_config import Pi0Config
    from openpi.training import config, optimizer, weight_loaders
    policy = Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False, image_channels=6)
    return dataclasses.replace(
        config.get_config('pi05_libero'), model=policy, batch_size=1,
        freeze_filter=policy.get_freeze_filter(), ema_decay=None, wandb_enabled=False,
        weight_loader=weight_loaders.SixChannelCheckpointWeightLoader(str(checkpoint), missing_regex='(?!)'),
        optimizer=optimizer.AdamW(), seed=42,
    )


def flat_params(params):
    if isinstance(params, nnx.State):
        params = params.to_pure_dict()
    return flax.traverse_util.flatten_dict(params, sep='/')


def require_trainable_stem(params, trainable_filter):
    selected = params.filter(trainable_filter)
    flat = flat_params(selected)
    if STEM not in flat:
        raise ValueError('Six-channel visual patch kernel is not trainable')
    if flat[STEM].ndim != 4 or flat[STEM].shape[2] != 6:
        raise ValueError('Trainable visual patch kernel must have six input channels')
    return selected


def require_optimizer_stem(state):
    """Check actual Adam moments, rather than only the optimizer configuration."""
    found = []
    def visit(value):
        if hasattr(value, 'mu') and hasattr(value, 'nu'):
            for name in ('mu', 'nu'):
                kernel = flat_params(getattr(value, name)).get(STEM)
                if kernel is not None and kernel.ndim == 4 and kernel.shape[2] == 6:
                    found.append(name)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
    visit(state)
    if sorted(found) != ['mu', 'nu']:
        raise ValueError('Cue kernel is missing from optimizer Adam moment state')
    return sorted(found)


def tensor_hash(value):
    array = np.asarray(jax.device_get(value))
    return hashlib.sha256(array.reshape(-1).view(np.uint8)).hexdigest()


def signatures(tree):
    return [{'path': jax.tree_util.keystr(path), 'shape': list(value.shape),
             'dtype': str(value.dtype), 'sha256': tensor_hash(value)}
            for path, value in jax.tree_util.tree_flatten_with_path(tree)[0]]


def comparison(actual, expected, *, atol, rtol):
    actual, expected = np.asarray(actual, np.float32), np.asarray(expected, np.float32)
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError('Nonfinite model outputs')
    difference = actual - expected
    result = {'max_abs': float(np.max(np.abs(difference))),
              'rms': float(np.sqrt(np.mean(difference**2))), 'atol': atol, 'rtol': rtol}
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)
    return result


def prepare_observation(args, config):
    from openpi import transforms
    from openpi.models import model
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.shared import normalize
    from rulervla.adapters.metric_cue import MetricViewAdapter, SixChannelInputs
    from rulervla.config import load_metric_method_config
    from rulervla.cues.metric_prompt import StrictPromptTokenizer
    with np.load(args.sample, allow_pickle=False) as data:
        sample = {key: data[key] for key in data.files}
    sample['task'] = str(sample['task'])
    # The saved fixture contains one demonstrated action. Repeat it only to
    # exercise the real horizon-10 FM objective; never claim a real future chunk.
    sample['actions'] = np.repeat(sample['actions'][None], config.model.action_horizon, axis=0)
    geometry, ruler, echo = load_metric_method_config(args.config)
    if echo.enabled:
        raise ValueError('G1 requires pure Ruler with Echo disabled')
    prepared = MetricViewAdapter(geometry, ruler, echo).prepare(sample)
    structured = SixChannelInputs(model_type=config.model.model_type)(prepared)
    stats = normalize.load(args.norm_stats)
    structured = transforms.Normalize(stats, use_quantiles=True, strict=True)(structured)
    tokenizer = StrictPromptTokenizer(PaligemmaTokenizer(config.model.max_token_len))
    structured = transforms.TokenizePrompt(tokenizer, discrete_state_input=False)(structured)
    structured = transforms.PadStatesAndActions(config.model.action_dim)(structured)
    batch = jax.tree.map(lambda value: jnp.asarray(value)[None], structured)
    observation = model.Observation.from_dict(batch)
    zero = dataclasses.replace(observation, images={
        key: image.at[..., 3:].set(0) for key, image in observation.images.items()})
    rgb = dataclasses.replace(observation, images={
        key: image[..., :3] for key, image in observation.images.items()})
    return observation, zero, rgb, batch['actions'], {
        'episode': int(sample['episode_index']), 'frame': int(sample['frame_index']),
        'token_count': int(observation.tokenized_prompt_mask.sum()),
        'action_labels': 'repeat current demonstrated action 10 times; diagnostic FM target, not a future chunk',
        'normalization': str(args.norm_stats.resolve()),
        'normalization_sha256': hashlib.sha256((args.norm_stats/'norm_stats.json').read_bytes()).hexdigest(),
        'sample_sha256': hashlib.sha256(args.sample.read_bytes()).hexdigest(),
        'ruler': dataclasses.asdict(ruler), 'echo_enabled': False,
    }, stats


def run(args):
    from openpi.models import model
    from openpi.training import checkpoints
    from openpi.training.utils import TrainState
    from openpi.training.weight_loaders import merge_six_channel_params
    from rulervla.training.accumulation import apply_gradients
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    report = {'state': 'running', 'formal_training_started': False, 'optimizer_updates': 0,
              'checkpoint': str(args.checkpoint), 'backend': str(jax.devices()),
              'model_source': str(Path(model.__file__).resolve()), 'seed': 42,
              'sampling_steps': args.sampling_steps, 'gate_learning_rate': args.learning_rate,
              'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    def stage(name, **values):
        report.update(stage=name, elapsed_seconds=time.time()-started, **values)
        (args.out/'status.json').write_text(json.dumps(report, indent=2))
        print(json.dumps({'stage':name,'seconds':round(time.time()-started,2),**values}), flush=True)
    try:
        if str(ROOT/'src/openpi') not in str(Path(model.__file__).resolve()):
            raise ValueError('Wrong OpenPI runtime source')
        (args.out/'git.diff').write_bytes(subprocess.check_output(['git','diff','--binary'],cwd=ROOT))
        (args.out/'git_status.txt').write_bytes(subprocess.check_output(['git','status','--porcelain=v1'],cwd=ROOT))
        sources = sorted(p for folder in ('src/openpi', 'src/rulervla') for p in (ROOT/folder).rglob('*.py'))
        sources += [Path(__file__), args.config]
        source_manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
        (args.out/'source_sha256.json').write_text(json.dumps(source_manifest,indent=2))
        config = build_gate_config(args.checkpoint)
        observation, zero_observation, rgb_observation, actions, input_info, norm_stats = prepare_observation(args, config)
        stage('loading_real_checkpoint', input=input_info)
        old_config = dataclasses.replace(config.model, image_channels=3)
        abstract_model = nnx.eval_shape(config.model.create, jax.random.key(42))
        graph, abstract_params = nnx.split(abstract_model)
        reference = abstract_params.to_pure_dict()
        # Use the actual strict loader that a new six-channel training run uses.
        params6 = config.weight_loader.load(reference)
        stage('six_channel_checkpoint_loaded')
        params3 = model.restore_params(args.checkpoint, restore_type=np.ndarray)
        old_reference = nnx.state(nnx.eval_shape(old_config.create, jax.random.key(42))).to_pure_dict()
        params3 = merge_six_channel_params(params3, old_reference, missing_regex='(?!)')
        old_flat, new_flat = flat_params(params3), flat_params(params6)
        if set(old_flat) != set(new_flat):
            raise ValueError('Parameter sets changed during migration')
        for path, original in old_flat.items():
            actual = new_flat[path][..., :3, :] if path == STEM else new_flat[path]
            if tensor_hash(original) != tensor_hash(actual):
                raise ValueError(f'Original checkpoint parameter changed: {path}')
        if np.any(new_flat[STEM][..., 3:, :] != 0):
            raise ValueError('New kernel channels were not initialized to zero')
        stage('migration_verified', parameter_leaves=len(new_flat),
              parameter_count=sum(int(np.prod(x.shape)) for x in new_flat.values()),
              old_kernel_shape=list(old_flat[STEM].shape), new_kernel_shape=list(new_flat[STEM].shape),
              original_parameters_exact=True, new_kernel_exact_zero=True)
        policy3 = old_config.load(params3, remove_extra_params=False)
        policy6 = config.model.load(params6, remove_extra_params=False)
        graph3, state3 = nnx.split(policy3)
        graph6, state6 = nnx.split(policy6)
        state3, state6 = jax.device_put(state3), jax.device_put(state6)
        del policy3, policy6, params3, params6, old_flat, new_flat
        noise = jax.random.normal(jax.random.key(77), actions.shape)
        rng = jax.random.key(123)
        def evaluator(graphdef):
            def evaluate(params, obs):
                policy = nnx.merge(graphdef, params)
                policy.eval()
                prefix, _, _ = policy.embed_prefix(obs)
                sampled = policy.sample_actions(rng, obs, noise=noise, num_steps=args.sampling_steps)
                return prefix, sampled
            return jax.jit(evaluate)
        evaluate3, evaluate6 = evaluator(graph3), evaluator(graph6)
        stage('three_channel_complete_forward')
        prefix3, predicted3 = jax.block_until_ready(evaluate3(state3, rgb_observation))
        stage('six_channel_zero_cue_complete_forward')
        prefix6, predicted6 = jax.block_until_ready(evaluate6(state6, zero_observation))
        np.savez_compressed(args.out/'zero_cue_forward_outputs.npz',
                            prefix3=np.asarray(prefix3,np.float32), prefix6=np.asarray(prefix6,np.float32),
                            actions3=np.asarray(predicted3), actions6=np.asarray(predicted6))
        report['zero_cue_equivalence'] = {
            'vision_prefix': comparison(prefix6, prefix3, atol=args.atol, rtol=args.rtol),
            'actions': comparison(predicted6, predicted3, atol=args.atol, rtol=args.rtol)}
        stage('six_channel_nonzero_cue_initial_forward', zero_cue_equivalence=report['zero_cue_equivalence'])
        prefix_initial, prediction_initial = jax.block_until_ready(evaluate6(state6, observation))
        report['zero_initialized_nonzero_cue_equivalence'] = {
            'vision_prefix': comparison(prefix_initial, prefix3, atol=args.atol, rtol=args.rtol),
            'actions': comparison(prediction_initial, predicted3, atol=args.atol, rtol=args.rtol)}
        del state3, prefix3, prefix6, prefix_initial, predicted3, predicted6, evaluate3
        selected = require_trainable_stem(state6, config.trainable_filter)
        trainable_flat = flat_params(selected)
        report['trainability'] = {
            'recipe': 'pi05_libero full fine-tuning; explicit Pi0Config.get_freeze_filter; EMA disabled for G1',
            'stem_in_trainable_filter': True,
            'frozen_parameter_count': sum(int(np.prod(x.shape)) for x in jax.tree.leaves(
                state6.filter(nnx.All(nnx.Param, config.freeze_filter)))),
            'trainable_parameter_count': sum(int(np.prod(x.shape)) for x in trainable_flat.values()),
            'vision_trainable_parameter_count': sum(int(np.prod(x.shape)) for k,x in trainable_flat.items() if k.startswith('PaliGemma/img/'))}
        stage('full_action_loss_backward', trainability=report['trainability'])
        def loss_and_gradient(params, obs, labels):
            policy = nnx.merge(graph6, params)
            policy.train()
            def loss_fn(module):
                return module.compute_loss(rng, obs, labels, train=True).mean()
            return nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, config.trainable_filter))(policy)
        backward = jax.jit(loss_and_gradient)
        loss, gradients = jax.block_until_ready(backward(state6, observation, actions))
        kernel_gradient = np.asarray(flat_params(gradients)[STEM])
        cue_gradient = kernel_gradient[..., 3:, :]
        gradient_finite = all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(gradients))
        if not gradient_finite or not np.isfinite(float(loss)) or not np.any(cue_gradient != 0):
            raise ValueError('Action loss did not produce finite nonzero Cue-kernel gradients')
        report['backward'] = {
            'action_fm_loss': float(loss), 'all_gradients_finite': gradient_finite,
            'cue_kernel_gradient_l2': float(np.linalg.norm(cue_gradient)),
            'cue_kernel_gradient_max_abs': float(np.max(np.abs(cue_gradient))),
            'cue_kernel_gradient_nonzero_count': int(np.count_nonzero(cue_gradient)),
            'gradient_norm': float(optax.global_norm(gradients))}
        stage('fresh_optimizer_initialization', backward=report['backward'])
        tx = config.optimizer.create(args.learning_rate)
        opt_state = jax.jit(tx.init)(selected)
        jax.block_until_ready(opt_state)
        report['optimizer_stem_moments'] = require_optimizer_stem(opt_state)
        state = TrainState(step=jnp.asarray(0), params=state6, model_def=graph6,
                           opt_state=opt_state, tx=tx, ema_decay=None)
        stage('one_optimizer_update')
        update = jax.jit(lambda current, gradient: apply_gradients(config, current, gradient))
        state = jax.block_until_ready(update(state, gradients))
        learned_cue = np.asarray(flat_params(state.params)[STEM])[..., 3:, :]
        if not np.isfinite(learned_cue).all() or not np.any(learned_cue != 0):
            raise ValueError('Optimizer failed to update Cue weights')
        report['updated_cue_kernel'] = {'l2': float(np.linalg.norm(learned_cue)),
            'max_abs':float(np.max(np.abs(learned_cue))), 'sha256':tensor_hash(learned_cue)}
        report['optimizer_updates'] = int(state.step)
        require_optimizer_stem(state.opt_state)
        stage('saving_full_six_channel_training_state', updated_cue_kernel=report['updated_cue_kernel'])
        expected_signatures = signatures(state)
        (args.out/'expected_state_sha256.json').write_text(json.dumps(expected_signatures,indent=2))
        class Assets:
            def data_config(self):
                return SimpleNamespace(norm_stats=norm_stats, asset_id='physical-intelligence/libero')
        manager, _ = checkpoints.initialize_checkpoint_dir(args.out/'checkpoints', keep_period=None,
                                                           overwrite=False, resume=False)
        checkpoints.save_state(manager, state, Assets(), 1)
        manager.wait_until_finished()
        stage('restoring_full_training_state')
        restored = checkpoints.restore_state(manager, state, Assets(), step=1)
        jax.block_until_ready(restored)
        if signatures(restored) != expected_signatures:
            raise ValueError('Restored full parameters/optimizer state are not bitwise equal')
        require_trainable_stem(restored.params, config.trainable_filter)
        require_optimizer_stem(restored.opt_state)
        stage('restored_complete_action_forward', full_state_restore_bitwise=True)
        prefix_restored, prediction_restored = jax.block_until_ready(evaluate6(restored.params, observation))
        prefix_updated, prediction_updated = jax.block_until_ready(evaluate6(state.params, observation))
        report['recovered_prediction_equivalence'] = comparison(
            prediction_restored, prediction_updated, atol=0, rtol=0)
        prefix_blank, prediction_blank = jax.block_until_ready(evaluate6(restored.params, zero_observation))
        vision_effect = float(np.max(np.abs(np.asarray(prefix_restored,np.float32)-np.asarray(prefix_blank,np.float32))))
        action_effect = float(np.max(np.abs(np.asarray(prediction_restored)-np.asarray(prediction_blank))))
        if vision_effect == 0 or action_effect == 0:
            raise ValueError('Learned Cue does not affect restored vision/action outputs')
        report['restored_cue_effect'] = {'vision_prefix_max_abs':vision_effect,
                                       'actions_max_abs':action_effect}
        report.update(state='complete', full_state_restore_bitwise=True,
                      checkpoint_saved=str(args.out/'checkpoints/1'), all_g1_checks_passed=True)
        manager.close()
        stage('g1_complete', restored_cue_effect=report['restored_cue_effect'])
    except BaseException as error:
        stage('failed', state='failed', error=repr(error))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--sample',type=Path,required=True)
    parser.add_argument('--norm-stats',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/rulervla_metric_cue.yaml')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--learning-rate',type=float,default=1e-4)
    parser.add_argument('--sampling-steps',type=int,default=10)
    parser.add_argument('--atol',type=float,default=1e-2)
    parser.add_argument('--rtol',type=float,default=1e-2)
    run(parser.parse_args())


if __name__=='__main__':
    main()
