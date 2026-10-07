"""Loaded official full pi05 GPU smoke with effective batch 32; no formal run."""
import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def main(supplemental=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--v03-root', type=Path, required=True)
    parser.add_argument('--rgbd-root', type=Path, required=True)
    parser.add_argument('--geometry-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    choices = (['CameraXYZ-only', 'Ruler-only', 'ActionCoord-only'] if supplemental
        else ['Baseline', 'AimBot', 'AxisGuide-adapted', 'RulerVLA-main'])
    parser.add_argument('--method', choices=choices, required=True)
    parser.add_argument('--aimbot-source', type=Path, required=True)
    parser.add_argument('--axisguide-source', type=Path, required=True)
    parser.add_argument('--micro-batch', type=int, default=1)
    parser.add_argument('--accumulation', type=int, default=32)
    parser.add_argument('--fsdp-devices', type=int, default=2)
    parser.add_argument('--save-restore', action='store_true')
    parser.add_argument('--cpu-optimizer', action='store_true')
    args = parser.parse_args()
    if supplemental:
        assert args.fsdp_devices == 1 and args.cpu_optimizer and args.save_restore
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'src'))
    sys.path.insert(0, str(args.runtime / 'packages/openpi-client/src'))
    sys.path.insert(0, str(args.runtime / 'src'))
    sys.path.insert(0, str(args.v03_root / 'repo/actioncoord_experiments/src'))
    args.out.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (args.out / '.run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = {'state': 'running', 'stage': 'importing_runtime', 'pid': os.getpid(), 'method': args.method,
        'gpu_ids': os.environ.get('CUDA_VISIBLE_DEVICES'), 'optimizer_updates': 0,
        'micro_batch': args.micro_batch, 'gradient_accumulation': args.accumulation,
        'effective_batch': args.micro_batch * args.accumulation, 'fsdp_devices': args.fsdp_devices,
        'optimizer_and_ema_on_cpu': args.cpu_optimizer,
        'started_at_unix': time.time(), 'formal_training_started': False}
    write_json(args.out / 'status.json', status)
    try:
        import flax.nnx as nnx
        import jax
        import jax.numpy as jnp
        from openpi import transforms
        from openpi.models import model
        from openpi.models.pi0_config import Pi0Config
        from openpi.models.tokenizer import PaligemmaTokenizer
        from openpi.shared import normalize
        from openpi.training import config, optimizer, sharding, weight_loaders
        from rulervla.adapters.main_comparison import MainComparisonInputs, MAIN_IMAGE_KEYS
        from rulervla.adapters.main_cues import MainCueAdapter
        from rulervla.config import load_config
        from rulervla.training.accumulation import accumulated_gradients, apply_gradients
        assert status['effective_batch'] == 32
        assert jax.device_count() == args.fsdp_devices
        status['devices'] = [str(device) for device in jax.devices()]
        status.update(stage='checking_gpu_collectives', nccl_transport={key: os.environ.get(key) for key in
            ('NCCL_P2P_DISABLE', 'NCCL_IB_DISABLE', 'NCCL_SHM_DISABLE', 'NCCL_SOCKET_IFNAME', 'NCCL_CUMEM_HOST_ENABLE')})
        write_json(args.out/'status.json', status)
        mesh = sharding.make_mesh(args.fsdp_devices)
        replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        weight_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('fsdp'))
        collective = jax.jit(lambda value: value.sum(axis=0),
            in_shardings=weight_sharding, out_shardings=replicated)
        warmup = np.arange(16, dtype=np.float32).reshape(4,4)
        np.testing.assert_array_equal(jax.device_get(collective(warmup)), warmup.sum(axis=0))
        status['gpu_collective_passed'] = True
        base_params = '/home/Dataset2/yqt/openpi/checkpoints/Official/pi05_base/params'
        policy_config = Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False)
        training_config = dataclasses.replace(config.get_config('pi05_libero'), model=policy_config,
            batch_size=32, seed=42, num_train_steps=30000, fsdp_devices=args.fsdp_devices,
            optimizer=optimizer.AdamW(b1=.9, b2=.95, eps=1e-8, weight_decay=1e-10, clip_gradient_norm=1),
            lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=1000, peak_lr=5e-5,
                decay_steps=30000, decay_lr=5e-6), ema_decay=.999,
            weight_loader=weight_loaders.CheckpointWeightLoader(base_params), wandb_enabled=False)
        status.update(stage='preparing_real_batch', initial_checkpoint=base_params,
            full_finetuning=True, lora=False, supervised_components=7, action_horizon=10)
        write_json(args.out / 'status.json', status)
        rgbd_manifest = json.loads((args.rgbd_root / 'regen_manifest.json').read_text())
        geometry_manifest = json.loads((args.geometry_root / 'manifest.json').read_text())
        assert geometry_manifest['complete'] and len(geometry_manifest['episodes']) == 432
        assert geometry_manifest['total_frames'] == 52970
        rows = sorted(rgbd_manifest['episodes'], key=lambda row: row['episode_index'])
        geometry = {row['episode_index']: row for row in geometry_manifest['episodes']}
        # Compute once from all training frames, shared by all methods. Chunk
        # statistics exactly weight the labels used at every frame, including
        # repeat-last padding within each episode. No evaluation scenes enter.
        stats_dir = (root/'outputs/main_comparison/gpu_smoke_20261006/shared_spatial_norm'
            if supplemental else args.out.parent / 'shared_spatial_norm')
        if supplemental:
            assert (stats_dir/'norm_stats.json').exists(), 'Reuse frozen norm; never regenerate'
        if not (stats_dir / 'norm_stats.json').exists():
            state_stats, action_stats = normalize.RunningStats(), normalize.RunningStats()
            for row in rows:
                with np.load(geometry[row['episode_index']]['geometry_npz']) as z:
                    state_stats.update(z['state8'])
                with np.load(row['npz']) as z:
                    actions = z['actions']
                    indices = np.minimum(np.arange(len(actions))[:, None] + np.arange(10), len(actions)-1)
                    action_stats.update(actions[indices])
            normalize.save(stats_dir, {'state': state_stats.get_statistics(), 'actions': action_stats.get_statistics()})
            write_json(stats_dir / 'source.json', {'episodes': 432, 'frames': 52970,
                'rgbd_manifest_sha256': hashlib.sha256((args.rgbd_root/'regen_manifest.json').read_bytes()).hexdigest(),
                'geometry_manifest_sha256': hashlib.sha256((args.geometry_root/'manifest.json').read_bytes()).hexdigest(),
                'sample_policy': 'all training frames; action horizon repeat-last within episode'})
        norm_stats = normalize.load(stats_dir)
        status['norm_stats_sha256'] = hashlib.sha256((stats_dir/'norm_stats.json').read_bytes()).hexdigest()
        if supplemental:
            from rulervla.adapters.small_ablation import SmallAblationAdapter
            cue = SmallAblationAdapter(load_config(root/'configs/rulervla_main.yaml'), args.method)
        else:
            cue = MainCueAdapter(load_config(root/'configs/rulervla_main.yaml'), args.method,
                aimbot_source=args.aimbot_source, axisguide_source=args.axisguide_source)
        transform = transforms.compose([MainComparisonInputs(model_type=policy_config.model_type),
            transforms.Normalize(norm_stats, use_quantiles=True, strict=True),
            transforms.TokenizePrompt(PaligemmaTokenizer(policy_config.max_token_len), discrete_state_input=False),
            transforms.PadStatesAndActions(32)])
        row = rows[0]
        with np.load(row['npz']) as rgbd, np.load(geometry[row['episode_index']]['geometry_npz']) as geo:
            entries = []
            from PIL import Image
            for frame in range(32):
                sample = {key: np.asarray(rgbd[key][frame]).copy() for key in
                    ('agentview_rgb', 'wrist_rgb', 'agentview_depth_m', 'ee_world_m')}
                sample.update({key: np.asarray(geo[key][frame]).copy() for key in
                    ('state8', 'eef_quaternion_xyzw', 'wrist_depth_m', 'intrinsic', 'camera_to_world',
                     'wrist_intrinsic', 'wrist_camera_to_world')})
                indices = np.minimum(frame + np.arange(10), row['frames']-1)
                sample.update(actions=np.asarray(rgbd['actions'][indices], np.float32), task=row['task'])
                prepared = cue.prepare(sample)
                if frame == 0:
                    Image.fromarray(prepared['observation/image']).save(args.out/'agent_first.png')
                    Image.fromarray(prepared['observation/wrist_image']).save(args.out/'wrist_first.png')
                    for key, name in [('observation/spatial_image', 'agent_aux_first'), ('observation/wrist_cue', 'wrist_aux_first')]:
                        Image.fromarray(np.round((prepared[key]+1)*127.5).astype(np.uint8)).save(args.out/(name+'.png'))
                entries.append(transform(prepared))
        data = jax.tree.map(lambda *values: np.stack(values).reshape((args.accumulation, args.micro_batch)+values[0].shape), *entries)
        batch = (model.Observation.from_dict(data), data['actions'])
        assert set(batch[0].images) == set(MAIN_IMAGE_KEYS)
        assert batch[1].shape == (args.accumulation, args.micro_batch, 10, 32)
        mesh = sharding.make_mesh(args.fsdp_devices)
        replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        batch_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, sharding.BATCH_AXIS))
        spec = importlib.util.spec_from_file_location('main_stock_train', args.runtime/'scripts/train.py')
        train = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(train)
        status.update(stage='loading_pi05_base')
        write_json(args.out/'status.json', status)
        optimizer_mesh = mesh
        init_rng = jax.random.key(42)
        if args.cpu_optimizer:
            optimizer_mesh = jax.sharding.Mesh(np.array(jax.devices('cpu')).reshape(1,1), ('batch', 'fsdp'))
            init_rng = jax.device_put(init_rng, jax.sharding.NamedSharding(optimizer_mesh, jax.sharding.PartitionSpec()))
        state, state_sharding = train.init_train_state(training_config, init_rng, optimizer_mesh, resume=False)
        jax.block_until_ready(state)
        assert not jax.tree.leaves(state.params.filter(training_config.freeze_filter))
        assert len(jax.tree.leaves(state.params.filter(training_config.trainable_filter))) == len(jax.tree.leaves(state.params))
        assert all(not isinstance(value, jax.ShapeDtypeStruct) for value in jax.tree.leaves(state.params))
        assert not any('lora' in jax.tree_util.keystr(path).lower() for path, _ in jax.tree_util.tree_flatten_with_path(state.params)[0])
        status.update(stage='compiling_forward_backward', official_weights_loaded=True,
            parameter_count=int(sum(np.prod(value.shape) for value in jax.tree.leaves(state.params))),
            four_real_input_slots=True, batch_source_episode=row['episode_index'], batch_source_frames=list(range(32)))
        write_json(args.out/'status.json', status)
        graph = state.model_def
        gradient_sharding = sharding.fsdp_sharding(jax.eval_shape(lambda: state.params), mesh)
        gradient = jax.jit(lambda rng, step, params, values:
            accumulated_gradients(training_config, rng, graph, params, step, values),
            in_shardings=(replicated, replicated, gradient_sharding, batch_sharding),
            out_shardings=(gradient_sharding, replicated))
        update = jax.jit(lambda current, gradients: apply_gradients(training_config, current, gradients),
            in_shardings=(state_sharding, state_sharding.params), out_shardings=state_sharding,
            donate_argnums=(0, 1))
        batch = jax.device_put(batch, batch_sharding)
        metrics = []
        with sharding.set_mesh(mesh):
            for index in range(2):
                started = time.time()
                gpu_params = jax.device_put(state.params, gradient_sharding)
                gradients, info = gradient(jax.random.key(42), jax.device_put(state.step, replicated), gpu_params, batch)
                jax.block_until_ready(gradients)
                if args.cpu_optimizer:
                    gradients_cpu = jax.device_put(gradients, state_sharding.params)
                    jax.block_until_ready(gradients_cpu)
                    del gradients, gpu_params
                    state = update(state, gradients_cpu)
                else:
                    del gpu_params
                    state = update(state, gradients)
                jax.block_until_ready(state)
                info = {key: float(value) for key, value in jax.device_get(info).items()}
                assert all(np.isfinite(value) for value in info.values()) and info['grad_norm'] > 0
                assert int(jax.device_get(state.step)) == index+1
                metrics.append(dict(update=index+1, seconds=time.time()-started, **info))
                status.update(stage='backward_running', optimizer_updates=index+1, metrics=metrics)
                write_json(args.out/'status.json', status)
                print(json.dumps(metrics[-1]), flush=True)
        if args.save_restore:
            if supplemental:
                save_lock = (root/'outputs/main_comparison/checkpoint_staging.lock').open('w')
                fcntl.flock(save_lock, fcntl.LOCK_EX)
            status['stage'] = 'saving_checkpoint'
            write_json(args.out/'status.json', status)
            from openpi.training import checkpoints
            from actioncoord.integration.v03_checkpoint_snapshot import host_snapshot
            from actioncoord.integration.v03_storage import publish_checkpoint, verify_checkpoint
            stage = Path('/dev/shm/rulervla_main_smoke_%d' % os.getpid())
            if stage.exists():
                raise FileExistsError(stage)
            snapshot, snapshot_evidence = host_snapshot(state)
            write_json(args.out/'snapshot.json', snapshot_evidence)
            manager, _ = checkpoints.initialize_checkpoint_dir(stage, keep_period=None, overwrite=False, resume=False)
            class LoaderAssets:
                def data_config(self):
                    return config.DataConfig(asset_id='physical-intelligence/libero', norm_stats=norm_stats)
            checkpoints.save_state(manager, snapshot, LoaderAssets(), 1)
            manager.wait_until_finished()
            archive_family = 'small_ablation_v1' if supplemental else 'main_comparison_v1'
            archive = Path('/home/Dataset2/yqt/rulervla')/archive_family/'smoke_checkpoints'/args.out.name/'1'
            publish_checkpoint(stage/'1', archive, completed_updates=2)
            verify_checkpoint(archive, expected_updates=2)
            status.update(stage='restoring_checkpoint', checkpoint_saved=True, checkpoint=str(archive),
                checkpoint_archive_sha_verified=True)
            write_json(args.out/'status.json', status)
            shapes, shards = train.init_train_state(training_config, init_rng, optimizer_mesh, resume=True)
            shapes = jax.tree.map(lambda shape, shard: jax.ShapeDtypeStruct(shape.shape, shape.dtype, sharding=shard), shapes, shards)
            del state
            import gc
            gc.collect()
            restored = checkpoints.restore_state(manager, shapes, LoaderAssets(), step=1)
            assert int(jax.device_get(restored.step)) == 2
            # Compare full optimizer/EMA/weights, not just the inference params.
            for actual, expected in zip(jax.tree.leaves(restored), jax.tree.leaves(snapshot), strict=True):
                np.testing.assert_array_equal(jax.device_get(actual), expected)
            status.update(checkpoint_restored=True, full_train_state_restore_bitwise=True, staging_path=str(stage))
            manager.close()
            if supplemental:
                import shutil
                assert stage.resolve().parent == Path('/dev/shm') and stage.name == 'rulervla_main_smoke_%d' % os.getpid()
                shutil.rmtree(stage)
                save_lock.close()
        status.update(state='complete', stage='smoke_complete', finished_at_unix=time.time(),
            forward_backward_passed=True, effective_batch_32_verified=True)
    except BaseException as error:
        status.update(state='failed', error=repr(error), finished_at_unix=time.time())
        raise
    finally:
        write_json(args.out/'status.json', status)


if __name__ == '__main__':
    main()
