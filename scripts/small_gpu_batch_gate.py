"""GPU input staging gate only: real batches, no model load or optimizer update."""
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from main_gpu_smoke import write_json
from small_training_contract import sha, validate_run


def main():
    root = Path(__file__).resolve().parents[1]
    runtime = root/'outputs/main_comparison/runtime_v3'
    sys.path[:0] = [str(root/'src'), str(runtime/'src'), str(runtime/'packages/openpi-client/src')]
    out = root/'outputs/small_ablation_v1/gpu_batch_gate'
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (out/'.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = out/'status.json'
    source_paths = ['src/rulervla/adapters/small_ablation.py', 'src/rulervla/adapters/openpi.py',
        'src/rulervla/adapters/main_comparison.py', 'src/rulervla/geometry/action_coord.py',
        'src/rulervla/training/main_data.py', 'configs/rulervla_main.yaml',
        'configs/v03_geometry.json', 'configs/v03_encoding.json',
        'scripts/small_gpu_batch_gate.py', 'scripts/main_spatial_train.py', 'scripts/main_gpu_smoke.py',
        'scripts/small_spatial_train.py', 'scripts/small_gpu_smoke.py', 'scripts/small_training_contract.py',
        'scripts/small_training_queue.py', 'src/rulervla/training/accumulation.py',
        'src/rulervla/training/blocking_snapshot.py',
        'experiments/small_ablation_v1/run_matrix.json']
    sources = {relative:sha(root/relative) for relative in source_paths}
    if path.exists():
        previous = json.loads(path.read_text())
        if previous['state'] == 'complete' and previous['source_sha256'] == sources:
            print('Reuse verified GPU batch gate'); return
        raise RuntimeError('Existing gate requires inspection; no overwrite')
    status = dict(state='running', stage='loading_inputs', pid=os.getpid(),
        gpu_ids=os.environ['CUDA_VISIBLE_DEVICES'], source_sha256=sources, methods_checked=[],
        started_at_unix=time.time(), optimizer_updates=0, policy_inferences=0, rollouts=0,
        official_weights_loaded=False, formal_training_started=False)
    write_json(path, status)
    try:
        import jax
        from openpi import transforms
        from openpi.models import model
        from openpi.models.pi0_config import Pi0Config
        from openpi.models.tokenizer import PaligemmaTokenizer
        from openpi.shared import normalize
        from rulervla.adapters.small_ablation import SmallAblationAdapter
        from rulervla.adapters.main_comparison import MainComparisonInputs, MAIN_IMAGE_KEYS
        from rulervla.config import load_config
        from rulervla.training.main_data import SpatialDataset, update_indices
        assert len(jax.devices('gpu')) == 1
        device = jax.devices('gpu')[0]
        status['devices'] = [str(device)]
        dataset = SpatialDataset('/home/Dataset2/yqt/openpi-aimbot/spatialcue_depth_source/regenerated_rgbd_v02',
            '/home/Dataset2/yqt/rulervla/main_comparison_v1/data_preparation/spatial_geometry')
        policy = Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False)
        norm = normalize.load(root/'outputs/main_comparison/gpu_smoke_20261006/shared_spatial_norm')
        transform = transforms.compose([MainComparisonInputs(model_type=policy.model_type),
            transforms.Normalize(norm,use_quantiles=True,strict=True),
            transforms.TokenizePrompt(PaligemmaTokenizer(policy.max_token_len),discrete_state_input=False),
            transforms.PadStatesAndActions(32)])
        indices = update_indices(len(dataset),42,0)
        samples = [dataset[int(i)] for i in indices]
        config = load_config(root/'configs/rulervla_main.yaml')
        runs = json.loads((root/'experiments/small_ablation_v1/run_matrix.json').read_text())['runs']
        records = []
        for run in runs:
            if run['conditional']: continue
            validate_run(root,run,require_gpu_gate=False)
            adapter = SmallAblationAdapter(config,run['method'])
            entries = [transform(adapter.prepare(sample)) for sample in samples]
            data = jax.tree.map(lambda *items:np.stack(items)[:,None],*entries)
            batch = (model.Observation.from_dict(data),data['actions'])
            assert set(batch[0].images) == set(MAIN_IMAGE_KEYS)
            assert batch[1].shape == (32,1,10,32)
            status['stage'] = 'gpu_batch_roundtrip'; write_json(path,status)
            staged = jax.device_put(batch,device); jax.block_until_ready(staged)
            restored = jax.device_get(staged)
            conversions = []
            for a,b in zip(jax.tree.leaves(batch),jax.tree.leaves(restored),strict=True):
                assert np.isfinite(a).all()
                # Match the main trainer's device_put with JAX x64 disabled.
                # Float64 normalization stats produce float64 state on host;
                # CUDA inputs are intentionally canonical float32, not float64.
                expected = np.asarray(a).astype(jax.dtypes.canonicalize_dtype(a.dtype))
                assert expected.dtype == b.dtype
                np.testing.assert_array_equal(expected,b)
                if a.dtype != expected.dtype:
                    conversions.append(dict(host_dtype=str(a.dtype),device_dtype=str(expected.dtype),
                        max_cast_error=float(np.max(np.abs(a.astype(np.float64)-expected.astype(np.float64))))))
            assert all(np.asarray(mask).all() for mask in restored[0].image_masks.values())
            records.append(dict(method=run['method'], batch_shape=list(batch[1].shape),
                sample_indices_sha256=__import__('hashlib').sha256(indices.tobytes()).hexdigest(),
                image_keys=list(MAIN_IMAGE_KEYS),gpu_canonical_roundtrip_bitwise=True,
                main_dtype_canonicalization=conversions,jax_enable_x64=bool(jax.config.x64_enabled)))
            status['methods_checked'].append(run['method']); write_json(path,status)
            del staged,restored,data,batch,entries
        write_json(out/'batches.json',records)
        status.update(state='complete',stage='complete',gpu_batch_gate_passed=True,
            finished_at_unix=time.time(),episodes=len(dataset.rows),frames=len(dataset))
    except BaseException as error:
        status.update(state='failed',error=repr(error),finished_at_unix=time.time()); raise
    finally: write_json(path,status)


if __name__ == '__main__': main()
