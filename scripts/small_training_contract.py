"""Release checks for frozen S22/S23 specs; no training settings are invented."""
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_run(root, run, require_gpu_gate=True):
    root = Path(root)
    specs = json.loads((root/'experiments/small_ablation_v1/run_matrix.json').read_text())
    planned = {item['run_id']: item for item in specs['runs']}
    assert run == planned[run['run_id']], 'Run differs from frozen spec'
    assert not run['conditional'], 'P1 requires its separate outcome gate'
    assert run['suite'] == 'libero_spatial' and run['seed'] == 42
    expected_updates = {'CameraXYZ-only': 10000, 'Ruler-only': 30000, 'ActionCoord-only': 30000}
    shared = run['shared_training']
    assert shared['optimizer_updates'] == expected_updates[run['method']]
    assert (shared['micro_batch'], shared['gradient_accumulation'], shared['effective_batch']) == (1, 32, 32)
    reference = json.loads((root/'experiments/main_comparison_v1/runs/libero_spatial_baseline_seed42.json').read_text())['shared_training']
    metadata = {'optimizer_updates', 'micro_batch', 'gradient_accumulation', 'data_parallel_replicas',
        'memory_configuration_status', 'checkpoint_updates', 'checkpoint_retention', 'formal_checkpoint_index', 'training_seeds'}
    assert {k:v for k,v in shared.items() if k not in metadata} == {
        k:v for k,v in reference.items() if k not in metadata}, 'Common settings differ from L06'
    assert shared['checkpoint_updates'] == [100] + list(range(1000, shared['optimizer_updates']+1, 1000))
    assert shared['checkpoint_retention'] == {'latest': 2, 'milestone_updates': [10000, 20000, 30000]}
    assert shared['formal_checkpoint_index'] == shared['optimizer_updates']-1
    assert shared['training_seeds'] == [42] and shared['data_parallel_replicas'] == 1
    assert Path(run['checkpoint_root']).is_relative_to('/home/Dataset2/yqt/rulervla/small_ablation_v1/checkpoints')
    norm = root/'outputs/main_comparison/gpu_smoke_20261006/shared_spatial_norm/norm_stats.json'
    assert sha(norm) == 'b5e3345ee3d2a4698348ea8f8988031f2d003fdcd36c4716a7f7abc2d0834820'
    s26 = json.loads((root/'outputs/small_ablation_v1/S26_表示核验/status.json').read_text())
    assert s26['state'] == 'complete' and s26['input_contract_passed'] and s26['encoding_gate_passed']
    if require_gpu_gate:
        gate = json.loads((root/'outputs/small_ablation_v1/gpu_batch_gate/status.json').read_text())
        assert gate['state'] == 'complete' and gate['gpu_batch_gate_passed']
        assert run['method'] in gate['methods_checked']
        for relative, expected in gate['source_sha256'].items():
            assert sha(root/relative) == expected, 'GPU gate source changed: '+relative


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    matrix = json.loads((root/'experiments/small_ablation_v1/run_matrix.json').read_text())
    for run in matrix['runs']:
        if not run['conditional']:
            validate_run(root, run)
    print('All three P0 runs match frozen shared training and GPU input gate')
