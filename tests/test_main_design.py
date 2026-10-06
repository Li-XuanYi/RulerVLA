import hashlib
import json
from pathlib import Path


def test_design_manifest_is_portable_and_verifies_frozen_files():
    root = Path(__file__).parents[1]
    manifest = root / 'experiments/main_comparison_v1/DESIGN_MANIFEST.sha256'
    for line in manifest.read_text().splitlines():
        digest, name = line.split('  ', 1)
        assert '\\' not in name and not Path(name).is_absolute()
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest


def test_all_48_runs_have_identical_training_budget_and_no_launch_claim():
    folder = Path(__file__).parents[1] / 'experiments/main_comparison_v1/runs'
    runs = [json.loads(path.read_text()) for path in folder.glob('*.json')]
    assert len(runs) == 48
    assert len({(r['suite'], r['method'], r['seed']) for r in runs}) == 48
    assert len({json.dumps(r['shared_training'], sort_keys=True) for r in runs}) == 1
    assert all(not r['execution_ready'] and r['status'] == 'planned' for r in runs)
