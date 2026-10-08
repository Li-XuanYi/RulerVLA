"""User-approved retention for new small-experiment recovery points only."""
from pathlib import Path
import shutil


def prune_verified_checkpoints(root, schedule, retention, current_updates, verifier):
    root = Path(root).resolve()
    assert (root.is_relative_to('/home/Dataset2/yqt/rulervla/small_ablation_v1/checkpoints') or
            root.is_relative_to('/home/Dataset2/yqt/rulervla/main_comparison_lora_v1/checkpoints'))
    assert current_updates in schedule
    verifier(root/str(current_updates-1), expected_updates=current_updates)
    saved = sorted(int(p.name)+1 for p in root.iterdir()
                   if p.name.isdigit() and p.is_dir())
    assert all(update in schedule for update in saved), 'Unowned checkpoint: no pruning'
    keep = set(saved[-retention['latest']:]) | set(retention['milestone_updates'])
    removed = []
    for update in saved:
        if update in keep:
            continue
        path = root/str(update-1)
        assert not path.is_symlink() and path.resolve().parent == root
        verifier(path, expected_updates=update)
        shutil.rmtree(path)
        removed.append(update)
    return removed
