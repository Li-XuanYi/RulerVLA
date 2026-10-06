"""Read the frozen run matrix and receipts; never launch or overwrite runs."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def audit(root):
    design = root / 'experiments/main_comparison_v1'
    records = []
    for config_path in sorted((design / 'runs').glob('*.json')):
        config = json.loads(config_path.read_text())
        checkpoint = Path(config['checkpoint_root']) / '29999'
        # A directory alone is not full training provenance or a valid eval result.
        train_receipt = root / 'outputs/main_comparison/runs' / config['run_id'] / 'training_verified.json'
        eval_receipt = root / 'outputs/main_comparison/runs' / config['run_id'] / 'evaluation_verified.json'
        records.append({'run_id': config['run_id'], 'suite': config['suite'], 'method': config['method'],
            'seed': config['seed'], 'planned_checkpoint': str(checkpoint), 'checkpoint_exists': checkpoint.is_dir(),
            'training_receipt_exists': train_receipt.is_file(), 'evaluation_receipt_exists': eval_receipt.is_file(),
            'action': 'needs_provenance_review' if checkpoint.is_dir() or train_receipt.is_file() else 'needs_finetuning'})
    assert len(records) == 48
    return {'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'run_count': 48,
        'checkpoint_directories_found': sum(row['checkpoint_exists'] for row in records),
        'training_receipts_found': sum(row['training_receipt_exists'] for row in records),
        'evaluation_receipts_found': sum(row['evaluation_receipt_exists'] for row in records),
        'records': records, 'legacy_evidence': 'Old LoRA/batch4 E1 and task2 pilots remain separate; not main full/batch32/four-slot results.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    report = audit(Path(__file__).resolve().parents[1])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'records'}))
