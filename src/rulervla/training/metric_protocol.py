import hashlib
import json
from pathlib import Path


def validate_data_audit(root, protocol, release):
    status_path = Path(root) / release['data_audit_status']
    status = json.loads(status_path.read_text())
    if (status['state'] != 'complete' or status['episodes'] != protocol['data']['episodes']
            or status['frames'] != protocol['data']['frames']
            or status['manifest_sha256']['rgbd'] != protocol['data']['rgbd_manifest_sha256']
            or status['manifest_sha256']['geometry'] != protocol['data']['geometry_manifest_sha256']):
        raise ValueError('Complete Spatial data audit differs from the protocol')
    digest = hashlib.sha256((status_path.parent / 'episode_receipts.jsonl').read_bytes()).hexdigest()
    if digest != release['data_audit_receipts_sha256']:
        raise ValueError('Spatial data audit receipts changed')


def load_formal_metric_run(root, arm, seed, run):
    root = Path(root).resolve()
    protocol_path = root / 'experiments/metric_cue_v1/protocol.json'
    release_path = root / 'experiments/metric_cue_v1/release.json'
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    release = json.loads(release_path.read_text())
    if protocol['schema'] != 'rulervla_metric_cue_v1' or release['schema'] != 'rulervla_metric_cue_release_v1':
        raise ValueError('Metric Cue protocol or release schema differs')
    if release['protocol_sha256'] != hashlib.sha256(protocol_bytes).hexdigest():
        raise ValueError('Metric Cue release belongs to another protocol revision')
    if arm not in protocol['arms'] or seed not in protocol['training']['seeds']:
        raise ValueError('Unknown formal arm or training seed')
    missing = [gate for gate in protocol['launch_gates'] if not release['gate_status'].get(gate)]
    if not release['training_released'] or missing:
        raise PermissionError('Formal Metric Cue training gates remain closed: ' + ', '.join(missing))
    validate_data_audit(root, protocol, release)
    receipt_path = release['gpu_smoke_receipt']
    if receipt_path is None:
        raise ValueError('Full model GPU receipt is required')
    receipt = json.loads((root / receipt_path).read_text())
    if (receipt['state'] != 'complete' or not receipt['full_state_restore_bitwise']
            or receipt['optimizer_updates'] < 1):
        raise ValueError('Full model GPU update and checkpoint recovery are unverified')
    training = protocol['training']
    if (training['batch_size'] != training['micro_batch_size'] * training['gradient_accumulation']
            or training['finetuning'] != 'full' or training['image_channels'] != 6
            or training['action_horizon'] != 10
            or training['checkpoint_updates'][-1] != training['num_steps']):
        raise ValueError('Formal Metric Cue training recipe is inconsistent')
    if (run['data']['rgbd_manifest_sha256'] != protocol['data']['rgbd_manifest_sha256']
            or run['data']['geometry_manifest_sha256'] != protocol['data']['geometry_manifest_sha256']
            or run['norm_stats_sha256'] != protocol['data']['norm_stats_sha256']):
        raise ValueError('Formal Metric Cue data differs from the frozen Spatial data')
    run = dict(run)
    run['training'] = dict(training, seed=seed)
    run['cue_config'] = str(root / protocol['arms'][arm])
    run['formal_run_id'] = f'L09_spatial_{arm}_seed{seed}'
    run['protocol_sha256'] = hashlib.sha256(protocol_bytes).hexdigest()
    return run, protocol
