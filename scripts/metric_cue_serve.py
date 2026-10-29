import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser(description='L09 六通道策略在线评测服务')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--gpu-index', type=int, required=True)
    args = parser.parse_args()
    protocol_path = ROOT / 'experiments/metric_cue_v1/protocol.json'
    protocol = json.loads(protocol_path.read_text())
    release = json.loads((protocol_path.parent / 'release.json').read_text())
    if release['protocol_sha256'] != hashlib.sha256(protocol_path.read_bytes()).hexdigest():
        raise ValueError('Evaluation release belongs to another protocol revision')
    from rulervla.training.metric_protocol import require_evaluation_release
    require_evaluation_release(protocol, release)
    from rulervla.training.metric_protocol import validate_data_audit
    validate_data_audit(ROOT, protocol, release)
    checkpoint = args.checkpoint.resolve()
    metadata = json.loads((checkpoint / 'metric_workflow.json').read_text())
    if (metadata.get('schema') != protocol['training']['lora']['checkpoint_format']
            or metadata['run_manifest']['run_config']['training'].get('lora') != protocol['training']['lora']):
        raise ValueError('Evaluation requires the matching LoRA checkpoint recipe')
    if metadata['completed_updates'] != protocol['training']['num_steps']:
        raise ValueError('Evaluation checkpoint differs from the fixed training update')
    run_id = metadata['run_manifest']['run_config']['formal_run_id']
    allowed = {entry['run_id'] for entry in json.loads((protocol_path.parent / 'run_matrix.json').read_text())['runs']}
    if run_id not in allowed or checkpoint.parent.parent.name != run_id:
        raise ValueError('Evaluation checkpoint has an unknown L09 run identity')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_index)
    os.environ['JAX_PLATFORMS'] = 'cuda,cpu'
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from rulervla.evaluation.metric_rollout import MetricRolloutSession
    from rulervla.training.metric_workflow import load_metric_policy
    policy = load_metric_policy(checkpoint, seed=protocol['evaluation']['policy_noise_seed'])
    session = MetricRolloutSession(policy)
    WebsocketPolicyServer(session, host=args.host, port=args.port,
                          metadata={'experiment_id': 'L09', 'run_id': run_id,
                                    'checkpoint': str(checkpoint),
                                    'pipeline_sha256': metadata['pipeline_sha256']}).serve_forever()


if __name__ == '__main__':
    main()
