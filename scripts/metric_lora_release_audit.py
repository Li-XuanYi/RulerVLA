import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ['JAX_PLATFORMS'] = 'cpu'

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def audit(smoke_root):
    from rulervla.adapters.metric_pipeline import MetricPipeline
    from rulervla.training.metric_protocol import require_lora_recipe
    from rulervla.training.metric_workflow import make_training_config, training_recipe

    protocol_path = ROOT / 'experiments/metric_cue_v1/protocol.json'
    protocol = read_json(protocol_path)
    arms = {}
    shared = None
    peaks = []
    for arm, cue_path in protocol['arms'].items():
        directory = smoke_root / arm
        status = read_json(directory / 'status.json')
        manifest = read_json(directory / 'run_manifest.json')
        run = manifest['run_config']
        require_lora_recipe(run, protocol)
        frozen_protocol = subprocess.check_output(
            ['git', 'show', manifest['git_commit'] + ':experiments/metric_cue_v1/protocol.json'], cwd=ROOT)
        if hashlib.sha256(frozen_protocol).hexdigest() != digest(protocol_path):
            raise ValueError('Smoke runtime belongs to another frozen protocol: ' + arm)
        if (run['norm_stats_sha256'] != protocol['data']['norm_stats_sha256']
                or digest(Path(run['norm_stats_dir']) / 'norm_stats.json') != run['norm_stats_sha256']):
            raise ValueError('Smoke normalization differs from the frozen data: ' + arm)
        validation = protocol['training']['resource_validation']
        budget = validation['initial_gpu_free_mib_per_device']
        sampled = status['sampled_process_gpu_peak_mib']
        if (run['resources']['min_free_mib_fsdp'] != budget or len(sampled) != 2
                or run['resources']['min_cpu_available_gib'] != validation['initial_cpu_available_gib']
                or status['gpu_samples'] < 1 or not all(0 < value < budget for value in sampled.values())):
            raise ValueError('Shared LoRA resource budget was not validated: ' + arm)
        if (status['state'] != 'complete' or status['optimizer_updates'] != 3
                or status['formal_training_started'] or run['training']['num_steps'] != 2
                or run['training']['seed'] != 42
                or status['trainable_parameters'] != 53509280
                or any(not status[name] for name in ('full_state_restore_bitwise',
                    'frozen_backbone_unchanged', 'delta_inference_verified',
                    'restored_optimizer_continued', 'resource_profile_valid'))):
            raise ValueError('Incomplete real LoRA smoke: ' + arm)
        if Path(run['cue_config']).resolve() != (Path(run['_project_root']) / cue_path).resolve():
            raise ValueError('Smoke evidence belongs to another comparison arm: ' + arm)
        expected_data = {key: protocol['data'][key + '_manifest_sha256'] for key in ('rgbd', 'geometry')}
        if manifest['dataset_manifests'] != expected_data:
            raise ValueError('Smoke dataset differs: ' + arm)
        pipeline = MetricPipeline.from_files(ROOT / cue_path, run['norm_stats_dir'])
        if pipeline.fingerprint() != manifest['pipeline_sha256']:
            raise ValueError('Smoke input pipeline differs from frozen arm: ' + arm)
        metadata_path = directory / 'checkpoints/3/metric_workflow.json'
        metadata = read_json(metadata_path)
        if (metadata['schema'] != protocol['training']['lora']['checkpoint_format']
                or metadata['training_recipe'] != training_recipe(make_training_config(run))
                or metadata['pipeline_sha256'] != manifest['pipeline_sha256']
                or metadata['completed_updates'] != 3 or metadata['run_manifest'] != manifest):
            raise ValueError('Smoke checkpoint identity differs: ' + arm)
        expected = read_json(directory / 'full_state_expected_sha256.json')
        if read_json(directory / 'full_state_sha256.json') != expected:
            raise ValueError('Full restored state hashes differ: ' + arm)
        initial = read_json(directory / 'initial_trainable_sha256.json')
        before_b = {value['path']: value['sha256'] for value in initial if 'lora_b' in value['path']}
        after_b = {value['path']: value['sha256'] for value in metadata['inference_sha256']
                   if 'lora_b' in value['path']}
        if set(before_b) != set(after_b) or not any(before_b[key] != after_b[key] for key in before_b):
            raise ValueError('LoRA B did not update from its zero initialization: ' + arm)
        metrics = status['metrics']
        if len(metrics) != 3 or [record['update'] for record in metrics] != [1, 2, 3]:
            raise ValueError('Three real optimizer updates are required: ' + arm)
        for record in metrics:
            if (not record['gradients_finite'] or not math.isfinite(record['loss'])
                    or record['lora_b_grad_l2'] <= 0
                    or (arm == 'M0' and record['cue_grad_l2'] != 0)
                    or (arm != 'M0' and record['cue_grad_l2'] <= 0)):
                raise ValueError('LoRA/Cue gradient contract failed: ' + arm)
        if metrics[2]['lora_a_grad_l2'] <= 0:
            raise ValueError('Restored LoRA A did not receive a gradient: ' + arm)
        inference = read_json(directory / 'delta_inference.json')
        if not inference['finite'] or inference['shape'] != [1, 10, 32]:
            raise ValueError('Real restored policy inference is unverified: ' + arm)
        common = dict(git_commit=manifest['git_commit'], sources=manifest['loaded_source_sha256'],
                      dependency=manifest['dependency_lock_sha256'], initial=initial,
                      frozen=metadata['frozen_sha256'], base=metadata['base_reference'],
                      trainable=metadata['trainable_parameter_contract'])
        if shared is not None and common != shared:
            raise ValueError('Four arms do not share the same initialization/source/base/trainable contract')
        shared = common
        cpu_peak = status['peak_cpu_rss_gib']
        if cpu_peak <= 0 or not math.isfinite(cpu_peak):
            raise ValueError('CPU peak is invalid: ' + arm)
        peaks.append(cpu_peak)
        names = ('status.json', 'run_manifest.json', 'initial_trainable_sha256.json',
                 'gpu_equivalence.json', 'full_state_expected_sha256.json',
                 'full_state_sha256.json', 'delta_inference.json',
                 'checkpoints/2/metric_workflow.json', 'checkpoints/3/metric_workflow.json')
        evidence = {name: dict(path=str(directory / name), sha256=digest(directory / name)) for name in names}
        measured = {key: status[key] for key in ('state', 'optimizer_updates', 'trainable_parameters',
            'full_state_restore_bitwise', 'frozen_backbone_unchanged', 'delta_inference_verified',
            'restored_optimizer_continued', 'resource_profile_valid', 'peak_cpu_rss_gib',
            'sampled_process_gpu_peak_mib', 'gpu_sampling_interval_seconds', 'gpu_samples',
            'gpu_sampling_started_at', 'gpu_sampling_finished_at', 'sampling_errors')}
        arms[arm] = dict(measured, arm=arm, cue_config=cue_path,
                         pipeline_sha256=pipeline.fingerprint(), evidence=evidence)
    return dict(schema='rulervla_metric_lora_smoke_release_v1', state='complete',
                protocol_sha256_during_smoke=digest(protocol_path), runtime_commit=shared['git_commit'],
                loaded_source_sha256=shared['sources'], dependency_lock_sha256=shared['dependency'],
                lora_recipe=protocol['training']['lora'], optimizer_updates=3,
                full_state_restore_bitwise=True, arms=arms,
                formal_min_cpu_available_gib=max(64, math.ceil(max(peaks) * 1.25)),
                gpu_peak_measurement='two_second_current_process_samples',
                base_reference=shared['base'])


def main():
    parser = argparse.ArgumentParser(description='核验四组真实LoRA证据并生成发布receipt')
    parser.add_argument('--smoke-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    receipt = audit(args.smoke_root.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        stream.write(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(state=receipt['state'], arms=list(receipt['arms']),
                         formal_min_cpu_available_gib=receipt['formal_min_cpu_available_gib'])))


if __name__ == '__main__':
    main()
