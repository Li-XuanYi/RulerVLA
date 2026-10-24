"""Linux resource/data admission checks. This module never imports JAX or CUDA."""
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[3]


def load_run_config(path):
    path = Path(path).resolve()
    value = yaml.safe_load(path.read_text())
    if value.get('schema') != 'rulervla_metric_workflow_v1':
        raise ValueError('Unknown metric workflow schema')
    value['_config_path'] = str(path)
    value['_project_root'] = str(ROOT)
    for key in ('cue_config', 'norm_stats_dir', 'initial_checkpoint', 'checkpoint_receipt'):
        item = Path(value[key])
        value[key] = str(item if item.is_absolute() else ROOT/item)
    value.setdefault('path_mappings', {})
    return value


def probe_file(path, *, parse_json=False):
    program = ('import hashlib,json,sys;from pathlib import Path;'
               'p=Path(sys.argv[1]);b=p.read_bytes();'
               'v={"path":str(p),"bytes":len(b),"sha256":hashlib.sha256(b).hexdigest()};'
               'v.update({"json":json.loads(b)} if sys.argv[2]=="1" else {});print(json.dumps(v))')
    try:
        result = subprocess.run(['timeout', '-k', '1', '5', sys.executable, '-c', program,
                                 str(path), '1' if parse_json else '0'],
                                text=True, capture_output=True, timeout=8)
        if result.returncode:
            return {'path': str(path), 'error': result.stderr.strip() or f'probe exit {result.returncode}'}
        return json.loads(result.stdout)
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
        return {'path': str(path), 'error': str(error)}


def query_gpus():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free,memory.total,utilization.gpu',
                                 '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=8, check=True)
        return [dict(index=int(row[0]), free_mib=int(row[1]), total_mib=int(row[2]), utilization=int(row[3]))
                for row in csv.reader(io.StringIO(result.stdout))]
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        return {'error': str(error)}


def _gpu_status(resource, gpu_query=query_gpus):
    gpus = gpu_query()
    reasons = []
    if isinstance(gpus, dict):
        reasons.append('GPU query unavailable: '+gpus['error'])
    else:
        indexed = {row['index']: row for row in gpus}
        for index in resource['gpu_indices']:
            if index not in indexed or indexed[index]['free_mib'] < resource['min_free_mib']:
                reasons.append(f'GPU {index} requires {resource["min_free_mib"]} MiB free')
    return gpus, reasons


def inference_preflight(value, checkpoint=None, *, g1_weights_bridge=False):
    if g1_weights_bridge and checkpoint is None:
        raise ValueError('G1 weights bridge requires a checkpoint')
    gpus, reasons = _gpu_status(value['resources'])
    checked = {}
    if checkpoint is not None:
        directory = Path(checkpoint)
        names = ['params/_METADATA']
        if not g1_weights_bridge:
            names.append('metric_workflow.json')
        for name in names:
            checked[name] = probe_file(directory/name)
    if checkpoint is None or g1_weights_bridge:
        checked['cue_config'] = probe_file(value['cue_config'])
        checked['norm_stats'] = probe_file(Path(value['norm_stats_dir'])/'norm_stats.json')
        if checked['norm_stats'].get('sha256') != value['norm_stats_sha256']:
            reasons.append('Normalization statistics unavailable or hash mismatch')
    for name, result in checked.items():
        if 'error' in result:
            reasons.append(f'Inference input unavailable: {name}: {result["error"]}')
    return dict(schema='rulervla_metric_inference_preflight_v1', launch_allowed=not reasons,
                blockers=reasons, inputs=checked, gpu_snapshot=gpus, gpu_job_started=False)


def preflight(value, *, gpu_query=query_gpus, file_probe=probe_file):
    reasons, data = [], {}
    for name, filename in (('rgbd', 'regen_manifest.json'), ('geometry', 'manifest.json')):
        path = Path(value['data'][name+'_root'])/filename
        checked = file_probe(path, parse_json=True)
        data[name] = {k: v for k, v in checked.items() if k != 'json'}
        if 'error' in checked:
            reasons.append(f'{name} manifest unavailable: {checked["error"]}')
        else:
            expected = value['data'].get(name+'_manifest_sha256')
            if not expected or checked['sha256'] != expected:
                reasons.append(f'{name} manifest identity is not pinned or does not match')
            manifest = checked['json']
            if len(manifest.get('episodes', [])) != 432:
                reasons.append(f'{name} manifest must identify all 432 Spatial training episodes')
            if name == 'geometry' and (not manifest.get('complete') or manifest.get('total_frames') != 52970):
                reasons.append('Geometry manifest is not the complete 52970-frame Spatial cache')
    norm = file_probe(Path(value['norm_stats_dir'])/'norm_stats.json')
    if 'error' in norm or norm.get('sha256') != value.get('norm_stats_sha256'):
        reasons.append('Normalization statistics unavailable or hash mismatch')
    norm_source = file_probe(Path(value['norm_stats_dir'])/'source.json', parse_json=True)
    if ('error' in norm_source or any(norm_source.get('json', {}).get(name+'_manifest_sha256')
            != value['data'].get(name+'_manifest_sha256') for name in ('rgbd', 'geometry'))):
        reasons.append('Normalization source differs from the pinned training data')
    checkpoint = file_probe(Path(value['initial_checkpoint'])/'_METADATA')
    receipt = file_probe(value['checkpoint_receipt'], parse_json=True)
    if 'error' in checkpoint or 'error' in receipt:
        reasons.append('Initial checkpoint metadata or verified source receipt unavailable')
    elif receipt.get('sha256') != value.get('checkpoint_receipt_sha256'):
        reasons.append('Initial checkpoint source receipt hash mismatch')
    elif checkpoint.get('sha256') != next((item['sha256'] for item in receipt.get('json', {}).get('files', [])
                                          if item['path'] == '_METADATA'), None):
        reasons.append('Initial checkpoint metadata does not match the verified base artifact')
    resource = value['resources']
    gpus, gpu_reasons = _gpu_status(resource, gpu_query)
    reasons.extend(gpu_reasons)
    available = None
    meminfo = Path('/proc/meminfo')
    if meminfo.exists():
        available = next(int(line.split()[1]) for line in meminfo.read_text().splitlines()
                         if line.startswith('MemAvailable:'))/1024**2
        if available < resource.get('min_cpu_available_gib', 100):
            reasons.append('CPU memory below the optimizer-offload threshold')
    return dict(schema='rulervla_metric_preflight_v1', launch_allowed=not reasons, blockers=reasons,
                data=data, norm_stats=norm, norm_source={k:v for k,v in norm_source.items() if k!='json'},
                checkpoint_metadata=checkpoint, checkpoint_receipt={k:v for k,v in receipt.items() if k!='json'},
                gpu_snapshot=gpus, cpu_available_gib=available, gpu_job_started=False)
