import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser(description='核验 L09 全部真实 Spatial 训练 episode')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/rulervla_metric_workflow.yaml')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    from rulervla.training.main_data import SpatialDataset
    from rulervla.training.metric_resources import load_run_config
    run = load_run_config(args.config)
    protocol = json.loads((ROOT / 'experiments/metric_cue_v1/protocol.json').read_text())
    dataset = SpatialDataset(run['data']['rgbd_root'], run['data']['geometry_root'],
                             max_cache_bytes=1 << 30, path_mappings=run['path_mappings'])
    if (dataset.manifest_hashes['rgbd'] != protocol['data']['rgbd_manifest_sha256'] or
            dataset.manifest_hashes['geometry'] != protocol['data']['geometry_manifest_sha256']):
        raise ValueError('Training manifests differ from the L09 protocol')
    args.out.mkdir(parents=True, exist_ok=False)
    index = 0
    with (args.out / 'episode_receipts.jsonl').open('x') as stream:
        for row in dataset.rows:
            sample = dataset[index]
            identity = dataset.sample_identity(index)
            if (identity['episode_index'] != row['episode_index'] or
                    sample['episode_index'] != row['episode_index'] or
                    sample['frame_index'] != 0):
                raise ValueError('Episode identity differs from the manifest')
            stream.write(json.dumps(identity, allow_nan=False) + '\n')
            stream.flush()
            index += row['frames']
    if index != len(dataset) or len(dataset.rows) != protocol['data']['episodes']:
        raise ValueError('Full Spatial episode or frame count differs')
    (args.out / 'status.json').write_text(json.dumps({
        'state': 'complete', 'episodes': len(dataset.rows), 'frames': index,
        'manifest_sha256': dataset.manifest_hashes}, indent=2) + '\n')


if __name__ == '__main__':
    main()
