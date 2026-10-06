"""Reuse the verified E1 action-sequence matcher for the other three suites.

Download fixed-revision demonstrations into Dataset2, with LFS size/SHA checks.
No training, no policy rollout, no modification of LeRobot or old E1 sources.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import time
from urllib.request import urlopen


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def normal(name):
    return re.sub(r'^(?:LIVING_ROOM|KITCHEN|STUDY)_SCENE\d+_', '', name).replace('_', ' ').strip().lower()


def listing(repo, revision, directory):
    url = 'https://hf-mirror.com/api/datasets/%s/tree/%s/%s' % (repo, revision, directory)
    with urlopen(url, timeout=30) as response:
        rows = json.load(response)
    return {row['path']: row for row in rows if row['type'] == 'file'}


def download_verified(reference, repo, revision, filename, record, root):
    path = root / filename
    expected = record.get('lfs', {}).get('oid')
    size = record['size']
    if not expected or len(expected) != 64:
        raise ValueError('Missing fixed-revision LFS SHA256: ' + filename)
    if path.exists():
        if path.stat().st_size != size or reference.hash_file(path) != expected:
            raise ValueError('Existing source failed integrity; refusing overwrite: ' + str(path))
        print(json.dumps({'event': 'reuse_verified_source', 'path': str(path)}), flush=True)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + '.partial')
    url = 'https://hf-mirror.com/datasets/%s/resolve/%s/%s' % (repo, revision, filename)
    print(json.dumps({'event': 'download_start', 'path': str(path), 'size_bytes': size}), flush=True)
    subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error',
        '--connect-timeout', '15', '--max-time', '1800', '--retry', '2', '--retry-delay', '5',
        '--continue-at', '-', '--output', str(partial), url], check=True)
    if partial.stat().st_size != size or reference.hash_file(partial) != expected:
        raise ValueError('Downloaded source failed size/SHA256: ' + filename)
    if path.exists():
        raise FileExistsError('Source appeared; refusing overwrite')
    os.replace(partial, path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lerobot', required=True, type=Path)
    parser.add_argument('--task-map', required=True, type=Path)
    parser.add_argument('--store', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--suite', required=True, choices=('libero_object', 'libero_goal', 'libero_10'))
    args = parser.parse_args()
    import fcntl
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / '.coverage.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    reference_path = Path(__file__).parent / 'references/spatialcue_depth_coverage.py'
    spec = importlib.util.spec_from_file_location('frozen_source_matcher', reference_path)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    mapping = None
    for node in ast.parse(args.task_map.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'libero_task_map' for target in node.targets):
            mapping = ast.literal_eval(node.value)
    assert mapping is not None
    all_episodes = [json.loads(line) for line in (args.lerobot / 'meta/episodes.jsonl').read_text().splitlines()]
    task_names = {normal(stem): stem for stem in mapping[args.suite]}
    subset = [ep for ep in all_episodes if any(normal(task) in task_names for task in ep['tasks'])]
    expected = {'libero_object': (454, 66984), 'libero_goal': (428, 52042), 'libero_10': (379, 101469)}[args.suite]
    assert (len(subset), sum(ep['length'] for ep in subset)) == expected
    status = {'state': 'running', 'pid': os.getpid(), 'suite': args.suite, 'training_started': False,
        'expected_episodes': len(subset), 'covered_episodes': 0, 'expected_frames': expected[1],
        'completed_tasks': 0, 'current_task': None, 'started_at_unix': time.time()}
    status_path = args.out / 'status.json'
    write_json(status_path, status)
    try:
        modified_files = listing(ref.MODIFIED_REPO, ref.MODIFIED_REV, args.suite + '_no_noops')
        original_files = None
        report = {'suite': args.suite, 'subset_episodes': expected[0], 'subset_frames': expected[1],
            'modified_repo': ref.MODIFIED_REPO, 'modified_revision': ref.MODIFIED_REV,
            'original_repo': ref.ORIGINAL_REPO, 'original_revision': ref.ORIGINAL_REV,
            'matcher_sha256': ref.hash_file(reference_path), 'tasks': {}}
        for task, stem in task_names.items():
            selected = [ep for ep in subset if any(normal(t) == task for t in ep['tasks'])]
            assert selected
            instruction = selected[0]['tasks'][0]
            status['current_task'] = instruction
            write_json(status_path, status)
            modified_name = args.suite + '_no_noops/' + stem + '_demo.hdf5'
            cache = {}
            item = {'task': instruction, 'benchmark_stem': stem, 'episodes': len(selected)}
            if modified_name in modified_files:
                path = download_verified(ref, ref.MODIFIED_REPO, ref.MODIFIED_REV, modified_name,
                    modified_files[modified_name], args.store)
                item['modified'] = ref.match_file(path, selected, args.lerobot, cache)
                matched = {row['episode_index'] for row in item['modified']['matches']}
            else:
                matched = set()
            leftovers = [ep for ep in selected if ep['episode_index'] not in matched]
            if leftovers:
                if original_files is None:
                    original_files = listing(ref.ORIGINAL_REPO, ref.ORIGINAL_REV, args.suite)
                original_name = args.suite + '/' + stem + '_demo.hdf5'
                path = download_verified(ref, ref.ORIGINAL_REPO, ref.ORIGINAL_REV, original_name,
                    original_files[original_name], args.store / 'original')
                item['original_exceptions'] = ref.original_match(path, leftovers, args.lerobot, cache)
                matched.update(row['episode_index'] for row in item['original_exceptions']['matches'])
            if matched != {ep['episode_index'] for ep in selected}:
                raise ValueError('Unmatched/ambiguous official action sequences: ' + instruction)
            report['tasks'][instruction] = item
            write_json(args.out / 'coverage.partial.json', report)
            status['covered_episodes'] += len(matched)
            status['completed_tasks'] += 1
            write_json(status_path, status)
            print(json.dumps({'event': 'task_matched_exactly', 'task': instruction, 'episodes': len(matched)}), flush=True)
        report.update(complete=True, covered_episode_count=status['covered_episodes'], covered_frame_count=expected[1])
        write_json(args.out / 'coverage.json', report)
        status.update(state='complete', current_task=None, finished_at_unix=time.time())
    except BaseException as error:
        status.update(state='failed', error=repr(error), finished_at_unix=time.time())
        raise
    finally:
        write_json(status_path, status)


if __name__ == '__main__':
    main()
