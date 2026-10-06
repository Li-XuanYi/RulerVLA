"""Create a separate four-slot OpenPI source snapshot; never modify V0.3."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def patch_model(source):
    marker = '    "right_wrist_0_rgb",\n)'
    if source.count(marker) != 1:
        raise ValueError('Unexpected OpenPI IMAGE_KEYS source')
    source = source.replace(marker, '    "right_wrist_0_rgb",\n    "aux_wrist_cue_rgb",\n)')
    start = source.index('        if train:\n', source.index('def preprocess_observation('))
    end = source.index('        out_images[key] = image', start)
    # All arms enter with final 224x224 values. No random raster or colour transforms.
    return source[:start] + '        # Main Comparison v1: image augmentation is disabled for every slot.\n' + source[end:]


def patch_spec(source):
    marker = '                    "right_wrist_0_rgb": '
    if source.count(marker) != 2:
        raise ValueError('Unexpected OpenPI inputs_spec source')
    return source.replace(marker + 'image_spec,', marker + 'image_spec,\n                    "aux_wrist_cue_rgb": image_spec,').replace(
        marker + 'image_mask_spec,', marker + 'image_mask_spec,\n                    "aux_wrist_cue_rgb": image_mask_spec,')


def prepare(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents:
        raise ValueError('Snapshot must be outside the frozen source runtime')
    if output.exists():
        raise FileExistsError('Refusing to replace an existing runtime: ' + str(output))
    if not (source / 'src/openpi/models/model.py').is_file():
        raise ValueError('Missing source OpenPI runtime')
    shutil.copytree(source, output, ignore=shutil.ignore_patterns('.git', '__pycache__', '*.pyc', '.venv', 'checkpoints', '.pytest_cache'))
    modifications = {}
    for name, patch in [('model.py', patch_model), ('pi0_config.py', patch_spec)]:
        relative = 'src/openpi/models/' + name
        original = source / relative
        target = output / relative
        target.write_text(patch(original.read_text()), encoding='utf-8', newline='\n')
        modifications[relative] = {'source_sha256': sha(original), 'snapshot_sha256': sha(target)}
    manifest = {'schema': 'rulervla_main_runtime_v1', 'source_runtime': str(source),
                'snapshot_runtime': str(output), 'changes': modifications,
                'parameter_architecture_changed': False, 'training_started': False,
                'image_slots': 4, 'image_augmentation': False,
                'source_files_sha256': {p.relative_to(source).as_posix(): sha(p)
                    for p in sorted(source.rglob('*.py')) if '__pycache__' not in p.parts}}
    (output / 'RULERVLA_RUNTIME_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.out), indent=2))
