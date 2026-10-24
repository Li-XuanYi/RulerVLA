"""Verify vendored code and the pinned Transformers overlay; shared environments are read-only."""
import argparse
from importlib.metadata import version
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply-transformers-overlay',action='store_true')
    args=parser.parse_args()
    source=json.loads((ROOT/'docs/metric_cue/OPENPI_SOURCE.json').read_text())
    import transformers
    if version('transformers')!=source['transformers_version']:
        raise ValueError('Transformers version differs from the dependency lock')
    if args.apply_transformers_overlay:
        if sys.prefix==sys.base_prefix or not Path(sys.prefix).resolve().is_relative_to(ROOT):
            raise ValueError('Apply is restricted to a repository-local virtual environment')
    verified=[]
    for relative,expected in source['transformers_override_sha256'].items():
        template=ROOT/'src/openpi/models_pytorch/transformers_replace'/relative.removeprefix('transformers/')
        target=Path(transformers.__file__).parent/relative.removeprefix('transformers/')
        if hashlib.sha256(template.read_bytes()).hexdigest()!=expected:
            raise ValueError('Vendored override template changed: '+relative)
        if args.apply_transformers_overlay and target.read_bytes()!=template.read_bytes():
            shutil.copyfile(template,target)
        if hashlib.sha256(target.read_bytes()).hexdigest()!=expected:
            raise ValueError('Installed override differs: '+relative)
        verified.append(relative)
    import openpi.models.model as model
    import openpi_client.image_tools as images
    for module in (model,images):
        if not Path(module.__file__).resolve().is_relative_to(ROOT/'src'):
            raise ValueError('External editable runtime shadowed vendored code')
    print(json.dumps(dict(python=sys.version,openpi_model=model.__file__,openpi_client=images.__file__,
        verified_transformers_overlays=verified,packages={name:version(name) for name in
        ('jax','jaxlib','flax','optax','orbax-checkpoint','transformers','numpy','torch')})),flush=True)


if __name__=='__main__':main()
