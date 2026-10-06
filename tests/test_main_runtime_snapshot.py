import importlib.util
from pathlib import Path
from uuid import uuid4

import pytest


def module():
    path = Path(__file__).parents[1] / 'scripts/prepare_main_runtime.py'
    spec = importlib.util.spec_from_file_location('prepare_main_runtime', path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_snapshot_retains_input_values_and_original_source():
    m = module()
    # Default inherited Windows ACLs; pytest's chmod(0700) temp fixture is
    # unreadable in the Windows managed sandbox.
    tmp_path = Path(__file__).parents[1] / 'outputs' / ('runtime_test_' + uuid4().hex)
    tmp_path.mkdir(parents=True)
    source = tmp_path / 'source'
    directory = source / 'src/openpi/models'
    directory.mkdir(parents=True)
    model = 'IMAGE_KEYS = (\n    "right_wrist_0_rgb",\n)\ndef preprocess_observation():\n    for key in keys:\n        image = images[key]\n        if train:\n            image = distort(image)\n        out_images[key] = image\n'
    config = 'images={\n                    "right_wrist_0_rgb": image_spec,\n},\nmasks={\n                    "right_wrist_0_rgb": image_mask_spec,\n}\n'
    (directory / 'model.py').write_text(model)
    (directory / 'pi0_config.py').write_text(config)
    prefix = 'def embed_prefix(obs):\n        for name in obs.images:\n            pass\n'
    (directory / 'pi0.py').write_text(prefix)
    training = source/'src/openpi/training'
    training.mkdir()
    (training/'sharding.py').write_text('return jax.sharding.PartitionSpec(DATA_AXIS)\n')
    output = tmp_path / 'snapshot'
    m.prepare(source, output)
    assert (directory / 'model.py').read_text() == model
    assert (directory / 'pi0_config.py').read_text() == config
    assert (directory / 'pi0.py').read_text() == prefix
    patched = (output / 'src/openpi/models/model.py').read_text()
    assert 'aux_wrist_cue_rgb' in patched and 'distort(image)' not in patched
    assert 'out_images[key] = image' in patched
    with pytest.raises(FileExistsError):
        m.prepare(source, output)


def test_snapshot_refuses_unrecognized_upstream_layout():
    with pytest.raises(ValueError):
        module().patch_spec('different spec')
