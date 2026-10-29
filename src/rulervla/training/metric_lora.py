import hashlib
import json
import math
from pathlib import Path
import re

import flax.nnx as nnx
import flax.traverse_util
import jax
import jax.numpy as jnp

from openpi.shared import nnx_utils

TRAINABLE_PATTERN = (
    r'PaliGemma/llm/.*lora_[ab]|PaliGemma/img/embedding/(kernel|bias)|'
    r'(action_in_proj|action_out_proj|time_mlp_in|time_mlp_out)/(kernel|bias)'
)
EXPECTED_TRAINABLE_COUNT = 53509280
BASE_RECEIPT_SHA256 = 'd048e7298f3cfe1687d1025ad886a28581a77f353253853aa8740e15c0793005'


def parameter_contract(params):
    if isinstance(params, nnx.State):
        params = params.to_pure_dict()
    flat = flax.traverse_util.flatten_dict(params, sep='/')
    result = {name: dict(shape=list(value.shape), dtype=str(value.dtype))
              for name, value in sorted(flat.items())}
    if (any(not re.fullmatch(TRAINABLE_PATTERN, name) for name in flat)
            or sum(math.prod(value.shape) for value in flat.values()) != EXPECTED_TRAINABLE_COUNT):
        raise ValueError('LoRA trainable delta differs from the fixed parameter allowlist')
    return result


def validate_delta(params, reference):
    def flatten(value):
        return flax.traverse_util.flatten_dict(
            value.to_pure_dict() if isinstance(value, nnx.State) else value, sep='/')
    actual, expected = flatten(params), flatten(reference)
    if (set(actual) != set(expected) or any(actual[name].shape != value.shape
            or actual[name].dtype != value.dtype for name, value in expected.items())):
        raise ValueError('LoRA delta names/shapes/dtypes differ from the complete model')


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_base_reference(reference, *, verify_files=True):
    if (reference['receipt_sha256'] != BASE_RECEIPT_SHA256
            or file_sha256(reference['receipt_path']) != BASE_RECEIPT_SHA256):
        raise ValueError('LoRA base receipt identity differs from verified pi05_base')
    receipt = json.loads(Path(reference['receipt_path']).read_text())
    root = Path(reference['params_path']).resolve()
    if verify_files:
        for entry in receipt['files']:
            path = (root / entry['path']).resolve()
            if (not path.is_relative_to(root) or path.stat().st_size != entry['bytes']
                    or file_sha256(path) != entry['sha256']):
                raise ValueError('LoRA frozen base file identity differs: ' + entry['path'])
    return reference


def base_reference(config, *, verify_files=True):
    path = Path(config.weight_loader.params_path).resolve()
    return verify_base_reference(dict(params_path=str(path),
        receipt_path=str(path.parent / 'SOURCE_VERIFIED.json'),
        receipt_sha256=BASE_RECEIPT_SHA256), verify_files=verify_files)


def is_lora(config):
    variants = (config.model.paligemma_variant, config.model.action_expert_variant)
    if variants == ('gemma_2b_lora', 'gemma_300m_lora'):
        return True
    if variants == ('gemma_2b', 'gemma_300m'):
        return False
    raise ValueError('Unsupported Metric Cue model variants')


def lora_freeze_filter():
    return nnx.Not(nnx_utils.PathRegex(TRAINABLE_PATTERN))


def initialize_lora_parameters(params, seed):
    flat = flax.traverse_util.flatten_dict(params, sep='/')
    keys = sorted(name for name in flat if re.fullmatch(r'PaliGemma/llm/.*lora_[ab]', name))
    if not keys:
        raise ValueError('LoRA parameters are missing from the complete model')
    for index, name in enumerate(keys):
        value = flat[name]
        if not isinstance(value, jax.ShapeDtypeStruct):
            raise ValueError('Base checkpoint unexpectedly contains LoRA parameters: ' + name)
        flat[name] = (jnp.zeros(value.shape, value.dtype) if name.endswith('lora_b')
                      else jax.random.normal(jax.random.fold_in(jax.random.key(seed), index),
                                             value.shape, value.dtype) * 0.01)
    return flax.traverse_util.unflatten_dict(flat, sep='/')
