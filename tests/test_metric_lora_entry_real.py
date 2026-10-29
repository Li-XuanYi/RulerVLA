import json
from pathlib import Path

import pytest

from rulervla.training.metric_resources import load_run_config

ROOT = Path(__file__).resolve().parents[1]


def test_explicit_lora_profile_matches_frozen_recipe_and_low_memory_data_cache():
    from rulervla.training.metric_protocol import require_lora_recipe

    run = load_run_config(ROOT / 'configs/metric_workflow_lora.yaml')
    protocol = json.loads((ROOT / 'experiments/metric_cue_v1/protocol.json').read_text())
    require_lora_recipe(run, protocol)
    assert run['data']['max_cache_bytes'] == 512 << 20
    assert run['resources']['min_cpu_available_gib'] == 48
    assert protocol['training']['resource_validation']['initial_cpu_available_gib'] == 48
    run['resources']['min_cpu_available_gib'] = 47
    with pytest.raises(ValueError, match='resource'):
        require_lora_recipe(run, protocol)


def test_old_full_tuning_and_mismatched_lora_rank_are_rejected():
    from rulervla.training.metric_protocol import require_lora_recipe

    protocol = json.loads((ROOT / 'experiments/metric_cue_v1/protocol.json').read_text())
    with pytest.raises(ValueError, match='LoRA'):
        require_lora_recipe(load_run_config(ROOT / 'configs/metric_workflow_node145.yaml'), protocol)
    run = load_run_config(ROOT / 'configs/metric_workflow_lora.yaml')
    run['training']['lora']['paligemma']['rank'] = 8
    with pytest.raises(ValueError, match='lora'):
        require_lora_recipe(run, protocol)


def test_twelve_lora_run_ids_are_unique_and_evaluation_remains_closed():
    from rulervla.training.metric_protocol import require_evaluation_release

    protocol = json.loads((ROOT / 'experiments/metric_cue_v1/protocol.json').read_text())
    family = ROOT / 'experiments/metric_cue_v1'
    matrix = json.loads((family / 'run_matrix.json').read_text())
    actual = {entry['run_id'] for entry in matrix['runs']}
    assert actual == {f'L09_spatial_{arm}_lora_seed{seed}' for arm in protocol['arms']
                      for seed in protocol['training']['seeds']}
    with pytest.raises(PermissionError):
        require_evaluation_release(protocol, json.loads((family / 'release.json').read_text()))
