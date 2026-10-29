import json
from pathlib import Path

import pytest

from rulervla.training.metric_resources import load_run_config, query_gpus

ROOT = Path(__file__).resolve().parents[1]


def test_real_gpu_allocator_configuration_matches_the_installed_native_plugin():
    from jaxlib import xla_client
    from rulervla.training.metric_resources import configure_lora_gpu_allocator

    snapshot = query_gpus()
    assert isinstance(snapshot, list)
    run = load_run_config(ROOT / 'configs/metric_workflow_lora.yaml')
    selected = sorted(snapshot, key=lambda gpu: gpu['free_mib'], reverse=True)[:2]
    assert len(selected) == 2 and all(gpu['free_mib'] >= 28672 for gpu in selected)
    run['resources']['gpu_indices'] = [gpu['index'] for gpu in selected]
    contract = configure_lora_gpu_allocator(run, snapshot)
    native = xla_client.generate_pjrt_gpu_plugin_options()
    assert native['allocator'] == 'bfc' and native['preallocate']
    assert contract['pool_target_mib_per_device'] == 26624
    assert contract['initial_reservation_mib_per_device'] == 24576
    assert contract['gpu_budget_mib_per_device'] == 28672
    assert native['memory_fraction'] == pytest.approx(contract['memory_fraction'])
    assert contract['selected_gpu_uuids'] == [gpu['uuid'] for gpu in selected]
    json.dumps(contract, allow_nan=False)
