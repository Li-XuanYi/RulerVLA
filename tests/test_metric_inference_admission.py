from pathlib import Path

import pytest

from rulervla.training.metric_resources import inference_preflight, load_run_config


def inference_config():
    root = Path(__file__).resolve().parents[1]
    run = load_run_config(root/'configs/rulervla_metric_workflow.yaml')
    return {key: run[key] for key in ('resources', 'cue_config', 'norm_stats_dir', 'norm_stats_sha256')}


def test_prepare_uses_committed_assets_and_actual_device_query():
    result = inference_preflight(inference_config())
    assert set(result['inputs']) == {'cue_config', 'norm_stats'}
    assert all('error' not in value for value in result['inputs'].values())
    assert all(reason.startswith(('GPU query unavailable:', 'GPU ')) for reason in result['blockers'])
    assert result['launch_allowed'] == (not result['blockers'])
    assert result['gpu_job_started'] is False


def test_missing_checkpoint_blocks_without_training_configuration(tmp_path):
    result = inference_preflight(inference_config(), tmp_path/'missing_checkpoint')
    assert set(result['inputs']) == {'params/_METADATA', 'metric_workflow.json'}
    assert all('error' in value for value in result['inputs'].values())
    assert not result['launch_allowed']
    assert result['gpu_job_started'] is False


def test_bridge_requires_explicit_checkpoint():
    with pytest.raises(ValueError, match='requires a checkpoint'):
        inference_preflight(inference_config(), g1_weights_bridge=True)
