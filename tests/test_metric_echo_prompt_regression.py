"""Prompt must describe the Echo raster that is actually shown in the current views."""
import numpy as np
import pytest

from rulervla.adapters.metric_cue import MetricViewAdapter
from rulervla.cues.metric_ruler import MetricCueConfig
from rulervla.cues.motion_echo import MotionEchoConfig, MotionHistory


def setup_sample(sample):
    sample['wrist_intrinsic'] = sample['intrinsic'].copy()
    sample['wrist_camera_to_world'] = np.eye(4)
    sample['episode_index'] = 3
    sample['frame_index'] = 10
    return sample


def test_stationary_history_does_not_claim_a_visible_arrow(sample, config):
    sample = setup_sample(sample)
    history = MotionHistory(True, 3, 10, 5, sample['ee_world_m'], sample['ee_world_m'])
    adapter = MetricViewAdapter(config['geometry'], MetricCueConfig(), MotionEchoConfig(enabled=True))
    result = adapter.prepare(sample, history=history)
    assert all(not view['echo']['rendered'] for view in result['diagnostics'].values())
    assert 'below the Motion Echo display threshold' in result['prompt']
    assert 'The yellow arrow shows' not in result['prompt']


def test_valid_moving_history_outside_both_views_does_not_claim_an_arrow(sample, config):
    sample = setup_sample(sample)
    sample['camera_to_world'][:2, 3] = [10., 10.]
    sample['wrist_camera_to_world'][:2, 3] = [10., 10.]
    past = sample['ee_world_m'] - np.array([.1, 0., 0.])
    history = MotionHistory(True, 3, 10, 5, sample['ee_world_m'], past)
    adapter = MetricViewAdapter(config['geometry'], MetricCueConfig(), MotionEchoConfig(enabled=True))
    result = adapter.prepare(sample, history=history)
    assert all(not view['echo']['rendered'] for view in result['diagnostics'].values())
    assert 'No Motion Echo is visible in the current camera views.' in result['prompt']
    assert 'The yellow arrow shows' not in result['prompt']


def test_history_from_a_different_current_frame_is_rejected_even_at_same_position(sample, config):
    sample = setup_sample(sample)
    history = MotionHistory(True, 3, 9, 4, sample['ee_world_m'], sample['ee_world_m'])
    adapter = MetricViewAdapter(config['geometry'], MetricCueConfig(), MotionEchoConfig(enabled=True))
    with pytest.raises(ValueError, match='current observation'):
        adapter.prepare(sample, history=history)


def test_clipped_shaft_prompt_does_not_locate_current_ee_at_the_border(sample, config):
    sample = setup_sample(sample)
    sample['camera_to_world'][0, 3] = -1.
    sample['wrist_camera_to_world'][0, 3] = -1.
    past = sample['ee_world_m'] - np.array([1., 0., 0.])
    history = MotionHistory(True, 3, 10, 5, sample['ee_world_m'], past)
    adapter = MetricViewAdapter(config['geometry'], MetricCueConfig(), MotionEchoConfig(enabled=True))
    result = adapter.prepare(sample, history=history)
    assert all(view['echo']['policy_visible'] and not view['echo']['arrowhead_drawn']
               for view in result['diagnostics'].values())
    assert 'Visible yellow line segments' in result['prompt']
    assert 'Clipped endpoints do not mark the current end-effector position.' in result['prompt']
    assert 'The yellow arrow shows' not in result['prompt']


def test_moving_echo_has_no_invented_metric_numbers_and_off_mode_is_unchanged(sample, config):
    sample = setup_sample(sample)
    past = sample['ee_world_m'] - np.array([.05, 0., 0.])
    history = MotionHistory(True, 3, 10, 5, sample['ee_world_m'], past)
    enabled = MetricViewAdapter(config['geometry'], MetricCueConfig(), MotionEchoConfig(enabled=True))
    moving = enabled.prepare(sample, history=history)
    assert 'The yellow arrow shows' in moving['prompt']
    assert 'Measured displacement:' not in moving['prompt']
    disabled = MetricViewAdapter(config['geometry'], MetricCueConfig())
    without = disabled.prepare(sample)
    with_history = disabled.prepare(sample, history=history)
    assert 'Motion Echo:' not in with_history['prompt']
    assert without['prompt'] == with_history['prompt']
    for key in ('observation/image', 'observation/wrist_image', 'observation/agent_cue', 'observation/wrist_cue'):
        np.testing.assert_array_equal(without[key], with_history[key])
