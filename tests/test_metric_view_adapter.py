import numpy as np

from rulervla.adapters.metric_cue import MetricViewAdapter
from rulervla.cues.metric_ruler import MetricCueConfig


def test_both_synchronized_views_keep_rgb_and_receive_separate_metric_cues(sample, config):
    sample['wrist_intrinsic'] = sample['intrinsic'].copy()
    sample['wrist_camera_to_world'] = np.eye(4)
    sample['wrist_camera_to_world'][0, 3] = 0.1
    original_agent = sample['agentview_rgb'].copy()
    original_wrist = sample['wrist_rgb'].copy()
    adapter = MetricViewAdapter(config['geometry'], MetricCueConfig(axis_extent_m=((-0.1, 0.1),) * 3))
    result = adapter.prepare(sample)
    assert result['observation/image'].shape == (224, 224, 3)
    assert result['observation/wrist_image'].shape == (224, 224, 3)
    assert result['observation/agent_cue'].shape == (224, 224, 3)
    assert result['observation/wrist_cue'].shape == (224, 224, 3)
    assert result['observation/image'].dtype == result['observation/wrist_image'].dtype == np.uint8
    assert result['observation/agent_cue'].dtype == result['observation/wrist_cue'].dtype == np.float32
    assert np.any(result['observation/agent_cue'] != result['observation/wrist_cue'])
    np.testing.assert_array_equal(sample['agentview_rgb'], original_agent)
    np.testing.assert_array_equal(sample['wrist_rgb'], original_wrist)
    assert result['diagnostics']['agentview']['x']['valid_ticks'] > 0
    assert result['diagnostics']['wrist']['x']['valid_ticks'] > 0


def test_wrist_calibration_is_required(sample, config):
    adapter = MetricViewAdapter(config['geometry'], MetricCueConfig())
    try:
        adapter.prepare(sample)
    except (KeyError, ValueError):
        pass
    else:
        raise AssertionError('Missing synchronized wrist calibration was silently accepted')


def test_ee_rotation_does_not_rotate_world_axis_cue(sample, config):
    sample['wrist_intrinsic'] = sample['intrinsic'].copy()
    sample['wrist_camera_to_world'] = np.eye(4)
    adapter = MetricViewAdapter(config['geometry'], MetricCueConfig())
    first = adapter.prepare(sample)['observation/agent_cue']
    sample['eef_quaternion_xyzw'] = np.array([0., 0., np.sqrt(.5), np.sqrt(.5)])
    second = adapter.prepare(sample)['observation/agent_cue']
    np.testing.assert_array_equal(first, second)
