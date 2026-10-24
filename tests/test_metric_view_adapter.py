import numpy as np
import sys

from rulervla.adapters.metric_cue import MetricViewAdapter
from rulervla.cues.metric_ruler import MetricCueConfig
from rulervla.adapters.openpi import resize_rgb
from rulervla.geometry.action_coord import raster


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


def test_disabled_echo_skips_rendering_and_composition_with_identical_rgb(sample, config):
    sample['wrist_intrinsic'] = sample['intrinsic'].copy()
    sample['wrist_camera_to_world'] = np.eye(4)
    calls = []

    def trace_calls(frame, event, argument):
        if event == 'call' and frame.f_code.co_name in {'render_motion_echo', 'compose_cues'}:
            calls.append(frame.f_code.co_name)

    previous = sys.getprofile()
    try:
        sys.setprofile(trace_calls)
        result = MetricViewAdapter(config['geometry'], MetricCueConfig()).prepare(sample)
    finally:
        sys.setprofile(previous)
    assert calls == []
    assert 'Motion Echo:' not in result['prompt']
    for rgb_key, image_key in [('agentview_rgb', 'observation/image'),
                               ('wrist_rgb', 'observation/wrist_image')]:
        historical = resize_rgb(raster(raster(sample[rgb_key], 'hflip'), 'hflip'))
        np.testing.assert_array_equal(result[image_key], historical)
