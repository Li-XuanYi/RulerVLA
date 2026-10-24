import numpy as np
import pytest

from rulervla.adapters.libero import LiberoAdapter
from rulervla.adapters.metric_cue import MetricViewAdapter
from rulervla.cues.metric_ruler import MetricCueConfig
from rulervla.data.schema import validate_sample


def live_inputs(sample):
    sample['wrist_intrinsic'] = sample['intrinsic'].copy()
    sample['wrist_camera_to_world'] = np.eye(4)
    observation = {
        'agentview_image': sample['agentview_rgb'][::-1, ::-1].copy(),
        'robot0_eye_in_hand_image': sample['wrist_rgb'][::-1, ::-1].copy(),
        'robot0_eef_pos': sample['ee_world_m'].copy(),
    }
    calibration = {
        'intrinsic': sample['intrinsic'], 'camera_to_world': sample['camera_to_world'],
        'wrist_intrinsic': sample['wrist_intrinsic'],
        'wrist_camera_to_world': sample['wrist_camera_to_world'],
        'state8': sample['state8'], 'instruction': sample['task'],
        'episode_index': 2, 'control_step': 10, 'calibration_step': 10,
    }
    return observation, calibration


def test_live_without_depth_or_quaternion_preserves_model_inputs(sample, config):
    observation, calibration = live_inputs(sample)
    live = LiberoAdapter(config['geometry']).from_metric_observation(observation, **calibration)
    assert not {'agentview_depth_m', 'wrist_depth_m', 'eef_quaternion_xyzw'} & live.keys()
    assert live['episode_index'] == 2 and live['frame_index'] == 10
    adapter = MetricViewAdapter(config['geometry'], MetricCueConfig())
    stored_inputs = adapter.prepare(sample)
    live_inputs_prepared = adapter.prepare(live)
    for key in ('observation/image', 'observation/wrist_image', 'observation/agent_cue',
                'observation/wrist_cue', 'observation/state'):
        assert stored_inputs[key].shape == live_inputs_prepared[key].shape
        assert stored_inputs[key].dtype == live_inputs_prepared[key].dtype
        np.testing.assert_array_equal(stored_inputs[key], live_inputs_prepared[key])
    assert stored_inputs['prompt'] == live_inputs_prepared['prompt']
    with pytest.raises(KeyError, match='agentview_depth_m'):
        validate_sample(live)


def test_optional_depths_keep_their_actual_view_and_stored_raster(sample, config):
    observation, calibration = live_inputs(sample)
    front = np.arange(256 * 256, dtype=np.float32).reshape(256, 256)
    wrist = front + 100000
    observation['robot0_eef_quat'] = sample['eef_quaternion_xyzw']
    live = LiberoAdapter(config['geometry']).from_metric_observation(
        observation, **calibration, depth_m=front, wrist_depth_m=wrist)
    np.testing.assert_array_equal(live['agentview_depth_m'], front[::-1, ::-1])
    np.testing.assert_array_equal(live['wrist_depth_m'], wrist[::-1, ::-1])
    np.testing.assert_array_equal(live['eef_quaternion_xyzw'], sample['eef_quaternion_xyzw'])
    validate_sample(live)


@pytest.mark.parametrize('key', ['depth_m', 'wrist_depth_m'])
@pytest.mark.parametrize('depth', [np.zeros((256, 256), np.uint8), np.zeros((255, 256), np.float32)])
def test_optional_depths_are_validated_when_supplied(sample, config, key, depth):
    observation, calibration = live_inputs(sample)
    with pytest.raises(ValueError, match='metric floating'):
        LiberoAdapter(config['geometry']).from_metric_observation(
            observation, **calibration, **{key: depth})


@pytest.mark.parametrize('key', ['episode_index', 'control_step', 'calibration_step'])
@pytest.mark.parametrize('identity', [10., 10.9, True, np.bool_(True), -1, '10'])
def test_live_identity_cannot_be_truncated_or_coerced(sample, config, key, identity):
    observation, calibration = live_inputs(sample)
    calibration[key] = identity
    with pytest.raises(ValueError, match='integer scalars'):
        LiberoAdapter(config['geometry']).from_metric_observation(observation, **calibration)


def test_live_numpy_integer_identities_and_stale_calibration(sample, config):
    observation, calibration = live_inputs(sample)
    calibration.update(episode_index=np.int64(2), control_step=np.array(10), calibration_step=np.int64(10))
    live = LiberoAdapter(config['geometry']).from_metric_observation(observation, **calibration)
    assert type(live['episode_index']) is type(live['frame_index']) is int
    calibration['calibration_step'] = 9
    with pytest.raises(ValueError, match='same control step'):
        LiberoAdapter(config['geometry']).from_metric_observation(observation, **calibration)


def test_historical_ingress_keeps_depth_and_quaternion_requirements(sample, config):
    observation, calibration = live_inputs(sample)
    adapter = LiberoAdapter(config['geometry'])
    historical_calibration = {key: calibration[key] for key in
                              ('intrinsic', 'camera_to_world', 'state8', 'instruction')}
    with pytest.raises(TypeError, match='depth_m'):
        adapter.from_observation(observation, **historical_calibration)
    with pytest.raises(KeyError, match='robot0_eef_quat'):
        adapter.from_observation(observation, **historical_calibration,
                                 depth_m=np.ones((256, 256), np.float32))
