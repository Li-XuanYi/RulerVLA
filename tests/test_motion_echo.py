from collections import OrderedDict

import numpy as np

from rulervla.cues.motion_echo import (EpisodeHistoryBuffer, MotionEchoConfig,
                                        MotionHistory, compose_cues, render_motion_echo)
from rulervla.data.schema import Camera
from rulervla.training.main_data import SpatialDataset
from rulervla.adapters.metric_cue import MetricViewAdapter
from rulervla.cues.metric_ruler import MetricCueConfig
import pytest


def camera(translation=(0., 0., 0.)):
    k = np.array([[200., 0., 127.5], [0., 200., 127.5], [0., 0., 1.]])
    t = np.eye(4)
    t[:3, 3] = translation
    return Camera(k, t)


def test_training_history_uses_only_same_episode_past_control_frames():
    dataset = SpatialDataset.__new__(SpatialDataset)
    dataset.rows = [{'episode_index': 1, 'frames': 7, 'task': 'a'},
                    {'episode_index': 2, 'frames': 7, 'task': 'b'}]
    dataset.ends = np.array([7, 14])
    dataset.cache = OrderedDict()
    dataset.cache_bytes = 0
    dataset.max_cache_bytes = None
    for episode in (1, 2):
        ee = np.stack([np.array([step * .01, episode, 1.]) for step in range(7)])
        dataset.cache[episode] = {'ee_world_m': ee, 'actions': np.zeros((7, 7), np.float32)}
    history = dataset.get_history(5, history_steps=5)
    assert history.valid and history.episode_index == 1
    assert history.past_step == 0 and history.current_step == 5
    np.testing.assert_allclose(history.displacement_m, [.05, 0., 0.])
    assert not dataset.get_history(7, history_steps=5).valid


def test_online_buffer_clears_on_reset_and_step_gap():
    buffer = EpisodeHistoryBuffer(history_steps=5)
    buffer.reset('episode_a')
    for step in range(6):
        buffer.observe(step, [step * .01, 0., 1.])
    assert buffer.history().valid
    buffer.reset('episode_b')
    assert not buffer.history().valid
    buffer.observe(0, [0., 0., 1.])
    buffer.observe(2, [.02, 0., 1.])
    assert not buffer.history().valid


@pytest.mark.parametrize('control_step', [True, np.bool_(True), 0., .9, -1, '0'])
def test_online_history_rejects_noninteger_control_identity(control_step):
    buffer = EpisodeHistoryBuffer()
    buffer.reset('episode_a')
    with pytest.raises(ValueError, match='integer scalars'):
        buffer.observe(control_step, [0., 0., 1.])


def test_online_history_accepts_numpy_integer_control_identity():
    buffer = EpisodeHistoryBuffer(history_steps=1)
    buffer.reset('episode_a')
    buffer.observe(np.int64(0), [0., 0., 1.])
    buffer.observe(np.int64(1), [.01, 0., 1.])
    assert buffer.history().valid
    assert buffer.history().episode_index == 'episode_a'


def test_valid_stationary_differs_from_missing_history_and_current_camera_is_used():
    buffer = EpisodeHistoryBuffer(history_steps=5)
    buffer.reset('one')
    invalid = buffer.history()
    for step in range(6):
        buffer.observe(step, [0., 0., 1.])
    stationary = buffer.history()
    assert not invalid.valid and stationary.valid
    cue, info = render_motion_echo(stationary, camera(), MotionEchoConfig(enabled=True))
    assert not np.any(cue) and info['stationary']
    buffer.reset('two')
    for step in range(6):
        buffer.observe(step, [step * .01, 0., 1.])
    moving = buffer.history()
    cue, info = render_motion_echo(moving, camera((.1, 0., 0.)), MotionEchoConfig(enabled=True))
    assert info['rendered'] and np.any(cue[..., 0]) and np.any(cue[..., 1])
    assert not np.any(cue[..., 2])
    # Past point is projected into the CURRENT shifted camera: u = 127.5 - 20.
    np.testing.assert_allclose(info['past_uv'], [107.5, 127.5])


def test_composer_echo_priority_and_disabled_identity():
    ruler = np.zeros((8, 8, 3), np.float32)
    ruler[3, 3] = [1., 0., 0.]
    echo = np.zeros_like(ruler)
    echo[3, 3] = [1., 1., 0.]
    np.testing.assert_array_equal(compose_cues(ruler, echo, enabled=False), ruler)
    result = compose_cues(ruler, echo, enabled=True)
    np.testing.assert_array_equal(result[3, 3], [1., 1., 0.])


def test_adapter_rejects_stale_or_cross_episode_history(sample, config):
    sample['wrist_intrinsic'] = sample['intrinsic'].copy()
    sample['wrist_camera_to_world'] = np.eye(4)
    sample['episode_index'] = 2
    adapter = MetricViewAdapter(config['geometry'], MetricCueConfig(), MotionEchoConfig(enabled=True))
    wrong = MotionHistory(True, 1, 5, 0, sample['ee_world_m'], sample['ee_world_m'])
    with pytest.raises(ValueError, match='episode'):
        adapter.prepare(sample, history=wrong)
