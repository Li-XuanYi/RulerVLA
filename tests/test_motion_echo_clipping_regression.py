"""Raster regressions for visible Echo segments and non-destructive composition."""
import numpy as np
import pytest

from rulervla.cues.motion_echo import MotionEchoConfig, MotionHistory, compose_cues, render_motion_echo
from rulervla.data.schema import Camera


def camera():
    intrinsic = np.array([[100., 0., 50.], [0., 100., 50.], [0., 0., 1.]])
    return Camera(intrinsic, np.eye(4), (101, 101))


def history(past, current):
    return MotionHistory(True, 'episode', 5, 0, np.array(current, float), np.array(past, float))


def moving_config():
    return MotionEchoConfig(enabled=True, near_clip_m=.1)


def assert_horizontal_shaft_only(cue):
    # Pillow's width=2 horizontal shaft occupies these two rows. A triangular
    # arrowhead at a clipped endpoint would paint additional off-shaft pixels.
    assert np.any(cue[50:52, :, 0])
    assert not np.any(cue[:50, :, 0])
    assert not np.any(cue[52:, :, 0])
    np.testing.assert_array_equal(cue[..., 0], cue[..., 1])
    assert not np.any(cue[..., 2])


def test_past_behind_near_plane_keeps_visible_segment_and_real_arrowhead():
    measured = history([.04, 0., -.1], [0., 0., 1.])
    before = measured.past_m.copy()
    cue, info = render_motion_echo(measured, camera(), moving_config())
    assert np.any(cue[47:50, 50:58, 0])
    assert info['rendered'] and info['arrowhead_drawn']
    assert info['past_near_plane_clipped']
    assert not info['current_near_plane_clipped']
    assert np.isnan(info['past_uv']).all()  # Original endpoint is behind camera.
    np.testing.assert_array_equal(info['current_uv'], [50., 50.])
    np.testing.assert_array_equal(measured.past_m, before)


def test_current_behind_near_plane_draws_shaft_without_false_arrowhead():
    measured = history([0., 0., 1.], [.04, 0., -.1])
    cue, info = render_motion_echo(measured, camera(), moving_config())
    assert_horizontal_shaft_only(cue)
    assert info['rendered'] and not info['arrowhead_drawn']
    assert info['current_near_plane_clipped']
    assert np.isnan(info['current_uv']).all()


def test_both_endpoints_behind_near_plane_remain_empty():
    measured = history([0., 0., -.1], [.04, 0., -.2])
    cue, info = render_motion_echo(measured, camera(), moving_config())
    assert not info['rendered'] and not info['arrowhead_drawn']
    assert info['reason'] == 'behind_near_plane'
    assert not np.any(cue)


def test_current_outside_viewport_keeps_shaft_without_boundary_arrowhead():
    measured = history([0., 0., 1.], [1., 0., 1.])
    cue, info = render_motion_echo(measured, camera(), moving_config())
    assert_horizontal_shaft_only(cue)
    assert info['rendered'] and not info['arrowhead_drawn']
    np.testing.assert_array_equal(info['current_uv'], [150., 50.])
    assert cue[50, 100, 0] == 1


def test_true_visible_endpoint_still_has_arrowhead():
    measured = history([-.25, 0., 1.], [.25, 0., 1.])
    cue, info = render_motion_echo(measured, camera(), moving_config())
    assert info['rendered'] and info['arrowhead_drawn']
    assert np.any(cue[47:50, 68:75, 0])


def test_stationary_and_disabled_echo_do_not_draw():
    stationary = history([0., 0., 1.], [.001, 0., 1.])
    cue, info = render_motion_echo(stationary, camera(), moving_config())
    assert not np.any(cue) and info['stationary'] and not info['arrowhead_drawn']
    np.testing.assert_allclose(info['displacement_m'], [.001, 0., 0.])
    cue, info = render_motion_echo(stationary, camera(), MotionEchoConfig())
    assert not np.any(cue) and not info['rendered'] and info['reason'] == 'disabled'


def test_composition_preserves_z_and_existing_ruler_values_without_mutation():
    ruler = np.zeros((8, 8, 3), np.float32)
    ruler[3, 3] = [.25, .5, 1.]
    echo = np.zeros_like(ruler)
    echo[3, 3] = [1., 1., 0.]
    original_ruler, original_echo = ruler.copy(), echo.copy()
    result = compose_cues(ruler, echo, enabled=True)
    np.testing.assert_array_equal(result[3, 3], [1., 1., 1.])
    assert np.all(result >= ruler)
    assert not np.shares_memory(result, ruler)
    np.testing.assert_array_equal(ruler, original_ruler)
    np.testing.assert_array_equal(echo, original_echo)
    disabled = compose_cues(ruler, echo, enabled=False)
    np.testing.assert_array_equal(disabled, ruler)
    assert not np.shares_memory(disabled, ruler)


@pytest.mark.parametrize('bad_value', [np.nan, -0.01, 1.01])
def test_composition_rejects_nonfinite_or_out_of_range_cue(bad_value):
    ruler = np.zeros((3, 3, 3), np.float32)
    echo = np.zeros_like(ruler)
    echo[1, 1, 0] = bad_value
    with pytest.raises(ValueError, match='finite'):
        compose_cues(ruler, echo, enabled=True)
