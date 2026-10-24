"""Raster-level direction glyph checks for intermediate physical ruler ticks."""
import numpy as np
import pytest

from rulervla.cues.metric_ruler import MetricCueConfig, render_metric_cue
from rulervla.data.schema import Camera


def ruler_scene(oblique):
    intrinsic = np.array([[400., 0., 128.], [0., 400., 128.], [0., 0., 1.]])
    transform = np.eye(4)
    if oblique:
        angle = np.pi / 4
        c, s = np.cos(angle), np.sin(angle)
        transform[:3, :3] = [[c, 0., s], [0., 1., 0.], [-s, 0., c]]
    # Keep the TCP centred in both cameras; oblique world Z is not the optical axis.
    origin = transform[:3, 2].copy()
    return origin, Camera(intrinsic, transform)


@pytest.mark.parametrize('oblique,axis', [(False, 0), (False, 1), (True, 0), (True, 1), (True, 2)])
def test_intermediate_negative_tick_centres_stay_hollow(oblique, axis):
    origin, camera = ruler_scene(oblique)
    config = MetricCueConfig(axis_extent_m=((-0.2, 0.2),) * 3)
    cue, projection = render_metric_cue(origin, camera, config)
    for index in (-1, -2, -3):
        tick = next(t for t in projection.ticks if (t.axis, t.index) == (axis, index))
        assert tick.visible
        u, v = np.rint(tick.uv).astype(int)
        # Another outward axis segment starts at each of these centres.
        assert cue[v, u, axis] == 0, (oblique, axis, index)
        radius = config.tick_radius + (abs(index) % config.major_tick_every == 0)
        patch = cue[v-radius-1:v+radius+2, u-radius-1:u+radius+2, axis]
        assert np.any(patch == 1), 'A hollow tick must retain a visible rim'


@pytest.mark.parametrize('oblique,axis', [(False, 0), (False, 1), (True, 0), (True, 1), (True, 2)])
def test_positive_ticks_and_origin_remain_filled(oblique, axis):
    origin, camera = ruler_scene(oblique)
    cue, projection = render_metric_cue(origin, camera, MetricCueConfig(axis_extent_m=((-0.2, 0.2),) * 3))
    for index in (1, 2, 3, 4):
        tick = next(t for t in projection.ticks if (t.axis, t.index) == (axis, index))
        assert tick.visible
        u, v = np.rint(tick.uv).astype(int)
        assert cue[v, u, axis] == 1
    u, v = np.rint(projection.origin_uv).astype(int)
    assert np.all(cue[v, u] == 1)
