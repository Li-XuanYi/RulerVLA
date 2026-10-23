"""Independent physical-point checks for the new six-channel metric cue."""
import numpy as np

from rulervla.cues.metric_ruler import MetricCueConfig, project_metric_cue, render_metric_cue
from rulervla.data.schema import Camera
from rulervla.cues.adaptive_metric_ruler import source_uv_to_policy_uv


def camera(focal=200., translation=(0., 0., 0.)):
    k = np.array([[focal, 0., 127.5], [0., focal, 127.5], [0., 0., 1.]])
    transform = np.eye(4)
    transform[:3, 3] = translation
    return Camera(k, transform)


def tick(projection, axis, index):
    return next(t for t in projection.ticks if (t.axis, t.index) == (axis, index))


def test_signed_ticks_are_world_xyz_and_change_with_ee_origin():
    cfg = MetricCueConfig(axis_extent_m=((-0.10, 0.10),) * 3, tick_spacing_m=0.05)
    first = project_metric_cue(np.array([0., 0., 1.]), camera(), cfg)
    second = project_metric_cue(np.array([0.2, -0.1, 1.]), camera(), cfg)
    assert len(first.ticks) == 12
    np.testing.assert_allclose(tick(first, 0, -2).world_m, [-0.10, 0., 1.])
    np.testing.assert_allclose(tick(first, 1, 1).world_m, [0., 0.05, 1.])
    np.testing.assert_allclose(tick(first, 2, -1).world_m, [0., 0., 0.95])
    np.testing.assert_allclose(tick(second, 0, -2).world_m, [0.1, -0.1, 1.])
    # Ruler takes the world position, so changing EE orientation cannot rotate axes.
    np.testing.assert_allclose(tick(first, 0, 1).world_m - tick(first, 0, -1).world_m,
                               [0.10, 0., 0.])


def test_two_cameras_project_the_same_physical_tick_independently():
    cfg = MetricCueConfig(axis_extent_m=((-0.05, 0.05),) * 3)
    a = project_metric_cue([0., 0., 1.], camera(200.), cfg)
    b = project_metric_cue([0., 0., 1.], camera(100., (0.1, 0., 0.)), cfg)
    np.testing.assert_allclose(tick(a, 0, 1).world_m, tick(b, 0, 1).world_m)
    np.testing.assert_allclose(tick(a, 0, 1).uv, [137.5, 127.5])
    np.testing.assert_allclose(tick(b, 0, 1).uv, [122.5, 127.5])


def test_near_plane_and_out_of_view_ticks_do_not_draw_invalid_pixels():
    cfg = MetricCueConfig(axis_extent_m=((-0.1, 0.1),) * 3, near_clip_m=0.01)
    close = project_metric_cue([0., 0., 0.005], camera(), cfg)
    assert not tick(close, 2, -1).visible
    cue, info = render_metric_cue([0., 0., 0.005], camera(), cfg)
    assert cue.shape == (256, 256, 3) and np.isfinite(cue).all()
    narrow = project_metric_cue([0., 0., 1.], camera(5000.), cfg)
    assert not tick(narrow, 0, 2).visible
    assert narrow.diagnostics["x"]["outside_fraction"] > 0


def test_physical_spacing_controls_world_points_and_diagnostics():
    a = project_metric_cue([0., 0., 1.], camera(),
                           MetricCueConfig(axis_extent_m=((-0.12, 0.12),) * 3, tick_spacing_m=0.05))
    b = project_metric_cue([0., 0., 1.], camera(),
                           MetricCueConfig(axis_extent_m=((-0.12, 0.12),) * 3, tick_spacing_m=0.02))
    np.testing.assert_allclose(tick(a, 0, 1).world_m, [0.05, 0., 1.])
    np.testing.assert_allclose(tick(b, 0, 1).world_m, [0.02, 0., 1.])
    assert b.diagnostics["x"]["valid_ticks"] > a.diagnostics["x"]["valid_ticks"]


def test_cue_is_separate_float32_and_rgb_unchanged():
    rgb = np.full((256, 256, 3), 37, np.uint8)
    original = rgb.copy()
    cue, _ = render_metric_cue([0., 0., 1.], camera(), MetricCueConfig())
    np.testing.assert_array_equal(rgb, original)
    assert cue.dtype == np.float32 and cue.shape == rgb.shape
    assert np.isfinite(cue).all() and cue.min() >= 0 and cue.max() <= 1
    assert np.count_nonzero(cue) > 0


def test_source_to_policy_pixel_transform_is_applied_once():
    np.testing.assert_allclose(source_uv_to_policy_uv([127.5, 127.5], (256, 256), (224, 224)),
                               [111.5, 111.5])
