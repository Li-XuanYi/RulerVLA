import hashlib

import numpy as np
import pytest

from rulervla.cues.metric_ruler import MetricCueConfig, project_metric_cue, render_metric_cue
from rulervla.data.schema import Camera, rotation_from_xyzw


def projection_scene(name):
    intrinsic = np.array([[400., 0., 128.], [0., 400., 128.], [0., 0., 1.]])
    transform = np.eye(4)
    origin_optical = np.array([0., 0., 1.])
    options = {}
    if name in {"oblique", "near_crossing", "behind", "asymmetric"}:
        quaternion = np.array([.2, -.35, .1, .9])
        quaternion /= np.linalg.norm(quaternion)
        transform[:3, :3] = rotation_from_xyzw(quaternion)
        transform[:3, 3] = [.17, -.09, .21]
        intrinsic[0, 0], intrinsic[1, 1] = 390., 415.
        intrinsic[0, 2], intrinsic[1, 2] = 129.75, 126.25
        origin_optical = np.array([.015, -.028, .8])
    if name == "viewport":
        origin_optical = np.array([.32, .04, .7])
    elif name == "near_to_front":
        origin_optical = np.array([.0009, -.0005, .0005])
    elif name == "near_crossing":
        origin_optical = np.array([.002, .002, .035])
        options = {"near_clip_m": .01, "tick_spacing_m": .025}
    elif name == "behind":
        origin_optical = np.array([.003, -.002, -.012])
        options = {"near_clip_m": .01, "tick_spacing_m": .025}
    elif name == "exact_near":
        origin_optical = np.array([0., 0., .001])
    elif name == "asymmetric":
        options = {"axis_extent_m": ((-.12, .24), (-.18, .09), (-.06, .21)),
                   "tick_spacing_m": .03, "major_tick_every": 3}
    elif name == "empty_ticks":
        options = {"tick_spacing_m": 1.}
    elif name == "disabled":
        options = {"enabled": False}
    origin = transform[:3, 3] + transform[:3, :3] @ origin_optical
    return origin, Camera(intrinsic, transform), MetricCueConfig(**options)


SCENES = ("front", "oblique", "viewport", "near_to_front", "near_crossing",
          "behind", "exact_near", "asymmetric", "empty_ticks", "disabled")


@pytest.mark.parametrize("name", SCENES)
def test_signed_world_ticks_match_individual_camera_projection(name):
    origin, camera, config = projection_scene(name)
    projection = project_metric_cue(origin, camera, config)
    uv, depth = camera.project(origin[None])
    np.testing.assert_allclose(projection.origin_uv, uv[0], rtol=0, atol=1e-10)
    assert projection.origin_optical_depth_m == depth[0]
    expected_keys = []
    for axis, (low, high) in enumerate(config.axis_extent_m):
        count_negative = int(np.floor((-low + 1e-10) / config.tick_spacing_m))
        count_positive = int(np.floor((high + 1e-10) / config.tick_spacing_m))
        expected_keys.extend((axis, index) for index in range(-1, -count_negative - 1, -1))
        expected_keys.extend((axis, index) for index in range(1, count_positive + 1))
    assert [(tick.axis, tick.index) for tick in projection.ticks] == expected_keys
    for tick in projection.ticks:
        point = origin.copy()
        point[tick.axis] += tick.index * config.tick_spacing_m
        np.testing.assert_array_equal(tick.world_m, point)
        expected_uv, expected_depth = camera.project(point[None])
        np.testing.assert_allclose(tick.uv, expected_uv[0], rtol=0, atol=1e-10)
        assert tick.optical_depth_m == expected_depth[0]
        expected_visible = bool(expected_depth[0] >= config.near_clip_m
                                and np.isfinite(expected_uv[0]).all()
                                and 0 <= expected_uv[0, 0] <= camera.resolution[1] - 1
                                and 0 <= expected_uv[0, 1] <= camera.resolution[0] - 1)
        assert tick.visible == expected_visible


# 这些散列在优化前由 v2 的真实投影、裁剪和 PIL 绘制函数生成。
RASTER_SHA256 = {
    "front": "60cf8b2923428e8c1ad3abd4a7aed8ac7d085aa3755cbae1886a4f3d20276f12",
    "oblique": "ba83ea807439acc83d4f6e6f3f78489544bede1be94899b4be392d13f2c576a9",
    "viewport": "75a02cd0b5492ca5b609d1f91bc2cfc92cab35c0736bec3e75755537ae49cdf7",
    "near_to_front": "68fa8dd4d70fd22c03b0f5045a57ba5c32ce5a82372b2de0fe83278329ee092d",
    "near_crossing": "9c671b669404a66f5e05bc9909bf7cbf5f024bc83ee4bcadbe548543a610c2cb",
    "behind": "3381de4ca9f3a477f25989dfc8b744e7916046b7aa369f61a9a2f7dc0963ec9e",
    "exact_near": "f395c7436e24b1497fa56afaa855937b0398014811b664d88ab52447766519e4",
    "asymmetric": "0be17fd4f88e633a1114592af9fcee51228e65eab7cfede5bd8561fbc74d285b",
    "empty_ticks": "3fae9732c755a7733630afed0c523d1810fe66178154d9aef03060fb4c43a195",
    "disabled": "3381de4ca9f3a477f25989dfc8b744e7916046b7aa369f61a9a2f7dc0963ec9e",
}


@pytest.mark.parametrize("name", SCENES)
def test_raster_preserves_v2_clipping_dashes_and_tick_order(name):
    origin, camera, config = projection_scene(name)
    cue, projection = render_metric_cue(origin, camera, config)
    assert cue.dtype == np.float32
    assert cue.shape == camera.resolution + (3,)
    assert np.isfinite(cue).all()
    assert np.isin(cue, [0., 1.]).all()
    raster = (cue * 255).astype(np.uint8)
    assert hashlib.sha256(raster.tobytes()).hexdigest() == RASTER_SHA256[name]
    if name == "near_crossing":
        assert any(tick.optical_depth_m < config.near_clip_m for tick in projection.ticks)
        assert np.count_nonzero(cue) > 0
    if name == "viewport":
        assert any(not tick.visible for tick in projection.ticks)
        assert np.count_nonzero(cue[:, -1]) > 0
