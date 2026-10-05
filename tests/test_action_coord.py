import numpy as np
import pytest

from rulervla.geometry.action_coord import canonical_geometry, encode
from rulervla.geometry.dense import action_coordinates, backproject_depth, validate_transform


def test_metric_transform_and_ee_subtraction():
    k = np.eye(3)
    points, valid = backproject_depth(np.array([[1., 2.], [3., 4.]]), k)
    rotation = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
    t = np.eye(4)
    t[:3, :3] = rotation
    t[:3, 3] = [.02, -.01, .005]
    ee = np.array([.1, .2, .3])
    action, mask = action_coordinates(points, t, ee, valid)
    np.testing.assert_allclose(action[0, 1], rotation @ [2., 0, 2] + t[:3, 3] - ee)
    assert mask.all()


def test_axis_signs_use_world_xyz_not_ee_rotation(config, sample):
    first = canonical_geometry(sample, config["geometry"])
    sample["eef_quaternion_xyzw"] = np.array([0., 0, 1, 0])
    second = canonical_geometry(sample, config["geometry"])
    np.testing.assert_array_equal(first["actioncoord"], second["actioncoord"])
    assert first["actioncoord"][128, 129, 0] > first["actioncoord"][128, 128, 0]
    assert first["actioncoord"][129, 128, 1] > first["actioncoord"][128, 128, 1]


def test_invalid_depth_is_zero_and_shared_encoding_is_float32():
    values, valid = backproject_depth(np.array([[0., -1., np.nan, np.inf, 1.]]), np.eye(3))
    assert valid.tolist() == [[False, False, False, False, True]]
    np.testing.assert_array_equal(values[~valid], 0)
    action, valid = action_coordinates(values, np.eye(4), [.1, .2, .3], valid)
    out = encode(action, valid, {"frozen": True, "radius_m": 3.25, "dtype": "float32"})
    assert out.dtype == np.float32 and np.isfinite(out).all()
    np.testing.assert_array_equal(out[~valid], 0)


def test_principal_point_offset_and_no_input_mutation(config, sample):
    old = sample["intrinsic"].copy()
    maps = canonical_geometry(sample, config["geometry"])
    np.testing.assert_allclose(maps["camera_xyz"][128, 128], [.005, .005, 2])
    np.testing.assert_array_equal(sample["intrinsic"], old)


@pytest.mark.parametrize("key,value", [("action_axes", "ee_xyz"),
                                       ("principal_point_offset_pixels", [0, 0]),
                                       ("canonical_to_policy", "identity")])
def test_refuse_changed_frozen_conventions(config, sample, key, value):
    config["geometry"][key] = value
    with pytest.raises(ValueError):
        canonical_geometry(sample, config["geometry"])


def test_reject_reflection_transform():
    t = np.eye(4)
    t[0, 0] = -1
    with pytest.raises(ValueError):
        validate_transform(t)
