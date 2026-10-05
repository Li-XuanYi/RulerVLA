"""Metric geometry for ActionCoord; lengths are metres and transforms are SE(3).

The camera frame uses positive optical z. Integer (u, v) pixel centres and the
supplied K are used verbatim; image flips, pixel-centre conventions, and resizing
must be reflected in K by the caller. Action axes are determined by the actual
controller, never inferred from the end-effector orientation.
"""
from __future__ import annotations

import numpy as np


def _finite_array(value, shape, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError("{} must be a finite array with shape {}; got {}".format(name, shape, array.shape))
    return array


def validate_intrinsics(intrinsic):
    """Return a float64 pinhole K, including any nonzero image-plane skew."""
    k = _finite_array(intrinsic, (3, 3), "intrinsic")
    if not np.allclose(k[2], [0.0, 0.0, 1.0], rtol=0.0, atol=1e-10):
        raise ValueError("intrinsic bottom row must be [0, 0, 1]; arbitrary scale is not inferred")
    if k[0, 0] <= 0.0 or k[1, 1] <= 0.0:
        raise ValueError("intrinsic focal lengths must be positive")
    if abs(np.linalg.det(k)) < 1e-12:
        raise ValueError("intrinsic must be invertible")
    return k


def _validate_rotation(rotation, name):
    rotation = _finite_array(rotation, (3, 3), name)
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-6):
        raise ValueError("{} must be an orthogonal rotation, without scale or shear".format(name))
    if not np.isclose(np.linalg.det(rotation), 1.0, rtol=0.0, atol=1e-6):
        raise ValueError("{} must have determinant +1".format(name))
    return rotation


def validate_transform(transform):
    """Reject malformed homogeneous matrices, reflections, scale, and shear."""
    matrix = _finite_array(transform, (4, 4), "transform")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], rtol=0.0, atol=1e-10):
        raise ValueError("transform bottom row must be [0, 0, 0, 1]")
    _validate_rotation(matrix[:3, :3], "transform rotation")
    return matrix


def _coerce_mask(valid_mask, shape):
    if valid_mask is None:
        return np.ones(shape, dtype=bool)
    mask = np.asarray(valid_mask)
    if mask.shape != shape:
        raise ValueError("valid_mask must have shape {}; got {}".format(shape, mask.shape))
    if mask.dtype != np.bool_:
        if not np.issubdtype(mask.dtype, np.number) or not np.isfinite(mask).all() or not np.isin(mask, [0, 1]).all():
            raise ValueError("valid_mask must contain booleans or finite 0/1 values")
    return mask.astype(bool, copy=True)


def backproject_depth(depth_m, intrinsic, valid_mask=None):
    """Return (H x W x 3 camera XYZ, H x W validity), computed in float64.

    Depth is optical-axis z in metres, not ray length or a normalized simulator
    depth buffer. Invalid (nonfinite or <= 0) depths and masked pixels become
    exactly zero. An explicit validity mask can exclude background pixels.
    """
    depth = np.asarray(depth_m, dtype=np.float64)
    if depth.ndim != 2 or min(depth.shape, default=0) == 0:
        raise ValueError("depth_m must be a nonempty H x W array of metric optical-axis depth")
    k = validate_intrinsics(intrinsic)
    valid = _coerce_mask(valid_mask, depth.shape) & np.isfinite(depth) & (depth > 0.0)
    vv, uu = np.indices(depth.shape, dtype=np.float64)
    pixels = np.stack((uu, vv, np.ones_like(uu)), axis=-1)
    rays = pixels @ np.linalg.inv(k).T
    # K's validated bottom row makes each ray's third coordinate exactly one.
    camera_xyz = np.zeros(depth.shape + (3,), dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        camera_xyz[valid] = rays[valid] * depth[valid, None]
    valid &= np.isfinite(camera_xyz).all(axis=-1)
    camera_xyz[~valid] = 0.0
    return camera_xyz, valid


def transform_points(points, transform):
    """Apply T_target_from_source to a finite (..., 3) array in float64."""
    points = np.asarray(points, dtype=np.float64)
    if points.shape[-1:] != (3,) or not np.isfinite(points).all():
        raise ValueError("points must be finite with final dimension 3")
    matrix = validate_transform(transform)
    result = points @ matrix[:3, :3].T + matrix[:3, 3]
    if not np.isfinite(result).all():
        raise ValueError("transformed points exceed the finite numeric range")
    return result


def action_coordinates(camera_xyz, camera_to_action, ee_position_action, valid_mask=None):
    """Return (point_A - ee_A, valid), both in the controller's action basis.

    For LIBERO OSC_POSITION/OSC_POSE translation, action axes are world axes:
    camera_to_action is C2W and ee_position_action is world site_xpos. Rotating
    coordinates by the EE quaternion would be incorrect for that controller.
    """
    points = np.asarray(camera_xyz, dtype=np.float64)
    if points.shape[-1:] != (3,):
        raise ValueError("camera_xyz must have final dimension 3")
    matrix = validate_transform(camera_to_action)
    ee = _finite_array(ee_position_action, (3,), "ee_position_action")
    valid = _coerce_mask(valid_mask, points.shape[:-1]) & np.isfinite(points).all(axis=-1)
    result = np.zeros_like(points, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        result[valid] = points[valid] @ matrix[:3, :3].T + matrix[:3, 3] - ee
    valid &= np.isfinite(result).all(axis=-1)
    result[~valid] = 0.0
    return result, valid


def action_frame_from_context(context):
    """Adapt the existing CueContext without importing renderer dependencies.

    ``action_axes_world`` stores the action basis vectors as rows. This adapter
    rotates both world points and the world EE position into that same basis.
    The returned transform origin remains the world origin; EE centring occurs
    only in action_coordinates.
    """
    validate_intrinsics(context.intrinsic)
    w2c = validate_transform(context.world_to_camera)
    axes = _validate_rotation(context.action_axes_world, "action_axes_world")
    ee_world = _finite_array(context.ee_position_world, (3,), "ee_position_world")
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = w2c[:3, :3].T
    c2w[:3, 3] = -w2c[:3, :3].T @ w2c[:3, 3]
    c2a = np.eye(4, dtype=np.float64)
    c2a[:3, :3] = axes @ c2w[:3, :3]
    c2a[:3, 3] = axes @ c2w[:3, 3]
    return c2a, axes @ ee_world


def action_coordinates_from_context(camera_xyz, context, valid_mask=None):
    """Action coordinates compatible with cues.geometry.CueContext."""
    camera_to_action, ee_position_action = action_frame_from_context(context)
    return action_coordinates(camera_xyz, camera_to_action, ee_position_action, valid_mask)
