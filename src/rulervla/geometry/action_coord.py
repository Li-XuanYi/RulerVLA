"""Frozen V0.3 raster construction; geometry is built before policy resizing.

Adapted from actioncoord/geometry/v03.py at ab3fb89. The dense primitives
in this package are copied unchanged from the same frozen source.
"""
from __future__ import annotations

import numpy as np

from rulervla.geometry.dense import action_coordinates, backproject_depth


def raster(array, operation):
    if operation == "identity":
        return np.ascontiguousarray(array)
    if operation == "vflip":
        return np.ascontiguousarray(array[::-1])
    if operation == "hflip":
        return np.ascontiguousarray(array[:, ::-1])
    if operation == "rot180":
        return np.ascontiguousarray(array[::-1, ::-1])
    raise ValueError("Unknown raster transform: " + operation)


def validate_contract(contract):
    expected = {
        "frozen": True,
        "action_axes": "world_xyz",
        "stored_rgb_to_canonical": "hflip",
        "stored_depth_to_canonical": "hflip",
        "canonical_to_policy": "hflip",
        "principal_point_offset_pixels": [-0.5, -0.5],
        "source_resolution": [256, 256],
        "policy_resolution": [224, 224],
        "pixel_coordinates": "integer array-index pixel centres",
        "depth": "metric optical z",
        "ee_origin": "gripper0_grip_site",
        "spatial_resize": "nearest_after_geometry",
    }
    for key, value in expected.items():
        if contract.get(key) != value:
            raise ValueError(f"Frozen V0.3 geometry contract changed: {key}")


def canonical_geometry(sample, contract):
    validate_contract(contract)
    rgb = raster(sample["agentview_rgb"], contract["stored_rgb_to_canonical"])
    wrist = raster(sample["wrist_rgb"], contract["stored_rgb_to_canonical"])
    depth = raster(sample["agentview_depth_m"], contract["stored_depth_to_canonical"])
    if rgb.shape != (256, 256, 3) or wrist.shape != rgb.shape or depth.shape != rgb.shape[:2]:
        raise ValueError("Geometry must be constructed at source resolution 256")
    intrinsic = np.array(sample["intrinsic"], dtype=np.float64, copy=True)
    intrinsic[:2, 2] += np.asarray(contract["principal_point_offset_pixels"])
    camera, valid = backproject_depth(depth, intrinsic)
    action, valid = action_coordinates(camera, sample["camera_to_world"], sample["ee_world_m"], valid)
    return dict(rgb=rgb, wrist=wrist, depth=depth, camera_xyz=camera, actioncoord=action, valid=valid)


def encode(values, valid, encoding):
    """V0.3 shared float encoding; invalid values and neutral background are zero."""
    if (encoding.get("frozen") is not True or encoding.get("dtype") != "float32"
            or encoding.get("radius_m") != 3.25):
        raise ValueError("Frozen shared float32 encoding with R=3.25m is required")
    values = np.asarray(values)
    if values.ndim == 2:
        values = np.repeat(values[..., None], 3, axis=-1)
    valid = np.asarray(valid, dtype=bool)
    if values.shape != valid.shape + (3,):
        raise ValueError("Spatial map and validity shape mismatch")
    result = np.clip(values / 3.25, -1, 1).astype(np.float32)
    result[~valid] = 0
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite spatial encoding")
    return result
