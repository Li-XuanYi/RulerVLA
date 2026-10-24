from __future__ import annotations

from dataclasses import dataclass
import operator

import numpy as np

from rulervla.geometry.dense import validate_intrinsics, validate_transform


def finite(value, shape, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return array


def observation_index(value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("Observation identities must be nonnegative integer scalars")
    try:
        index = operator.index(value)
    except TypeError as error:
        raise ValueError("Observation identities must be nonnegative integer scalars") from error
    if index < 0:
        raise ValueError("Observation identities must be nonnegative integer scalars")
    return index


@dataclass(frozen=True)
class Camera:
    """Canonical raster camera with integer pixel centres and corrected K.

    camera_to_action is the optical-camera to world-XYZ controller transform.
    No EE-relative rotation is applied. K is never silently flipped or rescaled.
    """
    intrinsic: np.ndarray
    camera_to_action: np.ndarray
    resolution: tuple[int, int] = (256, 256)

    def __post_init__(self):
        k = validate_intrinsics(self.intrinsic).copy()
        t = validate_transform(self.camera_to_action).copy()
        if len(self.resolution) != 2 or min(self.resolution) < 1:
            raise ValueError("Camera resolution must be positive (height, width)")
        k.setflags(write=False)
        t.setflags(write=False)
        object.__setattr__(self, "intrinsic", k)
        object.__setattr__(self, "camera_to_action", t)

    def project(self, points_action):
        points = np.asarray(points_action, dtype=np.float64)
        if points.shape[-1:] != (3,) or not np.isfinite(points).all():
            raise ValueError("World/action points must be finite (...,3)")
        t = self.camera_to_action
        optical = (points - t[:3, 3]) @ t[:3, :3]
        z = optical[..., 2]
        uv = np.full(points.shape[:-1] + (2,), np.nan)
        forward = z > 0
        homogeneous = optical[forward] @ self.intrinsic.T
        uv[forward] = homogeneous[..., :2] / homogeneous[..., 2, None]
        return uv, z

    def remove_bias(self, bias_xyz_m):
        t = self.camera_to_action.copy()
        t[:3, 3] -= finite(bias_xyz_m, (3,), "shared translation bias")
        return Camera(self.intrinsic, t, self.resolution)


@dataclass(frozen=True)
class NominalEEReference:
    """Same-observation proprioceptive EE; no visual anchor or correction."""

    position_action_m: np.ndarray

    def __post_init__(self):
        position = finite(self.position_action_m, (3,), "EE world position").copy()
        position.setflags(write=False)
        object.__setattr__(self, "position_action_m", position)


@dataclass(frozen=True)
class CorrectedEEReference:
    position_action_m: np.ndarray
    bias_xyz_m: np.ndarray
    accepted: bool

    def __post_init__(self):
        position = finite(self.position_action_m, (3,), "EE world position").copy()
        bias = finite(self.bias_xyz_m, (3,), "EE reference shared bias").copy()
        if not self.accepted and np.any(bias):
            raise ValueError("Rejected reference must not carry a correction")
        position.setflags(write=False)
        bias.setflags(write=False)
        object.__setattr__(self, "position_action_m", position)
        object.__setattr__(self, "bias_xyz_m", bias)


def rotation_from_xyzw(quaternion):
    q = finite(quaternion, (4,), "EE quaternion xyzw")
    if not np.isclose(np.linalg.norm(q), 1, atol=2e-5):
        raise ValueError("EE quaternion must be normalized")
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1 - 2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1 - 2*(x*x+y*y)],
    ])


def validate_sample(sample, *, require_depth=True):
    for key in ("agentview_rgb", "wrist_rgb"):
        image = np.asarray(sample[key])
        if image.shape != (256, 256, 3) or image.dtype != np.uint8:
            raise ValueError(f"{key} must be uint8[256,256,3]")
    for key in ("agentview_depth_m", "wrist_depth_m"):
        if key not in sample and (key != "agentview_depth_m" or not require_depth):
            continue
        depth = np.asarray(sample[key])
        if depth.shape != (256, 256) or not np.issubdtype(depth.dtype, np.floating):
            raise ValueError(f"{key} must be metric floating [256,256]; not a simulator depth buffer")
    state = finite(sample["state8"], (8,), "state8")
    ee = finite(sample["ee_world_m"], (3,), "EE world position")
    if not np.allclose(state[:3], ee, atol=2e-6, rtol=0):
        raise ValueError("state8[:3] and the geometry EE must share the same observation")
    validate_intrinsics(sample["intrinsic"])
    validate_transform(sample["camera_to_world"])
    if not isinstance(sample["task"], str) or not sample["task"].strip():
        raise ValueError("Language instruction must be a nonempty string")
