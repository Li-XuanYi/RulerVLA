"""GT robot surface landmarks for geometry-only checks; no learned detector yet."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from rulervla.data.schema import Camera, finite
from rulervla.geometry.dense import _validate_rotation


@dataclass(frozen=True)
class AnchorObservation:
    uv: np.ndarray
    confidence: np.ndarray
    visible: np.ndarray

    def __post_init__(self):
        uv = np.asarray(self.uv, dtype=np.float64)
        conf = np.asarray(self.confidence, dtype=np.float64)
        visible = np.asarray(self.visible)
        n = len(uv)
        if uv.shape != (n, 2) or conf.shape != (n,) or visible.shape != (n,):
            raise ValueError("Anchor shapes must be uv[N,2], confidence[N], visible[N]")
        if visible.dtype != np.bool_ or not np.isfinite(conf).all() or np.any((conf < 0) | (conf > 1)):
            raise ValueError("Visibility must be bool and confidence finite in [0,1]")
        if not np.isfinite(uv[visible]).all():
            raise ValueError("Visible anchor pixels must be finite")
        object.__setattr__(self, "uv", uv)
        object.__setattr__(self, "confidence", conf)
        object.__setattr__(self, "visible", visible)


class AnchorProvider(Protocol):
    def detect(self, rgb: np.ndarray) -> AnchorObservation: ...


def sample_depth(depth_m, uv):
    """Nearest integer-centre lookup; out-of-image pixels are invalid, not clamped."""
    depth = np.asarray(depth_m, dtype=np.float64)
    pixels = np.asarray(uv, dtype=np.float64)
    if depth.ndim != 2 or pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError("Expected depth[H,W], uv[N,2]")
    finite_uv = np.isfinite(pixels).all(axis=1)
    indices = np.zeros_like(pixels, dtype=np.int64)
    h, w = depth.shape
    # Bound before conversion to avoid overflow or negative wrapping.
    bounded = finite_uv & (pixels[:, 0] >= 0) & (pixels[:, 0] <= w - 1)
    bounded &= (pixels[:, 1] >= 0) & (pixels[:, 1] <= h - 1)
    indices[bounded] = np.rint(pixels[bounded]).astype(np.int64)
    z = np.zeros(len(pixels))
    z[bounded] = depth[indices[bounded, 1], indices[bounded, 0]]
    valid = bounded & np.isfinite(z) & (z > 0)
    z[~valid] = 0
    return z, valid, indices


class SimulatorGTAnchorProvider:
    """Project independently specified robot rigid-surface points using simulator GT.

    A caller binds the *true* camera, same-state EE pose and canonical depth.
    local_points_ee_m must come from robot geometry, not backprojected scene
    pixels. Depth-consistency gating checks visibility/occlusion. This is an
    oracle backend for mechanism verification, not a deployable RGB detector.
    """
    def __init__(self, local_points_ee_m, ee_world_m, ee_rotation_world, camera: Camera,
                 depth_m, *, occlusion_tolerance_m=0.01):
        self.points = np.asarray(local_points_ee_m, dtype=np.float64)
        if self.points.ndim != 2 or self.points.shape[1] != 3 or not np.isfinite(self.points).all():
            raise ValueError("Rigid robot surface points must be finite [N,3]")
        self.ee = finite(ee_world_m, (3,), "EE world position")
        self.rotation = _validate_rotation(ee_rotation_world, "EE rotation")
        self.camera = camera
        self.depth = np.asarray(depth_m)
        if self.depth.shape != camera.resolution or occlusion_tolerance_m <= 0:
            raise ValueError("Depth resolution or occlusion tolerance mismatch")
        self.tolerance = occlusion_tolerance_m

    def detect(self, rgb):
        rgb = np.asarray(rgb)
        if rgb.shape != self.camera.resolution + (3,):
            raise ValueError("Anchor RGB must share the canonical camera raster")
        world = self.points @ self.rotation.T + self.ee
        uv, expected_z = self.camera.project(world)
        measured_z, valid, _ = sample_depth(self.depth, uv)
        visible = valid & (expected_z > 0) & (np.abs(measured_z - expected_z) <= self.tolerance)
        return AnchorObservation(uv, visible.astype(np.float64), visible)


class LearnedAnchorProvider:
    """Deliberate placeholder. An untrained detector must fail explicitly."""
    def detect(self, rgb):
        raise NotImplementedError("LearnedAnchorProvider has no trained detector or valid weights")
