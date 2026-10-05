"""Robust shared translation from robot anchors; the EE/action rotation is fixed."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rulervla.data.schema import Camera, finite
from rulervla.geometry.dense import _validate_rotation
from rulervla.geometry.visual_anchor import AnchorObservation, sample_depth


@dataclass(frozen=True)
class CorrectionConfig:
    min_anchors: int = 3
    min_confidence: float = 0.5
    huber_delta_m: float = 0.002
    max_fit_residual_m: float = 0.005
    max_iterations: int = 50
    convergence_m: float = 1e-8

    def __post_init__(self):
        if self.min_anchors < 1 or not 0 <= self.min_confidence <= 1:
            raise ValueError("Invalid anchor thresholds")
        if (not np.isfinite([self.huber_delta_m, self.max_fit_residual_m, self.convergence_m]).all()
                or min(self.huber_delta_m, self.max_fit_residual_m, self.convergence_m) <= 0
                or self.max_iterations < 1):
            raise ValueError("Fit tolerances and iterations must be positive")


@dataclass(frozen=True)
class CorrectionResult:
    bias_xyz_m: np.ndarray
    residuals_m: np.ndarray
    weights: np.ndarray
    accepted: bool
    valid_anchors: np.ndarray
    raw_residuals_m: np.ndarray
    reason: str


def estimate_translation(residuals_m, confidence=None, valid=None,
                         config: CorrectionConfig = CorrectionConfig()):
    """Coordinate-median initialization and confidence-weighted vector Huber IRLS.

    No bias-size threshold is used. A large consistent bias is admissible.
    Rejected estimates return exactly zero bias; masked residual rows are zero.
    """
    r = np.asarray(residuals_m, dtype=np.float64)
    if r.ndim != 2 or r.shape[1] != 3:
        raise ValueError("Residuals must have shape [N,3]")
    n = len(r)
    conf = np.ones(n) if confidence is None else np.asarray(confidence, dtype=np.float64)
    mask = np.ones(n, dtype=bool) if valid is None else np.asarray(valid)
    if conf.shape != (n,) or mask.shape != (n,) or mask.dtype != np.bool_:
        raise ValueError("Confidence and boolean validity must have shape [N]")
    if not np.isfinite(conf).all() or np.any((conf < 0) | (conf > 1)):
        raise ValueError("Confidence must be finite in [0,1]")
    mask = mask & np.isfinite(r).all(axis=1) & (conf >= config.min_confidence) & (conf > 0)
    raw = np.zeros_like(r)
    raw[mask] = r[mask]
    weights = np.zeros(n)
    if mask.sum() < config.min_anchors:
        return CorrectionResult(np.zeros(3), raw, weights, False, mask, raw, "insufficient_anchors")
    points, base = r[mask], conf[mask]
    bias = np.median(points, axis=0)
    for _ in range(config.max_iterations):
        distances = np.linalg.norm(points - bias, axis=1)
        w = base * np.minimum(1, config.huber_delta_m / np.maximum(distances, 1e-15))
        updated = np.sum(points * w[:, None], axis=0) / w.sum()
        converged = np.linalg.norm(updated - bias) <= config.convergence_m
        bias = updated
        if converged:
            break
    distances = np.linalg.norm(points - bias, axis=1)
    weights[mask] = base * np.minimum(1, config.huber_delta_m / np.maximum(distances, 1e-15))
    inliers = distances <= config.max_fit_residual_m
    accepted = (inliers.sum() >= config.min_anchors
                and np.median(distances) <= config.max_fit_residual_m)
    if not accepted:
        return CorrectionResult(np.zeros(3), raw, weights, False, mask, raw, "inconsistent_residuals")
    fitted = np.zeros_like(r)
    fitted[mask] = r[mask] - bias
    return CorrectionResult(bias, fitted, weights, True, mask, raw, "accepted")


def correct_from_anchors(anchors: AnchorObservation, local_points_ee_m, depth_m,
                         camera: Camera, ee_world_m, ee_rotation_world,
                         config: CorrectionConfig = CorrectionConfig()):
    local = np.asarray(local_points_ee_m, dtype=np.float64)
    if local.shape != (len(anchors.uv), 3) or not np.isfinite(local).all():
        raise ValueError("Robot rigid-surface points must correspond to every detected anchor")
    ee = finite(ee_world_m, (3,), "EE world position")
    rotation = _validate_rotation(ee_rotation_world, "EE rotation")
    z, depth_valid, pixel_indices = sample_depth(depth_m, anchors.uv)
    valid = anchors.visible & depth_valid
    # Distinct raster evidence only; repeated detections cannot inflate anchor count.
    seen = set()
    for i in np.flatnonzero(valid):
        key = tuple(pixel_indices[i])
        if key in seen:
            valid[i] = False
        seen.add(key)
    observed = np.zeros_like(local)
    pixels = np.c_[anchors.uv[valid], np.ones(valid.sum())]
    rays = pixels @ np.linalg.inv(camera.intrinsic).T
    point_c = rays * z[valid, None]
    t = camera.camera_to_action
    observed[valid] = point_c @ t[:3, :3].T + t[:3, 3]
    known = local @ rotation.T + ee
    residuals = np.zeros_like(local)
    residuals[valid] = observed[valid] - known[valid]
    return estimate_translation(residuals, anchors.confidence, valid, config)


def apply_correction(nominal_actioncoord, valid, result: CorrectionResult):
    values = np.asarray(nominal_actioncoord, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if values.shape != mask.shape + (3,):
        raise ValueError("ActionCoord and validity shape mismatch")
    corrected = values.copy()
    if result.accepted:
        corrected[mask] -= result.bias_xyz_m
    corrected[~mask] = 0
    return corrected
