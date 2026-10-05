"""Deterministic observable geometry evidence; no corruption or outcome labels."""
from dataclasses import dataclass

import numpy as np

from rulervla.geometry.correction import CorrectionResult


@dataclass(frozen=True)
class ReliabilityConfig:
    patch_size: int = 16
    residual_scale_m: float = 0.005
    min_fit_score: float = 0.2
    min_anchors: int = 3

    def __post_init__(self):
        if self.patch_size < 1 or self.min_anchors < 1:
            raise ValueError("Patch size and anchor count must be positive")
        if not np.isfinite(self.residual_scale_m) or self.residual_scale_m <= 0:
            raise ValueError("Residual scale must be positive")
        if not 0 <= self.min_fit_score <= 1:
            raise ValueError("Fit-score threshold must be in [0,1]")


@dataclass(frozen=True)
class GeometryReliability:
    correction_valid: bool
    fit_score: float
    patch_weights: np.ndarray


def geometry_reliability(result: CorrectionResult, confidence, depth_valid,
                         config: ReliabilityConfig = ReliabilityConfig()):
    confidence = np.asarray(confidence, dtype=np.float64)
    valid = np.asarray(depth_valid)
    if (confidence.shape != result.weights.shape or not np.isfinite(confidence).all()
            or np.any((confidence < 0) | (confidence > 1))):
        raise ValueError("Confidence must correspond to anchors and lie in [0,1]")
    if valid.ndim != 2 or valid.dtype != np.bool_ or valid.size == 0:
        raise ValueError("Depth validity must be a nonempty boolean raster")
    used = result.valid_anchors
    score = 0.0
    if result.accepted and used.sum() >= config.min_anchors:
        distances = np.linalg.norm(result.residuals_m[used], axis=1)
        residual_quality = np.exp(-np.median(distances) / config.residual_scale_m)
        agreement = np.mean(distances <= config.residual_scale_m)
        score = float(np.clip(confidence[used].mean() * residual_quality * agreement, 0, 1))
    accepted = bool(result.accepted and score >= config.min_fit_score and used.sum() >= config.min_anchors)
    h, w = valid.shape
    p = config.patch_size
    patches = np.zeros(((h + p - 1) // p, (w + p - 1) // p), dtype=np.float32)
    for y in range(patches.shape[0]):
        for x in range(patches.shape[1]):
            fraction = valid[y*p:min((y+1)*p, h), x*p:min((x+1)*p, w)].mean()
            patches[y, x] = fraction * score if accepted else 0
    return GeometryReliability(accepted, score, patches)


def raster_weights(reliability: GeometryReliability, shape, patch_size):
    weights = np.repeat(np.repeat(reliability.patch_weights, patch_size, axis=0), patch_size, axis=1)
    if weights.shape[0] < shape[0] or weights.shape[1] < shape[1]:
        raise ValueError("Reliability patch grid does not cover the source raster")
    return weights[:shape[0], :shape[1]]
