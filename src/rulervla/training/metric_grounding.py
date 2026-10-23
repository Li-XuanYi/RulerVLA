"""Metric reading labels from current ruler geometry, never future actions."""
from dataclasses import dataclass

import numpy as np

from rulervla.cues.adaptive_metric_ruler import orient_uv, source_uv_to_policy_uv
from rulervla.cues.metric_ruler import MetricCueConfig
from rulervla.data.schema import Camera, finite


@dataclass(frozen=True)
class RulerQuery:
    view: str
    pixel_uv_policy: np.ndarray
    axis: int
    sign: int
    signed_ticks: float
    displacement_m: float
    world_point_m: np.ndarray

    def model_input(self):
        """Inference query contains only a view and pixel; no metric answer."""
        return {'view': self.view, 'pixel_uv_policy': self.pixel_uv_policy.astype(np.float32).copy()}


def make_ruler_query(ee_world_m, point_world_m, axis: int, camera: Camera,
                     ruler: MetricCueConfig, *, view: str, canonical_to_policy: str):
    """Label a visible possibly fractional position on a world-action ruler axis."""
    if view not in {'agentview', 'wrist'} or axis not in (0, 1, 2):
        raise ValueError('Unknown view or axis')
    ee = finite(ee_world_m, (3,), 'EE world position')
    point = finite(point_world_m, (3,), 'Metric query world point')
    offset = point - ee
    displacement = float(offset[axis])
    if np.max(np.abs(np.delete(offset, axis))) > 1e-6 or abs(displacement) < 1e-9:
        raise ValueError('Query must lie on one nonzero world-action axis')
    low, high = ruler.axis_extent_m[axis]
    if not low <= displacement <= high:
        raise ValueError('Query lies outside physical ruler extent')
    uv, depth = camera.project(point[None])
    uv = uv[0]
    height, width = camera.resolution
    if (depth[0] < ruler.near_clip_m or not np.isfinite(uv).all()
            or not 0 <= uv[0] <= width-1 or not 0 <= uv[1] <= height-1):
        raise ValueError('Query point is not visible in the selected camera')
    oriented = orient_uv(uv, camera.resolution, canonical_to_policy)
    policy_uv = source_uv_to_policy_uv(oriented, camera.resolution, (224, 224))
    return RulerQuery(view, np.asarray(policy_uv, dtype=np.float32), axis,
                      1 if displacement > 0 else -1, displacement / ruler.tick_spacing_m,
                      displacement, point.copy())
