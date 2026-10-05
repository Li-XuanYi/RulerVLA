"""Physical 5/10/15cm world-action-axis ruler, projected before policy resize."""
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from rulervla.data.schema import Camera, CorrectedEEReference, finite


@dataclass(frozen=True)
class RulerProjection:
    points_action_m: np.ndarray
    origin_uv: np.ndarray
    ticks_uv: np.ndarray
    optical_depth_m: np.ndarray
    valid: np.ndarray


def project_metric_ruler(corrected_ee_reference: CorrectedEEReference, camera: Camera,
                         lengths_m=(0.05, 0.10, 0.15)):
    if not corrected_ee_reference.accepted:
        raise ValueError("Metric ruler requires an accepted visually anchored EE reference")
    lengths = np.asarray(lengths_m, dtype=np.float64)
    if (lengths.ndim != 1 or len(lengths) == 0 or not np.isfinite(lengths).all()
            or np.any(lengths <= 0) or np.any(np.diff(lengths) <= 0)):
        raise ValueError("Metric ruler lengths must be positive increasing metres")
    origin = finite(corrected_ee_reference.position_action_m, (3,), "EE origin")
    # Physical action axes remain world XYZ, never the rotated EE body frame.
    points = origin[None, None, :] + np.eye(3)[:, None, :] * lengths[None, :, None]
    corrected_camera = camera.remove_bias(corrected_ee_reference.bias_xyz_m)
    origin_uv, origin_depth = corrected_camera.project(origin[None])
    uv, depth = corrected_camera.project(points)
    h, w = camera.resolution
    visible = (depth > 0) & np.isfinite(uv).all(axis=-1)
    visible &= (uv[..., 0] >= 0) & (uv[..., 0] <= w - 1)
    visible &= (uv[..., 1] >= 0) & (uv[..., 1] <= h - 1)
    if origin_depth[0] <= 0 or not np.isfinite(origin_uv).all():
        visible[:] = False
    return RulerProjection(points, origin_uv[0], uv, depth, visible)


def render_metric_ruler(rgb, corrected_ee_reference, camera,
                        lengths_m=(0.05, 0.10, 0.15), *, line_width=2, tick_radius=2):
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.shape != camera.resolution + (3,):
        raise ValueError("Ruler RGB must be canonical uint8 matching the camera raster")
    if line_width < 1 or tick_radius < 1:
        raise ValueError("Ruler line width and tick radius must be positive")
    projection = project_metric_ruler(corrected_ee_reference, camera, lengths_m)
    image = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(image)
    colors = ((255, 48, 48), (48, 220, 48), (48, 100, 255))
    for axis, color in enumerate(colors):
        previous = projection.origin_uv
        previous_z = camera.remove_bias(corrected_ee_reference.bias_xyz_m).project(
            corrected_ee_reference.position_action_m[None])[1][0]
        for i, uv in enumerate(projection.ticks_uv[axis]):
            current_z = projection.optical_depth_m[axis, i]
            if previous_z > 0 and current_z > 0 and np.isfinite(previous).all() and np.isfinite(uv).all():
                draw.line([tuple(previous), tuple(uv)], fill=color, width=line_width)
            if projection.valid[axis, i]:
                u, v = uv
                draw.ellipse((u-tick_radius, v-tick_radius, u+tick_radius, v+tick_radius), fill=color)
            previous, previous_z = uv, current_z
    return np.asarray(image).copy()
