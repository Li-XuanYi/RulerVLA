from dataclasses import dataclass, fields, replace

import cv2
import numpy as np
from PIL import Image, ImageDraw

from rulervla.data.schema import Camera, CorrectedEEReference, NominalEEReference, finite
from rulervla.geometry.camera import clip_segment_to_viewport, project_visible_segment


@dataclass(frozen=True)
class RulerProjection:
    points_action_m: np.ndarray
    origin_uv: np.ndarray
    ticks_uv: np.ndarray
    optical_depth_m: np.ndarray
    valid: np.ndarray


def _reference_camera(reference, camera):
    if isinstance(reference, NominalEEReference):
        return camera
    if not isinstance(reference, CorrectedEEReference) or not reference.accepted:
        raise ValueError("Corrected ruler requires an accepted visually anchored EE reference")
    return camera.remove_bias(reference.bias_xyz_m)


def project_metric_ruler(corrected_ee_reference, camera: Camera, lengths_m=(.05, .10, .15)):
    reference_camera = _reference_camera(corrected_ee_reference, camera)
    lengths = np.asarray(lengths_m, dtype=np.float64)
    if (lengths.ndim != 1 or len(lengths) == 0 or not np.isfinite(lengths).all()
            or np.any(lengths <= 0) or np.any(np.diff(lengths) <= 0)):
        raise ValueError("Metric ruler lengths must be positive increasing metres")
    origin = finite(corrected_ee_reference.position_action_m, (3,), "EE origin")
    points = origin[None, None, :] + np.eye(3)[:, None, :] * lengths[None, :, None]
    origin_uv, origin_depth = reference_camera.project(origin[None])
    uv, depth = reference_camera.project(points)
    h, w = camera.resolution
    visible = (depth > 0) & np.isfinite(uv).all(axis=-1)
    visible &= (uv[..., 0] >= 0) & (uv[..., 0] <= w - 1)
    visible &= (uv[..., 1] >= 0) & (uv[..., 1] <= h - 1)
    if origin_depth[0] <= 0 or not np.isfinite(origin_uv).all():
        visible[:] = False
    return RulerProjection(points, origin_uv[0], uv, depth, visible)


def render_metric_ruler(rgb, corrected_ee_reference, camera,
                        lengths_m=(.05, .10, .15), *, line_width=2, tick_radius=2):
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.shape != camera.resolution + (3,):
        raise ValueError("Ruler RGB must be canonical uint8 matching the camera raster")
    if line_width < 1 or tick_radius < 1:
        raise ValueError("Ruler line width and tick radius must be positive")
    projection = project_metric_ruler(corrected_ee_reference, camera, lengths_m)
    reference_camera = _reference_camera(corrected_ee_reference, camera)
    image = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(image)
    for axis, color in enumerate(((255, 48, 48), (48, 220, 48), (48, 100, 255))):
        previous = projection.origin_uv
        previous_z = reference_camera.project(corrected_ee_reference.position_action_m[None])[1][0]
        for index, uv in enumerate(projection.ticks_uv[axis]):
            current_z = projection.optical_depth_m[axis, index]
            if previous_z > 0 and current_z > 0 and np.isfinite(previous).all() and np.isfinite(uv).all():
                draw.line([tuple(previous), tuple(uv)], fill=color, width=line_width)
            if projection.valid[axis, index]:
                u, v = uv
                draw.ellipse((u-tick_radius, v-tick_radius, u+tick_radius, v+tick_radius), fill=color)
            previous, previous_z = uv, current_z
    return np.asarray(image).copy()


@dataclass(frozen=True)
class MetricRulerConfig:
    enabled: bool = True
    mode: str = "metric"
    origin: str = "ee_tcp"
    frame: str = "action_world_xyz"
    extent_m: float = .15
    tick_spacing_m: float = .05
    near_clip_m: float = .001
    line_width_px: int = 2
    tick_half_length_px: int = 3
    min_tick_spacing_px: float = 6.
    origin_radius_px: float = 2.
    direction_length_px: int = 28

    def __post_init__(self):
        if not isinstance(self.enabled, bool) or self.enabled != (self.mode != "blank"):
            raise ValueError("Ruler mode and enabled switch must agree")
        if self.mode not in {"blank", "direction", "metric"}:
            raise ValueError("Unknown ruler mode")
        if self.origin != "ee_tcp" or self.frame != "action_world_xyz":
            raise ValueError("Metric ruler requires EE TCP and robot world XYZ")
        sizes = (self.extent_m, self.tick_spacing_m, self.near_clip_m,
                 self.min_tick_spacing_px, self.origin_radius_px)
        if not np.isfinite(sizes).all() or min(sizes) <= 0 or self.extent_m < self.tick_spacing_m:
            raise ValueError("Ruler lengths and pixel sizes must be finite and positive")
        for value in (self.line_width_px, self.tick_half_length_px, self.direction_length_px):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError("Stroke sizes must be positive integer pixels")
        if self.min_tick_spacing_px < self.line_width_px + 2:
            raise ValueError("Tick spacing must separate raster strokes")

    @classmethod
    def from_mapping(cls, value):
        unknown = set(value) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError("Unsupported ruler configuration fields: " + ", ".join(sorted(unknown)))
        return cls(**value)


@dataclass(frozen=True)
class MetricRulerTick:
    axis: int
    index: int
    world_m: np.ndarray
    uv: np.ndarray
    optical_depth_m: float
    visible: bool


@dataclass(frozen=True)
class MetricRulerProjection:
    origin_uv: np.ndarray
    origin_optical_depth_m: float
    ticks: tuple[MetricRulerTick, ...]
    diagnostics: dict


def project_ruler_cue(ee_world_m, camera: Camera, config: MetricRulerConfig):
    origin = finite(ee_world_m, (3,), "EE world position")
    origin_uv, origin_depth = camera.project(origin[None])
    height, width = camera.resolution
    origin_visible = bool(origin_depth[0] >= config.near_clip_m and np.isfinite(origin_uv).all()
                          and np.all(origin_uv[0] >= 0) and np.all(origin_uv[0] <= [width-1, height-1]))
    ticks, diagnostics = [], {}
    count = int(np.floor((config.extent_m + 1e-10) / config.tick_spacing_m))
    for axis, name in enumerate("xyz"):
        points = origin + np.eye(3)[axis] * (np.arange(1, count+1) * config.tick_spacing_m)[:, None]
        uv, depth = camera.project(points)
        visible = ((depth >= config.near_clip_m) & np.isfinite(uv).all(axis=-1)
                   & (uv >= 0).all(axis=-1) & (uv <= [width-1, height-1]).all(axis=-1))
        adjacent = []
        previous, previous_valid = origin_uv[0], origin_visible
        for index in range(count):
            ticks.append(MetricRulerTick(axis, index+1, points[index], uv[index],
                                        float(depth[index]), bool(visible[index])))
            if previous_valid and visible[index]:
                adjacent.append(float(np.linalg.norm(uv[index] - previous)))
            previous, previous_valid = uv[index], visible[index]
        diagnostics[name] = dict(valid_ticks=int(visible.sum()), adjacent_tick_px=adjacent,
            outside_fraction=float(1-visible.mean()), behind_near_plane=int((depth < config.near_clip_m).sum()),
            visible_axis_length_px=sum(adjacent), degenerate=not adjacent or min(adjacent) < config.min_tick_spacing_px,
            origin_visible=origin_visible,
            origin_uv=origin_uv[0].tolist() if np.isfinite(origin_uv).all() else None,
            origin_optical_depth_m=float(origin_depth[0]))
    return MetricRulerProjection(origin_uv[0], float(origin_depth[0]), tuple(ticks), diagnostics)


def _draw_cue_line(plane, start, end, width):
    first, last = (tuple(np.rint(point * 256).astype(int)) for point in (start, end))
    cv2.line(plane, first, last, 255, width, cv2.LINE_AA, 8)


def _projected_axis_directions(origin, camera, near_clip_m):
    optical = (origin - camera.camera_to_action[:3, 3]) @ camera.camera_to_action[:3, :3]
    if optical[2] < near_clip_m:
        return np.zeros((3, 2))
    q = camera.intrinsic @ optical
    dq = camera.camera_to_action[:3, :3] @ camera.intrinsic.T
    return (dq[:, :2] * q[2] - q[:2] * dq[:, 2, None]) / q[2] ** 2


def render_ruler_cue(ee_world_m, camera: Camera, config: MetricRulerConfig):
    info = project_ruler_cue(ee_world_m, camera, config)
    origin = finite(ee_world_m, (3,), "EE world position")
    planes = [np.zeros(camera.resolution, np.uint8) for _ in range(3)]
    if not config.enabled:
        return np.stack(planes, axis=-1).astype(np.float32), info
    directions = _projected_axis_directions(origin, camera, config.near_clip_m)
    height, width = camera.resolution
    diagnostics = {}
    for axis, name in enumerate("xyz"):
        plane = planes[axis]
        segment = project_visible_segment(origin, origin + np.eye(3)[axis] * config.extent_m,
                                           camera, config.near_clip_m)
        segments = [] if segment is None else [dict(near_clipped=segment.near_clipped,
            viewport_clipped=segment.viewport_clipped,
            endpoints_uv=[segment.start_uv.tolist(), segment.end_uv.tolist()])]
        if config.mode == "direction":
            length = float(np.linalg.norm(directions[axis]))
            endpoint = (info.origin_uv + directions[axis] / length * config.direction_length_px
                        if length > 1e-8 else info.origin_uv)
            clipped = (clip_segment_to_viewport(info.origin_uv,
                       endpoint, camera.resolution) if length > 1e-8 else None)
            segments = []
            if clipped is not None:
                _draw_cue_line(plane, *clipped, config.line_width_px)
                segments.append(dict(near_clipped=False,
                    viewport_clipped=not np.allclose([info.origin_uv, endpoint], clipped, atol=1e-7, rtol=0),
                    endpoints_uv=[point.tolist() for point in clipped]))
        elif segment is not None:
            _draw_cue_line(plane, segment.start_uv, segment.end_uv, config.line_width_px)
        vector = np.zeros(2) if segment is None else segment.end_uv - segment.start_uv
        length = float(np.linalg.norm(vector))
        normal = np.array([-vector[1], vector[0]]) / length if length > 1e-6 else None
        accepted = [info.origin_uv] if info.diagnostics[name]["origin_visible"] else []
        selected, omitted, separation = [], [], []
        for tick in (item for item in info.ticks if item.axis == axis and config.mode == "metric"):
            reason = None
            distance = min((float(np.linalg.norm(tick.uv - point)) for point in accepted), default=None)
            if tick.optical_depth_m < config.near_clip_m:
                reason = "behind_near_plane"
            elif not tick.visible:
                reason = "outside_viewport"
            elif normal is None:
                reason = "projection_degenerate"
            else:
                ends = tick.uv + np.array([-1, 1])[:, None] * normal * config.tick_half_length_px
                margin = config.line_width_px / 2 + 1
                if np.any(ends < margin) or np.any(ends > [width-1-margin, height-1-margin]):
                    reason = "tick_glyph_clipped"
                elif distance is not None and distance < config.min_tick_spacing_px:
                    reason = "crowded"
            if reason:
                omitted.append(dict(index=tick.index, offset_m=tick.index*config.tick_spacing_m,
                    world_m=tick.world_m.tolist(), uv=tick.uv.tolist() if np.isfinite(tick.uv).all() else None,
                    reason=reason))
            else:
                _draw_cue_line(plane, *ends, config.line_width_px)
                selected.append(tick.index)
                accepted.append(tick.uv)
                if distance is not None:
                    separation.append(distance)
        if info.diagnostics[name]["origin_visible"]:
            centre = tuple(np.rint(info.origin_uv*256).astype(int))
            cv2.circle(plane, centre, round(config.origin_radius_px*256), 255, -1, cv2.LINE_AA, 8)
        angles = {}
        for other, other_name in enumerate("xyz"):
            if other == axis:
                continue
            denominator = np.linalg.norm(directions[axis]) * np.linalg.norm(directions[other])
            angles[other_name] = (float(np.degrees(np.arccos(np.clip(
                abs(np.dot(directions[axis], directions[other])) / denominator, 0, 1))))
                if denominator > 1e-12 else None)
        optical_angle = float(np.degrees(np.arccos(np.clip(abs(camera.camera_to_action[axis, 2]), 0, 1))))
        flags = []
        if optical_angle < 5:
            flags.append("close_to_optical_axis")
        if any(value is not None and value < 10 for value in angles.values()):
            flags.append("projected_axes_close")
        if any(item["reason"] == "crowded" for item in omitted):
            flags.append("crowded_ticks")
        if not selected and config.mode == "metric":
            flags.append("no_readable_ticks")
        if not info.diagnostics[name]["origin_visible"]:
            flags.append("origin_outside_view")
        diagnostics[name] = dict(**info.diagnostics[name], rendered_ticks=len(selected),
            rendered_tick_indices=selected, omitted_ticks=omitted, rendered_tick_separation_px=separation,
            tick_half_length_px=config.tick_half_length_px, clipped_fraction=info.diagnostics[name]["outside_fraction"],
            axis_angles_deg=angles, optical_axis_angle_deg=optical_angle, segments=segments,
            segment_unit="pixels" if config.mode == "direction" else "metres", degeneracy_flags=flags)
    return np.stack(planes, axis=-1).astype(np.float32)/255., replace(info, diagnostics=diagnostics)


def preview_metric_cue(cue):
    value = np.asarray(cue)
    if (value.ndim != 3 or value.shape[-1] != 3 or not np.isfinite(value).all()
            or value.min() < 0 or value.max() > 1):
        raise ValueError("Metric cue must be finite HWC3 in [0,1]")
    return np.rint(value*255).astype(np.uint8)
