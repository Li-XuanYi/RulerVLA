"""World-action-axis rulers, including the independent metric cue pilot."""
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from rulervla.data.schema import Camera, CorrectedEEReference, NominalEEReference, finite


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


def project_metric_ruler(corrected_ee_reference: CorrectedEEReference | NominalEEReference, camera: Camera,
                         lengths_m=(0.05, 0.10, 0.15)):
    """Project a nominal EE ruler or an explicitly accepted anchored ruler."""
    reference_camera = _reference_camera(corrected_ee_reference, camera)
    lengths = np.asarray(lengths_m, dtype=np.float64)
    if (lengths.ndim != 1 or len(lengths) == 0 or not np.isfinite(lengths).all()
            or np.any(lengths <= 0) or np.any(np.diff(lengths) <= 0)):
        raise ValueError("Metric ruler lengths must be positive increasing metres")
    origin = finite(corrected_ee_reference.position_action_m, (3,), "EE origin")
    # Physical action axes remain world XYZ, never the rotated EE body frame.
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
                        lengths_m=(0.05, 0.10, 0.15), *, line_width=2, tick_radius=2):
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.shape != camera.resolution + (3,):
        raise ValueError("Ruler RGB must be canonical uint8 matching the camera raster")
    if line_width < 1 or tick_radius < 1:
        raise ValueError("Ruler line width and tick radius must be positive")
    projection = project_metric_ruler(corrected_ee_reference, camera, lengths_m)
    reference_camera = _reference_camera(corrected_ee_reference, camera)
    image = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(image)
    colors = ((255, 48, 48), (48, 220, 48), (48, 100, 255))
    for axis, color in enumerate(colors):
        previous = projection.origin_uv
        previous_z = reference_camera.project(
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


@dataclass(frozen=True)
class MetricCueConfig:
    """One physical ruler definition shared by projection, rendering and prompt."""

    enabled: bool = True
    origin: str = "ee_tcp"
    frame: str = "action_world_xyz"
    axis_extent_m: tuple[tuple[float, float], ...] = ((-.30, .30),) * 3
    tick_spacing_m: float = .05
    major_tick_every: int = 2
    positive_style: str = "solid_filled"
    negative_style: str = "dashed_hollow"
    near_clip_m: float = .001
    line_width: int = 2
    tick_radius: int = 2

    def __post_init__(self):
        if not isinstance(self.enabled, bool) or self.origin != "ee_tcp" or self.frame != "action_world_xyz":
            raise ValueError("Metric cue requires EE TCP and world action XYZ")
        extents = np.asarray(self.axis_extent_m, dtype=np.float64)
        if extents.shape != (3, 2) or not np.isfinite(extents).all() or np.any(extents[:, 0] >= 0) or np.any(extents[:, 1] <= 0):
            raise ValueError("Each axis extent must be finite [negative, positive] metres")
        if not np.isfinite(self.tick_spacing_m) or self.tick_spacing_m <= 0:
            raise ValueError("Tick spacing must be positive finite metres")
        if not isinstance(self.major_tick_every, int) or self.major_tick_every < 1:
            raise ValueError("Major tick period must be a positive integer")
        if self.positive_style != "solid_filled" or self.negative_style != "dashed_hollow":
            raise ValueError("Unsupported direction style")
        if not np.isfinite(self.near_clip_m) or self.near_clip_m <= 0:
            raise ValueError("Near clip must be positive finite optical metres")
        if self.line_width < 1 or self.tick_radius < 1:
            raise ValueError("Stroke sizes must be positive")
        object.__setattr__(self, "axis_extent_m", tuple(tuple(float(x) for x in row) for row in extents))

    @classmethod
    def from_mapping(cls, value):
        fields = {key: value[key] for key in cls.__dataclass_fields__ if key in value}
        if "axis_extent_m" in fields and isinstance(fields["axis_extent_m"], dict):
            fields["axis_extent_m"] = tuple(fields["axis_extent_m"][axis] for axis in "xyz")
        return cls(**fields)


@dataclass(frozen=True)
class MetricCueTick:
    axis: int
    index: int
    world_m: np.ndarray
    uv: np.ndarray
    optical_depth_m: float
    visible: bool


@dataclass(frozen=True)
class MetricCueProjection:
    origin_uv: np.ndarray
    origin_optical_depth_m: float
    ticks: tuple[MetricCueTick, ...]
    diagnostics: dict


def project_metric_cue(ee_world_m, camera: Camera, config: MetricCueConfig):
    """Project signed, fixed-metre world ticks into one calibrated current view."""
    origin = finite(ee_world_m, (3,), "EE world position")
    origin_uv, origin_z = camera.project(origin[None])
    height, width = camera.resolution
    ticks = []
    diagnostics = {}
    for axis, name in enumerate("xyz"):
        low, high = config.axis_extent_m[axis]
        negative = range(-1, -int(np.floor((-low + 1e-10) / config.tick_spacing_m)) - 1, -1)
        positive = range(1, int(np.floor((high + 1e-10) / config.tick_spacing_m)) + 1)
        axis_ticks = []
        for index in (*negative, *positive):
            point = origin + np.eye(3)[axis] * (index * config.tick_spacing_m)
            uv, z = camera.project(point[None])
            uv, z = uv[0], float(z[0])
            visible = bool(z >= config.near_clip_m and np.isfinite(uv).all()
                           and 0 <= uv[0] <= width-1 and 0 <= uv[1] <= height-1)
            axis_ticks.append(MetricCueTick(axis, index, point, uv, z, visible))
        ticks.extend(axis_ticks)
        adjacent = []
        visible_length = 0.
        origin_visible = bool(origin_z[0] >= config.near_clip_m and np.isfinite(origin_uv[0]).all()
                              and 0 <= origin_uv[0, 0] <= width-1 and 0 <= origin_uv[0, 1] <= height-1)
        for sign in (-1, 1):
            previous = origin_uv[0]
            previous_valid = origin_visible
            for item in sorted((t for t in axis_ticks if t.index * sign > 0), key=lambda t: abs(t.index)):
                if previous_valid and item.visible:
                    distance = float(np.linalg.norm(item.uv - previous))
                    adjacent.append(distance)
                    visible_length += distance
                previous, previous_valid = item.uv, item.visible
        valid_count = sum(t.visible for t in axis_ticks)
        diagnostics[name] = dict(valid_ticks=valid_count, adjacent_tick_px=adjacent,
                                 visible_axis_length_px=visible_length,
                                 outside_fraction=(len(axis_ticks)-valid_count)/len(axis_ticks) if axis_ticks else 1.,
                                 degenerate=not adjacent or float(np.median(adjacent)) < 2.,
                                 behind_near_plane=sum(t.optical_depth_m < config.near_clip_m for t in axis_ticks))
    return MetricCueProjection(origin_uv[0], float(origin_z[0]), tuple(ticks), diagnostics)


def _draw_segment(draw, start, end, *, dashed, width):
    if not dashed:
        draw.line([tuple(start), tuple(end)], fill=255, width=width)
        return
    vector = np.asarray(end) - np.asarray(start)
    distance = float(np.linalg.norm(vector))
    if distance < 1e-8:
        return
    for begin in np.arange(0., distance, 9.):
        first = np.asarray(start) + vector * (begin / distance)
        last = np.asarray(start) + vector * (min(begin + 5., distance) / distance)
        draw.line([tuple(first), tuple(last)], fill=255, width=width)


def render_metric_cue(ee_world_m, camera: Camera, config: MetricCueConfig):
    """Return independent float32 RGB-semantic cue in [0,1] and diagnostics."""
    from rulervla.cues.adaptive_metric_ruler import clip_segment_to_viewport

    info = project_metric_cue(ee_world_m, camera, config)
    height, width = camera.resolution
    planes = [Image.new("L", (width, height), 0) for _ in range(3)]
    if not config.enabled:
        return np.zeros((height, width, 3), np.float32), info
    origin = finite(ee_world_m, (3,), "EE world position")
    for axis in range(3):
        draw = ImageDraw.Draw(planes[axis])
        # Complete the axis lines before cutting hollow tick interiors out of them.
        for sign in (-1, 1):
            previous_world = origin
            previous_z = info.origin_optical_depth_m
            for item in sorted((t for t in info.ticks if t.axis == axis and t.index * sign > 0),
                               key=lambda t: abs(t.index)):
                current_world, current_z = item.world_m, item.optical_depth_m
                if max(previous_z, current_z) >= config.near_clip_m:
                    start, end = previous_world.copy(), current_world.copy()
                    if min(previous_z, current_z) < config.near_clip_m:
                        fraction = (config.near_clip_m - previous_z) / (current_z - previous_z)
                        crossing = start + fraction * (end - start)
                        if previous_z < config.near_clip_m:
                            start = crossing
                        else:
                            end = crossing
                    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                        uv, _ = camera.project(np.stack((start, end)))
                    clipped = clip_segment_to_viewport(*uv, camera.resolution)
                    if clipped is not None:
                        _draw_segment(draw, *clipped, dashed=sign < 0, width=config.line_width)
                previous_world, previous_z = current_world, current_z
        for item in (t for t in info.ticks if t.axis == axis and t.visible):
            u, v = item.uv
            radius = config.tick_radius + (1 if abs(item.index) % config.major_tick_every == 0 else 0)
            bounds = (u-radius, v-radius, u+radius, v+radius)
            if item.index > 0:
                draw.ellipse(bounds, fill=255)
            else:
                draw.ellipse(bounds, fill=0, outline=255, width=1)
        if info.origin_optical_depth_m >= config.near_clip_m and np.isfinite(info.origin_uv).all():
            u, v = info.origin_uv
            draw.ellipse((u-2, v-2, u+2, v+2), fill=255)
    cue = np.stack([np.asarray(plane, dtype=np.float32)/255. for plane in planes], axis=-1)
    return cue, info


def preview_metric_cue(cue):
    """Human-readable red/X, green/Y, blue/Z preview, without changing RGB."""
    value = np.asarray(cue)
    if value.ndim != 3 or value.shape[-1] != 3 or not np.isfinite(value).all() or value.min() < 0 or value.max() > 1:
        raise ValueError("Metric cue must be finite HWC3 in [0,1]")
    return np.rint(value * 255).astype(np.uint8)
