"""Fixed physical units, perspective-consistent rendering on a canonical raster.

V2 reuses project_metric_ruler for all 3-D tick construction and projection.
Adaptation means perspective/viewport handling, never a changing metre unit.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from PIL import Image, ImageDraw
from rulervla.cues.metric_ruler import RulerProjection, project_metric_ruler
from rulervla.data.schema import Camera, NominalEEReference, finite

AXIS_COLORS = ((255, 48, 48), (48, 220, 48), (48, 100, 255))


@dataclass(frozen=True)
class AdaptiveRulerConfig:
    unit_m: float = 0.05
    tick_indices: tuple[int, ...] = (1, 2, 3)
    perspective_consistent: bool = True
    clip_to_viewport: bool = True
    depth_occlusion: bool = False
    near_clip_m: float = 0.001
    render_unit_label: bool = False

    def __post_init__(self):
        if not np.isfinite(self.unit_m) or self.unit_m <= 0:
            raise ValueError("unit_m must be positive finite metres")
        ticks = tuple(self.tick_indices)
        if not ticks or any(isinstance(n, bool) or not isinstance(n, (int, np.integer)) for n in ticks):
            raise ValueError("tick_indices must be positive increasing integers")
        if ticks != tuple(range(1, len(ticks) + 1)):
            raise ValueError("Consecutive indices 1..N preserve the shared tick interval")
        object.__setattr__(self, "tick_indices", ticks)
        if not np.isfinite(self.near_clip_m) or self.near_clip_m <= 0:
            raise ValueError("near_clip_m must be positive finite optical metres")
        for name in ("perspective_consistent", "clip_to_viewport", "depth_occlusion", "render_unit_label"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(name + " must be boolean")
        if not self.perspective_consistent:
            raise ValueError("V2 requires calibrated perspective projection")

    @property
    def lengths_m(self):
        return tuple(n * self.unit_m for n in self.tick_indices)


@dataclass(frozen=True)
class ProjectedSegment:
    axis: int
    tick_index: int
    start_uv: np.ndarray
    end_uv: np.ndarray
    near_plane_clipped: bool
    viewport_clipped: bool


@dataclass(frozen=True)
class AdaptiveRenderInfo:
    unit_m: float
    projection: RulerProjection
    segments: tuple[ProjectedSegment, ...]
    origin_optical_depth_m: float

    @property
    def lengths_m(self):
        return tuple((i + 1) * self.unit_m for i in range(self.projection.points_action_m.shape[1]))


def source_uv_to_policy_uv(uv, source_resolution, policy_resolution):
    """Exact integer-footprint PIL resize/pad mapping for pixel centres (H,W)."""
    dims = (*source_resolution, *policy_resolution)
    if len(dims) != 4 or any(not isinstance(n, (int, np.integer)) or n < 1 for n in dims):
        raise ValueError("Resolutions must contain positive integer H,W")
    source_h, source_w, target_h, target_w = dims
    ratio = max(source_w / target_w, source_h / target_h)
    resized_w, resized_h = int(source_w / ratio), int(source_h / ratio)
    if min(resized_w, resized_h) < 1:
        raise ValueError("Resize would collapse a raster dimension")
    left, top = max(0, int((target_w-resized_w)/2)), max(0, int((target_h-resized_h)/2))
    points = np.asarray(uv, np.float64)
    if points.shape[-1:] != (2,):
        raise ValueError("uv must end in two pixel coordinates")
    return (points + .5) * [resized_w/source_w, resized_h/source_h] - .5 + [left, top]


def orient_uv(uv, resolution, operation):
    """Map the same projected endpoints through the existing raster operation."""
    points = np.asarray(uv, np.float64).copy()
    if points.shape[-1:] != (2,):
        raise ValueError("uv must end in two pixel coordinates")
    h, w = resolution
    if operation in {"hflip", "rot180"}: points[..., 0] = w - 1 - points[..., 0]
    if operation in {"vflip", "rot180"}: points[..., 1] = h - 1 - points[..., 1]
    if operation not in {"identity", "hflip", "vflip", "rot180"}:
        raise ValueError("Unknown raster operation")
    return points


def clip_segment_to_viewport(start, end, resolution):
    """Liang-Barsky clipping; never reconnect across a nonfinite/hidden segment."""
    a, b = np.asarray(start, np.float64), np.asarray(end, np.float64)
    if not np.isfinite([a, b]).all(): return None
    h, w = resolution
    with np.errstate(over="ignore", invalid="ignore"):
        delta = b - a
    if not np.isfinite(delta).all(): return None
    low, high = 0., 1.
    for p, q in ((-delta[0], a[0]), (delta[0], w-1-a[0]),
                 (-delta[1], a[1]), (delta[1], h-1-a[1])):
        if p == 0:
            if q < 0: return None
        else:
            ratio = q / p
            if p < 0: low = max(low, ratio)
            else: high = min(high, ratio)
            if low > high: return None
    first, last = a + low*delta, a + high*delta
    return np.clip(first, [0, 0], [w-1, h-1]), np.clip(last, [0, 0], [w-1, h-1])


def project_adaptive_metric_ruler(ee_action_m, camera: Camera, config=AdaptiveRulerConfig()):
    origin = finite(ee_action_m, (3,), "EE world XYZ")
    reference = NominalEEReference(origin)
    # The original correct implementation is the single source of 3-D/2-D ticks.
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        raw = project_metric_ruler(reference, camera, config.lengths_m)
        _, origin_z = camera.project(origin[None])
    h, w = camera.resolution
    valid = (raw.optical_depth_m >= config.near_clip_m) & np.isfinite(raw.ticks_uv).all(axis=-1)
    valid &= ((raw.ticks_uv[..., 0] >= 0) & (raw.ticks_uv[..., 0] <= w-1) &
              (raw.ticks_uv[..., 1] >= 0) & (raw.ticks_uv[..., 1] <= h-1))
    projection = RulerProjection(raw.points_action_m, raw.origin_uv, raw.ticks_uv, raw.optical_depth_m, valid)
    segments = []
    for axis in range(3):
        points = np.concatenate([origin[None], raw.points_action_m[axis]])
        depths = np.concatenate([origin_z, raw.optical_depth_m[axis]])
        for i in range(len(config.tick_indices)):
            a, b = points[i].copy(), points[i+1].copy()
            za, zb = depths[i:i+2]
            if max(za, zb) < config.near_clip_m: continue
            clipped = min(za, zb) < config.near_clip_m
            if clipped:
                fraction = (config.near_clip_m - za) / (zb - za)
                crossing = a + fraction * (b - a)
                if za < config.near_clip_m: a = crossing
                else: b = crossing
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                uv, z = camera.project(np.stack([a, b]))
            if not np.isfinite(uv).all() or not np.all(z > 0): continue
            visible = clip_segment_to_viewport(*uv, camera.resolution)
            if visible is None: continue
            outside = not (np.all((uv >= [0, 0]) & (uv <= [w-1, h-1])))
            if not config.clip_to_viewport and outside: continue
            segments.append(ProjectedSegment(axis, config.tick_indices[i], *visible, clipped, outside))
    return AdaptiveRenderInfo(config.unit_m, projection, tuple(segments), float(origin_z[0]))


def draw_metric_scale_label(rgb, unit_m):
    """Optional constant-unit annotation, after orientation; off by default."""
    if not np.isfinite(unit_m) or unit_m <= 0: raise ValueError("Invalid metre unit")
    image = Image.fromarray(np.asarray(rgb, np.uint8).copy())
    draw = ImageDraw.Draw(image)
    label = f"1 tick = {unit_m * 100:g} cm"
    bounds = draw.textbbox((0, 0), label)
    width, height = bounds[2]-bounds[0], bounds[3]-bounds[1]
    if image.width >= width+8 and image.height >= height+8:
        draw.rectangle((2, 2, width+7, height+7), fill=(28, 28, 28))
        draw.text((5, 4-bounds[1]), label, fill=(255, 255, 255))
    return np.asarray(image).copy()


def render_adaptive_metric_ruler(rgb, ee_action_m, camera, config=AdaptiveRulerConfig(), *,
                                  line_width=2, tick_radius=2, depth_m=None):
    image = np.asarray(rgb)
    if image.dtype != np.uint8 or image.shape != camera.resolution + (3,):
        raise ValueError("RGB must match the calibrated canonical camera raster")
    if line_width < 1 or tick_radius < 1: raise ValueError("Invalid visual stroke size")
    if config.depth_occlusion:
        raise NotImplementedError("Depth occlusion is an optional, unreleased interface")
    # Background depth (including invalid pixels) cannot change ruler geometry.
    info = project_adaptive_metric_ruler(ee_action_m, camera, config)
    canvas = Image.fromarray(image.copy()); draw = ImageDraw.Draw(canvas)
    for axis, color in enumerate(AXIS_COLORS):
        for index, (uv, valid) in enumerate(zip(info.projection.ticks_uv[axis], info.projection.valid[axis])):
            for segment in info.segments:
                if segment.axis == axis and segment.tick_index == config.tick_indices[index]:
                    draw.line([tuple(segment.start_uv), tuple(segment.end_uv)], fill=color, width=line_width)
            if valid:
                u, v = uv
                draw.ellipse((u-tick_radius, v-tick_radius, u+tick_radius, v+tick_radius), fill=color)
    output = np.asarray(canvas).copy()
    if config.render_unit_label: output = draw_metric_scale_label(output, config.unit_m)
    return output, info
