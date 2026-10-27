from dataclasses import dataclass

import numpy as np

from rulervla.data.schema import Camera


def source_uv_to_policy_uv(uv, source_resolution, policy_resolution):
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


def resize_camera(camera, resolution):
    reference = source_uv_to_policy_uv([[0., 0.], [1., 1.]], camera.resolution, resolution)
    sx, sy = reference[1] - reference[0]
    affine = np.array([[sx, 0, reference[0, 0]], [0, sy, reference[0, 1]], [0, 0, 1.]])
    return Camera(affine @ camera.intrinsic, camera.camera_to_action, tuple(resolution))


def orient_uv(uv, resolution, operation):
    points = np.asarray(uv, np.float64).copy()
    if points.shape[-1:] != (2,):
        raise ValueError("uv must end in two pixel coordinates")
    h, w = resolution
    if operation in {"hflip", "rot180"}:
        points[..., 0] = w - 1 - points[..., 0]
    if operation in {"vflip", "rot180"}:
        points[..., 1] = h - 1 - points[..., 1]
    if operation not in {"identity", "hflip", "vflip", "rot180"}:
        raise ValueError("Unknown raster operation")
    return points


def clip_segment_to_viewport(start, end, resolution):
    a, b = np.asarray(start, np.float64), np.asarray(end, np.float64)
    if not np.isfinite([a, b]).all():
        return None
    h, w = resolution
    with np.errstate(over="ignore", invalid="ignore"):
        delta = b - a
    if not np.isfinite(delta).all():
        return None
    low, high = 0., 1.
    for p, q in ((-delta[0], a[0]), (delta[0], w-1-a[0]),
                 (-delta[1], a[1]), (delta[1], h-1-a[1])):
        if p == 0:
            if q < 0:
                return None
        else:
            ratio = q / p
            if p < 0:
                low = max(low, ratio)
            else:
                high = min(high, ratio)
            if low > high:
                return None
    first, last = a + low*delta, a + high*delta
    return np.clip(first, [0, 0], [w-1, h-1]), np.clip(last, [0, 0], [w-1, h-1])


@dataclass(frozen=True)
class VisibleSegment:
    start_uv: np.ndarray
    end_uv: np.ndarray
    near_clipped: bool
    viewport_clipped: bool


def project_visible_segment(start, end, camera, near_clip_m):
    points = np.stack((start, end)).copy()
    _, depth = camera.project(points)
    if np.max(depth) < near_clip_m:
        return None
    near_clipped = bool(np.min(depth) < near_clip_m)
    if near_clipped:
        fraction = (near_clip_m - depth[0]) / (depth[1] - depth[0])
        points[int(depth[1] < near_clip_m)] = points[0] + fraction * (points[1] - points[0])
    uv, _ = camera.project(points)
    clipped = clip_segment_to_viewport(*uv, camera.resolution)
    if clipped is None:
        return None
    return VisibleSegment(*clipped, near_clipped, not np.allclose(uv, clipped, atol=1e-7, rtol=0))
