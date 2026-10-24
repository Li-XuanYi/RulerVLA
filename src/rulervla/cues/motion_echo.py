"""Past measured EE displacement projected into the current camera view."""
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from rulervla.cues.adaptive_metric_ruler import clip_segment_to_viewport
from rulervla.data.schema import Camera, finite, observation_index


@dataclass(frozen=True)
class MotionEchoConfig:
    enabled: bool = False
    history_steps: int = 5
    min_displacement_m: float = .003
    near_clip_m: float = .001
    arrow_width: int = 2
    arrow_head_px: int = 6

    def __post_init__(self):
        if not isinstance(self.enabled, bool) or not isinstance(self.history_steps, int) or self.history_steps < 1:
            raise ValueError("Echo needs a boolean switch and positive control-step history")
        if (not np.isfinite(self.min_displacement_m) or self.min_displacement_m < 0
                or not np.isfinite(self.near_clip_m) or self.near_clip_m <= 0):
            raise ValueError("Invalid echo physical thresholds")
        if self.arrow_width < 1 or self.arrow_head_px < 1:
            raise ValueError("Invalid echo stroke")


@dataclass(frozen=True)
class MotionHistory:
    valid: bool
    episode_index: int | str | None
    current_step: int | None
    past_step: int | None
    current_m: np.ndarray | None
    past_m: np.ndarray | None
    reason: str | None = None

    @property
    def displacement_m(self):
        if not self.valid:
            raise ValueError("Invalid history has no measured displacement")
        return finite(self.current_m, (3,), "Current EE") - finite(self.past_m, (3,), "Past EE")


class EpisodeHistoryBuffer:
    """One observation per environment control step, never a predicted chunk."""

    def __init__(self, history_steps=5):
        if not isinstance(history_steps, int) or history_steps < 1:
            raise ValueError("history_steps must be positive")
        self.history_steps = history_steps
        self.episode_index = None
        self._positions = {}
        self._last_step = None

    def reset(self, episode_index):
        self.episode_index = episode_index
        self._positions.clear()
        self._last_step = None

    def observe(self, control_step, ee_world_m):
        if self.episode_index is None:
            raise ValueError("Reset history at episode start")
        control_step = observation_index(control_step)
        position = finite(ee_world_m, (3,), "Measured EE").copy()
        if self._last_step is not None and control_step != self._last_step + 1:
            self._positions.clear()
        self._positions[control_step] = position
        self._last_step = control_step
        for step in tuple(self._positions):
            if step < control_step - self.history_steps:
                del self._positions[step]

    def history(self):
        if self._last_step is None:
            return MotionHistory(False, self.episode_index, None, None, None, None, "no_observation")
        target = self._last_step - self.history_steps
        current = self._positions[self._last_step]
        if target not in self._positions:
            return MotionHistory(False, self.episode_index, self._last_step, target,
                                 current.copy(), None, "insufficient_or_discontinuous_history")
        return MotionHistory(True, self.episode_index, self._last_step, target,
                             current.copy(), self._positions[target].copy())


def render_motion_echo(history: MotionHistory, camera: Camera, config: MotionEchoConfig):
    height, width = camera.resolution
    blank = np.zeros((height, width, 3), np.float32)
    if not config.enabled:
        return blank, {"rendered": False, "arrowhead_drawn": False, "reason": "disabled"}
    if not history.valid:
        return blank, {"rendered": False, "arrowhead_drawn": False,
                       "reason": history.reason or "invalid_history"}
    displacement = history.displacement_m
    if np.linalg.norm(displacement) < config.min_displacement_m:
        return blank, {"rendered": False, "arrowhead_drawn": False, "stationary": True,
                       "reason": "valid_stationary", "displacement_m": displacement.tolist()}
    points = np.asarray((history.past_m, history.current_m), dtype=np.float64)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        uv, depth = camera.project(points)
    info = {"rendered": False, "arrowhead_drawn": False, "stationary": False, "past_uv": uv[0].tolist(),
            "current_uv": uv[1].tolist(), "displacement_m": displacement.tolist()}
    if not np.isfinite(depth).all():
        info["reason"] = "nonfinite_projection"
        return blank, info
    if np.max(depth) < config.near_clip_m:
        info["reason"] = "behind_near_plane"
        return blank, info

    # Endpoint visibility is checked before clipping: a boundary intersection
    # must never acquire an arrowhead that impersonates the measured current EE.
    current_visible = bool(depth[1] >= config.near_clip_m and np.isfinite(uv[1]).all()
                           and 0 <= uv[1, 0] <= width-1 and 0 <= uv[1, 1] <= height-1)
    past_near_clipped = bool(depth[0] < config.near_clip_m)
    current_near_clipped = bool(depth[1] < config.near_clip_m)
    if past_near_clipped or current_near_clipped:
        fraction = (config.near_clip_m-depth[0]) / (depth[1]-depth[0])
        crossing = points[0] + fraction * (points[1]-points[0])
        points[0 if past_near_clipped else 1] = crossing
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        segment_uv, _ = camera.project(points)
    info.update(past_near_plane_clipped=past_near_clipped,
                current_near_plane_clipped=current_near_clipped,
                clipped_past_uv=segment_uv[0].tolist(), clipped_current_uv=segment_uv[1].tolist())
    if not np.isfinite(segment_uv).all():
        info["reason"] = "nonfinite_projection"
        return blank, info
    clipped = clip_segment_to_viewport(*segment_uv, camera.resolution)
    if clipped is None:
        info["reason"] = "outside_view"
        return blank, info
    start, end = clipped
    direction = end - start
    norm = float(np.linalg.norm(direction))
    if norm < 1:
        info["reason"] = "subpixel_projection"
        return blank, info
    direction /= norm
    normal = np.array([-direction[1], direction[0]])
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.line([tuple(start), tuple(end)], fill=255, width=config.arrow_width)
    if current_visible:
        back = end - direction * min(config.arrow_head_px, norm)
        draw.polygon([tuple(end), tuple(back + normal * config.arrow_head_px/2),
                      tuple(back - normal * config.arrow_head_px/2)], fill=255)
        info["arrowhead_drawn"] = True
    yellow = np.asarray(mask, dtype=np.float32) / 255.
    blank[..., 0] = yellow
    blank[..., 1] = yellow
    info["rendered"] = True
    if not current_visible:
        info["reason"] = ("current_endpoint_behind_near_plane" if current_near_clipped
                          else "current_endpoint_outside_view")
    return blank, info


def compose_cues(ruler, echo, *, enabled):
    ruler = np.asarray(ruler)
    echo = np.asarray(echo)
    if (ruler.dtype != np.float32 or echo.dtype != np.float32 or ruler.shape != echo.shape
            or ruler.ndim != 3 or ruler.shape[-1] != 3 or not np.isfinite(ruler).all()
            or not np.isfinite(echo).all() or ruler.min() < 0 or ruler.max() > 1
            or echo.min() < 0 or echo.max() > 1):
        raise ValueError("Composer needs matching finite float32 HWC3 cues in [0,1]")
    if not enabled:
        return ruler.copy()
    # Yellow remains the Echo color; its zero blue channel must not erase Z.
    return np.maximum(ruler, echo)
