from dataclasses import dataclass

import cv2
import numpy as np

from rulervla.geometry.camera import project_visible_segment
from rulervla.data.schema import Camera, finite


ECHO_RENDER_CONTRACT = "antialiased_channel_max"


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
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 1
               for value in (self.arrow_width, self.arrow_head_px)):
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
        if not isinstance(control_step, int) or control_step < 0:
            raise ValueError("Control step must be a nonnegative integer")
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


class MetricInferenceHistory:
    def __init__(self, config: MotionEchoConfig):
        if not config.enabled:
            raise ValueError("Online history requires enabled Motion Echo")
        self.buffer = EpisodeHistoryBuffer(config.history_steps)

    def reset_episode(self, episode_index):
        self.buffer.reset(int(episode_index))

    def observe_sample(self, sample):
        episode_index = int(sample["episode_index"])
        step = int(sample["frame_index"])
        if self.buffer.episode_index != episode_index:
            raise ValueError("Reset history before observing a new episode")
        if self.buffer._last_step is None:
            if step != 0:
                raise ValueError("Episode history must begin at control step zero")
        elif step != self.buffer._last_step + 1:
            raise ValueError("Observe every environment control step")
        self.buffer.observe(step, sample["ee_world_m"])

    def for_sample(self, sample):
        history = self.buffer.history()
        if (history.episode_index != int(sample["episode_index"])
                or history.current_step != int(sample["frame_index"])
                or history.current_m is None
                or not np.allclose(history.current_m, sample["ee_world_m"], atol=2e-6, rtol=0)):
            raise ValueError("Inference history must match the current observation")
        return history


def render_motion_echo(history: MotionHistory, camera: Camera, config: MotionEchoConfig):
    height, width = camera.resolution
    blank = np.zeros((height, width, 3), np.float32)
    if not config.enabled:
        return blank, {"rendered": False, "reason": "disabled"}
    if not history.valid:
        return blank, {"rendered": False, "history_valid": False,
                       "reason": history.reason or "invalid_history"}
    displacement = history.displacement_m
    if np.linalg.norm(displacement) < config.min_displacement_m:
        return blank, dict(rendered=False, history_valid=True, stationary=True, reason="valid_stationary",
                           displacement_m=displacement.tolist(), displacement_norm_m=float(np.linalg.norm(displacement)))
    uv, depth = camera.project(np.stack((history.past_m, history.current_m)))
    visible = ((depth >= config.near_clip_m) & np.isfinite(uv).all(axis=1)
               & (uv >= [0, 0]).all(axis=1) & (uv <= [width-1, height-1]).all(axis=1))
    segment = project_visible_segment(history.past_m, history.current_m, camera, config.near_clip_m)
    info = dict(rendered=False, history_valid=True, stationary=False, head_rendered=False,
        past_uv=uv[0].tolist() if np.isfinite(uv[0]).all() else None,
        current_uv=uv[1].tolist() if np.isfinite(uv[1]).all() else None,
        optical_depth_m=depth.tolist(), displacement_m=displacement.tolist(),
        displacement_norm_m=float(np.linalg.norm(displacement)), current_visible=bool(visible[1]),
        past_visible=bool(visible[0]), near_clipped=bool(np.any(depth < config.near_clip_m)),
        viewport_clipped=False, endpoints_uv=None, visible_length_px=0., head_length_px=0.)
    if segment is None:
        info["reason"] = "behind_near_plane" if np.max(depth) < config.near_clip_m else "outside_view"
        return blank, info
    start, end = segment.start_uv, segment.end_uv
    info.update(viewport_clipped=segment.viewport_clipped, endpoints_uv=[start.tolist(), end.tolist()])
    direction = end - start
    norm = float(np.linalg.norm(direction))
    info["visible_length_px"] = norm
    if norm < 1:
        info["reason"] = "subpixel_projection"
        return blank, info
    direction /= norm
    mask = np.zeros((height, width), np.uint8)
    pixels = np.rint(np.stack((start, end)) * 256).astype(np.int32)
    cv2.line(mask, tuple(pixels[0]), tuple(pixels[1]), 255, config.arrow_width, cv2.LINE_AA, 8)
    # 头部始终锚定真实当前端点；裁剪交点只用于绘制可见线段。
    if visible[1]:
        length = min(config.arrow_head_px, norm * .4)
        normal = np.array([-direction[1], direction[0]])
        back = uv[1] - direction * length
        triangle = np.stack((uv[1], back + normal*length/2, back - normal*length/2))
        cv2.fillConvexPoly(mask, np.rint(triangle*256).astype(np.int32), 255, cv2.LINE_AA, 8)
        info.update(head_rendered=True, head_length_px=length)
    yellow = mask.astype(np.float32) / 255.
    blank[..., 0] = yellow
    blank[..., 1] = yellow
    info.update(rendered=True, reason="visible")
    return blank, info


def compose_cues(ruler, echo, *, enabled):
    ruler = np.asarray(ruler)
    echo = np.asarray(echo)
    if (ruler.dtype != np.float32 or echo.dtype != np.float32 or ruler.shape != echo.shape
            or ruler.ndim != 3 or ruler.shape[-1] != 3 or not np.isfinite(ruler).all()
            or not np.isfinite(echo).all() or ruler.min() < 0 or ruler.max() > 1
            or echo.min() < 0 or echo.max() > 1):
        raise ValueError("Composer needs matching finite float32 HWC3 cues in [0,1]")
    return np.maximum(ruler, echo) if enabled else ruler.copy()
