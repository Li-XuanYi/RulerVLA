"""Synchronized two-view RGB and separate metric cue preparation."""
from dataclasses import dataclass
import numpy as np

from rulervla.adapters.libero import LiberoAdapter
from rulervla.adapters.main_comparison import MAIN_IMAGE_KEYS
from rulervla.adapters.openpi import resize_float_map, resize_rgb
from rulervla.cues.metric_ruler import MetricCueConfig, render_metric_cue
from rulervla.cues.metric_prompt import build_scale_prompt
from rulervla.cues.motion_echo import MotionEchoConfig, MotionHistory, compose_cues, render_motion_echo
from rulervla.data.schema import validate_sample
from rulervla.geometry.action_coord import raster


class MetricViewAdapter:
    def __init__(self, geometry_contract, ruler: MetricCueConfig,
                 echo: MotionEchoConfig = MotionEchoConfig()):
        self.geometry = geometry_contract
        self.libero = LiberoAdapter(geometry_contract)
        self.ruler = ruler
        self.echo = echo
        self.last_diagnostics = None

    def prepare(self, sample, *, history: MotionHistory | None = None):
        validate_sample(sample)
        if self.echo.enabled and history is not None and history.valid:
            if (history.current_step - history.past_step != self.echo.history_steps
                    or not np.allclose(history.current_m, sample["ee_world_m"], atol=2e-6, rtol=0)
                    or ("episode_index" in sample and history.episode_index != int(sample["episode_index"]))):
                raise ValueError("Echo history must match the current observation and episode")
        to_policy = self.geometry["canonical_to_policy"]
        prepared = {}
        diagnostics = {}
        for view, rgb_key, image_key, cue_key in (
            ("agentview", "agentview_rgb", "observation/image", "observation/agent_cue"),
            ("wrist", "wrist_rgb", "observation/wrist_image", "observation/wrist_cue"),
        ):
            canonical = raster(sample[rgb_key], self.geometry["stored_rgb_to_canonical"])
            camera = self.libero.camera(sample, view)
            ruler_cue, projection = render_metric_cue(sample["ee_world_m"], camera, self.ruler)
            if history is None:
                history = MotionHistory(False, None, None, None, None, None, "no_history")
            echo_cue, echo_info = render_motion_echo(history, camera, self.echo)
            cue = compose_cues(ruler_cue, echo_cue, enabled=self.echo.enabled)
            prepared[image_key] = resize_rgb(raster(canonical, to_policy))
            prepared[cue_key] = resize_float_map(raster(cue, to_policy))
            diagnostics[view] = {**projection.diagnostics, "echo": echo_info}
        prepared["observation/state"] = np.asarray(sample["state8"], np.float32).copy()
        prepared["prompt"] = build_scale_prompt(sample["task"], self.ruler,
            echo_history_steps=self.echo.history_steps if self.echo.enabled else None,
            echo_valid=history.valid if self.echo.enabled else None)
        if "actions" in sample:
            prepared["actions"] = np.asarray(sample["actions"], np.float32).copy()
        prepared["diagnostics"] = diagnostics
        self.last_diagnostics = diagnostics
        return prepared


@dataclass(frozen=True)
class SixChannelInputs:
    """Keep four historical slots while filling two synchronized six-channel views."""

    model_type: object = None
    base_transform: object = None

    def __call__(self, data):
        transform = self.base_transform
        if transform is None:
            from openpi.policies.libero_policy import LiberoInputs
            transform = LiberoInputs(model_type=self.model_type)
        base = transform(data)
        images = {}
        for rgb_key, cue_key, slot in (
            ("observation/image", "observation/agent_cue", MAIN_IMAGE_KEYS[0]),
            ("observation/wrist_image", "observation/wrist_cue", MAIN_IMAGE_KEYS[1]),
        ):
            rgb = np.asarray(data[rgb_key])
            cue = np.asarray(data[cue_key])
            if rgb.shape != (224, 224, 3) or rgb.dtype != np.uint8:
                raise ValueError(f"{rgb_key} must be uint8[224,224,3]")
            if (cue.shape != rgb.shape or cue.dtype != np.float32 or not np.isfinite(cue).all()
                    or cue.min() < 0 or cue.max() > 1):
                raise ValueError(f"{cue_key} must be finite float32[224,224,3] in [0,1]")
            normalized_rgb = rgb.astype(np.float32) / 255. * 2. - 1.
            images[slot] = np.concatenate((normalized_rgb, cue), axis=-1)
        blank = np.zeros((224, 224, 6), np.float32)
        images[MAIN_IMAGE_KEYS[2]] = blank.copy()
        images[MAIN_IMAGE_KEYS[3]] = blank.copy()
        base["image"] = images
        base["image_mask"] = {key: np.bool_(index < 2) for index, key in enumerate(MAIN_IMAGE_KEYS)}
        return base
