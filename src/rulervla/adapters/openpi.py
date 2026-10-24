"""Three separate RGB/RGB/float32 spatial image slots, preserving V0.3 parsing."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from rulervla.adapters.libero import LiberoAdapter
from rulervla.config import validate_method
from rulervla.cues.metric_ruler import render_metric_ruler
from rulervla.data.schema import CorrectedEEReference, NominalEEReference, rotation_from_xyzw, validate_sample
from rulervla.geometry.action_coord import canonical_geometry, encode, raster
from rulervla.geometry.correction import CorrectionConfig, apply_correction, correct_from_anchors
from rulervla.geometry.reliability import ReliabilityConfig, geometry_reliability, raster_weights
from rulervla.geometry.visual_anchor import LearnedAnchorProvider, SimulatorGTAnchorProvider


def _resize_with_pad(image, height, width, method):
    """Frozen OpenPI PIL resize/pad footprint, without importing the VLA runtime."""
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return image
    ratio = max(cur_width / width, cur_height / height)
    resized = image.resize((int(cur_width / ratio), int(cur_height / ratio)), resample=method)
    canvas = Image.new(resized.mode, (width, height), 0)
    canvas.paste(resized, (max(0, int((width-resized.width)/2)), max(0, int((height-resized.height)/2))))
    return canvas


def resize_rgb(rgb, height=224, width=224):
    # The frozen code calls OpenPI resize_with_pad with its actual BILINEAR
    # default (enum 2). The geometry JSON's Lanczos label is inconsistent;
    # preserve executable behavior and verify equality against the V0.3 parser.
    return np.asarray(_resize_with_pad(Image.fromarray(rgb), height, width, Image.Resampling.BILINEAR)).copy()


def resize_float_map(spatial, height=224, width=224):
    spatial = np.asarray(spatial, dtype=np.float32)
    return np.stack([
        np.asarray(_resize_with_pad(Image.fromarray(spatial[..., i]), height, width,
                                    Image.Resampling.NEAREST), dtype=np.float32)
        for i in range(3)
    ], axis=-1)


class OpenPIAdapter:
    def __init__(self, config):
        validate_method(config["method"])
        self.config = config
        self.libero = LiberoAdapter(config["geometry"])
        self.last_diagnostics = {}

    def prepare(self, sample, *, anchor_provider=None, local_points_ee_m=None):
        validate_sample(sample)
        method = self.config["method"]
        maps = canonical_geometry(sample, self.config["geometry"])
        rgb = maps["rgb"]
        selected = method["representation"]
        spatial_m = np.zeros_like(maps["actioncoord"]) if selected == "neutral" else maps[selected].copy()
        result = reliability = reference = None
        if method["visual_anchor"]["enabled"] and method["visual_anchor"]["backend"] == "learned":
            LearnedAnchorProvider().detect(rgb)
        if method["correction"]["enabled"]:
            if not isinstance(anchor_provider, SimulatorGTAnchorProvider) or local_points_ee_m is None:
                raise ValueError("Bind same-state SimulatorGTAnchorProvider and rigid robot surface points")
            nominal_camera = self.libero.camera(sample)
            anchors = anchor_provider.detect(rgb)
            rotation = rotation_from_xyzw(sample["eef_quaternion_xyzw"])
            fit_config = CorrectionConfig(**{k: v for k, v in method["correction"].items()
                                             if k not in {"enabled", "mode"}})
            result = correct_from_anchors(anchors, local_points_ee_m, maps["depth"], nominal_camera,
                                          sample["ee_world_m"], rotation, fit_config)
            spatial_m = apply_correction(spatial_m, maps["valid"], result)
            ref_valid = result.accepted
            if method["reliability"]["enabled"]:
                cfg = ReliabilityConfig(**{k: v for k, v in method["reliability"].items() if k != "enabled"})
                reliability = geometry_reliability(result, anchors.confidence, maps["valid"], cfg)
                ref_valid = reliability.correction_valid
            reference = CorrectedEEReference(np.asarray(sample["ee_world_m"]),
                                             result.bias_xyz_m if ref_valid else np.zeros(3), ref_valid)
        ruler = method["metric_ruler"]
        ruler_rendered = False
        if ruler["enabled"]:
            if ruler.get("reference", "corrected_ee") == "nominal_ee":
                reference = NominalEEReference(sample["ee_world_m"])
            if reference is not None and (isinstance(reference, NominalEEReference) or reference.accepted):
                rgb = render_metric_ruler(rgb, reference, self.libero.camera(sample), ruler["lengths_m"],
                                          line_width=ruler["line_width"], tick_radius=ruler["tick_radius"])
                ruler_rendered = True
        spatial = encode(spatial_m, maps["valid"], self.config["encoding_config"])
        if reliability is not None:
            weights = raster_weights(reliability, spatial.shape[:2], method["reliability"]["patch_size"])
            # Blend encoded coordinates toward V0.3 neutral zero, without adding channels.
            spatial *= weights[..., None]
        to_policy = self.config["geometry"]["canonical_to_policy"]
        output = {
            "observation/image": resize_rgb(raster(rgb, to_policy)),
            "observation/wrist_image": resize_rgb(raster(maps["wrist"], to_policy)),
            "observation/spatial_image": resize_float_map(raster(spatial, to_policy)),
            "observation/state": np.asarray(sample["state8"], dtype=np.float32).copy(),
            "prompt": sample["task"],
        }
        if "actions" in sample:
            actions = np.asarray(sample["actions"], dtype=np.float32)
            if actions.shape != (10, 7) or not np.isfinite(actions).all():
                raise ValueError("Frozen action chunk must be finite float32[10,7]")
            output["actions"] = actions.copy()
        self.last_diagnostics = {"correction": result, "reliability": reliability,
                                 "reference": reference, "corrected_actioncoord_m": spatial_m,
                                 "ruler_rendered": ruler_rendered,
                                 "reference_source": ruler.get("reference", "corrected_ee") if ruler_rendered else None,
                                 # This is diagnostic metadata, not a model-side per-pixel mask.
                                 "valid_depth_fraction": float(np.mean(maps["valid"])),
                                 "invalid_depth_pixels": int(np.count_nonzero(~maps["valid"])),
                                 "per_pixel_validity_forwarded_to_policy": False}
        validate_prepared(output)
        return output


def validate_prepared(data):
    for key in ("observation/image", "observation/wrist_image", "observation/spatial_image"):
        image = np.asarray(data[key])
        expected = np.float32 if key.endswith("spatial_image") else np.uint8
        if image.shape != (224, 224, 3) or image.dtype != expected or not np.isfinite(image).all():
            raise ValueError(f"Invalid policy image {key}: {image.shape}, {image.dtype}")
    if np.max(np.abs(data["observation/spatial_image"])) > 1 + 1e-6:
        raise ValueError("Spatial encoding outside [-1,1]")


@dataclass(frozen=True)
class OpenPIThirdSlotInputs:
    """Drop-in equivalent of frozen V03LiberoInputs with optional lazy OpenPI import."""
    model_type: object = None
    base_transform: object = None

    def __call__(self, data):
        validate_prepared(data)
        transform = self.base_transform
        if transform is None:
            from openpi.policies.libero_policy import LiberoInputs
            transform = LiberoInputs(model_type=self.model_type)
        output = transform(data)
        output["image"]["right_wrist_0_rgb"] = data["observation/spatial_image"]
        output["image_mask"]["right_wrist_0_rgb"] = np.bool_(True)
        if len(output["image"]) != 3 or any(np.asarray(x).shape != (224, 224, 3)
                                             for x in output["image"].values()):
            raise ValueError("Expected exactly three three-channel OpenPI image slots")
        return output
