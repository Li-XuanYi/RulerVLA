from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import yaml

from rulervla.geometry.action_coord import validate_contract
from rulervla.geometry.correction import CorrectionConfig
from rulervla.geometry.reliability import ReliabilityConfig


def validate_method(method):
    if method["representation"] not in {"neutral", "depth", "camera_xyz", "actioncoord"}:
        raise ValueError("Unknown representation")
    for module in ("visual_anchor", "correction", "reliability", "metric_ruler"):
        if not isinstance(method[module]["enabled"], bool):
            raise ValueError(f"{module}.enabled must be boolean")
    anchor = method["visual_anchor"]
    if anchor["backend"] not in {"simulator_gt", "learned"}:
        raise ValueError("Unknown anchor backend")
    if not np.isfinite(anchor["occlusion_tolerance_m"]) or anchor["occlusion_tolerance_m"] <= 0:
        raise ValueError("Positive occlusion tolerance required")
    if method["correction"]["mode"] != "translation_huber":
        raise ValueError("Only shared translation Huber correction is implemented")
    CorrectionConfig(**{k: v for k, v in method["correction"].items() if k not in {"enabled", "mode"}})
    ReliabilityConfig(**{k: v for k, v in method["reliability"].items() if k != "enabled"})
    correction = method["correction"]["enabled"]
    if correction and (not anchor["enabled"] or method["representation"] != "actioncoord"):
        raise ValueError("Correction requires anchors and ActionCoord")
    if (method["reliability"]["enabled"] or method["metric_ruler"]["enabled"]) and not correction:
        raise ValueError("Reliability and anchored ruler require reference correction")
    ruler = method["metric_ruler"]
    lengths = np.asarray(ruler["lengths_m"], dtype=float)
    if (lengths.ndim != 1 or len(lengths) == 0 or not np.isfinite(lengths).all()
            or np.any(lengths <= 0) or np.any(np.diff(lengths) <= 0)
            or ruler["line_width"] < 1 or ruler["tick_radius"] < 1):
        raise ValueError("Invalid physical ruler specification")


def load_config(path):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text())
    config["geometry"] = json.loads((path.parent / config["geometry_contract"]).read_text())
    config["encoding_config"] = json.loads((path.parent / config["encoding"]).read_text())
    validate_contract(config["geometry"])
    encoding = config["encoding_config"]
    if (encoding.get("frozen") is not True or encoding.get("radius_m") != 3.25
            or encoding.get("dtype") != "float32"):
        raise ValueError("Shared encoding must remain frozen at R=3.25m, float32")
    validate_method(config["method"])
    return config


def profile(config, name):
    flags = {"V03": (False, False, False), "V05-C": (True, False, False),
             "V05-CR": (True, True, False), "V05-Full": (True, True, True)}
    if name not in flags:
        raise ValueError("Unknown controlled profile")
    out = deepcopy(config)
    correction, reliability, ruler = flags[name]
    out["method"]["visual_anchor"]["enabled"] = correction
    out["method"]["correction"]["enabled"] = correction
    out["method"]["reliability"]["enabled"] = reliability
    out["method"]["metric_ruler"]["enabled"] = ruler
    validate_method(out["method"])
    return out
