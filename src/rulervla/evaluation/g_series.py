"""Frozen geometry boundary interventions; never changes production estimators."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from rulervla.adapters.libero import LiberoAdapter
from rulervla.data.schema import Camera, rotation_from_xyzw
from rulervla.geometry.action_coord import canonical_geometry
from rulervla.geometry.correction import CorrectionConfig, apply_correction, correct_from_anchors
from rulervla.geometry.reliability import ReliabilityConfig, geometry_reliability
from rulervla.geometry.visual_anchor import AnchorObservation, sample_depth

GROUPS = {
    "G01": ("G01_translation", "Can shared translation be recovered?"),
    "G02": ("G02_anchor_count", "When do fewer anchors require rejection?"),
    "G03": ("G03_anchor_outlier", "Does Huber improve over the plain mean?"),
    "G04": ("G04_rotation", "What is the translation-only rotation boundary?"),
    "G05": ("G05_depth", "How does measured depth degradation affect correction?"),
    "G06": ("G06_anchor_depth_bias", "Can consistent anchor-local bias move a clean scene?"),
    "G07": ("G07_target_depth_bias", "Can anchor evidence detect target-only depth bias?"),
}


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def array_hash(array):
    a = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(json.dumps(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def unit_direction(name, design):
    vector = np.asarray(design["mixed_direction"] if name == "mixed"
                        else np.eye(3)["xyz".index(name)], dtype=float)
    return vector / np.linalg.norm(vector)


def conditions(design):
    matrix = [dict(experiment_id="G01", condition="clean_control", severity=0, direction="mixed", kind="translation")]
    for mm in design["translation_mm"]:
        for axis in design["translation_directions"]:
            matrix.append(dict(experiment_id="G01", condition=f"{axis}_{mm}mm", severity=mm, direction=axis, kind="translation"))
    for n in design["anchor_counts"]:
        for subset in design["subsets"]:
            matrix.append(dict(experiment_id="G02", condition=f"n{n}_{subset}", severity=n, subset=subset, kind="anchor_count"))
    for n in design["outlier_counts"]:
        for subset in design["subsets"]:
            matrix.append(dict(experiment_id="G03", condition=f"outliers{n}_{subset}", severity=n, subset=subset, kind="outlier"))
    for degrees in design["rotation_degrees"]:
        for axis in design["rotation_axes"]:
            matrix.append(dict(experiment_id="G04", condition=f"{axis}_{degrees}deg", severity=degrees, axis=axis, kind="rotation"))
    for mm in design["depth_noise_rms_mm"]:
        matrix.append(dict(experiment_id="G05", condition=f"noise_{mm}mm", severity=mm, kind="noise"))
    for fraction in design["missing_fractions"]:
        for mode in design["missing_modes"]:
            matrix.append(dict(experiment_id="G05", condition=f"missing_{mode}_{int(fraction*100)}pct", severity=fraction, kind="missing", mode=mode))
    for group, kind in (("G06", "anchor_bias"), ("G07", "target_bias")):
        for mm in design["local_depth_bias_mm"]:
            matrix.append(dict(experiment_id=group, condition=f"{kind}_{mm}mm", severity=mm, kind=kind))
    return matrix


def unique_visible_indices(uv, confidence, visible, depth):
    _, valid, pixels = sample_depth(depth, uv)
    seen, result = set(), []
    for i in np.flatnonzero(valid & visible & (confidence >= .5)):
        key = tuple(pixels[i])
        if key not in seen:
            seen.add(key)
            result.append(int(i))
    return np.asarray(result, dtype=int)


def spread_subsets(uv, pool, count=8):
    """Three deterministic farthest-point subsets; selection uses pixels only."""
    pool = np.asarray(pool, dtype=int)
    if len(pool) < count:
        return {name: pool.tolist() for name in ("A", "B", "C")}
    pixels = np.asarray(uv)[pool]
    starts = [0, int(np.argmax(pixels[:, 0])), int(np.argmax(pixels[:, 1]))]
    orders = {}
    for name, start in zip(("A", "B", "C"), starts):
        selected = [start]
        distance = np.full(len(pool), np.inf)
        while len(selected) < count:
            distance = np.minimum(distance, np.sum((pixels-pixels[selected[-1]])**2, axis=1))
            distance[selected] = -1
            selected.append(int(np.argmax(distance)))
        orders[name] = pool[selected].tolist()
    return orders


def pixel_region(shape, uv, radius):
    mask = np.zeros(shape, dtype=bool)
    for x, y in np.rint(uv).astype(int):
        if 0 <= x < shape[1] and 0 <= y < shape[0]:
            mask[max(0, y-radius):min(shape[0], y+radius+1),
                 max(0, x-radius):min(shape[1], x+radius+1)] = True
    return mask


def rotation(axis, degrees):
    theta = np.deg2rad(degrees)
    a = np.eye(3)["xyz".index(axis)]
    skew = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(theta)*skew + (1-np.cos(theta))*(skew@skew)


def perturb(depth, camera, points, anchors, rotation_ee, item, frame, design):
    """Return copied measured data; GT is used only to construct interventions."""
    measured = np.asarray(depth, dtype=np.float64).copy()
    t = camera.camera_to_action.copy()
    local = np.asarray(points).copy()
    valid = np.isfinite(depth) & (depth > 0)
    kind, severity = item["kind"], item["severity"]
    bias_gt = None
    if kind == "translation":
        bias_gt = unit_direction(item["direction"], design) * severity/1000
        t[:3, 3] += bias_gt
    elif kind in {"anchor_count", "outlier"}:
        bias_gt = unit_direction("mixed", design) * design["fixed_translation_mm"]/1000
        t[:3, 3] += bias_gt
        if kind == "outlier":
            for i in range(int(severity)):
                direction = np.asarray(design["outlier_directions"][i], dtype=float)
                direction /= np.linalg.norm(direction)
                # Wrong correspondence adds +delta to observed-minus-known residual.
                local[i] -= (direction * design["outlier_magnitude_mm"]/1000) @ rotation_ee
    elif kind == "rotation":
        t[:3, :3] = t[:3, :3] @ rotation(item["axis"], severity)
    elif kind == "noise":
        rng = np.random.default_rng(np.random.SeedSequence([design["seed"], frame["episode_id"], frame["frame_id"], 51]))
        noise = rng.standard_normal(int(valid.sum()))
        noise -= noise.mean()
        noise *= (severity/1000) / np.sqrt(np.mean(noise**2))
        measured[valid] += noise
    elif kind == "missing":
        count = int(round(severity * valid.sum()))
        if item["mode"] == "random":
            rng = np.random.default_rng(np.random.SeedSequence([design["seed"], frame["episode_id"], frame["frame_id"], 52]))
            indices = rng.permutation(np.flatnonzero(valid))[:count]
            measured.flat[indices] = 0
        else:
            h, w = depth.shape
            width = max(1, int(round(np.sqrt(severity)*w)))
            height = min(h, int(np.ceil(count/width)))
            centre = anchors.uv[anchors.visible].mean(axis=0) if anchors.visible.any() else np.array([w/2, h/2])
            x0 = int(np.clip(np.rint(centre[0])-width//2, 0, w-width))
            y0 = int(np.clip(np.rint(centre[1])-height//2, 0, h-height))
            measured[y0:y0+height, x0:x0+width] = 0
    elif kind == "anchor_bias":
        measured[frame["anchor_roi"]] += severity/1000
    elif kind == "target_bias":
        measured[frame["target_roi"]] += severity/1000
    else:
        raise ValueError("Unknown frozen intervention")
    return measured, Camera(camera.intrinsic, t, depth.shape), local, bias_gt


def error_statistics(reference, prediction, mask):
    if not np.any(mask):
        return {"rms_mm": None, "p95_mm": None, "pixels": 0}
    error = np.linalg.norm(prediction[mask]-reference[mask], axis=1)*1000
    return {"rms_mm": float(np.sqrt(np.mean(error**2))), "p95_mm": float(np.percentile(error, 95)), "pixels": int(mask.sum())}


def _distribution(values):
    values = np.asarray(values)
    return (float(np.median(values)), float(np.percentile(values, 95))) if len(values) else (None, None)


def run_condition(sample, frame, config, design, item):
    """Run existing correction and reliability; only measured depth/T/pairing vary."""
    clean = canonical_geometry(sample, config["geometry"])
    true_camera = LiberoAdapter(config["geometry"]).camera(sample)
    ree = rotation_from_xyzw(sample["eef_quaternion_xyzw"])
    all_uv = frame["anchor_uv"]
    indices = np.arange(len(all_uv))
    if item["kind"] in {"anchor_count", "outlier"}:
        requested = int(item["severity"]) if item["kind"] == "anchor_count" else 8
        indices = np.asarray(frame["subsets"][item["subset"]][:requested], dtype=int)
    anchors = AnchorObservation(all_uv[indices], frame["anchor_confidence"][indices], frame["anchor_visible"][indices])
    points = frame["local_points_ee_m"][indices]
    measured, camera, local, bias_gt = perturb(clean["depth"], true_camera, points, anchors, ree, item, frame, design)
    corrupted = dict(sample)
    corrupted["agentview_depth_m"] = np.ascontiguousarray(measured[:, ::-1])
    corrupted["camera_to_world"] = camera.camera_to_action
    nominal = canonical_geometry(corrupted, config["geometry"])
    correction_config = CorrectionConfig(**{k: v for k, v in config["method"]["correction"].items() if k not in {"enabled", "mode"}})
    result = correct_from_anchors(anchors, local, measured, camera, sample["ee_world_m"], ree, correction_config)
    corrected = apply_correction(nominal["actioncoord"], nominal["valid"], result)
    rel_config = ReliabilityConfig(**{k: v for k, v in config["method"]["reliability"].items() if k != "enabled"})
    rel = geometry_reliability(result, anchors.confidence, nominal["valid"], rel_config)
    accepted = bool(result.accepted and rel.correction_valid)
    scene = clean["valid"] & ~frame["robot_mask"]
    common = scene & nominal["valid"]
    before = error_statistics(clean["actioncoord"], nominal["actioncoord"], common)
    after = error_statistics(clean["actioncoord"], corrected, common)
    median, p95 = _distribution(np.linalg.norm(result.residuals_m[result.valid_anchors], axis=1)*1000)
    _, anchor_depth_valid, _ = sample_depth(measured, anchors.uv)
    known_world = points @ ree.T + sample["ee_world_m"]
    pool = np.asarray(frame["pool"], dtype=int)
    centroid = (frame["local_points_ee_m"][pool] @ ree.T + sample["ee_world_m"]).mean(axis=0)
    distances = np.linalg.norm(clean["actioncoord"] + sample["ee_world_m"] - centroid, axis=-1)
    q1, q2 = frame["distance_tertiles_m"]
    metrics = {}
    for name, mask in (("near", distances <= q1), ("middle", (distances > q1) & (distances <= q2)), ("far", distances > q2)):
        metrics[name+"_rms_after_mm"] = error_statistics(clean["actioncoord"], corrected, common & mask)["rms_mm"]
    for prefix, prediction in (("before", nominal["actioncoord"]), ("after", corrected)):
        target = error_statistics(clean["actioncoord"], prediction, frame["target_roi"] & nominal["valid"])
        metrics["target_rms_"+prefix+"_mm"] = target["rms_mm"]
    used_uv = anchors.uv[result.valid_anchors]
    spread_px = float(np.sqrt(np.mean(np.sum((used_uv-used_uv.mean(axis=0))**2, axis=1)))) if len(used_uv) else None
    record = dict(experiment_id=item["experiment_id"], episode_id=frame["episode_id"], frame_id=frame["frame_id"],
                  condition=item["condition"], severity=item["severity"], kind=item["kind"],
                  visible_configuration_id=item.get("subset"), num_anchor_total=len(indices),
                  num_anchor_valid=int((anchors.visible & anchor_depth_valid).sum()),
                  num_anchor_used=int(result.valid_anchors.sum()),
                  gt_bias_xyz_m=None if bias_gt is None else bias_gt.tolist(),
                  estimated_bias_xyz_m=result.bias_xyz_m.tolist(),
                  estimated_bias_norm_mm=float(np.linalg.norm(result.bias_xyz_m)*1000),
                  bias_error_mm=None if bias_gt is None else float(np.linalg.norm(result.bias_xyz_m-bias_gt)*1000),
                  scene_rms_before_mm=before["rms_mm"], scene_rms_after_mm=after["rms_mm"],
                  scene_p95_before_mm=before["p95_mm"], scene_p95_after_mm=after["p95_mm"],
                  anchor_residual_median_mm=median, anchor_residual_p95_mm=p95,
                  reliability_score=rel.fit_score, accepted=accepted, correction_accepted=result.accepted,
                  failure_reason=None if accepted else (result.reason if not result.accepted else "reliability_rejected"),
                  anchor_spatial_spread_px=spread_px,
                  scene_valid_coverage=float(common.sum()/scene.sum()),
                  measured_depth_missing_fraction=float(np.mean(clean["valid"] & ~nominal["valid"])),
                  mean_patch_weight=float(rel.patch_weights.mean()), target_roi_pixels=int(frame["target_roi"].sum()),
                  scene_pixels=before["pixels"],
                  intervention_roi_pixels=int(frame["anchor_roi" if item["kind"] == "anchor_bias" else "target_roi"].sum()) if item["kind"] in {"anchor_bias", "target_bias"} else None,
                  eligible=bool(frame["target_roi"].any()) if item["kind"] == "target_bias" else True,
                  **metrics)
    record["scene_harmed"] = bool(before["rms_mm"] is not None and after["rms_mm"] > before["rms_mm"]+design["scene_harm_margin_mm"])
    record["silent_wrong_correction"] = bool(accepted and (record["scene_harmed"] or
                                                        (record["bias_error_mm"] is not None and record["bias_error_mm"] > design["g2_bias_error_limit_mm"])))
    if item["kind"] == "outlier":
        if result.valid_anchors.any():
            mean_bias = np.mean(result.raw_residuals_m[result.valid_anchors], axis=0)
            mean_map = nominal["actioncoord"].copy()
            mean_map[nominal["valid"]] -= mean_bias
            record["mean_bias_xyz_m"] = mean_bias.tolist()
            record["mean_bias_error_mm"] = float(np.linalg.norm(mean_bias-bias_gt)*1000)
            record["mean_scene_rms_after_mm"] = error_statistics(clean["actioncoord"], mean_map, common)["rms_mm"]
        else:
            record.update(mean_bias_xyz_m=None, mean_bias_error_mm=None, mean_scene_rms_after_mm=None)
    return record, {"rgb": clean["rgb"], "scene": scene, "target": frame["target_roi"],
                    "anchor_roi": frame["anchor_roi"], "before_error_mm": np.linalg.norm(nominal["actioncoord"]-clean["actioncoord"], axis=-1)*1000,
                    "after_error_mm": np.linalg.norm(corrected-clean["actioncoord"], axis=-1)*1000}
