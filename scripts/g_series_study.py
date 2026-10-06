"""Prepare and run a frozen G1-G7 geometry study; never steps a policy."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import yaml

from rulervla.adapters.libero import LiberoAdapter, rigid_surface_points_from_sim
from rulervla.config import load_config
from rulervla.data.schema import rotation_from_xyzw
from rulervla.evaluation.g_series import (GROUPS, array_hash, conditions, file_hash,
    pixel_region, run_condition, spread_subsets, unique_visible_indices)
from rulervla.geometry.action_coord import canonical_geometry, raster
from rulervla.geometry.visual_anchor import SimulatorGTAnchorProvider


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def implementation_hashes():
    return {str(p.relative_to(ROOT)): file_hash(p) for p in sorted((ROOT/"src/rulervla").rglob("*.py"))
            if p.name != "g_series.py"}


def descendant_body_ids(model, root_id):
    ids = {root_id}
    for i in range(model.nbody):
        parent = i
        while parent:
            if parent == root_id:
                ids.add(i)
                break
            parent = int(model.body_parentid[parent])
    return ids


def prepare(args, design, config):
    """No estimator evaluation here: freeze all inputs before any G result."""
    manifest_path = ROOT/"experiments/base_frame_manifest.json"
    lock_path = ROOT/"experiments/g_series_protocol_lock.json"
    if manifest_path.exists() or lock_path.exists():
        raise FileExistsError("Frozen base set/protocol already exists; do not replace it")
    v03 = Path(design["v03_root"])
    sys.path.insert(1, str(v03/"repo/actioncoord_experiments/src"))
    sys.path.insert(1, str(v03/"repo/third_party/libero"))
    import h5py
    import mujoco
    from actioncoord.data.regenerated_rgbd_dataset import RegeneratedRGBDDataset
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.camera_utils import get_real_depth_map, get_camera_intrinsic_matrix, get_camera_extrinsic_matrix
    dataset = RegeneratedRGBDDataset(v03/"configs/v03_training_manifest.json")
    task = benchmark.get_benchmark_dict()["libero_spatial"]().get_task(2)
    env = OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path("bddl_files"))/task.problem_folder/task.bddl_file),
                            camera_heights=256, camera_widths=256, camera_depths=True,
                            render_gpu_device_id=args.device_id)
    frames, offset = [], 0
    cache = ROOT/"outputs/g_series/base_frames"
    cache.mkdir(parents=True, exist_ok=True)
    body_names = [env.sim.model.body_id2name(i) or "" for i in range(env.sim.model.nbody)]
    robot_bodies = [i for i, name in enumerate(body_names) if name.startswith(("robot0_", "gripper0_"))]
    robot_geoms = [i for i in range(env.sim.model.ngeom) if int(env.sim.model.geom_bodyid[i]) in robot_bodies]
    geom_type = int(mujoco.mjtObj.mjOBJ_GEOM)
    try:
        for row in dataset.episodes[:design["episodes"]]:
            target_id = env.sim.model.body_name2id(row["target_body_name"])
            target_bodies = descendant_body_ids(env.sim.model, target_id)
            target_geoms = [i for i in range(env.sim.model.ngeom) if int(env.sim.model.geom_bodyid[i]) in target_bodies]
            with h5py.File(row["source"], "r") as h5:
                group = h5["data"][row["source_demo"]]
                first = dataset[offset]
                env.seed(0)
                env.reset()
                env.set_init_state(np.asarray(group["states"][first["source_frame_index"]], dtype=np.float64))
                obs = env.env._get_observations(force_update=True)
                anchor_cfg = config["method"]["visual_anchor"]
                points = rigid_surface_points_from_sim(env.sim, obs["robot0_eef_pos"], rotation_from_xyzw(obs["robot0_eef_quat"]),
                    body_name=anchor_cfg["rigid_body_name"], max_points=anchor_cfg["max_surface_points"])
                for frame_id in np.linspace(0, row["frames"]-1, design["frames_per_episode"], dtype=int):
                    sample = dataset[offset+int(frame_id)]
                    source_state = np.asarray(group["states"][sample["source_frame_index"]], dtype=np.float64)
                    env.sim.set_state_from_flattened(source_state)
                    env.sim.forward()
                    obs = env.env._get_observations(force_update=True)
                    depth_raw = get_real_depth_map(env.sim, np.asarray(obs["agentview_depth"], dtype=np.float64).squeeze(-1)).astype(np.float32)
                    np.testing.assert_array_equal(raster(obs["agentview_image"], "rot180"), sample["agentview_rgb"])
                    np.testing.assert_array_equal(raster(obs["robot0_eye_in_hand_image"], "rot180"), sample["wrist_rgb"])
                    np.testing.assert_array_equal(raster(depth_raw, "rot180"), sample["agentview_depth_m"])
                    np.testing.assert_allclose(obs["robot0_eef_pos"], sample["ee_world_m"], atol=2e-6, rtol=0)
                    np.testing.assert_allclose(obs["robot0_eef_quat"], sample["eef_quaternion_xyzw"], atol=2e-6, rtol=0)
                    np.testing.assert_allclose(get_camera_intrinsic_matrix(env.sim, "agentview", 256, 256), sample["intrinsic"], atol=1e-9, rtol=0)
                    np.testing.assert_allclose(get_camera_extrinsic_matrix(env.sim, "agentview"), sample["camera_to_world"], atol=1e-9, rtol=0)
                    maps = canonical_geometry(sample, config["geometry"])
                    camera = LiberoAdapter(config["geometry"]).camera(sample)
                    ree = rotation_from_xyzw(sample["eef_quaternion_xyzw"])
                    provider = SimulatorGTAnchorProvider(points, sample["ee_world_m"], ree, camera, maps["depth"],
                        occlusion_tolerance_m=anchor_cfg["occlusion_tolerance_m"])
                    anchors = provider.detect(maps["rgb"])
                    pool = unique_visible_indices(anchors.uv, anchors.confidence, anchors.visible, maps["depth"])
                    if len(pool) < 8:
                        raise RuntimeError("Fixed frame has fewer than 8 eligible anchors; report preflight failure without replacing it")
                    seg_raw = env.sim.render(width=256, height=256, camera_name="agentview", segmentation=True)
                    seg = raster(seg_raw, "vflip")
                    is_geom = seg[..., 0] == geom_type
                    robot_mask = is_geom & np.isin(seg[..., 1], robot_geoms)
                    target_mask = is_geom & np.isin(seg[..., 1], target_geoms)
                    # Isolate interventions: target depth never touches an anchor lookup.
                    guard = pixel_region(maps["depth"].shape, anchors.uv[pool], design["anchor_region_radius_px"])
                    anchor_roi = guard & robot_mask & ~target_mask
                    target_roi = target_mask & ~guard & ~robot_mask & maps["valid"]
                    scene_mask = maps["valid"] & ~robot_mask
                    centroid = (points[pool] @ ree.T + sample["ee_world_m"]).mean(axis=0)
                    distance = np.linalg.norm(maps["actioncoord"]+sample["ee_world_m"]-centroid, axis=-1)
                    thresholds = np.quantile(distance[scene_mask], [1/3, 2/3])
                    frame_key = f"ep{row['episode_index']}_f{int(frame_id):04d}"
                    cache_path = cache/(frame_key+".npz")
                    payload = {k: sample[k] for k in ("agentview_rgb", "wrist_rgb", "agentview_depth_m", "intrinsic", "camera_to_world", "ee_world_m", "state8", "eef_quaternion_xyzw")}
                    payload.update(local_points_ee_m=points, anchor_uv=anchors.uv, anchor_confidence=anchors.confidence,
                                   anchor_visible=anchors.visible, robot_mask=robot_mask, target_roi=target_roi, anchor_roi=anchor_roi)
                    np.savez_compressed(cache_path, **payload)
                    frames.append(dict(frame_key=frame_key, episode_id=row["episode_index"], frame_id=int(frame_id), source_frame_id=sample["source_frame_index"],
                        source_path=row["source"], source_demo=row["source_demo"], rgbd_npz_path=row["npz"], rgbd_npz_sha256=row["npz_sha256"],
                        state_npz_path=row["state_npz"], state_npz_sha256=row["state_npz_sha256"], simulator_state_sha256=array_hash(source_state),
                        rgb_sha256=array_hash(sample["agentview_rgb"]), wrist_rgb_sha256=array_hash(sample["wrist_rgb"]), depth_sha256=array_hash(sample["agentview_depth_m"]),
                        ee_world_m=sample["ee_world_m"].tolist(), ee_quaternion_xyzw=sample["eef_quaternion_xyzw"].tolist(),
                        camera_K_before_pixel_offset=sample["intrinsic"].tolist(), camera_K_canonical=camera.intrinsic.tolist(), camera_to_world=sample["camera_to_world"].tolist(),
                        anchor_gt=dict(body=anchor_cfg["rigid_body_name"], local_points_ee_m=points.tolist(),
                                       uv=[uv.tolist() if np.isfinite(uv).all() else None for uv in anchors.uv],
                                       confidence=anchors.confidence.tolist(), visible=anchors.visible.tolist()),
                        pool=pool.tolist(), subsets=spread_subsets(anchors.uv, pool), distance_tertiles_m=thresholds.tolist(),
                        num_unique_visible_anchors=len(pool), target_visible_pixels=int(target_mask.sum()), target_roi_pixels=int(target_roi.sum()),
                        target_geom_ids=target_geoms, target_body_name=row["target_body_name"], robot_geom_ids=robot_geoms,
                        target_roi_sha256=array_hash(target_roi), robot_mask_sha256=array_hash(robot_mask), anchor_roi_sha256=array_hash(anchor_roi),
                        anchor_roi_pixels=int(anchor_roi.sum()), target_distance_to_ee_m=float(np.linalg.norm(env.sim.data.body_xpos[target_id]-sample["ee_world_m"])),
                        cache_path=str(cache_path.relative_to(ROOT)), cache_sha256=file_hash(cache_path)))
                offset += row["frames"]
                print(json.dumps({"prepared_episode": row["episode_index"], "frames": len(frames)}), flush=True)
    finally:
        env.close()
    previous = json.loads((ROOT/"docs/evidence/gt_anchor_smoke.json").read_text())
    expected = [(x["episode"], x["frame"]) for x in previous["frames"]]
    if [(x["episode_id"], x["frame_id"]) for x in frames] != expected:
        raise AssertionError("Base frames differ from prior validated smoke")
    manifest = dict(schema="rulervla_base_frames_v1", task=task.language, frames=frames, frames_total=len(frames),
        source_manifest_path=str(v03/"configs/v03_training_manifest.json"), source_manifest_sha256=file_hash(v03/"configs/v03_training_manifest.json"),
        selection=design["selection"], no_control_steps=True, no_policy_inference=True,
        gt_visibility="Frozen clean-render visibility; no corruption-dependent GT filtering",
        segmentation_convention="MuJoCo geom object type/id, raw-to-canonical vflip; exact same recorded state",
        coverage=dict(min_unique_anchors=min(x["num_unique_visible_anchors"] for x in frames),
                      frames_target_visible=sum(x["target_roi_pixels"] > 0 for x in frames),
                      ee_xyz_min_m=np.min([x["ee_world_m"] for x in frames], axis=0).tolist(),
                      ee_xyz_max_m=np.max([x["ee_world_m"] for x in frames], axis=0).tolist()))
    write_json(manifest_path, manifest)
    matrix = conditions(design)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    lock = dict(schema="rulervla_g_series_protocol_lock_v1", frozen_before_outcomes=True,
                runner_commit=head, design=design, conditions=matrix, conditions_total=len(matrix), expected_rows=len(matrix)*len(frames),
                base_manifest_sha256=file_hash(manifest_path), design_sha256=file_hash(ROOT/"configs/g_series.yaml"),
                method_config_sha256=file_hash(ROOT/"configs/v05_default.yaml"), method_config=config,
                implementation_sha256=implementation_hashes(), runner_sha256=file_hash(Path(__file__)),
                evaluator_sha256=file_hash(ROOT/"src/rulervla/evaluation/g_series.py"))
    write_json(lock_path, lock)
    for group, (directory, question) in GROUPS.items():
        p = ROOT/"experiments"/directory
        p.mkdir(parents=True, exist_ok=True)
        group_conditions = [x for x in matrix if x["experiment_id"] == group]
        (p/"config.yaml").write_text(yaml.safe_dump(dict(experiment_id=group, conditions=group_conditions, design=design), sort_keys=False))
        write_json(p/"manifest.json", dict(base_manifest="../base_frame_manifest.json", base_manifest_sha256=file_hash(manifest_path),
                                          protocol="../g_series_protocol_lock.json", protocol_sha256=file_hash(lock_path),
                                          conditions=group_conditions, expected_rows=len(group_conditions)*len(frames)))
        (p/"README.md").write_text(f"# {group}: {question}\n\nSame frozen 50 frames. Translation estimator and reliability thresholds remain at commit {design['base_implementation_commit']}.\n\nSee `config.yaml`, `manifest.json`, `raw/results.csv`, `summary.json`, `figures/`, and `conclusion.md`.\n")
    print(json.dumps({"prepared": True, "frames": len(frames), "conditions": len(matrix), "expected_rows": lock["expected_rows"],
                      "manifest_sha256": file_hash(manifest_path), "protocol_sha256": file_hash(lock_path), "coverage": manifest["coverage"]}), flush=True)


def verify_lock():
    lock = json.loads((ROOT/"experiments/g_series_protocol_lock.json").read_text())
    if file_hash(ROOT/"experiments/base_frame_manifest.json") != lock["base_manifest_sha256"]:
        raise AssertionError("Frozen frame manifest changed")
    if file_hash(ROOT/"configs/g_series.yaml") != lock["design_sha256"] or file_hash(ROOT/"configs/v05_default.yaml") != lock["method_config_sha256"]:
        raise AssertionError("Frozen design/method thresholds changed")
    if implementation_hashes() != lock["implementation_sha256"]:
        raise AssertionError("Production estimator or geometry changed")
    if file_hash(Path(__file__)) != lock["runner_sha256"] or file_hash(ROOT/"src/rulervla/evaluation/g_series.py") != lock["evaluator_sha256"]:
        raise AssertionError("Frozen runner changed; do not silently patch outcomes")
    return lock


def save_rows(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "null" if v is None else json.dumps(v) if isinstance(v, (list, dict, bool)) else v for k, v in row.items()})


def run(design, config):
    lock = verify_lock()
    manifest = json.loads((ROOT/"experiments/base_frame_manifest.json").read_text())
    output = {g: [] for g in GROUPS}
    root_out = ROOT/"outputs/g_series"
    status_path = root_out/"status.json"
    for i, row in enumerate(manifest["frames"]):
        cache = ROOT/row["cache_path"]
        if file_hash(cache) != row["cache_sha256"]:
            raise AssertionError("Frozen cached frame changed: "+row["frame_key"])
        with np.load(cache, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        sample = {k: arrays[k] for k in ("agentview_rgb", "wrist_rgb", "agentview_depth_m", "intrinsic", "camera_to_world", "ee_world_m", "state8", "eef_quaternion_xyzw")}
        sample["task"] = manifest["task"]
        frame = dict(row, **{k: arrays[k] for k in ("local_points_ee_m", "anchor_uv", "anchor_confidence", "anchor_visible", "robot_mask", "target_roi", "anchor_roi")})
        for item in lock["conditions"]:
            record, visual = run_condition(sample, frame, config, design, item)
            record.update(base_manifest_sha256=lock["base_manifest_sha256"], protocol_sha256=file_hash(ROOT/"experiments/g_series_protocol_lock.json"),
                          code_commit=lock["runner_commit"])
            output[item["experiment_id"]].append(record)
            # Preselected first frame, not a post-hoc best/worst case.
            if i == 0 and item["experiment_id"] in {"G06", "G07"} and item["severity"] == 10:
                np.savez_compressed(root_out/(item["experiment_id"]+"_representative.npz"), **visual)
        for group, rows in output.items():
            save_rows(ROOT/"experiments"/GROUPS[group][0]/"raw/results.csv", rows)
        write_json(status_path, dict(status="running", frames_completed=i+1, expected_frames=50, rows=sum(len(r) for r in output.values())))
        if (i+1) % 5 == 0:
            print(json.dumps({"frames_completed": i+1, "rows": sum(len(r) for r in output.values())}), flush=True)
    if sum(len(r) for r in output.values()) != lock["expected_rows"]:
        raise AssertionError("Incomplete condition matrix")
    verify_lock()
    write_json(status_path, dict(status="geometry_complete", frames_completed=50, rows=lock["expected_rows"],
                                 base_manifest_sha256=lock["base_manifest_sha256"], protocol_sha256=file_hash(ROOT/"experiments/g_series_protocol_lock.json")))
    print(json.dumps({"geometry_complete": True, "rows": lock["expected_rows"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--device-id", type=int, default=2)
    args = parser.parse_args()
    design = yaml.safe_load((ROOT/"configs/g_series.yaml").read_text())
    config = load_config(ROOT/"configs/v05_default.yaml")
    (prepare(args, design, config) if args.phase == "prepare" else run(design, config))


if __name__ == "__main__":
    main()
