"""Bounded geometry-only smoke: fixed bias, true robot surface anchors, no actions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from rulervla.adapters.libero import LiberoAdapter, rigid_surface_points_from_sim
from rulervla.adapters.openpi import OpenPIAdapter
from rulervla.config import load_config, profile
from rulervla.data.schema import rotation_from_xyzw
from rulervla.evaluation.corruption import inject_shared_translation
from rulervla.evaluation.geometry_metrics import geometry_error
from rulervla.geometry.action_coord import canonical_geometry, raster
from rulervla.geometry.visual_anchor import SimulatorGTAnchorProvider

BIAS = np.array([.02, -.01, .005])


def check_frame(sample, config, points, *, recovery_tolerance_m=.002):
    maps = canonical_geometry(sample, config["geometry"])
    camera = LiberoAdapter(config["geometry"]).camera(sample)
    rotation = rotation_from_xyzw(sample["eef_quaternion_xyzw"])
    provider = SimulatorGTAnchorProvider(points, sample["ee_world_m"], rotation, camera, maps["depth"],
                                        occlusion_tolerance_m=config["method"]["visual_anchor"]["occlusion_tolerance_m"])
    corrupted = dict(sample)
    corrupted["camera_to_world"] = inject_shared_translation(camera, BIAS).camera_to_action
    nominal = canonical_geometry(corrupted, config["geometry"])
    adapter = OpenPIAdapter(profile(config, "V05-C"))
    adapter.prepare(corrupted, anchor_provider=provider, local_points_ee_m=points)
    result = adapter.last_diagnostics["correction"]
    before = geometry_error(maps["actioncoord"], nominal["actioncoord"], maps["valid"])
    after = geometry_error(maps["actioncoord"], adapter.last_diagnostics["corrected_actioncoord_m"], maps["valid"])
    error = float(np.linalg.norm(result.bias_xyz_m-BIAS))
    if result.accepted and (error > recovery_tolerance_m or after["mean_m"] >= before["mean_m"]):
        raise AssertionError(f"Accepted correction failed to recover: {error}, {before}, {after}")
    if not result.accepted:
        np.testing.assert_array_equal(result.bias_xyz_m, 0)
    valid_residual = result.residuals_m[result.valid_anchors]
    return {"bias_gt_m": BIAS.tolist(), "bias_estimate_m": result.bias_xyz_m.tolist(),
            "bias_error_m": error, "before": before, "after": after,
            "residual_median_m": float(np.median(np.linalg.norm(valid_residual, axis=1))) if len(valid_residual) else None,
            "accepted": result.accepted, "reason": result.reason, "anchors_used": int(result.valid_anchors.sum())}


def synthetic_smoke(config):
    ee = np.array([0., 0, 1.])
    sample = dict(agentview_rgb=np.zeros((256, 256, 3), dtype=np.uint8),
                  wrist_rgb=np.zeros((256, 256, 3), dtype=np.uint8),
                  agentview_depth_m=np.full((256, 256), 2., dtype=np.float32),
                  intrinsic=np.array([[200., 0, 128.], [0, 200., 128.], [0, 0, 1.]]),
                  camera_to_world=np.eye(4), ee_world_m=ee, state8=np.r_[ee, np.zeros(5)],
                  eef_quaternion_xyzw=np.array([0., 0, 0, 1.]), task="synthetic geometry test")
    camera = LiberoAdapter(config["geometry"]).camera(sample)
    uv = np.array([[80, 90], [110, 90], [140, 90], [170, 90],
                   [80, 140], [110, 140], [140, 140], [170, 140]], dtype=float)
    points = np.c_[uv, np.ones(8)] @ np.linalg.inv(camera.intrinsic).T * 2 - ee
    return {"mode": "synthetic_independent_rigid_points", "frames": [check_frame(sample, config, points)]}


def real_smoke(args, config):
    """Read frozen NPZs, restore 50 recorded states, and use GT palm mesh points.

    No env.step, policy inference, task success query, model edit, or trajectory
    generation occurs. Surface point targets come from compiled robot geometry.
    """
    root = args.v03_root.resolve()
    sys.path.insert(1, str(root / "repo/actioncoord_experiments/src"))
    sys.path.insert(1, str(root / "repo/third_party/libero"))
    import h5py
    from actioncoord.data.regenerated_rgbd_dataset import RegeneratedRGBDDataset
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.camera_utils import get_real_depth_map

    dataset = RegeneratedRGBDDataset(root / "configs/v03_training_manifest.json")
    task = benchmark.get_benchmark_dict()["libero_spatial"]().get_task(2)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256,
                            camera_depths=True, render_gpu_device_id=args.device_id)
    report = {"mode": "restored_simulator_gt_rigid_hand_surface", "v03_root": str(root),
              "task": task.language, "selection": "first 10 manifest episodes, 5 evenly spaced frames each",
              "no_control_steps": True, "anchor_source": "compiled rigid-body mesh triangle centroids, not depth backprojection",
              "anchor_body": config["method"]["visual_anchor"]["rigid_body_name"],
              "frames": []}
    offset = 0
    try:
        for row in dataset.episodes[:args.episodes]:
            with h5py.File(row["source"], "r") as h5:
                group = h5["data"][row["source_demo"]]
                first = dataset[offset]
                env.seed(0)
                env.reset()
                env.set_init_state(np.asarray(group["states"][first["source_frame_index"]], dtype=np.float64))
                # Fixed surface coordinates per episode; exclude nonrigid fingers.
                obs = env.env._get_observations(force_update=True)
                rotation = rotation_from_xyzw(obs["robot0_eef_quat"])
                anchor_cfg = config["method"]["visual_anchor"]
                points = rigid_surface_points_from_sim(env.sim, obs["robot0_eef_pos"], rotation,
                    body_name=anchor_cfg["rigid_body_name"], max_points=anchor_cfg["max_surface_points"])
                for frame in np.linspace(0, row["frames"]-1, args.frames_per_episode, dtype=int):
                    sample = dataset[offset+int(frame)]
                    env.sim.set_state_from_flattened(group["states"][sample["source_frame_index"]])
                    env.sim.forward()
                    obs = env.env._get_observations(force_update=True)
                    # Audit same-state RGB/depth and exact rigid-surface/EE pose first.
                    np.testing.assert_array_equal(raster(obs["agentview_image"], "rot180"), sample["agentview_rgb"])
                    # Match frozen V0.3: promote the normalized buffer before
                    # the near/far-plane division, then store metric float32.
                    raw_depth = get_real_depth_map(env.sim, np.asarray(obs["agentview_depth"], dtype=np.float64).squeeze(-1)).astype(np.float32)
                    np.testing.assert_allclose(raster(raw_depth, "rot180"), sample["agentview_depth_m"], atol=2e-5, rtol=0)
                    np.testing.assert_allclose(obs["robot0_eef_pos"], sample["ee_world_m"], atol=2e-6, rtol=0)
                    np.testing.assert_allclose(obs["robot0_eef_quat"], sample["eef_quaternion_xyzw"], atol=2e-6, rtol=0)
                    record = check_frame(sample, config, points)
                    record["depth_replay_max_error_m"] = float(np.max(np.abs(raster(raw_depth, "rot180") - sample["agentview_depth_m"])))
                    record.update(episode=row["episode_index"], frame=int(frame), source_frame=sample["source_frame_index"])
                    report["frames"].append(record)
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.with_suffix(".partial.json").write_text(json.dumps(report, indent=2)+"\n")
                offset += row["frames"]
                print(json.dumps({"episode": row["episode_index"], "frames_checked": len(report["frames"])}), flush=True)
    finally:
        env.close()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v03-root", type=Path)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--frames-per-episode", type=int, default=5)
    parser.add_argument("--device-id", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("outputs/geometry_smoke.json"))
    args = parser.parse_args()
    if not 1 <= args.episodes <= 10 or not 5 <= args.frames_per_episode <= 10:
        raise ValueError("Geometry-only smoke is bounded to <=10 episodes and 5-10 frames each")
    config = load_config(Path(__file__).parents[1] / "configs/v05_default.yaml")
    report = real_smoke(args, config) if args.v03_root else synthetic_smoke(config)
    accepted = [row for row in report["frames"] if row["accepted"]]
    if not accepted:
        raise AssertionError("No independently visible robot anchors; correction mechanism remains unverified")
    report["summary"] = {"passed": True, "frames": len(report["frames"]), "accepted": len(accepted),
                          "rejected": len(report["frames"])-len(accepted),
                          "max_bias_recovery_error_m": max(row["bias_error_m"] for row in accepted),
                          "mean_before_m": float(np.mean([row["before"]["mean_m"] for row in accepted])),
                          "mean_after_m": float(np.mean([row["after"]["mean_m"] for row in accepted]))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
