"""Compare the extracted middleware against the authoritative frozen V0.3 parser."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from rulervla.adapters.openpi import OpenPIAdapter, OpenPIThirdSlotInputs
from rulervla.config import load_config, profile
from rulervla.geometry.action_coord import canonical_geometry


def compare(sample, config):
    from actioncoord.geometry.v03 import canonical_geometry as reference_geometry
    from actioncoord.integration.v03_inputs import prepare_sample
    old = reference_geometry(sample, config["geometry"])
    new = canonical_geometry(sample, config["geometry"])
    for key in ("rgb", "wrist", "depth", "camera_xyz", "actioncoord", "valid"):
        np.testing.assert_array_equal(new[key], old[key])
    errors = {}
    for arm, representation in (("R0", "neutral"), ("R1", "depth"), ("R2", "camera_xyz"), ("R3", "actioncoord")):
        cfg = profile(config, "V03")
        cfg["method"]["representation"] = representation
        expected = prepare_sample(sample, arm, cfg["geometry"], cfg["encoding_config"])
        actual = OpenPIAdapter(cfg).prepare(sample)
        for key in expected:
            if isinstance(expected[key], np.ndarray):
                np.testing.assert_array_equal(actual[key], expected[key])
            else:
                assert actual[key] == expected[key]
        errors[arm] = 0.0
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v03-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/v03_parity.json"))
    args = parser.parse_args()
    root = args.v03_root.resolve()
    sys.path.insert(1, str(root / "repo/actioncoord_experiments/src"))
    sys.path.insert(1, str(root / "pi05_runtime/packages/openpi-client/src"))
    sys.path.insert(1, str(root / "pi05_runtime/src"))
    config = load_config(Path(__file__).parents[1] / "configs/v05_default.yaml")
    from actioncoord.data.regenerated_rgbd_dataset import RegeneratedRGBDDataset
    from openpi.models.model import ModelType
    from actioncoord.integration.v03_inputs import V03LiberoInputs
    import openpi_client.image_tools as image_tools

    yy, xx = np.indices((256, 256))
    synthetic = dict(agentview_rgb=np.stack((xx, yy, (xx+3*yy) % 256), -1).astype(np.uint8),
                     wrist_rgb=np.stack((yy, xx, (2*xx+yy) % 256), -1).astype(np.uint8),
                     agentview_depth_m=np.where((xx < 100) & (yy < 130), 1., 2.).astype(np.float32),
                     intrinsic=np.array([[200., 0, 128.], [0, 200., 128.], [0, 0, 1.]]),
                     camera_to_world=np.eye(4), ee_world_m=np.array([0., 0, 1]),
                     state8=np.array([0, 0, 1, 0, 0, 0, .04, -.04], dtype=np.float32),
                     task="synthetic parity", actions=np.zeros((10, 7), dtype=np.float32))
    report = {"source_root": str(root), "synthetic": compare(synthetic, config), "real_frames": []}
    dataset = RegeneratedRGBDDataset(root / "configs/v03_training_manifest.json")
    previous_end = 0
    for ep, row in enumerate(dataset.episodes[:10]):
        for frame in np.linspace(0, row["frames"]-1, 5, dtype=int):
            sample = dataset[previous_end+int(frame)]
            report["real_frames"].append({"episode": row["episode_index"], "frame": int(frame),
                                          "max_errors": compare(sample, config)})
        previous_end += row["frames"]
    prepared = OpenPIAdapter(profile(config, "V03")).prepare(synthetic)
    old = V03LiberoInputs(ModelType.PI05, "R3")(prepared)
    new = OpenPIThirdSlotInputs(ModelType.PI05)(prepared)
    for key in old["image"]:
        np.testing.assert_array_equal(old["image"][key], new["image"][key])
    assert all(new["image_mask"].values()) and len(new["image"]) == 3
    report.update(passed=True, frames=len(report["real_frames"]), third_slot_parity=True,
                  source_dense_sha256=hashlib.sha256((root / "repo/actioncoord_experiments/src/actioncoord/geometry/dense.py").read_bytes()).hexdigest(),
                  rgb_resize_module=image_tools.__file__, rgb_resize_default=int(image_tools.resize_with_pad.__defaults__[0]),
                  metadata_discrepancy="Frozen JSON says Lanczos; actual frozen callable defaults to PIL BILINEAR enum=2. Executable behavior retained; JSON unchanged.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": True, "real_frames": report["frames"], "profiles": 4,
                      "geometry_max_error": 0., "policy_input_max_error": 0., "third_slot_parity": True}))


if __name__ == "__main__":
    main()
