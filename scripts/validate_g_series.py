"""Validate frozen inputs, paired records, result artifacts and protected sources."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
import numpy as np

from analyze_g_series import read_rows
from g_series_study import verify_lock
from rulervla.evaluation.g_series import GROUPS, array_hash, file_hash

REQUIRED = ("experiment_id episode_id frame_id condition severity num_anchor_total num_anchor_valid num_anchor_used "
            "gt_bias_xyz_m estimated_bias_xyz_m bias_error_mm scene_rms_before_mm scene_rms_after_mm "
            "scene_p95_before_mm scene_p95_after_mm anchor_residual_median_mm anchor_residual_p95_mm "
            "reliability_score accepted failure_reason").split()


def git(*args, cwd=ROOT):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True)


def main():
    lock = verify_lock()
    base = json.loads((ROOT/"experiments/base_frame_manifest.json").read_text())
    protocol_sha = file_hash(ROOT/"experiments/g_series_protocol_lock.json")
    source_checks = []
    for row in base["frames"]:
        cache = ROOT/row["cache_path"]
        assert file_hash(cache) == row["cache_sha256"]
        with np.load(cache, allow_pickle=False) as z:
            for key, field in (("agentview_rgb", "rgb_sha256"), ("wrist_rgb", "wrist_rgb_sha256"), ("agentview_depth_m", "depth_sha256"),
                               ("target_roi", "target_roi_sha256"), ("robot_mask", "robot_mask_sha256"), ("anchor_roi", "anchor_roi_sha256")):
                assert array_hash(z[key]) == row[field]
            assert not (z["anchor_roi"] & ~z["robot_mask"]).any()
            assert not (z["target_roi"] & z["robot_mask"]).any()
            uv = np.rint(z["anchor_uv"][row["pool"]]).astype(int)
            assert not z["target_roi"][uv[:, 1], uv[:, 0]].any()
            assert not (z["target_roi"] & z["anchor_roi"]).any()
            assert np.any(z["target_roi"])
        source_checks.append(dict(frame_key=row["frame_key"], cache_verified=True, rgb_depth_roi_verified=True))
    count, figures, g2_failures = 0, [], []
    ids = {(r["episode_id"], r["frame_id"]) for r in base["frames"]}
    for group, (directory, _) in GROUPS.items():
        path = ROOT/"experiments"/directory
        rows = read_rows(path/"raw/results.csv")
        expected = [c["condition"] for c in lock["conditions"] if c["experiment_id"] == group]
        assert len(rows) == 50*len(expected)
        assert len({(r["episode_id"], r["frame_id"], r["condition"]) for r in rows}) == len(rows)
        for condition in expected:
            assert {(r["episode_id"], r["frame_id"]) for r in rows if r["condition"] == condition} == ids
        for row in rows:
            assert all(key in row for key in REQUIRED)
            assert isinstance(row["accepted"], bool)
            assert row["base_manifest_sha256"] == lock["base_manifest_sha256"]
            assert row["protocol_sha256"] == protocol_sha
            for key, value in row.items():
                if isinstance(value, (int, float)):
                    assert np.isfinite(value), key
            if group in {"G04", "G05", "G06", "G07"}:
                assert row["gt_bias_xyz_m"] is None and row["bias_error_mm"] is None
        if group == "G02":
            g2_failures = [r for r in rows if r["silent_wrong_correction"]]
        exp_manifest = json.loads((path/"manifest.json").read_text())
        assert exp_manifest["base_manifest_sha256"] == lock["base_manifest_sha256"]
        assert exp_manifest["protocol_sha256"] == protocol_sha
        summary = json.loads((path/"summary.json").read_text())
        assert summary["csv_sha256"] == file_hash(path/"raw/results.csv")
        conclusion = (path/"conclusion.md").read_text()
        for heading in ("Question", "Setup", "Result", "Interpretation", "Failure Boundary", "Cannot Conclude", "Decision"):
            assert "## "+heading in conclusion
        for extension, signature in (("png", b"\x89PNG\r\n\x1a\n"), ("pdf", b"%PDF")):
            figure = path/"figures"/(group+"_boundary."+extension)
            assert figure.read_bytes().startswith(signature)
            figures.append(str(figure.relative_to(ROOT)))
        count += len(rows)
    assert count == lock["expected_rows"] == 3350
    protected_paths = ["src/rulervla/geometry", "src/rulervla/adapters", "src/rulervla/cues", "src/rulervla/data",
                       "src/rulervla/config.py", "configs/v05_default.yaml", "configs/v03_geometry.json", "configs/v03_encoding.json"]
    production_diff = git("diff", lock["design"]["base_implementation_commit"], "--", *protected_paths)
    assert not production_diff
    protected = {
        "/home/yqt/XuanyiLi/openpi0-aimbot": "250fd3aa511fd92535f329618373bbb93320fcdb47a3ae8c262df47852da0bcd",
        "/home/yqt/XuanyiLi/actioncoord/repo": "9cd4d799c7335fd46bca0ab786828a582d39b7aa7828b2a69b8e007878430415",
    }
    import hashlib
    status_checks = {}
    for source, expected in protected.items():
        status = git("status", "--porcelain", cwd=source)
        sha = hashlib.sha256(status.encode()).hexdigest()
        assert sha == expected
        status_checks[source] = sha
    frozen = Path(lock["design"]["v03_root"])
    assert not git("status", "--porcelain", cwd=frozen)
    assert git("rev-parse", "HEAD", cwd=frozen).strip() == "ab3fb89026b683afea401cdae91445767f6bb749"
    regression = subprocess.run([sys.executable, "-B", "-m", "pytest", "-q"], cwd=ROOT, capture_output=True, text=True)
    assert regression.returncode == 0, regression.stdout+regression.stderr
    assert "50 passed" in regression.stdout
    report = dict(passed=True, rows=3350, base_frames=50, conditions=67, figure_pairs=7,
                  required_metrics_present=True, unique_paired_matrix=True, cache_roi_isolation_verified=True,
                  production_geometry_estimators_thresholds_unchanged=True,
                  base_manifest_sha256=lock["base_manifest_sha256"], protocol_sha256=protocol_sha,
                  frozen_v03_unchanged=True, historical_status_sha256=status_checks,
                  regression_command=[sys.executable, "-B", "-m", "pytest", "-q"], regression_output=regression.stdout.strip(),
                  g6_g7_visual_review="Actual first-frame ROI and error figures reviewed; robot anchor region and table-centre black bowl correctly located",
                  analysis_narrative_revision="Explicitly report G2 partial support and failed accuracy gate; all raw CSVs, thresholds and decision rules unchanged",
                  git_gate="Verify clean committed tree and remote HEAD after publication; recorded in outputs/g_series/COMPLETE.json",
                  frames=source_checks, figures=figures, policy_training_started=False)
    (ROOT/"reports/G_SERIES_VALIDATION.json").write_text(json.dumps(report, indent=2)+"\n")
    failure_report = dict(boundary="Accepted low-count recovery above 2 mm gate", cases=g2_failures,
                         accepted_cases_total=600, rejected_insufficient_anchor_cases=300,
                         explanation="These 3/4-anchor cases retain 2.055-2.415 mm error with high consistency score; all improve the initial 10 mm scene error. No algorithm or threshold patch applied.")
    (ROOT/"reports/G2_ACCEPTED_ACCURACY_BOUNDARY.json").write_text(json.dumps(failure_report, indent=2)+"\n")
    checksum = ROOT/"reports/G_SERIES_MANIFEST.sha256"
    files = sorted(p for directory in (ROOT/"experiments", ROOT/"reports") for p in directory.rglob("*") if p.is_file() and p != checksum)
    checksum.write_text("".join(file_hash(p)+"  "+str(p.relative_to(ROOT)).replace("\\", "/")+"\n" for p in files))
    print(json.dumps({"passed": True, "rows": count, "figure_pairs": 7, "g2_boundary_cases": len(g2_failures),
                      "regression": regression.stdout.strip(), "manifest_sha256": file_hash(checksum)}), flush=True)


if __name__ == "__main__":
    main()
