#!/usr/bin/env python3
"""Audit exact simulator-state source coverage for the frozen Spatial subset.

Only downloads HDF5 source files to a separate Dataset2 directory. The
LeRobot dataset and trained checkpoints are read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import time

import h5py
from huggingface_hub import HfApi, hf_hub_download
import numpy as np
from PIL import Image
import pyarrow.parquet as pq


MODIFIED_REPO = "clip-rt/modified_libero_hdf5"
MODIFIED_REV = "6a6659f8ac7d580fd594173a0e3abf880c843130"
ORIGINAL_REPO = "yifengzhu-hf/LIBERO-datasets"
ORIGINAL_REV = "f13aa24a3da8c43c7225569f28c562979fa0e35a"


def hash_file(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(8 << 20), b""):
            sha.update(data)
    return sha.hexdigest()


def download(repo: str, revision: str, filename: str, root: Path) -> Path:
    output = root / filename
    if output.is_file():
        return output
    print(json.dumps({"event": "download_start", "repo": repo, "file": filename, "time": time.time()}), flush=True)
    path = Path(hf_hub_download(repo_id=repo, repo_type="dataset", filename=filename,
                                revision=revision, local_dir=root))
    assert path == output and path.is_file()
    print(json.dumps({"event": "download_complete", "file": filename,
                      "size": path.stat().st_size, "time": time.time()}), flush=True)
    return path


def load_episode(lerobot: Path, index: int, cache: dict) -> dict:
    if index in cache:
        return cache[index]
    parquet = lerobot / f"data/chunk-{index // 1000:03d}/episode_{index:06d}.parquet"
    t = pq.read_table(parquet, columns=["actions", "state", "image"])
    out = {"actions": np.asarray(t.column("actions").to_pylist(), dtype=np.float32),
           "state": np.asarray(t.column("state").to_pylist(), dtype=np.float32),
           "images": t.column("image").to_pylist()}
    cache[index] = out
    return out


def match_file(hdf5: Path, episodes: list, lerobot: Path, cache: dict) -> dict:
    result = {"source": str(hdf5), "sha256": hash_file(hdf5), "size": hdf5.stat().st_size,
              "episode_count": len(episodes), "matches": [], "unmatched": []}
    with h5py.File(hdf5, "r") as h5:
        demos = h5["data"]
        names = sorted(demos, key=lambda x: int(x.split("_")[-1]))
        actions = {name: np.asarray(demos[name]["actions"], dtype=np.float32) for name in names}
        result["hdf5_demos"] = len(names)
        for episode in episodes:
            idx = episode["episode_index"]
            ep = load_episode(lerobot, idx, cache)
            matches = [name for name, a in actions.items() if a.shape == ep["actions"].shape and np.array_equal(a, ep["actions"])]
            if len(matches) != 1:
                result["unmatched"].append({"episode_index": idx, "candidate_count": len(matches), "length": len(ep["actions"])})
                continue
            name = matches[0]
            demo = demos[name]
            frame_ids = sorted({0, len(ep["actions"]) // 2, len(ep["actions"]) - 1})
            rgb = []
            for frame in frame_ids:
                encoded = ep["images"][frame]["bytes"]
                le = np.asarray(Image.open(io.BytesIO(encoded)).convert("RGB"), dtype=np.uint8)
                ref = np.asarray(demo["obs/agentview_rgb"][frame], dtype=np.uint8)
                rgb.append({"frame": frame,
                            "rot180_mae": float(np.abs(le.astype(np.int16) - ref[::-1, ::-1].astype(np.int16)).mean()),
                            "raw_mae": float(np.abs(le.astype(np.int16) - ref.astype(np.int16)).mean())})
            state = np.concatenate([demo["obs/ee_states"][()], demo["obs/gripper_states"][()]], axis=1)
            result["matches"].append({"episode_index": idx, "demo": name, "length": len(ep["actions"]),
                                      "state_max_abs_error": float(np.max(np.abs(ep["state"] - state))),
                                      "sample_rgb": rgb})
    return result


def original_match(path: Path, episodes: list, lerobot: Path, cache: dict) -> dict:
    result = {"source": str(path), "sha256": hash_file(path), "size": path.stat().st_size,
              "matches": [], "unmatched": []}
    with h5py.File(path, "r") as h5:
        demos = h5["data"]
        actions = {name: np.asarray(demos[name]["actions"], dtype=np.float32) for name in demos}
        filtered = {}
        for name, raw in actions.items():
            kept, indices = [], []
            for i, action in enumerate(raw):
                previous = kept[-1] if kept else None
                noop = np.linalg.norm(action[:-1]) < 1e-4 and (previous is None or action[-1] == previous[-1])
                if not noop:
                    kept.append(action)
                    indices.append(i)
            filtered[name] = (np.asarray(kept, dtype=np.float32), indices)
        for episode in episodes:
            idx = episode["episode_index"]
            ep_actions = load_episode(lerobot, idx, cache)["actions"]
            matches = []
            for name, raw in actions.items():
                if raw.shape == ep_actions.shape and np.array_equal(raw, ep_actions):
                    matches.append((name, "raw", list(range(len(raw)))))
                    continue
                filt, indices = filtered[name]
                if filt.shape == ep_actions.shape and np.array_equal(filt, ep_actions):
                    matches.append((name, "filtered", indices))
            if len(matches) != 1:
                result["unmatched"].append({"episode_index": idx, "candidate_count": len(matches)})
            else:
                name, mode, indices = matches[0]
                result["matches"].append({"episode_index": idx, "demo": name,
                                          "length": len(ep_actions), "match_mode": mode,
                                          "source_frame_indices": indices})
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lerobot", type=Path, required=True)
    p.add_argument("--store", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--task-limit", type=int, default=10)
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    all_episodes = [json.loads(line) for line in (args.lerobot / "meta/episodes.jsonl").read_text().splitlines()]
    subset = [e for e in all_episodes if 1261 <= e["episode_index"] <= 1692]
    tasks = sorted({t for ep in subset for t in ep["tasks"]})[:args.task_limit]
    api = HfApi()
    modified_info = api.repo_info(MODIFIED_REPO, repo_type="dataset", revision=MODIFIED_REV, files_metadata=True)
    original_info = api.repo_info(ORIGINAL_REPO, repo_type="dataset", revision=ORIGINAL_REV, files_metadata=True)
    modified_files = {s.rfilename: s.size for s in modified_info.siblings}
    original_files = {s.rfilename: s.size for s in original_info.siblings}
    report = {"modified_repo": MODIFIED_REPO, "modified_revision": MODIFIED_REV,
              "original_repo": ORIGINAL_REPO, "original_revision": ORIGINAL_REV,
              "subset_episodes": len(subset), "subset_frames": sum(e["length"] for e in subset),
              "tasks": {}, "created_at": time.time()}
    cache = {}
    for task in tasks:
        stem = task.replace(" ", "_") + "_demo.hdf5"
        mod_file = "libero_spatial_no_noops/" + stem
        orig_file = "libero_spatial/" + stem
        selected = [ep for ep in subset if task in ep["tasks"]]
        item = {"task": task, "episodes": len(selected), "frames": sum(e["length"] for e in selected),
                "modified_filename": mod_file, "modified_expected_size": modified_files.get(mod_file),
                "original_filename": orig_file, "original_expected_size": original_files.get(orig_file)}
        if mod_file not in modified_files:
            item["error"] = "modified file absent"
        else:
            mod = download(MODIFIED_REPO, MODIFIED_REV, mod_file, args.store)
            item["modified"] = match_file(mod, selected, args.lerobot, cache)
            leftovers = [ep for ep in selected if ep["episode_index"] in
                         {x["episode_index"] for x in item["modified"]["unmatched"]}]
            if leftovers and orig_file in original_files:
                orig = download(ORIGINAL_REPO, ORIGINAL_REV, orig_file, args.store / "original")
                item["original_exceptions"] = original_match(orig, leftovers, args.lerobot, cache)
        report["tasks"][task] = item
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"event": "task_complete", "task": task,
                          "modified_match": len(item.get("modified", {}).get("matches", [])),
                          "original_exception_match": len(item.get("original_exceptions", {}).get("matches", [])),
                          "expected": len(selected), "time": time.time()}), flush=True)
    covered = set()
    for item in report["tasks"].values():
        covered.update(r["episode_index"] for r in item.get("modified", {}).get("matches", []))
        covered.update(r["episode_index"] for r in item.get("original_exceptions", {}).get("matches", []))
    report["covered_episode_count"] = len(covered)
    report["covered_frame_count"] = sum(ep["length"] for ep in subset if ep["episode_index"] in covered)
    report["uncovered_episode_indices"] = [ep["episode_index"] for ep in subset if ep["episode_index"] not in covered]
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"event": "complete", "covered_episodes": len(covered),
                      "covered_frames": report["covered_frame_count"],
                      "uncovered": report["uncovered_episode_indices"]}), flush=True)


if __name__ == "__main__":
    main()
