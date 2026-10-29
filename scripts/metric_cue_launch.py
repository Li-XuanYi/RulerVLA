import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description="等待真实资源准入后启动一次L09正式训练")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/metric_workflow_lora.yaml")
    parser.add_argument("--arm", choices=("M0", "M1", "M2", "M3"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpu-candidates", type=int, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--wait-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    if args.poll_seconds < 15 or len(set(args.gpu_candidates)) != len(args.gpu_candidates):
        raise ValueError("Use distinct GPU candidates and polling of at least 15 seconds")
    from rulervla.training.metric_resources import load_run_config, preflight, query_gpus
    from rulervla.training.metric_protocol import load_formal_metric_run
    run, protocol = load_formal_metric_run(
        ROOT, args.arm, args.seed, load_run_config(args.config))
    if args.out.name != run["formal_run_id"] or args.out.exists():
        raise ValueError("A new formal run directory with the frozen run ID is required")
    if len(args.gpu_candidates) < protocol["training"]["gradient_sharding"]["devices"]:
        raise ValueError("Not enough candidate GPUs")
    lock_dir = ROOT / "outputs/metric_cue_pilot/locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock = (lock_dir / (run["formal_run_id"] + ".lock")).open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.set_inheritable(lock.fileno(), True)
    args.wait_dir.mkdir(parents=True, exist_ok=False)
    attempt = 0
    while True:
        # 每轮重新核验协议和真实数据，发布条件变化时立即停止。
        run, protocol = load_formal_metric_run(
            ROOT, args.arm, args.seed, load_run_config(args.config))
        gpus = query_gpus()
        if isinstance(gpus, dict):
            raise RuntimeError(gpus["error"])
        available = sorted(
            (gpu for gpu in gpus if gpu["index"] in args.gpu_candidates),
            key=lambda gpu: gpu["free_mib"], reverse=True)
        devices = protocol["training"]["gradient_sharding"]["devices"]
        if len(available) < devices:
            raise RuntimeError("Candidate GPU disappeared")
        selected = [gpu["index"] for gpu in available[:devices]]
        run["resources"]["gpu_indices"] = selected
        admission = preflight(run)
        attempt += 1
        (args.wait_dir / f"preflight_{attempt:04d}.json").write_text(
            json.dumps(admission, indent=2))
        if admission["input_blockers"]:
            phase = "blocked_inputs"
        elif isinstance(admission["gpu_snapshot"], dict):
            phase = "blocked_runtime"
        else:
            phase = "launching" if admission["launch_allowed"] else "waiting_resources"
        status = dict(state=phase,
                      attempt=attempt, pid=os.getpid(), selected_gpu_indices=selected,
                      blockers=admission["blockers"], jax_imported="jax" in sys.modules,
                      protocol_sha256=run["protocol_sha256"])
        (args.wait_dir / "status.json").write_text(json.dumps(status, indent=2))
        print(json.dumps(status), flush=True)
        if phase == "blocked_inputs":
            raise RuntimeError("; ".join(admission["input_blockers"]))
        if phase == "blocked_runtime":
            raise RuntimeError(admission["gpu_snapshot"]["error"])
        if admission["launch_allowed"]:
            if "jax" in sys.modules:
                raise RuntimeError("Resource waiting must not initialize JAX")
            command = [
                sys.executable, str(ROOT / "scripts/metric_cue_train.py"),
                "--config", str(args.config.resolve()), "--mode", "train",
                "--arm", args.arm, "--seed", str(args.seed),
                "--out", str(args.out.resolve()), "--gpu-indices", *map(str, selected)]
            (args.wait_dir / "command.json").write_text(json.dumps(command, indent=2))
            os.execv(sys.executable, command)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
