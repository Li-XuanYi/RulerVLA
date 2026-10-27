import argparse
import json
from pathlib import Path
import subprocess
import sys

import flax.nnx as nnx
import jax
import numpy as np

from rulervla.training.metric_resources import load_run_config
from rulervla.training.metric_workflow import make_training_config


def tree_bytes(tree):
    return sum(int(np.prod(value.shape)) * np.dtype(value.dtype).itemsize
               for value in jax.tree.leaves(tree))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/metric_workflow_pilot.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run = load_run_config(args.config)
    protocol = json.loads((root / "experiments/metric_cue_v1/protocol.json").read_text())
    run["training"] = dict(protocol["training"], seed=42)
    config = make_training_config(run)
    params = nnx.state(nnx.eval_shape(config.model.create, jax.random.key(42)))
    trainable = params.filter(config.trainable_filter)
    tx = config.optimizer.create(config.lr_schedule.create())
    optimizer = jax.eval_shape(tx.init, trainable)
    parameters = tree_bytes(params)
    gradients = tree_bytes(trainable)
    moments = tree_bytes(optimizer)
    ema = parameters if config.ema_decay is not None else 0
    steady = parameters + moments + ema
    tracked = set(subprocess.check_output(["git", "ls-files"], cwd=root, text=True).splitlines())
    changed = set(subprocess.check_output(
        ["git", "diff", "--name-only", "HEAD"], cwd=root, text=True).splitlines())
    runtime = []
    for name, module in sorted(dict(sys.modules).items()):
        path = getattr(module, "__file__", None)
        if path and name.startswith(("rulervla.", "openpi.", "openpi_client.")) and path.endswith(".py"):
            relative = str(Path(path).resolve().relative_to(root))
            runtime.append(dict(module=name, path=relative, tracked=relative in tracked,
                                changed=relative in changed))
    result = dict(
        state="complete", parameter_count=sum(int(np.prod(x.shape)) for x in jax.tree.leaves(params)),
        parameter_bytes=parameters, gradient_bytes=gradients, optimizer_bytes=moments, ema_bytes=ema,
        cpu_steady_bytes=steady, cpu_update_donated_state_and_gradient_bytes=steady + gradients,
        cpu_update_without_reuse_estimate_bytes=2 * steady + gradients,
        cpu_restore_with_two_full_templates_estimate_bytes=3 * steady,
        gpu_per_device_parameter_and_gradient_bytes=(parameters + gradients) // config.fsdp_devices,
        gpu_budget_note="不包含激活、FSDP聚合、编译、通信及临时缓冲，不能作为显存峰值",
        effective_batch=config.batch_size, microbatch=protocol["training"]["micro_batch_size"],
        runtime_sources=runtime,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key != "runtime_sources"}))


if __name__ == "__main__":
    main()
