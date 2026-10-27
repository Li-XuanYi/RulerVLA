import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gpu-indices", type=int, nargs=2, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from rulervla.training.metric_resources import load_run_config, preflight, runtime_capacity
    run = load_run_config(args.config)
    run["resources"]["gpu_indices"] = args.gpu_indices
    args.out.mkdir(parents=True, exist_ok=False)
    admission = preflight(run)
    (args.out / "preflight.json").write_text(json.dumps(admission, indent=2))
    if not admission["launch_allowed"]:
        raise RuntimeError("; ".join(admission["blockers"]))
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpu_indices))
    os.environ["JAX_PLATFORMS"] = "cuda,cpu"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    import jax
    import numpy as np
    from rulervla.adapters.metric_pipeline import MetricPipeline
    from rulervla.training.main_data import SpatialDataset
    from rulervla.training.metric_workflow import prepare_batch, source_provenance
    if len(jax.devices("gpu")) != 2:
        raise ValueError("Exactly two GPU devices are required")
    pipeline = MetricPipeline.from_files(run["cue_config"], run["norm_stats_dir"])
    dataset = SpatialDataset(run["data"]["rgbd_root"], run["data"]["geometry_root"],
                             max_cache_bytes=2 << 30, path_mappings=run["path_mappings"])
    with jax.default_device(jax.devices("cpu")[0]):
        batch = prepare_batch(dataset, [20, 21], pipeline)
    images = np.asarray(batch.observation.images["base_0_rgb"])
    manifest = source_provenance(ROOT)
    manifest.update(sample_identities=batch.identities, environment={
        name: os.environ.get(name) for name in (
            "PATH", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "JAX_PLATFORMS",
            "NCCL_DEBUG", "NCCL_P2P_DISABLE", "NCCL_CUMEM_HOST_ENABLE")})
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    started = time.monotonic()
    print("real_image_all_gather_started", flush=True)
    gathered = jax.pmap(lambda image: jax.lax.all_gather(image, "devices"),
                       axis_name="devices")(images)
    gathered = np.asarray(jax.block_until_ready(gathered))
    np.testing.assert_array_equal(gathered, np.stack((images, images)))
    measured = runtime_capacity(run)
    receipt = dict(state="complete", seconds=time.monotonic() - started,
                   input_shape=images.shape, output_shape=gathered.shape,
                   bitwise_equal=True, resources=measured)
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
