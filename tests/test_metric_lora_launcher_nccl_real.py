import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(
    os.name != "posix" or not os.environ.get("METRIC_LORA_TEST_NCCL_ROOT"),
    reason="需要真实Linux动态加载器和已核验的NCCL库",
)
@pytest.mark.parametrize(
    ("launcher", "config", "extra_args", "rejection"),
    [
        (
            "start_metric_cue_amax.sh",
            "configs/metric_workflow_lora_node145.yaml",
            (),
            "Formal Metric Cue training gates remain closed",
        ),
        (
            "run_metric_lora_smokes_amax.sh",
            "configs/metric_workflow_node145.yaml",
            ("nccl_binding_rejection",),
            "require explicit LoRA finetuning",
        ),
    ],
)
def test_real_launcher_honors_nccl_override_before_closed_gate(
    launcher, config, extra_args, rejection
):
    release = json.loads((ROOT / "experiments/metric_cue_v1/release.json").read_text())
    assert release["training_released"] is False
    nccl_root = Path(os.environ["METRIC_LORA_TEST_NCCL_ROOT"]).resolve()
    library = nccl_root / "libnccl.so.2"
    assert library.is_file()
    environment = dict(
        os.environ,
        METRIC_RUNTIME_ROOT=str(ROOT),
        METRIC_OPENPI_ROOT=str(Path(sys.prefix).parent),
        METRIC_WORKFLOW_CONFIG=config,
        METRIC_NCCL_ROOT=str(nccl_root),
        CUDA_VISIBLE_DEVICES="",
        JAX_PLATFORMS="cpu",
        METRIC_NCCL_SHA256="1792291e26d2b27fe51e59cf4dbd361dbfa84cbfe8ee8196c087c7ebf5146d48",
    )
    environment.pop("LD_PRELOAD", None)
    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / launcher), *extra_args],
        cwd=ROOT, env=environment, text=True, capture_output=True, timeout=10,
    )
    assert result.returncode != 0
    assert rejection in result.stderr
    profiles = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    assert len(profiles) == 1
    assert profiles[0]["schema"] == "rulervla_metric_nccl_runtime_v1"
    assert profiles[0]["library_path"] == str(library)
    assert profiles[0]["version"] == 22809
    assert profiles[0]["sha256"] == environment["METRIC_NCCL_SHA256"]
