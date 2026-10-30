set -euo pipefail
metric_runtime_root="${METRIC_RUNTIME_ROOT:-/home/lixuanyi/XuanyiLi/work/rulervla-metric-lora-r5-runtime}"
metric_openpi_root="${METRIC_OPENPI_ROOT:-/home/lixuanyi/XuanyiLi/code/openpi}"
metric_workflow_config="${METRIC_WORKFLOW_CONFIG:-configs/metric_workflow_lora.yaml}"
metric_nccl_root="${METRIC_NCCL_ROOT:-$metric_openpi_root/.venv/lib/python3.11/site-packages/nvidia/nccl/lib}"
cd "$metric_runtime_root"
metric_smoke_key="${1:?需要本次smoke的唯一目录名}"
if [[ ! "$metric_smoke_key" =~ ^[A-Za-z0-9_]+$ ]]; then
    exit 2
fi
export LD_PRELOAD="$metric_nccl_root/libnccl.so.2"
export LD_LIBRARY_PATH="$metric_nccl_root"
export PATH="$metric_openpi_root/.venv/lib/python3.11/site-packages/nvidia/cuda_nvcc/bin:$PATH"
export CUDA_CACHE_MAXSIZE=4294967296
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export TMPDIR="$metric_runtime_root/build/metric_lora_cache/temp"
export CUDA_CACHE_PATH="$metric_runtime_root/build/metric_lora_cache/cuda"
export XDG_CACHE_HOME="$metric_runtime_root/build/metric_lora_cache/xdg"
export PYTHONPYCACHEPREFIX="$metric_runtime_root/build/metric_lora_cache/pycache"
mkdir -p "$TMPDIR" "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME" "$PYTHONPYCACHEPREFIX"
for metric_arm in M2 M0 M1 M3; do
    "$metric_openpi_root/.venv/bin/python" scripts/metric_nccl_runtime.py
    "$metric_openpi_root/.venv/bin/python" scripts/metric_cue_train.py \
        --config "$metric_workflow_config" --mode smoke --steps 2 \
        --arm "$metric_arm" --auto-gpus --wait-for-resources \
        --out "outputs/metric_cue_pilot/L09_lora_r5_smoke/$metric_smoke_key/$metric_arm"
done
