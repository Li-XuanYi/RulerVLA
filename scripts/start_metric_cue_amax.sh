set -euo pipefail
metric_runtime_root="${METRIC_RUNTIME_ROOT:-/home/lixuanyi/XuanyiLi/work/rulervla-metric-lora-r5-runtime}"
metric_openpi_root="${METRIC_OPENPI_ROOT:-/home/lixuanyi/XuanyiLi/code/openpi}"
metric_workflow_config="${METRIC_WORKFLOW_CONFIG:-configs/metric_workflow_lora.yaml}"
cd "$metric_runtime_root"
export LD_PRELOAD="$metric_openpi_root/.venv/lib/python3.11/site-packages/nvidia/nccl/lib/libnccl.so.2"
export LD_LIBRARY_PATH="$metric_openpi_root/.venv/lib/python3.11/site-packages/nvidia/nccl/lib"
export PATH="$metric_openpi_root/.venv/lib/python3.11/site-packages/nvidia/cuda_nvcc/bin:$PATH"
export CUDA_CACHE_MAXSIZE=4294967296
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export TMPDIR="$metric_runtime_root/build/metric_lora_cache/temp"
export CUDA_CACHE_PATH="$metric_runtime_root/build/metric_lora_cache/cuda"
export XDG_CACHE_HOME="$metric_runtime_root/build/metric_lora_cache/xdg"
export PYTHONPYCACHEPREFIX="$metric_runtime_root/build/metric_lora_cache/pycache"
mkdir -p "$TMPDIR" "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME" "$PYTHONPYCACHEPREFIX"
exec "$metric_openpi_root/.venv/bin/python" scripts/metric_cue_launch.py \
    --config "$metric_workflow_config" \
    --arm M2 --seed 42 --gpu-candidates 0 1 2 3 4 5 6 7 \
    --out outputs/metric_cue_pilot/L09_lora_r5/L09_spatial_M2_lora_seed42 \
    --wait-dir outputs/metric_cue_pilot/L09_lora_r5/launch_M2_seed42
