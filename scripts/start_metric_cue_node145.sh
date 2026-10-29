set -euo pipefail
metric_runtime_root=/home/yqt/XuanyiLi/rulervla-metric-runtime
cd "$metric_runtime_root"
export LD_PRELOAD="$metric_runtime_root/build/nccl2289/nvidia/nccl/lib/libnccl.so.2"
export LD_LIBRARY_PATH="$metric_runtime_root/build/nccl2289/nvidia/nccl/lib"
export PATH=/home/yqt/XuanyiLi/openpi/.venv/lib/python3.11/site-packages/nvidia/cuda_nvcc/bin:$PATH
export CUDA_CACHE_MAXSIZE=4294967296
export NCCL_DEBUG=INFO
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export TMPDIR="$metric_runtime_root/outputs/metric_cue_pilot/L09_restart_node145/temp"
export CUDA_CACHE_PATH="$metric_runtime_root/outputs/metric_cue_pilot/L09_restart_node145/cuda_cache"
export XDG_CACHE_HOME="$metric_runtime_root/outputs/metric_cue_pilot/L09_restart_node145/xdg_cache"
mkdir -p "$TMPDIR" "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME"
exec /home/yqt/XuanyiLi/openpi/.venv/bin/python scripts/metric_cue_launch.py \
    --config configs/metric_workflow_node145.yaml \
    --arm M2 --seed 42 --gpu-candidates 0 1 2 3 \
    --out outputs/metric_cue_pilot/L09_restart_node145/L09_spatial_M2_seed42 \
    --wait-dir outputs/metric_cue_pilot/L09_restart_node145/launch
