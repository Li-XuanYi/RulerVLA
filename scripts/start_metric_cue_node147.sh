set -euo pipefail
metric_runtime_root=/home/lixuanyi/XuanyiLi/metric_cue_runtime/RulerVLA
cd "$metric_runtime_root"
mkdir -p build/metric_runtime/temp
export LD_PRELOAD=/home/lixuanyi/XuanyiLi/metric_cue_runtime/nccl2289/nvidia/nccl/lib/libnccl.so.2
export LD_LIBRARY_PATH=/home/lixuanyi/XuanyiLi/metric_cue_runtime/nccl2289/nvidia/nccl/lib
export PATH=/home/lixuanyi/XuanyiLi/metric_cue_runtime/environment/.venv/lib/python3.11/site-packages/nvidia/cuda_nvcc/bin:$PATH
export CUDA_CACHE_MAXSIZE=4294967296
export NCCL_DEBUG=INFO
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export TMPDIR="$metric_runtime_root/build/metric_runtime/temp"
exec /home/lixuanyi/XuanyiLi/metric_cue_runtime/environment/.venv/bin/python scripts/metric_cue_launch.py \
    --arm M2 --seed 42 --gpu-candidates 0 1 2 3 4 5 6 \
    --out outputs/metric_cue_pilot/L09_restart_no_runtime_gate/L09_spatial_M2_seed42 \
    --wait-dir outputs/metric_cue_pilot/L09_restart_no_runtime_gate/launch
