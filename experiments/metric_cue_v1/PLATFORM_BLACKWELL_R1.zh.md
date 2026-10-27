# Blackwell 运行环境 revision1

[Code] 候选主机为 vla_7_6000（ZjuServer147）。独立目录为 /home/lixuanyi/XuanyiLi/metric_cue_runtime/RulerVLA，Python为同级environment/.venv/bin/python；从vla_6_6000已安装的CPython3.11环境复制后，以同级便携Python重建venv路径。已有共享环境及其他任务保持原状。outputs指向 /home/Dataset2/yqt/rulervla/metric_cue_runs，本地剩余空间不用于保存完整checkpoint。

[Experiment] 基础软件为JAX/JAXlib0.5.3、Flax0.10.2、Orbax0.11.13、PyTorch2.7.1+cu128、Transformers4.53.2。独立运行树的14项CPU检查通过，日志为outputs/metric_cue_pilot/node147_deploy/cpu_verified.log。主开发树40项回归及M0–M3真实输入合同已通过；数据缓存复制到共享outputs，研究计划按revision3已冻结哈希纳入Git。

[Experiment] 原环境NCCL2.26.2+cuda12.2在真实图像all_gather首次运行时先进行长时间PTX编译，随后抛出CUDA_ERROR_ILLEGAL_ADDRESS。原始输出为outputs/metric_cue_pilot/node147_deploy/transport_warm.log；GDB观察到驱动PTX编译线程，见transport_shm_gdb.log。等待提示本身不足以认定死锁。GPU更新尚未通过。

[Code] 独立NCCL2.28.9的版本和wheel SHA-256锁定于requirements/metric-cue-blackwell-nccl.lock。运行时通过进程级LD_PRELOAD明确加载nccl2289/nvidia/nccl/lib/libnccl.so.2；基础环境和旧库保持原位。日志核验实际库版本，平台receipt记录共享库哈希。ptxas使用环境中的CUDA12.9.41，显式放在PATH前部。

[Experiment] vla_7_6000 GPU1/3的真实图像all_gather已通过逐值比较：输入[2,224,224,6]，输出[2,2,224,224,6]，1.288秒，实际NCCL2.28.9+cuda12.9；原始记录在outputs/metric_cue_pilot/node147_transport_preload2289_89b7b5c/，日志为node147_deploy/transport_preload2289.log。该结果核验传输正确性，完整模型梯度、CPU optimizer峰值及恢复仍需单独通过。

[Protocol] 当前模型、batch8、microbatch1、累积8、FP32 AdamW/EMA、学习率、数据和评测协议保持revision3定义。环境修订的实测结果须与run manifest、平台库哈希和checkpoint共同保存；完整训练及评测release继续关闭。

参考：[JAX安装说明](https://docs.jax.dev/en/latest/installation.html)、[NCCL2.28.9发布说明](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2293/release-notes/rel_2-28-9.html)、[Blackwell兼容说明](https://docs.nvidia.com/cuda/archive/12.9.2/blackwell-compatibility-guide/index.html)。
