# L09 NAS路径修复与145节点重启

- [Experiment] 2026-10-29，用户要求重新开始实验。旧运行PID2296578在147节点已不存在；原始日志最后成功记录1547次更新，之后写metrics.jsonl时报旧/home/Dataset2/yqt路径FileNotFoundError，checkpoint目录为空。1548次更新是否已经在内存中完成未记录，不能计入已保存训练预算。
- [Experiment] 数据与旧输出现位于/home/Dataset2/yuqiuting。两个manifest及pi05_base来源receipt的SHA-256分别与冻结协议及原配置一致；原输出保持原位，不删除、不覆盖。不能从该次运行恢复完整训练状态。
- [Code] 新增configs/metric_workflow_node145.yaml，使用已核实的yuqiuting数据根目录，并通过既有path_mappings将manifest中的/home/Dataset2/yqt映射到当前路径。方法、数据manifest身份、batch8、累计8、full fine-tuning、EMA、学习率、CPU AdamW及30k预算保留；protocol revision4 SHA-256保持c924ef9cdf7ae2de11760371baf525d845edc4275020ec1eae6d68286e29e0e1。
- [Code] 新增scripts/start_metric_cue_node145.sh，使用145节点的独立运行树/home/yqt/XuanyiLi/rulervla-metric-runtime和既有Python环境。运行树outputs符号链接指向已核实的NAS新路径；旧目标路径记录保留在本说明中，数据本体不移动。
- [Experiment] 147节点nvidia-smi目前返回Failed to initialize NVML: Unknown Error；A6000主节点CPU可用约97 GiB，未满足160 GiB门槛。145节点已观察到CPU可用约346 GiB，两张候选Blackwell GPU超过每卡28672 MiB；正式启动仍须重新执行完整preflight及全部训练gates。
- [Code] 145根磁盘仅约1.9 GiB空闲；TMPDIR、CUDA_CACHE_PATH、XDG_CACHE_HOME、JAX编译缓存及训练输出全部放到NAS。继续通过LD_PRELOAD选择已验证的NCCL2.28.9，并选择环境中的CUDA ptxas。
- [Code] 新运行目录outputs/metric_cue_pilot/L09_restart_node145/L09_spatial_M2_seed42，日志同父目录launch.log，启动器记录同父目录launch/。仍使用共享run ID锁，旧运行目录保留。正式训练从已验证pi05_base重新开始，不计入旧未保存更新。
- [Code] scripts/metric_cue_train.py中八处启动后资源检查保持注释，不添加资源终止监视器。启动前数据、来源、GPU与CPU门槛继续生效，S29及策略评测保持关闭。
- [Experiment] 本文件记录重启准备及路径修复。实际是否启动和推进，须核验新目录的manifest、status.json、metrics.jsonl和对应receipt。
