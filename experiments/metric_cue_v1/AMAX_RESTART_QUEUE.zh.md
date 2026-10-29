# L09 A6000节点重启与资源等待

- [Experiment] 2026-10-29用户要求重新开始实验。旧147节点最后成功记录1547次更新，随后写入旧NAS路径metrics.jsonl时出现FileNotFoundError；checkpoint目录为空。新任务从已验证pi05_base开始，旧未保存更新不计入30k预算，原始输出保留。
- [Code] 已核实NAS当前根目录为/home/Dataset2/yuqiuting。两个数据manifest及基座来源receipt的哈希与冻结值一致；configs/metric_workflow_node145.yaml仅作为已校验的NAS路径配置使用，GPU编号由启动器按实际节点覆盖。既有path_mappings处理manifest中的旧/home/Dataset2/yqt路径。
- [Experiment] 147节点NVML初始化失败；145节点根磁盘已满，且真实GPU查询超时，查询进程进入os_acquire_rwlock_read内核等待，未通过启动前资源核验。对应145运行树和失败preflight留存；没有在145节点启动GPU训练。
- [Code] A6000主节点的GPU查询正常。正式重启使用独立、干净的运行树/home/lixuanyi/XuanyiLi/work/rulervla-metric-restart-runtime，保留开发树中的其他未提交修改。启动入口scripts/start_metric_cue_amax.sh；outputs指向/home/Dataset2/yuqiuting/rulervla/metric_cue_runs。
- [Code] A6000使用原依赖环境的NCCL库、CUDA ptxas和Python。TMPDIR、CUDA缓存、Python缓存、JAX缓存及全部训练产物放在NAS；新目录outputs/metric_cue_pilot/L09_restart_amax/L09_spatial_M2_seed42，启动日志同父目录launch.log，启动器记录同父目录launch/。
- [Code] 继续使用现有单次启动器。共享run ID锁防止重复启动；只有当前发布条件、真实输入和资源门槛全部通过后才执行训练。容量不足时每60秒重新检查，不导入JAX，不加载模型，不占用GPU；输入或运行时查询错误立即退出并保留错误。
- [Code] batch8、microbatch1、累计8、full fine-tuning、EMA .999、CPU AdamW、双卡FSDP、30000次更新及10000/20000/30000保存点保留。protocol revision4 SHA-256为c924ef9cdf7ae2de11760371baf525d845edc4275020ec1eae6d68286e29e0e1；S29及策略评测保持关闭。启动后的八处资源检查继续注释。
- [Experiment] 当前CPU可用约92 GiB，低于160 GiB门槛。启动器运行和waiting_resources状态表示正在等待资源；只有出现真实optimizer_updates才说明正式训练已开始，不能把排队状态记为训练成功。
