# L09 启动后资源检查修订

- [Code] 2026-10-28，用户要求通过WSL连接远程，注释启动训练后的显存检查，训练启动后不因资源检查主动停止，并重新启动正式训练。
- [Experiment] 原L09_spatial_M2_seed42在vla_7_6000、GPU0/4、PID216052完成13次更新后，在before_update_14被require_current_capacity终止。该次记录的GPU可用预算为16126/16621 MiB，阈值28672 MiB；日志没有将该次退出归因为CUDA OOM。checkpoints目录没有完整checkpoint。
- [Code] scripts/metric_cue_train.py的八处require_current_capacity调用全部注释，覆盖状态载入、GPU等价、每次更新前后、checkpoint保存及恢复后的资源检查。该函数保留用于查阅；训练路径不再调用它，也不再执行启动后的GPU/CPU资源阈值终止。
- [Code] 启动器及训练入口的启动前preflight保留：每卡至少28672 MiB、CPU可用至少160 GiB，并核验数据、来源及当前训练发布条件。原始模型/数据/训练计算错误继续正常报告。
- [Code] 协议revision4及其SHA-256 c924ef9cdf7ae2de11760371baf525d845edc4275020ec1eae6d68286e29e0e1保留；batch8、microbatch1、累计8、30000次更新、full fine-tuning、学习率、EMA、双卡FSDP及CPU AdamW保留。S29评测条件继续关闭。
- [Code] 原输出outputs/metric_cue_pilot/L09_spatial_M2_seed42及其启动日志、manifest和失败状态原位保留。原13次未保存更新不计入重启预算；本次从已核验pi05_base重新开始，不使用工程smoke状态。
- [Code] 新输出outputs/metric_cue_pilot/L09_restart_no_runtime_gate/L09_spatial_M2_seed42，日志同父目录launch.log，准入记录同父目录launch/。run ID继续使用原L09_spatial_M2_seed42，两个运行目录区分尝试；共享run ID锁继续防止重复启动。
- [Code] 启动环境由scripts/start_metric_cue_node147.sh固定，继续使用NCCL2.28.9的LD_PRELOAD及CUDA12.9.41 ptxas。运行源码须提交后部署到已停止的独立运行树；新manifest绑定新提交，保留上次来源e50e61c的记录。
- [Experiment] 验证目录为outputs/metric_cue_pilot/L09_restart_no_runtime_gate_verification/；原真实失败日志作为修改前证据，修改后检查语法、已有真实回归及实际完整模型更新。
- [Code] 本次用户要求同时约束后续操作：训练启动后，不以资源阈值为由发送终止信号，不添加自动停止训练的监视器。
- [Experiment] 原中止记录见receipts/L09_spatial_M2_seed42_runtime_gate_stop.json；本次启动状态以新输出中的status.json和metrics.jsonl及后续重启receipt为准。
