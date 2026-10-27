# 运行时冻结与资源实现修订

本次沿用 research/metric-cue-pilot 隔离开发树，处理用户指定的源码未commit和CPU/GPU资源门槛问题。

- 根据当前真实六通道π0.5参数形状核算参数、AdamW、EMA、梯度及恢复副本；输出在 outputs/metric_cue_pilot/resource_fix/memory_audit.json。
- 为CPU optimizer使用输入缓冲复用，首次EMA保持独立缓冲。原Flow Matching、full fine-tuning、学习率、FP32状态、梯度累积8和EMA .999不变。
- 恢复使用只含shape/dtype/CPU sharding的抽象模板；完整状态签名保存后释放旧状态，再从磁盘恢复。resume通过完整模型静态形状创建模板，不重新载入基座及完整optimizer状态。
- 运行中的资源检查按“当前空闲量＋本进程已用量”核对预算，并记录原始空闲量、本进程占用和CPU RSS；初始预检仍要求真实空闲资源。
- 当前工程pilot采用每卡28 GiB、CPU128 GiB的初始准入，执行M2、真实batch8、两次更新、EMA .999及完整保存恢复；依据node147首轮实际CPU峰值97.207 GiB及约21.2 GiB/卡的采样占用留出余量。初始20/72 GiB预算保存在首轮运行manifest，后续使用本次测量校准值。
- [Experiment] node147_M2_batch8_5d97c42已完成两次更新，loss为0.101758/0.137281，Cue梯度L2为0.046313/0.026327，梯度均有限。39GiB checkpoint已保存；旧Orbax抽象恢复缺少显式sharding而报错，完整恢复尚未通过。原日志和失败checkpoint保留。
- [Code] 恢复显式构建每个数组的CPU sharding、shape及dtype参数，沿用OpenPI的EMA split/merge与抽象模板类型检查上下文；实际恢复后核验CPU设备及完整签名。保存前先将期望签名写盘，便于核对失败边界。41项CPU回归通过，日志为outputs/metric_cue_pilot/resource_fix/restore_final.log。
- [Code] 当前与旧工作流配置统一为batch8、microbatch1、累积8、EMA .999、30k正式配方与双卡布局；工程pilot固定两次更新。正式release继续关闭。
- 只提交训练实际导入的当前运行时及相关Metric Cue配置、CLI、测试和实验说明。保留其他任务改动，不关闭source_provenance，也不提交outputs或checkpoint。
- 正式L09发布条件继续要求当前源码的真实GPU证据及位置场景条件。本次不会通过更改科学门槛启动未经验收的正式训练。
