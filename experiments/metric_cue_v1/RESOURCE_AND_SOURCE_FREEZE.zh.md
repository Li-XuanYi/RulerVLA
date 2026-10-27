# 运行时冻结与资源实现修订

本次沿用 research/metric-cue-pilot 隔离开发树，处理用户指定的源码未commit和CPU/GPU资源门槛问题。

- 根据当前真实六通道π0.5参数形状核算参数、AdamW、EMA、梯度及恢复副本；输出在 outputs/metric_cue_pilot/resource_fix/memory_audit.json。
- 为CPU optimizer使用输入缓冲复用，首次EMA保持独立缓冲。原Flow Matching、full fine-tuning、学习率、FP32状态、梯度累积8和EMA .999不变。
- 恢复使用只含shape/dtype/CPU sharding的抽象模板；完整状态签名保存后释放旧状态，再从磁盘恢复。resume通过完整模型静态形状创建模板，不重新载入基座及完整optimizer状态。
- 运行中的资源检查按“当前空闲量＋本进程已用量”核对预算，并记录原始空闲量、本进程占用和CPU RSS；初始预检仍要求真实空闲资源。
- 工程pilot先采用每卡20 GiB、CPU72 GiB的准入预算，执行当前M2、真实batch8、两次更新、EMA .999、保存恢复和源码身份核验；阶段性检查继续生效。该预算是实测前的工程范围，正式值须依据实测峰值和余量确认。
- 只提交训练实际导入的当前运行时及相关Metric Cue配置、CLI、测试和实验说明。保留其他任务改动，不关闭source_provenance，也不提交outputs或checkpoint。
- 正式L09发布条件继续要求当前源码的真实GPU证据及位置场景条件。本次不会通过更改科学门槛启动未经验收的正式训练。
