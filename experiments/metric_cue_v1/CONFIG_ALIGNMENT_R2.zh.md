# L09 当前配置对齐 r2

[Code] 本修订修复当前加载器与四组旧 YAML 的字段不兼容。当前方法为 EE/TCP 原点、固定 world/action 正 XYZ、0～15 cm、每 5 cm 等价横刻度、无标尺箭头；M0 空白，M1 固定 28 px 方向线段，M2 当前 MetricRuler，M3 加同 episode 五步实测 Motion Echo。默认工作流关闭 Echo。

## 修订影响与保留范围

- 旧协议、运行矩阵、发布状态及五份配置原字节保存在 revisions/r1/frozen_inputs.tar.gz，SHA-256 为 4b85026ac19c6c2ab63a1327a538336b893288ef9d6f5fd96a2c6de01ad64413。
- 四组 YAML 对齐当前字段；旧工作流入口指向当前默认 Cue。更新 protocol.json 的 revision、方法定义、研究计划哈希及 release.json 的协议哈希，运行矩阵记录相同修订身份。
- 数据、归一化、基座、评测场景和十二个 planned run ID 保留原定义。旧 checkpoint 继续使用其原始源码和合同；本修订不转换已有权重或恢复状态。
- 正式训练配方继续为 batch32、microbatch1、梯度累积32、30000次更新、full fine-tuning、双 GPU FSDP、CPU AdamW/EMA。batch8 是本次可行性评估的候选条件，未替换共同配方。
- 训练和评测发布标志保持关闭；输入合同在本修订重新核验后记录结果。GPU 数值、真实更新、完整状态恢复、位置场景和资源条件继续要求真实证据。

## 验证范围

按 tests/test_metric_formal_config_real.py 检查真实配置、严格合同、关闭状态的启动条件，以及八条真实连续动作输入和完整 π0.5 梯度累积静态形状。先保留旧配置下的失败记录，再执行修复后的相同检查。测试和中间文件写入 outputs/metric_cue_pilot/config_alignment_r2/，不使用 /tmp。

本修订仅修复配置及其身份记录；完整 GPU 更新、吞吐量、峰值显存和策略 SR 均需后续真实运行验证。

## 本轮实际验证

- [Experiment] 修复前，相同真实配置回归10项失败，均指向旧字段或旧研究计划哈希；原始记录为 red.log、red.xml。修复后新增11项全部通过。
- [Experiment] tests/test_metric_formal_config_real.py、tests/test_ruler_cue.py、tests/test_motion_echo_rendering.py 共37项通过，69.91秒；209条警告来自当前依赖的JAX/Flax弃用提示。该结果为上述定向回归，不表示重新执行全仓测试。regression.xml 的 SHA-256 为 c5dad939c3c03e56e0dea31f6625191567030a5bb688414296e749efdd1417bf。
- [Experiment] 四组真实输入入口通过，M0 Cue为空白，M1为方向线，M2为当前公制标尺，M3包含实测历史。real_arm_input.json 的 SHA-256 为 913fa3d69420e53cbe195769d4a7885a409d04a473c5020f8e1f50763fd0de6c，release.json 已记录该路径与身份。
- [Experiment] 四组 CLI preflight 均通过配置读取并到达数据与资源检查；全部因GPU2/5空闲显存及CPU可用内存不足而返回blocked，均未导入JAX或启动GPU。完整GPU更新、恢复及位置场景条件继续关闭。
- [Code] 当前协议 SHA-256 为 5bef268ce0ad87a56fd3b150cebcd5370917c963e4b4fc81fd7ba9e5bacd69a5，与运行矩阵和发布记录一致。Ruff及git diff --check通过，CodeGraph已同步。所有原始日志保存在 outputs/metric_cue_pilot/config_alignment_r2/。
