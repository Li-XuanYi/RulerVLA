# L09 公制提示四组实验

## 当前工程入口

[Code] 当前标尺为正 XYZ、0～15 cm、5 cm 等价横刻度和 EE 原点。统一配置为 `configs/metric_ruler.yaml`、`configs/metric_workflow.yaml`，代码为 `cues/metric_ruler.py` 与 `geometry/camera.py`，审计入口为 `scripts/audit_metric_ruler.py`。合同与清理证据见 `docs/metric_cue/RULER.md`、`reports/metric_ruler_cleanup/report.zh.md`。当前训练、推理共用此流程。

[Code/Experiment] 2026-10-28按用户要求注释启动后的八处资源检查，保留启动前准入；原训练在第14次更新前被显存预算阈值终止，已完成13次且没有完整checkpoint。重启从已验证pi05_base重新开始，输出目录为`outputs/metric_cue_pilot/L09_restart_no_runtime_gate/L09_spatial_M2_seed42`。修订与原始失败证据见[RUNTIME_RESOURCE_POLICY.zh.md](RUNTIME_RESOURCE_POLICY.zh.md)。当前运行身份与进度以run_matrix.json引用的最新receipt为准。

## 当前冻结协议 revision 4

[Code] 四组配置对齐当前正轴标尺；revision1 的协议、运行矩阵、发布状态和五份配置原字节保存在 revisions/r1/frozen_inputs.tar.gz；revision2 的协议、运行矩阵和发布状态保存在 revisions/r2/frozen_inputs.tar.gz。修订说明见 CONFIG_ALIGNMENT_R2.zh.md 与 REVISION3_BATCH8.zh.md。正式训练仍须通过当前协议的全部 gates。

研究依据是当前 `docs/metric_cue/RESEARCH_PLAN.md`，其 SHA-256 写入 protocol.json。本实验覆盖六通道 Metric Ruler、Motion Echo 和原 π0.5 Flow Matching 动作目标。M0 使用空白提示，M1 使用固定 28 像素的正 XYZ 方向线段，M2 使用 EE 原点、world XYZ 正轴 0～15 cm、每 5 cm 等价横刻度且无标尺箭头，M3 在 M2 上加入同一 episode 过去五个环境控制步的实测 EE 位移。

`protocol.json` 固定十二次训练、数据身份、共同配方、标准 Spatial 评测口径与进入实验的条件。`run_matrix.json` 保存全部计划运行。`release.json` 保存当前条件的核验状态。每次训练结果进入 `outputs/metric_cue_pilot/<run_id>/`；该目录中的 run manifest、模型 checkpoint 和实际执行记录需要在结果分析时核对。

[Protocol r4] 当前共同配方为effective batch8、microbatch1、累积8、30000次更新。源码8a699dc已在vla_7_6000完成M2两次真实更新、GPU零Cue等价、CPU设备恢复及207个数组的SHA-256一致性核验；CPU峰值121.304 GiB，GPU采样峰值约21.2 GiB/卡。工程receipt见receipts/L09_gpu_smoke_node147_20261027.json，平台和资源说明见PLATFORM_BLACKWELL_R1.zh.md与RESOURCE_AND_SOURCE_FREEZE.zh.md。用户已确认先开放标准Spatial训练；S29位置场景有效性继续约束评测。revision3原字节归档于revisions/r3/frozen_inputs.tar.gz，修订见REVISION4_TRAINING_RELEASE.zh.md。策略SR及稳定吞吐尚未验证。

当前四组真实输入、连续专家动作、M3 历史状态以及完整基座模型的 CPU 载入已经核验。432 个训练 episode、52970 帧的 RGB-D 与几何文件逐一完成哈希和内容检查，记录位于 `outputs/metric_cue_pilot/full_data_audit_20261025/`。标准 Spatial 的十个任务、每任务五十个官方初始状态已按 LIBERO 提交 `8f1084e` 锁定在 `paired_scene_manifest.json`，包含 BDDL、初始状态文件和全部 500 个状态的哈希。指定黑碗任务的 scene index 0 已完成真实离屏渲染、双视角 RGB-D 与相机参数采集，以及一个控制步，记录位于 `outputs/metric_cue_pilot/simulator_gate_task2_scene0_20261026/receipt.json`。完整模型GPU更新、零Cue等价和checkpoint恢复已通过；训练发布开放，位置场景有效性和完整rollout尚未通过，评测发布保持关闭。

训练输入使用 `scripts/metric_cue_full_data_audit.py` 核对完整数据；`tests/real_metric_arm_contract.py` 检查实际样本、正式批次、模型输入和在线会话。训练入口为 `scripts/metric_cue_train.py`。在线服务与模拟器客户端分别位于 `scripts/metric_cue_serve.py`、`scripts/metric_cue_eval.py`；两个入口在发布条件未通过时直接停止。模拟器使用独立 Python 3.11 环境和指定提交的 LIBERO 源码，依赖版本见 `requirements/metric-cue-eval-linux.lock`。LIBERO benchmark、robosuite 与单场景离屏运行已核验；完整 rollout 尚未运行。

黑碗目标位置泛化归入 S29。协议指定的 Spatial task index 2 对应“从桌面中心拿起黑碗并放到盘子上”。既有 S28 候选网格仍待完成可见性、无碰撞抓取姿态和放置稳定性检查。S29 的训练支持区域、内部留出区域和外部可行区域必须在执行前固定。三种主要比较是 M1−M0、M2−M1、M3−M2。任务成功率是主要结局；每个训练种子与任务均保留全部 planned rollout 的成功和工程异常记录。

M2−M1 同时改变标尺图案和单位说明，能够估计完整公制提示的增量。`reaserch.md` 要求的固定像素标尺控制尚需基于训练数据确定两路相机的固定像素刻度，并形成独立训练与评测记录；目前不能把四组任务表现单独解释为模型学会了真实物理刻度。

## ±15 cm 渲染版本

[Code/Experiment] 新增独立身份 `metric_ruler_15cm_v2`，固定 EE/world XYZ、±15 cm 和每 5 cm 等价刻度，使用 OpenCV 抗锯齿与确定性省略规则。真实 120 帧渲染及训练/推理输入核验见 `reports/metric_ruler_15cm_v2/report.zh.md`，合同见 `docs/metric_cue/RULER_15CM_V2.md`。旧 ±30 cm 配置、协议和 checkpoint 身份保持原定义；指定新工作流后选择旧正式 arm 会被明确拒绝。新版本正式训练尚需匹配的版本化协议和 gates，本次没有启动训练或 rollout，也没有修改当前 release 条件。

## 正轴 0～15 cm 渲染记录（归档）

[Code/Experiment] 该记录使用 `metric_ruler_positive15cm_v3`，只绘制正 XYZ 轴和 +5/+10/+15 cm 等价刻度。该阶段合同为 `docs/metric_cue/RULER_POSITIVE15CM_V3.md`；真实 120 帧及输入核验证据见 `reports/metric_ruler_positive15cm_v3/report.zh.md`。历史协议与运行矩阵继续保留原定义。
