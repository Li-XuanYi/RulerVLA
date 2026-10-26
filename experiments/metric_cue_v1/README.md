# L09 公制提示四组实验

研究依据是 2026-10-24 的 `reaserch.md`。本实验仅覆盖六通道 Metric Ruler、Motion Echo 和原 π0.5 Flow Matching 动作目标。M0 使用空白提示，M1 使用固定 28 像素的 XYZ 方向线段，M2 使用以当前 EE 为原点、沿 world XYZ 正负方向延伸 30 cm 的公制刻度，M3 在 M2 上加入同一 episode 过去五个环境控制步的实测 EE 位移。

`protocol.json` 固定十二次训练、数据身份、共同配方、标准 Spatial 评测口径与进入实验的条件。`run_matrix.json` 保存全部计划运行。`release.json` 保存当前条件的核验状态。每次训练结果进入 `outputs/metric_cue_pilot/<run_id>/`；该目录中的 run manifest、模型 checkpoint 和实际执行记录需要在结果分析时核对。

当前四组真实输入、连续专家动作、M3 历史状态以及完整基座模型的 CPU 载入已经核验。432 个训练 episode、52970 帧的 RGB-D 与几何文件逐一完成哈希和内容检查，记录位于 `outputs/metric_cue_pilot/full_data_audit_20261025/`。标准 Spatial 的十个任务、每任务五十个官方初始状态已按 LIBERO 提交 `8f1084e` 锁定在 `paired_scene_manifest.json`，包含 BDDL、初始状态文件和全部 500 个状态的哈希。指定黑碗任务的 scene index 0 已完成真实离屏渲染、双视角 RGB-D 与相机参数采集，以及一个控制步，记录位于 `outputs/metric_cue_pilot/simulator_gate_task2_scene0_20261026/receipt.json`。完整模型 GPU 更新、零提示数值等价、checkpoint 恢复、位置场景有效性和完整 rollout 尚无通过记录，训练与评测发布标志保持关闭。

训练输入使用 `scripts/metric_cue_full_data_audit.py` 核对完整数据；`tests/real_metric_arm_contract.py` 检查实际样本、正式批次、模型输入和在线会话。训练入口为 `scripts/metric_cue_train.py`。在线服务与模拟器客户端分别位于 `scripts/metric_cue_serve.py`、`scripts/metric_cue_eval.py`；两个入口在发布条件未通过时直接停止。模拟器使用独立 Python 3.11 环境和指定提交的 LIBERO 源码，依赖版本见 `requirements/metric-cue-eval-linux.lock`。LIBERO benchmark、robosuite 与单场景离屏运行已核验；完整 rollout 尚未运行。

黑碗目标位置泛化归入 S29。协议指定的 Spatial task index 2 对应“从桌面中心拿起黑碗并放到盘子上”。既有 S28 候选网格仍待完成可见性、无碰撞抓取姿态和放置稳定性检查。S29 的训练支持区域、内部留出区域和外部可行区域必须在执行前固定。三种主要比较是 M1−M0、M2−M1、M3−M2。任务成功率是主要结局；每个训练种子与任务均保留全部 planned rollout 的成功和工程异常记录。

M2−M1 同时改变标尺图案和单位说明，能够估计完整公制提示的增量。`reaserch.md` 要求的固定像素标尺控制尚需基于训练数据确定两路相机的固定像素刻度，并形成独立训练与评测记录；目前不能把四组任务表现单独解释为模型学会了真实物理刻度。
