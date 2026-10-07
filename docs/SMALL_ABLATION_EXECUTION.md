# S22/S23 GPU 执行记录（2026-10-07）

冻结研究设计见 `SMALL_ABLATION_PROTOCOL.md`。本记录说明实际工程放行和进程状态，不更改原始 run_matrix、条件列表和共同训练参数。

## 最新：正式训练已实际开始

下方为本日早先的等待记录。经过真实GPU5两步full-FT和完整保存恢复验收，已放行低缓存分配方式：

- **S23 ActionCoord-only：GPU5，训练PID276641，目标30000更新；核验时已完成3次正式更新。** loss 0.108287→0.096945→0.095144，gradient norm有限且非零。
- GPU5新队列PID276602；ActionCoord完成后接Ruler-only。GPU4新队列PID276486；CameraXYZ等待容量，当前GPU4有其他任务高占用。
- official pi05_base独立重新初始化，432 episodes/52970 frames，full FT、有效batch32、seed42、warmup1k/30k cosine、四槽、horizon10均未变。
- `XLA_PYTHON_CLIENT_ALLOCATOR=platform`减少缓存保留；优化器/EMA仍在CPU。每次更新间可能只看到很少新增GPU显存，应以真实metrics/status判断训练是否推进。
- checkpoint改为CPU不可变视图的严格阻塞保存，保存与Dataset2发布、SHA核验全部结束前不更新或donate state；保存前后SHA一致，完整weights/optimizer/EMA恢复逐字节一致。未改模型精度、架构或训练方式。
- 两步smoke使用了33.53亿参数，loss 0.074276/0.075231，grad_norm 0.993012/1.161824，无NaN/OOM。smoke权重没有用于正式warm start。
- 放行容量依实测调整为GPU空闲≥54369MiB、host≥220GiB，初始化仍串行；持续守护保留32GiB主机余量，只在不足时停止小实验子进程。旧380GiB/70000MiB等待门槛不再用于已验收的低缓存配置。
- 真实训练前两批的sample_indices SHA与GPU2/3的Baseline、AimBot完全一致。完整证据在 `reports/S22_S23_actual_training_20261007.json`。
- `/home2`有空间但当前yqt无可写目录，因此没有使用该盘或操作他人目录；暂存仍在自己的RAM目录，Dataset2归档保持完整。

```bash
tail -f /home/yqt/XuanyiLi/RulerVLA/outputs/small_ablation_v1/logs/S23_actioncoord_only_spatial_seed42.log
```

## 已完成

- 补充训练入口调用同一个 `main_spatial_train.main(supplemental=True)`；没有复制训练循环。唯一方法变化为 `SmallAblationAdapter`。
- 原有主实验入口默认 `supplemental=False`，仍限制30k和原有四种方法。
- 五项公平性阻断测试在本地和ZJU通过：改微批/累积、改scheduler、未通过GPU gate、未经结果门槛的P1均不能启动。
- GPU5完成三组真实第一个有效batch的核验：seed42、32个相同训练frame、四个active槽、动作形状32×1×10×32。按主训练JAX dtype canonicalization进行GPU传输，返回值逐字节一致。
- 该GPU核验没有加载模型权重、没有backward、没有optimizer update、没有rollout；不能冒充完整fine-tuning smoke。
- 首次核验将正常float64→float32 state转换误判为错误（最大舍入差约5.8e−8）；失败目录完整保留为 `gpu_batch_gate_failed_3041779`。修复检查逻辑，训练数值配置未改。

## 已启动的资源队列

| GPU | 队列PID / session | 分配 |
| --- | --- | --- |
| 4 | 3230031 | S22 CameraXYZ-only：10k |
| 5 | 3230032 | S23 ActionCoord-only：30k，随后Ruler-only：30k |

ActionCoord的9999快照用于S22；S23使用最终29999。三组均独立从官方pi05_base初始化。

**核验时两组都在等待资源，训练子进程为空，正式optimizer updates为0。** GPU4剩42090MiB且有其他任务计算，GPU5剩54369MiB；主机可用约240GiB。放行门槛保持已验证的单GPU配置：空闲显存≥70000MiB、利用率≤20%、主机可用≥380GiB，并确认现有训练已完成初始化。

资源满足后，每一臂首先运行两次full-FT update，保存checkpoint并验证完整weights/optimizer/EMA恢复逐字节一致；通过后再从base重新启动正式训练，smoke权重不用于warm start。

资源队列与大实验共用admission锁；直到smoke结束和正式训练50次update完成才释放，避免两个初始化同时消耗同一份host余量。checkpoint staging继续共用主实验锁。GPU2/3已有训练保持运行。

尚未有训练子进程的GPU4/5大实验等待队列已保留旧状态并重启，PID3230167/3230425。它们等待小实验优先队列完成后继续原分配。没有中断正式训练或其他用户的GPU任务。

## 查看实际进度

ZJU仓库：`/home/yqt/XuanyiLi/RulerVLA`

```bash
cd /home/yqt/XuanyiLi/RulerVLA
cat outputs/small_ablation_v1/training_queue_worker{0,1}_status.json
find outputs/small_ablation_v1/train_records -name status.json -exec cat {} \;
```

队列日志：`outputs/small_ablation_v1/logs/worker0.log`、`worker1.log`。每臂smoke/训练日志同目录，以run_id命名。

最终权重根目录：`/home/Dataset2/yqt/rulervla/small_ablation_v1/checkpoints/`，分别位于S22/camera_xyz_only_seed42、S23/actioncoord_only_seed42、S23/ruler_only_seed42。

逐臂GPU full-state smoke尚未执行，因此目前没有新的策略性能结果。正式评测仍须先放行L06共享evaluation contract；未放行前不能运行200条件主比较或复用不匹配的历史raw。S24仍受正向主要结果门槛限制，S25仍等待正式RulerVLA checkpoint。
