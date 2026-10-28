# Handoff

## 2026-10-28 L09 关闭启动后资源检查并重启

- [Code] 用户要求使用WSL连接远程，注释启动训练后的显存检查，启动后不可因资源检查主动停止训练。已通过WSL Ubuntu OpenSSH完成修改和部署，scripts/metric_cue_train.py的八处require_current_capacity调用全部注释，CPU/GPU运行中资源阈值均不再主动终止训练；启动前数据/来源/gates及每卡28 GiB、CPU160 GiB准入保留。后续操作不得添加运行中资源终止监视器或以资源阈值主动停止该训练。
- [Experiment] 原PID216052实际完成13次更新，在before_update_14因GPU0/4可用预算16126/16621 MiB低于28672 MiB主动退出；没有完整checkpoint。原目录、日志、manifest、失败状态保留，原13次未保存更新不计入新预算；记录见开发树experiments/metric_cue_v1/receipts/L09_spatial_M2_seed42_runtime_gate_stop.json。
- [Experiment] 新训练L09_spatial_M2_seed42在vla_7_6000 GPU4/3运行，PID2296578，已核实至少1次真实更新；首条loss0.10175843、Cue梯度L2 0.04631297、梯度有限。配方继续为batch8、microbatch1、累计8、full fine-tuning、EMA .999、CPU AdamW、双卡FSDP、30000次更新，源自已验证pi05_base。
- [Code] 冻结运行树/home/lixuanyi/XuanyiLi/metric_cue_runtime/RulerVLA当前源码为9519bafe8af1b76ebd1bb93d6b21ac0d9ba18d8a，manifest SHA-256为8a6476832f9fda75a8b025d0ab93ac695bdbb8579f135c1507e486f38611d43a，git_status为空。协议revision4 SHA-256保持c924ef9cdf7ae2de11760371baf525d845edc4275020ec1eae6d68286e29e0e1，S29及策略评测继续关闭。活动运行树的源码和Git HEAD保持冻结。
- [Experiment] 新输出outputs/metric_cue_pilot/L09_restart_no_runtime_gate/L09_spatial_M2_seed42，日志同父目录launch.log，启动前准入记录同父目录launch/；共享outputs仍指向NAS。启动时GPU4/3空闲93920/90680 MiB、CPU可用484.235 GiB，首轮准入无blockers，flock跨exec保留。新启动receipt为开发树experiments/metric_cue_v1/receipts/L09_spatial_M2_seed42_restart_no_runtime_gate.json，run matrix和registry已指向本次运行。
- [Experiment] 43项现有真实回归通过，233条已有依赖警告；Ruff、Python语法、bash -n及差异检查通过。CodeGraph确认require_current_capacity无调用者；原始验证目录outputs/metric_cue_pilot/L09_restart_no_runtime_gate_verification/。JAX部分donated buffers不可复用警告保留在日志。
- [Code] 维护说明位于开发树experiments/metric_cue_v1/RUNTIME_RESOURCE_POLICY.zh.md；开发树为vla_7_A6000:/home/lixuanyi/XuanyiLi/work/worktrees/rulervla-metric-cue。receipt和本交接为观察快照，后续进度须读取实际status.json及metrics.jsonl；当前尚无训练完成或SR结论。

## 2026-10-27 L09 revision4 正式训练已启动

- [Code] 用户明确确认修订协议先训练L09，S29位置场景有效性保留为评测条件。当前protocol revision4 SHA-256为c924ef9cdf7ae2de11760371baf525d845edc4275020ec1eae6d68286e29e0e1；revision3原字节归档于experiments/metric_cue_v1/revisions/r3/frozen_inputs.tar.gz。模型、数据、训练配方、checkpoint选择及评测定义经归档比对均未改变。
- [Experiment] L09_spatial_M2_seed42已在vla_7_6000 GPU0/4正式运行，PID216052，冻结运行代码e50e61c0fe9475f70a1b74ab62523325a1061983；已核实至少6次优化器更新，目标30000。第6次loss0.10732248、Cue梯度L2 0.02945974、梯度有限，该次更新23.64秒。此为启动观察，不表示训练完成或稳定吞吐。
- [Code] 配方为full fine-tuning、batch8、microbatch1、累计8、EMA .999、原warmup-cosine、CPU AdamW、双卡FSDP。源自已验证pi05_base，工程smoke更新不计入正式预算；计划10000/20000/30000保存完整checkpoint。
- [Experiment] 运行树/home/lixuanyi/XuanyiLi/metric_cue_runtime/RulerVLA，输出及原始metrics位于outputs/metric_cue_pilot/L09_spatial_M2_seed42；启动日志为同级L09_spatial_M2_seed42.launch.log，启动器准入记录为L09_spatial_M2_seed42_launch_r4。outputs仍指向共享NAS。对应receipt为experiments/metric_cue_v1/receipts/L09_spatial_M2_seed42_launch_r4.json，绑定run manifest SHA-256 d5633b61e4aa6d0fb7552c8000d27876820743e79a921bbbd127a8584a717ab8。
- [Experiment] 最终43项真实回归通过，两个真实评测CLI均被position_scene_validity_lock拒绝；日志在outputs/metric_cue_pilot/L09_launch_r4/。启动器对真实输入/receipt身份错误立即退出，只有容量不足才等待；flock跨exec保留。JAX日志存在部分donated buffers不可复用警告，训练已持续更新，原始警告保留。
- [Code] training_released=true，evaluation_released=false，position_scene_validity_lock=false。不改变活动运行树的源码和Git HEAD；后续文档记录提交保留在主开发树，恢复正式训练须使用e50e61c冻结运行时及同一manifest身份。

## 2026-10-27 MetricRuler 异节点 GPU 验证完成

- [Code] 运行时源码已提交并通过source_provenance，实际完整smoke提交为8a699dca1d3bee962b26e1ab0fe1bdb704b0a9c7。当前与旧工作流配置统一为batch8、microbatch1、累积8、EMA .999、warmup-cosine和双卡FSDP；CPU optimizer使用输入缓冲复用，恢复使用显式CPU sharding抽象模板，保存前写入期望签名。
- [Experiment] vla_7_6000（ZjuServer147）GPU1/3完成M2 seed42的GPU零Cue等价、两次完整π0.5更新、非零且有限的Cue梯度、完整checkpoint保存及CPU恢复。207个状态数组SHA-256全部一致；CPU峰值121.304 GiB，GPU每秒采样峰值21662/21660 MiB。后续准入每卡28 GiB、CPU160 GiB。
- [Code/Experiment] 独立运行树为/home/lixuanyi/XuanyiLi/metric_cue_runtime/RulerVLA，Python为同级environment/.venv/bin/python。Blackwell需要进程级LD_PRELOAD选择已锁定的NCCL2.28.9+cuda12.9，并将环境CUDA12.9.41 ptxas置于PATH前部。实际库版本、哈希及进程环境已记录；源共享环境保持原位。
- [Experiment] receipt为experiments/metric_cue_v1/receipts/L09_gpu_smoke_node147_20261027.json；原始目录outputs/metric_cue_pilot/node147_M2_batch8_8a699dc，checkpoint为checkpoints/2。运行树outputs指向/home/Dataset2/yqt/rulervla/metric_cue_runs；基座20个文件逐一验证，数据审计、真实输入与模拟器receipt已同步并核验哈希。旧失败日志与checkpoint保留。
- [Experiment] 主开发树41项CPU回归通过，日志outputs/metric_cue_pilot/resource_fix/restore_final.log；独立运行树已有14项CPU检查。GPU两步耗时135.47/80.35秒，仅有一个编译后的更新观察，稳定吞吐尚未建立。
- [Code] release更新GPU等价、更新、恢复和资源gates为通过，training_released/evaluation_released继续为false；正式30k训练仍等待position_scene_validity_lock，当前没有训练或采样监视进程。冻结protocol revision3及其SHA-256保持原字节。

## 2026-10-27 冒烟失败原因核查

- [Experiment] 最新batch8_split_attempt_20261027/preflight_M0停在资源准入：GPU2/5空闲8221/23549 MiB，要求每卡32768 MiB；CPU可用69.8140 GiB，要求100 GiB；未导入JAX或启动GPU。
- [Experiment] 本轮CPU执行source_provenance(Path.cwd())，复现 Runtime source is not committed: src/rulervla/geometry/camera.py。资源满足后仍有当前运行时代码身份阻塞；本轮未提交、改写源码或放宽检查。
- [Experiment] 历史ef3c277版本的gpu_fsdp_0_2_reuse_rgb_20261026已有gpu_equivalence.json，且源码哈希与manifest相符；两项allclose通过后才写入JSON。该原始证据覆盖旧handoff中“未取得等价检查数值”的概述，但不能用于当前revision3发布。
- [Unknown] 历史smoke没有首条成功更新记录，最终stderr/退出码未找到；旧running字段为遗留状态，当前无训练进程。现有证据不能确定历史中断是GPU/CPU OOM、NCCL异常还是外部终止。
- 详细证据与范围见开发树 experiments/metric_cue_v1/SMOKE_FAILURE_ANALYSIS.zh.md。本轮仅核查与CPU源码身份诊断，未启动GPU smoke、训练或rollout，全部门槛保持原状态。


## 2026-10-27 M2 seed42 正式启动检查

- [Experiment] 用户要求开始预训练；目标为当前protocol revision3的L09_spatial_M2_seed42：有效batch8、microbatch1、累计8、30k更新，full fine-tuning，关闭Echo。实际运行preflight及正式train入口，启动返回blocked，尚未发生参数更新或创建训练checkpoint。
- [Experiment] 本次GPU3/5空闲12259/23549 MiB，要求每卡32768 MiB；CPU可用66.93 GiB，要求100 GiB。两个manifest、统计量和基座metadata/receipt身份通过。preflight未导入JAX或启动GPU。
- [Code/Experiment] 正式入口因五项未通过条件抛出PermissionError：GPU零Cue等价、GPU梯度/优化器更新、完整状态恢复、位置场景有效性、资源准入。release的训练/评测开关保持关闭，没有降低门槛、停止其他作业或启动后台训练队列。
- 记录位于开发树 experiments/metric_cue_v1/receipts/L09_spatial_M2_seed42_start.json；原始日志位于 outputs/metric_cue_pilot/launch_M2_seed42_r3_20261027/。运行矩阵将该run标为blocked；其他run保持原状态。


## 2026-10-27 revision3：协议采用 batch8

- [Code] protocol.json 已升至 revision3：effective batch8、microbatch1、gradient accumulation8、30000 updates。run_matrix.json 与 release.json 使用相同 protocol revision 和 SHA-256；十二个 run ID、学习率、优化器、数据和评测条件保留。revision2 的协议、运行矩阵和 release 状态归档于 experiments/metric_cue_v1/revisions/r2/frozen_inputs.tar.gz。协议 SHA-256：a5188add4e9839c17de991a8062fb780f50c8ca3a7f967cafcd91ad9cc2b80e4。
- [Status] 训练与评测仍未发布；本次没有启动训练。GPU更新、资源条件、完整状态恢复和SR仍未验证。

## 2026-10-27 batch8 评估与旧配置修复

- [Code] 修改仅位于 MetricRuler 开发树。L09四组Cue和旧工作流入口对齐正XYZ、0～15 cm、5 cm等价横刻度及当前字段；协议更新为revision2，研究计划、协议、运行矩阵与release身份同步。旧协议与五份配置原字节保存在 experiments/metric_cue_v1/revisions/r1/frozen_inputs.tar.gz，SHA-256为4b85026ac19c6c2ab63a1327a538336b893288ef9d6f5fd96a2c6de01ad64413。
- [Experiment] 新增真实配置回归先复现10项旧字段/哈希失败，修复后11项通过；连同真实标尺与Echo回归共37 passed、209条当前JAX/Flax弃用警告。实际四组输入入口通过，receipt为 outputs/metric_cue_pilot/config_alignment_r2/real_arm_input.json。四组CLI preflight都能读取配置，随后因GPU2/5每卡32768 MiB及CPU100 GiB门槛而停止；四组preflight均未导入JAX或启动GPU。计算图与回归仅使用CPU，本轮没有正式训练。
- [Inference] 有效batch8按microbatch1、累计8解释；八条真实连续动作及完整π0.5梯度累积静态形状通过。30k更新抽取240000个起始帧窗口，等效4.5309轮；batch32为960000个、18.1235轮。参数/梯度/CPU AdamW/EMA占用基本保持，不能用batch降低解除当前资源限制；吞吐量、峰值显存、数值稳定性与SR仍未知。
- [Code] 该记录后由revision3更新：当前共同配方采用effective batch8、microbatch1、累计8、30k。详细依据见 experiments/metric_cue_v1/BATCH8_30K_ASSESSMENT.zh.md 与 REVISION3_BATCH8.zh.md。旧产物、checkpoint及失败结果保持原位，训练与评测发布标志继续关闭。Ruff、差异检查通过，CodeGraph已同步。


## 2026-10-27 CodeGraph 远程索引

- 用户已授权在远程主仓库和 MetricRuler 开发树建立索引。CodeGraph 0.9.4 已隔离安装到 /home/lixuanyi/.local/opt/codegraph/0.9.4，远程命令入口为 /home/lixuanyi/.local/bin/codegraph，自带运行时；没有修改系统 Node 或本地 MCP 配置。
- 主仓库 /home/lixuanyi/XuanyiLi/code/RulerVLA 与开发树 /home/lixuanyi/XuanyiLi/work/worktrees/rulervla-metric-cue 分别完成 init -i。status 记录主仓库1997个节点、3881条边，开发树4138个节点、8146条边；两者 pendingChanges 均为零。
- 开发树首轮 errors.log 的4个 ENOENT 均对应原有 Git 删除项：src/rulervla/training/metric_grounding.py、tests/test_metric_cue.py、tests/test_metric_grounding_labels.py、tests/test_metric_prompt.py。文件保持删除状态；当前 MetricRulerConfig 与主仓库 project_metric_ruler 的实际符号查询均返回源码位置。
- 两个工作区的 .gitignore 均新增 /.codegraph/。安装缓存和日志在主仓库 outputs/codegraph_setup/20261027/；开发树初始化日志在其 outputs/codegraph_setup/20261027/。中间文件均位于被忽略的 outputs 内，没有使用 /tmp。
- 后续通过 SSH 使用 /home/lixuanyi/.local/bin/codegraph，并切换到目标工作区执行 context/query/sync/status；本地会话中的 CodeGraph MCP 仍指向本地目录，不能自动读取远程索引。本轮没有启动训练、GPU 作业、rollout 或历史队列。


## 2026-10-27 MetricRuler 训练可行性核对

- [Code] 当前开发树 HEAD 为 c65323387dae06ae6c14036310643948649ca522，另有未提交修改。当前入口 configs/metric_ruler.yaml、configs/metric_workflow.yaml 使用 EE/TCP 原点、固定 world/action 正 XYZ、0～15 cm、每 5 cm 等价横刻度、无标尺箭头；双视角 RGB 加独立 Cue，经六通道 SigLIP 输入，Echo 默认关闭，目标为原 Flow Matching 动作损失。
- [Code/Inference] 可从官方 pi05_base 继续训练，使 π0.5 学习使用当前 Cue；MetricRuler 是确定性几何渲染模块。现有正式配方为 full fine-tuning，当前工作流的 batch1/2 updates 是工程检查配置。L09 仍引用旧 ±30 cm 字段，实际执行 preflight --arm M2 在读取冻结配置时因 axis_extent_m 等旧字段被拒绝；正式训练需先对齐版本化协议及 gates。
- [Experiment] 本日重新运行无 JAX/GPU 初始化的 preflight，数据两个 manifest、统计量及基座来源身份检查通过。GPU2/5 空闲分别为15759/23549 MiB，低于每卡32768 MiB；CPU可用63.35 GiB，低于100 GiB。launch_allowed=false。原始记录位于开发树 outputs/metric_cue_pilot/pretraining_feasibility_20261027/preflight/。
- [Experiment] release.json 当前 training_released=false、evaluation_released=false；完整模型 GPU 零Cue等价、梯度及优化器更新、完整训练状态恢复、位置场景有效性与资源条件仍未通过。120帧几何和输入报告为已存证据，本轮未重新执行该审计；尚无当前标尺的正式训练或 rollout SR。
- 本轮读取当前源码、配置、历史证据并完成资源及冻结配置检查；未修改代码、协议、发布条件或 registry，未启动训练、GPU smoke、policy inference 或 rollout。主仓库及开发树交接同步记录本次核查。


> 更新日期：2026-10-20 · 工作流首次部署交接

## 一句话现状
RulerVLA 已有实验总览、协议、run matrix 和 36 项历史实验索引；本次部署 research-workflow Skill 并接入项目文档，尚未核查当前 GPU 进程或实验是否运行。

## 已完成
- ✅ 确认主方法定义与证据边界，见 README.md、docs/EXPERIMENT_OVERVIEW.zh.md 和逐实验协议。
- ✅ Skill 已部署于 .agents/skills/research-workflow/，上游提交 14c882d。
- ✅ 已创建 AGENTS.md、handoff.md、brainstorm.md、dataset.md、experiment/results.md 和 experiment/evaluation.md。
- ✅ 原有 .gitignore 保留并追加数据/论文产物忽略规则；现有 experiments/ 未迁移或改名。

## 状态时间边界
- README 与实验总览的状态摘要整理于 2026-10-11，不可直接当作 10-20 实时状态。
- 部署时分支 main 比 origin/main 超前 38 个提交，且有大量已修改和未跟踪文件；不得 reset、清理或覆盖。
- 本次没有检查训练进程、checkpoint 完整性、NAS 可用性，也没有运行训练、推理或评测。
- README 说明仓库不含策略权重或演示数据；实际数据路径需从当前 protocol/manifest 核实。

## 下一步
刷新 git status；读实验总览和 registry；对目标 S/L 编号核对最新版 protocol、manifest、receipt、checkpoint/hash 和结果；开跑前核实进程、资源与所有 gates。若状态与 10-11 总览不同，先更新证据和总览再继续。

## 踩过的坑
1. 总览不是实时监控；追到 protocol、receipt 和输出。
2. LoRA/full fine-tuning 是训练配方，不是方法。
3. oracle-anchor 几何、输入正确性和 policy rollout SR 是不同证据。
4. Skill 模板用 experiment/，仓库原实验在 experiments/；分开维护。

## 给下一个窗口的第一句话
> 先读 AGENTS.md、handoff.md、docs/EXPERIMENT_OVERVIEW.zh.md 和 experiments/experiment_registry.json。核对未提交实验材料和运行状态，按目标 S/L 编号读取冻结协议；保留已有 outputs、checkpoint、manifest 和失败记录，launch gates 未通过时不要启动训练或 rollout。


## 2026-10-20 实验可读性整理
- 新增 docs/EXPERIMENT_GUIDE.zh.md，按研究问题说明旧 S/L 实验和新增 Adaptive Ruler/π0/A 系列目录，补上缩写解释与友好显示名。
- 显示名仅作阅读别名；未重命名实验 ID、目录、run ID、checkpoint、日志或原始结果。
- 状态边界：旧 S/L 登记整理于 2026-10-11；π0 状态 JSON 最后修改于 2026-10-14，10-17 资源快照为等待空闲 GPU。这些不是实时训练监控；本次未核验当前进程和完整 checkpoint receipt。
- 本次只整理文档，没有启动/停止训练或评测，也没有移动/覆盖实验产物。

## 2026-10-20 文件位置规范
- 已盘点远端根目录和主要子树：包源码在 src/rulervla；CLI/调度器在 scripts；配置在 configs；实验记录在 experiments；过程审计在 reports；忽略的生成物在 outputs；本地数据占位 DataSet 当前为空。
- 新增 docs/PROJECT_STRUCTURE.zh.md 和 experiments/README.zh.md，说明规范职责、实验记录模板和当前目录分支树；README.md 与 AGENTS.md 增加入口/放置约定。
- 保留 reports/ 中日期化脚本快照、scripts/ 中历史队列脚本和 outputs/ 中 UUID 测试产物；可能被绝对路径、systemd、run manifest、hash 或恢复流程引用，未做批量迁移。
- 本轮仅更新文档和放置规范；未移动源代码或实验产物，未运行测试、训练或评测。

## 2026-10-23 公制 Cue 六通道试点（开发 worktree）

- 唯一修改工作树：`/home/lixuanyi/XuanyiLi/.worktrees/rulervla-metric-cue`，分支 `research/metric-cue-pilot`，HEAD `c6ddada0`；原 `main` 脏工作区与备用节点未改。
- 试点实现：双视角 EE 世界轴公制标尺、独立三通道 Cue、四槽六通道输入、统一尺度 Prompt、默认关闭的实测 Motion Echo、读尺标签，以及受控的 JAX patch embedding 三→六通道初始化。详细证据和限制见 `docs/metric_cue/STATUS.md`。
- 相关回归 34 passed；JAX 局部契约／权重迁移／卷积等价与梯度 4 passed；一帧真实数据 CPU 输入 smoke 通过，Prompt 95/200 token。冻结 runtime 的 123 个哈希仍全部匹配；`git diff --check` 通过。
- 数据挂载在主节点为 `Host is down`；8 张 A6000 当时占用较高。未运行长训练、GPU rollout、正式 benchmark 或 push。
- 下一个工作单元先验证真实 checkpoint 的完整 π0.5 前后向、动作零 Cue 等价、新通道梯度和保存恢复；再统一正式训练与推理入口。随后实现 P5 训练 head/loss，才能准备 M0–M3 受控命令。不要把当前 CPU 输入 smoke 当作策略已可训练。
- 远程环境无 `codegraph` CLI；本次按用户许可在本地源码镜像建索引。历史 `runtime_v3` 和所有 checkpoint／结果／队列保持不动。


## 2026-10-23 G1 真实 checkpoint 端到端验收

- [Experiment] G1 已通过，详见 docs/metric_cue/G1.md。官方 pi05_base 从公开存储缓存到本地，20 个对象 MD5 核对并记录 SHA-256；未依赖故障 Dataset2。
- [Experiment] 完整 π0.5 的三／六通道视觉 prefix 和动作输出零 Cue 最大差均为 0；Action FM Loss 的 Cue kernel 梯度 L2=0.0560932，所有梯度有限。full-FT 配置包含全部图像参数及新 kernel，Adam mu/nu 覆盖新通道；仅做 1 次诊断更新。
- [Experiment] 完整训练状态的 155 个张量叶子恢复后 shape/dtype/SHA-256 一致；同输入动作恢复前后差为 0。恢复后的非零 Cue 影响动作；独立推理参数恢复检查确认 LIBERO 前七维均有非零影响。
- [Code/Experiment] 首次普通六通道卷积未通过完整零 Cue 等价；定位为 FP32 归约顺序跨 BF16 舍入边界的放大。新方法 SigLIP 保留单一六通道 kernel，将 RGB/Cue 分别卷积后求和以保留 RGB 路径数值。失败记录保留。
- [Experiment] 回归 43 passed；冻结 runtime 123/123 哈希匹配；git diff --check 通过。验收源码 manifest 与当前实现逐项相符，版本记录在 outputs/metric_cue_pilot/g1_environment.json。
- [Code] 新诊断入口 scripts/metric_cue_model_gate.py；新 checkpoint 为 outputs/metric_cue_pilot/g1_real_checkpoint_stable_stem/checkpoints/1。不要把旧三通道优化器状态当作六通道恢复状态。
- [Inference] 此证据确认模型具备通过 Action Loss 学习 Cue 的条件；尚无厘米读尺能力、泛化或 rollout 成功率证据。fixture 当前动作被重复到 horizon=10，仅用于工程验收。GPU 路径、LoRA 配方、G2 正式入口、G3 与 G4 尚未验收，未启动正式训练。


## 2026-10-24 G2 统一工作流交付：工程通过，GPU 验收阻塞

- [Code] G1 来源冻结提交 4ccdbabe；统一训练／推理实现 27281a2d；依赖整理后的源码提交 cc7a1855。完整报告见 docs/metric_cue/G2.md，入口说明见 docs/metric_cue/WORKFLOW.md。本轮没有 push。
- [Experiment] 原始 manifest 的 episode 1261/1262、样本 20/108/109/110 核验真实连续 actions 与 repeat-last；不跨 episode。四个 NPZ 的 SHA-256 与原 manifest 相同，缓存仅用于审计，不代表 Dataset2 已修复。
- [Experiment] Train/Inference 的 RGB、Cue、state、tokens、mask、dtype、shape 最大差均为0。新 CPU 推理入口显式桥接 G1 六通道 checkpoint，完成一次 float32[10,7] 预测；没有 rollout 或实际环境动作。
- [Experiment] 真实 B=2 batch 的76参数小模型 fixture 通过生产训练编排：Cue 梯度非零、更新、9叶子完整状态恢复、恢复后下一次更新逐值相同。该证据不是完整 pi05 GPU 训练通过。轻量回归46 passed；冻结 runtime 123/123 哈希相符，G1 原始证据不改。
- [Code] 依赖锁270项，OpenPI及client均由本仓库源码提供；共享环境未使用的RLinf排除。wheel由提交tree构建并隔离导入，119个Python文件与提交源码相符；全新依赖环境安装尚未实测。
- [Experiment] GPU smoke未启动：Dataset2的两个manifest报Host is down，GPU7空闲21351MiB低于45000MiB门槛。实际preflight jax_imported=false。GPU数值误差未测，预先容差atol=rtol=0.02，不宣称GPU位级等价。
- [Inference] G2整体尚未完成GPU验收；先修复数据访问并提供足够显存，再显式执行新入口的1–3更新smoke。G3只保留objective/embed_prefix/身份接口，未添加新头、loss或启动正式M0–M3、30k训练、批量rollout、历史队列。

## 2026-10-25 当前 Plan 与实现审计

- 当前实际开发树为 `/home/lixuanyi/XuanyiLi/work/worktrees/rulervla-metric-cue`，HEAD `a09e04a052f4bf5dd581a2569bfea4edaab63db2`。本轮按本地2026-10-24 `reaserch.md` 审计，完整报告见 `docs/metric_cue/PLAN_AUDIT_20261025.zh.md`，原始Plan和核验产物在 `outputs/metric_cue_pilot/plan_audit_20261025/`。
- [Code] 核心六通道公制Cue架构基本符合Plan；M0关闭Ruler被Prompt Builder拒绝，M1尚无独立Direction实现，M3被统一Pipeline拒绝且未贯通history。正方向箭头尚未绘制。M0与M3的拒绝路径已实际复现。
- [Experiment] 使用真实缓存两个episode的4个样本重新运行data audit：真实连续actions、repeat-last、Train/Inference全部输入一致；反归一化误差最大1.192092896e-7。环境及4个Transformers补丁检查通过；冻结runtime按snapshot manifest核验123/123匹配。
- [Experiment] 当前preflight读取到Dataset2两份manifest且哈希匹配；尚未逐文件核验整个数据集。当前阻塞是GPU7空闲3857MiB低于45000MiB、CPU可用67.89GiB低于100GiB。旧Host is down和旧环境路径为历史状态；Python现为 `/home/lixuanyi/XuanyiLi/code/openpi/.venv/bin/python`。无GPU初始化或作业启动。
- [Code] 当前Plan仅使用原FM loss，旧G2记录里的G3辅助head/loss不作为当前Plan必需项。正式训练预算、M0–M3配置、位置split和六通道rollout协议仍待冻结；CLI当前只有末尾保存，正式长训练还需确定周期保存与恢复流程。
- AutoResearch在主仓库／开发树状态均为not_initialized；本轮调用其状态核验并采用ARS审计，未开始自动优化。报告记录H1–H3的主要对照和固定像素控制、Prompt公平性、配对场景及分层重复单位，尚未成为正式协议。
- 本轮未修改模型／训练源码、registry或冻结协议，未启动GPU smoke、训练、policy inference、rollout或历史队列；保留已有未提交内容和全部历史证据。

## 2026-10-25 research.md 符合性复核

- 本地 `reaserch.md` 的 SHA-256 仍为 `9239b62fe26f9702731dfaecbf16c82f3915fedfbbf34f6193d91f3be1051a9b`；开发树 HEAD 仍为 `a09e04a052f4bf5dd581a2569bfea4edaab63db2`。复核了当前 Metric Ruler、Prompt、Motion Echo、统一输入及训练入口。
- [Code] M0 关闭标尺时被 Prompt Builder 拒绝；M1 缺少独立 Direction Cue；M3 在统一 Pipeline 中被拒绝，训练和推理入口尚未传递历史状态；正方向箭头尚未绘制。正式 M0–M3 协议和位置泛化评测仍缺少可执行记录。
- [Experiment] 使用真实缓存样本 20、108、109、110 再次运行 CPU 数据审计，产物位于 `outputs/metric_cue_pilot/recheck_20261025/`。4 个样本的训练与推理输入逐值一致，动作反归一化最大误差为 `1.1920928955078125e-7`；没有启动 GPU 作业、训练或 rollout。

## 2026-10-25 六通道实验准备

- [Code] 开发工作树 `/home/lixuanyi/XuanyiLi/work/worktrees/rulervla-metric-cue` 增加 M0 空白、M1 固定像素方向、M2 双向公制刻度和 M3 公制刻度加五步实测 Motion Echo。四组共用六通道模型、归一化和原 Flow Matching 动作目标；M3 从同一 episode 的真实历史读取训练输入，在线入口要求逐环境控制步观测及 episode 重置。
- [Code] `experiments/metric_cue_v1/` 保存 L09 四组、三训练种子的协议、运行矩阵和关闭状态的发布门槛；S29 黑碗位置泛化继续等待场景有效性检查。正式训练入口固定有效批量 32、微批量 1、梯度累积 32、30000 次更新、EMA 0.999、10000／20000／30000 次完整状态保存与恢复。未通过发布门槛时入口立即拒绝启动。
- [Experiment] 使用来源哈希已核对的真实缓存样本验证 M0–M3 训练／推理输入一致、M3 五步位移、真实 32 条动作训练批次、完整基座模型 CPU 载入及完整模型梯度累积的静态形状；使用真实连续观测和完整基座模型取得有限的在线十步动作块。逐一读取全部 432 个训练 episode 和 52970 帧，核对 RGB-D／几何哈希及内容，432 条 receipt 的 SHA-256 为 `73dbf88aa9abb7b412f8cbe5468e88dc928708466f87cbd510b2acfe693a1849`。未执行完整模型 GPU 更新。
- [Experiment] 2026-10-25 新 preflight 中数据 manifest、统计量、基座来源检查通过；GPU 7 空闲显存低于 45000 MiB，训练和 GPU smoke 均未启动。在线评测服务与 Spatial 配对场景客户端已加入，发布门槛关闭时均会拒绝启动。Python 3.11 模型环境当前没有 LIBERO 与 robosuite；协议指定的 LIBERO 源码路径对当前 SSH 用户不可读取，正式模拟器入口和 S29 场景锁定仍待验证。
- [Inference] 当前证据确认输入、协议和启动保护可检查；完整模型 GPU 数值检查、真实更新及 checkpoint 恢复、在线 simulator rollout、位置场景有效性和 SR 仍未取得证据。`release.json` 保持关闭。
- [Code] 本单元实现提交为 `6efe098`、`42a4fd5`、`fc871fb`、`23a4745`、`5961485`；旧数据审计入口保留，完整数据核验使用新入口 `scripts/metric_cue_full_data_audit.py`。`experiments/experiment_registry.json` 已在开发树补入 L09／S29，主工作树内容保持原位。在线策略按 episode 身份重置采样随机状态，配对场景使用相同的噪声起点。
- [Experiment] 最后一次 GPU 查询显示 GPU 7 空闲 3862 MiB，GPU 4 空闲 38153 MiB，均低于当前 45000 MiB 准入门槛。完整数据核验发布条件已通过，其余 GPU、模拟器和场景发布条件继续关闭。

## 2026-10-26 六通道模型双 GPU 分片

- [Code] 开发工作树分支 `research/metric-cue-pilot` 的提交 `fcd7149`、`f3f0b38`、`45e8326`、`efb922d`、`ef3c277`、`e62d2b5`、`14f7cd4` 实现可选择 GPU 的 FSDP 参数／梯度分片，CPU 上的 AdamW／EMA 状态，以及三通道／六通道等价检查。三通道对照直接使用六通道参数中的 RGB 权重。正式协议固定两张 GPU，允许每次运行选择具体设备并在 manifest 中记录。
- [Code] 资源检查对所选设备逐卡要求空闲 32768 MiB；模型载入之后和等价检查之后再次检查。JAX 编译缓存写在被 Git 忽略的 `outputs/metric_cue_pilot/jax_compilation_cache/`。单设备原有 45000 MiB 门槛保留。门槛属于启动保护，完整更新通过后须结合实际峰值显存核验。
- [Experiment] 两个真实 CPU JAX 设备、完整基座模型参数及真实 32 条 Spatial 样本通过分片结构、三通道参数构造和梯度累积形状检查；Ruff 与 `git diff --check` 通过。GPU 0／2 的真实完整模型等价检查进入 XLA 编译，两卡中本进程各占约 7 GiB；其他作业随后占用了两卡，进程在执行参数更新之前终止。没有取得等价检查数值、完整 GPU 梯度更新、checkpoint 恢复或峰值显存。最近一次 preflight 对 GPU 0／2 返回 `blocked`，`jax_imported=false`、`gpu_job_started=false`。
- [Experiment] `experiments/metric_cue_v1/release.json` 中 GPU 与训练发布条件继续关闭。两张设备均满足门槛且能持续供给时，执行真实完整模型 GPU smoke、参数更新、保存恢复及显存峰值检查；保持现有其他 GPU 任务和未提交文件原样。

## 2026-10-26 LIBERO 场景与模拟器条件

- [Code] 在独立依赖工作树 `/home/lixuanyi/XuanyiLi/work/deps/libero-8f1084` 检出 LIBERO 提交 `8f1084e3132a39270c3a13ebe37270a43ece2a01`。评测专用 Python 3.11 环境位于被忽略的 `outputs/metric_cue_pilot/eval_venv/`，依赖版本写入 `requirements/metric-cue-eval-linux.lock`；`pip check` 和 benchmark 导入通过。
- [Experiment] `experiments/metric_cue_v1/paired_scene_manifest.json` 冻结十个 Spatial 任务、每任务五十个官方初始状态、BDDL、场景种子及来源哈希；重复来源核验通过，SHA-256 为 `8b3788f3402f4ef0f41d4ba26fe629e6455b3d07afedfff24229097fa8fe8fe9`。官方 `.pruned_init` 文件由 `torch.load` 读取。
- [Experiment] 真实 OSMesa 模拟器在 Spatial task index 2、scene index 0 完成初始状态设置、双视角 RGB-D、相机参数和 state8 采集，并执行一个控制步。运行记录 `outputs/metric_cue_pilot/simulator_gate_task2_scene0_20261026/receipt.json` 的 SHA-256 为 `1cc6ba376368747b3afea973af119782390097df30213be2372fb7ad4dce5a52`。该记录证明此场景的运行接口，尚未取得完整 rollout 或策略成功率。
- [Code] 正式训练和评测入口增加配对场景记录与模拟器运行记录的哈希和内容核验；`release.json` 中对应两个条件通过。完整模型 GPU 数值、参数更新、完整状态恢复、S29 场景有效性及资源条件仍未通过；正式训练与评测保持关闭。当前资源检查没有满足两张 GPU 各空闲 32768 MiB 且 CPU 可用 100 GiB 的组合。
- [Experiment] 配对场景来源重复核验、模拟器记录哈希核验、评测入口关闭条件和依赖完整性检查通过。固定 CPU 后运行训练资源、状态恢复和输入工作流的 12 项回归全部通过；测试临时文件写入被忽略的 `outputs/metric_cue_pilot/pytest_temp/`。
- [Experiment] 读取主工作树 `outputs/a_series_four_suite_v1/S28_workspace_v1/` 的现有候选记录，`plan.json` 与 `candidate_initial_states.npz` 的 SHA-256 分别为 `a2477a02d481c4e57d0d25d7fc1ace673910f3ac58a2fdfc4a3022c5f7cd90b5` 和 `3f96a1573d041825bcafef6bbddffeec2210846c2ff1f1dcbcb9a60ec612a8d0`，训练位置文件的哈希与计划所记一致。99 个候选位置中 27 个通过既有几何筛选；遮挡、抓取过程碰撞、放置稳定性及最终分组尚未完成实际核验，S29 场景条件继续关闭。

## 2026-10-26 标尺与 Motion Echo 视觉示例

- [Experiment] 使用真实 LIBERO Spatial 黑碗样本 `outputs/metric_cue_pilot/real_sample_1261.npz`，SHA-256 为 `b588df1e3024159b81fecb6fbf8306dd622e00e51358bfc9bfca8b3c51880c56`；当前为 episode 1261 控制帧 20，实测历史为控制帧 15 至 20，EE 位移 `[0.0556159038, 0.0008702865, -0.0393905640]` 米。
- [Code/Experiment] `scripts/metric_cue_visual_demo.py` 使用现有 `MetricViewAdapter` 分别生成 M2 标尺、空白标尺加真实 Echo、M3 标尺加 Echo。三种情况的双视角 RGB 逐值相同，Echo 在两个视角均已渲染。图像和来源记录位于被忽略的 `outputs/metric_cue_pilot/visual_demo_20261026/rendered/`，包含三张独立 PNG、对照图和 `manifest.json`。上排 RGB 叠加仅用于观察；下排显示实际三通道 Cue。Echo 单独示例是可视化组合，当前 L09 运行矩阵没有对应训练组。

## 2026-10-26 ±15 cm Metric Ruler 渲染版本

- [Code] 工作树仍为 `/home/lixuanyi/XuanyiLi/work/worktrees/rulervla-metric-cue`。新增 `metric_ruler_15cm_v2` 及 `configs/rulervla_metric_cue_15cm_v2.yaml`、`configs/rulervla_metric_workflow_15cm_v2.yaml`。EE 为原点，world XYZ，每轴 ±15 cm、每 5 cm 等价横刻度；独立通道，正实线/负虚线及正箭头。OpenCV 亚像素 AA 直接在 224×224 标定上绘制，RGB 路径和原模型/FM 目标保持原状。
- [Code] 旧 ±30 cm 配置继续使用原渲染和原合同，历史 Cue 两视角哈希及 pipeline fingerprint 均与已有记录一致。新配置 fingerprint 为 `036530fc989926dd2ae0190077aa6a6738ef9e40d0e804fde19f0efff48e30e7`。新身份会拒绝无版本 G1 权重桥接，以及旧正式 arm 对新配置的静默替换。
- [Experiment] 120 个真实帧、240 个视角、4320 个刻度：最大物理误差 `1.1102230246251565e-16 m`，独立投影误差 `1.3642420526593924e-12 px`。4 个真实连续动作样本在训练/推理及 SigLIP 前逐值一致；六通道 float32、Cue [0,1]、朝向、mask 和真实 tokenizer 核验通过。边界测试使用真实标定独立检查 near 交点、viewport 端点及可见轴线像素。
- [Experiment] 本次新增真实数据测试 12 passed，相关回归 49 passed；全仓 192 passed、16 failed、96 warnings，16 个失败在修改前独立源码快照中逐项复现。详细名称、限制及数值表见 `reports/metric_ruler_15cm_v2/report.zh.md`，命令见同目录 `commands.md`；原始日志和最终图片位于 `outputs/metric_cue_pilot/ruler15_v2_validation/delivery_verified/` 及其上级。Ruff、git diff --check 和只读代码审查通过。
- [Inference] 当前 L09 protocol/run matrix 仍为旧 ±30 cm，release 条件未改变。新工作流已经过渲染和输入核验；正式训练需先冻结匹配的新协议并通过 gates。本次没有启动正式训练、GPU smoke、policy predict 或 rollout。Wrist 的视野/近面裁剪、光轴接近及投影轴夹角退化均保留真实位置并记录；不据此声明模型尺度能力或 SR。

## 2026-10-26 正 XYZ 轴标尺 v3

- [Code] 按最新要求，当前版本只显示从 EE 原点沿正 XYZ 轴的 0～15 cm 标尺，刻度为 +5/+10/+15 cm。新增独立身份 `metric_ruler_positive15cm_v3`，配置为 `configs/rulervla_metric_cue_positive15cm_v3.yaml` 和 `configs/rulervla_metric_workflow_positive15cm_v3.yaml`；v2 与历史配置、checkpoint 及记录保留原定义。
- [Code] 本地 `reaserch.md` 的公式、范围、刻度、Prompt、M1/M2/M3 描述已同步为当前正轴定义；远端副本为 `docs/metric_cue/RESEARCH_PLAN.md`。合同见 `docs/metric_cue/RULER_POSITIVE15CM_V3.md`。
- [Experiment] 120 个真实帧、240 视角、2160 正刻度通过核验；最大物理误差 `1.1102230246251565e-16 m`，独立投影误差 `1.3073986337985843e-12 px`。双视角没有负向轴线像素，四个真实样本的训练/推理及 SigLIP 前输入逐值一致。新 fingerprint 为 `de1ac6d2bb7a64bc64a8ab642da45b0cb657584c36d683e5693f14bc4c152dbb`。
- [Experiment] 新增真实测试 4 项通过，相关回归 35 passed；全仓 196 passed、16 个既有失败、96 warnings，失败名称与上一轮相同。Ruff、diff 检查与只读审查通过。详细记录在 `reports/metric_ruler_positive15cm_v3/report.zh.md`，图片和原始日志在 `outputs/metric_cue_pilot/positive15_v3_validation/`。
- [Inference] 本次没有启动训练或 rollout，旧正式协议和 gates 未修改。Wrist 正 Z 的刻度在当前示例中因实际视野与近平面不可见，保留真实裁剪及省略记录。

## 2026-10-26 无箭头标尺与代码整理

- [Code] 当前 Cue 仅保留正 XYZ 实线、+5/+10/+15 cm 等价横刻度及 EE 原点。参数类统一为 `MetricRulerConfig`，当前投影/渲染为 `project_ruler_cue`、`render_ruler_cue`；共用相机几何位于 `geometry/camera.py`。删除多套 Cue 分派、标尺箭头参数/函数、旧类名及未使用的读尺辅助实现。
- [Code] 所有当前 CLI 默认采用 `configs/metric_ruler.yaml`、`configs/metric_workflow.yaml`，审计入口为 `scripts/audit_metric_ruler.py`；版本后缀配置和重复测试合并清理。仍被 RGB/Adaptive 对照使用的实现与冻结协议引用配置保留。修改前源码完整归档于 `outputs/metric_cue_pilot/ruler_cleanup/before/source.tar.gz`。
- [Experiment] 120 个真实帧、240 视角、2160 个刻度检查通过，物理误差最大 `1.1102230246251565e-16 m`，独立投影误差最大 `1.3073986337985843e-12 px`。4 个真实连续动作样本的 Train/Inference 与 SigLIP 前输入一致；新输入 fingerprint 为 `4a0c879cec9dff924465417a64a8623c1676b4a3f357563e03a2aa73e52e3775`。
- [Experiment] 当前11项真实标尺测试、checkpoint保存恢复和RGB/Adaptive对照通过；最终全仓180 passed、16个既有失败、96 warnings，失败名单与此前相同。Ruff、diff检查和独立审查通过。原始日志、预览与输入在 `outputs/metric_cue_pilot/ruler_cleanup/final_render/` 及其上级；说明见 `reports/metric_ruler_cleanup/report.zh.md`，当前合同见 `docs/metric_cue/RULER.md`。
- [Code] 本地 `reaserch.md` 与远端 `docs/metric_cue/RESEARCH_PLAN.md` 同步改为当前名称和无箭头标尺。当前配置拒绝旧输入字段，旧 checkpoint 使用匹配的冻结源码；训练/rollout gates 未改变，本轮未启动正式训练或 rollout。

## 2026-10-26 Motion Echo 实现检查

- [Code] Echo 默认关闭；启用后使用同一 episode 内当前与五个环境控制步前的实测 EE 位置，在每个当前相机视角分别投影并绘制黄色净位移箭头。训练通过 `SpatialDataset.get_history` 读取历史，推理通过 `MetricInferenceHistory` 逐控制步维护历史；历史不足或位移小于 3 mm 时 Cue 留空。
- [Code] 当前实现存在四项待处理问题：Pillow 在最终分辨率直接绘图，未使用抗锯齿；任一端点位于近平面之后时丢弃整个线段，且视野裁剪后仍在裁剪端点画箭头；Echo 像素整组三通道覆盖 Ruler，可清除重叠位置的 Z 通道；Prompt 只依据历史有效性，未依据实际渲染或静止状态。本次仅检查实现，没有修改 Echo 源码或配置。
- [Experiment] 本次运行 `tests/test_motion_echo.py`，5 passed，日志为 `outputs/metric_cue_pilot/echo_review/tests.log`。另核对上一轮两个真实样本的 M0–M3 输入检查记录 `outputs/metric_cue_pilot/ruler_cleanup/real_arms_verified.log`；现有证据没有覆盖上述全部 Echo 可读性与裁剪问题，也不代表 policy 或 rollout 成功率。

## 2026-10-26 Motion Echo 渲染改进

- [Code] `render_motion_echo` 采用共用三维近平面／视野裁剪与 OpenCV 亚像素抗锯齿，保留实测五步 EE 净位移；头部仅锚定可见的真实当前端点，大小最多为可见长度 40%。`compose_cues` 逐通道取最大值，Z 逐值保持；Prompt 改由两个视角的实际渲染诊断生成，移除 `echo_valid`。无效投影诊断使用 JSON null。研究计划与 `docs/metric_cue/MOTION_ECHO.md` 已同步。
- [Code] Echo 默认关闭，关闭 fingerprint 仍为 `4a0c879cec9dff924465417a64a8623c1676b4a3f357563e03a2aa73e52e3775`；启用时保存 `motion_echo_rendering: antialiased_channel_max`，fingerprint 为 `775d450f78263bd1e0d44a1dd1edb9393afc08f7dea7ed9908d44f2c7e4c8413`，旧 Echo 身份在资产读取前被拒绝。六通道模型与 Flow Matching 目标不变。
- [Experiment] 120 个真实帧、240 视角：Agentview/Wrist 可见 Echo 为 107/109，历史不足各 10，不足 1 px 各 3/1；Wrist 有 45 个视野裁剪。独立投影误差最大 `1.1368683772161603e-12 px`，累计 12,376 个蓝通道重叠像素逐值保留。8 个真实样本 Train/Inference/SigLIP 前输入完全一致、RGB 不变、mask TTFF；启用预处理资产保存恢复通过。
- [Experiment] 新增 15 项回归在旧源码失败、新源码通过；Echo 与相关接口 35 项全通过。最终全仓 194 passed、16 个既有失败、96 warnings；与前轮 XML 按名称逐项比对失败集合相同。真实 M0–M3 入口、Ruff、diff 检查及独立只读审查通过。报告为 `reports/motion_echo/report.zh.md`；原始日志、配置和张量在 `outputs/metric_cue_pilot/echo_render/verified/` 及上级，最终对比图为 `outputs/metric_cue_pilot/echo_render/delivery/render_comparison.png`。修改前源码完整归档在同运行目录的 `before/source.tar.gz`。
- [Inference] 本轮没有训练或 rollout；共享 Cue 通道可能混色，不能分别恢复两个组件；边界压力用实测轨迹与受控标定参数变体单独验证。结果不构成模型尺度能力或 SR 结论，冻结协议和 gates 未修改。
