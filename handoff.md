# Handoff

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


## 2026-10-24 Metric Cue 审计修复（G2 限制保留）

- [Code] 基于 main 的4bc1466在独立detached工作树修复；原有脏工作区、历史证据和checkpoint不改。
- [Experiment] 新增回归在旧实现上16 failed / 9 passed；修复后83 passed。修复空心负刻度被后续线段覆盖、Echo抹除Z通道、近裁剪面丢线和视口边界假箭头、静止/不可见Echo的Prompt矛盾，以及同位置旧帧历史被接受。
- [Code] Echo默认与G2启用限制保留；未接通新实验样式、位移数值文本、辅助训练头或正式训练。np.maximum仅避免覆盖，不代表Echo获得独立通道。
- [Code] 默认Cue像素发生修正，预处理版本升级metric_cue_observation_v2；旧v1 checkpoint须使用原提交运行时，新恢复入口拒绝静默版本替换。
- [Experiment] CPU轻量测试记录位于outputs/metric_cue_pilot/review_fixes/；没有GPU作业、大型G1重跑或rollout。真实样本输入一致性将在源码提交后记录。
