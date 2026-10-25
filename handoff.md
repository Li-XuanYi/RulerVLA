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
