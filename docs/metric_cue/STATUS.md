# RulerVLA 公制 Cue 试点实施与验收报告

时间：2026-10-23。以下“通过”只指注明的测试范围，不代表策略收益或正式训练就绪。

## A. 工程审计与整理

### 工作树与来源

- **主节点**：通过 WSL 的 SSH 直连 `vla_7_A6000`（主机 `amax`，用户 `lixuanyi`）；仓库 `/home/lixuanyi/XuanyiLi/RulerVLA`。备用入口 `ssh zju` 指向 `ZjuServer145`（用户 `yqt`），其历史副本仅用于只读核对和提取一帧样本。
- 两端核对时 HEAD 均为 `c6ddada0fbbc9b98fb7523c62f3c3fca0fa1945f`。主仓库 `main` 领先 `origin/main` 38 个提交，并含大量未提交修改；原始工作区未重置、未覆盖。
- **唯一修改位置**：同一正式仓库的 worktree `/home/lixuanyi/XuanyiLi/.worktrees/rulervla-metric-cue`，分支 `research/metric-cue-pilot`。建树后复制了主工作区现有的相关脏源码，避免把它们当成可丢弃文件。该 worktree 目前仍有未提交改动；本次没有 Git push。
- **历史运行时**：`/home/lixuanyi/XuanyiLi/RulerVLA/outputs/main_comparison/runtime_v3`。按其 `RULERVLA_RUNTIME_MANIFEST.json` 逐一重算 123 个源码哈希，0 不匹配；四图像槽保持原样。没有编辑该冻结目录、checkpoint、历史结果 JSON、协议或队列。
- **新方法源码**：worktree 内 `src/openpi`，从冻结运行时复制，供新方法明确导入。实测 CPU 输入脚本加载路径为 worktree 的 `src/openpi/models/model.py`。外部 `/home/lixuanyi/XuanyiLi/openpi` 的当前 HEAD 是 `f781c5575811c365000e520b324e7b66be9e405f`；它与冻结源码有差异，不能将此 HEAD 冒充冻结 runtime 的确切源码提交。新方法还没有统一的训练／推理启动入口。
- 主节点 `/home/Dataset2` 在检查时返回 `Host is down`。备用节点同名挂载可读，真实示例来自该处。主节点 8 张 A6000 当时已有高占用，未启动 GPU 作业、长训练或 rollout。
- 远程没有可执行的 `codegraph` 命令；经用户允许，在本地同步的源码镜像 `work/source_mirror` 建立了 CodeGraph 索引（146 文件、1779 节点）。索引是开发辅助，不是远程运行依赖。

### 执行路径与文件

历史入口 `scripts/main_spatial_train.py` 经 `SpatialDataset/SuiteDataset → MainCueAdapter/OpenPIAdapter → MainComparisonInputs → runtime_v3`；其 RulerVLA-main 把 Ruler 叠在 agentview RGB，并把 ActionCoord 放到辅助槽。新试点的已验证路径为 `scripts/metric_cue_cpu_smoke.py → MetricViewAdapter → SixChannelInputs → StrictPromptTokenizer → src/openpi` 的输入契约；尚未接到动作前向与训练入口。

本次主要新增／修改：

| 职责 | worktree 中的文件 |
| --- | --- |
| 几何、双相机适配与六通道 | `src/rulervla/cues/metric_ruler.py`、`src/rulervla/adapters/libero.py`、`src/rulervla/adapters/metric_cue.py` |
| 配置、Prompt、Echo | `configs/rulervla_metric_cue.yaml`、`src/rulervla/config.py`、`src/rulervla/cues/metric_prompt.py`、`src/rulervla/cues/motion_echo.py` |
| 历史索引、读尺标签 | `src/rulervla/training/main_data.py`、`src/rulervla/training/metric_grounding.py` |
| JAX 图像契约与受控权重迁移 | `src/openpi/models/pi0_config.py`、`src/openpi/training/weight_loaders.py` |
| CPU 入口及测试 | `scripts/metric_cue_cpu_smoke.py`、7 个 `tests/test_metric_*.py`／`test_six_channel_*.py` 相关新测试文件 |

保留了历史 ActionCoord、AdaptiveRuler、AimBot、AxisGuide 实验路径。没有删除或合并历史调用链；新入口与旧入口目前并存。

## B. 一帧真实 Observation 的数据流

示例为 LIBERO-Spatial 黑碗任务、episode 1261、控制帧 20。源 agentview/wrist RGB 各 `[256,256,3] uint8`，深度各 `[256,256]` 米，EE 世界位置 `[3]` 米，两个视角各有同步内外参。经 canonical raster 与 resize，保留的 RGB 是 `[224,224,3] uint8`。XYZ Ruler 按同一个 EE 和世界轴物理刻度投影两次，每视角产生独立 `[224,224,3] float32` Cue，范围 `[0,1]`；Echo 默认关闭，启用时在同一三通道 Cue 中按固定优先级合成。RGB 未被 Cue 覆盖。

适配器先把 RGB 按模型约定映射到 `[-1,1]`，再拼接 Cue；实际 JAX 输入的四个槽均为 `[1,224,224,6] float32`。有效 `base_0_rgb` 与 `left_wrist_0_rgb` 的 mask 为 true；未使用的 `right_wrist_0_rgb` 与 `aux_wrist_cue_rgb` 是全零六通道，mask 为 false。实际 RGB 值域分别为 `[-0.9373,0.9922]`、`[-0.9922,0.7490]`；Cue 均为 `[0,1]`。状态张量 `[1,32]`。同一配置生成的任务＋尺度 Prompt 经真实 PaliGemma tokenizer 为 95/200 token，未截断。

该真实帧通过了 `Observation.from_dict` 和 `Pi0Config.inputs_spec`。**本次尚未在完整 π0.5 上完成图像编码、动作预测或训练反向**，因此数据流证据止于模型输入。

## C. 几何和视觉结果

默认配置：EE/TCP 原点，world/action XYZ，三个轴均 ±0.30 m，基础分度 0.05 m，隔 2 格为主刻度；正向实线实心、负向虚线空心。world XYZ 未随 EE 姿态旋转。两个视角分别投影同一组米制世界点，处理近平面、视野裁剪和像素变换。独立几何测试覆盖刻度位置、EE 平移／旋转、相机视角、裁剪、源到策略像素及 RGB 不变。

真实帧的 agentview：X/Y/Z 各 12 个有效刻度，最小相邻间距约 7.0/18.2/10.0 px，均无视野外刻度。wrist：X/Y/Z 有效刻度分别为 2/2/6；X/Y 的视野外比例均为 83.3%，Z 为 50%，并有 5 个刻度落在近平面之后。这说明 5 cm 在该帧可用作试点，但 wrist 读尺监督需可见性筛选。2 cm 候选在 agent X 的最小相邻间距约 2.75 px、wrist Z 约 0.85 px；3 cm 的 wrist Z 约 1.31 px。不能把 2 cm 默认当作可读训练尺度。

`metric_cue_real_frame.png` 给出两视角的原始 RGB、X/Y/Z 独立通道、三通道预览以及开启 Echo 的对比。真实历史从同一 episode 的控制帧 15 到 20，测得位移约 `[+0.0556,+0.0009,-0.0394]` m；在当前两相机中投影，wrist 过去点超出画面并被裁剪。默认 Echo 关闭。

## D. 模型、Prompt 与专项标签验收

- JAX 配置的 `inputs_spec` 支持六通道并保留四槽；`SixChannelCheckpointWeightLoader` 只允许指定的 `PaliGemma/img/embedding/kernel` 输入轴由 3 扩到 6，初始化 `[W_RGB,0]`，其余参数集合与形状不符会报错。原历史 loader 未修改。
- 4 项 JAX CPU 小测试通过：六通道 spec、受控迁移、非指定形状失配拒绝、**卷积层**零 Cue 与旧三通道卷积的数值等价及新增通道非零梯度。这是局部权重证据，**不是**完整 SigLIP 表征或动作输出的零 Cue 等价性。
- Prompt 从同一 Ruler 配置构造；5 cm、3 cm 等分度与文字一致，实际样本 95/200 token。Echo 文本只说明已测位移，不给未来动作或精确数值。
- `EpisodeHistoryBuffer` 与训练集历史索引只接受同 episode 的过去控制步，重置／不连续步／历史不足分别处理。关闭 Echo 时纯 Ruler Cue 不变。
- P5 完成了读尺查询和监督标签生成，未把距离答案放入模型查询输入。真实帧上同一个 +6 cm 世界点、同一查询像素，在 5 cm 和 3 cm 标尺下分别为 +1.2 与 +2 格，米制位移保持 0.06 m。**Metric Grounding head、loss、共享表征梯度、warm-up、联合训练和过拟合检查均未实现。**
- 最终相关几何、适配器、Prompt、Echo、标签和历史回归 **34 passed**；另一次独立 JAX 核验 **4 passed**。冻结运行时哈希 **123/123 匹配**；`git diff --check` 通过。真实帧 CPU 输入 smoke 成功。没有完整模型 checkpoint 载入、保存恢复或动作前后向的通过记录。

## E. 未解决问题与下一道门槛

1. 主节点数据挂载不可用；需要恢复并核对数据 manifest 后才能在主节点做训练集规模读取。备用节点只用于本次只读取样。
2. 必须用真实基座 checkpoint 验证受控三→六通道迁移、完整视觉与动作零 Cue 等价、完整前后向中新增通道及动作共享表征的梯度、六通道优化器新建、checkpoint 保存／恢复。
3. 必须把同一个 adapter 与 Prompt Builder 接到正式训练和在线推理入口，并逐项核对输入一致性。当前试点只有 CPU 输入 smoke。
4. P5 训练头与 loss 尚无实现；按“纯 Ruler 六通道先可训练”的先决条件，暂未接入。2 cm 分度在真实帧过于密集，需更系统的可读性统计后才考虑使用。
5. P6 的 M0–M3 消融、位置分布审计、训练／评估协议、置信区间和 Figure A/B 尚未准备。没有完整任务成功率证据。

## F. 可复现入口与正式实验状态

当前可执行的最小核验入口是在主节点 dev worktree 设置 `PYTHONPATH=$PWD/src JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=''`，再用 OpenPI venv 的 Python 执行 `python -m pytest -q tests/test_metric_cue.py tests/test_metric_view_adapter.py tests/test_six_channel_inputs.py tests/test_metric_prompt.py tests/test_motion_echo.py tests/test_metric_grounding_labels.py`。真实帧输入脚本为 `scripts/metric_cue_cpu_smoke.py`，其输入样本需按仓库脚本参数提供；本次实测报告见同目录 `real_frame_input_report.json`。

**目前没有经过验收、可以诚实给出的 M0–M3 正式训练与评估命令。** 在真实 checkpoint 完整模型验收、数据挂载、训练／推理统一入口、P5 head/loss 及冻结评估协议完成前，编造四组命令会产生不可归因的实验。达到这些门槛后，M0–M3 应共用同一基座、数据、预算与四槽六通道契约：M0 空 Cue；M1 明确定义的 Direction Cue；M2 公制 Ruler；M3 相同 Ruler 加 Metric Grounding。M4 Echo 单独作为后续增量实验。预计产物应为各组版本化配置、数据与基座 hash、运行 manifest、checkpoint、读尺指标、配对 rollout 成功率及位置图；本次均未生成。


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
