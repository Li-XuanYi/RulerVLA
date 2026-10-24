# Metric Cue 代码精简验收

日期：2026-10-24。基线提交：`23b231821b4ded18ebb5ca0580b2bda1a554cc59`。源码提交身份记录于 `outputs/metric_cue_pilot/simplification/source_commit.json`，交付分支为 `main`。

## 实现

- 公制原点与刻度批量投影；绘制复用端点的 UV 与深度，近面交点按实际需要投影。绘线、空心刻度与原点的顺序保留。
- 固定几何契约中的两次 RGB 水平翻转相消，存储 RGB 直接使用既有缩放函数。Cue 保留规定的相机与模型方向变换。
- Echo 关闭时直接使用 Ruler 输出；六通道转换直接生成 state、prompt、可选 actions、四槽图像与 masks。空槽分别分配数组。
- 实时入口支持省略深度和四元数；提供深度时校验其形状与类型。实时入口、Echo 和在线控制步共用严格整数身份核验。完整 RGB-D 入口保留原深度要求。
- 训练和推理分别检查实际需要的资源。checkpoint 推理使用目标权重和预处理资产，GPU 预检使用选定设备的显存条件。
- Cue YAML 解析一次；predict 恢复一次策略和预处理，`infer_prepared` 复用同一次准备的输入。优化器结构在初始化与恢复时检查，每次更新继续检查 loss 和梯度有限性。

## 保留的契约

- 六通道权重迁移、SigLIP 的 RGB/Cue 数值处理及模型源码保持原提交内容。
- 两个有效相机视图、四槽顺序与显式 masks 保持不变：`True, True, False, False`。
- RGB 为 `float32[-1,1]`，Cue 为 `float32[0,1]`，每槽 `[224,224,6]`；state 为 `[32]`，tokens 为 `int32[200]`。
- 模型版本、checkpoint schema、归一化统计、tokenizer、模型结构与 fingerprint 检查保留。v1 checkpoint 的版本拒绝规则保留。
- 预处理版本保持 `metric_cue_observation_v2`，指纹保持 `84631c2391dc7c99814be998972cdef9c6d7deaf8b2fa1041506c30ea57fdfd8`。
- G2 的 Echo 关闭规则保留；真实动作目标继续使用同 episode 的连续 horizon=10，末尾填充该 episode 最后一条动作。

## 真实样本检查

原始 manifest 的 episode 1261/1262 缓存已按固定 SHA-256 核对。精简前保存完整模型输入，精简后逐值比较所有字段及 dtype、shape。

| 全局索引 | episode / 当前帧 | 动作来源 | 模型输入最大差 |
| --- | --- | --- | ---: |
| 20 | 1261 / 20 | 20–29 | 0 |
| 108 | 1261 / 108 | 108、109，随后重复 109 | 0 |
| 109 | 1261 / 109 | 109，末尾填充 | 0 |
| 110 | 1262 / 0 | 0–9 | 0 |

四个样本同时通过训练／推理一致性、动作身份和动作反归一化检查。反归一化最大绝对误差为 `1.1920928955078125e-7`。实时入口使用真实样本的 RGB、末端状态与相机参数，省略深度和四元数后得到相同模型输入。

## 回归和恢复检查

- 141 项回归通过，87 条依赖弃用警告。覆盖公制几何、近面与视口裁剪、负空心刻度、Echo、四槽 masks、六通道数值、BF16 RGB 保持、Prompt、输入身份、动作反归一化、推理资源条件和配置单次解析。
- CPU 与 GPU 两种 CLI 参数路径均会在缺少显式 checkpoint 的 G1 bridge 请求处报错。
- 使用真实 pipeline 的统计、tokenizer 字节和契约保存预处理资产，再由生产 `pipeline_from_checkpoint` 恢复，指纹及真实模型输入逐值一致；v1 契约按规则拒绝。
- 本轮恢复检查针对 checkpoint 预处理资产。完整 π₀.₅ 参数和 Adam 状态的原 G1 证据保留，权重迁移与模型源码未改。
- 实际调用推理预检检查既有 G1 六通道 checkpoint 的元数据和设备状态，配置只提供推理所需字段；该检查没有读取训练 manifest 或初始基座路径。

## CPU 预处理测量

同一 episode 1261/frame 20，在同一远程 Python 环境和 16 个允许 CPU 核心上，预热 3 次并测量 30 次。

| 指标 | 基线 | 精简后 |
| --- | ---: | ---: |
| 中位耗时 | 19.594 ms | 13.292 ms |
| `Camera.project` 调用次数 | 142 | 3 |
| Echo 渲染／组合调用次数 | 2 / 2 | 0 / 0 |
| 标准三通道转换调用次数 | 1 | 0 |
| 图像方向变换调用次数 | 6 | 2 |

中位耗时减少约 32%。测量范围为 CPU 预处理，不包含数据集加载、完整模型前向或环境动作，不能据此推断 rollout 速度。

## 修改文件与证据

生产修改为 `metric_ruler.py`、`motion_echo.py`、`metric_cue.py`、`libero.py`、`schema.py`、`config.py`、`metric_resources.py`、`metric_workflow.py` 和 `scripts/metric_cue_infer.py`。相应测试、WORKFLOW 和 handoff 同步更新。

证据目录为审计工作树的 `outputs/metric_cue_pilot/simplification/`：`baseline.json`、四份 `baseline_input_*.npz`、`real_verification.json`、`regressions.json`、`regressions.log`、`cli_contract.json`、`inference_admission.json`、`source_commit.json` 以及提交后正式数据审计和 CLI prepare 的结果。

本轮优化器更新、训练、GPU 模型作业和 rollout 数量均为 0。
