# 六通道 Metric Cue 工作流

本页定义 G2 工程入口；正式 M0–M3 协议留待 G4。默认训练命令只报告资源门槛。

## 预处理版本

当前版本为 `metric_cue_observation_v2`。本轮修复改为先绘轴线再绘刻度，负刻度的空心中心不会再被下一段线覆盖；公制几何、相机约定、六通道 shape 和权重迁移方式不变，但部分 Cue 像素会变化。旧 v1 checkpoint 必须使用其原提交的运行时，不能自动套用 v2 渲染；新入口会拒绝该版本不匹配。历史证据和 checkpoint 保留。G2 仍拒绝开启 Echo；新的负半轴样式、位移数值提示和训练实验不属于本轮修复。

## 来源重建

当前方法只导入本仓库 `src/openpi` 和 `src/openpi_client`。完整源代码由 Git tree 固定，上游比较 commit、历史快照和项目补丁见 [OPENPI_SOURCE.json](OPENPI_SOURCE.json)；不能只从比较 commit 重建未知历史补丁。Linux CPython 3.11 依赖固定在 `requirements/metric-cue-linux.lock`，包括不可变的 LeRobot Git commit。

锁文件排除由本仓库提供的两个 OpenPI 包和共享环境中未用于本方法的 RLinf。RLinf 的依赖在共享环境中不完整，不能据此声称整个共享环境无依赖冲突。构建检查从已提交的 Git tree 生成 wheel，并隔离导入其中的三个包；全新依赖环境安装尚未实测。

原环境含四个 Transformers 4.53.2 补丁，模板与 SHA-256 清单已提交。以下步骤在新环境执行；本轮没有重装共享环境。overlay 写入仅允许仓库内虚拟环境，默认检查只读。

```bash
python3.11 -m venv .venv-metric
.venv-metric/bin/python -m pip install -r requirements/metric-cue-linux.lock
.venv-metric/bin/python -m pip install --no-deps -e .
PYTHONPATH=src .venv-metric/bin/python scripts/check_metric_environment.py --apply-transformers-overlay
PYTHONPATH=src JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' \
  .venv-metric/bin/python scripts/check_metric_environment.py
```

权重和大数据不进入 Git。基座由 G1 `SOURCE_VERIFIED.json` 的对象 generation、MD5 和 SHA-256 标识；数据由原 manifest 的 episode 哈希标识。冻结规范化统计和其数据来源位于 `configs/metric_cue/spatial_norm/`，不重新估算历史统计。

## 统一调用链

训练：`SpatialDataset -> MetricPipeline.prepare_train -> MetricBatch -> MetricTrainer -> pi05.compute_loss -> AdamW -> checkpoint`。

推理：`当前同步观测 -> 同一 MetricPipeline.prepare_inference -> pi05.sample_actions -> 相同统计反归一化 -> LIBERO [10,7]`。

两个模式共用 MetricViewAdapter、SixChannelInputs 和 Prompt Builder。有效槽是 base_0_rgb、left_wrist_0_rgb；其余两槽全零且 mask=false。RGB 为 [-1,1] float32，Cue 为 [0,1] float32；state8 归一化后补齐到 32，Prompt 拒绝超过 200 token 的截断。G2 要求 Echo 关闭。

实时调用 `MetricPolicy.infer_observation(observation, ...)` 经 `LiberoAdapter.from_metric_observation` 接入同一处理流程。调用方提供当前两路 RGB、K/T、state8、episode_index、control_step 和 calibration_step；身份字段必须是非负整数，标定步与当前步不一致即拒绝。深度和末端四元数为可选字段，提供深度时继续校验形状与类型。完整 RGB-D 数据集入口保留原深度要求。

固定几何契约下，存储 RGB 直接缩放；公制刻度批量投影并复用已投影端点，近面交点按实际需要投影。Echo 关闭时直接使用 Ruler。SixChannelInputs 一次生成最终四槽六通道图像与 masks。预处理版本与指纹保持 v2，真实样本输出逐值一致。

动作目标是同 episode 的 actions[t:t+10]，到尾部才重复该 episode 最后一条动作；同时记录动作 frame 索引和 valid mask。该 mask 用于诊断，重复末尾目标继续参与原动作 loss，保持既有 SpatialDataset 语义。训练身份包含 manifest、RGB-D／几何哈希和 episode/frame。七维动作归一化后补零为 32 维；推理先用同一统计反归一化，再取前七维，不额外转换动作参考系或隐式裁剪。

## 入口与资源门槛

```bash
PYTHONPATH=src /path/to/python scripts/metric_cue_train.py \
  --mode preflight --out outputs/metric_cue_pilot/new_preflight
# 仅在 preflight 就绪后运行；smoke 最多三次更新。
PYTHONPATH=src /path/to/python scripts/metric_cue_train.py \
  --mode smoke --steps 2 --out outputs/metric_cue_pilot/new_gpu_smoke
```

完整 manifest、固定统计、基座身份、CPU 内存和选定 GPU 显存全部通过后，才允许初始化 GPU。当前训练实现为单 GPU 计算梯度、CPU 管理 AdamW，45000 MiB 空闲显存是保守准入阈值，实际峰值待 GPU smoke 验证。GPU 视觉与动作等价预先使用 atol=0.02、rtol=0.02，记录实测误差；不从 CPU 逐值相同推断 GPU 位级等价，不自动放宽容差。

正式训练使用显式 `--mode train` 和经过协议核对的配置／步数，本轮不执行。保存包括参数、完整 Adam 状态、step、输入 contract、规范化 assets、Git 和源码哈希；恢复 API 拒绝不同的模型、统计或优化器配置。旧三通道优化器不能恢复六通道模型，也不自动恢复历史队列。

```bash
PYTHONPATH=src /path/to/python scripts/metric_cue_infer.py \
  --mode predict --backend cpu --checkpoint /path/to/g2/checkpoints/2 \
  --sample /path/to/current_observation.npz --out outputs/metric_cue_pilot/new_inference
```

G2 推理从 checkpoint 自身恢复几何、Ruler、Prompt 参数、SentencePiece 模型字节和统计，并核对 contract 哈希。predict 模式恢复一次策略和预处理，通过 `MetricPolicy.infer_prepared` 使用同一次准备的模型输入。GPU 推理预检检查目标 checkpoint 和选定 GPU，预处理 assets 由恢复函数校验。训练数据和初始基座的可用性由训练入口检查。显式 `--g1-weights-bridge` 必须提供 checkpoint，用于既有 G1 六通道权重的工程检查。

## G3 接口

MetricBatch.observation/actions/identities 分离输入、动作目标和身份。MetricTrainer 的 objective 默认使用原 action_loss；G3 可以在同一个 NNX differentiated model 内组合辅助 loss，使用现有 model.embed_prefix 共享表征。查询和监督应另外传给 objective，不能放入推理 observation，也不能使用 detached 表征训练外部头。本轮没有辅助头或辅助 loss。
