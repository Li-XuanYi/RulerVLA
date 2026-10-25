# 六通道 Metric Cue 工作流

本页定义六通道工程入口。M0–M3 的训练和评测条件见 `experiments/metric_cue_v1/protocol.json`，发布条件见同目录的 `release.json`。默认训练命令只报告资源门槛。

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

两个模式共用 MetricViewAdapter、SixChannelInputs 和 Prompt Builder。有效槽是 base_0_rgb、left_wrist_0_rgb；其余两槽全零且 mask=false。RGB 为 [-1,1] float32，Cue 为 [0,1] float32；state8 归一化后补齐到 32，Prompt 拒绝超过 200 token 的截断。M0 的 Cue 为空白；M1 绘制固定像素长度的方向线段；M2 绘制公制标尺；M3 加入实测历史位移。

实时调用 `MetricPolicy.infer_observation(observation, ...)` 经 `LiberoAdapter.from_metric_observation` 接入同一链路。调用方必须提供当前两路公制 depth、K/T、state8、episode_index、control_step 和 calibration_step；标定步与当前步不一致即拒绝。M3 在每个 episode 开始时调用 `reset_episode`，并在每个环境控制步调用 `observe_control_step` 或 `observe_observation`；历史不足时保留明确的无效状态。此接口不读取未来动作，不执行机器人控制。

动作目标是同 episode 的 actions[t:t+10]，到尾部才重复该 episode 最后一条动作；同时记录动作 frame 索引和 valid mask。该 mask 用于诊断，重复末尾目标继续参与原动作 loss，保持既有 SpatialDataset 语义。训练身份包含 manifest、RGB-D／几何哈希和 episode/frame。七维动作归一化后补零为 32 维；推理先用同一统计反归一化，再取前七维，不额外转换动作参考系或隐式裁剪。

## 入口与资源门槛

完整训练数据先执行 `scripts/metric_cue_full_data_audit.py`；该入口逐一核对 432 个真实 episode 的文件哈希、几何配对、形状和帧数。协议与运行矩阵位于 `experiments/metric_cue_v1/`。完整模型 CPU 输入、梯度累积结构和在线会话使用 `tests/real_metric_arm_contract.py` 验证。

```bash
PYTHONPATH=src /path/to/python scripts/metric_cue_train.py \
  --mode preflight --out outputs/metric_cue_pilot/new_preflight
# 仅在 preflight 就绪后运行；smoke 最多三次更新。
PYTHONPATH=src /path/to/python scripts/metric_cue_train.py \
  --mode smoke --steps 2 --out outputs/metric_cue_pilot/new_gpu_smoke
```

完整 manifest、固定统计、基座身份、CPU 内存和选定 GPU 显存全部通过后，才允许初始化 GPU。当前训练实现为单 GPU 计算梯度、CPU 管理 AdamW，45000 MiB 空闲显存是保守准入阈值，实际峰值待 GPU smoke 验证。GPU 视觉与动作等价预先使用 atol=0.02、rtol=0.02，记录实测误差；不从 CPU 逐值相同推断 GPU 位级等价，不自动放宽容差。

正式训练使用显式 `--mode train --arm M0|M1|M2|M3 --seed 42|43|44` 和协议指定的 30000 次更新、有效批量 32、一次更新累积 32 个真实样本。入口要求 `release.json` 中全部发布条件通过，以及完整模型 GPU 更新和恢复记录。训练在 10000、20000、30000 次更新保存参数、完整 Adam 状态、EMA、step、输入 contract、规范化 assets、Git 和源码哈希；`--resume` 仅恢复同一运行身份的完整状态。当前发布条件未通过，因此未执行正式训练。

```bash
PYTHONPATH=src /path/to/python scripts/metric_cue_infer.py \
  --mode predict --backend cpu --checkpoint /path/to/g2/checkpoints/2 \
  --sample /path/to/current_observation.npz --out outputs/metric_cue_pilot/new_inference
```

G2 推理从 checkpoint 自身恢复几何、Ruler、Prompt 参数、SentencePiece 模型字节和统计，并核对 contract 哈希；tokenizer 不依赖恢复时的网络下载内容。显式 `--g1-weights-bridge` 仅供既有 G1 六通道权重的工程检查，不恢复 G1 优化器。

正式 rollout 使用 `scripts/metric_cue_serve.py` 提供策略，并由 `scripts/metric_cue_eval.py` 读取冻结的 Spatial 配对场景。客户端在每个环境控制步提交同步的双视角 RGB-D、相机 K/T 与 state8；服务端按 episode 重置历史，再返回十步动作块，客户端每五步重新查询。两端先检查 `release.json` 的训练、场景、模拟器和资源条件。当前模拟器源码路径对 SSH 用户不可读取，在线 rollout 尚未核验或启动。

## G3 接口

MetricBatch.observation/actions/identities 分离输入、动作目标和身份。MetricTrainer 的 objective 默认使用原 action_loss；G3 可以在同一个 NNX differentiated model 内组合辅助 loss，使用现有 model.embed_prefix 共享表征。查询和监督应另外传给 objective，不能放入推理 observation，也不能使用 detached 表征训练外部头。本轮没有辅助头或辅助 loss。
