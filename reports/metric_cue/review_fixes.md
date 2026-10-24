# RulerVLA Metric Cue 审计修复

日期：2026-10-24。范围：修复已确认缺陷，保留 G2 的 Echo 关闭限制。

## 结论

修复已经提交并通过 91 项轻量 CPU 测试。四个真实样本通过训练／推理输入一致性、连续动作块和反归一化核验。本轮没有执行完整模型 G1 重验、GPU 作业、正式训练、rollout 或 G3。

来稿称“7 个 Bug”，正文实际编号为 6 项；其中包含 G2 的既定限制和新增实验功能，不能全部作为缺陷移除。

## 审计逐项处理

| 来稿问题 | 核验与处理 |
| --- | --- |
| Pipeline 禁用 Echo、没有传递 history | 属于本阶段明确约束。保留 `MetricPipeline` 的拒绝逻辑和默认关闭设置，没有把历史缓冲接入 G2。 |
| 负向空心刻度被后续线段填实 | 已确认。改为每个轴通道先绘全部线段，再绘刻度，最后绘原点；保留既有正负样式和公制投影。 |
| Echo 覆盖清除 Z 通道 | 已确认。组合改用逐元素最大值，避免黄色的零蓝色值清除原有 Z 像素。保留黄色编码，没有加入任意蓝色分量。 |
| 近面缺少裁剪、视口边界出现假箭头 | 已确认。先裁剪三维线段到近面，再裁剪到视口；仅原始当前端点确实可见时绘箭头头部，裁剪交点不会冒充当前末端位置。 |
| 不支持新的负轴样式 | 属于新实验功能。本轮保留固定样式，`solid_hollow` 等仍被拒绝。 |
| 静止时 Prompt 仍声称显示箭头 | 已确认。补充静止／视图不可见／只有裁剪线段的文字分支。没有新增位移数值提示。 |

额外修复：即使当前位置数值相同，也必须拒绝来自其他帧的有效 Echo 历史；历史与当前观测的 frame/episode 身份使用严格非负整数校验，避免 `int(10.9)` 等静默截断。支持 NumPy 整数标量，拒绝浮点、布尔、字符串和一维数组身份。

## 输入版本与 checkpoint

预处理版本从 `metric_cue_observation_v1` 升为 `metric_cue_observation_v2`，因为默认 Cue 像素确实发生变化。六通道 shape、几何单位、相机约定和 G1 验证过的受控权重迁移方式未改。

旧 v1 checkpoint 保留，须使用其原 Git 提交运行时。新恢复入口在加载 assets 前拒绝 v1 预处理契约，避免静默换用 v2。没有自动转换旧 checkpoint，也没有声称 v2 重新完成了完整模型 GPU 或动作学习验收。

## 证据

- 新增最初 25 项回归在旧实现上：16 failed、9 passed；修复后连同既有测试为 83 passed。随后加入 8 项身份类型校验，最终 19 个测试文件共 **91 passed、94 warnings**。没有测试失败；警告原文保存在日志。
- 真实数据来自此前按原 manifest 哈希核对过的 episode 1261/1262 缓存。manifest 索引总量为 52,970 帧；本轮仅核验下面四个样本，没有扫描完整训练集。

| 全局索引 | episode / 当前帧 | horizon=10 的动作来源 |
| --- | --- | --- |
| 20 | 1261 / 20 | 20–29，真实连续动作 |
| 108 | 1261 / 108 | 108、109，随后仅重复该 episode 的 109 |
| 109 | 1261 / 109 | 最后一帧 109，尾部填充 |
| 110 | 1262 / 0 | 0–9，没有跨 episode |

- 四个样本的 RGB、Cue、state、prompt tokens、image/token masks、dtype 和 shape 在训练／推理模式逐值相同；只有训练目标 actions 被单独加入。
- 真实 batch 为 `actions=float32[2,10,32]`，四槽图像为 `float32[2,224,224,6]`；两个有效槽，其余槽全零、mask=false。
- 动作恢复为 LIBERO 七维接口，四个样本反归一化最大绝对误差 `1.1920928955078125e-7`。帧 20 当前原始观测与 G1 样本逐值相同。
- 同一真实观测与旧 v1 已保存模型输入比较：RGB、state、tokens 和 masks 逐值相同；base/wrist 的 Cue 分别有 **54 / 16 个通道值**变化，空槽无变化。这是修复后的刻度光栅差异，不能宣称 v1/v2 Cue 相同。
- 新 v2 pipeline 指纹：`84631c2391dc7c99814be998972cdef9c6d7deaf8b2fa1041506c30ea57fdfd8`。
- 历史冻结 runtime 的 123 个文件按原 manifest 的 snapshot 哈希核对，0 不匹配；9 份 G1 证据哈希与 G2 留存基线一致。

## 修改文件

| 文件 | 内容 |
| --- | --- |
| `src/rulervla/cues/metric_ruler.py` | 线段／刻度绘制顺序 |
| `src/rulervla/cues/motion_echo.py` | 近面与视口裁剪、原始端点可见性、非破坏组合及诊断 |
| `src/rulervla/cues/metric_prompt.py` | 与静止、不可见、裁剪后的 Echo 对齐 |
| `src/rulervla/adapters/metric_cue.py` | 当前观测身份校验、可见性与 Prompt 联动 |
| `src/rulervla/adapters/metric_pipeline.py` | v2 预处理标识；G2 Echo 限制保留 |
| `src/rulervla/training/metric_workflow.py` | checkpoint 预处理版本拒绝逻辑 |
| `tests/test_metric_ruler_rendering_regression.py` | 刻度绘制回归 |
| `tests/test_motion_echo_clipping_regression.py` | 裁剪、箭头、通道保留回归 |
| `tests/test_metric_echo_prompt_regression.py` | Prompt 和历史身份回归 |
| `tests/test_metric_review_contract.py` | G2 与版本限制回归 |
| `docs/metric_cue/WORKFLOW.md` | v2 和旧 checkpoint 使用说明 |
| `handoff.md`、`reports/metric_cue/review_fixes.md` | 工程交接和本报告 |

## 来源与重现

基线：`4bc1466078e47b8005d3250ef930abc360bfb1ff`。

源码修复提交：

1. `c865768e6ea47685d6f2e4461a69b82eea22e3d9`：渲染、Prompt、版本与 G2 限制。
2. `777f1e874f21cb64407d7c22b00cebf3ba9c9207`：严格整数身份校验，最终测试与真实样本审计基于此提交。

独立审计工作树：`/home/lixuanyi/XuanyiLi/.worktrees/rulervla-metric-review`。沿 main 历史保存提交，未创建新的远程功能分支，本轮未 push。原有两个工作区的历史修改、checkpoint 和结果保留。

运行时：`/home/lixuanyi/XuanyiLi/openpi/.venv/bin/python`，导入当前工作树的 `src/openpi`。依赖仍按仓库的 `requirements/metric-cue-linux.lock` 和 `docs/metric_cue/OPENPI_SOURCE.json` 固定；没有安装或修改共享依赖。

证据在审计工作树的 `outputs/metric_cue_pilot/review_fixes/`：`red.log`、`green_final.log`、`test_results_final.json`、`real_data_audit_final/report.json`、`v1_v2_comparison.json`、`preservation.json`。数据审计可用原入口和新的空输出目录重现：

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' /path/to/python \
  scripts/metric_cue_data_audit.py \
  --cache-root /path/to/verified/segmented_data_cache \
  --reference-g1-frame /path/to/real_sample_1261.npz \
  --out /path/to/new_audit
```

## 证据边界与后续接口

- `np.maximum` 只防止原像素被清零，不提供独立 Echo 语义通道；重叠可能变白，也可能填入标尺空心刻度。G2 关闭 Echo，因此这不是本阶段的模型输入变化。
- 原始端点位于相机后方时，Camera 的原始 UV 诊断仍可能含 NaN；严格 JSON 导出应转换为 null 等显式表示，不能把裁剪点当作原始测量点。
- 箭头头部可见性尚未单独核验缩放后的头部像素；默认 6px 头按当前回归工作，极小头部等新实验配置需要另行验证。
- 投影退化、刻度重叠仍可能降低可读性；这里没有学习有效性、读尺精度或相机泛化结论。
- 本轮没有重新检查 Dataset2/GPU 当前资源，也没有启动 GPU smoke；此前 G2 的完整 GPU 验收仍需独立完成。CPU 结果不能推广为 GPU 位级等价。
- G3 仍可使用 `MetricBatch.observation/actions/identities`、`MetricTrainer.objective` 和共享 `embed_prefix` 接口；本轮未添加 Metric Grounding 头、loss 或数值标签提示。

附带的渲染对比图使用合成标定相机 fixture，仅展示三类缺陷的像素变化，不是 LIBERO rollout 或模型输出证据。
