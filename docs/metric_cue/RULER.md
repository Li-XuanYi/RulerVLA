# 正轴公制标尺

[Code] 当前只绘制 EE 原点、正 XYZ 实线及 +5/+10/+15 cm 等价横刻度。XYZ 分别保留在独立 Cue 通道。物理范围 0～15 cm，几何保持真实相机 K/T 的投影关系。

## 当前入口

| 内容 | 位置 |
|---|---|
| 配置 | `configs/metric_ruler.yaml` |
| 训练/推理工作流配置 | `configs/metric_workflow.yaml` |
| 参数与投影数据 | `MetricRulerConfig`、`MetricRulerTick`、`MetricRulerProjection` |
| 几何与渲染 | `cues/metric_ruler.py::project_ruler_cue`、`render_ruler_cue` |
| 相机及裁剪工具 | `geometry/camera.py` |
| 双视角及六通道适配 | `MetricViewAdapter`、`MetricPipeline` |
| 真实样本审计 | `scripts/audit_metric_ruler.py` |
| 真实数据契约测试 | `tests/test_ruler_cue.py` |

[Code] 所有当前 CLI 默认读取上述配置。Cue 使用 OpenCV 亚像素抗锯齿，直接绘制到 224×224；RGB 保持原处理，Cue 为 float32[0,1]。拥挤与图形跨边界时省略刻度，并记录原始世界位置、像素和原因。读取配置时拒绝未知字段，checkpoint 校验完整输入合同的 fingerprint。

[Code] `blank` 生成空白 Cue，`direction` 生成固定像素长度的正方向实线，`metric` 生成真实物理长度及横刻度。Motion Echo 的历史位移表达是独立组件，按其原定义保留。

[Code] 仍被 OpenPI 和 Adaptive 对照调用的 RGB 叠加标尺函数继续保留。L09 revision3 采用有效 batch8、microbatch1、梯度累积8、30k updates，四组 Cue 配置已对齐当前字段。revision1 与 revision2 的冻结输入分别保存在 experiments/metric_cue_v1/revisions/r1/ 和 revisions/r2/；修订记录见 CONFIG_ALIGNMENT_R2.zh.md 与 REVISION3_BATCH8.zh.md。旧 checkpoint 使用其匹配的冻结源码，当前程序继续拒绝旧输入字段。正式训练和 rollout 的发布标志保持关闭。

[Experiment] 本次整理前的源码和配置完整保存在 `outputs/metric_cue_pilot/ruler_cleanup/before/source.tar.gz`。详细清理清单、测试、真实渲染产物见 `reports/metric_ruler_cleanup/report.zh.md`。旧报告、输出和 checkpoint 保持原位。
