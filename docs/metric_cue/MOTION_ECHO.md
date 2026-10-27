# Motion Echo 输入合同

[Code] `MotionHistory` 保存同一 episode 当前与五个环境控制步前的实测 EE 世界坐标。`SpatialDataset.get_history` 和 `MetricInferenceHistory` 提供训练与推理历史，共用 `MetricViewAdapter.prepare`。净位移为 `current_m - past_m`；两个端点都投影到当前时刻的各相机视角。

[Code] `render_motion_echo` 复用 `geometry.camera.project_visible_segment`，先裁剪三维近平面再裁剪视野。线段使用 OpenCV `LINE_AA`、8 位亚像素坐标；线宽 2 px，头部长度最大 6 px 且不超过可见长度的 40%，头部宽度等于其长度。真实当前点可见时绘制头部；当前点不可见时保留可见线段。位移小于 3 mm 或可见投影小于 1 px 时留空。所有位置来自实际物理投影。

[Code] 独立 Echo 为 `float32[H,W,3]`，R 与 G 相等、B 为零、范围 `[0,1]`。`compose_cues` 对 Ruler 与 Echo 逐通道取最大值，Z 通道逐值保留。重叠区域的颜色可能混合，共享三通道不能独立解码两个组件；RGB 不变。Agentview 与 Wrist 使用相同物理定义和绘制参数，各自应用当前相机标定和统一图像方向转换。

[Code] 诊断记录当前／历史点像素位置及深度、物理位移、实际可见线段端点、像素长度、近平面与视野裁剪、当前点可见性、头部状态和省略原因。相机后方点的像素位置为 JSON `null`；输出支持 `allow_nan=False`。Prompt 依据两个实际渲染状态描述历史不足、低于阈值、可见箭头、裁剪线段或不可见。

[Code] Echo 默认关闭。启用时 `MetricPipeline.contract()` 额外保存 `motion_echo_rendering: antialiased_channel_max`；`pipeline_from_checkpoint` 在读取资产之前拒绝缺失或不匹配的 Echo 渲染身份。关闭时合同保持原样。当前六通道结构、state、mask、π0.5、Action Expert 与 Flow Matching 目标保持原实现。

验证入口：`tests/test_motion_echo_rendering.py`、`scripts/audit_motion_echo.py`；执行证据见 `reports/motion_echo/`。边界压力测试使用真实 EE 端点和受控相机参数变体，单独于原始相机样本统计报告。现有正式协议和训练／rollout gates 继续生效。
