## 后续执行更正

- GPU5并行Camera短测的第一轮被host reserve停止；第二轮Camera和ActionCoord进程消失，退出原因不明。user-1015 cgroup OOM计数为0；不能仅由过程判断OOM。
- ActionCoord在1372步退出且无5000步checkpoint，不能从1372续训。
- Camera改为GPU5单独训练；已完成两步真实更新，loss0.0724165/0.0725232。Dataset2完整step2状态写入成功，复制归档EHOSTDOWN失败，现正从保存前SHA逐leaf验证恢复。
- 缓存8GiB、磁盘staging、流式恢复SHA、释放初始化缓存已经部署为仅supplemental可选参数，既有Main运行默认配置不变。全状态重建测试4项、缓存逐样本一致性测试1项、冻结训练契约测试5项均通过。
- GPU5后续顺序为Camera10k、ActionCoord重新从base30k、Ruler-only30k。

# 小实验内存减负核验（2026-10-08）

目标：优先完成S22表示对照和S23四格消融；不改变full-FT、batch32、数据、cue、seed、优化器或训练预算。

## 已做

- GPU4 CameraXYZ full-state placement smoke（GPU optimizer/EMA），保留三个attempt及日志；全部未进入正式训练。
- 直接GPU初始化发生RESOURCE_EXHAUSTED；CPU初始化转GPU的第一个attempt暴露状态树静态元数据问题，修正后再次探测。
- 过程中GPU4资源发生变化，新VLLM进程3842867占用85218MiB，剩余约12GiB，无法继续验证GPU放置容量。
- 短测child均已退出。正式训练共用runner恢复到d7cf06d已验证版本，旧队列SHA契约保留。
- 数据loader的无限episode缓存可选改为字节数上限；缓存驱逐测试确认sample、动作chunk和repeat-last padding逐元素不变。当前仅在本地工程中验证，尚未部署到正式训练。

## 当前可验收状态

- ActionCoord-only继续运行在GPU5；不将显存间歇释放解释为任务空闲。
- GPU4 CameraXYZ正式任务尚未启动；既有资源队列继续等待。
- Ruler-only仍排队。S24依赖S22冻结结果触发条件，不能提前放行。
- 不将GPU optimizer方案称为已通过，不降低现有CPU内存门槛，不干扰其他用户VLLM。

## 后续放行门槛

1. 可选缓存上限的CPU/GPU内存短测。
2. 两步真实梯度和完整checkpoint保存/校验/恢复。
3. 绑定新源代码SHA并记录放置配置；满足实测内存余量后才启动正式训练。
