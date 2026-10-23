
## 2026-10-23 公制 Cue 六通道试点（开发 worktree）

- 唯一修改工作树：`/home/lixuanyi/XuanyiLi/.worktrees/rulervla-metric-cue`，分支 `research/metric-cue-pilot`，HEAD `c6ddada0`；原 `main` 脏工作区与备用节点未改。
- 试点实现：双视角 EE 世界轴公制标尺、独立三通道 Cue、四槽六通道输入、统一尺度 Prompt、默认关闭的实测 Motion Echo、读尺标签，以及受控的 JAX patch embedding 三→六通道初始化。详细证据和限制见 `docs/metric_cue/STATUS.md`。
- 相关回归 34 passed；JAX 局部契约／权重迁移／卷积等价与梯度 4 passed；一帧真实数据 CPU 输入 smoke 通过，Prompt 95/200 token。冻结 runtime 的 123 个哈希仍全部匹配；`git diff --check` 通过。
- 数据挂载在主节点为 `Host is down`；8 张 A6000 当时占用较高。未运行长训练、GPU rollout、正式 benchmark 或 push。
- 下一个工作单元先验证真实 checkpoint 的完整 π0.5 前后向、动作零 Cue 等价、新通道梯度和保存恢复；再统一正式训练与推理入口。随后实现 P5 训练 head/loss，才能准备 M0–M3 受控命令。不要把当前 CPU 输入 smoke 当作策略已可训练。
- 远程环境无 `codegraph` CLI；本次按用户许可在本地源码镜像建索引。历史 `runtime_v3` 和所有 checkpoint／结果／队列保持不动。
