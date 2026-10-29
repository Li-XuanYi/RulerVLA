# L09 revision6：CPU48资源修订与实施计划

> 使用superpowers:executing-plans在现有开发树执行；本次用户明确要求降低CPU内存门槛。

**Goal:** 将四组LoRA的共同CPU起始门槛由64 GiB降至48 GiB，并同步真实验收、发布审计及等待队列。
**Architecture:** 仅修订资源准入。正式门槛从四组完整真实smoke的最高CPU峰值校准，保留25%余量和48 GiB下限；不改变训练计算。
**Tech Stack:** 现有WSL SSH、Python/JAX、原始运行证据与冻结protocol/release/run matrix。
**Spec:** 用户本轮降低CPU门槛的指令；protocol.json revision6；revision5的LoRA数学配方继续适用。

## 依据与约束

- [User] 用户已授权降低CPU准入；四组仍须公平，启动后不得按资源阈值主动停止训练。
- [Experiment] V5/V6均有两次真实更新及CP2保存/恢复，推理阶段被终止；此前WSL读取V6的/proc/14586/status得到VmHWM35548836KiB（约33.9GiB）。终止前最终峰值未知，完整四组资源验收仍未通过。
- [Inference] 48GiB是本次选择的初始工程预算，覆盖已观察峰值加25%后的约42.4GiB；不能视作已验证峰值结论。当前约21GiB可用内存仍不满足48GiB。
- [Code] 正式CPU门槛为max(48, ceil(1.25 * 四组完整smoke最高CPU峰值GiB))；审计器从冻结协议读取下限，避免继续硬编码64。
- [Code] M0/M1/M2/M3、三个种子及未来主表对照共享规则；双卡每卡28672MiB、micro1/累计8、有效batch8、30k、LoRA白名单、AdamW/EMA、数据与评测保持一致。
- [Code] training_released/evaluation_released保持false；新协议SHA必须由新四组smoke实际绑定，旧full-FT及失败LoRA记录不能放行。

## 执行与验证

- [ ] 归档revision5 CPU64的原始protocol/release/run matrix/config/计划，记录SHA。
- [ ] 先将真实配置合同测试改为48及拒绝47，确认旧配置产生失败。
- [ ] 更新共同LoRA配置、protocol revision6、release与run matrix的版本和SHA，保留原输出/run ID。
- [ ] 审计器按冻结协议下限计算正式CPU门槛；原25%余量及其他验收不变。
- [ ] 执行真实CPU合同检查、ruff/diff/JSON校验，并确认正式训练gates继续关闭。
- [ ] 仅在旧等待器仍未导入JAX、未启动GPU且进程身份吻合时取消等待器，保留取消记录；不能停止已启动训练。
- [ ] 提交并在无本次活动进程时前向部署干净运行树，以唯一新目录启动48GiB四组等待队列。
- [ ] 更新handoff、registry、receipt及后续automation；如资源不足，报告真实等待状态，不降低其余gates。
