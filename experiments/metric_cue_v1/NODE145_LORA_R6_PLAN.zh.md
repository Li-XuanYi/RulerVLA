# L09 LoRA revision6 的145节点部署

- [Experiment] 2026-10-30 06:59 UTC，145出现四张空闲超过28672MiB的RTX PRO 6000 Blackwell，CPU可用约500GiB；A6000原等待器仍受48GiB CPU门槛阻塞。用户此前已授权在其他节点运行并持续完成LoRA重启，本次沿用该授权执行部署。
- [Code] protocol6原字节及SHA9d35dbd9cd94c1ff1e380a7bcef3a37d1a5643e7d41c2f32683e3084b9cb4c5b不变。LoRA、有效batch8/micro1/累计8、30k、AdamW/EMA范围、rank/alpha、数据、初始化、损失及评测保持一致。CPU起始48GiB、双卡各28672MiB、26GiB BFC目标和初始原生pool至少24GiB保持一致。
- [Code] 新配置configs/metric_workflow_lora_node145.yaml仅把基座和其receipt的绝对路径改为相同NAS对象；其余解析后的字段须与原LoRA配置一致。启动器支持METRIC_RUNTIME_ROOT、METRIC_OPENPI_ROOT、METRIC_WORKFLOW_CONFIG，原默认路径保持原行为。
- [Experiment] 145的Python环境使用/home/yqt/XuanyiLi/openpi/.venv，JAX/jaxlib0.5.3、Flax0.10.2、Optax0.2.4、Orbax0.11.13、NumPy1.26.4与A6000一致；实际NCCL预加载使用该venv自身的libnccl.so.2。硬件和驱动单独写入运行receipt，四组均在145以相同新冻结提交重新验收。

## 执行顺序

1. [Code] 只在开发树修改两个启动器、增加节点配置；模型、训练CLI、allocator及八处已注释的启动后资源终止调用保持原字节。A6000活动运行树保持76e9b4c，不热更新。
2. [Code] 核验旧node145配置被LoRA协议拒绝；新配置与原配置除两条绝对路径外完全相同，实际基座receipt哈希相同，bash语法和正式闭门检查通过后提交。
3. [Code] 将干净提交通过NAS Git bundle部署到/home/yqt/XuanyiLi/rulervla-metric-lora-r6-runtime；outputs链接到既有NAS运行根，缓存和TMPDIR放运行树ignored build目录。确认没有该运行树的活动任务。
4. [Experiment] 在145依次执行四组实际preflight；数据、norm、基座身份及共同资源门槛均通过后，重新核实A6000等待器身份、工作目录、waiting_resources、JAXfalse/GPUfalse。若已进入真实GPU阶段，保留原队列并放弃切换；只有仍未启动GPU的本次等待器可以取消。
5. [Experiment] 原等待器及控制器均退出后，145启动唯一的新队列，顺序M2/M0/M1/M3，目录L09_lora_r5_smoke/smoke_cpu48_r6_node145。启动后不按CPU/GPU资源阈值终止任务，不停止其他作业。
6. [Experiment] 四组各自完成零Cue等价、两次有限更新、冻结参数不变、CP2完整恢复、独立GPU推理、恢复后第三步及资源记录后，使用CPU环境运行既有release audit。共同正式CPU门槛仍为max(48,ceil(1.25*四组最高CPU峰值GiB))，不能把此前A6000部分结果计入本队列。
7. [Experiment] 在本次smoke全部自然退出后，部署与四组审计的模型/CLI/allocator哈希匹配的干净正式运行树；通过同一启动器及三个路径覆盖启动M2 seed42正式训练，输出保留L09_lora_r5/L09_spatial_M2_lora_seed42。先查重，至少两个真实有限更新才报告正式启动。
8. [Experiment] release/registry/handoff与receipt随证据更新；evaluation_released保持false，S29位置场景待验收。自然失败时保存全部日志和checkpoint，并按新的原始错误定位；不无依据重复相同失败。

本文件记录工程部署，不改变冻结训练协议或实验预算。资源窗口不等于GPU验收通过，预检或PID不等于正式训练已开始。
