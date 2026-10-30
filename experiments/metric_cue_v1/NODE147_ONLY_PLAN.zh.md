# L09 LoRA revision6 仅使用147节点

- [User] 2026-10-30最新指令禁止使用145，后续GPU验证及正式训练只在147运行。145不再用于计算、等待、备用或SSH转发。只允许为结束本次既有任务进行必要的停止控制，不能恢复145计算。
- [Experiment] WSL可登录lixuanyi@10.82.1.147:8864，hostname为ZjuServer147。账号uid/gid1060，补充组仅1060；GPU设备root:omniai(2000)/0660、NAS入口/home/Dataset2为root:omniai(2000)/0770，当前账号没有两者的访问权限；nvidia-smi报NVML Unknown Error。设备权限已确认缺失；权限恢复后的NVML及实际显存状态仍须验证。
- [Experiment] 145停止控制使用原属主yqt及既有密钥，直连和经A6000的stop-only连接均被publickey认证拒绝，控制命令未执行。停止状态未知，不能用共享NAS的旧running或complete替代实际进程核验。需任务属主或管理员恢复控制并结束仅属于本次cohort的活动控制器及训练子进程。
- [Code] protocol6原字节和SHA9d35dbd9cd94c1ff1e380a7bcef3a37d1a5643e7d41c2f32683e3084b9cb4c5b保持不变；所有组统一LoRA、有效batch8/micro1/累计8、30k及53509280个可训练参数，CPU起始48GiB、双卡各28672MiB、BFC目标26GiB及初始每卡至少24GiB不变。正式CPU门槛仍按max(48,ceil(1.25*四组完整smoke最高CPU峰值GiB))校准。
- [Code] configs/metric_workflow_lora_node147.yaml与已核验node145配置原字节一致，使用同一NAS基座、数据和receipt；名称表示目标部署，不代表147权限、环境或GPU验证已经通过。

## 执行约束及顺序

1. [Experiment] 通过WSL Ubuntu SSH直连147，后续任务启动前核验hostname、命令、工作目录及本人活动进程。不得经145跳转。由管理员恢复lixuanyi合法资源组权限，重新SSH登录后验证id、GPU/NAS访问、nvidia-smi、数据与基座身份；不自行sudo、修改设备ACL、重置GPU或重启节点。
2. [Experiment] 核实145旧本次cohort已退出，保留所有日志与checkpoint。只按实际cmdline/cwd/cohort匹配控制及停止本人任务，不能按2344109或旧M2 PID盲目终止进程。当前没有完成这项核验，不启动并行147 GPU队列。
3. [Code] 147拟用独立运行树/home/lixuanyi/XuanyiLi/metric_cue_runtime/RulerVLA_lora_r6_node147；部署前检查既有目录及活动任务，不能覆盖旧产物。冻结源以bf96eef96f3447d51ce837829ce22dfc356fde2d及本次147配置提交为基础，模型/训练CLI/allocator须与已审计76e9b4c原字节一致，八处启动后资源终止调用保持注释。147现有两条旧openpi venv路径不存在，独立匹配环境尚未准备，不能把拟用路径当作已部署。
4. [Code] 为147准备匹配JAX/jaxlib0.5.3、Flax0.10.2、Optax0.2.4、Orbax0.11.13、NumPy1.26.4的独立环境；按实际安装结果冻结METRIC_OPENPI_ROOT，不修改其他任务的环境。NCCL采用隔离2.28.9、version22809、lib SHA1792291e26d2b27fe51e59cf4dbd361dbfa84cbfe8ee8196c087c7ebf5146d48；官方wheel及bundle可从已验证NAS资产读取，历史node145目录名不表示使用145主机。任何本次活动任务期间不得热更新其运行树或库。
5. [Experiment] 147同一冻结源码及环境重新执行M2/M0/M1/M3，拟用NAS目录metric_cue_pilot/L09_lora_r5_smoke/smoke_cpu48_r6_node147_nccl2289。显式设置147的METRIC_RUNTIME_ROOT、METRIC_OPENPI_ROOT、METRIC_WORKFLOW_CONFIG=configs/metric_workflow_lora_node147.yaml、METRIC_NCCL_ROOT及固定METRIC_NCCL_SHA256，NCCL_DEBUG=INFO并写独立日志。四组各自核对CPU profile和实际训练PID的NCCL通信版本；只在原CLI全部准入通过后启动GPU。
6. [Experiment] 各组必须完成零Cue等价、两次有限更新、冻结主干不变、CP2全状态恢复、独立GPU推理、再次恢复同一CP2全部状态签名一致及第三步，有完整资源记录后执行既有CPU release audit。145的磁盘结果只保留为历史工程证据，不充当147验收。release的training_released/evaluation_released当前均false，S29场景未验收，不能提前放行。
7. [Experiment] 全部smoke自然退出且核实没有本次活动进程后，部署匹配四组审计源码/CLI/allocator与NCCL哈希的干净运行树，在147通过scripts/start_metric_cue_amax.sh启动L09_spatial_M2_lora_seed42。保留NAS正式输出metric_cue_pilot/L09_lora_r5/L09_spatial_M2_lora_seed42，先查重，至少两次有限真实更新才报告并结束自动检查；绝不因启动后资源阈值停止训练。

[Experiment] 当前没有在147启动GPU任务。权限和145停止核验阻塞记录见receipts/L09_lora_node147_only_20261030.json。自动检查已更新为147唯一目标；本文件是部署约束更新，不改变冻结实验数学。
