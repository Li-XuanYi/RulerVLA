# L09 LoRA 暂停交接记录

记录日期：2026-10-30。文档核对时间为14:10 UTC（北京时间22:10）；147最近一次实际权限检查为14:09:49 UTC。以下运行结果均注明观察时间，恢复工作时必须重新核验。

## 当前状态与暂停范围

- [User] 用户要求暂停执行并整理交接。自动检查l09-lora已设为PAUSED；只有用户明确恢复后，才继续部署、GPU验证或正式训练。
- [Experiment] 本次暂停没有发送训练停止命令，没有重置GPU、重启节点或修改资源权限。145旧任务是否已经退出仍未知；暂停自动检查不能证明旧进程停止。
- [User] 后续只允许使用147的GPU。145禁止用于计算、等待、备用及SSH转发；只允许为结束本次既有任务做必要的停止控制。
- [Experiment] 147目前可SSH登录，但GPU及NAS权限缺失，匹配Python环境和独立运行树尚未部署。本次LoRA正式训练没有取得已放行、至少两个有限真实optimizer updates的证据。
- [Code] release中training_released=false、evaluation_released=false。S29位置场景仍未验收。

## 需要完成的目标

近期目标是在147以正确的协议、代码、数据、初始化和NCCL身份启动正式任务L09_spatial_M2_lora_seed42，并实际完成至少两个有限真实optimizer updates。PID或配置存在不能作为完成依据。

完整实验目标是在同一训练条件下完成M0/M1/M2/M3四组、seed42/43/44共12个正式run，各30000 updates；按协议保存checkpoint并进行公平比较。位置泛化评测还需先完成S29场景验收，再报告policy rollout SR、计数、分组指标及不确定性。当前没有可以支持方法性能结论的完整正式训练与rollout结果。

## 本次L09实际设计

[Code] 当前执行口径为在已验证的π0.5基座上做统一LoRA微调。四组使用相同六通道模型接口、训练配方、数据和评测口径，保留原flow matching损失，action horizon为10。

| 组别 | 输入提示 | Motion Echo |
|---|---|---|
| M0 | blank，关闭MetricRuler | 关闭 |
| M1 | 仅方向提示，direction_length_px=28 | 关闭 |
| M2 | EE/TCP为原点，世界/动作正XYZ方向0–15cm标尺，5cm等间隔刻度，无箭头 | 关闭 |
| M3 | 与M2相同的公制标尺 | 开启，使用过去5个control steps的实测历史位移 |

[Code] 提示配置分别为configs/rulervla_metric_cue_m0.yaml、configs/rulervla_metric_cue_m1.yaml、configs/rulervla_metric_cue.yaml、configs/rulervla_metric_cue_m3.yaml。协议记录Motion Echo采用抗锯齿及channel max。项目总览还讨论ActionCoord；本次L09的M0–M3含义由以上冻结协议定义，交接时须保持这套分组。

## 已完成的工作

### 训练配方和公平性

- [Code] 按用户要求将当前对照统一为LoRA，保留12个正式run ID；不使用旧full-FT结果替代LoRA验收，也不允许从full-FT checkpoint续接本次LoRA状态。
- [Code] 有效batch=8，micro batch=1，gradient accumulation=8，30000 updates；四组和未来主表比较保持一致，没有为单个组减少batch、rank或预算。
- [Code] PaliGemma采用gemma_2b_lora，attn/ffn的rank=16、alpha=16；action expert采用gemma_300m_lora，rank=32、alpha=32，rslora=false。同seed的初始化在各组配对，LoRA A为normal_std_0.01、B为零。
- [Code/Experiment] 白名单参数共53509280，包含LoRA、六通道图像stem及action/time投影模块；AdamW和EMA仅维护白名单参数，其他基座参数冻结。真实恢复回归核验过该参数数目。
- [Code] AdamW b1=0.9、b2=0.95、eps=1e-8、weight_decay=1e-10、梯度裁剪1.0，EMA decay=0.999；warmup1000 updates，peak LR=5e-5、end LR=5e-6，cosine至30000。
- [Code] 协议规定10000/20000/30000更新保存checkpoint，所有组和seed统一选择30000更新的LoRA checkpoint进行主比较。

### 配置、资源与运行记录

- [Code] 修订并冻结LoRA协议revision5，随后按用户要求将共同CPU起始门槛从64GiB降至48GiB，形成当前revision6。旧CPU64输入完整归档，原协议、失败输出和checkpoint保留。
- [Code] 当前共同GPU准入为两张卡各空闲至少28672MiB；BFC目标每卡26GiB，原生初始pool_bytes每卡至少24GiB。正式CPU门槛为max(48,ceil(1.25*四组完整smoke最高CPU峰值GiB))。
- [Code] 八处启动后按资源阈值终止的调用保持注释。训练开始后不主动因CPU/GPU阈值停止训练；用户最新停用145的指令仅对本次145任务构成明确例外。
- [Code] 两个既有启动器支持运行树、Python环境、workflow配置和NCCL路径覆盖。147配置configs/metric_workflow_lora_node147.yaml与已验证node145配置原字节一致，未改变训练数学或资源门槛。
- [Experiment] 一直区分真实进程、status文件、日志、manifest、指标和checkpoint；发现过进程被Killed/abort后status仍为running的情况。PID和历史status不能证明任务当前存活。
- [Experiment] A6000原等待队列已在确认未导入JAX、未启动GPU后取消；不得恢复旧队列或与新节点重复启动。145曾作为获准替代节点使用，现已由用户撤销其计算授权。

### pinned host与checkpoint恢复修复

- [Experiment] 历史V5/V6的M2均取得两次有限更新及CP2，但后续独立推理附近出现pinned host registration错误并被Killed，未完成完整验收。
- [Code] 独立推理路径先释放训练状态，将加载的参数显式放置到CPU jax.Array、逐叶核验并释放NumPy引用，再传入FSDP；增加delta_gpu_prediction阶段以区分加载和真实GPU预测。
- [Experiment] 后续6bfd12c版本的M2完成两次更新、CP2完整恢复、真实独立GPU推理及再次恢复CP2，但在continue_restored_optimizer又出现pinned错误并被Killed，第三步未确认。推理输出形状[1,10,32]且有限，不能计完整smoke通过。
- [Code] 确认LoRA恢复重新加载的冻结基座仍为NumPy，旧设备检查跳过NumPy。提交76e9b4c将其先device_put到目标CPU再覆盖delta，并使设备检查拒绝np.ndarray；保留device=None、Tracer、ShapeDtypeStruct行为及全部参数/AdamW/EMA/step/冻结签名核验。
- [Experiment] 真实CP2回归先复现DID NOT RAISE，修复后6 passed/79.82s，CPU放置前后字节签名一致；ruff、diff与只读复核通过。GPU效果需由匹配条件的完整smoke证明，不能仅用这些CPU检查宣布所有GPU故障已解决。

### Blackwell与实际NCCL库固定

- [Experiment] 145使用旧NCCL2.26.2时，首次零Cue等价通信发生CUDA out of memory，随后XLA abort，0次更新。准入CPU、GPU空闲量及BFC初始池当时均达标；终末峰值、其他分配及终止来源不明。
- [Code/Hypothesis] 旧库原生cubin最高SM90，官方2.28.9含SM120；旧PTX可以向新架构JIT，不能据此断言旧库完全不支持Blackwell或已经证明其导致OOM。
- [Code] 用独立库副本固定实际NCCL2.28.9，不替换共享venv。scripts/metric_nccl_runtime.py读取全局ncclGetVersion并记录实际库路径和SHA；指定SHA不匹配会在原训练CLI之前失败。
- [Experiment] 真实库绑定回归RED为2 failed/38.35s，GREEN为2 passed/2.47s，145部署后2 passed/1.29s；默认旧版本行为和错误SHA拒绝均核验。CPU profile本身不能替代训练PID的NCCL INFO通信日志。
- [Code] 现有release audit未自动比较额外NCCL profile，放行前还必须逐组人工核对version=22809、sha256、expected_sha256，以及各训练PID的实际NCCL2.28.9通信日志。

## 已保存的GPU结果与证据边界

[Experiment] 下表来自2026-10-30 12:36 UTC经A6000读取的共享NAS快照，属于145历史工程证据；本次暂停没有重新查询145实际进程。

| 组别 | 已记录更新 | status中的结果 | 最后记录/限制 |
|---|---:|---|---|
| M2 | 3 | complete；冻结权重不变、独立推理、完整恢复及恢复后继续均为true | status更新于11:53:31 UTC |
| M0 | 3 | 同上 | status更新于12:00:19 UTC |
| M1 | 3 | 同上 | status更新于12:07:00 UTC |
| M3 | 1 | running，stage=training_updates | status停在12:08:34 UTC，实时进程和后续结果未知 |

[Experiment] M2三次loss为0.1017097011、0.1376684159、0.0835810602；M0/M1记录的loss也有限。三组记录CPU峰值分别约38.801、34.657、36.906GiB。M3未取得完整验收，不能据三组峰值提前冻结正式CPU门槛，也未生成可放行147的四组聚合receipt。

[Experiment] 145已有结果保留，不能冒充147的GPU验收。当前147的新四组GPU验证及正式M2两个有限更新目标均未完成。

## 暂停时的具体阻塞

### 147：GPU及NAS访问权限缺失

[Experiment] WSL可登录lixuanyi@10.82.1.147:8864，hostname=ZjuServer147；uid/gid1060，补充组仅[1060]，不属于omniai(gid2000)。/dev/nvidia0、/dev/nvidia1为root:2000、0660，/home/Dataset2为root:2000、0770；os.access确认GPU读写与NAS访问均失败。14:09:49 UTC新SSH会话检查仍相同，nvidia-smi exit255，输出Failed to initialize NVML: Unknown Error。

[Inference] 账号资源权限缺失已经确认；权限恢复后的NVML状态、驱动健康及真实双卡空闲量尚未知。由管理员恢复合法组权限，例如在147执行sudo usermod -aG omniai lixuanyi，再重新登录核验。不能自行修改设备/NAS ACL或复制受限数据绕过权限。

### 145：旧任务是否停止尚未确认

[Experiment] 使用原属主yqt和既有密钥的必要停止连接被Permission denied(publickey)拒绝，停止脚本未执行。历史控制PID2344109、M2 PID2344133仅用于追溯，绝不能据此盲目终止进程。需要任务属主或管理员按实际UID、cmdline、cwd和cohort确认，只结束本次旧控制器及子任务，保留checkpoint和日志。

[User] 恢复工作后仍禁止145计算和转发；确认旧任务退出前不启动并行147 GPU队列。本次暂停后不再自动重试停止控制。

### 147：独立运行环境尚未部署

[Experiment] 拟用运行树/home/lixuanyi/XuanyiLi/metric_cue_runtime/RulerVLA_lora_r6_node147尚未部署。此前检查的两条旧openpi venv路径不存在，不能照抄145的/home/yqt路径或把拟用路径当作真实环境。权限及输入门槛恢复后，准备独立匹配环境，并按实际位置冻结METRIC_OPENPI_ROOT。

## 文件、版本和身份

所有仓库相对路径以下列开发树为根。原有未提交修改、删除和未跟踪文件很多，继续工作时必须保留，不得整体reset或清理。

| 用途 | 位置 |
|---|---|
| 开发树 | vla_7_A6000:/home/lixuanyi/XuanyiLi/work/worktrees/rulervla-metric-cue |
| 主仓库/交接入口 | vla_7_A6000:/home/lixuanyi/XuanyiLi/code/RulerVLA |
| 本地交付副本 | D:/Users/aa133/Desktop/RulerVLA/L09_LoRA_handoff_20261030.zh.md |
| 147 SSH | WSL Ubuntu，ssh -i ~/.ssh/zju_rsa -p 8864 lixuanyi@10.82.1.147 |
| NAS运行根 | /home/Dataset2/yuqiuting/rulervla/metric_cue_runs |
| 147拟用smoke根 | NAS下metric_cue_pilot/L09_lora_r5_smoke/smoke_cpu48_r6_node147_nccl2289 |
| 正式M2输出 | NAS下metric_cue_pilot/L09_lora_r5/L09_spatial_M2_lora_seed42 |
| 145旧结果根 | NAS下metric_cue_pilot/L09_lora_r5_smoke/smoke_cpu48_r6_node145_nccl2289 |
| 历史NCCL资产 | NAS下metric_cue_pilot/L09_lora_r6_node145_deployment；目录名不代表允许使用145主机 |

- [Code] 核心恢复实现：76e9b4c6c5e357b22d63d80cb4fde3847a5ca5e9。
- [Code] 路径覆盖及NCCL固定版本：bf96eef96f3447d51ce837829ce22dfc356fde2d，其src与训练CLI沿用上述核心实现。
- [Code] 147专用配置/计划及权限记录：9e93c97；13:12 UTC复查元数据：b23f4b4。交接前开发树HEAD为b23f4b4，交接文档自身提交在此之后。
- [Code] 重点实现：src/rulervla/cues/metric_ruler.py、src/rulervla/training/metric_lora.py、metric_workflow.py、metric_gpu_allocator.py、metric_protocol.py。
- [Code] 入口：scripts/run_metric_lora_smokes_amax.sh、scripts/start_metric_cue_amax.sh、scripts/metric_cue_train.py、scripts/metric_cue_launch.py、scripts/metric_lora_release_audit.py。
- [Code] 配置/计划：configs/metric_workflow_lora_node147.yaml、experiments/metric_cue_v1/NODE147_ONLY_PLAN.zh.md、REVISION5_LORA_PLAN.zh.md、REVISION6_CPU48_PLAN.zh.md。
- [Code] 协议记录libero_spatial共432 episodes、52970 frames；RGBD/geometry/norm身份在protocol和workflow配置中锁定，尚未在147重新验证读取。

### 必须保持的SHA

| 对象 | SHA256 |
|---|---|
| protocol revision6 | 9d35dbd9cd94c1ff1e380a7bcef3a37d1a5643e7d41c2f32683e3084b9cb4c5b |
| 暂停前release.json | 55a4324d3c29ca97b4cd8c2908c065ead1d1f12c28b027ac74696d37dc6eaee5 |
| 暂停前run_matrix.json | 8c8da9c06fa63c68c11c78b9c5c84441179230a913b4509d390401d056d268b3 |
| 固定NCCL2.28.9库 | 1792291e26d2b27fe51e59cf4dbd361dbfa84cbfe8ee8196c087c7ebf5146d48 |
| 官方NCCL wheel | 485776daa8447da5da39681af455aa3b2c2586ddcf4af8772495e7c532c7e5ab |
| 旧CPU64输入归档 | ca5ebefe6da43ed4c67b2dd327409538a8d9a083c5a08180fe7f496792771f2a |

[Code] protocol原字节保持冻结；release和run matrix只在真实验收后有依据地更新，并记录新SHA。旧CPU64归档为experiments/metric_cue_v1/revisions/r5_cpu64/frozen_inputs.tar.gz，不能重新把64GiB当活动下限。

### 证据入口

- experiments/metric_cue_v1/receipts/L09_lora_restore_device_fix_20261030.json：真实CP2恢复缺陷、修复和回归。
- experiments/metric_cue_v1/receipts/L09_lora_nccl2289_restart_20261030.json：首次NCCL OOM、隔离库、启动器绑定验证；其中等待状态为10:23 UTC历史快照。
- experiments/metric_cue_v1/receipts/L09_lora_node147_only_20261030.json：仅使用147的指令、145停止认证失败及12:36 UTC磁盘结果。
- experiments/metric_cue_v1/receipts/L09_lora_node147_access_recheck_20261030T131201Z.json：13:12 UTC权限复查。
- experiments/metric_cue_v1/receipts/L09_lora_pause_handoff_20261030.json：本次暂停、14:09:49 UTC最新权限检查和自动检查PAUSED记录。
- outputs/metric_cue_pilot/L09_lora_node147_only_20261030/node145_disk_observation.json：开发树本地磁盘观察，SHA3c1282a4d923145d4aa336068a0061f38ce28226b0bc243e8365aa213f8b2e93。开发树outputs是本地目录，旧冻结树outputs链接NAS，不能混淆。
- outputs/metric_cue_pilot/L09_lora_r5_implementation/restore_device_red.log、restore_device_green.log及inference_cpu_placement.log；NCCL绑定日志在outputs/metric_cue_pilot/L09_lora_r6_nccl_implementation。

## 用户明确恢复后的执行顺序

1. 先重新读取AGENTS.md、handoff.md、本文件、registry、当前protocol/release/run matrix及最新receipt，核实Git状态和实际主机。所有远程操作通过WSL Ubuntu SSH；A6000直连，不经145。每次读取以本地timeout -k 1s 25s及SSH ConnectTimeout=5/ServerAliveInterval=5/ServerAliveCountMax=3限时，不持续轮询旧挂起会话。
2. 恢复147合法资源权限，重新登录验证omniai组、GPU/NAS访问及nvidia-smi；核实两卡共同准入及CPU48GiB。核实145本次旧任务确实退出；必要停止控制只处理经过UID/cmdline/cwd/cohort核验的本次任务。
3. 检查147既有目录及本人任务，部署干净、独立的运行树；准备JAX/jaxlib0.5.3、Flax0.10.2、Optax0.2.4、Orbax0.11.13、NumPy1.26.4的匹配环境及隔离NCCL2.28.9库。核对模型、训练CLI和allocator源码哈希，活动运行树与库禁止热更新。
4. 在147重新核验实际数据、基座receipt、norm及LoRA配置，全部输入门槛通过后启动唯一M2/M0/M1/M3队列。统一显式设置METRIC_RUNTIME_ROOT、METRIC_OPENPI_ROOT、METRIC_WORKFLOW_CONFIG、METRIC_NCCL_ROOT、METRIC_NCCL_SHA256，NCCL_DEBUG=INFO并保存各训练PID日志。
5. 四组分别完成零Cue等价、两次有限更新、冻结主干不变、CP2全部参数/AdamW/EMA/step恢复、独立GPU推理、再次恢复同一CP2全部状态签名一致及恢复后第三步；记录有效CPU/GPU资源峰值，逐组核对实际NCCL通信。
6. 四组全部通过后在CPU环境运行既有release audit，按完整四组峰值校准共同正式CPU门槛；接入真实receipt及SHA、更新训练gates。evaluation_released继续false，直到S29位置场景等评测前提全部满足。
7. 全部本次smoke自然退出并确认无活动任务后，部署与验收源码和库匹配的干净正式运行树。在147通过既有正式入口启动M2 seed42，先查重，不提前创建CLI要求不存在的run目录。
8. 用正式run manifest、实际进程身份、有限loss及至少两次真实optimizer updates证明近期目标完成。随后结束自动检查本身，保留正式训练继续运行；继续完成30k及其余11个公平对照，之后再按协议推进评测。

## 继续工作时的操作约束

不降低其他gate，不使用旧full-FT或CPU/几何结果替代GPU/策略验收，不无依据重复相同pinned/OOM失败。若自然失败，先保留新日志、完整错误和checkpoint再分析；终止来源或终末峰值未知时明确记为未知。

修改仅用文件编辑工具或native patch，不通过Python/sed/perl/heredoc程序化修改代码；禁止读写/tmp，TMPDIR和测试缓存置于仓库ignored目录。不回滚Git、不覆盖用户改动、不删除实验数据与失败结果。每个工作单元更新现有handoff、registry和真实receipt。

暂停期间只保存本次交接材料；等待用户明确恢复后才执行上述步骤。
