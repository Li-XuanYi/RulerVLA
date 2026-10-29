# L09 revision5：统一LoRA微调实施计划

> 使用superpowers:executing-plans按任务执行；本次用户明确要求修订计划，暂不使用full fine-tuning，所有比较组统一使用LoRA。

**Goal:** 使L09的M0/M1/M2/M3及后续纳入同一主比较的对照共享LoRA训练配方，并让S29使用对应LoRA checkpoint。
**Architecture:** 沿用现有六通道π0.5、Flow Matching、真实数据和评测接口；冻结预训练主干，只训练两套LoRA、六通道输入层和既有动作/时间投影。AdamW和EMA仅维护可训练参数，冻结权重通过已验证基座身份恢复。
**Tech Stack:** 当前OpenPI/Flax NNX、JAX、Optax、Orbax及既有真实数据审计。
**Spec:** docs/metric_cue/RESEARCH_PLAN.md；protocol.json revision5。

## 共同配方与公平性

| 项目 | 四组共同规定 |
|---|---|
| 组别与种子 | M0/M1/M2/M3，各42、43、44，共12次计划训练 |
| 微调方式 | LoRA；本轮full fine-tuning暂停 |
| PaliGemma | gemma_2b_lora，attention和FFN，rank16、alpha16 |
| Action Expert | gemma_300m_lora，attention和FFN，rank32、alpha32 |
| 初始化 | LoRA A沿用normal std0.01；B统一置零；同一seed在各组共享完全相同的初始可训练参数 |
| 视觉输入层 | 四组都训练PaliGemma/img/embedding的kernel和bias，完整六通道kernel采用[W_RGB,0]初始化 |
| 其他可训练层 | action_in_proj、action_out_proj、time_mlp_in、time_mlp_out的kernel和bias |
| 冻结参数 | 上述白名单之外的全部参数，包括SigLIP其余层和LLM原始权重 |
| 批次与预算 | effective batch8、microbatch1、累计8、30000 updates、相同采样顺序和随机种子 |
| 优化器与EMA | 原AdamW、warmup1000、peak5e-5、end5e-6、decay30000、EMA .999；状态仅覆盖可训练参数 |
| 数据与接口 | 相同432 episodes/52970帧、manifest/统计量/Tokenizer身份、六通道结构、TTFF mask、action horizon10 |
| checkpoint与评测 | 保存10000/20000/30000，统一使用30000；相同500个配对初始状态和原SR/CI口径 |

[Code] CPU上对真实完整LoRA模型执行jax.eval_shape，按上述白名单得到总参数3,404,098,832、可训练参数53,509,280（约1.57%），其中LoRA为49,987,584、输入层为1,355,904。其余2,165,792为动作/时间投影。原始记录outputs/metric_cue_pilot/L09_lora_plan_r5/parameter_shapes.json；这是结构核验，不是GPU训练或峰值内存测量。

[Code] 六通道新增Cue权重为零，输入层必须可训练；M0也采用相同的输入层白名单。M0空白Cue分支梯度应为零，M1/M2/M3有可见提示的真实批次应产生有限、非零Cue梯度。不能直接沿用只冻结LLM、仍全量训练SigLIP的默认filter。

[Inference] 按FP32数组计，冻结参数加可训练参数、梯度、AdamW两状态和可训练EMA约13.48 GiB，未包含激活、载入临时副本、编译和数据缓存。LoRA验证以CPU可用64 GiB及原双卡每卡28 GiB为起始预算；正式门槛取max(64 GiB, 四组完整smoke最高CPU峰值的1.25倍向上取整)。旧full-FT的160 GiB门槛及GPU训练receipt不直接作为LoRA资源结论。

- 所有进入同一主结果表的Baseline、方向、公制标尺、Echo及后续固定像素等对照，均采用同一LoRA、白名单、dtype、预算和checkpoint选择；不混入历史full-FT结果。
- 动态资源适配优先调整缓存和设备选择，保持effective batch8。需要改有效batch时，全部比较组采用同一新版本并重新开始，不在组间独立降低batch、rank或训练预算。
- S29使用L09同一LoRA配方的checkpoint；位置场景有效性仍是评测前提。
- 原full-FT的13次、1547次记录、失败输出及等待记录保留归档，不能作为LoRA训练或恢复状态；新的run ID包含lora，禁止跨配方resume。

## 发布状态

本次完成计划与冻结协议修订，LoRA运行时尚待实现和验收。training_released=false、evaluation_released=false；旧full-FT等待PID24503已取消，取消前jax_imported=false、未开始训练。不得仅修改finetuning文字后沿用实际仍全量训练的入口。

revision4原计划、协议、run matrix、release和配置原字节保存在revisions/r4/frozen_inputs.tar.gz。几何/数据/场景证据可在身份不变时复用；LoRA训练参数、GPU更新、冻结权重不变性、delta状态恢复和资源证据必须重新取得。

## 实施任务

### 任务1：模型、初始化和可训练参数合同
文件：src/rulervla/adapters/metric_pipeline.py、src/rulervla/training/metric_workflow.py、tests/test_metric_lora_contract_real.py。

- [ ] 先用真实完整模型结构写失败测试：两套LoRA rank/alpha正确；四组白名单、叶子名称/shape/dtype及53,509,280参数计数一致；输入层和四个投影存在，其余主干被冻结。
- [ ] 让make_training_config显式读取finetuning和两套LoRA变体；新增严格白名单，未知模式或多余可训练参数立即失败。
- [ ] 基座载入只允许LoRA新参数缺失；B置零、六通道[W_RGB,0]与相同seed初始化按合同执行，记录基座/adapter/输入层身份。
- [ ] CPU结构测试通过后提交；不能用小模型或mock代替完整模型。

### 任务2：梯度、AdamW与EMA内存
文件：src/rulervla/training/accumulation.py、src/rulervla/training/metric_workflow.py、tests/test_metric_lora_training_real.py。

- [ ] 失败测试先覆盖：optimizer/gradient/EMA只含白名单参数，冻结参数不产生对应AdamW状态；八个microbatch只进行一次clip/AdamW/schedule/EMA更新。
- [ ] EMA .999只存可训练参数，冻结部分由基座恢复；避免复制冻结主干和全量EMA，保持原Flow Matching目标。
- [ ] 用真实完整模型和同一批真实动作完成至少两次GPU更新，核对冻结权重逐值不变、LoRA和输入层的有限梯度。B零初始化时首步A梯度可能为零，不以首步A未变化判错。
- [ ] M0与有可见Cue的M1/M2/M3分别检查预期梯度；记录实际CPU/GPU峰值和数据缓存设置。

### 任务3：可恢复checkpoint与配方身份
文件：src/rulervla/training/metric_workflow.py、scripts/metric_cue_serve.py、tests/test_metric_lora_checkpoint_real.py。

- [ ] 失败测试先覆盖：缺少LoRA/输入层/AdamW/EMA/step、错误基座、错误rank/filter，以及full-FT checkpoint均不能恢复成LoRA。
- [ ] 使用Orbax保存基座来源引用、全部可训练参数、AdamW、可训练EMA、step、采样/调度配置和manifest；大型冻结权重不重复写入adapter状态。
- [ ] 恢复时核验基座文件身份，恢复完整推理模型，并对可训练状态、优化器、EMA、step和冻结基座逐一校验SHA-256。
- [ ] 以真实完整checkpoint执行保存、CPU恢复、继续一步和推理入口验证，禁止只检查文件存在。

### 任务4：正式入口与四组共同配置
文件：src/rulervla/training/metric_protocol.py、scripts/metric_cue_train.py、scripts/metric_cue_launch.py、configs/metric_workflow_lora.yaml、tests/test_metric_formal_config_real.py。

- [ ] 失败测试先覆盖：旧full-FT配置/身份拒绝；12个LoRA run ID及相同共同配方正确；训练与评测仍受各自gates限制。
- [ ] 新增明确的LoRA工作流配置，使用已验证的yuqiuting路径和旧manifest路径映射；正式run ID为L09_spatial_<arm>_lora_seed<seed>。
- [ ] 四组统一使用512 MiB的episode缓存上限和同步数据读取；Python及编译热缓存放在运行树被忽略的本地build/metric_lora_cache目录，数据、权重和checkpoint持久保存到NAS，检查本地目录可写及空间余量。
- [ ] start_metric_cue_amax.sh明确改用configs/metric_workflow_lora.yaml，并在模型载入前拒绝full-FT或缺少微调模式的配置；旧等待启动器不复用。
- [ ] 资源preflight使用经新smoke校准的LoRA门槛；等待期不初始化JAX，启动后继续不按资源阈值主动终止训练。
- [ ] 训练/服务入口读取同一LoRA配方、白名单和delta checkpoint身份，避免离线训练与在线评测模型不同。

### 任务5：真实验收、发布和执行
文件：experiments/metric_cue_v1/release.json、run_matrix.json、receipts/、scripts/start_metric_cue_amax.sh及handoff.md。

- [ ] 四组运行相同真实输入、零Cue初始化等价、LoRA梯度/优化器、冻结参数不变性、完整状态恢复与资源smoke，保存全部原始receipt。
- [ ] 依据四组最高峰值冻结共同资源门槛；数据、几何、输入和场景的既有证据逐项核对身份，S29评测继续关闭。
- [ ] 验收通过后提交冻结运行时，才将LoRA训练发布为true；先运行M2 seed42，再按run matrix完成全部四组和三个种子。
- [ ] 每次以实际更新、manifest与checkpoint确认进度；报告全部对照、macro/micro SR、配对差异、CI与工程失败。

## 自查

计划分别覆盖了零初始化Cue学习、M0零Cue梯度、LoRA首步A零梯度、默认视觉filter过宽、EMA/optimizer误包含冻结主干、full-FT混用、不同rank/批次/更新预算、NAS路径迁移及checkpoint只有目录没有完整状态等情况。本轮没有将上述待实施任务报告为已实现或已通过。
