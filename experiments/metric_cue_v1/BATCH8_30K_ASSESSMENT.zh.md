# batch8、30k updates 可行性评估与协议决定

[Protocol r3] 依据本评估，L09 正式共同配方现采用有效 batch8、microbatch1、梯度累积8、30k updates。此决定同步适用于 M0/M1/M2/M3 与全部三个 seed。GPU 实际更新、吞吐量、峰值显存和 SR 仍未核验，release gates 继续关闭。

[Code/Inference] batch 指有效批量。当前训练器支持 microbatch1、累计8次梯度后执行一次 AdamW、学习率与 EMA 更新；双 GPU FSDP 分片参数和梯度，不会把该有效批量再乘二。八条真实动作输入和完整 π0.5 梯度累积的 CPU 静态形状检查通过。实际 GPU 更新、吞吐量、峰值显存与优化稳定性尚未验收。

## 数据与更新预算

数据身份沿用 L09：432 episodes、52970 个训练起始帧，动作 horizon=10，末尾沿用同 episode 的 repeat-last。起始帧以确定性打乱的完整 epoch 顺序抽取，恢复由 seed 与 update 决定。

| 条件 | 当前正式有效 batch8 | revision2 历史有效 batch32 |
|---|---:|---:|
| 优化器更新数 | 30000 | 30000 |
| microbatch | 1 | 1 |
| 每次更新累积数 | 8 | 32 |
| 起始帧样本抽取总数 | 240000 | 960000 |
| 等效数据遍历次数 | 4.5309 | 18.1235 |
| 1000 次 warmup 对应样本抽取数 | 8000 | 32000 |

[Inference] batch8 保留30k次优化器更新，梯度累积的前后向次数为当前batch32的四分之一。每次优化器与EMA更新、数据准备及通信等耗时仍然存在，不能直接推出四倍吞吐提升。相邻动作窗口存在重叠，上述抽取次数不表示独立轨迹数量。

## 内存与准入

[Code] 保持 microbatch1 时，模型参数、梯度累积器、CPU AdamW 的 mu/nu 与 EMA 形状不随有效 batch 改变。单个微批次的激活规模也保持。降低有效 batch 主要减少累积次数以及整批输入缓存。

[Inference] 仅计算四槽 float32[224,224,6] 图像张量，batch8约36.75 MiB，batch32约147 MiB，相差约110.25 MiB；此数不含模型权重、梯度、优化器、临时缓冲或激活，不能作为峰值显存估计。

[Experiment] 本日四组 CLI preflight 均通过配置读取，阻塞集中在 GPU2/5 每卡空闲32768 MiB门槛与CPU可用100 GiB门槛。M2记录的CPU可用63.09 GiB，launch_allowed=false；全部 preflight 的 jax_imported=false、gpu_job_started=false。降低有效 batch 不能据此解除现有资源条件。记录在 outputs/metric_cue_pilot/config_alignment_r2/preflight_M0 至 preflight_M3。

## 配方与结论边界

[Inference] batch8、30k 已作为当前 π0.5 适配协议采用，仍沿用 full fine-tuning、双 GPU FSDP、CPU AdamW/EMA 与 warmup-cosine 参数。相较历史 batch32 配方，每次训练的数据暴露量不同；目前没有数值更新或 SR 证据证明两种 batch 效果等价，也没有小 batch 下学习率稳定性的实测结果。

[Protocol r3] 当前正式协议固定有效batch8、microbatch1、累计8、30k更新；四组对照和三个seed共用该配方。学习率、优化器、full fine-tuning、双GPU FSDP、CPU AdamW/EMA及数据身份保持不变。batch32仅作为revision2历史条件保留。完整模型GPU更新和恢复检查仍需通过后续launch gates；训练与评测发布标志保持关闭。
