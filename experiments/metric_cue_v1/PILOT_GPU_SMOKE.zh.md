# L09 工程 pilot：batch8 连续两次 GPU 更新

[Code] 配置为 configs/metric_workflow_pilot.yaml：有效batch8、双卡FSDP、microbatch1、累积8、连续两次optimizer update、full fine-tuning、CPU AdamW、EMA .999。学习率与revision3的warmup-cosine相同。当前初始准入为每卡28672 MiB、CPU128 GiB，依据首轮完整更新实测留出余量。

[Code] CPU optimizer 复用输入状态和梯度缓冲，EMA 初始缓冲独立；完整状态恢复使用抽象模板并在恢复前释放旧状态。运行中按空闲量与本进程占用之和检查预算，初始预检仍核验实际空闲量。

[Inference] 使用 M2、seed42 的真实 LIBERO 样本核验 GPU 零 Cue 等价、batch8 更新、非零 Cue 梯度、完整状态保存恢复。工程 pilot 不计入正式十二个 run，不能给出 SR 结论。正式训练及评测继续受 release.json 控制。

[Experiment] 修复后的CPU回归41项通过，日志在outputs/metric_cue_pilot/resource_fix/restore_final.log。node147首轮已完成两次完整模型更新并保存checkpoint；完整恢复在旧Orbax参数处失败，修复后需重新执行全部流程。GPU通信与平台身份见PLATFORM_BLACKWELL_R1.zh.md。

[Experiment] 先前 L09_P0_batch8_step1 在源码身份检查处停止，没有加载基座或执行更新；保留原始记录。当前部署要求先提交所有实际导入的运行时文件，保持 source_provenance 的严格检查。

[Code] 用户同意考虑其他节点后，候选运行主机为 vla_6_6000（ZjuServer145），Python环境 /home/yqt/XuanyiLi/openpi/.venv/bin/python；JAX/JAXlib0.5.3、Flax0.10.2、Orbax0.11.13与主开发环境一致。使用同一Git提交的独立运行目录，不修改该节点已有RulerVLA工作区或共享环境。outputs和基座权重位于 /home/Dataset2/yqt/rulervla/metric_cue_runs，避免本地磁盘20GiB余量不足。GPU编号通过 --gpu-indices 显式指定，启动前重新准入。
