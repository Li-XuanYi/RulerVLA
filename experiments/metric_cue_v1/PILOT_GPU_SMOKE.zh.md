# L09 工程 pilot：batch8 连续两次 GPU 更新

[Code] 配置为configs/metric_workflow_pilot.yaml：有效batch8、双卡FSDP、microbatch1、累积8、两次optimizer update、full fine-tuning、CPU AdamW、EMA .999，采用revision3的warmup-cosine。后续初始准入为每卡28672 MiB、CPU160 GiB，依据完整更新和保存恢复峰值留出余量。

[Code] CPU optimizer 复用输入状态和梯度缓冲，EMA 初始缓冲独立；完整状态恢复使用抽象模板并在恢复前释放旧状态。运行中按空闲量与本进程占用之和检查预算，初始预检仍核验实际空闲量。

[Inference] 使用 M2、seed42 的真实 LIBERO 样本核验 GPU 零 Cue 等价、batch8 更新、非零 Cue 梯度、完整状态保存恢复。工程 pilot 不计入正式十二个 run，不能给出 SR 结论。正式训练及评测继续受 release.json 控制。

[Experiment] 当前源码8a699dc已在vla_7_6000 GPU1/3完成全部工程smoke：GPU零Cue等价、两次更新、非零且有限的Cue梯度、完整状态保存恢复、207个数组签名一致、恢复至CPU。CPU峰值121.304 GiB，GPU每秒采样峰值21662/21660 MiB。receipt为receipts/L09_gpu_smoke_node147_20261027.json，原始目录为outputs/metric_cue_pilot/node147_M2_batch8_8a699dc/。41项CPU回归通过；旧恢复失败及其checkpoint保留。

[Experiment] 先前 L09_P0_batch8_step1 在源码身份检查处停止，没有加载基座或执行更新；保留原始记录。当前部署要求先提交所有实际导入的运行时文件，保持 source_provenance 的严格检查。

[Code] 当前已验证主机为vla_7_6000（ZjuServer147），独立目录/home/lixuanyi/XuanyiLi/metric_cue_runtime/RulerVLA，Python为同级environment/.venv/bin/python。outputs与基座权重位于/home/Dataset2/yqt/rulervla/metric_cue_runs。需使用PLATFORM_BLACKWELL_R1.zh.md指定的NCCL LD_PRELOAD与ptxas PATH，并显式传入--gpu-indices 1 3；每次启动重新准入。

[Inference] 第二次更新耗时80.35秒，仅含一个编译后的更新观察，未建立稳定吞吐。按此值外推30k约27.9天，且未计批次准备和正式checkpoint时间；共享GPU、CPU及NAS负载会影响该数值。正式长程训练须另行评估持续吞吐。
