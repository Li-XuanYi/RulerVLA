# 145节点首次NCCL通信失败与隔离验证

- [Experiment] 冻结提交1c06f6df8f4787e0c47cafea0609be8297fe4917的M2在gpu_zero_cue_equivalence阶段自然Aborted，控制1693357、训练进程1693359均已不存在；原status的running是陈旧记录。0次optimizer updates，没有metrics；checkpoint目录存在但为空，没有可恢复CP2。未主动停止训练。
- [Experiment] 原始目录为NAS metric_cue_pilot/L09_lora_r5_smoke/smoke_cpu48_r6_node145，launch.log SHA为5d42c23b70bcc0322dc8a0b89b81c6946f0857a67508a74470977d3ee7019a64。failure_observation_20261030T090157Z.json保存实际进程缺失及原status。
- [Experiment] 准入时CPU可用423.44690704345703GiB，GPU3/5分别空闲49990/48870MiB；每卡原生BFC pool_bytes为27735597056，达到共同24GiB下限。随后NCCL2.26.2+cuda12.2首次通信发生CUDA error2 out of memory，XLA因其他副本尚未结束而主动abort。最终显存峰值和失败时其他分配未知，不能归咎其他作业，也不能把前面的rendezvous等待提示当作已确定死锁。
- [Code] CUDA12.9.82的cuobjdump实际核验：原2.26.2库原生cubin最高SM90；官方2.28.9库含SM120原生cubin。NVIDIA说明旧PTX可在新架构运行时编译，因此缺少SM120原生cubin不等于完全不支持Blackwell：[Blackwell Compatibility Guide](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html)。
- [Hypothesis] 原库在Blackwell的首次通信需要PTX运行时编译或其他额外资源，可能与当前OOM有关；日志不能单独证明该因果关系。选择只改变实际NCCL库版本，保留模型、数据、LoRA数学及共同资源门槛，并增加INFO通信日志进行隔离验证。

## 已核验的依赖

- [Experiment] NVIDIA PyPI官方wheel nvidia_nccl_cu12-2.28.9-py3-none-manylinux_2_18_x86_64.whl的SHA485776daa8447da5da39681af455aa3b2c2586ddcf4af8772495e7c532c7e5ab已验证；解包库SHA1792291e26d2b27fe51e59cf4dbd361dbfa84cbfe8ee8196c087c7ebf5146d48，实际ncclGetVersion为22809。原库SHAe4a7aee9c3eecf53fac780441d2f03b578ab8db8874b71f8e391bcec7adb2899，版本22602。
- [Experiment] wheel、来源JSON、RECORD核对、旧/新cubin列表及nccl2289_verified_artifact.json均保存在NAS metric_cue_pilot/L09_lora_r6_node145_deployment。新库解包在运行树ignored build/nccl2289；没有替换共享venv中的库。
- [Code] 两个启动器支持METRIC_NCCL_ROOT，保留原默认路径；每组及正式启动前记录全局实际加载的NCCL版本及库SHA，METRIC_NCCL_SHA256存在时不匹配即失败。NCCL_DEBUG允许显式INFO，其余训练代码保持原字节。
- [Code] 真实CPU回归RED为2 failed38.35s（原正式闭门/旧full-FT配置拒绝条件成立，但缺少实际库指纹），GREEN为2 passed2.47s，核实新库版本22809及SHA。默认路径仍读取22602，以旧库真实SHA校验新库会立即拒绝；ruff/diff/bash语法及只读复核通过，无Critical/Important。该profile来自独立CPU进程，实际训练通信版本仍须由每组NCCL INFO日志确认；核心release audit尚未自动比较新profile，放行前须逐组核对指定SHA及实际通信版本，正式启动也须保持一致。

## 执行与验收

1. [Code] 真实CPU加载器检查使用官方2.28.9库，分别运行正式闭门和历史full-FT配置拒绝路径；禁止mock、替身解释器或虚构模型。RED须真实失败，GREEN验证实际库版本/路径/SHA和原拒绝条件同时成立。此检查不代表GPU通过。
2. [Experiment] 确认所有本次smoke/train/等待器均已退出后，前向部署干净提交。新四组统一使用METRIC_NCCL_ROOT=/home/yqt/XuanyiLi/rulervla-metric-lora-r6-runtime/build/nccl2289/nvidia/nccl/lib及METRIC_NCCL_SHA256=1792291e26d2b27fe51e59cf4dbd361dbfa84cbfe8ee8196c087c7ebf5146d48，NCCL_DEBUG=INFO。
3. [Experiment] 唯一新队列目录为smoke_cpu48_r6_node145_nccl2289，顺序M2/M0/M1/M3；每组均记录库指纹和独立NCCL日志。LoRA、batch8/micro1累计8、30k、CPU48GiB、双卡各28672MiB、BFC26GiB目标及初始原生pool至少24GiB全部不变。
4. [Experiment] 所有组需重新完成完整GPU验收，正式训练使用相同NCCL路径及SHA并核对实际启动日志；protocol6原字节保持不变，训练/评测发布关闭。只有完整GPU证据和聚合审计可放行，不能把原失败、依赖结构或CPU加载器检查计为GPU验收成功。若相同OOM再现，先分析新INFO日志，不能无依据连续重试。
