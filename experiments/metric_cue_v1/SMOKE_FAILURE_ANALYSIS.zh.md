# Metric Cue 冒烟失败原因核查

核查日期：2026-10-27（Asia/Shanghai）。工作树：vla_7_A6000 上 /home/lixuanyi/XuanyiLi/work/worktrees/rulervla-metric-cue。本轮只读取运行证据与源码，并以CPU执行源码身份诊断，没有启动GPU smoke、训练或rollout，没有修改模型、门槛或协议。

## 1. 最新记录停在资源准入

[Experiment] outputs/metric_cue_pilot/batch8_split_attempt_20261027/preflight_M0/status.json 记录 mode=preflight、state=blocked、jax_imported=false、gpu_job_started=false、formal_training_started=false。该目录只有预检产物，没有smoke更新记录。

| 资源 | 该次预检的可用量 | 配置门槛 |
|---|---:|---:|
| GPU2空闲显存 | 8221 MiB | 32768 MiB |
| GPU5空闲显存 | 23549 MiB | 32768 MiB |
| CPU可用内存 | 69.8140 GiB | 100 GiB |

两份数据manifest、归一化统计和基座metadata/receipt身份均通过该次检查。preflight.json SHA-256 为 f7bf016e330fd094fb66cba78eea8ac2c6499dc96388a27e1a751914f0905710。

[Code/Inference] scripts/metric_cue_train.py 在导入JAX之前读取准入结果并返回blocked。因此这次记录证明预设资源条件未满足，不能据此判定实际训练发生显存OOM、loss发散或梯度异常。32768 MiB/100 GiB是当前配置的准入门槛；完整更新的实际峰值仍未验收。

## 2. 当前重新运行还有源码身份错误

[Experiment] 本轮在 JAX_PLATFORMS=cpu、CUDA_VISIBLE_DEVICES为空的环境导入生产模块并执行 source_provenance(Path.cwd())，实际得到：

```text
ValueError: Runtime source is not committed: src/rulervla/geometry/camera.py
```

[Code] metric_workflow.py 的 source_provenance 同时要求导入的运行时代码已经纳入Git且内容等于当前提交。当前geometry/camera.py是未跟踪文件，其他运行时文件也有未提交修改。正式入口在资源准入后、创建完整训练状态之前调用该检查。

[Inference] 单纯释放GPU/CPU资源仍不足以完成当前入口；运行时代码需要按当前方法冻结并与run manifest的提交和哈希一致。保持源码身份检查有效，不以关闭检查取得表面通过。本轮仅定位问题，没有提交或改写代码。

## 3. 一次历史GPU smoke已通过零Cue等价检查

[Experiment] outputs/metric_cue_pilot/gpu_fsdp_0_2_reuse_rgb_20261026/run_manifest.json 指向提交 ef3c277582351b81d62e13fbe4313b1189c00fa0；其metric_workflow.py SHA-256为5e96667a2ef195ff6c6f63be2d9226d5363efc3ec54f150d5eb88f0be705f7fe，与该提交原始文件相符。运行使用M3、batch1、一次计划更新、GPU0/2。

该运行的gpu_equivalence.json已存在，SHA-256为ac9d40f117637288c009f1871ddf481bfd294aa5b83b448e255c7cefc3216f0e：

| 比较 | 最大绝对误差 | RMS |
|---|---:|---:|
| 视觉prefix | 0.03125 | 0.0000713932 |
| 动作 | 0.00422034 | 0.000930531 |

[Code] 对应版本在两项np.testing.assert_allclose均返回后才写入该JSON。验收使用atol=0.02、rtol=0.02，逐元素允许误差为atol+rtol×绝对参考值，不能只将视觉max_abs与atol作比较。

[Experiment] status.json仍停在stage=gpu_zero_cue_equivalence、optimizer_updates=0、metrics为空。对应训练入口在第一次trainer.step及Cue梯度检查之后才写入training_updates状态。该证据将中断位置限制在等价检查完成后、第一条成功更新记录写入前；它没有证明第一步内部的编译、反向传播或更新具体执行到哪里。

[Unknown] 在已检查的运行目录、相关日志目录和用户journal中，没有找到这次退出的原始stderr或退出码。旧handoff提到其他作业随后重新占用GPU，但缺少与本次终止对应的原始错误记录，无法进一步认定为CUDA OOM、CPU OOM、NCCL错误或外部终止。

## 4. 状态记录与证据边界

[Code] 入口没有统一的异常退出状态写入逻辑；进程异常退出后，最后一份status.json可能继续显示running。本轮进程查询未发现metric_cue_train.py训练进程，因此历史running字段不能作为当前运行证据。

[Evidence] 原始gpu_equivalence.json覆盖旧handoff中“没有取得等价检查数值”的概述。该数值只属于上述历史提交及输入身份，不能直接填入当前正轴标尺、batch8、revision3的release gates。当前完整GPU更新、checkpoint恢复和正式SR仍无通过证据。

[Code/Inference] 有效batch32降至8、同时保持microbatch1，主要减少累积次数和整批输入缓存；参数、梯度累积器、AdamW mu/nu和EMA形状保持。因此该改动不会按四倍比例减少持续内存占用，也不会自动解除上述准入与源码身份阻塞。
