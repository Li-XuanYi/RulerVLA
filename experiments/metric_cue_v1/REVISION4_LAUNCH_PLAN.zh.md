# L09 revision4 实施计划

> 使用superpowers:executing-plans逐项执行；规格为L09_TRAINING_SCOPE_PROPOSAL.zh.md，用户已在当前会话明确选择修订协议并先训练L09。

**Goal:** 启动L09_spatial_M2_seed42的batch8、30k正式训练，同时保持S29评测关闭。

**Architecture:** 保留launch_gates作为训练准入，新增evaluation_launch_gates供评测入口使用。训练配方、数据、模型与现有GPU验收沿用；原字节归档revision3。单次启动器只在当前协议和实时资源均通过后执行正式CLI。

**Tech Stack:** Python3.11、JSON、现有JAX/Orbax训练入口、真实NVIDIA资源查询。

## 全局约束

- full fine-tuning、batch8、microbatch1、累积8、EMA .999、原warmup-cosine、双卡FSDP、CPU AdamW、30k不变。
- position_scene_validity_lock保持false，evaluation_released保持false；训练不读取S29场景。
- 保留历史protocol、receipt、smoke和失败输出；不停止其他任务，不使用mock，不读写/tmp。

## 核查重点

训练须仍核验数据和完整状态receipt；评测flag误开也须被S29 gate阻止；资源不足不得导入JAX或创建训练状态；运行源码必须已提交；出现运行错误应明确退出。

## 执行步骤

- [x] 修改tests/test_metric_formal_config_real.py，使用真实数据审计及GPUreceipt证明训练准入；验证评测在S29未通过时拒绝启动。先运行并记录预期失败。
- [x] 归档revision3 protocol/run matrix/release，更新revision4及对应SHA；更新metric_protocol.py的评测检查并接入metric_cue_eval.py、metric_cue_serve.py。
- [x] 运行真实配置、完整模型静态形状、内存与渲染回归；运行两个真实评测CLI确认在CUDA导入前拒绝请求。
- [x] 提交和部署冻结源码。运行单次正式启动器，实时检查资源，保存身份和原始日志；实际训练出现更新后才报告已启动。
- [x] 更新run matrix、registry和主仓库/开发树handoff，记录实际进程、更新数和剩余评测条件。
