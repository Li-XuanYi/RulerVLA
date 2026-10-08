# 统一 LoRA 协议（2026-10-08）

用户授权将当前主比较与补充小实验全部从 full fine-tuning 改为 LoRA。
48 个主比较 run、3 个 P0 小实验及 2 个条件性 P1 配置均统一修改。
旧 A0/A1/A2/A3、B/G/S 实验及旧 full-FT 记录保留，不改写成 LoRA 结果。

## 共享设置

- 同一官方 pi05_base；重新初始化，不接续本轮 full-FT 状态。
- 官方 `gemma_2b_lora`（rank=16, alpha=16）和 `gemma_300m_lora`（rank=32, alpha=32）。
- 冻结范围采用 `Pi0Config.get_freeze_filter()`：冻结 LLM/action expert 的非 LoRA 基础权重。
  视觉编码器、投影等非 LLM 模块依官方 recipe 继续训练；不宣称只有 adapter 可训练。
- 同一训练数据、seed、AdamW、EMA、LR、scheduler、action horizon、四输入槽与正确 orientation。
- micro-batch=1、梯度累积=32、effective batch=32。
- 原训练预算保持：CameraXYZ P0 10k；ActionCoord/Ruler P0 30k；主比较每模型 30k。
- 100 updates 首存，此后每 1000 updates 保存；保留最近两份和 10k/20k/30k 里程碑。
- 新 run id 后缀 `_lora`，新输出目录隔离，不混合 full-FT 与 LoRA 结果。

## 开跑门槛

重跑真实 GPU 输入 gate。每个小实验先完成 2 updates smoke：官方 base 加载、
LoRA 参数存在、冻结参数前后 SHA 一致、loss/gradient 有限，以及完整
optimizer/EMA/weights 保存和恢复后逐叶 SHA 一致。
主比较仍须通过新的 LoRA smoke，禁止用旧 full-FT smoke 放行。

先完成小实验。主比较暂不启动。显存消耗以实际 LoRA smoke 为准，不保证单卡 20GB。
