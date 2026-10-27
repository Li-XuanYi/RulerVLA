# L09 训练范围修订（用户已确认）

[Code] 当前冻结protocol revision3（SHA-256 a5188add4e9839c17de991a8062fb780f50c8ca3a7f967cafcd91ad9cc2b80e4）将position_scene_validity_lock列入全部启动条件。S29清单仍为null，旧S28仅完成几何预览，尚缺遮挡、无碰撞抓取姿态、放置稳定性及配对重复的最终核验。

[Inference] L09训练读取已冻结的标准Spatial数据：432个episode、52970帧，不读取S29改变目标XY位置的场景。当前阻塞需要明确是否继续把S29评测场景验收作为L09训练前提。

## revision4范围

- 首次正式运行仍为L09_spatial_M2_seed42：full fine-tuning、batch8、microbatch1、累计8、30k updates、EMA .999、原warmup-cosine、CPU AdamW及双卡FSDP；模型、数据、种子、checkpoint选择与全部对照保持原定义。
- 训练启动条件保留真实输入、完整数据审计、GPU零Cue等价、真实更新、完整状态恢复、标准Spatial配对场景、模拟器接口和实时资源检查。
- 将S29位置场景有效性保留为评测条件，训练不使用S29位置或其策略反馈；评测发布保持关闭，场景冻结后才能运行标准策略评测与S29位置评测。
- 先按原字节归档revision3的protocol、run matrix及release，再版本化更新训练/评测条件并核验哈希与回归。position_scene_validity_lock继续为false，不填写虚构场景manifest。
- 通过预检后从已验证的pi05_base开始正式训练，保存源码、依赖库、数据及配方身份。工程smoke的两次更新不计入正式30k。
- 若保持revision3，须先完成S29的场景定义和全部验收，再开放训练；当前不启动正式训练。

[Scope] 用户已在本次会话明确选择“修订协议，先训练L09”。对应revision4现已实现；正式作业仍须通过实时资源预检。
