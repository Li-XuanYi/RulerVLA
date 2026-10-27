# L09 revision3：有效 batch8

protocol.json revision3 将所有 L09 正式运行统一为有效 batch8、microbatch1、梯度累积8、30000次更新，适用于 M0/M1/M2/M3 和 seeds 42/43/44。学习率、优化器、full fine-tuning、双 GPU FSDP、CPU AdamW/EMA、数据、评测场景与十二个 run ID 保持原协议定义。

revision2 的 batch32 协议、运行矩阵和 release 状态原样归档于 revisions/r2/frozen_inputs.tar.gz；归档 SHA-256 为 3b70b8cca73abef1c02d1a4389f6ca93d6a076dc85b145f838b38ac95043dc92。

当前 protocol SHA-256 为 a5188add4e9839c17de991a8062fb780f50c8ca3a7f967cafcd91ad9cc2b80e4；run_matrix.json 与 release.json 指向相同修订和哈希。训练与评测仍未发布，GPU 更新、峰值显存、吞吐量、完整状态恢复和 SR 均未验证。
