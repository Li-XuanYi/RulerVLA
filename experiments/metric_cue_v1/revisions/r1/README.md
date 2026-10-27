# L09 revision 1 冻结输入

frozen_inputs.tar.gz 保留配置修复前的原始字节，SHA-256：

4b85026ac19c6c2ab63a1327a538336b893288ef9d6f5fd96a2c6de01ad64413

内容为四组 configs/rulervla_metric_cue*.yaml、configs/rulervla_metric_workflow.yaml，以及 experiments/metric_cue_v1 下的 protocol.json、run_matrix.json、release.json。原协议 SHA-256 为 63498060346a1bad1c460aadacc256215c98f48915cda2c2bb6bcb183e142632。

该修订采用旧 ±30 cm 字段、batch32与30k更新。当前入口使用 revision2 的正轴0～15 cm配置。旧产物继续使用其匹配的冻结源码和原输入合同；此归档不提供当前程序的旧格式兼容转换。
