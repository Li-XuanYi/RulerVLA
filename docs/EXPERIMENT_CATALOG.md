# 实验总目录（2026-10-07）

## 命名和使用规则

统一使用 **`S编号_中文简称`** 或 **`L编号_中文简称`**，中文简称最多五个字。

- **S：小实验**。机制验证、开发 pilot、消融、诊断、输入/几何/保存验收。分类依据研究用途，不把“训练达到 30k”自动当作正式 benchmark。
- **L：大实验**。冻结的正式训练与正式批量评测，包括早期单任务正式评测；只有 L06 才是四套 LIBERO 主比较。
- A0–A3、B0–B5、R2/R3 是历史 **方法/arm 标签**，保留 checkpoint 身份；实验标题统一改用 S/L。
- 原始权重、日志、排除规则和正在运行的路径保留。新名称与旧名称通过 `experiments/experiment_registry.json` 对应；不移动运行目录、不覆盖失败证据。
- 完成、部分完成、失败、待训练、计划中分别记录。工程验收通过不等于策略成功；旧协议结果不并入新协议主表。

## 大实验 L

| 新名称 | 旧标识 / 做什么 | 当前结果或状态 |
| --- | --- | --- |
| **L01_标尺微调** | E1 A0/A1/A2，Spatial 三臂 LoRA | 三组独立 pi05_base → 30k，batch 4；已完成、归档。不是当前 full-FT/batch32 主协议。 |
| **L02_位移评测** | E1 corrected-orientation，三臂目标 XY shift | 567 rollouts：A0 181/189，A1 173/189，A2 186/189。旧朝向错误的 formal_567 是同实验无效 attempt，保留审计、不用于比较。 |
| **L03_像素微调** | A3 Pixel Ruler | 30k LoRA，训练集刻度中位数固定为 12.69/17.65/14.17 px；完成、归档。 |
| **L04_像素评测** | A2/A3 corrected ID | A2 186/189，A3 首次 180/189；后续景深零位移复测 A3 178/189。保留复现波动，不挑成绩。 |
| **L05_景深对照** | A2/A3，五档光轴相机位移 | 1890 raw JSON/NPZ 对已完成归档。四档 OOD：A2 729/756、A3 725/756；真实 metric sparse ruler 没有稳定的景深优势证据。 |
| **L06_四套主比** | Main Comparison v1 | Spatial/Object/Goal/Long × 四方法 × 三 seeds = 48 模型；full FT、30k、effective batch32。Spatial Baseline/AimBot 正在训练；其他训练在资源队列，无完整主表。 |
| **L07_扰动主比** | Main LIBERO-Plus v1 | Level2、七类别，正式协议 2202 条任务/训练 seed；待相应 L06 checkpoint。历史单任务 Camera pilot 不能替代此主表。 |

L02 支持 A2 优于 A1 的配对证据；A2 对 A0 只有较高点估计，scene CI 触及零。L05 是负消融结果，不等于否定密集 ActionCoord。

## 历史小实验 S

| 新名称 | 旧标识 / 内容 | 已查证状态、结论和处理 |
| --- | --- | --- |
| **S01_朝向排错** | E1 初期全零 SR 排查 | simulator RGB 朝向错误已修复；无效结果隔离。沿用正确链路，不重新跑旧错误流水线。 |
| **S02_权重核验** | pi05_base / 数据 / A0–A3 smoke | 官方 base、Spatial 432 episodes/52970 frames、最终 29999 身份记录可复用；不下载或训练新的官方 base。 |
| **S03_相机试点** | 旧 LIBERO-Plus Camera smoke/pilot | 三臂 smoke 各 0/4；pilot 各 0/40。仅旧诊断记录，cue/环境路由与主协议适配未得到等价验收，不用作正式方法比较、不重跑。 |
| **S04_相机偏移** | 标准 LIBERO yaw+15° pilot | 现存 40 条/arm、零工程错误：A0 13/40，A1 6/40，A2 5/40；单任务旧 exploratory 条件，不扩样、不充当 L07。 |
| **S05_位移动作** | M1，目标位置→首次 action chunk | 直接复用 L02；A2 未显示稳定方向/单调优势。动作命令不是米，最近接近点不是实际抓取点。 |
| **S06_错尺干预** | M2，A2 tick ×0.5/.75/1/1.25/1.5 | 严格回放 50 scenes，正确输入 hash 与 E1 一致；改尺会改动作，但没有对称连续的 metric 调节证据。初版哈希不符 attempt 不纳入结论。 |
| **S07_深度补录** | SpatialCue depth source / replay / state8 / K | GT optical depth、原动作匹配、方向与半像素中心工程证据可复用；当前四套 1693 episodes 的数据准备已完成。 |
| **S08_稠密试点** | V02 B0/B1/B2/B3 | 5k full FT，warmup10k，10 scenes×4 shifts：RGB 14/40、Depth 16/40、CameraXYZ 27/40、ActionCoord 17/40。旧 pilot 不支持 ActionCoord，不能推断合理训练预算下谁更优。 |
| **S09_组合试点** | V02 B4/B5 | 5k full FT：CameraXYZ+Ruler 18/40，ActionCoord+Ruler 14/40；差 −10pp，scene CI [−22.5,+2.5]pp。早期保存阻断随后修复；最终 5k checkpoint/80 raw 对可复用为历史证据。 |
| **S10_组合加训** | “B4/B5_lora_30k_20261005” | 两组均 failed、2001 条 metrics，ENOSPC。目录虽写 lora，实际配置 Nothing() freeze filter、非 LoRA gemma variants，即 full FT；无30k最终结果。不得称已完成 LoRA。 |
| **S11_同步复核** | V03 R2/R3 | 正确同帧 RGB-D/geometry，warmup500/5k。最新 R2 完成5000且reload_verified；R3仅4241条metrics，进程−11退出。旧“未跑”报告已过时；无完整配对策略结果，不混用两个预算。 |
| **S12_位置预检** | V03 task2 location split | 46 demonstrations 的初始目标支持为一维；10 ID+10 interpolation+10 extrapolation 通过 scripted feasibility。属于几何可行性，非 policy SR。 |
| **S13_锚点核验** | V05初始 GT-anchor / V03 parity | 50帧几何与接口核验，shared translation 修正可恢复；没有策略级效果结果。 |
| **S14_平移修正** | G1 | 800/800 接受，最大恢复误差0.493mm；限定共享平移误差。 |
| **S15_锚数边界** | G2 | 1–2锚点0/300接受；仍有9个3/4锚点已接受结果超过固定2mm误差门槛。 |
| **S16_离群稳健** | G3 | Huber相对mean的偏差误差−2.944mm，episode cluster CI [−2.989,−2.902]mm；仅声明已测离群模型。 |
| **S17_旋转边界** | G4 | 100%接受，但远处残留误差42.162mm；局部平移拟合无法保证旋转错误的全场恢复。 |
| **S18_深度失真** | G5 | 深度噪声/缺失逐条件coverage与acceptance已归档；一致性检测不是深度重建。 |
| **S19_局部偏差** | G6 | anchor-only bias 对原干净场景引入7.392mm RMS；高score不能证明全场正确。 |
| **S20_目标偏差** | G7 | target-only误差7.603→7.854mm，score不变。G系列严格门槛总决策NO-GO，停止扩展此分支。 |
| **S21_统一验收** | Main full-FT smoke / 40-task preflight / inference API | 最坏四槽 full-state 保存恢复通过；四方法接口动作10×7有限；三套新数据cue/统计/视觉审核通过。是L06工程验收，不是正式SR。 |

## 补充小实验（按优先级）

| 新名称 | 问题 | 预算 / 复用 | 状态 |
| --- | --- | --- | --- |
| **S22_表示对照** | 无Ruler的CameraXYZ vs ActionCoord | 两臂都比较精确10k快照、warmup1k，同一30k cosine曲线前10k；ActionCoord快照来自S23。额外独立训练CameraXYZ一个模型。 | P0，协议冻结；资源/逐臂smoke/评测合约门槛前不训练。 |
| **S23_四格消融** | RGB / Ruler / ActionCoord / Ruler+ActionCoord | 30k full FT；复用L06同suite/seed的Baseline与RulerVLA。只新增Ruler-only和ActionCoord-only两个模型。 | P0，协议冻结。 |
| **S24_坐标拆解** | CameraXYZ / CameraEE / ActionXYZ / ActionCoord | 10k同预算，复用S22两臂快照；条件成立后新增CameraEE、ActionXYZ。 | P1，条件执行，不立即排训练。 |
| **S25_输入干预** | zero map / remove ruler / swap x-y / ruler ×.5/×2 | 同一完整RulerVLA checkpoint、相同scene和推理noise；无需重训。 | 待L06完整RulerVLA checkpoint；不能拿旧B5充当主方法。 |
| **S26_表示核验** | 四格输入与坐标变换是否公平、正确 | CPU，10任务50真实训练帧；四槽/共享编码/共同mask；主实验两端逐字节一致。 | 本轮已完成：误差<1e−15m，抽样各表示裁剪率0；零train/inference/rollout。 |
| **S27_旧案交互** | 已有B2/B3/B4/B5补充旧协议的交互描述 | CPU，160原raw；跨run配置与state/RGB配对通过。 | 本轮完成：Ruler在Camera下−22.5pp，CI[−40,−7.5]；在Action下−7.5pp，CI[−25,+7.5]；交互+15pp，CI[−5,+32.5]。旧不足预算结果，不能替代S23。 |

详细补充协议：`docs/SMALL_ABLATION_PROTOCOL.md`。新的小实验产物统一存于 `outputs/small_ablation_v1/S编号_中文简称/`。

## 复用和结论边界

1. S08/S09/S10/S11与L06的训练数据、输入槽、预算或微调方式不同，不可替代新协议checkpoint。
2. L01–L05是稀疏ruler旧研究阶段；作为历史结果完整保存，不为追求新假设重跑。
3. S22/S23可共享 **ActionCoord的前10k快照**，但只有把CameraXYZ的scheduler也冻结为相同30k曲线前10k才成立。不得把10k cosine和30k cosine的checkpoint直接比较。
4. 相同scene的多shift/multiple policy-noise trials不是独立训练replicate。单training seed的小实验只估计场景条件下的变异，不能写成三seed稳健结论。
5. 输入干预引起test-time distribution shift，只能支持使用/敏感性证据；组件训练收益由S23来检验。

## 权威证据入口

- `/home/yqt/XuanyiLi/openpi0-aimbot/third_party/AimBot/Xuanyi/reports/E1_final_results_oriented.md`
- `/home/yqt/XuanyiLi/openpi0-aimbot/third_party/AimBot/Xuanyi/reports/A2_A3_depth_final_20261001.md`
- `/home/yqt/XuanyiLi/openpi0-aimbot/third_party/AimBot/Xuanyi/reports/e1_mechanism_20260930/`
- `/home/yqt/XuanyiLi/actioncoord/reports/V02_pilot_results.md` 和 `V02_pilot_training.md`
- `/home/yqt/XuanyiLi/actioncoord/fixes/framework_fix_20261002T151206Z/repo/third_party/AimBot/Xuanyi/v02_actioncoord/reports/B4_B5_final_20261005.md`
- `/home/Dataset2/yqt/actioncoord-v02-20260928/pilot_B{4,5}_lora_30k_20261005/run_metadata.json`
- `/home/Dataset2/yqt/actioncoord-v03-20261003T164806/background/20261004T070106Z/status.json`，同launch的R2/R3 metrics
- `reports/G_SERIES_SUMMARY.md`、`docs/MAIN_COMPARISON_PROTOCOL_V1.md`、`reports/MAIN_PARALLEL_RELEASE_20261007.md`

本目录依据本轮只读核验快照；运行中进度以live status为准。尚未核实完整性的新结果不得标记为completed。
