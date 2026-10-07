# 补充小实验协议 v1（2026-10-07）

关联：S22_表示对照、S23_四格消融、S24_坐标拆解、S25_输入干预。本轮冻结设计并执行CPU核验，不占用现有正式训练资源去启动新的full-FT。用户后续“利用空闲资源做小实验”允许低成本诊断直接执行；full-FT仍须满足实测容量和验收门槛。

## 1. 三层问题和唯一变量

### P0：S22_表示对照

无Ruler、Anchor、Correction、Reliability。比较CameraXYZ-only与ActionCoord-only。

两者共用同一GT optical-Z、K/T、valid-pixel mask、RGB双视角、state8、语言、actions、四个active image slots、float32 metres/3.25编码和image preprocessing。agent辅助槽分别为：

`CameraXYZ = p_camera`

`ActionCoord = R_camera_to_action × p_camera + t_camera_to_action − EE_action`

LIBERO action basis是world XYZ，不能用EE quaternion旋转成另一个局部坐标系。此比较同时改变basis和origin，不把结果解释成单独axis效应。

**两臂分析快照均固定为9999，即精确10,000次optimizer updates。** 为减少新模型，S23的ActionCoord-only保存10k checkpoint供本实验复用；CameraXYZ独立从base训练到10k。两者使用同一warmup1000、decay_steps30000的cosine曲线前10k，避免scheduler混杂。这里不是“10k内完整走完cosine”，也不声称已经收敛。若要完整10k decay试验，则两臂都需要额外匹配训练，不能偷用S23快照。

主要量：同条件SR差ActionCoord−CameraXYZ及配对CI；min EE-target distance、D0−D5/D0−D10、真实contact event（若记录可靠）为辅助量。不能把closest approach叫作grasp error。

### P0：S23_四格消融

| | 无ActionCoord | 有ActionCoord |
| --- | --- | --- |
| 无Ruler | Baseline | ActionCoord-only |
| 有Ruler | Ruler-only | RulerVLA-main |

四臂均固定30k full-FT。Baseline和RulerVLA-main复用L06的Spatial seed42最终checkpoint；只新增Ruler-only及ActionCoord-only。与L06完全相同的训练frames、norm、seed、sample order、微批/累积、optimizer/EMA/scheduler、horizon、编码和runtime。

Ruler-only必须调用相同的nominal-EE renderer，保持agent辅助槽neutral，不能改成AxisGuide。ActionCoord-only只是关闭相同ruler绘制，不能改变depth/encoding或有效像素。

报告四臂SR及：

- 无ActionCoord时Ruler效应：B−A；有ActionCoord时：D−C。
- 无Ruler时ActionCoord效应：C−A；有Ruler时：D−B。
- 交互项：`D−C−B+A`。报告原始pp尺度和CI，不能仅用“D比A好”证明互补。

### P1：S24_坐标拆解

只有在S22按冻结主要评测显示正向且scene-cluster 95% CI下界>0，并且工程和support验证通过后进入训练。若没有该证据，报告阴性/不确定结果并停止自动扩展；不能选择另一个正向secondary取代触发条件。单seed触发只授权下一层pilot，不是稳健性结论。

- CameraXYZ：相机basis，相机origin。
- CameraEE：相机basis，减去同帧相机坐标EE。
- ActionXYZ：action/world basis，world origin。
- ActionCoord：action/world basis，减去同帧world/action EE。

CameraEE→ActionCoord只改变basis；ActionXYZ→ActionCoord只改变origin。四臂分析快照均10k、同一30k曲线前10k，无ruler。复用S22两臂；新训练CameraEE、ActionXYZ各一组。变换同源几何的等价性是设计门槛，不是VLA性能收益证据。

## 2. 共同训练约束和新增数量

- suite固定LIBERO-Spatial：episode1261–1692，432 episodes，52970 frames，10 tasks；official pi05_base独立初始化。
- training seed42；full fine-tuning，microbatch1×accumulation32=effective32；horizon10；7监督action分量，padding32。
- AdamW beta1=.9、beta2=.95、eps1e−8、weight_decay1e−10、clip1；EMA=.999；LR warmup1k、peak5e−5、end5e−6、decay30k。
- raw/stored raster合约、K−.5px、RGB实际bilinear与dense nearest、四槽共享encoder均按冻结L06 executable runtime。注意geometry JSON的历史Lanczos标签与实际bilinear不符；不照标签改变执行行为。
- 共享Spatial训练norm，SHA `b5e3345ee3d2a4698348ea8f8988031f2d003fdcd36c4716a7f7abc2d0834820`。
- S23保存full state/EMA/optimizer在5k/10k/15k/20k/25k/30k；S22/S24保留5k/10k。最终与分析快照不依据成功率挑选。
- frozen common scheduler的前10k复用是**参数更新前缀复用**；禁止A→B或B→C跨arm warm start。

P0新增**三个独立模型**：CameraXYZ-only(10k)、Ruler-only(30k)、ActionCoord-only(30k)。追加70000次模型optimizer updates；无需重训现有Baseline/RulerVLA。若P1触发再新增两个10k模型。多training seeds的确认性消融后续另立协议，不能把此seed42 pilot标成三seed实验。

`experiments/small_ablation_v1/protocol.json`和`run_matrix.json`是冻结设计规格，**不是已放行的训练命令或已经启动的模型**。还需把small adapter接入隔离训练入口并完成逐臂2–5-step backward/save/reload门槛，才允许释放相应训练。

## 3. 共同评测和复用门槛

主要评测固定Spatial每任务前20个official初态，10任务×20=200条件/arm。同一case跨方法/seed用同state、instruction、camera和policy-noise protocol。条件来自已保存L06 paired_scene_plan的前缀，必须在看到补充结果前保存精确case列表及SHA。

控制频率、最大步数、chunk执行与policy sampling设置沿用L06最终冻结评测合约；该合约尚须放行，不能在不同设置的旧raw中复用计分。L06标准500条件/arm若已完成，可以只读取前20状态的200条**严格匹配**结果，不再rollout；不迁移E1的target shift与11个排除到这张表。

Novel-location作为预先声明的独立secondary：优先复用S12的30个已通过物理可行性检查的task2场景，但必须先对当前46条task2训练轨迹的support与来源重验，重新锁定ID/interpolation/extrapolation类别和exact states。此门槛之前不能将旧46-demo位置分类直接声明适用于任意当前训练。

结果按task和scene配对。SR先给每task计数，再给macro与micro平均。CI以task为外层、scene为内层配对重采样10000次，bootstrap seed42；只有10tasks和一个training seed，要明确区间不包含训练变异。task2 novel-location按scene cluster，额外采样noise不能视为新增training seed。工程异常保留原condition；技术重试不按success筛选。

## 4. S25_输入干预（无需训练）

对象固定为L06 Spatial RulerVLA-main seed42最终30k checkpoint；不使用旧B5或官方base产生主机制结论。首个完整checkpoint到达后，可先做同协议工程smoke；主要机制结果仍使用固定30k。

保持同一原RGB、wrist、depth、K/T、EE、target、instruction、state与相同policy noise。每个条件包含正常输入、正常输入重复，以及以下一次只动一个变量的干预：

1. zero ActionCoord：agent辅助三通道变为neutral0。
2. remove Ruler：从同帧干净RGB重构，禁止涂黑已画线条。
3. swap x/y：只交换ActionCoord x/y通道，valid mask不变。
4. ruler ×0.5 / ×2：重新投影真实长度2.5/5/7.5cm或10/20/30cm，EE原点与ActionCoord不变。

先10任务各前3个official初态（30个冻结scene）进行同observation的首chunk probe（正常两次+五种干预=210次inference，零控制step）；固定scene/noise顺序前保存case manifest。报告10×7完整chunk、前五平移动作、gripper、动作差norm和signed alignment；明确command非米。只有原RGB可重构且正常重复通过噪声控制才执行。闭环辅助结果另存stage，不混入主SR或把test-time OOD下降等同训练组件贡献。

## 5. 本轮执行与资源门槛

- 直接执行S26_表示核验、S27_旧案交互，CPU为主，单线程，读取原证据；不重跑旧raw。S26已完成，S27先做跨run匹配门槛。
- GPU5低成本inference须使用完整且身份匹配的checkpoint。现在缺少完整L06 RulerVLA，因此S25等待；不为了占满GPU做无关probe。
- 小实验训练不得抢占GPU2/3正式进程，不得清退其他用户。当前约66GB模型占用/CPU优化器放置沿用已验证容量；不足时不降低微批、改LoRA或绕过host gate。
- 若未来另行验证更低显存placement，数值/随机性/保存恢复必须通过统一验收后再复用；不能只给某臂特殊precision。
- G1–G7保持NO-GO，不增加Correction policy训练；不重跑L01–L05或S08–S11来追求正结果。

执行结果、验证gate和冻结来源写入同名S输出目录，并在实验总目录更新状态。
