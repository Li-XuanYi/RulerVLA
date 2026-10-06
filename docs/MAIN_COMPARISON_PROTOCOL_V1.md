# Main Comparison Protocol v1

## Decision and status

The user approved suite-specific training (48 models) and independent cue image
slots retaining both camera views and the original pi05_base vision input layer.
The scientific design below is fixed before any main-comparison outcomes.
**Execution is not ready:** aligned data, upstream-renderer parity, four-slot
runtime consumption, gradient accumulation and checkpoint storage require gates.
The run matrix is a plan, not evidence of launched training.

The earlier G-series NO-GO applies to the anchored shared-correction mechanism.
Its archived results and thresholds remain unchanged. This main comparison removes
that mechanism and asks whether Ruler + ActionCoord is competitive as a method.

## Methods and information

| Method | Agent RGB | Wrist RGB | Agent auxiliary | Wrist auxiliary |
|---|---|---|---|---|
| Baseline | unchanged | unchanged | neutral zero | neutral zero |
| AimBot | official shooting line | official reticle | neutral zero | neutral zero |
| AxisGuide-adapted | unchanged | unchanged | independent xyz direction cue | independent xyz direction cue |
| RulerVLA-main | nominal-EE metric ruler | unchanged | ActionCoord | neutral zero |

All methods have four active three-channel 224x224 image slots using the same
shared pi0.5 visual encoder; there are no new convolution weights or action heads.
Neutral auxiliary values are float32 zero. Cue RGB is mapped to [-1,1]; ActionCoord
uses the frozen V0.3 metres/3.25 encoding. Different input information is the
method intervention, so this table cannot isolate ruler scale from dense depth.

RulerVLA-main disables VisualAnchor, Correction and Reliability, including spatial
value scaling. Its centre is the same-observation proprioceptive EE position. It
projects +world/action x/y/z at 0.05/0.10/0.15 m, without pixel-length normalization.
Rendering is at canonical 256x256 before policy orientation and resizing; width
and tick radius are 2 px. The RGB, wrist, K, transforms, state8, instruction and
actions must share the exact original demonstration state.

AxisGuide-adapted preserves the paper's EE-centred, image-normalized axis directions
in both views. Freeze the checked reference's local projection epsilon at 0.02 m,
arrow length at 60 source pixels, thickness 2, head length 8, head width 6. The
intentional adaptation is independent image slots instead of six-channel early
fusion; it is not the official SmolVLA implementation or its reported score.
Reference source: AxisGuide-code/src/lerobot/datasets/visual_cue_utils.py.
Upstream parity must be measured before training; no unverified renderer is a baseline.

AimBot reuses crosshair/reticle_builder.py and CONFIG_DICT's
large_crosshair_dynamic_default_color, with front/wrist tolerance 10 and the
upstream projection/visibility/gripper-state semantics. Freeze complete resolved
renderer configuration and source hashes after parity checks, before main training.
No learned target position, task-specific object GT, segmentation or anchor GT
may enter any policy input. Simulator optical-Z depth and camera calibration are
declared auxiliary sensor sources used by AimBot/RulerVLA, not RGB-only evidence.

## Training design

- Four separate suites: libero_spatial, libero_object, libero_goal, libero_10 (Long).
- Four methods x three training seeds [42,43,44] x four suites = 48 independent models.
- Every run initializes independently from the same local official pi05_base params.
  No pi05_libero weight loading and no arm-to-arm initialization.
- Full fine-tuning: SigLIP, language backbone and action expert all train; no LoRA,
  frozen backbone or method-specific freeze filters.
- 30,000 optimizer updates; effective global batch 32 for every method.
  micro_batch x accumulation x data_parallel_replicas = 32. FSDP sharding is not
  another data-parallel multiplier. Microbatch and accumulation are selected jointly
  for all methods after the worst-case four-slot memory smoke.
- Average unscaled microbatch gradients; clip once, then one AdamW/EMA/scheduler
  update per accumulated global batch. Verify numerical equivalence on a small
  differentiable fixture and actual counter/sample accounting in the model smoke.
- Common pi0.5 horizon 10, action_dim 32 (7 supervised LIBERO components), matching
  existing action/state transforms; no extra delta action transform.
- AdamW beta1=.9, beta2=.95, eps=1e-8, weight_decay=1e-10, clip norm=1.0,
  EMA=.999. Common warmup/cosine: 1,000 updates, peak 5e-5, end 5e-6 at 30k.
  These common choices are not claimed to reproduce AimBot's pi0 LR.
- Preserve correct V0.3 orientation: raw->policy rot180; canonical->policy hflip;
  K offset [-.5,-.5] applied once; RGB bilinear, spatial nearest.
- Disable random crop, rotation and colour jitter for all methods in v1. In particular,
  stock RGB augmentation must not distort float metric maps or misalign cue and RGB.
- Same episode IDs, frame indices, sample order, tail padding, train-only state/action
  normalization assets and loss for each suite/seed across all four methods.
- All selected official suite episodes are training data. No perturbed evaluation
  scene or LIBERO-Plus demonstration enters training. v1 uses a fixed 30k budget;
  no automatic 50k extension or model selection from main evaluation scores.
- Save complete checkpoints after 10k/20k/30k updates (index 9999/19999/29999).
  Formal results always use 29999 and the same EMA convention for all methods.
- Checkpoints and hashes are archived under Dataset2/rulervla/main_comparison_v1;
  NAS/local writer and resume must pass save->load verification before formal runs.

## Evaluation

Standard LIBERO: 10 tasks x 50 official initial states per suite, shared fixed scene
seeds across methods and all training seeds. 500 rollouts/suite/model = 24,000 total.
Use official suite-specific horizons/control/success predicates; freeze them and
the simulator/assets commits in the evaluation lock. Preserve original instructions.
No E1 custom target shifts or its 11 exclusions are transferred to this main table.

LIBERO-Plus: freeze Level 2 and all seven categories using the installed official
task_classification.json, not ranking-dependent selection. The audited checkout
contains L1-L5, 10,030 tasks, and 2,202 Level-2 entries. Use one official trial per
perturbed task as upstream specifies, giving 26,424 planned rollouts for 4 methods
x3 training seeds. Map each condition to its original suite checkpoint and base task.
Freeze exact task IDs, initial states, category metadata and source hashes before
inference. Keep perturbed language instructions for the Language category.
Do not silently replace the official noise/depth/calibration perturbation path with
clean inputs. Capture actual same-state observations after each perturbation.

Report per-suite/category SR as mean and standard deviation across three training
seeds; Avg is an equal-weight macro average over four suites or seven categories.
Also report counts and the category-imbalanced micro SR. Method differences use
paired cells; uncertainty retains task/seed clusters, not pixels or pooled IID
rollouts. Robustness drop uses the category's matched base-task mixture for ID.
Record action trajectories, first contact/grasp errors and engineering exceptions.
Engineering retries retain the original condition; unresolved errors are reported
with valid counts and cannot be silently excluded or counted as completed success.

## Required launch gates

1. Complete suite manifests for all selected episodes with RGB, both-view depth/K/T,
   same-state state8 and action provenance; verify replay/orientation across 40 tasks.
2. Nominal Ruler and unscaled ActionCoord tests; official AimBot/AxisGuide parity.
3. Four-slot inputs_spec, preprocess, loss and inference parity using the unchanged
   parameter shapes; show no auxiliary slot is dropped or colour-jittered.
4. Full-parameter loader/filter evidence and the exact same resolved optimizer,
   effective batch, update counters, horizon and normalization for all methods.
5. 2-5 update engineering smoke per method with backward, finite loss and gradients,
   actual save/load/resume; no result-driven method changes.
6. Data/storage/device capacity and checkpoint staging validated; do not launch 48
   runs into the presently nearly-full system disk.
7. Freeze renderer source hashes, runtime/data/eval locks and resolved run configs.

The design file and run matrix are frozen now; unresolved executable specifications
are clearly marked pending. A launch lock is written only after these gates pass.

## Primary sources

- https://arxiv.org/html/2508.08113v1 (AimBot simulation budget and official cue)
- https://arxiv.org/html/2606.06761v1 (AxisGuide geometry, six-channel early fusion)
- https://github.com/sylvestf/LIBERO-plus (official classification and trial convention)
