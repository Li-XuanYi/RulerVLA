# Main Comparison v1 execution preparation — 2026-10-06

## Authorization and inventory

The user authorized reusing existing compatible training/evaluation evidence,
fine-tuning missing models on available GPUs, and evaluating compatible models
that do not already have evaluation records. The frozen design remains four
methods x three seeds x four suites, full pi0.5 fine-tuning, effective batch 32,
30,000 optimizer updates, independent official pi05_base initialization.

`MAIN_RUN_INVENTORY_20261006.json` checks all 48 planned checkpoint roots and
main-comparison receipts: **0 final checkpoint directories, 0 training receipts,
0 evaluation receipts**. Old E1 LoRA/batch-4 checkpoints/results and task-2 pilots
remain separate evidence; they do not satisfy the new full/batch-32/four-slot
protocol. No old policy evaluations were repeated.

## Actual work completed

- Created a separate source snapshot under
  `/home/yqt/XuanyiLi/RulerVLA/outputs/main_comparison/runtime_v1` from frozen V0.3.
  Only default image keys, inputs_spec and image augmentation are changed. The
  source runtime and model parameter architecture remain unchanged.
- Actual OpenPI preprocessing retains all four slots with identical values in
  train and inference mode; no image augmentation occurs.
- Shape-traced the real full pi0.5 architecture through its image/language prefix
  and flow-matching loss: prefix `[1,1224,2048]`, loss `[1,10]`, actions `[1,10,32]`.
  This is **not** a loaded-weight numerical forward/backward or optimizer smoke.
- Ten Spatial tasks passed first/middle/last replay checks: **10 episodes, 30 frames**,
  exact source state, RGB pair, agent depth and state8 position.
- ZJU regression: **66 tests passed**. Frozen design and G evidence SHA manifests
  passed verification. Four pre-existing untracked Chinese documents are preserved.

## Background processes and latest snapshot

These are preparation processes, **not formal training**.

| Worker | Queue PID/session | Work | Latest verified progress |
|---|---:|---|---|
| Spatial geometry | 1745257 | GPU 3 rendering; both-view K/T, wrist depth, state8 and quaternion | 15/432 episodes, 1,749/52,970 frames; child PID 1746535 |
| Other suite sources | 1336482 | CPU/download; Object -> Goal -> Long | Object 2/10 tasks, 89/454 episodes matched exactly; child PID 1336485 |

Source files are pinned to the already validated E1 revisions:
`clip-rt/modified_libero_hdf5@6a6659f8ac7d580fd594173a0e3abf880c843130`
and original exceptions
`yifengzhu-hf/LIBERO-datasets@f13aa24a3da8c43c7225569f28c562979fa0e35a`.
Downloads require exact LFS byte count and SHA-256; trajectory matching reuses the
archived E1 matcher and requires an unambiguous exact float32 action sequence.

All new large data are saved under:
`/home/Dataset2/yqt/rulervla/main_comparison_v1/data_preparation/`.
The original LeRobot data and cached RGB-D are read-only.

## Engineering failures resolved and retained

1. A float32 depth intermediate differed from the frozen generator by up to about
   0.032 mm. Restored its exact float64 intermediate; agent depth equality remains
   mandatory, without loosening a depth tolerance.
2. The mirror rejected urllib's default client with HTTP 403. The existing curl
   approach succeeds for fixed-revision metadata and downloads; files retain
   exact size/SHA validation.
3. Full replay episode 1271/frame 13 had two wrist RGB channel values lower by
   one uint8 unit. Signed subtraction confirms 1, not the misleading unsigned
   difference of 255. Simulator state and agent depth still match exactly.
   The cached training RGB is unchanged for every method. The preparation gate
   records sparse renderer quantization (maximum channel delta 1, changed-channel
   fraction <=1e-4); larger raster/colour disagreements stop the queue. Each
   quantization exception is retained in a diagnostic JSON and episode receipt,
   and such an episode is not described as bitwise RGB-exact. This numerical
   replay check is resolved before any new model training or policy outcome.

Initial failures, resumed status receipts and append-only logs are retained.

## Remaining work before formal training

- Complete Spatial geometry and all other suite source coverage. The other suites
  still need same-state RGB-D/geometry generation and across-task validation.
- Reuse and verify official AimBot and AxisGuide renderers with the shared input
  contract and freeze resolved cue/runtime/data configurations.
- Verify full pi05_base weight loading, full trainable parameters, numerical
  forward/backward, averaged gradient accumulation/effective batch 32, and
  complete checkpoint save/load/resume for all methods.
- Choose a common GPU/microbatch/FSDP layout after memory smoke; save full
  checkpoints to Dataset2 rather than the nearly-full system filesystem.
- Then launch missing 30k runs and evaluate completed compatible runs exactly
  once. There is currently no automatic formal-training launch from these
  preparation workers; pending gates are not represented as passed.

## Inspect progress

On ZJU:

```bash
cd /home/yqt/XuanyiLi/RulerVLA
cat outputs/main_comparison/spatial_geometry_status.json
cat /home/Dataset2/yqt/rulervla/main_comparison_v1/data_preparation/coverage/libero_object/status.json
tail -f outputs/main_comparison/logs/spatial_all_432_geometry.log
```

Worker status: `outputs/main_comparison/spatial_preparation_status.json` and
`sources_preparation_status.json`; stage logs live under
`outputs/main_comparison/logs/`. Completion counts above are a snapshot, not final
data completion or training/evaluation results.
