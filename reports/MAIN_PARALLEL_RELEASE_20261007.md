# Main comparison parallel release — 2026-10-07

## Running training

- Spatial Baseline seed 42: GPU 2, PID 527793, 547/30,000 optimizer updates at inspection.
- Spatial AimBot seed 42: GPU 3, PID 527802, 531/30,000 optimizer updates at inspection.
- Existing Spatial queue owners: PIDs 527466 and 527467; their 12 assignments are preserved.
- No new main-comparison model has completed 30k. Prior exploratory checkpoints are not substitutes for the frozen full-FT main protocol.

## Work completed alongside training

1. Object, Goal and Long training-only shared normalization: complete, respectively 66,984 / 52,042 / 101,469 frames.
2. Four-method numerical cue audit and actual dual-view contact-sheet review: passed for all 10 tasks in each of these three suites. Review image hashes and release records are in `outputs/main_comparison/suite_release/<suite>/`.
3. Corrected-orientation RGB-D, camera transform and initialization preflight: complete for all 40 tasks; the paired 50-scene plan per task is saved in `outputs/main_comparison/eval_environment_preflight/paired_scene_plan.json`. No policy rollouts were performed by this preflight.
4. Actual four-slot policy inference API smoke on GPU 5: complete for Baseline, AimBot, AxisGuide-adapted and RulerVLA-main; every output is finite with shape 10×7. It used official pi05_base for engineering validation only, performed zero rollouts and is not a success-rate result. PID 1323963 exited; results are in `outputs/main_comparison/inference_api_smoke/`.
5. Remote focused checks passed: 7 tests covering suite loading/chunks, frame order, transient storage handling and gradient accumulation.

## Remaining-suite resource queue

PID/session 1323962 is running `scripts/main_remaining_suite_queue.py`. It owns the remaining 36 Object/Goal/Long runs, disjoint from the 12 Spatial assignments. It allows one additional training process at a time. The first planned model is Object RulerVLA-main seed 42.

At inspection it was **waiting**, with no training child: GPU 5 had 49,892 MiB free, GPU 4 had 6,448 MiB free, and host MemAvailable was 221 GiB. These do not meet the release thresholds (70,000 MiB free GPU; 380 GiB available host RAM for Object/Goal or 420 GiB for Long; stable existing Spatial owners). GPU utilization alone does not indicate that a full training process fits. Other users' GPU processes were preserved.

All new models retain the same official base initialization, full fine-tuning, micro-batch 1 × accumulation 32, 30k updates, horizon 10 and frozen four-slot protocol. No LoRA conversion, batch reduction or new ablation was used to obtain concurrency.

Queue status: `outputs/main_comparison/remaining_suite_queue_status.json`.
Queue log: `outputs/main_comparison/logs/remaining_suite_queue.log`.
Launch receipt: `outputs/main_comparison/parallel_launch_20261007.json`.

## Next dependent work

Formal success-rate evaluation requires each frozen 30k checkpoint. Once available, evaluate the saved paired scenes and skip technically valid existing results. LIBERO-Plus baseline difficulty calibration depends on the corresponding trained baseline. These are not reported as completed experiments.

## User-approved GPU 4 / GPU 5 split (supersedes the single spare queue)

The former waiting queue was stopped only after confirming it had no training child; its status and log were preserved under `outputs/main_comparison/recovery/`. The 36 pending models are now partitioned without overlap:

| GPU | Methods across Object / Goal / Long, seeds 42/43/44 | Runs | Queue PID/session |
| --- | --- | --- | --- |
| 4 | RulerVLA-main, Baseline | 18 | 1686892 |
| 5 | AxisGuide-adapted, AimBot | 18 | 1687056 |

Both queue processes were verified alive with `child_pid=null`, waiting for resources. Latest checks showed 39,564 MiB free on GPU 4, 49,598 MiB on GPU 5 and about 238 GiB available host RAM. Neither additional formal training has started yet. Their fixed-GPU resource gates retain the validated training protocol; admission is serialized and peer initialization must stabilize before another model loads. A CPU-only scheduler platform override is explicitly removed from training-child environments.

Statuses and logs now use `remaining_suite_queue_worker0` / `remaining_suite_queue_worker1` filenames. Launch receipt: `outputs/main_comparison/dual_queue_launch_20261007.json`. GPU 2/3 existing Spatial training continues.
