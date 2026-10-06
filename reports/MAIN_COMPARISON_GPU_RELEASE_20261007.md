# Main Comparison v1: GPU release

## Frozen training

Four methods, three seeds (42/43/44), four separate suites: 48 planned models.
All runs independently initialize official `pi05_base/params`, use full
fine-tuning (no LoRA or frozen backbone), 30,000 optimizer updates, effective
batch 32, horizon 10, identical AdamW, cosine schedule, EMA and normalization.
Four active image slots and disabled image augmentation are shared by all arms.
Anchor, correction and reliability are disabled in RulerVLA-main.

Executable code at release: `331b16a`. The isolated runtime is `runtime_v3`;
the original V0.3 runtime, historical checkpoints and evaluation outputs remain
unchanged. Each run records exact code, runtime, data, cue and normalization
hashes in its own `execution_lock.json`.

## Completed release gates

- All 432 Spatial episodes (1261–1692), 52,970 frames, paired RGB-D and geometry
  are ready. No evaluation perturbations enter training.
- Four-method, dual-view cue audit across ten tasks passed, with reviewed actual
  RGB/contact sheets and independent axis projection checks.
- Official full checkpoint loaded: 3,353,433,872 trainable parameters.
- Worst-case RulerVLA-main real GPU smoke completed two optimizer updates at
  effective batch 32: loss 0.07281794 / 0.07416944; grad norm 0.96498853 /
  1.12576091. No NaN, shape error or OOM in the accepted placement.
- Complete model, Adam state and EMA were saved to Dataset2, SHA verified,
  restored and compared bitwise. Checkpoint:
  `/home/Dataset2/yqt/rulervla/main_comparison_v1/smoke_checkpoints/rulervla_main_attempt7/1`.
- Five focused tests passed for shared frame order/resume, gradient accumulation
  against an independent optimizer reference, and isolated runtime patching.
- Actual first and final training frames passed all four method adapters.

## Accepted placement and engineering findings

Two-GPU combined full-state JIT exhausted currently available GPU memory;
splitting gradient/update compilation exposed NCCL transport failures, including
an illegal CUDA memory access before any optimizer update. These failed attempts
are preserved as zero-update smoke evidence.

Accepted placement uses one GPU per model for forward/backward and CPU RAM for
Adam state, model master state and EMA. Microbatch 1 with 32 accumulated
microbatches produces one optimizer update at effective batch 32. The same
placement is used for every method. Independent jobs require no cross-GPU NCCL.
Optimizer and EMA mathematics are unchanged and were independently tested.

The accepted smoke's second update took 28.03 seconds. This is a smoke timing,
not a formal-run ETA: shuffled-frame I/O, two-worker CPU contention and
checkpoint publication add overhead. This placement prioritizes running the
frozen full-finetuning protocol with available memory.

## Launched Spatial queues

| Worker | GPU | Queue PID/session | Initial child PID | First run |
| --- | --- | --- | --- | --- |
| 0 | 4 | 3838220 | 3838433 | libero_spatial_baseline_seed42 |
| 1 | 3 | 3898761 | 3899089 | libero_spatial_aimbot_seed42 |

Worker 0: Baseline / AxisGuide-adapted for each seed.
Worker 1: AimBot / RulerVLA-main for each seed.
Each worker executes six models sequentially. At initial release both children
entered their first full forward/backward; consult live status for completed
optimizer updates. Launching a process is not evidence of a finished 30k run.

Root: `/home/yqt/XuanyiLi/RulerVLA`.

Status files:

- `outputs/main_comparison/spatial_training_queue_worker0_status.json`
- `outputs/main_comparison/spatial_training_queue_worker1_status.json`

Per-run logs: `outputs/main_comparison/logs/<run_id>.log`.
Per-run live status and metrics:
`/home/Dataset2/yqt/rulervla/main_comparison_v1/checkpoints/<run_id>/status.json`
and `metrics.jsonl`.

All six checkpoints per run are retained on Dataset2 at optimizer update
5k/10k/15k/20k/25k/30k (directory indices 4999/9999/14999/19999/24999/29999).
Publication is SHA verified and serialized to limit CPU/RAM-staging peaks.
Full-state resume is available explicitly via `main_spatial_train.py --resume`;
an incomplete existing run is never silently overwritten or restarted.

## Other suites

Object, Goal and Long original demonstration coverage/action matching is
complete. Exact state replay and GT RGB-D generation runs separately (queue
1404743, renderer 1404748, GPU 4). At the initial release Object was 221/454
episodes, 32,613/66,984 frames; Goal and Long followed in that preparation queue.
These other suites are not yet released for formal training. No new policy
evaluation rollouts were launched during this release.

## Inspection commands on ZJU

```bash
cd /home/yqt/XuanyiLi/RulerVLA
cat outputs/main_comparison/spatial_training_queue_worker{0,1}_status.json
tail -f outputs/main_comparison/logs/libero_spatial_baseline_seed42.log
tail -f outputs/main_comparison/logs/libero_spatial_aimbot_seed42.log
```
