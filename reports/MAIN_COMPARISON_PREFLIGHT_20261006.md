# Main Comparison v1 preflight — 2026-10-06

## Completed

- User selected per-suite training: four methods x three seeds x four suites = 48 models.
- User selected independent cue image slots retaining both camera views and the
  original pi0.5 input convolution. Common input contract has four active slots.
- Nominal-EE ruler is decoupled from anchors/correction; main ActionCoord values
  are not scaled by reliability. Legacy corrected-reference guards are retained.
- Design, suite episode manifests and 48 planned run configs are written.
- 60 local tests passed, including forbidden Anchor/Correction/Reliability calls,
  unchanged wrist/state/actions, physical tick projection and auxiliary range checks.
- ZJU regression also passed: 60 tests in 1.17 s. Input-only checks passed on all
  50 frozen real frames: wrist/state/unscaled ActionCoord match exactly, nominal
  ruler uses no anchor/correction/reliability. This does not test model forward,
  backward, the actual four-slot runtime or optimizer/save/load.
- Local official pi05_base exists with expected Orbax root structure. A model load
  is still a required smoke gate; filesystem presence is not a completed loader test.

## Actual training source coverage

| Suite | Tasks | Episodes | Frames | Existing RGB-D NPZ | Verified state8 sidecars |
|---|---:|---:|---:|---:|---:|
| Spatial | 10 | 432 | 52,970 | 432 | 46 (task 2 only) |
| Object | 10 | 454 | 66,984 | 0 | 0 |
| Goal | 10 | 428 | 52,042 | 0 | 0 |
| Long / libero_10 | 10 | 379 | 101,469 | 0 | 0 |
| Total | 40 | 1,693 | 273,465 | 432 | 46 |

The audited Spatial NPZ contains agent RGB, wrist RGB, agent depth, EE position,
actions and original frame indices. It does not supply wrist depth, per-frame wrist
extrinsics, EE quaternion or the full state8. The task-2 sidecars/geometry contract
cannot stand in for 40-task same-state evidence. All arms must use the same frame
manifest and regenerated RGB source rather than mixing original RGB and replay RGB.

## Benchmark and resources

- Installed official LIBERO-Plus classification: 10,030 task instances, levels 1-5;
  Level 2 has 2,202 entries across all seven categories. 121 entries have no level
  and are outside the Level-2 design. Exact source hashes are in the preflight JSON.
- Planned standard evaluation: 24,000 rollouts; Plus Level 2: 26,424 rollouts.
  These are planned counts, not completed results.
- Audit snapshot: GPUs 0/1 nearly full; GPU2 busy; GPU3/4/5 partly free. A snapshot
  is not a reservation or proof that full fine-tuning with batch 32 fits.
- Dataset2 has about 20 TB free, while the system filesystem has only about 33 GB.
  Existing local checkpoint staging cannot be assumed to hold full optimizer/EMA
  checkpoints. Validate a suitable writer/staging location with save/load/resume.

## Remaining launch gates

1. Supplement and verify full-suite RGB-D, both-view calibration and state8.
2. Reuse official AimBot/AxisGuide renderers and validate identical geometry/raster
   semantics; freeze resolved renderer configs and source hashes.
3. Wire and test all four slots through actual pi0.5 inputs_spec, preprocessing,
   training loss and inference. The checked stock preprocessing enumerates three
   default image keys and applies RGB colour jitter: it cannot be used unmodified.
4. Verify full trainable parameter filters, same initialization and train-only stats.
5. Verify averaged gradient accumulation, 32 samples/update, common memory layout,
   and 2-5 actual update/save/load/resume smokes for all four methods.
6. Freeze executable runtime/data/evaluation locks, then launch formal training.

**No formal training, optimizer smoke or policy rollout was started in this preflight.**
The G1-G7 results remain archived as evidence about the separate Anchor extension.
