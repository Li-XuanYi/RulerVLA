# G06 Conclusion

## Question

Can consistent anchor-local bias move a clean scene?

## Setup

Use the locked conditions in config.yaml and the same 50 base frames. K/T, EE pose, RGB, wrist, instruction and production thresholds remain fixed except the specified intervention.

## Result

- Effective acceptance: 100.0%.
- Scene RMS before/after (mm; episode cluster CI): 0.000 [0.000, 0.000] / 7.392 [7.338, 7.448].
- Anchor post-fit median residual (mm): 1.232 [1.194, 1.268].
- Reliability fit score: 0.751 [0.743, 0.759].
- Accepted scene-harming corrections: 100/100.

## Interpretation

Consistent robot-local depth bias is confounded with shared reference translation. Its agreement can cause a nonzero global correction that moves a previously clean nonrobot scene.

## Failure Boundary

Identifiability limitation: anchor-only evidence cannot establish that the measured bias is shared by the whole scene. Do not patch it using GT corruption labels.

## Cannot Conclude

No policy success-rate improvement, learned anchor-detector performance, full SE(3) recovery, unseen tasks or real-world calibration guarantee follows from this geometry-only study.

## Decision

modify hypothesis: require a shared-error assumption or independent scene evidence.
