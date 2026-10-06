# G04 Conclusion

## Question

What is the translation-only rotation boundary?

## Setup

Use the locked conditions in config.yaml and the same 50 base frames. K/T, EE pose, RGB, wrist, instruction and production thresholds remain fixed except the specified intervention.

## Result

- Effective acceptance: 100.0%.
- Scene RMS before/after (mm; episode cluster CI): 35.545 [35.439, 35.656] / 25.506 [25.403, 25.616].
- Anchor post-fit median residual (mm): 1.589 [1.544, 1.630].
- Reliability fit score: 0.702 [0.695, 0.709].
- Far scene RMS after: 42.162 [41.991, 42.352] mm; near: 6.765 [6.720, 6.813] mm.

## Interpretation

A locally fitted translation cannot eliminate rotation-induced position-dependent error. Low robot-cluster residual can coexist with large distant-scene error.

## Failure Boundary

Rotation is not repaired. Acceptance of rotation conditions is a reliability scope limitation, not proof of successful correction.

## Cannot Conclude

No policy success-rate improvement, learned anchor-detector performance, full SE(3) recovery, unseen tasks or real-world calibration guarantee follows from this geometry-only study.

## Decision

modify hypothesis: reliability describes local reference consistency.
