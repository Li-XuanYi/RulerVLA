# G03 Conclusion

## Question

Does Huber improve over the plain mean?

## Setup

Use the locked conditions in config.yaml and the same 50 base frames. K/T, EE pose, RGB, wrist, instruction and production thresholds remain fixed except the specified intervention.

## Result

- Effective acceptance: 100.0%.
- Scene RMS before/after (mm; episode cluster CI): 10.000 [10.000, 10.000] / 0.737 [0.662, 0.825].
- Anchor post-fit median residual (mm): 1.422 [1.353, 1.493].
- Reliability fit score: 0.622 [0.614, 0.630].
- Paired Huber minus mean bias error for 1-3 outliers: -2.944 mm, CI [-2.9894320931740768, -2.902415947767486].

## Interpretation

A negative paired difference supports robust-location resistance to these fixed correspondence errors. Plain mean remains an analytical baseline without acceptance gating.

## Failure Boundary

Only up to 3/8 outliers and the preselected +x/-y/+z 20 mm direction set were tested; coordinated adversarial errors are not covered.

## Cannot Conclude

No policy success-rate improvement, learned anchor-detector performance, full SE(3) recovery, unseen tasks or real-world calibration guarantee follows from this geometry-only study.

## Decision

continue within the tested outlier regime.
