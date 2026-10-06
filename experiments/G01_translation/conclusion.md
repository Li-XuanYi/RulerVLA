# G01 Conclusion

## Question

Can shared translation be recovered?

## Setup

Use the locked conditions in config.yaml and the same 50 base frames. K/T, EE pose, RGB, wrist, instruction and production thresholds remain fixed except the specified intervention.

## Result

- Effective acceptance: 100.0%.
- Scene RMS before/after (mm; episode cluster CI): 8.706 [8.706, 8.706] / 0.296 [0.268, 0.325].
- Anchor post-fit median residual (mm): 1.005 [0.962, 1.043].
- Reliability fit score: 0.818 [0.812, 0.825].
- Maximum accepted bias recovery error over 2-20 mm: 0.493 mm.

## Interpretation

Shared translation is supported on this fixed GT-anchor set if accepted recovery remains within 2 mm; it does not test arbitrary calibration errors.

## Failure Boundary

Clean-control fitting has a raster/mesh residual floor. The largest tested translation is 20 mm; no claim beyond it.

## Cannot Conclude

No policy success-rate improvement, learned anchor-detector performance, full SE(3) recovery, unseen tasks or real-world calibration guarantee follows from this geometry-only study.

## Decision

continue: retain as the positive mechanism control.
