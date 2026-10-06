# G07 Conclusion

## Question

Can anchor evidence detect target-only depth bias?

## Setup

Use the locked conditions in config.yaml and the same 50 base frames. K/T, EE pose, RGB, wrist, instruction and production thresholds remain fixed except the specified intervention.

## Result

- Effective acceptance: 100.0%.
- Scene RMS before/after (mm; episode cluster CI): 0.800 [0.784, 0.818] / 0.890 [0.868, 0.910].
- Anchor post-fit median residual (mm): 1.005 [0.962, 1.043].
- Reliability fit score: 0.818 [0.812, 0.825].
- Target ROI RMS before/after: 7.603 [7.598, 7.609] / 7.854 [7.798, 7.902] mm; eligible records 100/100.

## Interpretation

Correct robot anchors cannot diagnose a target-local depth bias. Unchanged anchor fit is evidence about the reference only, not the target's metric correctness.

## Failure Boundary

Target-only depth error remains undetected by this reliability mechanism; invisible-target rows have null target metrics and are not substituted.

## Cannot Conclude

No policy success-rate improvement, learned anchor-detector performance, full SE(3) recovery, unseen tasks or real-world calibration guarantee follows from this geometry-only study.

## Decision

modify hypothesis: distinguish reference consistency from full-scene correctness.
