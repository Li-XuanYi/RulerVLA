# G02 Conclusion

## Question

When do fewer anchors require rejection?

## Setup

Use the locked conditions in config.yaml and the same 50 base frames. K/T, EE pose, RGB, wrist, instruction and production thresholds remain fixed except the specified intervention.

## Result

- Effective acceptance: 66.7%.
- Scene RMS before/after (mm; episode cluster CI): 10.000 [10.000, 10.000] / 3.800 [3.755, 3.851].
- Anchor post-fit median residual (mm): 4.352 [4.320, 4.386].
- Reliability fit score: 0.540 [0.535, 0.545].
- Accepted with <=2 anchors: 0/300; accepted recovery errors >2 mm: 9/600 accepted records (9/900 total).

## Interpretation

Insufficient-anchor rejection is supported, but accepted 3/4-anchor recovery can exceed the predeclared 2 mm accuracy limit despite a small fit residual. Consistency does not guarantee accuracy. These cases still improve the original 10 mm error; they are not necessarily scene-harming corrections.

## Failure Boundary

Subset A/B/C are deterministic spatial selections, not an exhaustive occlusion distribution. Rejected bias error uses zero correction and must not be described as accepted recovery.

## Cannot Conclude

No policy success-rate improvement, learned anchor-detector performance, full SE(3) recovery, unseen tasks or real-world calibration guarantee follows from this geometry-only study.

## Decision

modify hypothesis: retain the accepted low-count accuracy boundary.
