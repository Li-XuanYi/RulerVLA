# G05 Conclusion

## Question

How does measured depth degradation affect correction?

## Setup

Use the locked conditions in config.yaml and the same 50 base frames. K/T, EE pose, RGB, wrist, instruction and production thresholds remain fixed except the specified intervention.

## Result

- Effective acceptance: 67.4%.
- Scene RMS before/after (mm; episode cluster CI): 2.567 [2.567, 2.568] / 2.711 [2.697, 2.729].
- Anchor post-fit median residual (mm): 2.612 [2.524, 2.697].
- Reliability fit score: 0.466 [0.454, 0.478].
- Retained nonrobot scene coverage: 0.899 [0.899, 0.899]; mean patch weight: 0.409 [0.399, 0.420].

## Interpretation

Report noise-induced error, removed coverage and rejection separately. Missing depth can make retained-point RMS small while usable geometry decreases.

## Failure Boundary

No denoising or depth inpainting is implemented. Global fit_score may stay high under random missing while patch weights decrease.

## Cannot Conclude

No policy success-rate improvement, learned anchor-detector performance, full SE(3) recovery, unseen tasks or real-world calibration guarantee follows from this geometry-only study.

## Decision

continue only with measured validity and coverage recorded.
