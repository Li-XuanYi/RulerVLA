# G1-G7 Geometry Boundary Study: pre-outcome protocol

This study evaluates geometry only. The translation estimator and reliability
remain at implementation commit `4cce5f08cadf92d89802e842019ea4624dbc48e5`.
The runner does not start a policy, take a control step or change a threshold.

## Base set and lock

Use exactly the same first ten V0.3 manifest episodes and five evenly spaced
frames per episode as the previous GT smoke. `prepare` checks exact RGB, wrist,
metric depth, EE, K/T replay and equality of the 50 previous frame IDs. It renders
geom segmentation from those recorded states to identify robot and target masks.
It projects independent rigid palm mesh points using the clean true camera.

Before computing any condition outcome, freeze:

- `experiments/base_frame_manifest.json`: sources, content hashes, poses,
  calibration, robot points, clean GT pixels/visibility, deterministic subsets,
  segmentation ROI hashes, cached-frame SHA-256 and distance tertiles.
- `experiments/g_series_protocol_lock.json`: full condition matrix, design,
  method config, implementation and runner SHA-256, expected 3,350 rows.
- Each experiment's config and manifest reference the same base SHA-256.

GT RGB anchor visibility and confidence are frozen from clean geometry. They
are not re-filtered against corrupted depth using known true camera geometry.
The correction sees frozen visual correspondences, measured (possibly corrupted)
depth and nominal (possibly corrupted) calibration. Its input does not contain
corruption labels, true severity or evaluation errors. This avoids an oracle
depth-error detector masking the G6/G7 failure boundary.

## Condition matrix: 67 conditions x 50 frames

| Group | Conditions |
|---|---|
| G1 | Clean control; 2/5/10/20 mm x world +x/+y/+z/fixed normalized `[1,-0.5,0.25]` |
| G2 | 10 mm mixed translation; retain 8/6/4/3/2/1 anchors x deterministic A/B/C subsets |
| G3 | 10 mm mixed translation; eight anchors; corrupt 0/1/2/3 correspondences x A/B/C; compare Huber vs plain mean |
| G4 | 0.5/1/3 degrees x optical camera x/y/z; right-multiply rotation, fixed camera translation |
| G5 | Clean calibration; independent depth noise 2/5/10 mm RMS; 10/30% random or anchor-centred block missing |
| G6 | Clean scene/calibration; +5/+10 mm optical depth only in anchor-local robot pixels |
| G7 | Clean anchors/calibration; +5/+10 mm optical depth only in visible bowl ROI, excluding anchor guard |

Subsets use deterministic farthest-pixel selection from clean distinct valid
anchors. Start choices A/B/C use first pool entry, largest u and largest v.
G3 corrupts observed-minus-known residuals by a fixed 20 mm world vector along
`+x`, `-y`, `+z`, through incorrect 3D correspondence reference points. It does
not add new estimator behavior. The plain mean is an unconditional analytical
baseline and does not have a fabricated acceptance/reliability score.

Noise uses fixed per-frame seeds and the same normalized realization at all
severities. Its empirical zero mean and RMS are controlled. Random missing uses
the same ordered permutation for nested fractions. Block missing is a rectangle
centred on the clean visible anchor cluster, clipped into the image. Record its
actual removed fraction, which can differ slightly from the requested fraction.

## Metrics and eligibility

Scene means clean valid **nonrobot** pixels. RMS and P95 are evaluated on the
pairwise common measured-valid support. Missing coverage is recorded separately;
zero-filled missing pixels are never called accurate geometry. G6 preserves
all nonrobot pixels; G7 preserves every anchor depth lookup. An invisible target
ROI is not replaced: target error is null and its G7 condition is ineligible.

Primary after-error is the physical corrected map **before** encoding and
reliability attenuation. Attenuating XYZ toward neutral zero does not prove
geometric recovery. Record fit score, mean patch weight and coverage separately.
`accepted` means estimator accepted AND reliability correction-valid; also keep
the estimator's acceptance separately. Post-fit residuals for rejected estimates
remain the uncorrected residuals, as in the frozen implementation.

GT uniform bias and bias-error are defined only for G1/G2/G3. Rotation,
independent noise and local depth biases have no single shared translation
ground truth: those fields are null, not invented zeros. Report estimated bias
norm and per-point scene/target errors instead.

G4 near/middle/far are fixed clean-scene distance tertiles about the clean robot
anchor centroid, retained for every rotation condition. G6/G7 representative
visualizations use the first base frame at 10 mm, selected before outcomes.

## Analysis and decision rules

All conditions are paired on the same frames. Confidence intervals resample
the ten episodes (2,000 deterministic cluster-bootstrap draws), retaining the
five frames and all subsets within each sampled episode. Pixels and 3,350
interventions are not 3,350 independent trajectories. CIs are descriptive for
this small deterministic base set, not task-success evidence.

Predefined reporting flags, not new estimator gates:

- G2 accepted bias-error >2 mm is a silent reference-recovery error.
- Scene RMS increase >0.5 mm is scene harm.
- Accepted scene harm is a silent wrong correction.
- Residual scene/target error >2 mm despite acceptance indicates unrepaired
  geometry, even if correction improves a larger initial error.

Keep failure boundaries. Do not tune thresholds or patch the algorithm based
on G6/G7 results. Verify complete rows, hashes, input isolation, CPU tests and a
clean committed tree. Report GO / CONDITIONAL GO / NO-GO; no decision starts
training automatically. This study cannot establish policy SR, a learned
detector's performance, general task transfer or arbitrary calibration recovery.
