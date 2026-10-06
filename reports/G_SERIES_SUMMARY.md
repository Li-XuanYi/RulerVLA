# G-Series Geometry Boundary Study

## Q1 — Which errors can V0.5 reliably repair?

Shared world translation at 2/5/10/20 mm along +x/+y/+z and the fixed mixed direction:
800/800 accepted, maximum accepted recovery error
0.493 mm. All conditions use the
same frozen ten episodes / fifty frames. Across 1-3 of eight 20 mm correspondence
outliers, paired Huber-minus-mean bias error is -2.944 mm, episode
cluster CI [-2.989, -2.902].

## Q2 — Which unrepaired errors can reliability detect?

With one or two retained anchors, 0/300
conditions accept; insufficient evidence falls back to zero correction.
However, G2 contains 9 accepted 3/4-anchor recoveries above
the fixed 2 mm accuracy limit. Their small fit residual and high score do not
certify recovery accuracy. They improve the original 10 mm error but fail the
predeclared accuracy gate, rather than necessarily worsening the scene.
Depth-noise and missing-depth acceptance/coverage are reported condition by
condition in G05. Anchor-centred block missing can remove reference evidence;
random missing can reduce patch weights without changing the global fit score.
This is validity/consistency detection, not depth reconstruction. Missing pixels
are excluded from retained-support RMS and their lost coverage is reported.

## Q3 — Which errors cannot be repaired or detected?

- Extrinsic rotation: acceptance 100.0%; after-error
  near 6.765 mm / far 42.162 mm.
  A robot-local translation fit leaves position-dependent error elsewhere.
- Anchor-only depth bias: scene RMS before 0.000 mm /
  after 7.392 mm, acceptance 100.0%.
  Consistent local bias can wrongly move a clean scene. This is an identifiability
  limit under the shared-error assumption, not a reason to add a GT-specific patch.
- Target-only depth bias: target RMS before 7.603 mm /
  after 7.854 mm; acceptance 100.0%.
  Maximum paired reliability-score change versus clean is 0.
  Reference consistency does not certify full-scene metric correctness.

## Q4 — Is the mechanism ready for a policy-level pilot?

**NO-GO**. This is a scope-limited research decision, not training authorization.
The analysis code committed before outcomes uses a strict zero-exceedance G2
gate at 2 mm; it remains unchanged. The observed G2 exceedances are not erased
by changing thresholds after reviewing outcomes. NO-GO blocks automatic pilot
progress under this protocol; it does not imply shared-translation correction
is ineffective.

1. Require the G1 recovery, G2 insufficiency and G3 robustness gates in the JSON summary.
2. Restrict any later pilot to a declared shared-translation regime with GT anchors;
   reliability must not be presented as a general geometry validator.
3. Preserve the G4/G6/G7 failures and add independent evidence before claiming
   recovery from rotation or local depth error.
4. Require complete frozen manifests, regression tests, committed results and a clean tree.
5. The user must separately approve training. No policy inference, rollout, detector
   training or policy training was performed in this study.

These confidence intervals resample ten episodes, preserving frames and subsets;
3,350 interventions and image pixels are not independent trajectories. The small
deterministic task-2 base set cannot establish policy SR, learned-detector quality,
general tasks, negative rotation angles or adversarial outlier distributions.

Base SHA-256: `74029f20cd2dd9ff06568bcac89a7a0f0b07708b485a17d462195117fe5f8706`.
Protocol SHA-256: `7da25721ade32c6aeb5792a0bab2047b3d2eeb978f49468351fd7c3c9270cdaf`.
