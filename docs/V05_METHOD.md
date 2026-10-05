# V0.5 method and current evidence

## Nominal geometry

For canonical pixel q = (u, v, 1), optical depth D in metres and fixed camera
calibration K and (R, t):

```text
P_camera = D K^-1 q
P_action = R P_camera + t
ActionCoord = P_action - p_EE
```

Action axes remain world XYZ. The current EE position provides the origin.
EE orientation transforms known rigid robot surface points, not action axes.

## Independent visual anchors

`AnchorProvider.detect(rgb)` returns pixel locations, confidence and visibility.
The simulator GT backend obtains local rigid surface points from compiled mesh
triangle centroids on `gripper0_right_gripper`. It excludes moving fingers and
does not derive the reference points by backprojecting the same depth being
tested. The true camera and same-state EE pose project those points into the
canonical raster. In-frame, positive-depth points consistent with rendered
depth within 3 mm are eligible anchors.

This oracle backend establishes the mathematical chain when anchors are
correct. It does not establish performance of a visual keypoint detector.
`LearnedAnchorProvider` is an explicit unimplemented stub.

## Shared translation correction

For known local surface point l_E and observed depth-backprojected point P_C:

```text
residual = R P_C + t - (p_EE + R_EE l_E)
bias estimate = robust location of residuals
corrected ActionCoord = nominal ActionCoord - bias estimate
```

First take the coordinate median. Iterate a confidence-weighted vector Huber
location update with delta 2 mm. Accept only at least three distinct raster
anchors with consistent post-fit residuals (default median threshold 5 mm).
Out-of-frame pixels are invalid, not clamped or wrapped. Duplicate raster
evidence cannot inflate anchor count. No threshold rejects bias solely for its
magnitude. A rejected estimate is exactly zero.

This implementation estimates three translation parameters. It does not
estimate rotation, full SE(3), focal length or depth scale.

## Reliability

The deterministic score uses eligible anchor count, confidence, median
post-fit residual and agreement. For each 16-by-16 source patch, its valid-depth
fraction multiplies the score. Encoded coordinates blend toward neutral zero
with this weight; no reliability channel is added. It reads neither injected
bias nor task outcomes. Large consistent bias can remain reliable.

Reliability is a configured engineering heuristic. Its effect on policy
performance is not yet measured.

## Optional ruler

The ruler projects physical 5, 10 and 15 cm ticks along world +x/+y/+z. It uses
the same bias estimate as the dense map: nominal camera translation becomes
t - bias, while the known EE provides the origin. There is no image-space
length normalization and no separate ruler origin. Ruler drawing occurs in
the canonical source raster before the existing policy flip and resize.
The default is disabled; rejected references do not draw a ruler.

## Validation completed

CPU regression: **40 tests passed in 0.75 s**. Tests include transform signs,
metres, EE subtraction, invalid depth, six correct/two incorrect anchors,
insufficient anchors, large consistent bias, physical ruler projection,
orientation, three separate image slots and float32 spatial parsing.

Frozen parity: **50 real frames x four profiles**, exact equality of canonical
geometry and policy input images; actual OpenPI third-slot parity passed.

Synthetic independent-point smoke recovered bias to numerical precision
(1.21e-16 m error).

Real GT-anchor geometry-only smoke:

```text
task: pick up the black bowl from table center and place it on the plate
selection: first 10 V0.3 manifest episodes x 5 evenly spaced frames
frames: 50
accepted: 50
rejected: 0
injected translation: [+0.020, -0.010, +0.005] m
mean map error before: 0.0229128785 m
mean map error after: 0.0002963521 m
maximum bias recovery error: 0.0004925929 m
control steps / policy inference / success evaluation: none
```

Accepted recovery error had to be at most **2 mm**, and geometry error had to
decrease. These gates were fixed before running the smoke. Remaining error is
consistent with finite raster sampling and surface depth lookup. See complete
per-frame evidence in `docs/evidence/gt_anchor_smoke.json`; do not infer policy
success or real-world calibration robustness from this result.

Two engineering issues were corrected during validation: bind the actual
rigid body name, and match frozen float64 depth conversion. Failed first
attempts did not change the source data, frozen code, bias injection or gates.

## Reproduction on ZJU

From `/home/yqt/XuanyiLi/RulerVLA`:

```bash
PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu \
  /home/yqt/XuanyiLi/openpi/.venv/bin/python -B -m pytest -q

PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu \
  /home/yqt/XuanyiLi/openpi/.venv/bin/python -B scripts/verify_geometry.py \
  --v03-root /home/yqt/XuanyiLi/actioncoord/work/v03_sync_rgbd_20261003T164806

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  NUMBA_CACHE_DIR=/home/yqt/XuanyiLi/RulerVLA/outputs/numba \
  LIBERO_CONFIG_PATH=/home/yqt/XuanyiLi/actioncoord/fixes/framework_fix_20261002T151206Z/libero_config \
  /home/yqt/XuanyiLi/openpi/examples/libero/.venv/bin/python -B scripts/smoke_pipeline.py \
  --v03-root /home/yqt/XuanyiLi/actioncoord/work/v03_sync_rgbd_20261003T164806 \
  --episodes 10 --frames-per-episode 5 --device-id 4 \
  --output outputs/gt_anchor_smoke.json
```

The primary package targets Python >=3.11. This bounded simulator harness was
also executed in the existing LIBERO Python3.8 environment. OpenPI and LIBERO
are optional external runtimes, not vendored dependencies.

## Next smallest experiment

Review the per-frame **GT-anchor geometry correction smoke** and, if needed,
repeat a bounded geometry-only sample with the same frozen contract. No policy
training, official rollout, learned detector training or superiority claim is
part of this stage.
