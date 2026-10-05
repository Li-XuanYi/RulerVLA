# RulerVLA architecture

RulerVLA is a spatial middleware package. It reads observations and produces
policy inputs. It does not own a policy, simulator, training loop or checkpoint.

```text
stored or live same-state LIBERO observation
  RGB + wrist RGB + optical depth + K + camera-to-world + EE pose + state8
                         |
                 frozen V0.3 geometry
                         |
           CameraXYZ / world XYZ minus current EE
                         |
          independently bound robot surface anchors
                         |
              robust shared translation fit
                         |
        corrected ActionCoord + observable reliability
                         |
          optional ruler using the same corrected camera
                         |
  agent RGB / wrist RGB / float32 spatial map -> OpenPI three image slots
```

## Ownership and boundaries

| Module | Responsibility |
|---|---|
| `data/schema.py` | Validate same-state RGB-D, calibration, state8, EE pose; immutable camera and corrected reference |
| `geometry/dense.py` | Unmodified frozen V0.3 backprojection and encoding primitives |
| `geometry/action_coord.py` | Frozen raster contract and canonical coordinate maps |
| `geometry/visual_anchor.py` | Anchor protocol, explicit GT backend and untrained detector stub |
| `geometry/correction.py` | Translation only: median initialization and confidence-weighted vector Huber IRLS |
| `geometry/reliability.py` | Deterministic fit and patch evidence; no outcome or corruption labels |
| `cues/metric_ruler.py` | Project physical world-axis ticks through the corrected reference |
| `adapters/libero.py` | Live-to-stored raster conversion and independent rigid mesh surface points |
| `adapters/openpi.py` | Canonical-to-policy images, nearest float map resize, existing OpenPI third slot |
| `evaluation/` | Inject translation into a copied camera; measure geometry errors |

## Frozen input contract

- Depth is optical Z in metres, not ray length or the normalized MuJoCo buffer.
- Optical camera axes are x-right, y-down, z-forward.
- Action axes are world XYZ. EE rotation is used for robot landmarks only;
  it does not rotate the action-coordinate map or ruler axes.
- EE origin is the simulator's `gripper0_grip_site` reference.
- Source raster is 256 by 256. Raw-to-stored is a 180-degree rotation;
  stored-to-canonical is a horizontal flip; canonical-to-policy is another
  horizontal flip. RGB and depth share the same geometry raster.
- The principal point offset (-0.5, -0.5) is applied once before backprojection.
- Encoding is `clip(coordinates / 3.25 m, -1, 1)`, with invalid pixels set to zero.
- Geometry calculations retain the frozen precision; policy spatial images are
  finite float32, nearest resized to 224 by 224 by 3.

The frozen JSON labels RGB resizing as Lanczos, while its executable OpenPI
function defaults to PIL **Bilinear**. The adapter preserves the executable
behavior and passes exact parity. The copied JSON is unchanged; see provenance.

## OpenPI integration

`OpenPIAdapter.prepare` returns `observation/image`,
`observation/wrist_image`, `observation/spatial_image`, state8 and the unchanged
prompt. The three image tensors each have shape [224, 224, 3]. The first two
remain uint8 RGB; the spatial tensor remains float32 in [-1, 1].

`OpenPIThirdSlotInputs` lazily uses the existing `LiberoInputs` transform and
assigns spatial data directly to `right_wrist_0_rgb`, enabling its mask. Spatial
data bypasses the RGB parser. No SigLIP channels, VLM layers, Action Expert or
Flow Matching objective are changed. The existing third slot is an interface,
not a second physical wrist camera.

## Profiles

| Profile | Translation correction | Reliability | Ruler |
|---|---|---|---|
| V03 | off | off | off |
| V05-C | on | off | off |
| V05-CR | on | on | off |
| V05-Full | on | on | on |

The YAML defaults to V05-CR with the ruler disabled. All thresholds, rigid body
selection and drawing parameters are configured in `configs/v05_default.yaml`.
Callers bind a same-state GT anchor provider for correction. The learned backend
raises `NotImplementedError` until an actual trained detector exists.

## Failure behavior

Insufficient or inconsistent anchors produce `accepted=False` and exactly zero
bias. With reliability disabled, rejected correction retains nominal geometry;
with reliability enabled, untrusted patches blend toward neutral encoded zero.
A rejected reference does not draw a ruler. A large, consistent bias is allowed.

GT anchors use true calibration and compiled robot mesh points. This is an
oracle mechanism test and must never be reported as a learned RGB detector.
