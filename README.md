# RulerVLA V0.5

Lightweight spatial middleware for LIBERO observations and OpenPI / pi0.5.
Frozen ActionCoord V0.3 geometry is the reference. V0.5 adds visual anchor
interfaces, robust shared translation correction, deterministic geometry
reliability and an optional metric action ruler. This repository contains no
policy weights, demonstrations, OpenPI fork, or training launchers.

Scientific status: geometry and software validation only. V0.3 R2 completed
a 5k pilot; the latest R3 pilot failed around 4.2k because of a runtime/native
issue. There is no completed CameraXYZ versus ActionCoord conclusion and no
V0.5 policy-performance claim.

```bash
python -m pip install -e '.[test,smoke]'
python -m pytest -q
python scripts/smoke_pipeline.py
```

See `docs/ARCHITECTURE.md`, `docs/V03_PROVENANCE.md` and `docs/V05_METHOD.md`
for geometry, integration and evidence boundaries.
