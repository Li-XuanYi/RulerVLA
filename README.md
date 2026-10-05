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

## Engineering validation

- 40 CPU tests passed.
- 50 real frames across R0-R3 match frozen V0.3 geometry and policy inputs
  exactly, including the actual OpenPI third-slot transform.
- 50 simulator states from ten task-2 episodes passed a geometry-only GT-anchor
  smoke. Injected translation `[+0.020, -0.010, +0.005] m` produced mean map
  error 22.913 mm before and 0.296 mm after correction; maximum bias recovery
  error was 0.493 mm. No control steps or policy inference were performed.

Per-frame JSON evidence and SHA-256 manifest are in `docs/evidence/`.
The learned anchor detector remains an explicit stub. This result validates
translation correction with oracle landmarks; it does not measure policy SR.

Repository: https://github.com/Li-XuanYi/RulerVLA (private). Commit identity is
`Lixuanyi`, using repository-local settings.
