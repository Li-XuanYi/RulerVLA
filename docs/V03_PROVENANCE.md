# Frozen V0.3 provenance

## Authority

```text
Source implementation:
/home/yqt/XuanyiLi/actioncoord/work/v03_sync_rgbd_20261003T164806
Source branch: v03/sync-rgbd
Source commit: ab3fb89026b683afea401cdae91445767f6bb749
Source working tree: clean before and after extraction
Upstream VLA: OpenPI / pi0.5
```

The geometry package lives under `repo/actioncoord_experiments/src/actioncoord/`.
RulerVLA copies `geometry/dense.py` byte for byte and extracts the executable
logic of `geometry/v03.py` and `integration/v03_inputs.py` into small adapters.
It does not copy the OpenPI fork, demonstrations or checkpoints.

## Source hashes

| Frozen source | SHA-256 |
|---|---|
| `geometry/dense.py` | `414691d3a25221d28c05fedaf3b3b793bbbb4551c701c536199e8dcaf03b568e` |
| `geometry/v03.py` | `b4590754757b65f28c684ad09d84ec9f217a469f48654545aab4310990ff69e5` |
| `integration/v03_inputs.py` | `2f378c3a30d666a40880e9cc92f6ef31c59d4a439ab58d49931be2d25eeb8f7f` |
| `configs/v03_geometry_contract.json` | `f4ca974937f4e479adf396f7ad76e5ea305031103c12d6383ae8d38e8252e692` |
| `configs/v03_encoding.json` | `e1ccc728cfeca2a8bbd3ab01ae74c92052146b221ede42ac566ff0b2108a4255` |

The configuration copies are `configs/v03_geometry.json` and
`configs/v03_encoding.json`. No frozen source file was edited.

## Verified executable behavior

`scripts/verify_geometry.py` compared a synthetic asymmetric image and 50 actual
frames (first ten manifest episodes, five evenly spaced frames each).
Canonical RGB, wrist RGB, depth, validity, CameraXYZ and ActionCoord match
exactly. All policy inputs match exactly for R0 neutral, R1 Depth, R2 CameraXYZ
and R3 ActionCoord. The actual OpenPI `LiberoInputs` third-slot hook also matches.

There is a pre-existing metadata discrepancy: the JSON says RGB uses Lanczos,
but frozen `prepare_sample` calls `openpi_client.image_tools.resize_with_pad`
with its actual Bilinear default (PIL enum 2). RulerVLA retains **Bilinear** and
records this discrepancy rather than changing the frozen JSON or RGB pipeline.

The replay smoke also retains V0.3's float64 promotion of the normalized depth
buffer before converting to metres and storing float32. Converting the buffer
directly in float32 initially caused a maximum 0.0324 mm discrepancy; this was
fixed in the new smoke script by matching V0.3, without relaxing its tolerance.

## Data used in this engineering validation

The authoritative `configs/v03_training_manifest.json` defines **46 regenerated
task-2 episodes / 5,343 frames**, with recorded same-state RGB-D and state8
sidecars. This differs from the earlier A0-A3 432-episode training subset.
This validation uses the V0.3 manifest as requested and does not alter either
dataset or start training.

RGB-D source:
`/home/Dataset2/yqt/openpi-aimbot/spatialcue_depth_source/regenerated_rgbd_v02`

State8 sidecars:
`/home/Dataset2/yqt/actioncoord-v03-20261003T164806/data/state8_task2_all_2`

The real smoke reads the original HDF5 states linked by `source_frame_index`.
It restores simulator states and renders observations; it performs no control
steps, policy inference or success evaluation.

## Scientific state

V0.3 R2 completed a 5k pilot. The latest R3 pilot failed around 4.2k due to a
runtime/native issue. There is therefore **no completed CameraXYZ versus
ActionCoord superiority conclusion**. V0.5 currently has software and geometry
evidence only, not a policy-performance result.

## Protected projects

The frozen source remained clean at the same HEAD. The existing dirty states
of the two historical repositories were preserved; their `git status
--porcelain` output SHA-256 remained unchanged:

```text
openpi0-aimbot:
250fd3aa511fd92535f329618373bbb93320fcdb47a3ae8c262df47852da0bcd
actioncoord/repo:
9cd4d799c7335fd46bca0ab786828a582d39b7aa7828b2a69b8e007878430415
```

No old checkpoint was deleted and no Dataset2 data was overwritten. CodeGraph
was not initialized, following the user's earlier instruction.

## Repository identity

The authenticated GitHub account is **Li-XuanYi**, ID **182248291**. The user
explicitly approved this account instead of the distinct similarly named
account. The repository is private at `Li-XuanYi/RulerVLA`.

All new commits use repository-local `user.name=Lixuanyi` and
`user.email=1483177339@qq.com`, read from the configured WSL identity. No global
Git configuration was changed and no server owner's GitHub credentials were
used for publication.
