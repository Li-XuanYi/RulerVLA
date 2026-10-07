"""Actual first-frame cue and projection audit for all ten Spatial tasks."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rgbd-root', type=Path)
    parser.add_argument('--geometry-root', type=Path)
    parser.add_argument('--suite-root', type=Path, help='Combined completed RGB-D/geometry for Object, Goal or Long')
    parser.add_argument('--aimbot-source', type=Path, required=True)
    parser.add_argument('--axisguide-source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root/'src'))
    from rulervla.adapters.main_cues import MainCueAdapter
    from rulervla.config import load_config
    from rulervla.adapters.openpi import resize_rgb
    args.out.mkdir(parents=True, exist_ok=True)
    if args.suite_root:
        assert args.rgbd_root is None and args.geometry_root is None
        manifest=json.loads((args.suite_root/'manifest.json').read_text())
        assert manifest['complete']
        rows=manifest['episodes']
        geo_rows=[dict(row,geometry_npz=row['npz']) for row in rows]
    else:
        assert args.rgbd_root is not None and args.geometry_root is not None
        rows = json.loads((args.rgbd_root/'regen_manifest.json').read_text())['episodes']
        geo_rows = json.loads((args.geometry_root/'manifest.json').read_text())['episodes']
    geometry = {row['episode_index']: row for row in geo_rows}
    methods = ('Baseline', 'AimBot', 'AxisGuide-adapted', 'RulerVLA-main')
    adapters = {name: MainCueAdapter(load_config(root/'configs/rulervla_main.yaml'), name,
        aimbot_source=args.aimbot_source, axisguide_source=args.axisguide_source) for name in methods}
    selected = [next(row for row in rows if row['task'] == task) for task in sorted({row['task'] for row in rows})]
    assert len(selected)==10
    sheet = Image.new('RGB', (224*8, 248*len(selected)), 'white')
    draw = ImageDraw.Draw(sheet)
    wrist_sheet = Image.new('RGB', (224*8, 248*len(selected)), 'white')
    wrist_draw = ImageDraw.Draw(wrist_sheet)
    results = []
    for task_index, row in enumerate(selected):
        with np.load(row['npz']) as rgbd, np.load(geometry[row['episode_index']]['geometry_npz']) as geo:
            sample = {key: rgbd[key][0].copy() for key in ('agentview_rgb', 'wrist_rgb', 'agentview_depth_m', 'ee_world_m')}
            sample.update({key: geo[key][0].copy() for key in ('state8', 'eef_quaternion_xyzw', 'wrist_depth_m',
                'intrinsic', 'camera_to_world', 'wrist_intrinsic', 'wrist_camera_to_world')})
            sample['task'] = row['task']
            sample['actions'] = rgbd['actions'][np.minimum(np.arange(10), row['frames']-1)].copy()
        prepared = {name: adapter.prepare(sample) for name, adapter in adapters.items()}
        baseline = prepared['Baseline']
        np.testing.assert_array_equal(baseline['observation/image'], resize_rgb(sample['agentview_rgb']))
        np.testing.assert_array_equal(baseline['observation/wrist_image'], resize_rgb(sample['wrist_rgb']))
        for name in methods:
            value = prepared[name]
            np.testing.assert_array_equal(value['observation/state'], baseline['observation/state'])
            np.testing.assert_array_equal(value['actions'], baseline['actions'])
            assert value['prompt'] == baseline['prompt']
            if name in {'Baseline', 'AimBot'}:
                assert not value['observation/spatial_image'].any() and not value['observation/wrist_cue'].any()
            if name in {'Baseline', 'AxisGuide-adapted', 'RulerVLA-main'}:
                np.testing.assert_array_equal(value['observation/wrist_image'], baseline['observation/wrist_image'])
            if name == 'AxisGuide-adapted':
                np.testing.assert_array_equal(value['observation/image'], baseline['observation/image'])
            if name == 'RulerVLA-main':
                assert not value['observation/wrist_cue'].any()
                assert value['observation/spatial_image'].any()
            for slot in ('observation/spatial_image', 'observation/wrist_cue'):
                assert np.isfinite(value[slot]).all() and np.max(np.abs(value[slot])) <= 1
        # Independent optical projection agrees with the upstream MuJoCo-axis
        # projection for EE and each +xyz probe in both cameras.
        axis_module = sys.modules.get('main_axisguide_reference')
        # importlib loader does not populate sys.modules; obtain the function's
        # original globals directly from the reused function object instead.
        project = adapters['AxisGuide-adapted'].axis.__globals__['project_world_point_to_pixel_cam_to_world']
        projection_errors = []
        for k_key, t_key in [('intrinsic', 'camera_to_world'), ('wrist_intrinsic', 'wrist_camera_to_world')]:
            k = sample[k_key].copy()
            k[:2, 2] -= .5
            transform = sample[t_key]
            for offset in [np.zeros(3), *(.02*np.eye(3))]:
                point = sample['ee_world_m'] + offset
                camera = np.linalg.inv(transform) @ np.r_[point, 1.]
                if camera[2] <= 1e-6:
                    continue
                expected = (k @ camera[:3])[:2]/camera[2]
                actual = project(k, transform @ np.diag([1., -1., -1., 1.]), point)
                np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=0)
                projection_errors.append(float(np.max(np.abs(actual-expected))))
        changes = {}
        for method_index, name in enumerate(methods):
            value = prepared[name]
            changes[name] = int(np.count_nonzero(value['observation/image'] != baseline['observation/image']))
            for view_index, key in enumerate(('observation/image', 'observation/spatial_image')):
                image = value[key] if view_index == 0 else np.round((value[key]+1)*127.5).astype(np.uint8)
                x, y = (method_index*2+view_index)*224, task_index*248
                sheet.paste(Image.fromarray(image), (x,y+24))
                draw.text((x+2,y+3), name + (' RGB' if view_index == 0 else ' aux'), fill='black')
            for view_index, key in enumerate(('observation/wrist_image', 'observation/wrist_cue')):
                image = value[key] if view_index == 0 else np.round((value[key]+1)*127.5).astype(np.uint8)
                x, y = (method_index*2+view_index)*224, task_index*248
                wrist_sheet.paste(Image.fromarray(image), (x,y+24))
                wrist_draw.text((x+2,y+3), name + (' wrist' if view_index == 0 else ' wrist aux'), fill='black')
        results.append(dict(task=row['task'], episode_index=row['episode_index'], shared_state_actions_prompt=True,
            max_axis_projection_error_px=max(projection_errors), agent_rgb_changed_channels=changes))
    sheet.save(args.out/'contact_sheet.png')
    wrist_sheet.save(args.out/'wrist_contact_sheet.png')
    from crosshair.config import CONFIG_DICT
    import dataclasses
    settings = CONFIG_DICT['large_crosshair_dynamic_default_color']
    report = dict(passed=True, tasks=10, methods=list(methods), records=results,
        reused_aimbot_renderer=True, reused_axisguide_renderer=True,
        aim_settings={key: dataclasses.asdict(value) for key,value in settings.items()},
        aim_gripper_open_rule='current measured finger qpos[0] > 0.02m; no future action input',
        axis_projection_contract='optical c2w @ diag(1,-1,-1,1) exactly once; -.5 principal offset once',
        official_sources_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
            [args.aimbot_source/'crosshair/reticle_builder.py', args.aimbot_source/'crosshair/config.py', args.axisguide_source]},
        visual_review_pending=True, formal_training_started=False)
    (args.out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
