"""Main-comparison cues, using the frozen geometry and official renderers."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import numpy as np

from rulervla.adapters.openpi import OpenPIAdapter, resize_rgb, resize_float_map
from rulervla.geometry.action_coord import canonical_geometry, raster


class MainCueAdapter:
    def __init__(self, config, method, *, aimbot_source=None, axisguide_source=None):
        self.method = method
        self.config = deepcopy(config)
        if method != 'RulerVLA-main':
            self.config['method']['representation'] = 'neutral'
            self.config['method']['metric_ruler']['enabled'] = False
        self.base = OpenPIAdapter(self.config)
        if method == 'AimBot':
            sys.path.insert(0, str(Path(aimbot_source)))
            from crosshair.reticle_builder import ReticleBuilder
            from crosshair.config import CONFIG_DICT
            settings = deepcopy(CONFIG_DICT['large_crosshair_dynamic_default_color'])
            self.builder = ReticleBuilder(settings['shooting_line'], settings['scope_reticle'])
        elif method == 'AxisGuide-adapted':
            spec = importlib.util.spec_from_file_location('main_axisguide_reference', axisguide_source)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.axis = module._make_motion_basis_wrist_axis_rgb_tensor_cam_to_world
        elif method not in {'Baseline', 'RulerVLA-main'}:
            raise ValueError('Unknown main method')

    def prepare(self, sample):
        output = self.base.prepare(sample)
        neutral = np.zeros((224, 224, 3), np.float32)
        output['observation/wrist_cue'] = neutral.copy()
        if self.method in {'Baseline', 'RulerVLA-main'}:
            return output
        contract = self.config['geometry']
        maps = canonical_geometry(sample, contract)
        k = np.array(sample['intrinsic'], np.float64, copy=True)
        wk = np.array(sample['wrist_intrinsic'], np.float64, copy=True)
        k[:2, 2] -= .5
        wk[:2, 2] -= .5
        policy_raster = contract['canonical_to_policy']
        if self.method == 'AimBot':
            # Official builder expects optical world-to-camera (+Z forward).
            agent, wrist = self.builder.render_on_front_and_wst_camera(
                front_camera_rgb=maps['rgb'], front_camera_depth=maps['depth'],
                front_camera_extrinsics=np.linalg.inv(sample['camera_to_world']), front_camera_intrinsics=k,
                wrist_camera_rgb=maps['wrist'],
                wrist_camera_depth=raster(sample['wrist_depth_m'], contract['stored_depth_to_canonical']),
                wrist_camera_extrinsics=np.linalg.inv(sample['wrist_camera_to_world']), wrist_camera_intrinsics=wk,
                gripper_pos=sample['ee_world_m'], gripper_quat=sample['eef_quaternion_xyzw'],
                gripper_open=bool(sample['state8'][6] > .02), gripper_qpos=sample['state8'][6:8],
                image_height=256, image_width=256, front_tolerance=10, wrist_tolerance=10)
            output['observation/image'] = resize_rgb(raster(agent, policy_raster))
            output['observation/wrist_image'] = resize_rgb(raster(wrist, policy_raster))
        else:
            import torch
            # The checked upstream projection explicitly expects MuJoCo axes
            # (+Y up, -Z forward); our cache stores optical axes. Undo exactly
            # that convention once, without changing RGB orientation.
            axis_correction = np.diag([1., -1., -1., 1.])
            pose = np.concatenate([sample['ee_world_m'], sample['eef_quaternion_xyzw']])
            for rgb, intrinsic, transform, key in [
                    (maps['rgb'], k, sample['camera_to_world'], 'observation/spatial_image'),
                    (maps['wrist'], wk, sample['wrist_camera_to_world'], 'observation/wrist_cue')]:
                tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.
                cue, _ = self.axis(tensor, intrinsic, transform @ axis_correction, pose,
                    origin_robot=True, arrow_len=60, line_thickness=2, return_overlay=False)
                cue = cue.permute(1, 2, 0).numpy().astype(np.float32) * 2 - 1
                output[key] = resize_float_map(raster(cue, policy_raster))
        return output
