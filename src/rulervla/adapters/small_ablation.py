"""Controlled supplements using the existing geometry, renderer and four slots.

This adapter does not change the frozen main-comparison methods.
"""
from copy import deepcopy

import numpy as np

from rulervla.adapters.openpi import OpenPIAdapter, resize_float_map
from rulervla.geometry.action_coord import canonical_geometry, encode, raster


METHODS = {
    'Baseline': ('neutral', False),
    'Ruler-only': ('neutral', True),
    'ActionCoord-only': ('actioncoord', False),
    'RulerVLA-main': ('actioncoord', True),
    'CameraXYZ-only': ('camera_xyz', False),
    'CameraEE-only': ('camera_ee', False),
    'ActionXYZ-only': ('action_xyz', False),
}


class SmallAblationAdapter:
    def __init__(self, config, method):
        representation, ruler = METHODS[method]
        self.representation = representation
        self.config = deepcopy(config)
        for key in ('visual_anchor', 'correction', 'reliability'):
            self.config['method'][key]['enabled'] = False
        self.config['method']['representation'] = representation if representation in {
            'neutral', 'actioncoord', 'camera_xyz'} else 'neutral'
        self.config['method']['metric_ruler'].update(enabled=ruler, reference='nominal_ee')
        self.base = OpenPIAdapter(self.config)

    def prepare(self, sample):
        output = self.base.prepare(sample)
        output['observation/wrist_cue'] = np.zeros((224, 224, 3), np.float32)
        if self.representation in {'camera_ee', 'action_xyz'}:
            maps = canonical_geometry(sample, self.config['geometry'])
            transform = np.asarray(sample['camera_to_world'], np.float64)
            rotation, translation = transform[:3, :3], transform[:3, 3]
            if self.representation == 'camera_ee':
                ee_camera = rotation.T @ (sample['ee_world_m'] - translation)
                values = maps['camera_xyz'] - ee_camera
            else:
                values = maps['camera_xyz'] @ rotation.T + translation
            spatial = encode(values, maps['valid'], self.config['encoding_config'])
            output['observation/spatial_image'] = resize_float_map(
                raster(spatial, self.config['geometry']['canonical_to_policy']))
        return output
