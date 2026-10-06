"""Common four-slot input contract; runtime integration must retain all slots."""
from dataclasses import dataclass

import numpy as np

MAIN_IMAGE_KEYS = ('base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb', 'aux_wrist_cue_rgb')


@dataclass(frozen=True)
class MainComparisonInputs:
    """Use the same RGB pair and two dedicated auxiliary slots for every arm.

    Auxiliaries are pre-encoded float32[-1,1]. Neutral input is exactly zero.
    This input hook alone does not extend OpenPI inputs_spec/preprocessing.
    A runtime parity gate must verify four slots survive training and inference.
    """

    model_type: object = None
    base_transform: object = None

    def __call__(self, data):
        for key in ('observation/image', 'observation/wrist_image'):
            value = np.asarray(data[key])
            if value.shape != (224, 224, 3) or value.dtype != np.uint8:
                raise ValueError(f'{key} must be uint8[224,224,3]')
        cues = [data['observation/spatial_image'], data['observation/wrist_cue']]
        for value in cues:
            array = np.asarray(value)
            if (array.shape != (224, 224, 3) or array.dtype != np.float32
                    or not np.isfinite(array).all() or np.max(np.abs(array)) > 1 + 1e-6):
                raise ValueError('Main auxiliary slots must be finite float32[224,224,3] in [-1,1]')
        transform = self.base_transform
        if transform is None:
            from openpi.policies.libero_policy import LiberoInputs
            transform = LiberoInputs(model_type=self.model_type)
        output = transform(data)
        output['image'] = {
            MAIN_IMAGE_KEYS[0]: output['image']['base_0_rgb'],
            MAIN_IMAGE_KEYS[1]: output['image']['left_wrist_0_rgb'],
            MAIN_IMAGE_KEYS[2]: cues[0],
            MAIN_IMAGE_KEYS[3]: cues[1],
        }
        output['image_mask'] = {key: np.bool_(True) for key in MAIN_IMAGE_KEYS}
        return output
