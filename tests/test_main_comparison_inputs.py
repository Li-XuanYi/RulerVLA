import numpy as np
import pytest

from rulervla.adapters.main_comparison import MAIN_IMAGE_KEYS, MainComparisonInputs
from rulervla.adapters.openpi import OpenPIAdapter
from rulervla.config import profile


def test_four_slots_preserve_both_cameras_aux_values_and_state(config, sample):
    prepared = OpenPIAdapter(profile(config, 'RulerVLA-main')).prepare(sample)
    prepared['observation/wrist_cue'] = np.zeros((224, 224, 3), np.float32)

    def base(data):
        return {'image': {'base_0_rgb': data['observation/image'],
                         'left_wrist_0_rgb': data['observation/wrist_image']},
                'state': data['observation/state'], 'actions': data['actions']}

    output = MainComparisonInputs(base_transform=base)(prepared)
    assert tuple(output['image']) == MAIN_IMAGE_KEYS
    assert all(output['image_mask'].values())
    assert output['image']['right_wrist_0_rgb'] is prepared['observation/spatial_image']
    assert output['image']['aux_wrist_cue_rgb'] is prepared['observation/wrist_cue']
    assert output['image']['left_wrist_0_rgb'] is prepared['observation/wrist_image']
    np.testing.assert_array_equal(output['state'], sample['state8'])
    np.testing.assert_array_equal(output['actions'], sample['actions'])


@pytest.mark.parametrize('value', [np.full((224, 224, 3), 2, np.float32),
                                 np.full((224, 224, 3), np.nan, np.float32),
                                 np.zeros((224, 224, 3), np.uint8)])
def test_auxiliary_contract_rejects_silent_rescaling(config, sample, value):
    prepared = OpenPIAdapter(profile(config, 'V03')).prepare(sample)
    prepared['observation/wrist_cue'] = value
    with pytest.raises(ValueError):
        MainComparisonInputs(base_transform=lambda _: {})(prepared)
