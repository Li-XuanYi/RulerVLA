import numpy as np
import pytest

from rulervla.adapters.metric_cue import SixChannelInputs


def prepared():
    return {'observation/image': np.full((224, 224, 3), 128, np.uint8),
            'observation/wrist_image': np.full((224, 224, 3), 255, np.uint8),
            'observation/agent_cue': np.full((224, 224, 3), .25, np.float32),
            'observation/wrist_cue': np.full((224, 224, 3), .5, np.float32),
            'observation/state': np.zeros(8, np.float32), 'prompt': 'pick bowl'}


def test_each_valid_view_is_normalized_rgb_then_independent_cue():
    data = prepared()
    source = data['observation/image'].copy()
    result = SixChannelInputs()(data)
    assert list(result['image']) == ['base_0_rgb', 'left_wrist_0_rgb',
                                     'right_wrist_0_rgb', 'aux_wrist_cue_rgb']
    assert [bool(result['image_mask'][key]) for key in result['image']] == [True, True, False, False]
    for image in result['image'].values():
        assert image.shape == (224, 224, 6) and image.dtype == np.float32
    np.testing.assert_allclose(result['image']['base_0_rgb'][..., :3], 128/255*2-1, atol=1e-6)
    np.testing.assert_allclose(result['image']['base_0_rgb'][..., 3:], .25)
    np.testing.assert_allclose(result['image']['left_wrist_0_rgb'][..., :3], 1.)
    np.testing.assert_allclose(result['image']['left_wrist_0_rgb'][..., 3:], .5)
    assert not np.any(result['image']['right_wrist_0_rgb'])
    np.testing.assert_array_equal(data['observation/image'], source)


def test_nonfinite_cue_and_wrong_shape_fail_closed():
    data = prepared()
    data['observation/agent_cue'][0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        SixChannelInputs()(data)
    data = prepared()
    data['observation/wrist_cue'] = data['observation/wrist_cue'][..., :2]
    with pytest.raises(ValueError):
        SixChannelInputs()(data)


@pytest.mark.parametrize('model_type', [None, 'pi0', 'pi05'])
def test_real_conversion_preserves_state_prompt_and_optional_actions(model_type):
    data = prepared()
    data['observation/state'] = np.arange(8, dtype=np.float64)
    data['actions'] = np.arange(70, dtype=np.float32).reshape(10, 7)
    result = SixChannelInputs(model_type=model_type)(data)
    assert set(result) == {'state', 'prompt', 'actions', 'image', 'image_mask'}
    assert result['state'] is data['observation/state']
    assert result['state'].dtype == np.float64
    assert result['prompt'] == data['prompt']
    assert result['actions'] is data['actions']
    np.testing.assert_array_equal(result['actions'], np.arange(70, dtype=np.float32).reshape(10, 7))
    del data['actions']
    del data['prompt']
    result = SixChannelInputs(model_type=model_type)(data)
    assert set(result) == {'state', 'image', 'image_mask'}
