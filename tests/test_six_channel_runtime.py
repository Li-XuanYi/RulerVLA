import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import pytest

from openpi.models.pi0_config import Pi0Config
from openpi.training.weight_loaders import merge_six_channel_params


PATH = 'PaliGemma/img/embedding/kernel'


def tree(kernel, other_shape=(2,)):
    return {'PaliGemma': {'img': {'embedding': {'kernel': kernel}}},
            'other': {'weight': np.ones(other_shape, np.float32)}}


def test_pi05_four_slot_spec_has_six_channels():
    observation, _ = Pi0Config(pi05=True).inputs_spec()
    assert list(observation.images) == ['base_0_rgb', 'left_wrist_0_rgb',
                                        'right_wrist_0_rgb', 'aux_wrist_cue_rgb']
    assert all(value.shape == (1, 224, 224, 6) and value.dtype == jnp.float32
               for value in observation.images.values())


def test_only_siglip_patch_kernel_expands_with_exact_rgb_prefix():
    old = np.arange(2*2*3*4, dtype=np.float32).reshape(2, 2, 3, 4)
    reference = tree(np.empty((2, 2, 6, 4), np.float32))
    migrated = merge_six_channel_params(tree(old), reference)
    kernel = migrated['PaliGemma']['img']['embedding']['kernel']
    np.testing.assert_array_equal(kernel[..., :3, :], old)
    np.testing.assert_array_equal(kernel[..., 3:, :], 0)
    np.testing.assert_array_equal(migrated['other']['weight'], 1)


def test_migration_rejects_any_other_shape_mismatch():
    reference = tree(np.empty((2, 2, 6, 4), np.float32))
    with pytest.raises(ValueError):
        merge_six_channel_params(tree(np.zeros((2, 2, 4, 4), np.float32)), reference)
    with pytest.raises(ValueError):
        merge_six_channel_params(tree(np.zeros((2, 2, 3, 4), np.float32), (3,)), reference)


def test_zero_new_kernel_gives_same_stem_output_and_new_channel_gradients():
    old = np.arange(2*2*3*2, dtype=np.float32).reshape(2, 2, 3, 2) / 100
    migrated = merge_six_channel_params(tree(old), tree(np.empty((2, 2, 6, 2), np.float32)))
    kernel = jnp.asarray(migrated['PaliGemma']['img']['embedding']['kernel'])
    rgb = jnp.arange(1*4*4*3, dtype=jnp.float32).reshape(1, 4, 4, 3) / 255
    cue = jnp.full((1, 4, 4, 3), .25, jnp.float32)
    conv = lambda x, w: lax.conv_general_dilated(x, w, (2, 2), 'VALID',
                                                  dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    np.testing.assert_allclose(conv(rgb, jnp.asarray(old)),
                               conv(jnp.concatenate((rgb, cue), -1), kernel), atol=1e-6)
    gradient = jax.grad(lambda w: conv(jnp.concatenate((rgb, cue), -1), w).sum())(kernel)
    assert np.any(np.asarray(gradient[..., 3:, :]) != 0)


def test_rgb_cue_patch_reduction_preserves_exact_rgb_numerics():
    from openpi.models.siglip import rgb_cue_conv_general_dilated
    rgb = jax.random.normal(jax.random.key(18), (1, 224, 224, 3))
    kernel = jax.random.normal(jax.random.key(19), (14, 14, 3, 32))
    cue = jax.random.uniform(jax.random.key(20), rgb.shape)
    expanded = jnp.concatenate((kernel, jnp.zeros_like(kernel)), axis=2)
    kwargs = dict(dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    expected = jax.jit(lambda x, w: lax.conv_general_dilated(x, w, (14, 14), 'VALID', **kwargs))(rgb, kernel)
    actual = jax.jit(lambda x, w: rgb_cue_conv_general_dilated(x, w, (14, 14), 'VALID', **kwargs))(
        jnp.concatenate((rgb, cue), axis=-1), expanded)
    np.testing.assert_array_equal(actual, expected)


def test_siglip_bf16_rgb_prefix_is_exact_with_zero_initialized_cue():
    import copy
    from openpi.models import siglip
    image = jax.random.uniform(jax.random.key(21), (1, 28, 28, 3))
    module = siglip.Module(num_classes=16, variant='mu/14', pool_type='none', scan=True,
                           dtype_mm='bfloat16', head_zeroinit=False)
    variables = module.init(jax.random.key(22), image, train=False)
    expanded = copy.deepcopy(variables)
    kernel = variables['params']['embedding']['kernel']
    expanded['params']['embedding']['kernel'] = jnp.concatenate((kernel, jnp.zeros_like(kernel)), axis=2)
    expected, _ = module.apply(variables, image, train=False)
    actual, _ = module.apply(expanded, jnp.concatenate((image, jnp.ones_like(image)), axis=-1), train=False)
    np.testing.assert_array_equal(actual, expected)
