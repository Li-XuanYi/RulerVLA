import importlib.util
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest


def checker():
    path = Path(__file__).parents[1] / 'scripts/prepare_main_spatial_geometry.py'
    spec = importlib.util.spec_from_file_location('spatial_geometry', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = Path(__file__).parents[1] / 'outputs' / ('rgb_validation_' + uuid4().hex)
    out.mkdir(parents=True)
    return module.check_rgb, out


def test_signed_quantization_is_recorded_and_reference_not_changed():
    check, out = checker()
    frozen = np.full((256, 256, 3), 62, np.uint8)
    actual = frozen.copy()
    actual[152, 4, :2] -= 1
    result = check(actual, frozen, out, 1271, 13, 'wrist')
    assert result['signed_max_abs_delta'] == 1
    assert result['mismatched_channels'] == 2
    assert result['within_rgb_quantization_bound']
    np.testing.assert_array_equal(frozen, np.full_like(frozen, 62))


def test_large_colour_or_raster_shift_stops_preparation():
    check, out = checker()
    frozen = np.full((256, 256, 3), 62, np.uint8)
    actual = frozen.copy()
    actual[0, 0, 0] += 2
    with pytest.raises(ValueError):
        check(actual, frozen, out, 1, 0, 'agent')
    actual = frozen + 1
    with pytest.raises(ValueError):
        check(actual, frozen, out, 1, 1, 'agent')
