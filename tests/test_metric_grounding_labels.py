import numpy as np
import pytest

from rulervla.cues.metric_ruler import MetricCueConfig
from rulervla.cues.metric_prompt import build_scale_prompt
from rulervla.data.schema import Camera
from rulervla.training.metric_grounding import make_ruler_query


def camera():
    k = np.array([[200., 0., 127.5], [0., 200., 127.5], [0., 0., 1.]])
    return Camera(k, np.eye(4))


def test_one_physical_point_has_spacing_dependent_reading_but_same_metres():
    ee = np.array([0., 0., 1.])
    point = np.array([.06, 0., 1.])
    a = make_ruler_query(ee, point, 0, camera(), MetricCueConfig(tick_spacing_m=.05),
                         view='agentview', canonical_to_policy='hflip')
    b = make_ruler_query(ee, point, 0, camera(), MetricCueConfig(tick_spacing_m=.03),
                         view='agentview', canonical_to_policy='hflip')
    assert a.signed_ticks == pytest.approx(1.2)
    assert b.signed_ticks == pytest.approx(2.0)
    assert a.displacement_m == b.displacement_m == pytest.approx(.06)
    np.testing.assert_allclose(a.pixel_uv_policy, b.pixel_uv_policy)
    assert '5 cm' in build_scale_prompt('pick bowl', MetricCueConfig(tick_spacing_m=.05))
    assert '3 cm' in build_scale_prompt('pick bowl', MetricCueConfig(tick_spacing_m=.03))
    assert set(a.model_input()) == {'view', 'pixel_uv_policy'}


def test_negative_axis_and_off_axis_or_outside_point():
    ee = np.array([0., 0., 1.])
    query = make_ruler_query(ee, [-.05, 0., 1.], 0, camera(), MetricCueConfig(),
                             view='agentview', canonical_to_policy='hflip')
    assert query.axis == 0 and query.sign == -1 and query.signed_ticks == pytest.approx(-1.)
    with pytest.raises(ValueError):
        make_ruler_query(ee, [.05, .01, 1.], 0, camera(), MetricCueConfig(),
                         view='agentview', canonical_to_policy='hflip')
    with pytest.raises(ValueError):
        make_ruler_query(ee, [.30, 0., 1.], 0, Camera(np.array([[5000., 0., 127.5],
            [0., 5000., 127.5], [0., 0., 1.]]), np.eye(4)), MetricCueConfig(),
            view='agentview', canonical_to_policy='hflip')
