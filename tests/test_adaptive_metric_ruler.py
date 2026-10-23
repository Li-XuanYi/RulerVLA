"""Regression coverage for the approved fixed-unit V2 renderer."""
import numpy as np
from rulervla.cues.adaptive_metric_ruler import AdaptiveRulerConfig, render_adaptive_metric_ruler, clip_segment_to_viewport
from rulervla.data.schema import Camera


def test_fixed_label_is_optional_and_constant():
    cam=Camera(np.array([[200.,0,127.5],[0,200.,127.5],[0,0,1]]),np.eye(4))
    image=np.zeros((256,256,3),np.uint8)
    close,a=render_adaptive_metric_ruler(image,[0,0,1],cam,AdaptiveRulerConfig(render_unit_label=True))
    far,b=render_adaptive_metric_ruler(image,[0,0,2],cam,AdaptiveRulerConfig(render_unit_label=True))
    assert a.unit_m == b.unit_m == .05
    np.testing.assert_array_equal(close[:25,:110],far[:25,:110])
    default,_=render_adaptive_metric_ruler(image,[0,0,1],cam)
    assert not default[:25,:110].any()


def test_viewport_crossing_and_parallel_invisible_segments():
    a,b=clip_segment_to_viewport([-10,40],[300,40],(180,320))
    np.testing.assert_allclose(a,[0,40]);np.testing.assert_allclose(b,[300,40])
    assert clip_segment_to_viewport([-10,-5],[300,-5],(180,320)) is None
    assert clip_segment_to_viewport([np.nan,0],[4,5],(180,320)) is None


def test_no_large_outside_line_when_viewport_clipping_disabled():
    cam=Camera(np.array([[240.,0,127.5],[0,240.,127.5],[0,0,1]]),np.eye(4))
    image=np.zeros((256,256,3),np.uint8)
    _,info=render_adaptive_metric_ruler(image,[.48,0,1],cam,AdaptiveRulerConfig(clip_to_viewport=False))
    assert not any(s.viewport_clipped for s in info.segments)
