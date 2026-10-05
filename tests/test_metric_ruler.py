import numpy as np
import pytest

from rulervla.data.schema import CorrectedEEReference
from rulervla.cues.metric_ruler import project_metric_ruler, render_metric_ruler
from rulervla.evaluation.corruption import inject_shared_translation


def test_physical_ticks_and_depth_order(sample, landmarks):
    _, provider = landmarks
    ref = CorrectedEEReference(sample["ee_world_m"], np.zeros(3), True)
    projection = project_metric_ruler(ref, provider.camera)
    for axis in range(3):
        expected = np.outer([.05, .10, .15], np.eye(3)[axis])
        np.testing.assert_allclose(projection.points_action_m[axis] - sample["ee_world_m"], expected)
    assert np.all(np.diff(projection.ticks_uv[0, :, 0]) > 0)
    assert np.all(np.diff(projection.ticks_uv[1, :, 1]) > 0)
    np.testing.assert_allclose(projection.optical_depth_m[2], [1.05, 1.10, 1.15])


def test_anchored_projection_removes_same_shared_bias(sample, landmarks):
    _, provider = landmarks
    bias = np.array([.02, -.01, .005])
    nominal = inject_shared_translation(provider.camera, bias)
    a = project_metric_ruler(CorrectedEEReference(sample["ee_world_m"], np.zeros(3), True), provider.camera)
    b = project_metric_ruler(CorrectedEEReference(sample["ee_world_m"], bias, True), nominal)
    np.testing.assert_allclose(a.ticks_uv, b.ticks_uv, atol=1e-12)


def test_camera_depth_changes_pixel_length_not_physical_length(sample, landmarks):
    _, provider = landmarks
    ref1 = CorrectedEEReference(np.array([0., 0, 1]), np.zeros(3), True)
    ref2 = CorrectedEEReference(np.array([0., 0, 2]), np.zeros(3), True)
    a, b = project_metric_ruler(ref1, provider.camera), project_metric_ruler(ref2, provider.camera)
    np.testing.assert_allclose(a.ticks_uv[0]-a.origin_uv, 2*(b.ticks_uv[0]-b.origin_uv))


def test_render_no_input_mutation_and_three_channels(sample, landmarks):
    _, provider = landmarks
    rgb = sample["agentview_rgb"].copy()
    output = render_metric_ruler(rgb, CorrectedEEReference(sample["ee_world_m"], np.zeros(3), True), provider.camera)
    assert output.shape == rgb.shape and output.dtype == np.uint8 and np.any(output != rgb)
    np.testing.assert_array_equal(rgb, sample["agentview_rgb"])


def test_ruler_refuses_unanchored_reference(sample, landmarks):
    _, provider = landmarks
    with pytest.raises(ValueError):
        project_metric_ruler(CorrectedEEReference(sample["ee_world_m"], np.zeros(3), False), provider.camera)
