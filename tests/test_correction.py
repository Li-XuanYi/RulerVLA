import numpy as np
import pytest

from rulervla.evaluation.corruption import inject_shared_translation
from rulervla.geometry.correction import apply_correction, correct_from_anchors, estimate_translation
from rulervla.geometry.visual_anchor import AnchorObservation


def test_recovers_two_minus_one_half_centimetre_bias(sample, landmarks):
    points, provider = landmarks
    bias = np.array([.02, -.01, .005])
    nominal = inject_shared_translation(provider.camera, bias)
    result = correct_from_anchors(provider.detect(sample["agentview_rgb"]), points, provider.depth,
                                  nominal, sample["ee_world_m"], np.eye(3))
    assert result.accepted
    np.testing.assert_allclose(result.bias_xyz_m, bias, atol=1e-12)
    ground_truth = np.arange(36).reshape(3, 4, 3) / 50
    valid = np.ones((3, 4), dtype=bool)
    np.testing.assert_allclose(apply_correction(ground_truth+bias, valid, result), ground_truth, atol=1e-12)


def test_six_correct_two_outliers_not_pulled_to_outlier_mean():
    bias = np.array([.02, -.01, .005])
    residuals = np.tile(bias, (8, 1))
    residuals[-2:] += [[.2, .3, -.1], [.1, -.2, .4]]
    result = estimate_translation(residuals)
    assert result.accepted
    np.testing.assert_allclose(result.bias_xyz_m, bias, atol=.001)
    assert result.weights[-2:].max() < .02


@pytest.mark.parametrize("count", [0, 1, 2])
def test_insufficient_anchors_zero_correction(count):
    result = estimate_translation(np.full((count, 3), .02))
    assert not result.accepted and result.reason == "insufficient_anchors"
    np.testing.assert_array_equal(result.bias_xyz_m, 0)


def test_inconsistent_residuals_reject():
    result = estimate_translation(np.eye(3) * .1)
    assert not result.accepted
    np.testing.assert_array_equal(result.bias_xyz_m, 0)


def test_large_consistent_bias_is_accepted():
    bias = [.4, -.3, .2]
    result = estimate_translation(np.tile(bias, (8, 1)))
    assert result.accepted
    np.testing.assert_allclose(result.bias_xyz_m, bias)


def test_duplicated_pixel_is_not_independent_evidence(sample, landmarks):
    points, provider = landmarks
    a = provider.detect(sample["agentview_rgb"])
    repeated = AnchorObservation(np.tile(a.uv[0], (8, 1)), a.confidence, a.visible)
    result = correct_from_anchors(repeated, np.tile(points[0], (8, 1)), provider.depth,
                                  provider.camera, sample["ee_world_m"], np.eye(3))
    assert not result.accepted and result.valid_anchors.sum() == 1


def test_invalid_and_low_confidence_residuals_not_used():
    residuals = np.tile([.02, -.01, .005], (5, 1))
    residuals[-1] = np.nan
    result = estimate_translation(residuals, confidence=[1, 1, 1, .1, 1])
    assert result.accepted and result.valid_anchors.sum() == 3
    np.testing.assert_array_equal(result.residuals_m[~result.valid_anchors], 0)
