import numpy as np
import pytest

from rulervla.geometry.visual_anchor import AnchorObservation, LearnedAnchorProvider, sample_depth


def test_gt_surface_projection_and_confidence(config, sample, landmarks):
    _, provider = landmarks
    anchors = provider.detect(sample["agentview_rgb"])
    assert anchors.visible.all() and anchors.confidence.tolist() == [1.] * 8
    np.testing.assert_allclose(anchors.uv[0], [80., 90.])


def test_occluded_and_invalid_anchors_are_hidden(sample, landmarks):
    _, provider = landmarks
    provider.depth = provider.depth.copy()
    provider.depth[90, 80] = .5
    provider.depth[90, 110] = np.nan
    anchors = provider.detect(sample["agentview_rgb"])
    assert anchors.visible.sum() == 6
    np.testing.assert_array_equal(anchors.visible[:2], False)


def test_lookup_does_not_clamp_or_wrap_outside_pixels():
    uv = np.array([[-1., 0], [10, 0], [np.nan, 0], [0, 0], [1e100, 0]])
    depth, valid, _ = sample_depth(np.ones((2, 2)), uv)
    assert valid.tolist() == [False, False, False, True, False]
    np.testing.assert_array_equal(depth[~valid], 0)


def test_untrained_learned_provider_fails():
    with pytest.raises(NotImplementedError):
        LearnedAnchorProvider().detect(np.zeros((256, 256, 3), dtype=np.uint8))


def test_visible_nonfinite_anchor_is_rejected():
    with pytest.raises(ValueError):
        AnchorObservation(np.array([[np.nan, 0.]]), np.ones(1), np.ones(1, dtype=bool))
