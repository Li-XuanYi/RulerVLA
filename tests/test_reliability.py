import numpy as np

from rulervla.geometry.correction import estimate_translation
from rulervla.geometry.reliability import ReliabilityConfig, geometry_reliability


def test_score_depends_on_fit_not_bias_magnitude():
    small = estimate_translation(np.tile([.02, -.01, .005], (8, 1)))
    large = estimate_translation(np.tile([.6, -.7, .8], (8, 1)))
    depth_valid = np.ones((256, 256), dtype=bool)
    a = geometry_reliability(small, np.ones(8), depth_valid)
    b = geometry_reliability(large, np.ones(8), depth_valid)
    assert a.correction_valid and b.correction_valid
    np.testing.assert_allclose(a.fit_score, b.fit_score, atol=1e-12)


def test_invalid_patch_suppressed_partial_patch_weighted():
    result = estimate_translation(np.tile([.02, 0, 0], (8, 1)))
    valid = np.ones((32, 32), dtype=bool)
    valid[:16, :16] = False
    valid[:8, 16:] = False
    reliability = geometry_reliability(result, np.ones(8), valid, ReliabilityConfig(patch_size=16))
    np.testing.assert_allclose(reliability.patch_weights, [[0, .5], [1, 1]])


def test_no_anchor_evidence_neutralizes_spatial_slot():
    result = estimate_translation(np.empty((0, 3)))
    reliability = geometry_reliability(result, np.empty(0), np.ones((256, 256), dtype=bool))
    assert not reliability.correction_valid and reliability.fit_score == 0
    assert not reliability.patch_weights.any()
