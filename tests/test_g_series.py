from pathlib import Path
import numpy as np
import yaml

from rulervla.evaluation.g_series import (conditions, error_statistics, perturb,
    pixel_region, rotation, run_condition, spread_subsets, unit_direction)
from rulervla.geometry.action_coord import canonical_geometry


def design():
    return yaml.safe_load((Path(__file__).parents[1]/"configs/g_series.yaml").read_text())


def frame(sample, points, provider):
    uv = provider.detect(np.zeros((256, 256, 3), dtype=np.uint8)).uv
    return dict(episode_id=1, frame_id=0, anchor_uv=uv, anchor_visible=np.ones(8, dtype=bool),
                anchor_confidence=np.ones(8), local_points_ee_m=points, pool=list(range(8)),
                subsets=spread_subsets(uv, np.arange(8)), distance_tertiles_m=[.5, 1.],
                robot_mask=pixel_region((256, 256), uv, 2), anchor_roi=pixel_region((256, 256), uv, 2),
                target_roi=np.indices((256, 256))[0] > 200)


def test_frozen_matrix_has_unique_conditions_and_expected_rows():
    items = conditions(design())
    assert len(items) == 67
    assert len({(x["experiment_id"], x["condition"]) for x in items}) == 67
    assert len(items)*50 == 3350


def test_mixed_translation_is_magnitude_not_componentwise(design_unused=None):
    assert np.isclose(np.linalg.norm(unit_direction("mixed", design())), 1)


def test_rotation_changes_rotation_without_translation(landmarks, sample):
    points, provider = landmarks
    f = frame(sample, points, provider)
    item = dict(kind="rotation", axis="y", severity=3)
    _, camera, _, bias = perturb(provider.depth, provider.camera, points, provider.detect(sample["agentview_rgb"]), np.eye(3), item, f, design())
    np.testing.assert_array_equal(camera.camera_to_action[:3, 3], provider.camera.camera_to_action[:3, 3])
    np.testing.assert_allclose(camera.camera_to_action[:3, :3].T@camera.camera_to_action[:3, :3], np.eye(3), atol=1e-14)
    assert bias is None


def test_missing_values_are_not_zero_error_points():
    gt = np.ones((2, 2, 3))
    pred = gt.copy()
    pred[0, 0] = 0
    mask = np.ones((2, 2), dtype=bool)
    mask[0, 0] = False
    assert error_statistics(gt, pred, mask)["rms_mm"] == 0
    assert error_statistics(gt, pred, mask)["pixels"] == 3
    assert error_statistics(gt, pred, np.zeros((2, 2), dtype=bool))["rms_mm"] is None


def test_target_bias_leaves_anchor_measurement_pixels_clean(sample, landmarks):
    points, provider = landmarks
    f = frame(sample, points, provider)
    anchors = provider.detect(sample["agentview_rgb"])
    depth, _, _, _ = perturb(provider.depth, provider.camera, points, anchors, np.eye(3), dict(kind="target_bias", severity=10), f, design())
    pixels = np.rint(anchors.uv).astype(int)
    np.testing.assert_array_equal(depth[pixels[:, 1], pixels[:, 0]], provider.depth[pixels[:, 1], pixels[:, 0]])


def test_anchor_only_bias_keeps_nonanchor_scene_clean(sample, landmarks):
    points, provider = landmarks
    f = frame(sample, points, provider)
    depth, _, _, _ = perturb(provider.depth, provider.camera, points, provider.detect(sample["agentview_rgb"]), np.eye(3), dict(kind="anchor_bias", severity=5), f, design())
    np.testing.assert_array_equal(depth[~f["anchor_roi"]], provider.depth[~f["anchor_roi"]])


def test_noise_rms_seed_and_severity_are_exact(sample, landmarks):
    points, provider = landmarks
    f = frame(sample, points, provider)
    arguments = (provider.depth, provider.camera, points, provider.detect(sample["agentview_rgb"]), np.eye(3), dict(kind="noise", severity=5), f, design())
    a = perturb(*arguments)[0]
    b = perturb(*arguments)[0]
    np.testing.assert_array_equal(a, b)
    assert np.isclose(np.sqrt(np.mean((a-provider.depth)**2)), .005)
    assert abs(np.mean(a-provider.depth)) < 1e-12


def test_deterministic_subsets_have_no_repeated_pixels(sample, landmarks):
    points, provider = landmarks
    f = frame(sample, points, provider)
    assert f["subsets"] == spread_subsets(f["anchor_uv"], np.arange(8))
    assert all(len(set(x)) == 8 for x in f["subsets"].values())
    assert len({tuple(x[:3]) for x in f["subsets"].values()}) == 3


def test_two_anchors_reject_without_mutating_input(config, sample, landmarks):
    points, provider = landmarks
    f = frame(sample, points, provider)
    before = sample["camera_to_world"].copy()
    row, _ = run_condition(sample, f, config, design(), dict(experiment_id="G02", condition="n2_A", kind="anchor_count", severity=2, subset="A"))
    assert not row["accepted"] and row["num_anchor_used"] == 2
    assert row["failure_reason"] == "insufficient_anchors"
    np.testing.assert_array_equal(row["estimated_bias_xyz_m"], [0, 0, 0])
    np.testing.assert_array_equal(sample["camera_to_world"], before)


def test_outlier_correspondence_adds_fixed_world_residual(sample, landmarks):
    points, provider = landmarks
    f = frame(sample, points, provider)
    anchors = provider.detect(sample["agentview_rgb"])
    _, _, changed, _ = perturb(provider.depth, provider.camera, points, anchors, np.eye(3), dict(kind="outlier", severity=2), f, design())
    np.testing.assert_allclose(points[:2]-changed[:2], [[.02, 0, 0], [0, -.02, 0]])
    np.testing.assert_array_equal(points[2:], changed[2:])
