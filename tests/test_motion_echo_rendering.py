from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from rulervla.adapters.libero import LiberoAdapter
from rulervla.adapters.metric_cue import MetricViewAdapter
from rulervla.config import load_metric_method_config
from rulervla.cues.metric_ruler import render_ruler_cue
from rulervla.cues.motion_echo import MotionHistory, compose_cues, render_motion_echo
from rulervla.data.schema import Camera
from rulervla.geometry.camera import resize_camera


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def measured():
    with np.load(ROOT / "outputs/metric_cue_pilot/real_sample_1261.npz", allow_pickle=False) as data:
        sample = {key: data[key] for key in data.files}
    sample["task"] = str(sample["task"])
    geometry, ruler, echo = load_metric_method_config(ROOT / "configs/metric_ruler.yaml")
    history = MotionHistory(True, int(sample["episode_index"]), int(sample["frame_index"]),
                            int(sample["frame_index"]) - 5, sample["ee_world_m"], sample["past_ee_world_m"])
    return sample, geometry, ruler, replace(echo, enabled=True), history


def view_camera(measured, view):
    sample, geometry, _, _, _ = measured
    return resize_camera(LiberoAdapter(geometry).camera(sample, view), (224, 224))


@pytest.mark.parametrize("view", ["agentview", "wrist"])
def test_measured_echo_antialias_and_independent_projection(measured, view):
    _, _, _, echo, history = measured
    camera = view_camera(measured, view)
    cue, info = render_motion_echo(history, camera, echo)
    assert info["rendered"] and info["head_rendered"]
    assert np.any((cue > 0) & (cue < 1))
    assert cue.shape == (224, 224, 3) and cue.dtype == np.float32
    assert np.isfinite(cue).all() and 0 <= cue.min() <= cue.max() <= 1
    assert not np.any(cue[..., 2])
    np.testing.assert_array_equal(cue[..., 0], cue[..., 1])
    inverse = np.linalg.inv(camera.camera_to_action)
    for name, point in (("past_uv", history.past_m), ("current_uv", history.current_m)):
        optical = (inverse @ np.r_[point, 1])[:3]
        pixel = camera.intrinsic @ optical
        np.testing.assert_allclose(info[name], pixel[:2] / pixel[2], atol=1e-9)
    repeated, repeated_info = render_motion_echo(history, camera, echo)
    np.testing.assert_array_equal(cue, repeated)
    assert info == repeated_info
    json.dumps(info, allow_nan=False)


@pytest.mark.parametrize("view", ["agentview", "wrist"])
def test_measured_overlap_preserves_all_ruler_channels(measured, view):
    _, _, ruler, echo, history = measured
    camera = view_camera(measured, view)
    reference, _ = render_ruler_cue(history.current_m, camera, ruler)
    motion, _ = render_motion_echo(history, camera, echo)
    result = compose_cues(reference, motion, enabled=True)
    assert np.any((reference[..., 2] > 0) & (motion[..., 0] > 0))
    assert np.all(result >= reference) and np.all(result >= motion)
    np.testing.assert_array_equal(result[..., 2], reference[..., 2])
    np.testing.assert_array_equal(compose_cues(reference, motion, enabled=False), reference)


@pytest.mark.parametrize("view", ["agentview", "wrist"])
def test_measured_segment_crosses_near_plane_without_losing_visible_part(measured, view):
    import cv2
    _, _, _, echo, history = measured
    camera = view_camera(measured, view)
    points = np.stack((history.past_m, history.current_m))
    _, depth = camera.project(points)
    near = float(depth.mean())
    cue, info = render_motion_echo(history, camera, replace(echo, near_clip_m=near))
    assert info["rendered"] and info["near_clipped"] and np.any(cue)
    assert info["head_rendered"] == bool(depth[1] >= near)
    intersection = (history.past_m + history.current_m) / 2
    projected, _ = camera.project(np.stack((intersection, points[int(depth[1] >= near)])))
    expected = projected if depth[0] < near else projected[::-1]
    first, last = np.rint(expected * 256).astype(int)
    visible, first, last = cv2.clipLine((0, 0, 223*256+1, 223*256+1), tuple(first), tuple(last))
    assert visible
    np.testing.assert_allclose(info["endpoints_uv"], np.array([first, last])/256, atol=.02)


@pytest.mark.parametrize("view", ["agentview", "wrist"])
def test_measured_current_endpoint_outside_view_has_no_false_border_arrow(measured, view):
    import cv2
    _, _, _, echo, history = measured
    camera = view_camera(measured, view)
    uv, _ = camera.project(np.stack((history.past_m, history.current_m)))
    intrinsic = camera.intrinsic.copy()
    axis = int(np.argmax(np.abs(uv[1] - uv[0])))
    boundary = -2. if uv[1, axis] < uv[0, axis] else 225.
    intrinsic[axis, 2] += boundary - uv[1, axis]
    shifted = Camera(intrinsic, camera.camera_to_action, camera.resolution)
    cue, info = render_motion_echo(history, shifted, echo)
    assert info["rendered"] and info["viewport_clipped"] and np.any(cue)
    assert not info["current_visible"] and not info["head_rendered"]
    assert 0 <= info["endpoints_uv"][1][axis] <= 223
    expected = np.zeros(camera.resolution, np.uint8)
    first, last = np.rint(np.array(info["endpoints_uv"])*256).astype(np.int32)
    cv2.line(expected, tuple(first), tuple(last), 255, echo.arrow_width, cv2.LINE_AA, 8)
    np.testing.assert_array_equal(cue[..., 0], expected.astype(np.float32)/255.)


@pytest.mark.parametrize("view", ["agentview", "wrist"])
@pytest.mark.parametrize("length_px", [.5, 3.])
def test_measured_short_projection_keeps_endpoint_and_scales_head(measured, view, length_px):
    _, _, _, echo, history = measured
    camera = view_camera(measured, view)
    uv, _ = camera.project(np.stack((history.past_m, history.current_m)))
    factor = length_px / np.linalg.norm(uv[1] - uv[0])
    intrinsic = camera.intrinsic.copy()
    intrinsic[:2, :2] *= factor
    intrinsic[:2, 2] = uv[1] - factor*(uv[1] - intrinsic[:2, 2])
    cue, info = render_motion_echo(history, Camera(intrinsic, camera.camera_to_action, camera.resolution), echo)
    np.testing.assert_allclose(info["current_uv"], uv[1], atol=1e-10)
    assert info["visible_length_px"] == pytest.approx(length_px)
    if length_px < 1:
        assert not np.any(cue) and info["reason"] == "subpixel_projection"
    else:
        assert info["rendered"] and info["head_rendered"]
        assert info["head_length_px"] <= .4*length_px + 1e-10


def test_measured_behind_camera_diagnostics_are_strict_json(measured):
    _, _, _, echo, history = measured
    camera = view_camera(measured, "agentview")
    _, depth = camera.project(np.stack((history.past_m, history.current_m)))
    transform = camera.camera_to_action.copy()
    transform[:3, 3] += transform[:3, 2] * (max(depth) + .01)
    cue, info = render_motion_echo(history, Camera(camera.intrinsic, transform, camera.resolution), echo)
    assert not np.any(cue) and not info["rendered"]
    assert info["past_uv"] is None and info["current_uv"] is None
    json.dumps(info, allow_nan=False)


def test_measured_prompt_reports_stationary_missing_and_view_visibility(measured):
    sample, geometry, ruler, echo, history = measured
    threshold = replace(echo, min_displacement_m=float(np.linalg.norm(history.displacement_m)) + .001)
    stationary = MetricViewAdapter(geometry, ruler, threshold).prepare(sample, history=history)
    assert "below the display threshold" in stationary["prompt"]
    assert "yellow arrow" not in stationary["prompt"]
    invalid = replace(history, valid=False, past_m=None, reason="insufficient_history")
    missing = MetricViewAdapter(geometry, ruler, echo).prepare(sample, history=invalid)
    assert "No valid Motion Echo" in missing["prompt"]
    depths = [max(view_camera(measured, view).project(np.stack((history.past_m, history.current_m)))[1])
              for view in ("agentview", "wrist")]
    clipped = MetricViewAdapter(geometry, ruler, replace(echo, near_clip_m=float(np.mean(depths))))
    prepared = clipped.prepare(sample, history=history)
    assert sum(item["echo"]["rendered"] for item in clipped.last_diagnostics.values()) == 1
    for view, info in clipped.last_diagnostics.items():
        assert f"{view.capitalize()}:" in prepared["prompt"]
        if not info["echo"]["rendered"]:
            assert f"{view.capitalize()}: not visible" in prepared["prompt"]


def test_enabled_echo_checkpoint_rejects_missing_render_contract_before_assets(measured, tmp_path):
    from rulervla.adapters.metric_pipeline import MetricPipeline
    from rulervla.training.metric_resources import load_run_config
    from rulervla.training.metric_workflow import CHECKPOINT_SCHEMA, pipeline_from_checkpoint

    sample, geometry, ruler, echo, history = measured
    run = load_run_config(ROOT / "configs/metric_workflow.yaml")
    base = MetricPipeline.from_files(run["cue_config"], run["norm_stats_dir"])
    pipeline = MetricPipeline(geometry, ruler, base.norm_stats, echo=echo)
    contract = pipeline.contract()
    contract.pop("motion_echo_rendering", None)
    (tmp_path / "metric_workflow.json").write_text(json.dumps(dict(
        schema=CHECKPOINT_SCHEMA, pipeline=contract, pipeline_sha256=pipeline.fingerprint())))
    with pytest.raises(ValueError, match="Echo rendering"):
        pipeline_from_checkpoint(tmp_path)
    from rulervla.training.main_data import SpatialDataset
    cache = ROOT / "outputs/metric_cue_pilot/g2_engineering/segmented_data_cache"
    dataset = SpatialDataset(cache / Path(run["data"]["rgbd_root"]).relative_to("/"),
        cache / Path(run["data"]["geometry_root"]).relative_to("/"),
        path_mappings={"/home/Dataset2": str(cache / "home/Dataset2")})
    sample, history = dataset[20], dataset.get_history(20)
    trained = pipeline.prepare_train(sample, history=history)
    inferred = pipeline.prepare_inference(sample, history=history)
    trained.pop("actions")
    import jax
    for actual, expected in zip(jax.tree.leaves(trained), jax.tree.leaves(inferred), strict=True):
        np.testing.assert_array_equal(actual, expected)
