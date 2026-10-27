from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rulervla.adapters.libero import LiberoAdapter
from rulervla.adapters.metric_cue import MetricViewAdapter
from rulervla.config import load_metric_method_config
from rulervla.cues.metric_ruler import MetricRulerConfig, project_ruler_cue, render_ruler_cue
from rulervla.cues.metric_prompt import build_scale_prompt
from rulervla.geometry.camera import resize_camera


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/metric_ruler.yaml"


@pytest.fixture(scope="module")
def real_sample():
    with np.load(ROOT / "outputs/metric_cue_pilot/real_sample_1261.npz", allow_pickle=False) as data:
        sample = {key: data[key] for key in data.files}
    sample["task"] = str(sample["task"])
    return sample


def test_ruler_configuration_rejects_removed_options():
    assert MetricRulerConfig().extent_m == .15
    for fields in ({"renderer_version": "metric_ruler_positive15cm_v3"},
                   {"arrow_length_px": 5}, {"major_tick_every": 2},
                   {"negative_style": "dashed"}):
        with pytest.raises(ValueError, match="Unsupported ruler configuration"):
            MetricRulerConfig.from_mapping(fields)
    with pytest.raises(ValueError):
        replace(MetricRulerConfig(), extent_m=-.15)


@pytest.mark.parametrize("view", ["agentview", "wrist"])
def test_real_ruler_keeps_positive_geometry_without_arrowheads(real_sample, view):
    geometry, ruler, _ = load_metric_method_config(CONFIG)
    camera = resize_camera(LiberoAdapter(geometry).camera(real_sample, view), (224, 224))
    cue, projection = render_ruler_cue(real_sample["ee_world_m"], camera, ruler)
    assert len(projection.ticks) == 9
    assert cue.shape == (224, 224, 3) and cue.dtype == np.float32
    assert np.isfinite(cue).all() and 0 <= cue.min() <= cue.max() <= 1
    for item in projection.ticks:
        assert item.index in (1, 2, 3)
        np.testing.assert_allclose(item.world_m - real_sample["ee_world_m"],
                                   np.eye(3)[item.axis] * item.index * .05, atol=1e-15, rtol=0)
    inverse = np.linalg.inv(camera.camera_to_action)
    for item in projection.ticks:
        optical = (inverse @ np.r_[item.world_m, 1])[:3]
        if optical[2] > 0:
            pixel = camera.intrinsic @ optical
            np.testing.assert_allclose(item.uv, pixel[:2] / pixel[2], atol=1e-9)
    for name in "xyz":
        info = projection.diagnostics[name]
        assert info["rendered_ticks"] + len(info["omitted_ticks"]) == 3
        assert all(value >= 6 - 1e-9 for value in info["rendered_tick_separation_px"])
        assert "arrow" not in info and "positive_arrow_rendered" not in info
    prepared = MetricViewAdapter(geometry, ruler).prepare(real_sample)
    key = "observation/agent_cue" if view == "agentview" else "observation/wrist_cue"
    with np.load(ROOT / "outputs/metric_cue_pilot/positive15_v3_validation/delivery/before_siglip_inputs.npz",
                 allow_pickle=False) as previous:
        old_cue = previous[f"frame20_{view}"][..., 3:]
    assert np.all(prepared[key] <= old_cue + 1e-7)
    assert np.any(prepared[key] < old_cue - .1)
    repeated, _ = render_ruler_cue(real_sample["ee_world_m"], camera, ruler)
    np.testing.assert_array_equal(cue, repeated)


def test_prompt_describes_lines_and_equivalent_ticks():
    _, ruler, _ = load_metric_method_config(CONFIG)
    prompt = build_scale_prompt("Pick up the black bowl.", ruler)
    assert "0 cm to +15 cm" in prompt and "5 cm" in prompt and "equivalent" in prompt
    assert all(word not in prompt.lower() for word in ("arrow", "negative", "major", "dashed"))


def test_real_tokenizer_refuses_truncation_and_keeps_echo_meaning(real_sample):
    from openpi.models.tokenizer import PaligemmaTokenizer
    from rulervla.cues.metric_prompt import StrictPromptTokenizer
    from rulervla.cues.motion_echo import MotionHistory, MotionEchoConfig

    ruler = MetricRulerConfig()
    prompt = build_scale_prompt("Pick up the black bowl.", ruler)
    with pytest.raises(ValueError, match="truncate"):
        StrictPromptTokenizer(PaligemmaTokenizer(5)).tokenize(prompt)
    tokens, mask = StrictPromptTokenizer(PaligemmaTokenizer(200)).tokenize(prompt)
    assert tokens.shape == mask.shape == (200,) and mask.sum() > 0
    geometry, _, _ = load_metric_method_config(CONFIG)
    adapter = MetricViewAdapter(geometry, ruler, MotionEchoConfig(enabled=True))
    history = MotionHistory(True, int(real_sample["episode_index"]), 20, 15,
                            real_sample["ee_world_m"], real_sample["past_ee_world_m"])
    valid = adapter.prepare(real_sample, history=history)["prompt"]
    missing = adapter.prepare(real_sample)["prompt"]
    assert "5 environment control steps" in valid and "No valid Motion Echo" in missing
    StrictPromptTokenizer(PaligemmaTokenizer(200)).tokenize(valid)


@pytest.mark.parametrize("view", ["agentview", "wrist"])
def test_real_camera_near_plane_has_visible_clipped_shaft(real_sample, view):
    import cv2

    geometry, ruler, _ = load_metric_method_config(CONFIG)
    camera = resize_camera(LiberoAdapter(geometry).camera(real_sample, view), (224, 224))
    inverse = np.linalg.inv(camera.camera_to_action)
    optical = (inverse @ np.r_[real_sample["ee_world_m"], 1])[:3]
    optical *= ruler.near_clip_m * 1.2 / optical[2]
    origin = camera.camera_to_action[:3, 3] + camera.camera_to_action[:3, :3] @ optical
    cue, info = render_ruler_cue(origin, camera, ruler)
    crossings = 0
    for axis, name in enumerate("xyz"):
        for segment in info.diagnostics[name]["segments"]:
            if not segment["near_clipped"]:
                continue
            stop = (inverse @ np.r_[origin + np.eye(3)[axis] * ruler.extent_m, 1])[:3]
            fraction = (ruler.near_clip_m - optical[2]) / (stop[2] - optical[2])
            points = np.stack((optical, optical + fraction * (stop - optical))) @ camera.intrinsic.T
            pixels = points[:, :2] / points[:, 2, None]
            first, last = np.rint(pixels * 256).astype(int)
            valid, first, last = cv2.clipLine((0, 0, 223 * 256 + 1, 223 * 256 + 1), tuple(first), tuple(last))
            assert valid
            expected = np.array([first, last]) / 256
            np.testing.assert_allclose(segment["endpoints_uv"], expected, atol=.02, rtol=0)
            u, v = np.rint(expected.mean(axis=0)).astype(int)
            assert np.any(cue[max(0, v-2):v+3, max(0, u-2):u+3, axis] > 0)
            crossings += 1
    assert np.isfinite(cue).all()
    assert crossings >= 1


def test_real_pipeline_matches_training_and_inference():
    import jax
    from openpi.models.model import Observation, preprocess_observation
    from rulervla.adapters.metric_pipeline import MetricPipeline
    from rulervla.training.main_data import SpatialDataset
    from rulervla.training.metric_resources import load_run_config
    from rulervla.training.metric_workflow import load_metric_policy

    run = load_run_config(ROOT / "configs/metric_workflow.yaml")
    cache = ROOT / "outputs/metric_cue_pilot/g2_engineering/segmented_data_cache"
    dataset = SpatialDataset(cache / Path(run["data"]["rgbd_root"]).relative_to("/"),
        cache / Path(run["data"]["geometry_root"]).relative_to("/"),
        path_mappings={"/home/Dataset2": str(cache / "home/Dataset2")})
    pipeline = MetricPipeline.from_files(run["cue_config"], run["norm_stats_dir"])
    with pytest.raises(ValueError, match="saved preprocessing"):
        load_metric_policy(ROOT / "outputs/metric_cue_pilot/g1_real_checkpoint_stable_stem/checkpoints/1",
                           explicit_pipeline=pipeline)
    for index in (20, 108, 109, 110):
        sample = dataset[index]
        train = pipeline.prepare_train(sample)
        infer = pipeline.prepare_inference(sample)
        train.pop("actions")
        for left, right in zip(jax.tree.leaves(train), jax.tree.leaves(infer), strict=True):
            np.testing.assert_array_equal(left, right)
            assert left.dtype == right.dtype
        observation = Observation.from_dict(jax.tree.map(lambda value: np.asarray(value)[None], infer))
        for left, right in zip(jax.tree.leaves(preprocess_observation(jax.random.key(42), observation, train=True)),
                               jax.tree.leaves(preprocess_observation(None, observation, train=False)), strict=True):
            np.testing.assert_array_equal(left, right)
        assert list(infer["image_mask"].values()) == [True, True, False, False]
        assert all(value.shape == (224, 224, 6) and value.dtype == np.float32 for value in infer["image"].values())


def test_projection_does_not_depend_on_rendering_style(real_sample):
    geometry, ruler, _ = load_metric_method_config(CONFIG)
    camera = LiberoAdapter(geometry).camera(real_sample, "agentview")
    first = project_ruler_cue(real_sample["ee_world_m"], camera, ruler)
    second = project_ruler_cue(real_sample["ee_world_m"], camera, replace(ruler, line_width_px=3))
    for left, right in zip(first.ticks, second.ticks, strict=True):
        np.testing.assert_array_equal(left.world_m, right.world_m)
        np.testing.assert_array_equal(left.uv, right.uv)


@pytest.mark.parametrize("view", ["agentview", "wrist"])
def test_real_direction_and_blank_modes_report_actual_output(real_sample, view):
    geometry, ruler, _ = load_metric_method_config(CONFIG)
    camera = resize_camera(LiberoAdapter(geometry).camera(real_sample, view), (224, 224))
    direction = replace(ruler, mode="direction")
    cue, info = render_ruler_cue(real_sample["ee_world_m"], camera, direction)
    assert np.any(cue)
    for name in "xyz":
        assert info.diagnostics[name]["segment_unit"] == "pixels"
        assert info.diagnostics[name]["rendered_ticks"] == 0
        for segment in info.diagnostics[name]["segments"]:
            start, end = np.asarray(segment["endpoints_uv"])
            assert np.linalg.norm(end-start) <= ruler.direction_length_px + 1e-7
    blank, _ = render_ruler_cue(real_sample["ee_world_m"], camera, replace(ruler, enabled=False, mode="blank"))
    assert not np.any(blank)
