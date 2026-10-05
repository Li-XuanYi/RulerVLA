from copy import deepcopy
import numpy as np
import pytest

from rulervla.adapters.openpi import OpenPIAdapter, OpenPIThirdSlotInputs
from rulervla.config import profile
from rulervla.evaluation.corruption import inject_shared_translation


@pytest.mark.parametrize("name", ["V03", "V05-C", "V05-CR", "V05-Full"])
def test_three_separate_images_float_spatial_and_no_wrist_change(config, sample, landmarks, name):
    points, provider = landmarks
    adapter = OpenPIAdapter(profile(config, name))
    clean_wrist = OpenPIAdapter(profile(config, "V03")).prepare(sample)["observation/wrist_image"]
    sample["camera_to_world"] = inject_shared_translation(provider.camera, [.02, -.01, .005]).camera_to_action
    output = adapter.prepare(sample, anchor_provider=provider, local_points_ee_m=points)
    for key in ("observation/image", "observation/wrist_image", "observation/spatial_image"):
        assert output[key].shape == (224, 224, 3)
    assert output["observation/spatial_image"].dtype == np.float32
    assert np.isfinite(output["observation/spatial_image"]).all()
    np.testing.assert_array_equal(output["observation/wrist_image"], clean_wrist)
    assert output["prompt"] == sample["task"]
    if name != "V03":
        np.testing.assert_allclose(adapter.last_diagnostics["correction"].bias_xyz_m, [.02, -.01, .005], atol=1e-10)
    assert adapter.last_diagnostics["ruler_rendered"] == (name == "V05-Full")


def test_corrected_output_restores_v03_encoding(config, sample, landmarks):
    points, provider = landmarks
    expected = OpenPIAdapter(profile(config, "V03")).prepare(sample)
    sample["camera_to_world"] = inject_shared_translation(provider.camera, [.02, -.01, .005]).camera_to_action
    actual = OpenPIAdapter(profile(config, "V05-C")).prepare(sample, anchor_provider=provider, local_points_ee_m=points)
    np.testing.assert_array_equal(actual["observation/image"], expected["observation/image"])
    np.testing.assert_allclose(actual["observation/spatial_image"], expected["observation/spatial_image"], atol=1e-7)


def test_downstream_three_slot_hook_preserves_float_spatial(config, sample):
    data = OpenPIAdapter(profile(config, "V03")).prepare(sample)
    def base(prepared):
        return {"image": {"base_0_rgb": prepared["observation/image"],
                          "left_wrist_0_rgb": prepared["observation/wrist_image"],
                          "right_wrist_0_rgb": np.zeros((224, 224, 3), dtype=np.uint8)},
                "image_mask": {"base_0_rgb": True, "left_wrist_0_rgb": True, "right_wrist_0_rgb": False}}
    output = OpenPIThirdSlotInputs(base_transform=base)(data)
    assert len(output["image"]) == 3 and all(output["image_mask"].values())
    assert output["image"]["right_wrist_0_rgb"] is data["observation/spatial_image"]
    assert output["image"]["right_wrist_0_rgb"].dtype == np.float32


def test_live_and_stored_raster_have_same_inputs(config, sample):
    from rulervla.adapters.libero import LiberoAdapter
    observation = {"agentview_image": sample["agentview_rgb"][::-1, ::-1],
                   "robot0_eye_in_hand_image": sample["wrist_rgb"][::-1, ::-1],
                   "robot0_eef_pos": sample["ee_world_m"],
                   "robot0_eef_quat": sample["eef_quaternion_xyzw"]}
    live = LiberoAdapter(config["geometry"]).from_observation(
        observation, depth_m=sample["agentview_depth_m"][::-1, ::-1],
        intrinsic=sample["intrinsic"], camera_to_world=sample["camera_to_world"],
        state8=sample["state8"], instruction=sample["task"])
    adapter = OpenPIAdapter(profile(config, "V03"))
    a, b = adapter.prepare(sample), adapter.prepare(live)
    for key in ("observation/image", "observation/wrist_image", "observation/spatial_image"):
        np.testing.assert_array_equal(a[key], b[key])


def test_reject_wrong_state_geometry_pair(config, sample):
    sample["state8"][0] += .1
    with pytest.raises(ValueError):
        OpenPIAdapter(profile(config, "V03")).prepare(sample)


def test_configured_learned_backend_is_explicit_stub(config, sample):
    config["method"]["visual_anchor"]["backend"] = "learned"
    with pytest.raises(NotImplementedError):
        OpenPIAdapter(config).prepare(sample)


def test_rejected_anchors_do_not_draw_ruler_or_random_correction(config, sample, landmarks):
    points, provider = landmarks
    provider.depth = np.zeros_like(provider.depth)
    adapter = OpenPIAdapter(profile(config, "V05-Full"))
    out = adapter.prepare(sample, anchor_provider=provider, local_points_ee_m=points)
    assert not adapter.last_diagnostics["correction"].accepted
    assert not adapter.last_diagnostics["ruler_rendered"]
    assert not out["observation/spatial_image"].any()
