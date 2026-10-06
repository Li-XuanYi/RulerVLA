from copy import deepcopy

import numpy as np
import pytest

from rulervla.adapters.openpi import OpenPIAdapter
from rulervla.config import profile, validate_method
from rulervla.cues.metric_ruler import project_metric_ruler
from rulervla.data.schema import CorrectedEEReference, NominalEEReference


def test_main_never_uses_anchor_correction_or_reliability(config, sample, monkeypatch):
    import rulervla.adapters.openpi as implementation

    def forbidden(*args, **kwargs):
        raise AssertionError("Anchor/Correction/Reliability must not enter the main method")

    monkeypatch.setattr(implementation, "correct_from_anchors", forbidden)
    monkeypatch.setattr(implementation, "geometry_reliability", forbidden)
    monkeypatch.setattr(implementation.LearnedAnchorProvider, "detect", forbidden)
    base = OpenPIAdapter(profile(config, "V03")).prepare(sample)
    adapter = OpenPIAdapter(profile(config, "RulerVLA-main"))
    actual = adapter.prepare(sample)
    np.testing.assert_array_equal(actual["observation/spatial_image"], base["observation/spatial_image"])
    np.testing.assert_array_equal(actual["observation/wrist_image"], base["observation/wrist_image"])
    np.testing.assert_array_equal(actual["observation/state"], base["observation/state"])
    np.testing.assert_array_equal(actual["actions"], base["actions"])
    assert actual["prompt"] == base["prompt"]
    assert np.any(actual["observation/image"] != base["observation/image"])
    assert adapter.last_diagnostics["ruler_rendered"]
    assert adapter.last_diagnostics["reference_source"] == "nominal_ee"
    assert adapter.last_diagnostics["correction"] is None
    assert adapter.last_diagnostics["reliability"] is None
    assert isinstance(adapter.last_diagnostics["reference"], NominalEEReference)


def test_nominal_projection_uses_physical_world_axes_and_uncorrected_camera(sample, landmarks):
    from rulervla.evaluation.corruption import inject_shared_translation

    _, provider = landmarks
    reference = NominalEEReference(sample["ee_world_m"])
    projection = project_metric_ruler(reference, provider.camera)
    anchored = project_metric_ruler(CorrectedEEReference(sample["ee_world_m"], np.zeros(3), True), provider.camera)
    np.testing.assert_array_equal(projection.ticks_uv, anchored.ticks_uv)
    expected = reference.position_action_m[None, None, :] + np.eye(3)[:, None, :] * np.array([.05, .10, .15])[None, :, None]
    np.testing.assert_array_equal(projection.points_action_m, expected)
    biased = inject_shared_translation(provider.camera, [.02, .01, 0])
    actual = project_metric_ruler(reference, biased)
    np.testing.assert_array_equal(actual.ticks_uv, biased.project(expected)[0])
    assert not np.allclose(actual.ticks_uv, projection.ticks_uv)


@pytest.mark.parametrize("module", ["visual_anchor", "correction", "reliability"])
def test_nominal_main_rejects_hidden_anchor_dependencies(config, module):
    main = profile(config, "RulerVLA-main")
    main["method"][module]["enabled"] = True
    with pytest.raises(ValueError):
        validate_method(main["method"])


def test_main_is_immutable_profile_and_retains_legacy_anchor_guard(config):
    before = deepcopy(config)
    profile(config, "RulerVLA-main")
    assert config == before
    invalid = profile(config, "V03")
    invalid["method"]["metric_ruler"]["enabled"] = True
    with pytest.raises(ValueError):
        validate_method(invalid["method"])
