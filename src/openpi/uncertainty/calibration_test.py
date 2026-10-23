import json

import numpy as np
import pytest

from openpi.uncertainty.accel import AccelConfig
from openpi.uncertainty.calibration import FORMAT
from openpi.uncertainty.calibration import AccelCalibration
from openpi.uncertainty.calibration import digest_std


def _score_config(**overrides) -> AccelConfig:
    payload = {"action_dim": 7, "n_exec": 10, "mode": "accel_prefix:3", "num_steps": 10}
    payload.update(overrides)
    return AccelConfig(**payload)


def _calibration(**overrides) -> AccelCalibration:
    payload = {
        "mu0": 1.8,
        "sigma": 0.4,
        "k": 0.1,
        "h": 2.7,
        "score": _score_config(),
        "fpr": 0.1,
        "k_sigma": 0.25,
        "protocol": "held-out",
        "conformal": True,
        "n_calib": 30,
        "n_succ": 120,
        "n_fail": 60,
        "metrics": {"tpr": 0.55, "median_lead": 3.0, "succ_fire_rate": 0.07},
    }
    payload.update(overrides)
    return AccelCalibration(**payload)


def test_round_trip_through_a_file(tmp_path):
    std = np.linspace(0.5, 1.5, 70, dtype=np.float32)
    original = _calibration(
        fixed_std_kind="npz",
        fixed_std_path=str(tmp_path / "accel_fixed_std.npz"),
        fixed_std_sha256=digest_std(std),
        fixed_std_length=70,
    )
    path = tmp_path / "calibration.json"
    original.save(path)
    assert path.exists()

    restored = AccelCalibration.load(path)
    assert restored.mu0 == original.mu0
    assert restored.h == original.h
    assert restored.protocol == "held-out"
    assert restored.conformal is True
    assert restored.n_calib == 30
    assert restored.score == original.score
    assert restored.fixed_std_sha256 == original.fixed_std_sha256
    assert restored.fixed_std_length == 70
    assert restored.metrics["tpr"] == 0.55
    assert restored.created_utc == original.created_utc

    # A directory path gets the canonical filename inside it.
    directory = tmp_path / "out"
    original.save(directory)
    assert (directory / "calibration.json").exists()
    assert AccelCalibration.load(directory).h == original.h


def test_provenance_keeps_the_score_identity():
    payload = _calibration().to_artifact()
    assert payload["format"] == FORMAT
    assert payload["provenance"] == {
        "mode": "accel_prefix:3",
        "action_dim": 7,
        "n_exec": 10,
        "num_steps": 10,
        "n_prefix_layers": 9,
    }
    # Round-tripping the artifact must not lose the identity it carries.
    assert AccelCalibration.from_artifact(payload).score == _score_config()


def test_mismatches_names_every_disagreeing_field():
    calibration = _calibration()
    assert calibration.mismatches(_score_config()) == []

    for overrides in (
        {"mode": "accel_prefix:7"},
        {"n_exec": 5},
        {"action_dim": 32},
        {"num_steps": 20},
    ):
        problems = calibration.mismatches(_score_config(**overrides))
        assert len(problems) == 1
        assert next(iter(overrides)) in problems[0]


def test_mismatches_on_the_reference_scale():
    std = np.ones(70, dtype=np.float32)
    calibration = _calibration(fixed_std_sha256=digest_std(std), fixed_std_length=70)

    # Same length, different values: only the digest catches this, and it is the
    # one check that keeps a score on the scale the threshold was fitted to.
    tampered = std.copy()
    tampered[0] = 1.0001
    problems = calibration.mismatches(_score_config(), fixed_std=tampered)
    assert any("digest mismatch" in problem for problem in problems)

    assert calibration.mismatches(_score_config(), fixed_std=std) == []

    # Calibrated with a fixed scale but scoring without one: the live score
    # self-normalises and is not on the calibrated scale.
    problems = calibration.mismatches(_score_config(), fixed_std=None)
    assert any("none is loaded" in problem for problem in problems)


def test_rejects_unusable_thresholds():
    for field in ("mu0", "sigma", "k", "h"):
        with pytest.raises(ValueError, match=f"calibration {field} must be finite"):
            _calibration(**{field: float("nan")})
    with pytest.raises(ValueError, match="fpr must be in"):
        _calibration(fpr=1.5)
    with pytest.raises(ValueError, match="unknown calibration protocol"):
        _calibration(protocol="guesswork")


def test_rejects_an_unrecognised_artifact_format(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"format": "openpi.accel.calibration/0", "mu0": 1.0}))
    with pytest.raises(ValueError, match="unsupported calibration format"):
        AccelCalibration.load(path)
    with pytest.raises(FileNotFoundError):
        AccelCalibration.load(tmp_path / "absent.json")
