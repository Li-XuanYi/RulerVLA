"""Serialisable CUSUM calibration for the Accel monitor.

The point of this module is that a threshold is only meaningful together with
the configuration that produced the scores it was fitted on.  ``fm-geometry``
states the contract plainly for its detector -- two runs agree bit-for-bit only
when ``mode`` / ``fixed_std`` / ``n_exec`` match -- and a calibration file that
does not carry that identity is a number waiting to be misapplied.  So the
artifact stores the fitted thresholds *and* the score identity
(:class:`~openpi.uncertainty.accel.AccelConfig`) plus a digest of the reference
scale, and :meth:`AccelCalibration.mismatches` is what a server should call
once at startup rather than re-deriving the check per request.
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import pathlib
from typing import Any

import numpy as np

from openpi.uncertainty.accel import AccelConfig

FORMAT = "openpi.accel.calibration/1"
FILENAME = "calibration.json"


def digest_std(std: np.ndarray) -> str:
    """Return a stable digest of a reference scale, for artifact checking."""

    return hashlib.sha256(np.ascontiguousarray(std, dtype=np.float32).tobytes()).hexdigest()


@dataclasses.dataclass(frozen=True)
class AccelCalibration:
    """Fitted CUSUM thresholds plus the identity of the scores they apply to."""

    mu0: float
    sigma: float
    k: float
    h: float
    score: AccelConfig
    fpr: float = 0.1
    k_sigma: float = 0.25
    protocol: str = "held-out"
    conformal: bool = True
    n_calib: int = 0
    n_succ: int = 0
    n_fail: int = 0
    fixed_std_kind: str = "none"
    fixed_std_path: str | None = None
    fixed_std_sha256: str | None = None
    fixed_std_length: int | None = None
    metrics: dict[str, float] = dataclasses.field(default_factory=dict)
    created_utc: str = ""

    def __post_init__(self) -> None:
        for name in ("mu0", "sigma", "k", "h", "fpr", "k_sigma"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"calibration {name} must be finite; got {value!r}")
        if not 0.0 < self.fpr < 1.0:
            raise ValueError(f"fpr must be in (0, 1), got {self.fpr}")
        if self.protocol not in {"held-out", "in-sample"}:
            raise ValueError(f"unknown calibration protocol {self.protocol!r}")
        if not self.created_utc:
            object.__setattr__(self, "created_utc", datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"))

    def mismatches(self, live: AccelConfig, *, fixed_std: np.ndarray | None = None) -> list[str]:
        """Return every way ``live`` disagrees with the configuration fitted here.

        An empty list means the thresholds apply as-is.  A non-empty list is not
        a warning: scores computed under a different identity are on a different
        scale, so the fitted ``h`` has no meaning for them.
        """

        problems = self.score.mismatches(live)
        if (
            fixed_std is not None
            and self.fixed_std_sha256 is not None
            and digest_std(fixed_std) != self.fixed_std_sha256
        ):
            problems.append(
                "fixed_std: the live reference scale differs from the one this calibration was fitted "
                "against (digest mismatch), so the scores are not on the calibrated scale"
            )
        if fixed_std is None and self.fixed_std_sha256 is not None:
            problems.append(
                "fixed_std: calibrated with a fixed reference scale but none is loaded, so scores will be "
                "self-normalised per chunk and are not on the calibrated scale"
            )
        return problems

    def to_artifact(self) -> dict[str, Any]:
        """Return the JSON-serialisable artifact."""

        payload: dict[str, Any] = {
            "format": FORMAT,
            "created_utc": self.created_utc,
            "provenance": self.score.to_artifact(),
            "fixed_std": {
                "kind": self.fixed_std_kind,
                "path": self.fixed_std_path,
                "sha256": self.fixed_std_sha256,
                "length": self.fixed_std_length,
            },
            "mu0": self.mu0,
            "sigma": self.sigma,
            "k": self.k,
            "h": self.h,
            "fpr": self.fpr,
            "k_sigma": self.k_sigma,
            "protocol": self.protocol,
            "conformal": self.conformal,
            "n_calib": self.n_calib,
            "n_succ": self.n_succ,
            "n_fail": self.n_fail,
            "metrics": self.metrics,
        }
        return payload

    @classmethod
    def from_artifact(cls, payload: dict[str, Any]) -> AccelCalibration:
        """Rebuild a calibration from :meth:`to_artifact` output."""

        fmt = payload.get("format")
        if fmt != FORMAT:
            raise ValueError(f"unsupported calibration format {fmt!r}; expected {FORMAT!r}")
        fixed_std = payload.get("fixed_std") or {}
        return cls(
            mu0=float(payload["mu0"]),
            sigma=float(payload["sigma"]),
            k=float(payload["k"]),
            h=float(payload["h"]),
            score=AccelConfig.from_artifact(payload["provenance"]),
            fpr=float(payload.get("fpr", 0.1)),
            k_sigma=float(payload.get("k_sigma", 0.25)),
            protocol=str(payload.get("protocol", "in-sample")),
            conformal=bool(payload.get("conformal", False)),
            n_calib=int(payload.get("n_calib", 0)),
            n_succ=int(payload.get("n_succ", 0)),
            n_fail=int(payload.get("n_fail", 0)),
            fixed_std_kind=str(fixed_std.get("kind", "none")),
            fixed_std_path=fixed_std.get("path"),
            fixed_std_sha256=fixed_std.get("sha256"),
            fixed_std_length=None if fixed_std.get("length") is None else int(fixed_std["length"]),
            metrics={str(key): float(value) for key, value in (payload.get("metrics") or {}).items()},
            created_utc=str(payload.get("created_utc", "")),
        )

    def save(self, path: pathlib.Path | str) -> None:
        """Write the artifact to ``path``.

        A path ending in ``.json`` is used as-is; any other path is treated as a
        directory and gets ``calibration.json`` inside it.
        """

        target = pathlib.Path(path)
        if target.suffix != ".json":
            target = target / FILENAME
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_artifact(), indent=2))

    @classmethod
    def load(cls, path: pathlib.Path | str) -> AccelCalibration:
        """Read an artifact written by :meth:`save`."""

        target = pathlib.Path(path)
        if target.suffix != ".json":
            target = target / FILENAME
        if not target.exists():
            raise FileNotFoundError(f"Accel calibration not found at: {target}")
        return cls.from_artifact(json.loads(target.read_text()))


__all__ = ["FILENAME", "FORMAT", "AccelCalibration", "digest_std"]
