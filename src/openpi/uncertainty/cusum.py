"""Causal CUSUM monitoring for per-decision Accel scores.

The formulas are the small, policy-independent part of
``fm-geometry/fmaccel/detection/cusum.py``. ``AccelMonitor`` is intentionally
stateful but action-agnostic: callers decide what to do after an alarm (stop,
replan, or log it). It never modifies a policy's action chunk itself.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import dataclasses
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from openpi.uncertainty.calibration import AccelCalibration


# -------------------------------------------------------------------diagnostics
def auroc(fail_vals: Sequence[float], succ_vals: Sequence[float]) -> tuple[float, int]:
    """P(random failure score > random success score); 0.5 = no signal.

    Mann-Whitney U with tie-averaged ranks. Returns ``(auroc, n_pairs)`` and
    ``(nan, 0)`` when either group is empty -- an empty group is a data problem,
    not a score of zero, and callers must see the pair count to tell them apart.
    Ported from ``fm-geometry`` so the two report the same statistic.
    """

    fail = np.asarray(fail_vals, float)
    succ = np.asarray(succ_vals, float)
    if fail.size == 0 or succ.size == 0:
        return float("nan"), 0
    allv = np.concatenate([fail, succ])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty_like(order, float)
    ranks[order] = np.arange(1, allv.size + 1)
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(cnt.size)
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    rank_sum = ranks[: fail.size].sum()
    u = rank_sum - fail.size * (fail.size + 1) / 2
    return float(u / (fail.size * succ.size)), int(fail.size * succ.size)


def cohen_d(fail_vals: Sequence[float], succ_vals: Sequence[float]) -> float:
    """Standardized mean difference of failures against successes."""

    fail = np.asarray(fail_vals, float)
    succ = np.asarray(succ_vals, float)
    if fail.size < 2 or succ.size < 2:
        return float("nan")
    pooled = np.sqrt(
        ((fail.size - 1) * fail.var(ddof=1) + (succ.size - 1) * succ.var(ddof=1)) / (fail.size + succ.size - 2)
    )
    return float((fail.mean() - succ.mean()) / (pooled + 1e-12))


def tpr_at_fpr(fail_vals: Sequence[float], succ_vals: Sequence[float], fpr: float) -> float:
    """Fraction of failures above the success score's ``(1-fpr)`` quantile."""

    fail = np.asarray(fail_vals, float)
    succ = np.asarray(succ_vals, float)
    if fail.size == 0 or succ.size == 0:
        return float("nan")
    tau = float(np.quantile(succ, 1.0 - float(fpr)))
    return float(np.mean(fail > tau))


def early_window_summary(
    episodes: Sequence[dict],
    *,
    early_ks: Sequence[int] = (1, 2, 3, 5, 8, 10, 15),
    min_survivors: int = 20,
) -> list[dict]:
    """Does the score over the FIRST k chunks predict eventual failure?

    Conditioned on survival to at least k chunks, otherwise the comparison is
    between episodes that were abandoned early and ones that ran long, which
    measures episode length rather than the score. This is the primary metric
    when the failure count is small: it draws on every episode, not only the
    failures, so it has far more power than a TPR estimated on ~40 samples.

    Each ``episode`` is ``{"success": bool, "scores": sequence}``. Rows with too
    few survivors are omitted rather than reported with misleading noise.
    """

    rows: list[dict] = []
    for k in early_ks:
        survivors = [e for e in episodes if len(e["scores"]) >= k]
        if len(survivors) < min_survivors:
            continue
        succ = [float(np.mean(e["scores"][:k])) for e in survivors if e["success"]]
        fail = [float(np.mean(e["scores"][:k])) for e in survivors if not e["success"]]
        score, pairs = auroc(fail, succ)
        rows.append(
            {
                "k": int(k),
                "n_surv": len(survivors),
                "sr_surv": len(succ) / len(survivors) if survivors else float("nan"),
                "succ_mean": float(np.mean(succ)) if succ else float("nan"),
                "fail_mean": float(np.mean(fail)) if fail else float("nan"),
                "auroc": score,
                "n_pairs": pairs,
                "cohen_d": cohen_d(fail, succ),
            }
        )
    return rows


def cusum_peak(stream: Iterable[float], mean0: float, slack: float) -> float:
    """Return the peak of the one-sided upper CUSUM for one episode."""

    state = peak = 0.0
    for value in np.asarray(list(stream), dtype=np.float64):
        state = max(0.0, state + (float(value) - mean0 - slack))
        peak = max(peak, state)
    return float(peak)


def cusum_alarm(
    stream: Iterable[float],
    threshold: float,
    mean0: float,
    slack: float,
) -> tuple[int | None, int | None]:
    """Return the first alarm index and the latest zero-state onset index."""

    state = 0.0
    last_zero = 0
    for index, value in enumerate(np.asarray(list(stream), dtype=np.float64)):
        state = max(0.0, state + (float(value) - mean0 - slack))
        if state == 0.0:
            last_zero = index
        if state > threshold:
            return index, last_zero
    return None, None


def conformal_height(peaks: Iterable[float], fpr: float = 0.1) -> float:
    """Compute the finite-sample split-conformal CUSUM height."""

    values = np.sort(np.asarray(list(peaks), dtype=np.float64))
    if values.size == 0:
        return float("nan")
    if not 0.0 < fpr < 1.0:
        raise ValueError(f"fpr must be in (0, 1), got {fpr}")
    rank = int(np.ceil((values.size + 1) * (1.0 - fpr)))
    return float(values[min(rank, values.size) - 1])


def _finite_streams(streams: Iterable[Iterable[float]]) -> list[np.ndarray]:
    """Coerce an iterable of score streams to finite-only float64 arrays."""

    arrays = [np.asarray(list(stream), dtype=np.float64) for stream in streams]
    return [array[np.isfinite(array)] for array in arrays if np.isfinite(array).any()]


def calibrate(
    streams: Iterable[Iterable[float]],
    *,
    calib_streams: Iterable[Iterable[float]] | None = None,
    fpr: float = 0.1,
    k_sigma: float = 0.25,
    conformal: bool | None = None,
) -> dict[str, float | str | bool | int]:
    """Fit ``mu0``, ``sigma``, ``k`` and ``h`` on successful episodes.

    The single calibration implementation behind :meth:`AccelMonitor.from_success_streams`
    and :func:`evaluate_streams`, so the online monitor and the offline report
    cannot drift apart.

    ``calib_streams`` is the held-out protocol and the one to use in practice:
    the successes being scored stay disjoint from the successes that set the
    threshold, which is what makes ``succ_fire_rate`` a measurement instead of a
    restatement of ``fpr``.  ``calib_streams=None`` keeps upstream's legacy
    in-sample path, where the successes calibrate themselves and the threshold
    is optimistic -- ``conformal`` defaults to ``True`` with a held-out set
    (small ``M`` needs the order statistic) and ``False`` in-sample, matching
    upstream so the historical tables stay comparable.
    """

    held_out = calib_streams is not None
    if conformal is None:
        conformal = held_out
    calib = _finite_streams(calib_streams) if held_out else _finite_streams(streams)
    if not calib:
        raise ValueError("at least one successful stream with a finite score is required")
    if not 0.0 < fpr < 1.0:
        raise ValueError(f"fpr must be in (0, 1), got {fpr}")
    pooled = np.concatenate(calib)
    mu0 = float(pooled.mean())
    sigma = float(pooled.std())
    k = float(k_sigma * sigma)
    peaks = np.array([cusum_peak(array, mu0, k) for array in calib])
    # The order statistic is the finite-sample guarantee; np.quantile is the
    # large-M in-sample shortcut upstream keeps for comparability. They differ
    # most at exactly the small M a held-out split implies.
    h = conformal_height(peaks, fpr) if conformal else float(np.quantile(peaks, 1.0 - fpr))
    return {
        "mu0": mu0,
        "sigma": sigma,
        "k": k,
        "h": h,
        "k_sigma": float(k_sigma),
        "fpr": float(fpr),
        "n_calib": len(calib),
        "calibration": "held-out" if held_out else "in-sample",
        "conformal": bool(conformal),
    }


def score_frozen(
    succ_streams: Iterable[Iterable[float]],
    fail_streams: Iterable[Iterable[float]],
    *,
    mu0: float,
    k: float,
    h: float,
) -> dict[str, float | int]:
    """Realized specificity, sensitivity and lead time for FROZEN parameters.

    The scoring half of :func:`evaluate_streams` with ``mu0``/``k``/``h``
    supplied instead of re-derived. That distinction is the whole point of a
    stored calibration: the thresholds were fitted somewhere else -- another
    run, or a held-out split of this one -- and applying them unchanged is what
    makes ``succ_fire_rate`` a *measurement* of the false-alarm rate rather
    than a restatement of the rate that was asked for. Re-fitting inside the
    evaluation would quietly turn an out-of-sample test into an in-sample one.

    ``succ_fire_rate`` is the realized false-positive rate: the fraction of
    SCORED successes that alarmed. ``median_lead`` is in chunks, counted back
    from the episode's last chunk, so it is what a caller could still have
    saved. Both are ``nan`` when their population is empty -- an empty group is
    a data problem, and the counts are returned so a caller can tell that from
    a genuine zero.
    """

    succ = _finite_streams(succ_streams)
    fail = _finite_streams(fail_streams)

    fired_succ = sum(cusum_alarm(stream, h, mu0, k)[0] is not None for stream in succ)
    leads: list[float] = []
    alarm_frac: list[float] = []
    onset_frac: list[float] = []
    fired_fail = 0
    for stream in fail:
        alarm_index, onset_index = cusum_alarm(stream, h, mu0, k)
        if alarm_index is None:
            continue
        fired_fail += 1
        length = len(stream)
        leads.append(length - 1 - alarm_index)
        alarm_frac.append(alarm_index / length)
        if onset_index is not None:
            onset_frac.append(onset_index / length)

    return {
        "n_succ": len(succ),
        "n_fail": len(fail),
        "n_succ_fire": int(fired_succ),
        "n_fail_fire": int(fired_fail),
        "succ_fire_rate": fired_succ / len(succ) if succ else float("nan"),
        "tpr": fired_fail / len(fail) if fail else float("nan"),
        "median_lead": float(np.median(leads)) if leads else float("nan"),
        "median_alarm_frac": float(np.median(alarm_frac)) if alarm_frac else float("nan"),
        "median_onset_frac": float(np.median(onset_frac)) if onset_frac else float("nan"),
    }


def evaluate_streams(
    succ_streams: Iterable[Iterable[float]],
    fail_streams: Iterable[Iterable[float]],
    *,
    calib_streams: Iterable[Iterable[float]] | None = None,
    fpr: float = 0.1,
    k_sigma: float = 0.25,
    conformal: bool | None = None,
) -> dict[str, float | str | bool | int]:
    """Calibrate on successes, then score failures online; return the metrics.

    Field names match ``fm-geometry``'s ``cusum_online`` so the two are
    comparable side by side.  ``tpr`` is the fraction of failure episodes that
    alarm, ``median_lead`` is how many chunks before the episode's last chunk
    the alarm fired (i.e. how much of the episode a caller could still save),
    and ``succ_fire_rate`` is the specificity: an out-of-sample measurement when
    ``calib_streams`` is given, and merely the target ``fpr`` when it is not.
    """

    succ = _finite_streams(succ_streams)
    fail = _finite_streams(fail_streams)
    held_out = calib_streams is not None
    if conformal is None:
        conformal = held_out
    calib = _finite_streams(calib_streams) if held_out else succ
    if not calib:
        succ = fail = []

    if not succ or not fail:
        return {
            "fpr": float(fpr),
            "tpr": float("nan"),
            "n_fail": len(fail),
            "n_succ": len(succ),
            "n_fail_fire": 0,
            "median_lead": float("nan"),
            "median_alarm_frac": float("nan"),
            "median_onset_frac": float("nan"),
            "succ_fire_rate": float("nan"),
            "mu0": float("nan"),
            "sigma": float("nan"),
            "k": float("nan"),
            "h": float("nan"),
            "n_calib": len(calib),
            "calibration": "held-out" if held_out else "in-sample",
            "conformal": bool(conformal),
        }

    fitted = calibrate(succ, calib_streams=calib if held_out else None, fpr=fpr, k_sigma=k_sigma, conformal=conformal)
    mu0, k, h = float(fitted["mu0"]), float(fitted["k"]), float(fitted["h"])

    # Measured on the SCORED successes -- out-of-sample whenever calib is held out.
    scored = score_frozen(succ, fail, mu0=mu0, k=k, h=h)

    return {
        "fpr": float(fpr),
        "tpr": scored["tpr"],
        "n_fail": scored["n_fail"],
        "n_succ": scored["n_succ"],
        "n_fail_fire": scored["n_fail_fire"],
        "median_lead": scored["median_lead"],
        "median_alarm_frac": scored["median_alarm_frac"],
        "median_onset_frac": scored["median_onset_frac"],
        "succ_fire_rate": scored["succ_fire_rate"],
        "mu0": mu0,
        "sigma": float(fitted["sigma"]),
        "k": k,
        "h": h,
        "n_calib": int(fitted["n_calib"]),
        "calibration": str(fitted["calibration"]),
        "conformal": bool(fitted["conformal"]),
    }


@dataclasses.dataclass
class AccelMonitor:
    """Causal state machine that turns Accel scores into alarms.

    ``mean0`` and ``std0`` should be estimated on held-out successful
    episodes. ``threshold`` can be obtained with :func:`conformal_height`
    over their episode CUSUM peaks. ``update`` uses only the current score and
    prior state, so it is safe to call during receding-horizon inference.

    The alarm is the *endpoint* here, exactly as upstream: nothing in this class
    decides what a caller does about it.  ``index``/``onset``/``alarm_index``
    exist so an online caller can report where in the episode the drift started
    and how much episode was left when it fired.
    """

    mean0: float
    std0: float
    threshold: float
    k_sigma: float = 0.25
    state: float = 0.0
    index: int = 0
    onset: int = 0
    alarm_index: int | None = None

    @property
    def slack(self) -> float:
        return float(self.k_sigma * self.std0)

    def reset(self) -> None:
        """Start a new episode: clear the running state, keep the calibration."""

        self.state = 0.0
        self.index = 0
        self.onset = 0
        self.alarm_index = None

    def update(self, score: float) -> dict[str, float | bool | int]:
        """Consume one Accel score and return ``score/cusum/alarm/onset/index``."""

        value = float(score)
        self.state = max(0.0, self.state + value - self.mean0 - self.slack)
        if self.state == 0.0:
            # Same convention as cusum_alarm: the last chunk that reset the sum
            # before the alarm is the maximum-likelihood change-point.
            self.onset = self.index
        alarm = bool(self.state > self.threshold)
        if alarm and self.alarm_index is None:
            self.alarm_index = self.index
        result: dict[str, float | bool | int] = {
            "score": value,
            "cusum": float(self.state),
            "alarm": alarm,
            "onset": int(self.onset),
            "index": int(self.index),
        }
        self.index += 1
        return result

    @classmethod
    def from_success_streams(
        cls,
        streams: Iterable[Iterable[float]],
        *,
        calib_streams: Iterable[Iterable[float]] | None = None,
        fpr: float = 0.1,
        k_sigma: float = 0.25,
        conformal: bool | None = None,
    ) -> AccelMonitor:
        """Calibrate from successful per-chunk streams.

        Pass ``calib_streams`` (successes that are *not* among the streams being
        monitored) for the held-out split-conformal height; without it the
        successes calibrate themselves and the height is the in-sample
        ``np.quantile``, matching upstream's legacy path.
        """

        fitted = calibrate(streams, calib_streams=calib_streams, fpr=fpr, k_sigma=k_sigma, conformal=conformal)
        return cls(
            mean0=float(fitted["mu0"]),
            std0=float(fitted["sigma"]),
            threshold=float(fitted["h"]),
            k_sigma=float(k_sigma),
        )

    @classmethod
    def from_calibration(cls, calibration: AccelCalibration) -> AccelMonitor:
        """Build a monitor from a fitted :class:`~openpi.uncertainty.calibration.AccelCalibration`."""

        return cls(
            mean0=float(calibration.mu0),
            std0=float(calibration.sigma),
            threshold=float(calibration.h),
            k_sigma=float(calibration.k_sigma),
        )


__all__ = [
    "AccelMonitor",
    "auroc",
    "calibrate",
    "cohen_d",
    "conformal_height",
    "cusum_alarm",
    "cusum_peak",
    "early_window_summary",
    "evaluate_streams",
    "score_frozen",
    "tpr_at_fpr",
]
