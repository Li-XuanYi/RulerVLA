"""Does ``accel`` track the resample divergence? The published Table 1 question.

``accel`` claims to read posterior spread off the geometry of a denoising path.
The check is a rank correlation against ``D_resample``, the measured spread of
chunks the same observation actually produces under fresh noise.

Three numbers come out, and the difference between them is the point:

* ``rho_full`` -- the whole denoising path against the divergence. A single
  pre-specified statistic, so it is the one to lead with.
* ``rho_best`` -- the best depth, chosen by taking the argmax over depths *on the
  same data*. That is what the published ``0.792`` is, and it is an in-sample
  maximum: it can only be optimistic, and by an amount that grows with the number
  of depths tried.
* ``rho_best_held_out`` -- the depth chosen on one half of the episodes and
  scored on the other. This is the honest version of the same claim, and if the
  gap to ``rho_best`` is large then the peak was mostly selection.

Splitting by *episode* rather than by decision matters: decisions inside one
episode share an observation stream, so a decision-level split would put near
duplicates on both sides and flatter the held-out number.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses

import numpy as np

EPS = 1e-12


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Rank correlation of two 1-D arrays; nan below 3 points or on a constant.

    Ranks are averaged within ties, which matters here because ``D_resample`` is
    a maximum over pairs and can repeat exactly when two decisions draw the same
    candidates.
    """

    x = np.asarray(a, float)
    y = np.asarray(b, float)
    if x.size < 3 or x.size != y.size:
        return float("nan")
    rx = _average_ranks(x)
    ry = _average_ranks(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denominator = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denominator) if denominator > EPS else float("nan")


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = values.argsort(kind="mergesort")
    ranks = np.empty(values.size, float)
    ranks[order] = np.arange(1, values.size + 1, dtype=float)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    sums = np.zeros(counts.size)
    np.add.at(sums, inverse, ranks)
    return (sums / counts)[inverse]


@dataclasses.dataclass
class PrefixCurve:
    """The rank correlation at every denoise depth."""

    columns: list[int]
    rho: list[float]
    n_decisions: int

    @property
    def best(self) -> tuple[int, float]:
        """``(column, rho)`` at the argmax -- the in-sample peak."""

        pairs = [(column, value) for column, value in zip(self.columns, self.rho, strict=True) if np.isfinite(value)]
        if not pairs:
            return -1, float("nan")
        return max(pairs, key=lambda pair: pair[1])

    def at(self, column: int) -> float:
        """The rho at one named depth, or nan if it was not computed."""

        for candidate, value in zip(self.columns, self.rho, strict=True):
            if candidate == int(column):
                return float(value)
        return float("nan")


def prefix_rho_curve(
    prefix_tables: np.ndarray, divergence: np.ndarray, *, columns: Sequence[int] | None = None
) -> PrefixCurve:
    """Rank correlation of every denoise-depth prefix against the divergence.

    ``prefix_tables`` is ``(n, T-1)`` where column ``j`` is ``accel_p`` at depth
    ``p = j + 2``, so column ``-1`` is the full path -- the statistic behind
    ``rho_full``.
    """

    tables = np.asarray(prefix_tables, float)
    div = np.asarray(divergence, float)
    if tables.ndim != 2:
        raise ValueError(f"prefix tables must be (n, T-1); got {tables.shape}")
    if tables.shape[0] != div.size:
        raise ValueError(f"{tables.shape[0]} prefix rows for {div.size} divergence values")
    indices = list(range(tables.shape[1])) if columns is None else [int(c) % tables.shape[1] for c in columns]
    rho = [spearman(tables[:, index], div) for index in indices]
    return PrefixCurve(columns=indices, rho=rho, n_decisions=int(div.size))


def held_out_best(
    prefix_tables: np.ndarray,
    divergence: np.ndarray,
    groups: Sequence,
    *,
    seed: int = 0,
    columns: Sequence[int] | None = None,
) -> dict:
    """Pick the best depth on one half of the episodes, score it on the other.

    ``groups`` names the episode each decision came from. Both directions are run
    and reported, because a single split leaves the choice of half doing work
    that has nothing to do with the claim.
    """

    tables = np.asarray(prefix_tables, float)
    div = np.asarray(divergence, float)
    group_array = np.asarray(list(groups), dtype=object)
    unique = np.unique(group_array)
    if unique.size < 4:
        return {"error": f"need at least 4 episodes to split; got {unique.size}"}
    shuffled = np.random.default_rng(int(seed)).permutation(unique)
    halves = [shuffled[::2], shuffled[1::2]]

    out: dict = {"n_episodes": int(unique.size), "splits": []}
    for choose_half, score_half in ((0, 1), (1, 0)):
        choose_mask = np.isin(group_array, halves[choose_half])
        score_mask = np.isin(group_array, halves[score_half])
        selected = prefix_rho_curve(tables[choose_mask], div[choose_mask], columns=columns)
        column, chosen_rho = selected.best
        if column < 0:
            continue
        scored = spearman(tables[score_mask, column], div[score_mask])
        out["splits"].append(
            {
                "chosen_column": column,
                "rho_on_choice_half": chosen_rho,
                "rho_on_held_out_half": scored,
                "n_choose": int(choose_mask.sum()),
                "n_score": int(score_mask.sum()),
            }
        )
    values = [split["rho_on_held_out_half"] for split in out["splits"] if np.isfinite(split["rho_on_held_out_half"])]
    out["rho_best_held_out"] = float(np.mean(values)) if values else float("nan")
    out["rho_best_held_out_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return out


__all__ = ["PrefixCurve", "held_out_best", "prefix_rho_curve", "spearman"]
