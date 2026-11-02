"""Experiment 518: what the encoder is scored on, as functions of plain arrays.

All inputs are arrays of the estimated step `level` (windows, 672) in spreads, so the same code
scores H, a perfect predictor, and a zero predictor in the self-test.

- A **candidate onset** is a half-hour where the mean estimated level over the next `RADIUS`
  half-hours exceeds the mean over the previous `RADIUS` (including this one's predecessor) by at
  least `THRESHOLD` spreads, and the difference is the largest within `RADIUS` half-hours either
  side. Averaging over four half-hours keeps a ramp of up to four half-hours from hiding a step.
  A step injected at half of its size for its first half-hours (a staircase) of under 0.5 spreads
  is not a candidate even for a perfect predictor, so every score is printed beside a ceiling: the
  same score of a predictor that outputs the injected step exactly.
- An injected event is **found within h hours** if a candidate lies within h hours of its onset. A
  step is **missed** if no candidate lies within 24 hours, the largest onset error on the noise
  curve of the plan.
- The **size error** is the relative difference between the median estimated level and the median
  injected step over the event's half-hours inside the window.
- The **repaired lag error** is the mean absolute difference between the estimated level and the
  injected step at the three lag positions (newest, one day, and one week back), against the mean
  absolute injected step. Rows with no shift touching a lag measure what the repair adds.
- The **planted-step precision** at rate `PLANT_RATE`: a window is flagged if it has a candidate.
  With recall `r` on windows that carry a planted step (a candidate within 2 hours of its onset)
  and false-flag rate `f` on real windows, precision is `0.02 r / (0.02 r + 0.98 f)`. The real
  windows hold events that NGED did not log, so `f` is an upper bound on the false-step rate.
"""

from dataclasses import dataclass
from typing import Final

import e00507_generator as gen  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view

THRESHOLD: Final[float] = 0.5
RADIUS: Final[int] = 4
FOUND_HOURS: Final[tuple[float, ...]] = (2.0, 6.0, 24.0)
LAG_POSITIONS: Final[tuple[int, int, int]] = (
    gen.HISTORY - 1,
    gen.HISTORY - 1 - 48,
    gen.HISTORY - 1 - 336,
)
PLANT_RATE: Final[float] = 0.02
BOOTSTRAPS: Final[int] = 400


def candidate_mask(*, level: np.ndarray, threshold: float = THRESHOLD) -> np.ndarray:
    """Return a boolean (windows, 672) array: True where a candidate onset begins at that index."""
    n, size = level.shape
    cum = np.concatenate([np.zeros((n, 1)), np.cumsum(level, axis=1, dtype=np.float64)], axis=1)
    diff = np.zeros((n, size))
    t = np.arange(RADIUS - 1, size - RADIUS)  # boundary between t and t + 1
    right = (cum[:, t + RADIUS + 1] - cum[:, t + 1]) / RADIUS
    left = (cum[:, t + 1] - cum[:, t + 1 - RADIUS]) / RADIUS
    diff[:, t + 1] = np.abs(right - left)
    padded = np.pad(diff, ((0, 0), (RADIUS, RADIUS)), constant_values=0.0)
    peak = sliding_window_view(padded, 2 * RADIUS + 1, axis=1).max(axis=-1)
    return (diff >= threshold) & (diff >= peak)


def true_events(
    *, ev: gen.Events, p: np.ndarray, d: np.ndarray, step_true: np.ndarray
) -> pl.DataFrame:
    """Return one row per injected event whose onset lies inside a window's history.

    Columns: `window` (index into the windows), `onset` (index inside the history), `family`,
    `size` (median injected step over the event's half-hours inside the history, in spreads).
    """
    newest = gen.newest_of(day=d)
    first = newest - (gen.HISTORY - 1)
    rows = []
    for slot in range(gen.SLOTS):
        fam = ev.family[p, d, slot]
        start = ev.start[p, d, slot]
        end = ev.end[p, d, slot]
        use = np.flatnonzero((fam != gen.NONE) & (start >= first) & (start <= newest))
        for w in use:
            lo = int(start[w] - first[w])
            hi = int(min(end[w], newest[w] + 1) - first[w])
            rows.append(
                {
                    "window": int(w),
                    "end": max(hi, lo + 1),
                    "onset": lo,
                    "family": int(fam[w]),
                    "size": float(np.median(step_true[w, lo:hi])),
                }
            )
    return pl.DataFrame(
        rows,
        schema={
            "window": pl.Int64,
            "end": pl.Int64,
            "onset": pl.Int64,
            "family": pl.Int64,
            "size": pl.Float64,
        },
    )


def score_events(
    *, level: np.ndarray, events: pl.DataFrame, step_true: np.ndarray, threshold: float = THRESHOLD
) -> pl.DataFrame:
    """Return the events with their onset error (hours; inf if none) and their size error.

    Args:
        level: The estimated step (windows, 672) in spreads.
        events: The output of `true_events`.
        step_true: The injected step (windows, 672) in spreads.
        threshold: The candidate threshold in spreads.

    Returns:
        `events` plus `onset_error_hours`, `size_error` (relative), and the estimated `size_hat`.
    """
    mask = candidate_mask(level=level, threshold=threshold)
    err, size_hat = [], []
    for row in events.iter_rows(named=True):
        cand = np.flatnonzero(mask[row["window"]])
        err.append(float(np.abs(cand - row["onset"]).min()) / 2.0 if len(cand) else np.inf)
        w, o = row["window"], row["onset"]
        covered = np.flatnonzero(np.abs(step_true[w, o : row["end"]]) > 1e-6) + o
        size_hat.append(float(np.median(level[w, covered])) if len(covered) else 0.0)
    out = events.with_columns(
        pl.Series("onset_error_hours", err, dtype=pl.Float64),
        pl.Series("size_hat", size_hat, dtype=pl.Float64),
    )
    return out.with_columns(
        (
            (pl.col("size_hat") - pl.col("size")).abs()
            / pl.col("size").abs().clip(lower_bound=1e-9)
        ).alias("size_error")
    )


def lag_errors(*, level: np.ndarray, step_true: np.ndarray, valid: np.ndarray) -> pl.DataFrame:
    """Return per window the repaired and unrepaired lag errors (spreads) and `touched`.

    A window is `touched` if the injected step is non-zero at any of the three lag positions.
    """
    pos = list(LAG_POSITIONS)
    s, h, ok = step_true[:, pos], level[:, pos], valid[:, pos]
    count = ok.sum(axis=1)
    return pl.DataFrame(
        {
            "repaired": np.where(
                count > 0, (np.abs(h - s) * ok).sum(axis=1) / np.maximum(count, 1), 0.0
            ),
            "unrepaired": np.where(
                count > 0, (np.abs(s) * ok).sum(axis=1) / np.maximum(count, 1), 0.0
            ),
            "touched": (np.abs(s) > 1e-6).any(axis=1) & (count > 0),
            "usable": count > 0,
        }
    )


def flagged(*, level: np.ndarray, threshold: float = THRESHOLD) -> np.ndarray:
    """Return True for each window with at least one candidate onset."""
    return candidate_mask(level=level, threshold=threshold).any(axis=1)


def rank_auc(*, score: np.ndarray, label: np.ndarray) -> float:
    """Return the area under the ROC curve, NaN if one class is empty."""
    pos, neg = score[label > 0.5], score[label <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="stable")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def precision_at_rate(*, recall: float, false_rate: float, rate: float = PLANT_RATE) -> float:
    """Return the precision of the flag when a share `rate` of windows carry a planted step."""
    hit = rate * recall
    return hit / max(hit + (1 - rate) * false_rate, 1e-12)


@dataclass(frozen=True)
class Interval:
    """A mean and its 95% cluster-bootstrap interval."""

    mean: float
    low: float
    high: float

    def __str__(self) -> str:
        """Return `mean (low to high)`."""
        return f"{self.mean:.3f} ({self.low:.3f} to {self.high:.3f})"


def cluster_mean(*, values: np.ndarray, cluster: np.ndarray, seed: int = 0) -> Interval:
    """Return the mean of `values` and a 95% interval from resampling whole clusters.

    Args:
        values: One number per row (booleans count as 0 and 1).
        cluster: The cluster of each row (here, the window shared by the three worlds).
        seed: The resampling seed.

    Returns:
        The mean and the interval; NaN everywhere if there are no rows.
    """
    if len(values) == 0:
        return Interval(float("nan"), float("nan"), float("nan"))
    ids, inverse = np.unique(cluster, return_inverse=True)
    total = np.bincount(inverse, weights=values.astype(np.float64), minlength=len(ids))
    count = np.bincount(inverse, minlength=len(ids)).astype(np.float64)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(ids), size=(BOOTSTRAPS, len(ids)))
    boot = total[pick].sum(axis=1) / count[pick].sum(axis=1)
    return Interval(
        float(values.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))
    )
