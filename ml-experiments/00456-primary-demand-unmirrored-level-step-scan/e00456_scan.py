"""Experiment 456, step 2: the step scan, its mirror classes, and its validation (no fit).

Pure functions on daily residual levels (points of effective capacity), plus `main`, which reads
`residuals_b.parquet` (step 1) and writes the tables of the plan. See the experiment's `result.md`
for the definitions; every parameter is pre-registered there.
"""

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl

GRID_START: Final[datetime] = datetime(2019, 9, 1, tzinfo=UTC)
LABEL_TABLE: Final[Path] = (
    Path(__file__).resolve().parents[1]
    / "00395-xgboost-level-state-winter-spread-anchor-stored-forecasts"
    / "sw_block_detector_label_table_corrected.csv"
)
MIN_VALID_HALF_HOURS: Final[int] = 24
FLANK_DAYS: Final[int] = 7


@dataclass(frozen=True)
class Params:
    """The scan's thresholds (baseline: the plan's pre-registered values)."""

    spread_multiple: float = 3.0
    floor_points: float = 2.0
    min_segment_days: int = 7
    penalty_multiplier: float = 3.0
    mirror_start_days: int = 1
    mirror_ratio: float = 2.0


@dataclass(frozen=True)
class Step:
    """One detected step at a Primary (day indices are days since 2019-09-01)."""

    site: str
    start_day: int
    end_day: int
    points: float
    mw: float


def daily_levels(*, grid: np.ndarray, residual: np.ndarray, n_days: int) -> np.ndarray:
    """Return the daily median residual in points, NaN on days with under 24 valid half-hours."""
    day = grid // 48
    out = np.full(n_days, np.nan)
    order = np.argsort(day, kind="stable")
    day_sorted, res_sorted = day[order], residual[order]
    bounds = np.flatnonzero(np.diff(day_sorted)) + 1
    for d, chunk in zip(np.split(day_sorted, bounds), np.split(res_sorted, bounds), strict=True):
        if len(chunk) >= MIN_VALID_HALF_HOURS:
            out[d[0]] = 100.0 * np.median(chunk)
    return out


def spread_of(*, levels: np.ndarray) -> float:
    """Return the unscaled median absolute deviation of the valid daily levels."""
    valid = levels[np.isfinite(levels)]
    return float(np.median(np.abs(valid - np.median(valid))))


def noise_sigma(*, levels: np.ndarray) -> float:
    """Return the robust daily noise scale from lag-7 differences (median-absolute, / sqrt 2)."""
    diff = levels[7:] - levels[:-7]
    diff = diff[np.isfinite(diff)]
    return float(1.4826 * np.median(np.abs(diff - np.median(diff))) / np.sqrt(2.0))


def partition(*, y: np.ndarray, penalty: float, min_len: int) -> list[int]:
    """Return segment start indices of the optimal squared-error partition of `y`.

    Optimal partitioning by dynamic programming over all start points (O(n^2)): the cost of a
    segment is its sum of squares about its mean, each extra segment adds `penalty`, and every
    segment holds at least `min_len` points.
    """
    n = len(y)
    csum = np.concatenate([[0.0], np.cumsum(y)])
    csq = np.concatenate([[0.0], np.cumsum(y * y)])
    best = np.full(n + 1, np.inf)
    best[0] = -penalty
    back = np.zeros(n + 1, dtype=int)
    for j in range(min_len, n + 1):
        starts = np.arange(0, j - min_len + 1)
        length = j - starts
        seg = csq[j] - csq[starts] - (csum[j] - csum[starts]) ** 2 / length
        total = best[starts] + seg + penalty
        k = int(np.argmin(total))
        best[j], back[j] = total[k], starts[k]
    cuts: list[int] = []
    j = n
    while j > 0:
        cuts.append(int(back[j]))
        j = int(back[j])
    return sorted(cuts)


def detect(
    *,
    site: str,
    levels: np.ndarray,
    capacity_mw: float,
    params: Params,
    spread: float | None = None,
    sigma: float | None = None,
) -> tuple[list[Step], float]:
    """Return the steps that pass the thresholds, and the Primary's spread (points).

    `spread` and `sigma` override the values measured on `levels`; validations pass the real
    series' values so that an injected or resampled series is judged by the real thresholds.
    """
    valid_days = np.flatnonzero(np.isfinite(levels))
    y = levels[valid_days]
    spread = spread_of(levels=levels) if spread is None else spread
    sigma = noise_sigma(levels=levels) if sigma is None else sigma
    penalty = params.penalty_multiplier * np.log(len(y)) * sigma**2
    starts = partition(y=y, penalty=penalty, min_len=params.min_segment_days)
    ends = [*starts[1:], len(y)]
    means = [float(np.mean(y[a:b])) for a, b in zip(starts, ends, strict=True)]
    steps: list[Step] = []
    for k in range(1, len(starts)):
        delta = means[k] - means[k - 1]
        if abs(delta) >= max(params.spread_multiple * spread, params.floor_points):
            steps.append(
                Step(
                    site=site,
                    start_day=int(valid_days[starts[k]]),
                    end_day=int(valid_days[ends[k] - 1]),
                    points=delta,
                    mw=delta * capacity_mw / 100.0,
                )
            )
    return steps, spread


def flank_change_mw(*, levels: np.ndarray, day: int, capacity_mw: float) -> float:
    """Return a Primary's raw change across `day`: median of the next 7 days minus the last 7."""
    after = levels[day : day + FLANK_DAYS]
    before = levels[max(0, day - FLANK_DAYS) : day]
    after, before = after[np.isfinite(after)], before[np.isfinite(before)]
    if len(after) < 4 or len(before) < 4:
        return np.nan
    return float((np.median(after) - np.median(before)) * capacity_mw / 100.0)


def neighbour_sets() -> dict[str, set[str]]:
    """Return each Primary's neighbours: the labels named in the same row of the corrected table."""
    graph: dict[str, set[str]] = {}
    for pair in pl.read_csv(LABEL_TABLE)["pair"].to_list():
        names = set(re.findall(r"D\d+", pair))
        for a in names:
            graph.setdefault(a, set()).update(names - {a})
    return graph


def classify(
    *,
    step: Step,
    levels_of: dict[str, np.ndarray],
    capacity_of: dict[str, float],
    neighbours: set[str],
    params: Params,
    start_days: int | None = None,
) -> tuple[str, str, float]:
    """Return (class, best mirror site, its opposite-sign change in MW) for one step.

    `mirrored` if some neighbour's raw flank change within `start_days` of the step's start has the
    opposite sign and a magnitude ratio between 1/ratio and ratio in MW; `unassessable` if no
    neighbour has data around the start; otherwise `unmirrored`.
    """
    window = params.mirror_start_days if start_days is None else start_days
    assessed = False
    best_site, best_mw = "", 0.0
    for other in sorted(neighbours):
        if other == step.site or other not in levels_of:
            continue
        for day in range(step.start_day - window, step.start_day + window + 1):
            change = flank_change_mw(
                levels=levels_of[other], day=day, capacity_mw=capacity_of[other]
            )
            if not np.isfinite(change):
                continue
            assessed = True
            if change * step.mw < 0:
                ratio = abs(change) / abs(step.mw)
                if 1.0 / params.mirror_ratio <= ratio <= params.mirror_ratio:
                    return "mirrored", other, change
                if abs(change) > abs(best_mw):
                    best_site, best_mw = other, change
    return ("unmirrored" if assessed else "unassessable"), best_site, best_mw


def mark_temporary(*, steps: list[Step], classes: dict[tuple[str, int], str]) -> None:
    """Relabel an unmirrored step and its reversal within 30 days (50% of its size) as temporary."""
    by_site: dict[str, list[Step]] = {}
    for s in steps:
        by_site.setdefault(s.site, []).append(s)
    for site_steps in by_site.values():
        site_steps.sort(key=lambda s: s.start_day)
        for a, b in zip(site_steps, site_steps[1:], strict=False):
            if (
                b.start_day - a.start_day < 30
                and a.points * b.points < 0
                and 0.5 <= abs(b.points) / abs(a.points) <= 2.0
                and classes[(a.site, a.start_day)] == "unmirrored"
                and classes[(b.site, b.start_day)] == "unmirrored"
            ):
                classes[(a.site, a.start_day)] = "temporary"
                classes[(b.site, b.start_day)] = "temporary"
