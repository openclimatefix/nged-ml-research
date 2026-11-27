"""Scoring of experiment 734: found shares, paired intervals, vetoes, calibration, and false calls.

`e00734_report.py` prints every number the page quotes and reads only the files under `OUT` and
the stored rows of experiments 680 and 670.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl

N_BOOT: Final[int] = 10_000
INTERVAL: Final[tuple[float, float]] = (1.25, 98.75)
"""Percentiles of a 97.5% interval (Bonferroni over the two counted contrasts)."""
C1_BAR: Final[float] = 0.10
VETO_RATIO: Final[float] = 1.10
COUNTED_CELL: Final[str] = "long, shallow"
KINDS: Final[tuple[str, ...]] = ("edge", "solo", "permanent_transfer", "permanent_shift")
MIN_UNITS: Final[float] = 2.0
DEEP_UNITS: Final[float] = 8.0
GRID_START: Final[datetime] = datetime(2022, 1, 1, tzinfo=UTC)
QUANTILES: Final[tuple[float, ...]] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
RECUR_WORLDS: Final[int] = 10
"""A candidate at one Primary and day that recurs in this many worlds is a real-record event."""
DAY: Final[int] = 48


def length_of(*, days: float | None) -> str:
    """Length band of experiment 665: long is over 7 days or no end, short is 1 day or less."""
    if days is None or days < 0 or days > 7:
        return "long"
    return "medium" if days > 1 else "short"


def add_cells(*, rows: pl.DataFrame) -> pl.DataFrame:
    """Keep injected positive events of 2 noise units or more and add experiment 665's cell."""
    d = rows.filter(pl.col("kind").is_in(KINDS) & (pl.col("units") >= MIN_UNITS))
    cells = [
        f"{length_of(days=r['days'])}, {'deep' if r['units'] >= DEEP_UNITS else 'shallow'}"
        for r in d.select("days", "units").iter_rows(named=True)
    ]
    month = [
        (GRID_START + timedelta(minutes=30 * int(s))).strftime("%Y-%m")
        for s in d["start"].to_list()
    ]
    return d.with_columns(
        cell=pl.Series(cells, dtype=pl.String), month=pl.Series(month, dtype=pl.String)
    )


def found_share(*, rows: pl.DataFrame, cell: str | None = None, col: str = "found") -> float:
    """MW-weighted share of rows with `col` true, in a cell (all rows when None)."""
    d = rows if cell is None else rows.filter(pl.col("cell") == cell)
    w = d["amp_mw"].abs()
    if d.height == 0 or float(w.sum()) == 0.0:
        return float("nan")
    return float((d[col].cast(pl.Float64) * w).sum()) / float(w.sum())


def _unit_table(*, rows: pl.DataFrame, by: str, units: list, col: str) -> np.ndarray:
    """Per unit (world or month): found megawatts and total megawatts."""
    out = np.zeros((len(units), 2))
    for i, u in enumerate(units):
        g = rows.filter(pl.col(by) == u)
        w = g["amp_mw"].abs()
        out[i] = [float((g[col].cast(pl.Float64) * w).sum()), float(w.sum())]
    return out


def _paired(
    *, a: np.ndarray, b: np.ndarray, rng: np.random.Generator
) -> tuple[float, float, float]:
    """Point difference of pooled shares and its percentile interval over unit resamples."""
    n = len(a)
    draws = rng.choice(n, size=(N_BOOT, n))
    sa, sb = a[draws].sum(axis=1), b[draws].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        diff = sa[:, 0] / sa[:, 1] - sb[:, 0] / sb[:, 1]
    point = a[:, 0].sum() / a[:, 1].sum() - b[:, 0].sum() / b[:, 1].sum()
    lo, hi = np.nanpercentile(diff, INTERVAL)
    return float(point), float(lo), float(hi)


@dataclass(frozen=True)
class Contrast:
    """A paired difference of found shares with both intervals."""

    point: float
    world_lo: float
    world_hi: float
    month_lo: float
    month_hi: float

    @property
    def counts(self) -> bool:
        """True when both intervals exclude zero."""
        return (self.world_lo > 0 or self.world_hi < 0) and (self.month_lo > 0 or self.month_hi < 0)


def contrast(
    *, a: pl.DataFrame, b: pl.DataFrame, cell: str, rng: np.random.Generator, col: str = "found"
) -> Contrast:
    """Method `a` minus method `b`, in a cell, pooled over worlds.

    Interval A resamples whole worlds; interval B assigns each truth row to the calendar month of
    its start, resamples months, and pools all worlds' rows of the drawn months. Both use
    `N_BOOT` draws and the percentiles of `INTERVAL`.

    Args:
        a: Rows of method `a` with `world`, `cell`, `month`, `amp_mw`, and `col`.
        b: The same for method `b`, on the same truth rows.
        cell: The experiment 665 cell.
        rng: The random generator.
        col: The flag column to share (`found` or `right_class`).

    Returns:
        The point estimate and both intervals.
    """
    ca, cb = a.filter(pl.col("cell") == cell), b.filter(pl.col("cell") == cell)
    worlds = sorted(set(ca["world"].to_list()) | set(cb["world"].to_list()))
    point, wlo, whi = _paired(
        a=_unit_table(rows=ca, by="world", units=worlds, col=col),
        b=_unit_table(rows=cb, by="world", units=worlds, col=col),
        rng=rng,
    )
    months = sorted(set(ca["month"].to_list()) | set(cb["month"].to_list()))
    _, mlo, mhi = _paired(
        a=_unit_table(rows=ca, by="month", units=months, col=col),
        b=_unit_table(rows=cb, by="month", units=months, col=col),
        rng=rng,
    )
    return Contrast(point, wlo, whi, mlo, mhi)


def leave_one_out(*, a: pl.DataFrame, b: pl.DataFrame, cell: str, by: str) -> pl.DataFrame:
    """The point estimate of `a - b` in a cell with each world (or each Primary) left out."""
    out = []
    for u in sorted(set(a[by].to_list())):
        pa, pb = a.filter(pl.col(by) != u), b.filter(pl.col(by) != u)
        out.append(
            {by: u, "point": found_share(rows=pa, cell=cell) - found_share(rows=pb, cell=cell)}
        )
    return pl.DataFrame(out)


def verdict(*, c1: Contrast, v1: bool, v2: bool, c1_upper_below_zero: bool) -> str:
    """Verdict word of the pre-registered decision rule."""
    if c1.counts and c1.point >= C1_BAR and v1 and v2:
        return "supported"
    if c1.counts and c1.point > 0 and v1 and v2:
        return "small gain"
    if c1.counts and c1.point > 0:
        return "partly supported"
    if c1_upper_below_zero:
        return "refuted"
    return "not supported"


def veto_ratios(
    *, h_rates: tuple[float, float], base_rates: tuple[float, float]
) -> tuple[tuple[float, float], bool]:
    """Ratios of both called rates over the benchmark's, and whether both are at most 1.10."""
    ratios = (h_rates[0] / base_rates[0], h_rates[1] / base_rates[1])
    return ratios, max(ratios) <= VETO_RATIO


def pinball(*, truth: np.ndarray, pred: np.ndarray, q: float) -> float:
    """Mean pinball loss of quantile `q`."""
    d = truth - pred
    return float(np.mean(np.maximum(q * d, (q - 1.0) * d)))


def calibration(*, truth: np.ndarray, quantiles: np.ndarray) -> dict[str, float]:
    """Pinball loss at each quantile and the coverage of q10 to q90 (the 80% interval).

    Args:
        truth: The injected change in megawatts at each block, `(n,)`.
        quantiles: The predicted switched megawatts minus the uninjected median, `(n, 7)`.

    Returns:
        A dict with `pinball_<q>` for each quantile and `coverage_80`.
    """
    out = {
        f"pinball_{q:.2f}": pinball(truth=truth, pred=quantiles[:, j], q=q)
        for j, q in enumerate(QUANTILES)
    }
    out["coverage_80"] = float(np.mean((truth >= quantiles[:, 1]) & (truth <= quantiles[:, 5])))
    return out


def event_overlap_share(*, start: int, end: int, windows: np.ndarray) -> float:
    """Share of the span `[start, end)` that lies inside the boolean `windows` array."""
    if end <= start:
        return 0.0
    return float(windows[start:end].mean())


def drop_recurring(*, candidates: pl.DataFrame) -> pl.DataFrame:
    """Drop candidates that recur at one Primary and day in 10 or more worlds.

    Args:
        candidates: Rows with `world`, `primary`, and `start` (half-hours).

    Returns:
        The rows whose Primary and day occur in fewer than `RECUR_WORLDS` worlds.
    """
    keyed = candidates.with_columns(day=pl.col("start") // DAY)
    counts = keyed.group_by("primary", "day").agg(n=pl.col("world").n_unique())
    recurring = counts.filter(pl.col("n") >= RECUR_WORLDS)
    return keyed.join(recurring.select("primary", "day"), on=["primary", "day"], how="anti")


def report(*, out: Path) -> None:
    """Print every number the page quotes from the scored files under `out` (stage 5)."""
    from e00734_report import report as run_report

    run_report(out=out)


if __name__ == "__main__":
    report(out=Path(sys.argv[1]))
