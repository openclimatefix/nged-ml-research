# ruff: noqa: E501
"""Experiment 514: the scaled residual series of a fold, its partner series, and the labels.

A fold is one of experiment 507's lagless mixture tables (`crossfit` rows up to the month's refit,
then `further` rows). The residual of a Primary is the cleaned power over effective capacity minus
the table's `q50`, divided by the Primary's median band half-width over its `crossfit` rows. The
cross-fit blocks are centred on their own median (history only), and the junction between the last
block and the `further` rows is detrended causally at each cutoff by `junction_offset`.
"""

import functools
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00428_data as data  # ty: ignore[unresolved-import]
import e00510_common as c510  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00514-load-shift-probability-detector-and-calibration" / "work"
)
E507_TABLES: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00507-history-encoder-first-test-oracle-repair-and-total-variation"
    / "work"
    / "lagless_era5_mixture"
)
EVENTS: Final[Path] = c510.worlds402.cf.EVENTS
FOLDS: Final[tuple[str, ...]] = ("2025-10", "2026-01", "2026-03", "2026-06")
BLOCKS: Final[int] = 6
"""Experiment 404's cross-fit blocks: six contiguous equal blocks of the history."""
FUTURE_STEPS: Final[int] = 62 * 48
CUT_STEPS: Final[int] = 7
"""The detector sees half-hours at least 3.5 hours (7 steps) before the cutoff."""
WEEK: Final[int] = 336
EDGE_WINDOW: Final[int] = 6
"""Half-hours either side of a partner series' edge statistic."""
EVENT_PAD: Final[int] = 96
SPLIT_END: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)
"""Residual history for training the classifier ends here."""


def sites() -> list[str]:
    """Return the 16 Primaries in Primary-index order."""
    return data.sites()


def grid_length() -> int:
    """Return the number of half-hours on the dense grid."""
    return len(common.grid_times())


def index_of(*, t: datetime) -> int:
    """Return the grid index of a UTC timestamp."""
    return int(common.grid_index(times=t))


def time_of(*, index: int) -> datetime:
    """Return the UTC timestamp of a grid index."""
    return common.GRID_START + timedelta(minutes=30 * index)


@functools.cache
def series_info() -> pl.DataFrame:
    """Return the 20 series: label, type, units, and effective capacity in MW."""
    return common.series_table().select(
        "site", "substation_type", "units", "effective_capacity_mw", "time_series_id"
    )


def primary_capacity_mw() -> np.ndarray:
    """Return the Primaries' effective capacity in MW, in Primary-index order."""
    table = series_info().filter(pl.col("substation_type") == "Primary")
    by_site = dict(zip(table["site"].to_list(), table["effective_capacity_mw"].to_list(), strict=True))
    return np.array([by_site[s] for s in sites()], dtype=np.float64)


def supply_sites() -> list[str]:
    """Return the four bulk and grid supply point labels, sorted."""
    table = series_info().filter(pl.col("substation_type") != "Primary")
    return sorted(table["site"].to_list(), key=lambda s: int(s[1:]))


def supply_capacity_mw() -> np.ndarray:
    """Return the four supply points' effective capacity in MW."""
    table = series_info()
    by_site = dict(zip(table["site"].to_list(), table["effective_capacity_mw"].to_list(), strict=True))
    return np.array([by_site[s] for s in supply_sites()], dtype=np.float64)


@functools.cache
def clean_primary_power() -> np.ndarray:
    """Return each Primary's cleaned power over capacity, NaN at flagged half-hours and before the cut."""
    return c510.clean_stack()


@functools.cache
def profile_difference_supply() -> np.ndarray:
    """Return each supply point's power over capacity minus its median of the 4 earlier weeks (causal)."""
    table = series_info().filter(pl.col("substation_type") != "Primary")
    arrays = common.load_power_arrays(series=table)
    out = []
    for site in supply_sites():
        x = arrays[site].power.astype(np.float64)
        past = np.stack([np.concatenate([np.full(k * WEEK, np.nan), x[: len(x) - k * WEEK]]) for k in range(1, 5)])
        with np.errstate(all="ignore"):
            med = np.nanmedian(past, axis=0)
        out.append((x - med).astype(np.float32))
    return np.stack(out)


def labelled_events() -> dict[int, list[tuple[int, int]]]:
    """Return each Primary's labelled events as (onset, end) grid indices."""
    return c510.labelled_events()


def labelled_event_table() -> pl.DataFrame:
    """Return the 184 labelled event rows."""
    return pl.read_parquet(EVENTS)


def event_mask(*, n: int, pad: int = EVENT_PAD) -> np.ndarray:
    """Return a (16, n) boolean array, True inside a labelled event padded `pad` half-hours each side."""
    mask = np.zeros((16, n), dtype=bool)
    for p, spans in labelled_events().items():
        for onset, end in spans:
            mask[p, max(0, onset - pad) : min(n, end + pad + 1)] = True
    return mask


def robust_sd(*, x: np.ndarray) -> float:
    """Return 1.4826 times the median absolute deviation of the finite values, or NaN if none."""
    v = x[np.isfinite(x)]
    if len(v) < 100:
        return float("nan")
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


@dataclass
class Fold:
    """One fold's scaled residuals and the history statistics fixed at its refit."""

    month: str
    refit: int
    """Grid index of the refit, and of the first `further` row."""
    z: np.ndarray
    """(16, n) scaled residual, NaN where unavailable; blocks centred; junction not yet detrended."""
    half_width: np.ndarray
    spread: np.ndarray
    """(16,) robust sd of `z` over the history (scaled units)."""
    supply: np.ndarray
    """(4, n) profile-differenced supply power divided by its history robust sd."""
    edge_sd: np.ndarray
    """(20,) robust sd of the partner edge statistic over the history (16 Primaries then 4 supply)."""
    block_edges: list[int]
    cap_mw: np.ndarray
    """(20,) capacity in MW, Primaries then supply points."""

    def partners(self) -> np.ndarray:
        """Return the (20, n) partner matrix: Primary `z` then the supply series."""
        return np.concatenate([self.z, self.supply])


def edge_statistic(*, x: np.ndarray, g: int = EDGE_WINDOW) -> np.ndarray:
    """Return mean(x[t:t+g]) - mean(x[t-g:t]) for each t, NaN where either side has under g/2 values."""
    ok = np.isfinite(x)
    v = np.where(ok, x, 0.0)
    cs = np.concatenate([[0.0], np.cumsum(v)])
    cn = np.concatenate([[0], np.cumsum(ok)])
    n = len(x)
    out = np.full(n, np.nan)
    t = np.arange(g, n - g + 1)
    n_after = cn[t + g] - cn[t]
    n_before = cn[t] - cn[t - g]
    good = (n_after >= g // 2) & (n_before >= g // 2)
    after = (cs[t + g] - cs[t]) / np.maximum(n_after, 1)
    before = (cs[t] - cs[t - g]) / np.maximum(n_before, 1)
    out[t[good]] = (after - before)[good]
    return out


def build_fold(*, month: str) -> Fold:
    """Build a fold's arrays from experiment 507's table and the cleaned power."""
    table = pl.read_parquet(E507_TABLES / f"month={month}" / "seed=0.parquet")
    n = grid_length()
    clean = clean_primary_power()
    refit = index_of(t=c510.month_start(month=month))
    z = np.full((16, n), np.nan, dtype=np.float64)
    half_width = np.zeros(16)
    for p in range(16):
        sub = table.filter(pl.col("primary") == p)
        grid = sub["grid"].to_numpy().astype(np.int64)
        q50 = sub["q50"].to_numpy().astype(np.float64)
        hw = (sub["q95"].to_numpy() - sub["q05"].to_numpy()).astype(np.float64) / 2.0
        cross = sub["source"].to_numpy() == "crossfit"
        half_width[p] = float(np.median(hw[cross]))
        z[p, grid] = (clean[p, grid] - q50) / half_width[p]
    edges = [int(np.ceil(k * refit / BLOCKS)) for k in range(1, BLOCKS)]
    bounds = [0, *edges, refit]
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        for p in range(16):
            block = z[p, a:b]
            if np.isfinite(block).sum() > 100:
                z[p, a:b] = block - np.nanmedian(block)
    spread = np.array([robust_sd(x=z[p, :refit]) for p in range(16)])
    supply = profile_difference_supply().astype(np.float64)
    for k in range(supply.shape[0]):
        sd = robust_sd(x=supply[k, :refit])
        supply[k] = supply[k] / sd
    partners = np.concatenate([z, supply])
    edge_sd = np.array([robust_sd(x=edge_statistic(x=partners[k, :refit])) for k in range(20)])
    cap = np.concatenate([primary_capacity_mw(), supply_capacity_mw()])
    return Fold(
        month=month,
        refit=refit,
        z=z,
        half_width=half_width,
        spread=spread,
        supply=supply,
        edge_sd=edge_sd,
        block_edges=edges,
        cap_mw=cap,
    )


def junction_offset(*, z: np.ndarray, junction: int, end: int) -> tuple[np.ndarray, int]:
    """Return per-Primary offsets for the `further` rows at a cutoff, and the first usable row.

    The offset matches the median of the first (up to 7) observed days after the junction to the
    median of the 7 days before it, using only rows below `end`. If fewer than 2 days exist after the
    junction, the rows after it are unusable (`first_usable` is `end`).
    """
    after_n = end - junction
    if after_n < 96:
        return np.zeros(len(z)), end
    m = min(WEEK, after_n)
    offset = np.zeros(len(z))
    for p in range(len(z)):
        before = z[p, junction - WEEK : junction]
        after = z[p, junction : junction + m]
        if np.isfinite(before).sum() > 24 and np.isfinite(after).sum() > 24:
            offset[p] = np.nanmedian(before) - np.nanmedian(after)
    return offset, junction


def series_at(*, fold: Fold, end: int) -> np.ndarray:
    """Return the (20, n) partner matrix as seen with rows below `end` only (NaN from `end` on).

    `end` is the cutoff index minus the 3.5 hour cut. The junction offset is applied to the Primaries'
    `further` rows. Rows between the junction and 2 days after it are NaN when `end` is under 2 days
    past the junction.
    """
    out = fold.partners().copy()
    out[:, end:] = np.nan
    offset, first = junction_offset(z=out[:16], junction=fold.refit, end=end)
    if first >= end and end > fold.refit:
        out[:16, fold.refit : end] = np.nan
    elif end > fold.refit:
        out[:16, fold.refit : end] += offset[:, None]
    return out


def cutoff_index(*, cutoff: datetime) -> int:
    """Return the grid index of a cutoff."""
    return index_of(t=cutoff)


def daily_cutoffs(*, month: str) -> list[datetime]:
    """Return the fold's daily 00:00 UTC cutoffs (the 511 daily cadence)."""
    start, stop = c510.month_start(month=month), c510.month_stop(month=month)
    return [start + timedelta(days=d) for d in range((stop - start).days)]


def neighbour_graph() -> dict[int, list[int]]:
    """Return each Primary index's neighbour Primary indices from experiment 395's corrected table."""
    import e00456_scan as scan  # ty: ignore[unresolved-import]

    names = sites()
    out: dict[int, list[int]] = {}
    for site, others in scan.neighbour_sets().items():
        if site in names:
            out[names.index(site)] = sorted(names.index(o) for o in others if o in names)
    return out
