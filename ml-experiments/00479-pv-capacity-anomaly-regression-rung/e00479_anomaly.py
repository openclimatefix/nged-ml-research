"""The anomaly-regression rung of experiment 479.

The rung estimates capacity from day-to-day weather variation within a month only.

For one Primary, the estimator removes from the reading and from every regressor the mean of the
same (calendar month, hour of day, weekday class), then regresses the reading's anomaly on the
physics PV shape's anomaly with one capacity coefficient per season-year (July to June). The
temperature columns enter the same regression, so heating and cooling load is fitted jointly with
the PV capacity. A month-block bootstrap resamples the months.

**Privacy.** The farms (labels A to F) are metered generators. Only ratios of recovered to injected
capacity and coarse distance bands may leave the private data folder.

The plan is in `docs/ml-experiments/00479-pv-capacity-anomaly-regression-rung/result.md`. Run
scripts with `PYTHONPATH` set to the folders of experiments 269, 129, and 474.
"""

from dataclasses import dataclass
from typing import Final

import e00269_common as c269  # ty: ignore[unresolved-import]
import e00269_fit as f269  # ty: ignore[unresolved-import]
import e00474_common as c474  # ty: ignore[unresolved-import]
import numpy as np
from studies.dp_features import FeatureCache
from studies.dp_gam import derated_unit_shape

HINGES_C: Final[tuple[float, ...]] = (5.0, 10.0, 15.0, 20.0)
"""Knots of the piecewise-linear temperature response, in degrees Celsius."""
HOUR_BANDS: Final[tuple[tuple[int, int], ...]] = ((0, 8), (8, 16), (16, 24))
"""Local hours of the three bands whose temperature response is fitted separately."""
RIDGE_RELATIVE: Final[float] = 1e-6
"""Ridge added to the normal equations, relative to the mean diagonal."""
N_BOOTSTRAP: Final[int] = 2000
"""Month-block bootstrap draws."""
FIT_START_D15_D18: Final[np.datetime64] = np.datetime64("2025-03-01")
"""D15 and D18 drop every row before this date (maintainer, 2026-10-30)."""
READ_OUT_SEASON_YEAR: Final[int] = c269.READ_OUT_SEASON_YEAR
"""The season-year (July 2024 to June 2025) at which injection recovery is read."""
CLOUD_LOAD_SHAPES: Final[tuple[str, ...]] = ("linear", "squared")
MIN_ROWS_PER_SEASON_YEAR: Final[int] = 4000
"""A season-year needs this many fitted half-hours to get its own capacity column."""
MIN_MONTHS_PER_SEASON_YEAR: Final[int] = 3
"""... and rows in at least this many calendar months."""


@dataclass(frozen=True)
class Prepared:
    """One Primary's regression inputs on the full grid.

    Attributes:
        site: The Primary's label.
        eff_mw: The Primary's effective capacity in megawatts (points to MW factor is eff / 100).
        y_mw: The reading in megawatts, `NaN` where the row is not trained on.
        shape: The derated physics unit shape per MW AC, `NaN` where CAMS is missing.
        raw_columns: Temperature columns (T, T24 and hinges, by hour band), shape (grid, k).
        cloud_columns: The two cloud columns of experiment 474, shape (grid, 2).
        group: Integer id of (month, hour, weekday class) per row.
        month: Month since 1970-01 per row.
        season_year: Season-year per row.
        usable: True where the row may be fitted (default row, reading and shape finite).
    """

    site: str
    eff_mw: float
    y_mw: np.ndarray
    shape: np.ndarray
    raw_columns: np.ndarray
    cloud_columns: np.ndarray
    group: np.ndarray
    month: np.ndarray
    season_year: np.ndarray
    usable: np.ndarray


def prepare(*, cache: FeatureCache, site: str) -> Prepared:
    """Return a Primary's inputs: reading, shape, temperature and cloud columns, group ids."""
    f = cache.sites[site]
    eff = float(f.table.effective_capacity_mw)
    n = len(f.y)
    grid = np.arange(n)
    shape = derated_unit_shape(features=f, index=grid)
    start = cache.end_times - np.timedelta64(30, "m")
    usable = f.default_rows & np.isfinite(f.y) & np.isfinite(shape)
    if site in ("D15", "D18"):
        usable &= start >= FIT_START_D15_D18
    cal = cache.calendar
    hour = (cal.half_hour // 2).astype(np.int64)
    days = cal.local_date.astype("datetime64[D]").astype(np.int64)
    weekday = (days + 3) % 7
    day_class = np.where(weekday < 5, 0, weekday - 4)
    month = cal.local_date.astype("datetime64[M]").astype(np.int64)
    key = (month - month.min()) * 144 + hour * 3 + day_class
    group = np.unique(key, return_inverse=True)[1]
    t = f.temperature_c
    t24 = f.smoothed_temperature_c[:, 1]
    base_columns = [t, t24]
    for knot in HINGES_C:
        base_columns.append(np.clip(t - knot, 0, None))
        base_columns.append(np.clip(knot - t, 0, None))
    base = np.nan_to_num(np.stack(base_columns, axis=1), nan=0.0)
    banded = []
    for lo, hi in HOUR_BANDS:
        in_band = ((hour >= lo) & (hour < hi))[:, None]
        banded.append(np.where(in_band, base, 0.0))
    raw = np.concatenate(banded, axis=1)
    return Prepared(
        site=site,
        eff_mw=eff,
        y_mw=np.where(usable, f.y * eff / 100.0, np.nan),
        shape=shape,
        raw_columns=raw,
        cloud_columns=c474.cloud_regressors(cache=cache, site=site),
        group=group,
        month=month,
        season_year=cal.season_year.astype(np.int64),
        usable=usable,
    )


def group_demean(*, values: np.ndarray, group: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Return `values` minus the mean of its group over `rows`; zero outside `rows`.

    `values` has shape (grid,) or (grid, k).
    """
    two_d = values.ndim == 2
    v = values if two_d else values[:, None]
    n_groups = int(group.max()) + 1
    counts = np.bincount(group[rows], minlength=n_groups).astype(np.float64)
    out = np.zeros_like(v, dtype=np.float64)
    for j in range(v.shape[1]):
        sums = np.bincount(group[rows], weights=v[rows, j], minlength=n_groups)
        mean = np.divide(sums, counts, out=np.zeros(n_groups), where=counts > 0)
        out[rows, j] = v[rows, j] - mean[group[rows]]
    return out if two_d else out[:, 0]


@dataclass(frozen=True)
class Design:
    """The anomaly design for one Primary on one row set.

    Attributes:
        rows: Grid positions of the fitted rows.
        x: The design, shape (n_rows, p): capacity columns, then nuisance columns.
        y: The reading's anomaly in megawatts, shape (n_rows,).
        years: The season-years that have a capacity column.
        n_nuisance: Number of nuisance columns after the capacity columns.
        month_slot: Row's month index in `months`.
        months: The months present.
        prepared: The inputs.
        mask: The boolean row mask on the full grid.
    """

    rows: np.ndarray
    x: np.ndarray
    y: np.ndarray
    years: np.ndarray
    n_nuisance: int
    month_slot: np.ndarray
    months: np.ndarray
    prepared: Prepared
    mask: np.ndarray


def build_design(
    *,
    prepared: Prepared,
    mask: np.ndarray,
    cloud_terms: bool,
    shape_override: np.ndarray | None = None,
    cloud_override: np.ndarray | None = None,
) -> Design:
    """Return the anomaly design on the rows in `mask` (a subset of `prepared.usable`)."""
    p = prepared
    rows = np.flatnonzero(mask)
    shape = p.shape if shape_override is None else shape_override
    shape_anomaly = group_demean(values=np.nan_to_num(shape, nan=0.0), group=p.group, rows=mask)
    nuisance = group_demean(values=p.raw_columns, group=p.group, rows=mask)
    if cloud_terms:
        raw_cloud = p.cloud_columns if cloud_override is None else cloud_override
        cloud = group_demean(values=raw_cloud, group=p.group, rows=mask)
        nuisance = np.concatenate([nuisance, cloud], axis=1)
    sy = p.season_year[rows]
    candidates = np.unique(sy)
    row_month = p.month[rows]
    years = np.array(
        [
            k
            for k in candidates
            if (sy == k).sum() >= MIN_ROWS_PER_SEASON_YEAR
            and len(np.unique(row_month[sy == k])) >= MIN_MONTHS_PER_SEASON_YEAR
        ]
    )
    capacity = np.zeros((len(rows), len(years)))
    for j, k in enumerate(years):
        capacity[:, j] = np.where(sy == k, shape_anomaly[rows], 0.0)
    x = np.concatenate([capacity, nuisance[rows]], axis=1)
    keep_rows = np.isin(sy, years)
    months = np.unique(p.month[rows[keep_rows]])
    month_slot = np.searchsorted(months, p.month[rows])
    kept_slot = month_slot[keep_rows]
    if not np.array_equal(months[kept_slot], p.month[rows][keep_rows]):
        raise ValueError("a kept row has a month outside the fitted months")
    y_anomaly = group_demean(values=np.nan_to_num(p.y_mw, nan=0.0), group=p.group, rows=mask)
    return Design(
        rows=rows[keep_rows],
        x=x[keep_rows],
        y=y_anomaly[rows][keep_rows],
        years=years,
        n_nuisance=nuisance.shape[1],
        month_slot=month_slot[keep_rows],
        months=months,
        prepared=p,
        mask=mask,
    )


def monthly_normal_equations(
    *, design: Design, extra_targets: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-month Gram matrices (M, p, p) and right-hand sides (M, p, T).

    Column 0 of the right-hand sides is the reading's anomaly. `extra_targets`, shape (T-1, grid),
    holds further series, each added to the reading (so a column is the reading plus the series,
    group-demeaned; the reading is column 0 plus the series' own term). Because
    the design columns are anomalies, the right-hand side of the demeaned sum equals the right-hand
    side of the raw sum, so the extra series need no demeaning.
    """
    d = design
    n_months = len(d.months)
    p = d.x.shape[1]
    t = 1 + (0 if extra_targets is None else extra_targets.shape[0])
    gram = np.zeros((n_months, p, p))
    rhs = np.zeros((n_months, p, t))
    for m in range(n_months):
        members = d.month_slot == m
        xm = d.x[members]
        gram[m] = xm.T @ xm
        rhs[m, :, 0] = xm.T @ d.y[members]
        if extra_targets is not None:
            for k in range(extra_targets.shape[0]):
                series = np.nan_to_num(extra_targets[k, d.rows[members]], nan=0.0)
                rhs[m, :, k + 1] = rhs[m, :, 0] + xm.T @ series
    return gram, rhs


def month_counts(*, months: np.ndarray, n_resamples: int, seed: int) -> np.ndarray:
    """Return month weights (n_resamples + 1, M); row 0 is all ones (the point fit)."""
    return f269.month_block_counts(months=months, n_resamples=n_resamples, seed=seed)


def solve(*, gram: np.ndarray, rhs: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Return coefficients (B, p, T) for month weights `counts` (B, M)."""
    g = np.einsum("bm,mpq->bpq", counts, gram)
    r = np.einsum("bm,mpt->bpt", counts, rhs)
    scale = np.trace(g, axis1=1, axis2=2) / g.shape[1]
    g = g + (RIDGE_RELATIVE * scale)[:, None, None] * np.eye(g.shape[1])[None]
    return np.linalg.solve(g, r)


def capacity_paths(*, coef: np.ndarray, design: Design) -> np.ndarray:
    """Return capacity (MW AC) per season-year, shape (B, n_years, T): the negated coefficient."""
    return -coef[:, : len(design.years), :]


def shrink_rises_only(*, paths: np.ndarray) -> np.ndarray:
    """Return the capacity paths with the capacity-rises prior: clip at zero, never fall.

    `paths` has the season-year on axis 1. The running maximum over the years resists a fall.
    """
    return np.maximum.accumulate(np.clip(paths, 0.0, None), axis=1)


def readout_column(*, design: Design, season_year: int) -> int:
    """Return the column of `season_year` in the capacity paths, or -1 if it has none."""
    where = np.flatnonzero(design.years == season_year)
    return int(where[0]) if len(where) else -1


def analytic_standard_error(*, design: Design, gram: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Return the ordinary-least-squares standard error of each capacity coefficient (MW)."""
    g = gram.sum(axis=0)
    g = g + RIDGE_RELATIVE * np.trace(g) / g.shape[0] * np.eye(g.shape[0])
    theta = np.linalg.solve(g, rhs.sum(axis=0)[:, 0])
    residual = design.y - design.x @ theta
    dof = max(len(design.y) - g.shape[0], 1)
    sigma2 = float(residual @ residual) / dof
    return np.sqrt(sigma2 * np.diag(np.linalg.inv(g))[: len(design.years)])


def variance_inflation(*, design: Design, gram: np.ndarray) -> np.ndarray:
    """Return the variance inflation factor of each capacity column against all other columns."""
    g = gram.sum(axis=0)
    diag = np.diag(g)
    inverse = np.linalg.inv(g + RIDGE_RELATIVE * np.trace(g) / g.shape[0] * np.eye(g.shape[0]))
    return (diag * np.diag(inverse))[: len(design.years)]


def signal_share(*, prepared: Prepared, mask: np.ndarray) -> float:
    """Return the share of the shape's variance that survives the group-mean removal."""
    shape = np.nan_to_num(prepared.shape, nan=0.0)
    anomaly = group_demean(values=shape, group=prepared.group, rows=mask)
    total = float(np.var(shape[mask]))
    return float(np.var(anomaly[mask])) / total if total > 0 else float("nan")


def shuffle_within_month(*, cache: FeatureCache, series: np.ndarray, seed: int) -> np.ndarray:
    """Return `series` with whole days permuted within each calendar year-month.

    Each half-hour slot keeps its time of day, and the series' mean for the month is unchanged,
    so after the group-mean removal only the day-to-day link to the reading is broken.
    """
    day_number, _, slot = c474.day_slot(cache=cache)
    month = (cache.end_times - np.timedelta64(30, "m")).astype("datetime64[M]").astype(np.int64)
    first = day_number.min()
    n_days = day_number.max() - first + 1
    table = np.full((n_days, 48), np.nan)
    table[day_number - first, slot] = series
    day_month = np.full(n_days, -1)
    day_month[day_number - first] = month
    mapping = np.arange(n_days)
    rng = np.random.default_rng(seed)
    for value in np.unique(day_month[day_month >= 0]):
        members = np.flatnonzero(day_month == value)
        mapping[members] = rng.permutation(members)
    shuffled = table[mapping[day_number - first], slot]
    return np.where(np.isfinite(series), shuffled, np.nan)


def recovery_factor(*, prepared: Prepared, design: Design, norm: np.ndarray) -> float:
    """Return the ratio of 99th percentiles (derated shape over the farm's normalised series)."""
    p = prepared
    rows = design.rows
    in_year = p.season_year[rows] == READ_OUT_SEASON_YEAR
    derated = p.shape[rows]
    daylight = in_year & (derated > c269.DAYLIGHT_UNIT_SHAPE) & (norm[rows] > 0.0)
    return float(np.percentile(derated[daylight], 99) / np.percentile(norm[rows][daylight], 99))
