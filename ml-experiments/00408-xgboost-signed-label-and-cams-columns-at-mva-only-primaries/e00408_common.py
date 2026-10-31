"""Shared pieces of experiment 408: cleaned series, CAMS columns, the signed-label rule, the fold.

- **Series.** `install_patches` makes experiment 87's row builders read the cleaned power of
  `studies.power.scan_power` for every Primary, cut D15 and D18 before 2025-03-01 (training rows,
  lags, inference inputs, and scoring actuals), and, in the known-answer mode, replace the readings
  of a signed-MW Primary by the apparent power of a series with a metered solar farm's output
  injected. The mode is read from the environment variable `E408_MODE` at import, so spawned
  workers patch themselves: `real`, `abs`, or `inj:<level>`.
- **CAMS columns.** `cams_arrays` returns the unit PV shape `u` and the clear-sky index `k` of each
  half-hour on experiment 87's dense grid, from the hourly CAMS global horizontal irradiance at the
  Primary's coordinates. `k` is the half-hour's CAMS irradiance over the Haurwitz clear-sky
  irradiance. CAMS arrives about one day late, so these columns describe the half-hour itself only
  in lab conditions.
- **Signed-label rule.** `fit_rule` and `reverse_mask` implement the rule of the brief from a
  refit's training window only. `fold_quantiles` folds 11 signed quantiles back into 11 apparent
  power quantiles.
"""

import dataclasses
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_fit_demand as fd  # ty: ignore[unresolved-import]
import e00087_regimes as regimes  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from scipy.stats import rankdata
from studies.baselines import haurwitz_w_m2
from studies.dp_features import FLEET
from studies.power import scan_power
from studies.pv_physics import half_hour_sun_and_sky, plant_power_mw
from studies.sources import ML_EXPERIMENTS_DATA_DIR, REPO_DATA_DIR

SLUG: Final[str] = "xgboost-signed-label-and-cams-columns-at-mva-only-primaries"
WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"00408-{SLUG}" / "work"
CAMS_PATH: Final[Path] = (
    REPO_DATA_DIR / "studies/downloads/reanalysis/CAMS/demand_points/cams_demand_points.parquet"
)
INJECTION_FRAME_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00039-pv-disaggregation-cams-injection" / "work" / "frame.parquet"
)
MODE_ENV: Final[str] = "E408_MODE"
POWER_END: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
TRAIN_END: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)
CUT_D15_D18: Final[datetime] = datetime(2025, 3, 1, tzinfo=UTC)
CUT_SITES: Final[tuple[str, ...]] = ("D15", "D18")
MVA_ONLY: Final[tuple[str, ...]] = ("D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22")
SIGNED_SITES: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16")
FARM_OF_SITE: Final[dict[str, str]] = {
    "D5": "A",
    "D9": "B",
    "D10": "C",
    "D11": "A",
    "D14": "B",
    "D16": "C",
}
"""Three farms rotate across the six signed-MW Primaries, so each farm's output is injected at two
Primaries. The letters are the anonymised labels of experiment 39's frame."""
FARM_MAX_ZENITH_DEG: Final[float] = 75.0
FARM_DROPOUT_GHI_W_M2: Final[float] = 100.0
SUN_UP_UNIT_SHAPE: Final[float] = 0.01

GATE_CORRELATION: Final[float] = 0.3
GATE_PERCENTILE_POINTS: Final[float] = 1.0
GATE_K_MIN: Final[float] = 0.7
DEMAND_REFERENCE_K_MAX: Final[float] = 0.3
MIN_EXCESS_POINTS: Final[float] = 2.0
MIN_RULE_ROWS: Final[int] = 200
MIN_UNIT_SHAPE: Final[float] = 0.1
"""Rows with a smaller unit shape do not set the capacity `C` (it divides by the shape)."""
MIDDAY_START_HOURS: Final[tuple[int, int]] = (10, 14)
SUMMER_MONTHS: Final[tuple[int, int]] = (4, 9)
SHUFFLE_SEED: Final[int] = 408
K_CLIP: Final[float] = 1.5


# --- CAMS columns -----------------------------------------------------------------------------


@dataclass(frozen=True)
class CamsArrays:
    """The unit PV shape and clear-sky index of each grid half-hour (NaN where CAMS is missing)."""

    u: np.ndarray
    k: np.ndarray


_CAMS_CACHE: dict[str, CamsArrays] = {}


def cams_arrays(*, row: dict) -> CamsArrays:
    """Return `u` and `k` on the dense grid for one series' row of `common.series_table()`."""
    site = row["site"]
    if site in _CAMS_CACHE:
        return _CAMS_CACHE[site]
    hourly = (
        pl.read_parquet(CAMS_PATH, columns=["site", "time", "ghi_w_m2"])
        .filter(pl.col("site") == site)
        .select("time", pl.col("ghi_w_m2").cast(pl.Float64))
        .drop_nulls()
        .sort("time")
    )
    sky = half_hour_sun_and_sky(hourly=hourly, latitude=row["latitude"], longitude=row["longitude"])
    clear = haurwitz_w_m2(apparent_zenith_deg=sky.zenith_deg)
    index = np.divide(sky.ghi_w_m2, clear, out=np.zeros_like(clear), where=clear > 1.0)
    stamps = pl.Series(sky.half_hour_end_time).dt.replace_time_zone("UTC")
    frame = pl.DataFrame(
        {
            "valid_time": stamps,
            "cams_u": plant_power_mw(sky=sky, parameters=FLEET),
            "cams_k": np.clip(index, 0.0, K_CLIP),
        }
    )
    joined = pl.DataFrame({"valid_time": common.grid_times()}).join(
        frame, on="valid_time", how="left", validate="1:1", maintain_order="left"
    )
    out = CamsArrays(
        u=joined["cams_u"].to_numpy().astype(np.float64),
        k=joined["cams_k"].to_numpy().astype(np.float64),
    )
    _CAMS_CACHE[site] = out
    return out


def start_hour_and_month(*, grid: pl.Series) -> tuple[np.ndarray, np.ndarray]:
    """Return the UTC hour and month of each half-hour's start (the grid stamps its end)."""
    starts = grid.dt.offset_by("-30m")
    return starts.dt.hour().to_numpy(), starts.dt.month().to_numpy()


# --- the signed-label rule ----------------------------------------------------------------------


@dataclass(frozen=True)
class SignedRule:
    """One Primary's rule at one refit, fitted on the training window only.

    Attributes:
        gate_open: Whether the training readings show a solar dip (experiment 366's rule).
        spearman: Spearman correlation of reading and `u` on sunny summer middays (NaN if too few).
        p1_points: The 1st percentile of those readings, in points of effective capacity.
        n_midday: The number of sunny summer-midday training half-hours behind the two statistics.
        d_ref: The reference demand by UTC start hour, in points (NaN for an hour with no rows).
        capacity: `C`, the fitted points of effective capacity per unit of the shape `u` (NaN if
            there were too few rows).
    """

    gate_open: bool
    spearman: float
    p1_points: float
    n_midday: int
    d_ref: np.ndarray
    capacity: float


def _summer_midday(*, hour: np.ndarray, month: np.ndarray) -> np.ndarray:
    return (
        (hour >= MIDDAY_START_HOURS[0])
        & (hour < MIDDAY_START_HOURS[1])
        & (month >= SUMMER_MONTHS[0])
        & (month <= SUMMER_MONTHS[1])
    )


def fit_rule(
    *,
    y_points: np.ndarray,
    u: np.ndarray,
    k: np.ndarray,
    hour: np.ndarray,
    month: np.ndarray,
    usable: np.ndarray,
) -> SignedRule:
    """Fit the rule on the half-hours where `usable` holds (the training window).

    Args:
        y_points: The readings in points of effective capacity (NaN where missing).
        u: The CAMS unit PV shape.
        k: The CAMS clear-sky index.
        hour: The UTC start hour of each half-hour.
        month: The UTC start month of each half-hour.
        usable: The training window: finite reading, shape and index, and inside the window.

    Returns:
        The fitted rule.
    """
    sunny = usable & _summer_midday(hour=hour, month=month) & (k >= GATE_K_MIN)
    n = int(sunny.sum())
    spearman, p1 = float("nan"), float("nan")
    gate = False
    if n >= MIN_RULE_ROWS:
        ranks_y, ranks_u = rankdata(y_points[sunny]), rankdata(u[sunny])
        spearman = float(np.corrcoef(ranks_y, ranks_u)[0, 1])
        p1 = float(np.percentile(y_points[sunny], 1.0))
        gate = spearman > GATE_CORRELATION or p1 < GATE_PERCENTILE_POINTS
    summer = usable & (month >= SUMMER_MONTHS[0]) & (month <= SUMMER_MONTHS[1])
    dark = summer & (k < DEMAND_REFERENCE_K_MAX)
    d_ref = np.full(24, np.nan)
    for h in range(24):
        in_hour = dark & (hour == h)
        if in_hour.sum() >= MIN_RULE_ROWS // 4:
            d_ref[h] = float(np.median(y_points[in_hour]))
    capacity = float("nan")
    clear = sunny & (u >= MIN_UNIT_SHAPE) & np.isfinite(d_ref[hour])
    if clear.sum() >= MIN_RULE_ROWS // 4:
        capacity = float(np.median((y_points[clear] + d_ref[hour][clear]) / u[clear]))
    return SignedRule(
        gate_open=gate, spearman=spearman, p1_points=p1, n_midday=n, d_ref=d_ref, capacity=capacity
    )


def reverse_mask(
    *,
    rule: SignedRule,
    tau: float,
    y_points: np.ndarray,
    u: np.ndarray,
    k: np.ndarray,
    hour: np.ndarray,
) -> np.ndarray:
    """Return the half-hours the rule calls reverse flow (false everywhere if the gate is shut).

    A reading is reverse flow when the gate is open, `k` is at least `tau`, `C u` exceeds the
    reference demand of the hour, and `C u` minus the reading is at least 2 points.
    """
    if not rule.gate_open or not np.isfinite(rule.capacity) or not np.isfinite(tau):
        return np.zeros(len(y_points), dtype=bool)
    d_ref = rule.d_ref[hour]
    modelled = rule.capacity * u
    with np.errstate(invalid="ignore"):
        return (
            np.isfinite(y_points)
            & np.isfinite(k)
            & (k >= tau)
            & (modelled > d_ref)
            & (modelled - y_points >= MIN_EXCESS_POINTS)
        )


def shuffled_k(
    *, k: np.ndarray, hour: np.ndarray, month: np.ndarray, year: np.ndarray
) -> np.ndarray:
    """Return `k` with its values permuted across days inside each (year, month, hour) group.

    This keeps each month's and hour's distribution of `k` and breaks its link to the day's
    readings, so an arm that labels with it measures how much the label owes to the real weather.
    """
    rng = np.random.default_rng(SHUFFLE_SEED)
    out = k.copy()
    keys = (year * 100 + month) * 24 + hour
    order = np.argsort(keys, kind="stable")
    bounds = np.flatnonzero(np.diff(keys[order])) + 1
    for group in np.split(order, bounds):
        out[group] = k[rng.permutation(group)]
    return out


# --- the fold -----------------------------------------------------------------------------------


def fold_quantiles(*, signed: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Fold signed quantiles into quantiles of the absolute value.

    Build the piecewise-linear distribution function `F` through `(signed quantile, level)`, zero
    below the lowest quantile and one above the highest, and set `H(x) = F(x) - F(-x)` for
    `x >= 0`. The folded quantile at level `a` is the inverse of `H`. It is the identity whenever
    the lowest quantile is at least zero.

    Args:
        signed: Shape `(rows, n_levels)`, non-decreasing along the second axis.
        levels: The `n_levels` probability levels.

    Returns:
        Shape `(rows, n_levels)`: the quantiles of `|x|` at the same levels.
    """
    rows, n = signed.shape
    ordered = np.sort(signed, axis=1)
    candidates = np.sort(np.concatenate([np.zeros((rows, 1)), np.abs(ordered)], axis=1), axis=1)

    def distribution(x: np.ndarray) -> np.ndarray:
        """Return `F(x)` for `x` of shape `(rows, m)`."""
        count = (x[:, :, None] >= ordered[:, None, :]).sum(axis=2)
        lower = np.clip(count - 1, 0, n - 2)
        q_lo = np.take_along_axis(ordered, lower, axis=1)
        q_hi = np.take_along_axis(ordered, lower + 1, axis=1)
        width = q_hi - q_lo
        frac = np.divide(x - q_lo, width, out=np.ones_like(width), where=width > 0)
        value = levels[lower] + np.clip(frac, 0.0, 1.0) * (levels[lower + 1] - levels[lower])
        return np.where(count == 0, 0.0, np.where(x > ordered[:, -1:], 1.0, value))

    h = distribution(candidates) - distribution(-candidates)
    h = np.maximum.accumulate(h, axis=1)
    out = np.empty_like(ordered)
    for i, a in enumerate(levels):
        above = (h >= a - 1e-9).argmax(axis=1)
        reached = (h >= a - 1e-9).any(axis=1)
        j = np.clip(above, 1, candidates.shape[1] - 1)
        h_lo = np.take_along_axis(h, j[:, None] - 1, axis=1)[:, 0]
        h_hi = np.take_along_axis(h, j[:, None], axis=1)[:, 0]
        x_lo = np.take_along_axis(candidates, j[:, None] - 1, axis=1)[:, 0]
        x_hi = np.take_along_axis(candidates, j[:, None], axis=1)[:, 0]
        span = h_hi - h_lo
        t = np.divide(a - h_lo, span, out=np.ones_like(span), where=span > 0)
        interpolated = x_lo + np.clip(t, 0.0, 1.0) * (x_hi - x_lo)
        out[:, i] = np.where(
            reached, np.where(above == 0, candidates[:, 0], interpolated), candidates[:, -1]
        )
    return out


# --- series patches -----------------------------------------------------------------------------


def farm_norm(*, farm: str, u: np.ndarray) -> np.ndarray:
    """Return a metered farm's output over its 99th percentile on the dense grid.

    NaN marks a half-hour where the farm reading is missing or has dropped out while the sun is
    up, so the Primary's row leaves training; other negative readings are set to zero. The farm's
    series stays in the private store.
    """
    everything = pl.read_parquet(INJECTION_FRAME_PATH).filter(
        pl.col("site") == farm, pl.col("kind") == "farm"
    )
    frame = everything.filter(pl.col("zenith") < FARM_MAX_ZENITH_DEG)
    power = frame.filter(pl.col("power").is_not_null())
    p99 = float(np.nanpercentile(power["power"].to_numpy(), 99))
    dropout = (pl.col("power") <= 0.0) & (pl.col("ghi_u") > FARM_DROPOUT_GHI_W_M2)
    daylight = frame.select(
        "time",
        norm=pl.when(pl.col("power").is_null() | dropout)
        .then(float("nan"))
        .otherwise(pl.col("power").clip(lower_bound=0.0) / p99),
    )
    grid = pl.DataFrame({"time": common.grid_times()})
    joined = grid.join(daylight, on="time", how="left", validate="1:1", maintain_order="left")
    norm = joined["norm"].fill_null(0.0).to_numpy().astype(np.float64)
    seen = grid.join(everything.select("time"), on="time", how="semi")["time"]
    missing = ~np.isin(grid["time"].to_numpy(), seen.to_numpy())
    norm[missing & (np.nan_to_num(u, nan=0.0) > SUN_UP_UNIT_SHAPE)] = np.nan
    return norm


@dataclass(frozen=True)
class Injected:
    """A signed-MW Primary with a farm injected: the truth, and what an MVA meter would read.

    Both are fractions of effective capacity on the dense grid; NaN where the Primary's reading or
    the farm's is missing. `scale` is the injected capacity in the same units.
    """

    truth: np.ndarray
    apparent: np.ndarray
    scale: float


def injection(*, row: dict, arrays: common.SeriesArrays, level: float) -> Injected:
    """Inject a farm at `level` times the median sunny-midday reading of the training window.

    The reading is `sqrt(P'^2 + Q^2)` with `Q = 0`, so the apparent power is `|P'|`. The farm's
    output never enters the model inputs, the rule, or the shape `u`.
    """
    cams = cams_arrays(row=row)
    hour, month = start_hour_and_month(grid=common.grid_times())
    valid = common.grid_times()
    window = (valid < TRAIN_END).to_numpy() & np.isfinite(arrays.power)
    sunny = window & _summer_midday(hour=hour, month=month) & (np.nan_to_num(cams.k) >= GATE_K_MIN)
    scale = level * float(np.median(np.abs(arrays.power[sunny])))
    norm = farm_norm(farm=FARM_OF_SITE[row["site"]], u=cams.u)
    truth = arrays.power.astype(np.float64) - scale * norm
    return Injected(truth=truth, apparent=np.abs(truth), scale=scale)


def cleaned_power(*, series: pl.DataFrame) -> pl.DataFrame:
    """Return the cleaned half-hourly power (MW) of every series before 2026-07-01."""
    ids = series["time_series_id"].to_list()
    return (
        scan_power()
        .filter(pl.col("time_series_id").is_in(ids) & (pl.col("time") < POWER_END))
        .select("time_series_id", "time", "power")
        .sort("time_series_id", "time")
        .collect()
    )


def cut_of(*, site: str) -> datetime:
    """Return the first valid time a Primary's rows may use (the D15 and D18 cut)."""
    return CUT_D15_D18 if site in CUT_SITES else datetime(2000, 1, 1, tzinfo=UTC)


def install_patches(*, mode: str) -> None:
    """Make experiment 87's row builders use cleaned power, the cuts, and the injection.

    Args:
        mode: `real` (cleaned readings), `abs` (the signed-MW Primaries' readings replaced by their
            absolute value, so they look like MVA readings; the signed reading is the truth), or
            `inj:<level>` (the signed-MW Primaries' readings replaced by the apparent power of a
            series with a farm injected).
    """
    if mode not in {"real", "abs"} and not mode.startswith("inj:"):
        msg = f"unknown mode {mode!r}; use real, abs, or inj:<level>"
        raise ValueError(msg)
    original_series_and_arrays = fd._series_and_arrays
    original_training_rows = fd.training_rows
    original_known_rows = fd.known_rows

    def series_and_arrays() -> tuple[pl.DataFrame, dict[str, common.SeriesArrays]]:
        first_call = "series" not in fd._CACHE
        series, arrays = original_series_and_arrays()
        if first_call:
            scanned = cleaned_power(series=series)
            truths: dict[str, np.ndarray] = {}
            for row in series.iter_rows(named=True):
                site = row["site"]
                mine = scanned.filter(pl.col("time_series_id") == row["time_series_id"])
                grid = np.full(len(arrays[site].power), np.nan, dtype=np.float32)
                index = np.asarray(common.grid_index(times=mine["time"]))
                grid[index] = (mine["power"] / row["effective_capacity_mw"]).to_numpy()
                cut = common.grid_index(times=cut_of(site=site))
                grid[: max(int(cut), 0)] = np.nan
                arrays[site] = dataclasses.replace(arrays[site], power=grid)
                if mode == "abs" and site in SIGNED_SITES:
                    truths[site] = arrays[site].power.astype(np.float64)
                    arrays[site] = dataclasses.replace(
                        arrays[site], power=np.abs(arrays[site].power)
                    )
                elif mode.startswith("inj:") and site in SIGNED_SITES:
                    injected = injection(row=row, arrays=arrays[site], level=float(mode[4:]))
                    truths[site] = injected.truth
                    arrays[site] = dataclasses.replace(
                        arrays[site], power=injected.apparent.astype(np.float32)
                    )
            fd._CACHE["truth"] = truths
            fd._CACHE["mode"] = mode
        return series, arrays

    def training_rows(
        *, row: dict, arrays: common.SeriesArrays, frozen: bool, a7: bool
    ) -> pl.DataFrame:
        rows = original_training_rows(row=row, arrays=arrays, frozen=frozen, a7=a7)
        index = np.asarray(common.grid_index(times=rows["valid_time"]))
        label = common.take(values=arrays.power, index=index) * row["effective_capacity_mw"]
        return rows.with_columns(power=pl.Series(label).cast(pl.Float32)).filter(
            pl.col("power").is_finite()
        )

    def known_rows(*, rows: pl.DataFrame, refit: datetime, config: object) -> pl.DataFrame:
        known = original_known_rows(rows=rows, refit=refit, config=config)
        series = fd._CACHE["series"]
        site = series.filter(pl.col("time_series_id") == rows["time_series_id"][0])["site"].item()
        return known.filter(pl.col("valid_time") >= cut_of(site=site))

    fd._series_and_arrays = series_and_arrays
    fd.training_rows = training_rows
    fd.known_rows = known_rows


def latest_valid(*, refit: datetime) -> datetime:
    """Return the newest valid time a refit may train on."""
    return refit - regimes.REFIT_CUTOFF_AGE


if MODE_ENV in os.environ:
    install_patches(mode=os.environ[MODE_ENV])
