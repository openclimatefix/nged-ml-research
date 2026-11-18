# ruff: noqa: E501
"""Experiment 706: the shared code of stages 0 and 1 (CPU only, nothing is fitted by a network).

The stored one-year backtest holds one forecast per Primary, per daily 12:00 UTC origin, per
half-hour of the next 14 days. This module lays those forecasts, the cleaned power, and the
ensemble weather out on dense arrays and implements the look-back offsets of the design:

- Power and quantiles are in points of capacity (percent of the series' capacity in the
  `series_capacity.parquet` of the backtest folder).
- The half-hour grid starts at 2025-11-01 00:00 UTC. Stamp `i` is the valid time `GRID_START + i *
  30 min`, which is the end of the half-hour, as in the stored files. Origin `o` is at stamp
  `ORIGIN_STAMP + PER_DAY * o` (12:00 UTC of day `o`), and its row `h` is the valid time `h`
  half-hours after the origin.
- The holdout is dropped on loading: no row with a valid time on or after 2026-07-01 is read.
- The residual of a past half-hour is the cleaned power minus the stored normal-running median of a
  forecast issued for it, so a positive residual is added to the forecast.
- A look-back offset for origin `t` reads residuals of half-hours up to `t` minus a lag (6 hours:
  power at 12:00 UTC is known to 06:00 UTC), over a window of 7, 14, or 28 days, and the weather
  of the ensemble run of 00:00 UTC on the day of each forecast that produced a residual.

Run any script from the repository root with `DATA_PATH_INTERNAL` and the dummy `NGED_S3_*` variables
set, as `CLAUDE.md` says.
"""

import datetime as dt
import resource
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Final, NamedTuple

import numpy as np
import polars as pl
from studies.eval_protocol.metrics import LEVELS, MEDIAN_INDEX, QUANTILE_COLUMNS, TAIL_COLUMNS
from studies.experiment_store import HOLDOUT_START
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = (
    "inference-time-level-tracking-part-1-look-back-offsets-by-driver-on-stored-forecasts"
)
EXPERIMENT_ID: Final[int] = 706
OUT_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"00706-{SLUG}"
"""The experiment's private data folder: predictions, the residual table, and score tables."""
BACKTEST_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "backtest_year_2025-11_to_2026-10"
E607_WORK: Final[Path] = next(ML_EXPERIMENTS_DATA_DIR.glob("00607-*")) / "work"
SITE_MAP_GLOB: Final[str] = "00521-*/site_map.csv"

ROW_SET_ID: Final[str] = "backtest_year_2025_12_to_2026_06_daily_12utc_pre_holdout"
"""The row set of C1 and C2: origins 2025-12-01 to 2026-06-30 (212 origins, seven month blocks)."""
ROW_SET_ID_FROM_JANUARY: Final[str] = "backtest_year_2026_01_to_2026_06_daily_12utc_pre_holdout"
"""The row set of C5 and of the 28-day arms: origins 2026-01-01 to 2026-06-30 (181 origins)."""

GRID_START: Final[dt.datetime] = dt.datetime(2025, 11, 1, tzinfo=dt.UTC)
"""Stamp 0: the valid time 2025-11-01 00:00 UTC."""
LIB_GRID_START: Final[dt.datetime] = dt.datetime(2019, 9, 1, tzinfo=dt.UTC)
"""The grid origin of experiment 668's scoring library and of the event spans of experiment 688."""
N_ORIGINS: Final[int] = 242
"""Daily origins from 2025-11-01 12:00 UTC to 2026-06-30 12:00 UTC."""
N_H: Final[int] = 672
"""Half-hours per forecast: valid times from the origin to 11:30 UTC 14 days later."""
PER_DAY: Final[int] = 48
N_STAMPS: Final[int] = N_ORIGINS * PER_DAY
ORIGIN_STAMP: Final[int] = 24
"""The stamp of origin 0 (12:00 UTC on 2025-11-01), 24 half-hours after `GRID_START`."""
STEP_US: Final[int] = 1_800_000_000
DAY_US: Final[int] = 86_400_000_000
GRID_START_US: Final[int] = int(GRID_START.timestamp() * 1_000_000)
ORIGIN0_US: Final[int] = GRID_START_US + ORIGIN_STAMP * STEP_US
LIB_OFFSET: Final[int] = int((GRID_START - LIB_GRID_START).total_seconds() // 1800)
"""Half-hours from the library grid start to stamp 0."""

FIRST_ORIGIN_DEC: Final[int] = 30
"""Origin index of 2025-12-01, the first scored origin of C1 and C2."""
FIRST_ORIGIN_JAN: Final[int] = 61
"""Origin index of 2026-01-01, the first scored origin of C5."""
MAX_LEAD_HALF_HOURS: Final[int] = 480
"""Scored rows have a lead above 0 and up to 240 hours, the protocol's headline band."""
LAG_6H: Final[int] = 12
"""Half-hours between the origin and the newest residual: the 06:00 UTC reading at a 12:00 origin."""
LAG_3H_CAP: Final[int] = 18
"""Half-hours for the 03:00 UTC cap of the stored prevailing rows (C5)."""
PENALTY_HALF_HOURS: Final[float] = 48.0
"""A ridge penalty equals this many half-hours of data at the driver's variance."""
MIN_ACTIVE: Final[int] = 24
"""A slope is zero when fewer window half-hours than this have its driver above the threshold."""
ACTIVE_THRESHOLD: Final[float] = 0.25
MIN_COVERAGE: Final[float] = 0.5
"""A window needs this share of its half-hours to hold a residual, else the offset is zero."""
FACTOR_LOOK_BACK: Final[float] = 0.75
"""The share of the look-back offset applied by G and D (recipe R, experiment 115)."""
SHORTWAVE_SCALE: Final[float] = 800.0
COLD_THRESHOLD_C: Final[float] = 5.0
COLD_SCALE_K: Final[float] = 5.0
PV_DRIVER: Final[int] = 0
COLD_DRIVER: Final[int] = 1
DRIVER_NAMES: Final[tuple[str, ...]] = ("x_pv", "x_cold")

BAND_UPPER_H: Final[tuple[int, ...]] = (48, 96, 192, 336, 480, 672)
"""Upper edges, in half-hours, of the six lead bands of `demand-v2`: (0, 24], (24, 48], (48, 96],
(96, 168], (168, 240], and (240, 336] hours."""
BAND_NAMES: Final[tuple[str, ...]] = (
    "day 1",
    "day 2",
    "days 3 to 4",
    "days 5 to 7",
    "days 8 to 10",
    "days 11 to 14",
)
N_BANDS: Final[int] = 6
BAND_OF_H: Final[np.ndarray] = np.searchsorted(np.array(BAND_UPPER_H), np.arange(N_H), side="left")
"""The lead band of the forecast row `h`: lead `h / 2` hours lies in `(low, high]` of the band."""
LOOK_BACK_LEADS: Final[int] = 14
"""A half-hour has a forecast from each of up to 14 origins: leads of 0 to 14 days."""

GROUP_SIGNED: Final[str] = "signed-MW six"
GROUP_MVA: Final[str] = "MVA-only eight"
GROUP_SHORT: Final[str] = "D15 and D18"
POOLED: Final[str] = "pooled 14"
"""The six signed-MW and eight MVA-only Primaries together."""
LEVELS32: Final[np.ndarray] = LEVELS.astype(np.float32)


# --------------------------------------------------------------------------- small helpers


def atomic_write_parquet(*, frame: pl.DataFrame, path: Path) -> None:
    """Write `frame` to `path` through a temporary file, so a crash leaves no partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.write_parquet(temporary)
    temporary.replace(path)


def atomic_write_text(*, text: str, path: Path) -> None:
    """Write `text` to `path` through a temporary file, so a crash leaves no partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def utc_series(*, micros: np.ndarray) -> pl.Series:
    """Return microseconds since the epoch as a UTC datetime series."""
    return pl.from_epoch(pl.Series(micros.astype(np.int64)), time_unit="us").dt.replace_time_zone(
        "UTC"
    )


def peak_memory_gb() -> float:
    """Return this process's peak resident memory in gigabytes (Linux reports kilobytes)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def stamp_to_utc(*, stamp: int) -> dt.datetime:
    """Return the valid time of grid stamp `stamp`."""
    return GRID_START + dt.timedelta(minutes=30 * int(stamp))


def origin_to_utc(*, origin: int) -> dt.datetime:
    """Return the 12:00 UTC forecast origin with index `origin`."""
    return stamp_to_utc(stamp=ORIGIN_STAMP + PER_DAY * int(origin))


def origin_months() -> np.ndarray:
    """Return each origin's UTC month label, such as `2025-12`."""
    return np.array([origin_to_utc(origin=o).strftime("%Y-%m") for o in range(N_ORIGINS)])


def origin_of_month(*, month: str) -> np.ndarray:
    """Return the indices of the origins whose UTC date is in `month`."""
    return np.flatnonzero(origin_months() == month)


def target_stamps() -> np.ndarray:
    """Return the stamp of every (origin, row) pair, shape (N_ORIGINS, N_H); it may exceed the grid."""
    return ORIGIN_STAMP + PER_DAY * np.arange(N_ORIGINS)[:, None] + np.arange(N_H)[None, :]


def stamp_to_target(*, per_stamp: np.ndarray) -> np.ndarray:
    """Map an array over stamps (last axis N_STAMPS) onto (…, N_ORIGINS, N_H), NaN or False beyond."""
    index = target_stamps()
    inside = index < N_STAMPS
    taken = per_stamp[..., np.minimum(index, N_STAMPS - 1)]
    fill: float | bool = False if per_stamp.dtype == bool else np.nan
    return np.where(inside, taken, fill)


# --------------------------------------------------------------------------- loading


@dataclass(frozen=True, slots=True, eq=False)
class Inputs:
    """The stored forecasts, cleaned power, and weather of the 16 Primaries on dense arrays."""

    ids: np.ndarray
    labels: tuple[str, ...]
    groups: tuple[str, ...]
    capacity: np.ndarray
    is_mva: np.ndarray
    truth: np.ndarray
    """Cleaned power in points of capacity, shape (K, N_STAMPS); NaN where flagged or absent."""
    q_nra: np.ndarray
    """Stored normal-running quantiles in points, shape (K, N_ORIGINS, N_H, 11)."""
    q_prv: np.ndarray | None
    """Stored prevailing quantiles in points, or None when not loaded."""
    weather: np.ndarray
    """Ensemble-mean drivers `x_pv` and `x_cold` of the run of each origin, (K, N_ORIGINS, N_H, 2)."""
    driver_var: np.ndarray
    """Variance of each driver per series over 2024-11 to 2025-10, shape (K, 2)."""
    null_counts: pl.DataFrame
    """Half-hours with no cleaned power per series and month (the `substation_zero` rows)."""


def read_site_map() -> pl.DataFrame:
    """Return the site map of the 16 Primaries: primary, label, time_series_id, capacity_mw, group."""
    return pl.read_csv(next(ML_EXPERIMENTS_DATA_DIR.glob(SITE_MAP_GLOB)))


def _positions(*, ids: np.ndarray, series: pl.Series) -> np.ndarray:
    """Return the position in `ids` of each entry of `series`."""
    mapping = {int(value): position for position, value in enumerate(ids)}
    return series.cast(pl.Int64).replace_strict(mapping, return_dtype=pl.Int64).to_numpy()


def dense_from_frame(*, frame: pl.DataFrame, ids: np.ndarray, capacity: np.ndarray) -> np.ndarray:
    """Lay a forecast frame (MW) out as quantiles in points, shape (K, N_ORIGINS, N_H, 11).

    Rows the frame lacks stay NaN.

    Args:
        frame: Columns `time_series_id`, `valid_time`, `power_fcst_init_time` and the 11 quantiles.
        ids: The series ids, in array order.
        capacity: The series' capacity in MW or MVA, in array order.

    Returns:
        The dense float32 array.

    Raises:
        ValueError: If an origin is not a 12:00 UTC origin of the grid, a row lies outside the 672
            half-hours of its origin, or the frame holds a key twice.
    """
    position = _positions(ids=ids, series=frame["time_series_id"])
    origin_us = frame["power_fcst_init_time"].dt.epoch("us").to_numpy()
    valid_us = frame["valid_time"].dt.epoch("us").to_numpy()
    day, day_remainder = np.divmod(origin_us - ORIGIN0_US, DAY_US)
    half_hour, half_remainder = np.divmod(valid_us - origin_us, STEP_US)
    if (day_remainder != 0).any() or (half_remainder != 0).any():
        msg = "an origin is not at 12:00 UTC or a valid time is off the half-hour"
        raise ValueError(msg)
    if (day < 0).any() or (day >= N_ORIGINS).any() or (half_hour < 0).any():
        msg = "a row lies outside the origins or the horizon of the grid"
        raise ValueError(msg)
    if (half_hour >= N_H).any():
        msg = "a row lies beyond the 672 half-hours of its origin"
        raise ValueError(msg)
    flat = np.ravel_multi_index((position, day, half_hour), (len(ids), N_ORIGINS, N_H))
    if len(np.unique(flat)) != len(flat):
        msg = "the frame holds a (series, origin, valid time) key twice"
        raise ValueError(msg)
    values = frame.select(list(QUANTILE_COLUMNS)).to_numpy().astype(np.float32)
    values *= (100.0 / capacity[position]).astype(np.float32)[:, None]
    dense = np.full((len(ids) * N_ORIGINS * N_H, len(QUANTILE_COLUMNS)), np.nan, np.float32)
    dense[flat] = values
    return dense.reshape(len(ids), N_ORIGINS, N_H, len(QUANTILE_COLUMNS))


def dense_to_frame(*, inputs: Inputs, quantiles: np.ndarray, origins: np.ndarray) -> pl.DataFrame:
    """Return the rows of a dense forecast, in MW, for the origins `origins`.

    Args:
        inputs: The loaded inputs, for the ids and capacities.
        quantiles: Quantiles in points, shape (K, N_ORIGINS, N_H, 11); NaN rows are left out.
        origins: Indices of the origins to write.

    Returns:
        A frame with the stored files' columns: `time_series_id`, `valid_time`, `nwp_init_time`,
        `power_fcst_init_time`, and the 11 quantiles as float32.
    """
    keep = np.isfinite(quantiles[..., MEDIAN_INDEX])
    wanted = np.zeros(N_ORIGINS, dtype=bool)
    wanted[origins] = True
    keep &= wanted[None, :, None]
    k, o, h = np.nonzero(keep)
    origin_us = ORIGIN0_US + o.astype(np.int64) * DAY_US
    values = quantiles[k, o, h, :] * (inputs.capacity[k] / 100.0).astype(np.float32)[:, None]
    return pl.DataFrame(
        {
            "time_series_id": pl.Series(inputs.ids[k], dtype=pl.Int8),
            "valid_time": utc_series(micros=origin_us + h.astype(np.int64) * STEP_US),
            "nwp_init_time": utc_series(micros=origin_us - 12 * 2 * STEP_US),
            "power_fcst_init_time": utc_series(micros=origin_us),
            **{
                name: pl.Series(values[:, index].astype(np.float32))
                for index, name in enumerate(QUANTILE_COLUMNS)
            },
        }
    )


def _read_forecast_kind(*, kind: str, ids: np.ndarray, capacity: np.ndarray) -> np.ndarray:
    """Return the stored `NRA` or `prevailing` rows before the holdout as dense quantiles."""
    frame = (
        pl.scan_parquet(BACKTEST_DIR / "primaries.parquet")
        .filter(pl.col("NRA_or_prevailing") == kind, pl.col("valid_time") < HOLDOUT_START)
        .select("time_series_id", "valid_time", "power_fcst_init_time", *QUANTILE_COLUMNS)
        .collect()
    )
    return dense_from_frame(frame=frame, ids=ids, capacity=capacity)


def _read_truth(*, ids: np.ndarray, capacity: np.ndarray) -> tuple[np.ndarray, pl.DataFrame]:
    """Return cleaned power in points on the stamp grid, and the null-power count per series and month."""
    frame = (
        pl.scan_parquet(BACKTEST_DIR / "ground_truth_primaries.parquet")
        .filter(pl.col("valid_time") >= GRID_START, pl.col("valid_time") < HOLDOUT_START)
        .select("time_series_id", "valid_time", "power", "drop_reason")
        .collect()
    )
    position = _positions(ids=ids, series=frame["time_series_id"])
    stamp, remainder = np.divmod(
        frame["valid_time"].dt.epoch("us").to_numpy() - GRID_START_US, STEP_US
    )
    if (remainder != 0).any() or (stamp < 0).any() or (stamp >= N_STAMPS).any():
        msg = "a ground-truth valid time is off the half-hour grid"
        raise ValueError(msg)
    power = frame["power"].cast(pl.Float64).to_numpy()
    truth = np.full((len(ids), N_STAMPS), np.nan, np.float64)
    truth[position, stamp] = power * (100.0 / capacity[position])
    counts = (
        frame.filter(pl.col("power").is_null())
        .with_columns(month=pl.col("valid_time").dt.strftime("%Y-%m"))
        .group_by("time_series_id", "month", "drop_reason")
        .agg(half_hours=pl.len())
        .sort("time_series_id", "month")
    )
    return truth, counts


def read_run(*, day: dt.date, ids: np.ndarray) -> pl.DataFrame:
    """Return the ensemble-summary rows of the run of 00:00 UTC on `day` for the given series."""
    path = E607_WORK / "ens_summary" / f"init={day:%Y-%m-%d}.parquet"
    return (
        pl.scan_parquet(path)
        .filter(pl.col("time_series_id").is_in([int(value) for value in ids]))
        .select(
            "time_series_id",
            "power_fcst_init_time",
            "valid_time",
            "mean_shortwave",
            "mean_temperature_2m",
        )
        .collect()
    )


def drivers_from(
    *, shortwave: np.ndarray, temperature: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return `x_pv` and `x_cold` from the ensemble-mean shortwave (W/m2) and 2 m temperature (deg C)."""
    x_pv = shortwave.astype(np.float64) / SHORTWAVE_SCALE
    x_cold = np.maximum(0.0, COLD_THRESHOLD_C - temperature.astype(np.float64)) / COLD_SCALE_K
    return x_pv, x_cold


def _check_celsius(*, temperature: np.ndarray, where: str) -> None:
    """Raise unless the ensemble-mean temperature looks like degrees Celsius."""
    finite = temperature[np.isfinite(temperature)]
    if len(finite) and not (float(finite.min()) > -40.0 and float(finite.max()) < 50.0):
        msg = (
            f"{where}: the 2 m temperature runs {finite.min():.1f} to {finite.max():.1f}, not deg C"
        )
        raise ValueError(msg)


def _fill_run_weather(*, weather: np.ndarray, ids: np.ndarray, origin: int) -> None:
    """Fill the drivers of the run of origin day `origin` into `weather`, in place.

    Raises:
        ValueError: If the run's `power_fcst_init_time` is not on the date of the run's file, or its
            temperature is not in degrees Celsius.
    """
    day = GRID_START.date() + dt.timedelta(days=origin)
    frame = read_run(day=day, ids=ids)
    if not (frame["power_fcst_init_time"].dt.date() == day).all():
        msg = f"run {day}: power_fcst_init_time is not on the run's date"
        raise ValueError(msg)
    position = _positions(ids=ids, series=frame["time_series_id"])
    origin_us = ORIGIN0_US + origin * DAY_US
    valid_us = frame["valid_time"].dt.epoch("us").to_numpy()
    half_hour, remainder = np.divmod(valid_us - origin_us, STEP_US)
    keep = (remainder == 0) & (half_hour >= 0) & (half_hour < N_H)
    temperature = frame["mean_temperature_2m"].to_numpy()
    _check_celsius(temperature=temperature, where=f"run {day}")
    x_pv, x_cold = drivers_from(
        shortwave=frame["mean_shortwave"].to_numpy(), temperature=temperature
    )
    weather[position[keep], origin, half_hour[keep], 0] = x_pv[keep]
    weather[position[keep], origin, half_hour[keep], 1] = x_cold[keep]
    _interpolate_gaps(weather=weather[:, origin])


def _interpolate_gaps(*, weather: np.ndarray) -> None:
    """Fill interior gaps of one run's drivers by linear interpolation along the rows, in place.

    The summary tables hold the ensemble's own steps where the weather model steps are coarser than
    a half-hour; a gap between two present rows is bridged, and rows outside the present span stay
    NaN. Shape (K, N_H, 2).
    """
    rows = np.arange(N_H)
    for series in range(weather.shape[0]):
        for driver in range(weather.shape[2]):
            values = weather[series, :, driver]
            present = np.isfinite(values)
            if present.sum() < 2 or present.all():
                continue
            first, last = np.flatnonzero(present)[[0, -1]]
            gaps = ~present & (rows > first) & (rows < last)
            values[gaps] = np.interp(rows[gaps], rows[present], values[present])


def _read_weather(*, ids: np.ndarray, n_jobs: int) -> np.ndarray:
    """Return the drivers of the run of each origin day, shape (K, N_ORIGINS, N_H, 2).

    Each run writes its own origin's slice, so the runs are read on `n_jobs` threads.
    """
    weather = np.full((len(ids), N_ORIGINS, N_H, 2), np.nan, np.float32)
    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        list(
            pool.map(
                lambda origin: _fill_run_weather(weather=weather, ids=ids, origin=origin),
                range(N_ORIGINS),
            )
        )
    return weather


def driver_variances(*, ids: np.ndarray) -> np.ndarray:
    """Return the variance of `x_pv` and `x_cold` per series over 2024-11 to 2025-10, shape (K, 2).

    A run's rows from 12 to under 36 hours after its 00:00 UTC start are the rows that a
    smallest-lead forecast uses. The table is cached in the experiment's folder.
    """
    path = OUT_DIR / "driver_variances.parquet"
    if path.exists():
        table = pl.read_parquet(path)
        return np.stack([table["var_pv"].to_numpy(), table["var_cold"].to_numpy()], axis=1)
    totals = np.zeros((len(ids), 2, 3))
    day = dt.date(2024, 11, 1)
    while day <= dt.date(2025, 10, 31):
        frame = read_run(day=day, ids=ids)
        run_us = (
            int(dt.datetime(day.year, day.month, day.day, tzinfo=dt.UTC).timestamp()) * 1_000_000
        )
        lead_us = frame["valid_time"].dt.epoch("us").to_numpy() - run_us
        keep = (lead_us >= 12 * 2 * STEP_US) & (lead_us < 36 * 2 * STEP_US)
        temperature = frame["mean_temperature_2m"].to_numpy()
        _check_celsius(temperature=temperature, where=f"spread run {day}")
        drivers = drivers_from(
            shortwave=frame["mean_shortwave"].to_numpy(), temperature=temperature
        )
        position = _positions(ids=ids, series=frame["time_series_id"])
        for column, values in enumerate(drivers):
            good = keep & np.isfinite(values)
            np.add.at(totals[:, column, 0], position[good], 1.0)
            np.add.at(totals[:, column, 1], position[good], values[good])
            np.add.at(totals[:, column, 2], position[good], values[good] ** 2)
        day += dt.timedelta(days=1)
    mean = totals[..., 1] / totals[..., 0]
    variance = totals[..., 2] / totals[..., 0] - mean**2
    atomic_write_parquet(
        frame=pl.DataFrame(
            {"time_series_id": ids, "var_pv": variance[:, 0], "var_cold": variance[:, 1]}
        ),
        path=path,
    )
    return variance


def load_inputs(*, load_prevailing: bool, n_jobs: int) -> Inputs:
    """Load the stored forecasts, cleaned power, and weather of the 16 Primaries.

    The holdout is dropped here, before any table is built: only valid times before 2026-07-01 are
    read, and the three months of origins that follow are never opened.

    Args:
        load_prevailing: Whether to load the stored prevailing rows as well (C5 and its arms).
        n_jobs: Threads that read the ensemble runs.

    Returns:
        The inputs, in the site map's order of Primaries.
    """
    site = read_site_map()
    ids = site["time_series_id"].to_numpy().astype(np.int64)
    capacity = site["capacity_mw"].to_numpy().astype(np.float64)
    stored = pl.read_parquet(BACKTEST_DIR / "series_capacity.parquet").filter(
        pl.col("time_series_id").is_in(ids.tolist())
    )
    stored_capacity = dict(
        zip(stored["time_series_id"].to_list(), stored["capacity"].to_list(), strict=True)
    )
    for series, mw in zip(ids.tolist(), capacity.tolist(), strict=True):
        if abs(stored_capacity[series] - mw) > 1e-3 * mw:
            msg = f"series {series}: site map capacity {mw} differs from series_capacity.parquet"
            raise ValueError(msg)
    units = dict(
        zip(stored["time_series_id"].to_list(), stored["power_unit"].to_list(), strict=True)
    )
    truth, counts = _read_truth(ids=ids, capacity=capacity)
    return Inputs(
        ids=ids,
        labels=tuple(site["label"].to_list()),
        groups=tuple(site["group"].to_list()),
        capacity=capacity,
        is_mva=np.array([units[int(series)] == "MVA" for series in ids]),
        truth=truth,
        q_nra=_read_forecast_kind(kind="NRA", ids=ids, capacity=capacity),
        q_prv=(
            _read_forecast_kind(kind="prevailing", ids=ids, capacity=capacity)
            if load_prevailing
            else None
        ),
        weather=_read_weather(ids=ids, n_jobs=n_jobs),
        driver_var=driver_variances(ids=ids),
        null_counts=counts,
    )


def era5_grid(*, ids: np.ndarray) -> np.ndarray:
    """Return the realised ERA5 shortwave (W/m2) and 2 m temperature (deg C) at every stamp.

    The values are analysis weather and enter only the known-answer injections and the slices of the
    score tables, never an estimator. Shape (K, N_STAMPS, 2).
    """
    frame = (
        pl.scan_parquet(E607_WORK / "era5_features.parquet")
        .filter(
            pl.col("time_series_id").is_in([int(value) for value in ids]),
            pl.col("valid_time") >= GRID_START - dt.timedelta(hours=3),
            pl.col("valid_time") < HOLDOUT_START + dt.timedelta(hours=3),
        )
        .select("time_series_id", "valid_time", "shortwave", "temperature_2m")
        .sort("time_series_id", "valid_time")
        .collect()
    )
    stamps = GRID_START_US + STEP_US * np.arange(N_STAMPS)
    out = np.full((len(ids), N_STAMPS, 2), np.nan, np.float32)
    for position, series in enumerate(ids):
        part = frame.filter(pl.col("time_series_id") == int(series))
        times = part["valid_time"].dt.epoch("us").to_numpy()
        for column, name in enumerate(("shortwave", "temperature_2m")):
            values = part[name].cast(pl.Float64).to_numpy()
            good = np.isfinite(values)
            out[position, :, column] = np.interp(stamps, times[good], values[good])
    _check_celsius(temperature=out[..., 1], where="ERA5")
    return out


# --------------------------------------------------------------------------- residuals


class Residuals(NamedTuple):
    """Look-back residuals in points of capacity, on the stamp grid."""

    banded: np.ndarray
    """Per lead band, the mean over the forecasts in the band of power minus the stored median,
    shape (K, 6, N_STAMPS)."""
    smallest: np.ndarray
    """Power minus the median of the forecast with the smallest lead (the 12:00 origin at most 24 h
    before the half-hour), shape (K, N_STAMPS)."""


def build_residuals(*, inputs: Inputs) -> Residuals:
    """Return the residuals of every stamp against the stored normal-running median.

    A stamp `i` has a forecast from origin `o = (i - 24) // 48 - j` at row `h = (i - 24) % 48 + 48 j`
    for each `j` from 0 to 13. The smallest-lead forecast is `j = 0`; the band residual averages the
    forecasts whose lead lies in the band. Every such forecast was issued at or before the stamp.
    """
    median = inputs.q_nra[..., MEDIAN_INDEX]
    stamps = np.arange(N_STAMPS)
    offset = stamps - ORIGIN_STAMP
    first_origin = offset // PER_DAY
    first_row = offset % PER_DAY
    n_series = len(inputs.ids)
    totals = np.zeros((n_series, N_BANDS, N_STAMPS))
    counts = np.zeros((n_series, N_BANDS, N_STAMPS))
    smallest = np.full((n_series, N_STAMPS), np.nan)
    for lead_day in range(LOOK_BACK_LEADS):
        origin = first_origin - lead_day
        row = first_row + PER_DAY * lead_day
        usable = (origin >= 0) & (origin < N_ORIGINS) & (row < N_H)
        forecast = np.full((n_series, N_STAMPS), np.nan)
        forecast[:, usable] = median[:, origin[usable], row[usable]]
        residual = inputs.truth - forecast
        if lead_day == 0:
            smallest = residual.copy()
        band = BAND_OF_H[np.minimum(row, N_H - 1)]
        finite = np.isfinite(residual)
        for index in range(N_BANDS):
            chosen = (band == index)[None, :] & finite
            totals[:, index, :] += np.where(chosen, residual, 0.0)
            counts[:, index, :] += chosen
    banded = np.where(counts > 0, totals / np.maximum(counts, 1), np.nan)
    return Residuals(banded=banded, smallest=smallest)


# --------------------------------------------------------------------------- designs and the window fit


def calendar_columns() -> np.ndarray:
    """Return the weekend flag and three time-of-day block flags of every stamp, shape (N_STAMPS, 4).

    Times are Europe/London local times of the stamp. The blocks are hours 0 to 7, 8 to 15, and 16
    to 23.
    """
    stamps = pl.datetime_range(
        GRID_START,
        GRID_START + dt.timedelta(minutes=30 * (N_STAMPS - 1)),
        interval="30m",
        time_zone="UTC",
        eager=True,
    ).dt.convert_time_zone("Europe/London")
    weekend = (stamps.dt.weekday() >= 6).to_numpy().astype(np.float64)
    block = (stamps.dt.hour() // 8).to_numpy()
    return np.stack([weekend, *[(block == index).astype(np.float64) for index in range(3)]], axis=1)


@dataclass(frozen=True, slots=True)
class ArmSpec:
    """One look-back offset arm."""

    name: str
    drivers: tuple[int, ...] = ()
    """Driver slopes in the ridge regression: 0 for `x_pv`, 1 for `x_cold`; empty for G."""
    calendar: bool = False
    """Whether the regression adds the weekend flag and three time-of-day blocks (D+)."""
    window_days: int = 14
    lag_half_hours: int = LAG_6H
    factor: float = FACTOR_LOOK_BACK
    band_matched: bool = True
    """One residual per lead band if True, else the smallest-lead residual for every row."""
    truncate_to_month: bool = False
    """Use only residuals of forecasts issued in the origin's own month (the refit month)."""
    permute_days: bool = False
    """Permute the look-back drivers in whole days against the residuals (placebo PL2)."""
    first_origin: int = FIRST_ORIGIN_DEC
    row_set_id: str = ROW_SET_ID


class WindowFit(NamedTuple):
    """Coefficients of the window regressions of one series."""

    coef: np.ndarray
    """Intercept and slopes per origin and residual band, shape (N_ORIGINS, B, 1 + p)."""
    se: np.ndarray
    """Standard error of each coefficient, same shape; NaN where the coefficient is zero by rule."""
    covered: np.ndarray
    """Whether the window had enough half-hours, shape (N_ORIGINS, B)."""


def _cumulative(*, values: np.ndarray) -> np.ndarray:
    """Return the cumulative sum along the last axis with a leading zero."""
    out = np.zeros((*values.shape[:-1], values.shape[-1] + 1))
    np.cumsum(values, axis=-1, out=out[..., 1:])
    return out


def fit_window(
    *,
    residual: np.ndarray,
    design: np.ndarray,
    penalty: np.ndarray,
    window_half_hours: int,
    lag_half_hours: int,
    truncate_stamps: np.ndarray | None = None,
) -> WindowFit:
    """Fit the ridge regression of the residual on the design in every origin's look-back window.

    The window of origin `t` holds the stamps from `t - lag - window + 1` to `t - lag`; stamps
    before the grid start are absent. The intercept is not penalised. A slope is set to zero when
    fewer than `MIN_ACTIVE` window half-hours have its driver above `ACTIVE_THRESHOLD`. With no
    slope the intercept is the window mean, so the arms nest.

    Args:
        residual: Residuals per band, shape (B, N_STAMPS), NaN where absent.
        design: Regressors without the intercept, shape (N_STAMPS, p); NaN rows are not used.
        penalty: Ridge penalty of each of the p slopes.
        window_half_hours: The window's length.
        lag_half_hours: Half-hours between the origin and the newest residual.
        truncate_stamps: If given, the first stamp the window may hold at each origin.

    Returns:
        The coefficients, their standard errors, and whether each window was covered.
    """
    n_bands = residual.shape[0]
    n_slopes = design.shape[1]
    full = np.concatenate([np.ones((N_STAMPS, 1)), design], axis=1)
    usable = np.isfinite(full).all(axis=1)
    full = np.where(usable[:, None], full, 0.0)
    weight = (np.isfinite(residual) & usable[None, :]).astype(np.float64)
    values = np.where(weight > 0, residual, 0.0)
    count = _cumulative(values=weight)
    squares = _cumulative(values=weight * values**2)
    gram = np.empty((n_bands, n_slopes + 1, n_slopes + 1, N_STAMPS + 1))
    for i in range(n_slopes + 1):
        for j in range(i, n_slopes + 1):
            gram[:, i, j] = gram[:, j, i] = _cumulative(values=weight * full[:, i] * full[:, j])
    cross = np.stack(
        [_cumulative(values=weight * full[:, i] * values) for i in range(n_slopes + 1)], axis=1
    )
    active = (
        np.stack(
            [
                _cumulative(values=weight * (full[:, 1 + i] > ACTIVE_THRESHOLD))
                for i in range(n_slopes)
            ],
            axis=1,
        )
        if n_slopes
        else np.zeros((n_bands, 0, N_STAMPS + 1))
    )

    last = ORIGIN_STAMP + PER_DAY * np.arange(N_ORIGINS) - lag_half_hours
    first = np.maximum(last - window_half_hours + 1, 0)
    if truncate_stamps is not None:
        first = np.maximum(first, truncate_stamps)
    stop = last + 1
    span = stop - first
    n_window = count[:, stop] - count[:, first]
    covered = (n_window >= MIN_COVERAGE * span[None, :]) & (span[None, :] >= PER_DAY)
    normal = np.moveaxis(gram[..., stop] - gram[..., first], -1, 1)
    rhs = np.moveaxis(cross[..., stop] - cross[..., first], -1, 1)
    n_active = np.moveaxis(active[..., stop] - active[..., first], -1, 1)
    sum_squares = squares[:, stop] - squares[:, first]

    keep = np.ones((n_bands, N_ORIGINS, n_slopes + 1), dtype=bool)
    keep[..., 1:] = n_active >= MIN_ACTIVE
    keep &= covered[..., None]
    ridge = np.zeros(n_slopes + 1)
    ridge[1:] = penalty
    eye = np.eye(n_slopes + 1)
    outer = keep[..., :, None] & keep[..., None, :]
    raw = np.where(outer, normal, 0.0)
    system = (
        raw
        + np.einsum("bop,pq->bopq", keep * ridge, eye)
        + np.einsum("bop,pq->bopq", (~keep).astype(np.float64), eye)
    )
    rhs = rhs * keep
    coef = np.linalg.solve(system, rhs[..., None])[..., 0]
    fitted = np.einsum("bop,bopq,boq->bo", coef, raw, coef)
    sse = np.maximum(sum_squares - 2.0 * (coef * rhs).sum(axis=-1) + fitted, 0.0)
    dof = np.maximum(n_window - keep.sum(axis=-1), 1.0)
    inverse_diagonal = np.einsum("...ii->...i", np.linalg.inv(system))
    se = np.sqrt((sse / dof)[..., None] * inverse_diagonal)
    se = np.where(keep, se, np.nan)
    return WindowFit(coef=np.swapaxes(coef, 0, 1), se=np.swapaxes(se, 0, 1), covered=covered.T)


def lookback_design(*, inputs: Inputs, series: int, spec: ArmSpec) -> np.ndarray:
    """Return the look-back regressors of one series on the stamp grid, shape (N_STAMPS, p).

    The driver of stamp `i` is the ensemble-mean weather of the 00:00 UTC run of the origin whose
    forecast has the smallest lead for `i`, so it is weather the origin of that forecast already had.
    """
    stamps = np.arange(N_STAMPS)
    offset = stamps - ORIGIN_STAMP
    origin = offset // PER_DAY
    row = offset % PER_DAY
    usable = (origin >= 0) & (origin < N_ORIGINS)
    columns: list[np.ndarray] = []
    for driver in spec.drivers:
        values = np.full(N_STAMPS, np.nan)
        values[usable] = inputs.weather[series, origin[usable], row[usable], driver]
        if spec.permute_days:
            permutation = np.random.default_rng(EXPERIMENT_ID).permutation(N_ORIGINS)
            shuffled = np.full(N_STAMPS, np.nan)
            shuffled[usable] = values[
                ORIGIN_STAMP + PER_DAY * permutation[origin[usable]] + row[usable]
            ]
            values = shuffled
        columns.append(values)
    if spec.calendar:
        columns.extend(calendar_columns().T)
    return np.stack(columns, axis=1) if columns else np.zeros((N_STAMPS, 0))


def target_design(*, inputs: Inputs, series: int, spec: ArmSpec) -> np.ndarray:
    """Return the regressors of the target rows of one series, shape (N_ORIGINS, N_H, p).

    Weather comes from the run of the row's own origin (available at 09:00 UTC); a missing value is
    NaN here and contributes nothing to the correction.
    """
    columns: list[np.ndarray] = [inputs.weather[series, :, :, driver] for driver in spec.drivers]
    if spec.calendar:
        index = np.minimum(target_stamps(), N_STAMPS - 1)
        columns.extend(calendar_columns().T[:, index])
    if not columns:
        return np.zeros((N_ORIGINS, N_H, 0))
    return np.stack(columns, axis=-1)


def penalties(*, inputs: Inputs, series: int, spec: ArmSpec) -> np.ndarray:
    """Return the ridge penalty of each slope: 48 half-hours of data at the regressor's variance."""
    values = [inputs.driver_var[series, driver] for driver in spec.drivers]
    if spec.calendar:
        share = calendar_columns().mean(axis=0)
        values.extend(share * (1.0 - share))
    return PENALTY_HALF_HOURS * np.array(values, dtype=np.float64)


def month_start_stamps() -> np.ndarray:
    """Return, per origin, the first stamp whose smallest-lead forecast is from the origin's month."""
    months = origin_months()
    first_of_month = {month: int(np.flatnonzero(months == month)[0]) for month in set(months)}
    return np.array(
        [ORIGIN_STAMP + PER_DAY * first_of_month[month] for month in months], dtype=np.int64
    )


def series_corrections(
    *, inputs: Inputs, residuals: Residuals, series: int, spec: ArmSpec
) -> tuple[np.ndarray, WindowFit]:
    """Return the correction added to every quantile of one series, shape (N_ORIGINS, N_H), and the fit.

    The correction at target row `(o, h)` is the factor times `a + b . x`, where `a` and `b` come
    from the window ending `lag` half-hours before origin `o` and `x` is the row's own drivers.
    """
    residual = (
        residuals.banded[series] if spec.band_matched else residuals.smallest[series][None, :]
    )
    fit = fit_window(
        residual=residual,
        design=lookback_design(inputs=inputs, series=series, spec=spec),
        penalty=penalties(inputs=inputs, series=series, spec=spec),
        window_half_hours=spec.window_days * PER_DAY,
        lag_half_hours=spec.lag_half_hours,
        truncate_stamps=month_start_stamps() if spec.truncate_to_month else None,
    )
    band = BAND_OF_H if spec.band_matched else np.zeros(N_H, dtype=np.int64)
    coef = fit.coef[:, band, :]
    slopes = np.nan_to_num(target_design(inputs=inputs, series=series, spec=spec))
    correction = coef[..., 0] + np.einsum("ohp,ohp->oh", coef[..., 1:], slopes)
    return spec.factor * correction, fit


def apply_correction(
    *, inputs: Inputs, quantiles: np.ndarray, correction: np.ndarray
) -> np.ndarray:
    """Add `correction` (K, N_ORIGINS, N_H) to every quantile, keeping apparent power non-negative."""
    shifted = quantiles + correction[..., None].astype(np.float32)
    return np.where(inputs.is_mva[:, None, None, None], np.maximum(shifted, 0.0), shifted)


def changepoint_levels(
    *,
    residual: np.ndarray,
    lag_half_hours: int,
    window_days: int = 28,
    min_segment_days: int = 3,
    improvement: float = 0.5,
    fallback_days: int = 7,
) -> np.ndarray:
    """Return the prevailing level per origin from the best single change point (arm P-cp).

    The window's daily mean residuals are split at the single change point that minimises the sum of
    squares of two constant segments of at least `min_segment_days` days. When the split cuts the
    sum of squares to `improvement` of the unsplit value, the level is the mean of the segment after
    the change (the slow part before it plus the whole step), else the mean of the last
    `fallback_days` days. A window with under half its days observed gives level 0.

    Args:
        residual: One series' smallest-lead residual on the stamp grid.
        lag_half_hours: Half-hours between the origin and the newest residual.
        window_days: The window in days.
        min_segment_days: The shortest segment on either side of the change point.
        improvement: The accepted ratio of split to unsplit sum of squares.
        fallback_days: Days averaged when no change point is accepted.

    Returns:
        The level per origin, in points, shape (N_ORIGINS,).
    """
    levels = np.zeros(N_ORIGINS)
    length = window_days * PER_DAY
    for origin in range(N_ORIGINS):
        last = ORIGIN_STAMP + PER_DAY * origin - lag_half_hours
        first = last - length + 1
        if first < 0:
            continue
        block = residual[first : last + 1].reshape(window_days, PER_DAY)
        seen = np.isfinite(block).sum(axis=1)
        daily = np.where(
            seen >= PER_DAY // 2, np.nansum(block, axis=1) / np.maximum(seen, 1), np.nan
        )
        days = daily[np.isfinite(daily)]
        if len(days) < window_days // 2:
            continue
        levels[origin] = _segment_level(
            days=days,
            min_segment_days=min_segment_days,
            improvement=improvement,
            fallback_days=fallback_days,
        )
    return levels


def _segment_level(
    *, days: np.ndarray, min_segment_days: int, improvement: float, fallback_days: int
) -> float:
    """Return the level of the last segment of `days`, or the mean of the last days if no change."""
    single = float(((days - days.mean()) ** 2).sum())
    best_sse, best_split = np.inf, -1
    for split in range(min_segment_days, len(days) - min_segment_days + 1):
        before, after = days[:split], days[split:]
        sse = float(((before - before.mean()) ** 2).sum() + ((after - after.mean()) ** 2).sum())
        if sse < best_sse:
            best_sse, best_split = sse, split
    if best_split > 0 and best_sse <= improvement * single:
        return float(days[best_split:].mean())
    return float(days[-fallback_days:].mean())


# --------------------------------------------------------------------------- losses and masks


def target_truth(*, inputs: Inputs) -> np.ndarray:
    """Return the cleaned power at every target row, shape (K, N_ORIGINS, N_H); NaN beyond the grid."""
    return stamp_to_target(per_stamp=inputs.truth.astype(np.float32))


def loss_arrays(*, truth: np.ndarray, quantiles: np.ndarray) -> dict[str, np.ndarray]:
    """Return the per-row losses in points: CRPS, median error, and the tail pinball and coverage.

    CRPS is twice the pinball loss averaged over the 11 levels. Coverage is 1 where the power is at
    or below the quantile, else 0. Every array is (K, N_ORIGINS, N_H) float32, NaN where power or
    the forecast is missing.
    """
    names = ["crps", "mae"]
    names += [f"pin_{name}" for name in TAIL_COLUMNS] + [f"cov_{name}" for name in TAIL_COLUMNS]
    out = {name: np.full(truth.shape, np.nan, np.float32) for name in names}
    for series in range(truth.shape[0]):
        error = truth[series][..., None] - quantiles[series]
        pinball = np.maximum(LEVELS32 * error, (LEVELS32 - 1.0) * error)
        out["crps"][series] = 2.0 * pinball.mean(axis=-1)
        out["mae"][series] = np.abs(error[..., MEDIAN_INDEX])
        for name in TAIL_COLUMNS:
            position = QUANTILE_COLUMNS.index(name)
            out[f"pin_{name}"][series] = pinball[..., position]
            below = np.where(np.isnan(error[..., position]), np.nan, error[..., position] <= 0.0)
            out[f"cov_{name}"][series] = below
    return out


def group_masks(*, inputs: Inputs) -> dict[str, np.ndarray]:
    """Return a mask over series for each metering group and for the 14 series of the verdict."""
    labels = np.array(inputs.groups)
    masks = {group: labels == group for group in (GROUP_SIGNED, GROUP_MVA, GROUP_SHORT)}
    masks[POOLED] = masks[GROUP_SIGNED] | masks[GROUP_MVA]
    return masks


def lead_mask() -> np.ndarray:
    """Return the rows with a lead above 0 and up to 240 hours, shape (N_H,)."""
    rows = np.arange(N_H)
    return (rows >= 1) & (rows <= MAX_LEAD_HALF_HOURS)


def event_masks(*, spans: pl.DataFrame, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per series and stamp whether it lies inside an event, and in the 7 days after one.

    Args:
        spans: Event spans with `time_series_id`, `start`, and `end` in half-hours on the library
            grid (the output of `event_spans` of experiment 688).
        ids: The series ids, in array order.

    Returns:
        Two boolean arrays of shape (K, N_STAMPS); the second excludes stamps in an event.
    """
    position = {int(value): index for index, value in enumerate(ids)}
    inside = np.zeros((len(ids), N_STAMPS), dtype=bool)
    after = np.zeros((len(ids), N_STAMPS), dtype=bool)
    for row in spans.iter_rows(named=True):
        index = position.get(int(row["time_series_id"]))
        if index is None:
            continue
        start, end = int(row["start"]) - LIB_OFFSET, int(row["end"]) - LIB_OFFSET
        inside[index, max(start, 0) : max(min(end, N_STAMPS), 0)] = True
        after[index, max(end, 0) : max(min(end + 7 * PER_DAY, N_STAMPS), 0)] = True
    return inside, after & ~inside


def window_overlap(*, flags: np.ndarray, window_half_hours: int, lag_half_hours: int) -> np.ndarray:
    """Return, per series and origin, whether any stamp of the look-back window has `flags` set."""
    cumulative = _cumulative(values=flags.astype(np.float64))
    last = ORIGIN_STAMP + PER_DAY * np.arange(N_ORIGINS) - lag_half_hours
    first = np.maximum(last - window_half_hours + 1, 0)
    return (cumulative[:, last + 1] - cumulative[:, first]) > 0


def load_event_masks(*, inputs: Inputs, kind: str) -> tuple[np.ndarray, np.ndarray, pl.DataFrame]:
    """Return the event masks of experiment 688 on the stamp grid, and the spans they come from.

    Needs the folders of experiments 668, 683, and 688 on `PYTHONPATH`, because the spans come from
    `event_spans` in `e00688_diagnose.py`.

    Args:
        inputs: The loaded inputs, for the series ids.
        kind: `primary` for the labelled events and the verified transfers, or `wide` to add the
            fitted spans of experiment 580 and the candidate events of experiment 685.

    Returns:
        The inside-an-event mask, the 7-days-after mask (both (K, N_STAMPS)), and the spans.
    """
    import e00688_diagnose as diagnose  # ty: ignore[unresolved-import]

    spans = diagnose.event_spans(kind=kind)
    inside, after = event_masks(spans=spans, ids=inputs.ids)
    return inside, after, spans
