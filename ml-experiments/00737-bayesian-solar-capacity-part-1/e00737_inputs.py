"""Inputs of experiment 737: the power, CAMS irradiance, ERA5 weather, and the baseline design.

Every loader cuts at `CUT` (2026-07-01 00:00 UTC, the end of the last half-hour of the grid), and
`assert_cut` checks the cut. The half-hour grid is experiment 653's: index `i` is the half-hour that
starts at `GRID_START + 30 min x i` and ends at `GRID_START + 30 min x (i + 1)`. Every array is on
that grid and every timestamp here is a half-hour end time, which is what
`half_hour_sun_and_sky` and `studies.pv_separation.baseline_design` take.

The fit reads: the cleaned signed power (experiment 327's store), CAMS irradiance at the Primary's
own coordinates for hours whose reliability is 0.9 or more, ERA5 2 m temperature and 10 m wind
speed, the holiday flag, the sun's position, and the calendar. It reads no forecast, no lead time,
and no past power. The stored forecast of experiment 327 is loaded only for arm R.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from functools import lru_cache
from typing import Final

import e00039_common as c39  # ty: ignore[unresolved-import]
import e00653_common as c653  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from e00737_common import (
    CUT,
    DAYLIGHT_GHI_W_M2,
    E327,
    FARM_IDS,
    FARMS_PATH,
    FEATURES_PATH,
    GRID_LENGTH,
    GRID_START,
    HALF_HOURS_PER_DAY,
    HDD_BASE_C,
    MIN_QUARTER_SHARE,
    MIN_RELIABILITY,
    N_QUARTERS,
    PRIMARIES,
    WEATHER_WARMUP_DAYS,
    grid_end_times,
    quarter_index,
)
from e00737_forward import FleetParams, fleet_unit_output, interpolate_hourly, trailing_mean
from studies.pv_dataset import pv_sites
from studies.pv_physics import SunAndSky, daylight, half_hour_sun_and_sky
from studies.pv_separation import RIDGE_PER_COLUMN, baseline_design

FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "valid_time",
    "time_series_id",
    "temperature_2m",
    "wind_speed_10m",
    "is_holiday",
)
"""The only columns read from experiment 6's `features.parquet`: no forecast or rolling mean."""
SEASONAL_COLUMNS: Final[int] = 336
"""Columns of `baseline_design(..., flexibility="seasonal")`: 144 day-type and 192 harmonic."""
N_TEMPERATURE_COLUMNS: Final[int] = 4
N_HOLIDAY_COLUMNS: Final[int] = 48
N_BASE_COLUMNS: Final[int] = SEASONAL_COLUMNS + N_TEMPERATURE_COLUMNS + N_HOLIDAY_COLUMNS + 17
"""The baseline's 405 columns: 336 seasonal, 4 temperature, 48 holiday, and 17 quarterly levels."""
TRAILING_24H: Final[int] = 48
TRAILING_72H: Final[int] = 144
PLACEBO_SHIFT_DAYS: Final[int] = 365


def assert_cut(*, times: np.ndarray, what: str) -> None:
    """Raise unless no timestamp is after the cut.

    Args:
        times: The timestamps, `datetime64`.
        what: A name for the message.

    Raises:
        ValueError: When a timestamp lies after `CUT`.
    """
    limit = np.datetime64(CUT.replace(tzinfo=None), "us")
    if len(times) and times.max() > limit:
        msg = f"{what} holds a timestamp after the cut {CUT.isoformat()}"
        raise ValueError(msg)


def primary_table() -> pl.DataFrame:
    """Return the series table rows of the six Primaries, in ascending D-number order.

    Returns:
        Columns `time_series_id`, `site`, `latitude`, `longitude`, `effective_capacity_mw`.
    """
    table = pl.read_parquet(c653.SERIES_PATH).filter(pl.col("site").is_in(PRIMARIES))
    order = {s: k for k, s in enumerate(PRIMARIES)}
    return table.sort(pl.col("site").replace_strict(order, return_dtype=pl.Int64)).select(
        "time_series_id", "site", "latitude", "longitude", "effective_capacity_mw"
    )


def hour_labels() -> pl.Series:
    """Return the CAMS hour labels (hour ends) that cover the grid: one per two half-hours.

    Returns:
        A UTC datetime series from 01:00 on the first day to `CUT`.
    """
    return pl.datetime_range(
        GRID_START + timedelta(hours=1), CUT, interval="1h", time_zone="UTC", eager=True
    )


def load_cams(*, site: str) -> pl.DataFrame:
    """Return the CAMS hourly rows of one Primary's point, cut at `CUT`.

    Args:
        site: The Primary's label, such as `D5`.

    Returns:
        Columns `time` (hour end, UTC), `ghi_w_m2`, `clear_sky_ghi_w_m2`, and `reliability`.
    """
    cams = (
        pl.scan_parquet(c39.CAMS_DEMAND_POINTS_PATH)
        .filter((pl.col("site") == site) & (pl.col("time") <= pl.lit(CUT)))
        .select("time", "ghi_w_m2", "clear_sky_ghi_w_m2", "reliability")
        .sort("time")
        .collect()
    )
    assert_cut(times=cams["time"].dt.replace_time_zone(None).to_numpy(), what="CAMS")
    return cams


def hourly_irradiance(*, cams: pl.DataFrame, shift_days: int = 0) -> pl.DataFrame:
    """Return one row per hour of the grid, with the irradiance of that hour (or a shifted one).

    An hour with no reliable CAMS value gets zero irradiance and `reliable` false, so that the
    two half-hours it holds are left out of the fit and are never taken as dark.

    Args:
        cams: `load_cams`'s output.
        shift_days: Take the irradiance of the hour this many days away (the placebo sky). A
            hour whose shifted label lies after `CUT` takes the value from the same number of days
            earlier.

    Returns:
        Columns `time`, `ghi_w_m2`, `clear_sky_ghi_w_m2`, and `reliable`, in `hour_labels` order.
    """
    labels = hour_labels()
    frame = pl.DataFrame({"time": labels})
    if shift_days:
        shifted = frame["time"].dt.offset_by(f"{shift_days}d")
        flipped = frame["time"].dt.offset_by(f"{-shift_days}d")
        frame = frame.with_columns(source=pl.when(shifted <= CUT).then(shifted).otherwise(flipped))
    else:
        frame = frame.with_columns(source=pl.col("time"))
    joined = frame.join(
        cams.rename({"time": "source"}), on="source", how="left", maintain_order="left"
    )
    reliable = (pl.col("reliability") >= MIN_RELIABILITY).fill_null(value=False)
    return joined.select(
        "time",
        ghi_w_m2=pl.when(reliable).then(pl.col("ghi_w_m2")).otherwise(0.0).cast(pl.Float64),
        clear_sky_ghi_w_m2=pl.col("clear_sky_ghi_w_m2").fill_null(0.0).cast(pl.Float64),
        reliable=reliable,
    )


def build_sky(*, hourly: pl.DataFrame, latitude: float, longitude: float) -> SunAndSky:
    """Return the sun and sky of every half-hour of the grid.

    Args:
        hourly: `hourly_irradiance`'s output.
        latitude: Degrees north.
        longitude: Degrees east.

    Returns:
        The sun and sky, `GRID_LENGTH` half-hours long.

    Raises:
        ValueError: When the half-hour end times differ from `grid_end_times`.
    """
    sky = half_hour_sun_and_sky(
        hourly=hourly.select("time", "ghi_w_m2"), latitude=latitude, longitude=longitude
    )
    if len(sky.ghi_w_m2) != GRID_LENGTH or not np.array_equal(
        sky.half_hour_end_time, grid_end_times()
    ):
        msg = "the half-hour end times of the sky differ from the grid's"
        raise ValueError(msg)
    return sky


@dataclass(frozen=True)
class Weather:
    """ERA5 on the half-hour grid, and the calendar flags that experiment 6 stores with it."""

    air_c: np.ndarray
    wind_ms: np.ndarray
    temperature_columns: np.ndarray
    """Heating and cooling degrees about 15.5 C and the 24-hour and 72-hour trailing means,
    each centred and scaled to unit spread, `(GRID_LENGTH, 4)`."""
    holiday: np.ndarray
    """True where the half-hour's start falls on a bank holiday."""


def load_weather(*, series_id: int) -> Weather:
    """Read ERA5 for one Primary and put it on the half-hour grid.

    The hourly values are interpolated linearly to the half-hour end times: a half-hour that ends on
    the hour takes that hour's value, and one that ends at the half-hour mark takes the mean of its
    two neighbouring hours. The trailing means are built here from `temperature_2m` over the
    half-hours that end at or before each half-hour's end. Rows after `CUT` are never read.

    Args:
        series_id: The Primary's `time_series_id`.

    Returns:
        The weather on the grid.

    Raises:
        ValueError: When a weather value is missing inside the range that is read.
    """
    start = GRID_START - timedelta(days=WEATHER_WARMUP_DAYS)
    frame = (
        pl.scan_parquet(FEATURES_PATH)
        .filter(
            (pl.col("time_series_id") == series_id)
            & (pl.col("valid_time") <= pl.lit(CUT))
            & (pl.col("valid_time") >= pl.lit(start))
        )
        .select(*FEATURE_COLUMNS)
        .sort("valid_time")
        .collect()
    )
    assert_cut(times=frame["valid_time"].dt.replace_time_zone(None).to_numpy(), what="ERA5")
    hourly = frame.filter(pl.col("valid_time").dt.minute() == 0)
    times = hourly["valid_time"].dt.replace_time_zone(None).to_numpy()
    n_warm = WEATHER_WARMUP_DAYS * HALF_HOURS_PER_DAY
    extended = np.concatenate(
        [
            grid_end_times()[0] - (np.arange(n_warm, 0, -1) * np.timedelta64(30, "m")),
            grid_end_times(),
        ]
    )
    temperature = interpolate_hourly(
        hourly_times=times,
        values=hourly["temperature_2m"].to_numpy().astype(float),
        at_times=extended,
    )
    wind = interpolate_hourly(
        hourly_times=times,
        values=hourly["wind_speed_10m"].to_numpy().astype(float),
        at_times=extended,
    )
    if not (np.isfinite(temperature).all() and np.isfinite(wind).all()):
        msg = "ERA5 has a missing value in the range that is read"
        raise ValueError(msg)
    columns = np.column_stack(
        [
            np.maximum(HDD_BASE_C - temperature, 0.0),
            np.maximum(temperature - HDD_BASE_C, 0.0),
            trailing_mean(values=temperature, window=TRAILING_24H),
            trailing_mean(values=temperature, window=TRAILING_72H),
        ]
    )[n_warm:]
    scaled = (columns - columns.mean(axis=0)) / columns.std(axis=0)
    starts = pl.DataFrame(
        {
            "valid_time": pl.datetime_range(
                GRID_START, CUT - timedelta(minutes=30), interval="30m", time_zone="UTC", eager=True
            )
        }
    )
    flags = starts.join(
        frame.select("valid_time", "is_holiday"), on="valid_time", how="left", maintain_order="left"
    )["is_holiday"].fill_null(value=False)
    return Weather(
        air_c=temperature[n_warm:],
        wind_ms=wind[n_warm:],
        temperature_columns=scaled,
        holiday=flags.to_numpy().astype(bool),
    )


def local_slot(*, end_time: np.ndarray) -> np.ndarray:
    """Return the local half-hour of the day (0 to 47) of each half-hour's start.

    Args:
        end_time: Half-hour end stamps in UTC.

    Returns:
        Integers 0 to 47 on the clock of Europe/London, as `baseline_design` counts them.
    """
    local = (
        pl.Series(end_time)
        .dt.replace_time_zone("UTC")
        .dt.convert_time_zone("Europe/London")
        .dt.offset_by("-30m")
    )
    return (local.dt.hour().cast(pl.Int64) * 2 + local.dt.minute().cast(pl.Int64) // 30).to_numpy()


def baseline_columns(
    *, end_time: np.ndarray, weather_columns: np.ndarray, holiday: np.ndarray, quarter: np.ndarray
) -> np.ndarray:
    """Return the baseline's 405 columns for the given half-hours.

    The columns are 336 from `studies.pv_separation.baseline_design` (144 day-type indicators and
    192 harmonic columns), 4 temperature columns, 48 bank-holiday indicators (one per local
    half-hour of the day), and 17 quarterly levels (the first quarter has none).

    Args:
        end_time: The half-hour end stamps in UTC.
        weather_columns: The 4 standardised temperature columns of those half-hours.
        holiday: Whether each half-hour starts on a bank holiday.
        quarter: Each half-hour's quarter, zero-based.

    Returns:
        The array, `(len(end_time), N_BASE_COLUMNS)`.
    """
    n = len(end_time)
    seasonal = baseline_design(half_hour_end_time=pl.Series(end_time), flexibility="seasonal")
    holiday_columns = np.zeros((n, N_HOLIDAY_COLUMNS))
    slot = local_slot(end_time=end_time)
    holiday_columns[np.flatnonzero(holiday), slot[holiday]] = 1.0
    levels = np.zeros((n, N_QUARTERS - 1))
    later = quarter >= 1
    levels[np.flatnonzero(later), quarter[later] - 1] = 1.0
    return np.hstack([seasonal, weather_columns, holiday_columns, levels])


@dataclass(frozen=True)
class PrimaryInputs:
    """Everything one Primary's worlds are built from, on the half-hour grid."""

    site: str
    series_id: int
    latitude: float
    longitude: float
    end_time: np.ndarray
    power: np.ndarray
    forecast: np.ndarray
    cams: pl.DataFrame
    hourly: pl.DataFrame
    sky: SunAndSky
    reliable: np.ndarray
    """Per half-hour: its CAMS hour is reliable (the fit leaves other half-hours out)."""
    weather: Weather
    quarter: np.ndarray
    sun_up: np.ndarray
    """Per half-hour: the sun is more than 5 degrees above the horizon."""
    usable_daylight: np.ndarray
    """Per half-hour: sun up, reliable CAMS, irradiance above 5 W/m2, and a finite power reading."""
    usable_share: np.ndarray
    """Per quarter: the usable share of its daylight half-hours (NaN with no daylight)."""
    scored: np.ndarray
    """Per quarter: whether the share reaches 60%."""


def quarter_usable_share(
    *, quarter: np.ndarray, sun_up: np.ndarray, usable_daylight: np.ndarray
) -> np.ndarray:
    """Return, for each quarter, the usable daylight half-hours over its daylight half-hours.

    Args:
        quarter: Each half-hour's quarter.
        sun_up: Whether the sun is above 5 degrees.
        usable_daylight: Whether the half-hour is a usable daylight half-hour.

    Returns:
        One share per quarter, NaN where a quarter has no daylight half-hour.
    """
    daylight_count = np.bincount(quarter[sun_up], minlength=N_QUARTERS).astype(float)
    usable_count = np.bincount(quarter[usable_daylight], minlength=N_QUARTERS).astype(float)
    return np.divide(
        usable_count, daylight_count, out=np.full(N_QUARTERS, np.nan), where=daylight_count > 0
    )


def scored_quarters(*, share: np.ndarray) -> np.ndarray:
    """Return which quarters are scored: those with at least 60% of daylight half-hours usable.

    Args:
        share: `quarter_usable_share`'s output.

    Returns:
        A boolean per quarter.
    """
    return np.nan_to_num(share, nan=0.0) >= MIN_QUARTER_SHARE


def store_to_grid(*, stored: np.ndarray) -> np.ndarray:
    """Move one of experiment 327's stored arrays onto the grid.

    Store index `i` holds the half-hour that ends at `GRID_START + 30 min x i`. Grid index `i` is
    the half-hour that ends at `GRID_START + 30 min x (i + 1)`, which is store index `i + 1`. The
    last grid half-hour therefore has no stored value and is NaN.

    Args:
        stored: A stored array of `GRID_LENGTH` values.

    Returns:
        The array on the grid.
    """
    return np.concatenate([stored[1:], [np.nan]])


@lru_cache(maxsize=2)
def load_primary(*, site: str) -> PrimaryInputs:
    """Load and align one Primary's inputs.

    The stored power and forecast are shifted by `store_to_grid`. The weather, the holiday flag, and
    the baseline's day-type columns are keyed on the grid's own stamps, so they already agree with
    the shifted power.

    Args:
        site: The Primary's label.

    Returns:
        The inputs.

    Raises:
        ValueError: When a stored array does not have `GRID_LENGTH` entries.
    """
    row = primary_table().filter(pl.col("site") == site).row(0, named=True)
    series_id = int(row["time_series_id"])
    stored_power = np.load(E327 / f"power/power_{series_id}.npy").astype(float)
    stored_forecast = np.load(E327 / f"forecast/C/forecast_{series_id}.npy").astype(float)
    if len(stored_power) != GRID_LENGTH or len(stored_forecast) != GRID_LENGTH:
        msg = (
            f"{site}: stored arrays have {len(stored_power)} and {len(stored_forecast)} half-hours"
        )
        raise ValueError(msg)
    power = store_to_grid(stored=stored_power)
    forecast = store_to_grid(stored=stored_forecast)
    cams = load_cams(site=site)
    hourly = hourly_irradiance(cams=cams)
    lat, lon = float(row["latitude"]), float(row["longitude"])
    sky = build_sky(hourly=hourly, latitude=lat, longitude=lon)
    reliable = np.repeat(hourly["reliable"].to_numpy(), 2)
    end_time = grid_end_times()
    quarter = quarter_index(end_time=end_time)
    sun_up = daylight(sky=sky)
    usable = sun_up & reliable & (sky.ghi_w_m2 > DAYLIGHT_GHI_W_M2) & np.isfinite(power)
    share = quarter_usable_share(quarter=quarter, sun_up=sun_up, usable_daylight=usable)
    return PrimaryInputs(
        site=site,
        series_id=series_id,
        latitude=lat,
        longitude=lon,
        end_time=end_time,
        power=power,
        forecast=forecast,
        cams=cams,
        hourly=hourly,
        sky=sky,
        reliable=reliable,
        weather=load_weather(series_id=series_id),
        quarter=quarter,
        sun_up=sun_up,
        usable_daylight=usable,
        usable_share=share,
        scored=scored_quarters(share=share),
    )


def placebo_sky(*, inputs: PrimaryInputs) -> tuple[SunAndSky, np.ndarray]:
    """Return the placebo sky (CAMS of the same hour a year later) and its reliability mask.

    Args:
        inputs: The Primary's inputs.

    Returns:
        The sky, and a boolean per half-hour that is true where both the real and the placebo
        CAMS hour are reliable.
    """
    hourly = hourly_irradiance(cams=inputs.cams, shift_days=PLACEBO_SHIFT_DAYS)
    sky = build_sky(hourly=hourly, latitude=inputs.latitude, longitude=inputs.longitude)
    both = hourly["reliable"].to_numpy() & inputs.hourly["reliable"].to_numpy()
    return sky, np.repeat(both, 2)


def unit_term(
    *, inputs: PrimaryInputs, fleet: FleetParams, sky: SunAndSky | None = None
) -> np.ndarray:
    """Return the output of one megawatt of a fleet on the Primary's sky and weather.

    Args:
        inputs: The Primary's inputs.
        fleet: The fleet's mix, tilt, and DC:AC ratio.
        sky: A sky that replaces the real one (the placebo sky).

    Returns:
        The output at every half-hour of the grid, megawatts per megawatt of AC capacity.
    """
    return fleet_unit_output(
        sky=sky if sky is not None else inputs.sky,
        air_c=inputs.weather.air_c,
        wind_ms=inputs.weather.wind_ms,
        fleet=fleet,
    )


@dataclass(frozen=True)
class FitData:
    """The part of a Primary's regression that no world changes.

    Attributes:
        rows: Grid indices of the half-hours the fit reads (a finite reading and a reliable CAMS
            hour).
        base: The 405 baseline columns of those rows.
        quarter: Each row's quarter.
        day: Each row's day number (the UTC day of its start).
        daylight: Whether each row is a daylight row of the noise model (CAMS above 5 W/m2).
        ridge_abs: The absolute ridge precision of the baseline's coefficients, before the division
            by the noise scale.
    """

    rows: np.ndarray
    base: np.ndarray
    quarter: np.ndarray
    day: np.ndarray
    daylight: np.ndarray
    ridge_abs: float


def build_fit_data(*, inputs: PrimaryInputs, reliable: np.ndarray | None = None) -> FitData:
    """Build the baseline design of the rows a fit of this Primary reads.

    Args:
        inputs: The Primary's inputs.
        reliable: A reliability mask that replaces the real one (the placebo sky's).

    Returns:
        The fit data.
    """
    ok = inputs.reliable if reliable is None else reliable
    rows = np.flatnonzero(np.isfinite(inputs.power) & ok)
    base = baseline_columns(
        end_time=inputs.end_time[rows],
        weather_columns=inputs.weather.temperature_columns[rows],
        holiday=inputs.weather.holiday[rows],
        quarter=inputs.quarter[rows],
    )
    return FitData(
        rows=rows,
        base=base,
        quarter=inputs.quarter[rows],
        day=rows // HALF_HOURS_PER_DAY,
        daylight=inputs.sky.ghi_w_m2[rows] > DAYLIGHT_GHI_W_M2,
        ridge_abs=ridge_precision(seasonal=base[:, :SEASONAL_COLUMNS]),
    )


def ridge_precision(*, seasonal: np.ndarray) -> float:
    """Return the ridge precision: 1e-3 times the mean squared column norm of the seasonal columns.

    Args:
        seasonal: The baseline's calendar columns of the rows that the fit reads.

    Returns:
        The precision in the units of the unweighted normal equations.
    """
    return float(RIDGE_PER_COLUMN * (seasonal**2).sum(axis=0).mean())


# --- Farms ----------------------------------------------------------------------------------
FARM_TIME_IS_PERIOD_END: Final[bool] = True
"""The farm file labels each half-hour with its end, like experiment 39's frame (the code review
confirms this convention)."""


def load_farm_per_unit(*, farm: int) -> np.ndarray:
    """Return one farm's output per unit of its effective capacity on the grid, NaN where missing.

    Args:
        farm: The farm's number, 0 to 5, counting from the lowest series identifier.

    Returns:
        An array of `GRID_LENGTH` values.
    """
    series_id = sorted(FARM_IDS)[farm]
    capacity = float(
        pv_sites().filter(pl.col("time_series_id") == series_id)["effective_capacity_mw"][0]
    )
    frame = (
        pl.scan_parquet(FARMS_PATH)
        .filter((pl.col("time_series_id") == series_id) & (pl.col("time") <= pl.lit(CUT)))
        .select("time", "power")
        .collect()
    )
    assert_cut(times=frame["time"].dt.replace_time_zone(None).to_numpy(), what="farm file")
    return farm_to_grid(
        times=frame["time"].dt.replace_time_zone(None).to_numpy(),
        power=frame["power"].to_numpy().astype(float),
        capacity_mw=capacity,
    )


def farm_to_grid(*, times: np.ndarray, power: np.ndarray, capacity_mw: float) -> np.ndarray:
    """Put a farm's rows on the grid, per unit of capacity.

    Args:
        times: The rows' timestamps, `datetime64[us]`.
        power: The rows' power in megawatts.
        capacity_mw: The farm's effective capacity.

    Returns:
        An array of `GRID_LENGTH` values, NaN where the farm has no finite reading.
    """
    offset = 1 if FARM_TIME_IS_PERIOD_END else 0
    start = np.datetime64(GRID_START.replace(tzinfo=None), "us")
    index = ((times - start) // np.timedelta64(30, "m")).astype(np.int64) - offset
    keep = (index >= 0) & (index < GRID_LENGTH) & np.isfinite(power)
    out = np.full(GRID_LENGTH, np.nan)
    out[index[keep]] = power[keep] / capacity_mw
    return out
