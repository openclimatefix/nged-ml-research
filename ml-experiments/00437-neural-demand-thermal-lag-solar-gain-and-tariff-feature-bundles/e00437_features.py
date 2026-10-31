# ruff: noqa: E501
r"""Trailing-window input columns for experiment 437, built once per Primary and cached.

The pooled plain network of experiment 425 sees one valid time per row, so a column about the hours
before the valid time has to be built outside it. This module builds 16 such columns, in three
bundles, for every (run, valid time) row of experiment 397's ensemble summaries, and stores them
in one table per Primary under this experiment's data folder.

**Causality (the point of the module).** A row belongs to the run whose forecast origin is `O`
(09:00 UTC). Its weather window ends at the valid time `t`. The window is built from one
"combined series" per run:

- for half-hours at or before `O`, the series holds the ensemble-mean value of the *freshest run
  that was published by `O` and covers that half-hour*, which is the previous run's first day
  (valid times in `(O_prev, O_prev + 24 h]`, so leads of about 9 to 33 hours). This is the
  "analysis proxy": no ERA5 is used, because ERA5 arrives about 5 days late and a column built from
  it could not be served;
- for half-hours after `O`, the series holds this run's own ensemble mean.

No half-hour after `O` ever comes from a later run, and no column reads the observed power. The
check `check` below shuffles nothing; it recomputes a sample of rows with every later run deleted
and requires an identical result.

Columns (the bundle each belongs to is in `BUNDLES`):

- T, thermal lag: trailing means of temperature over 3, 6, and 24 h (`t_mean_3h`, `t_mean_6h`,
  `t_mean_24h`); the temperature change over 3 and 6 h (`t_change_3h`, `t_change_6h`); the minimum
  and maximum over the Europe/London local day of the valid time (`t_min_day`, `t_max_day`, which
  use the later hours of the same run, as the run's own forecast is known at `O`); and the days
  since the trailing 24-hour mean last exceeded 5 degrees C, capped at 10 (`t_cold_snap_days`);
- S, solar gain: trailing means of shortwave over 3, 6, and 24 h (`sw_mean_3h`, `sw_mean_6h`,
  `sw_mean_24h`); the 3-hour change in the clear-sky index (`csi_change_3h`; the index is
  shortwave over Haurwitz clear-sky shortwave, defined where the clear-sky value exceeds
  20 W/m2, clipped to 0 to 1.5, and held at its last daylight value overnight); and the realised
  CAMS irradiance of the last complete UTC day before `O` (`cams_prev_day_mean` in W/m2 and
  `cams_prev_day_csi`, its ratio to the CAMS clear-sky irradiance), which is constant within a
  run and which production could read, because CAMS publishes the previous day at midnight UTC;
- C, calendar, tariff, and national signals: `economy_7`, 1 on the half-hours that start in 00:30
  to 07:30 UTC (the stamp is the period end, so 01:00 to 07:30), and `economy_7_cold`, that flag
  times `max(15.5 - temperature, 0)`; five Lincolnshire school-term columns of the valid time's
  local date from experiment 11's calendar (`school_term_flag`; the signed days to the nearest term
  start and end divided by 13 and the signed days from the year's first working day divided by 14,
  each 0 outside its window; `school_window_flag`, 1 inside any of the three windows); and seven
  national signals of the local day before the run's origin date, constant within a run, all of
  which exist by 09:00 UTC on the origin day: the mean, maximum, and 16:00 to 19:00 mean of the
  Elexon market index price (`price_prev_day_*`), a flag for a Demand Flexibility Service event
  (`dfs_prev_day`), and the day's mean national demand and mean embedded solar and wind generation
  from NESO's historic demand data. The 16 Primaries all lie in the Lincolnshire box of experiment
  11, so one county's calendar applies to every Primary.

- F, look-ahead: forward means of temperature and shortwave over the next 6 and 24 hours after the
  valid time (`f_t_mean_6h`, `f_t_mean_24h`, `f_sw_mean_6h`, `f_sw_mean_24h`), taken from the same
  run's ensemble mean only, so they exist at the origin; near the end of the run's horizon the
  window is truncated, and `f_cover_6h` and `f_cover_24h` hold the share of the window the run
  covers (a window with no covered half-hour takes the valid-time value).

Four diagnostic columns (`diag_*`) are stored beside the inputs for the scorer's slices; they are
never inputs.

Commands (run with `PYTHONPATH` as `e00437_fit.py` shows):

- `build`: write the table of every Primary (skips existing files);
- `check`: the causality check and the range checks on one Primary.
"""

import argparse
import time
from datetime import date
from itertools import pairwise
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00418_prep as prep418  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.baselines import haurwitz_w_m2
from studies.solar import zenith
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_DIR_NAME: Final[str] = (
    "00437-neural-demand-thermal-lag-solar-gain-and-tariff-feature-bundles"
)
WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / EXPERIMENT_DIR_NAME
TRAILING_DIR: Final[Path] = WORK_DIR / "work" / "trailing"
BUNDLES: Final[dict[str, tuple[str, ...]]] = {
    "T": (
        "t_mean_3h",
        "t_mean_6h",
        "t_mean_24h",
        "t_change_3h",
        "t_change_6h",
        "t_min_day",
        "t_max_day",
        "t_cold_snap_days",
    ),
    "S": (
        "sw_mean_3h",
        "sw_mean_6h",
        "sw_mean_24h",
        "csi_change_3h",
        "cams_prev_day_mean",
        "cams_prev_day_csi",
    ),
    "C": (
        "economy_7",
        "economy_7_cold",
        "school_term_flag",
        "school_term_start_days",
        "school_term_end_days",
        "school_return_days",
        "school_window_flag",
        "price_prev_day_mean",
        "price_prev_day_max",
        "price_prev_day_evening",
        "dfs_prev_day",
        "national_demand_prev_day",
        "embedded_solar_prev_day",
        "embedded_wind_prev_day",
    ),
    "F": (
        "f_t_mean_6h",
        "f_t_mean_24h",
        "f_sw_mean_6h",
        "f_sw_mean_24h",
        "f_cover_6h",
        "f_cover_24h",
    ),
}
INPUT_COLUMNS: Final[tuple[str, ...]] = tuple(c for b in BUNDLES.values() for c in b)
DIAGNOSTIC_COLUMNS: Final[tuple[str, ...]] = (
    "diag_temperature",
    "diag_csi",
    "diag_short_window_24h",
    "diag_missing_history",
)
STEP_US: Final[int] = 1_800_000_000
US_PER_MINUTE: Final[int] = 60_000_000
STEPS_PER_DAY: Final[int] = 48
LOOKBACK_STEPS: Final[int] = 12 * STEPS_PER_DAY
"""History kept in front of the origin: the 10-day cold-snap cap plus a 24-hour mean window."""
COLD_SNAP_CAP_DAYS: Final[float] = 10.0
COLD_SNAP_THRESHOLD_C: Final[float] = 5.0
HEATING_BASE_C: Final[float] = 15.5
MIN_CLEAR_SKY_W_M2: Final[float] = 20.0
MAX_CSI: Final[float] = 1.5
GRID_US: Final[int] = int(common.GRID_START.timestamp() * 1_000_000)
ECONOMY_7_FIRST_MINUTE: Final[int] = 60
ECONOMY_7_LAST_MINUTE: Final[int] = 450
"""A half-hour starting 00:30 ends at 01:00 (minute 60); the last one starting 07:00 ends 07:30."""


def grid_of(*, us: np.ndarray) -> np.ndarray:
    """Return the index on experiment 87's half-hourly grid of microsecond stamps."""
    return ((us - GRID_US) // STEP_US).astype(np.int64)


def trailing_mean(*, a: np.ndarray, n: int, at: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the mean of the finite values of `a` over the `n` steps ending at each position.

    Args:
        a: The series.
        n: The window length in steps; a window reaching before the start uses what exists.
        at: Positions to evaluate at.

    Returns:
        The means, and how many finite values each window held.
    """
    finite = np.isfinite(a)
    total = np.concatenate([[0.0], np.cumsum(np.where(finite, a, 0.0))])
    count = np.concatenate([[0], np.cumsum(finite)])
    start = np.maximum(at + 1 - n, 0)
    n_found = count[at + 1] - count[start]
    return (total[at + 1] - total[start]) / np.maximum(n_found, 1), n_found


def forward_mean(*, a: np.ndarray, n: int, at: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the mean of the finite values of `a` over the `n` steps after each position.

    Args:
        a: The series; steps past its end are not covered.
        n: The window length in steps.
        at: Positions to evaluate at (the window excludes the position itself).

    Returns:
        The means (NaN where no step is covered), and the covered share of the `n` steps.
    """
    finite = np.isfinite(a)
    total = np.concatenate([[0.0], np.cumsum(np.where(finite, a, 0.0))])
    count = np.concatenate([[0], np.cumsum(finite)])
    stop = np.minimum(at + 1 + n, len(a))
    n_found = count[stop] - count[at + 1]
    mean = np.where(n_found > 0, (total[stop] - total[at + 1]) / np.maximum(n_found, 1), np.nan)
    return mean, n_found / n


def local_date_codes() -> np.ndarray:
    """Return, per grid half-hour, the Europe/London date of the half-hour ending at the stamp."""
    return (
        common.grid_times()
        .dt.offset_by("-30m")
        .dt.convert_time_zone("Europe/London")
        .dt.date()
        .cast(pl.Int32)
        .to_numpy()
    )


def clear_sky_on_grid(*, latitude: float, longitude: float) -> np.ndarray:
    """Return the Haurwitz clear-sky irradiance (W/m2) at the midpoint of every grid half-hour."""
    stamps = common.grid_times().dt.offset_by("-15m")
    return haurwitz_w_m2(
        apparent_zenith_deg=zenith(stamps=stamps, latitude=latitude, longitude=longitude)
    )


def daily_dense(
    *, frame: pl.DataFrame, columns: tuple[str, ...]
) -> tuple[int, dict[str, np.ndarray]]:
    """Return the first date code and a dense array per column, from a table keyed by `local_date`."""
    frame = frame.sort("local_date").with_columns(code=pl.col("local_date").cast(pl.Int32))
    codes = frame["code"].to_numpy()
    first = int(codes.min())
    size = int(codes.max()) - first + 1
    dense = {c: np.full(size, np.nan) for c in columns}
    index = (codes - first).astype(np.int64)
    for c in columns:
        dense[c][index] = frame[c].cast(pl.Float64).to_numpy()
    return first, dense


def lookup(*, first: int, dense: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """Return `dense` at date `codes`, NaN outside the array."""
    index = codes - first
    inside = (index >= 0) & (index < len(dense))
    return np.where(inside, dense[np.clip(index, 0, len(dense) - 1)], np.nan)


def school_columns() -> tuple[int, dict[str, np.ndarray]]:
    """Return Lincolnshire's school calendar columns per local date, as dense arrays.

    The windowed signed-day columns of experiment 11 are divided by their window and set to 0
    outside it; `school_window_flag` is 1 where any of the three windows holds the date.
    """
    import e00011_term_dates as term_dates  # ty: ignore[unresolved-import]

    calendar = term_dates.calendar_frame(terms=term_dates.read_terms()).select(
        "local_date",
        "school_term_flag",
        "days_from_term_start_w",
        "days_from_term_end_w",
        "days_from_return_w",
    )
    window = (
        pl.col("days_from_term_start_w").is_not_null()
        | pl.col("days_from_term_end_w").is_not_null()
        | pl.col("days_from_return_w").is_not_null()
    )
    scaled = calendar.select(
        "local_date",
        "school_term_flag",
        school_term_start_days=pl.col("days_from_term_start_w").fill_null(0.0)
        / term_dates.WINDOW_DAYS,
        school_term_end_days=pl.col("days_from_term_end_w").fill_null(0.0) / term_dates.WINDOW_DAYS,
        school_return_days=pl.col("days_from_return_w").fill_null(0.0)
        / term_dates.RETURN_WINDOW_AFTER,
        school_window_flag=window.cast(pl.Float64),
    )
    return daily_dense(frame=scaled, columns=tuple(c for c in scaled.columns if c != "local_date"))


def national_daily() -> tuple[int, dict[str, np.ndarray]]:
    """Return the national signals per settlement date, as dense arrays.

    Prices come from the Elexon market index day files (`price_gbp_per_mwh`; the evening mean is
    settlement periods 33 to 38, 16:00 to 19:00 local), demand and embedded generation from NESO's
    historic demand files (a daily mean of half-hourly outturn), and the Demand Flexibility Service
    flag from the service-requirement table (1 on a date with any live requirement).
    """
    d = ML_EXPERIMENTS_DATA_DIR
    prices = (
        pl.scan_parquet(
            sorted(
                (d / "p3-price-download" / "elexon_mid_apx" / "_day_cache").glob(
                    "202[3-6]-*.parquet"
                )
            )
        )
        .group_by(pl.col("settlement_date").alias("local_date"))
        .agg(
            pl.col("price_gbp_per_mwh").mean().alias("price_prev_day_mean"),
            pl.col("price_gbp_per_mwh").max().alias("price_prev_day_max"),
            pl.col("price_gbp_per_mwh")
            .filter(pl.col("settlement_period").is_between(33, 38))
            .mean()
            .alias("price_prev_day_evening"),
        )
        .collect()
    )
    public = d / "00185-dp-price-dfs-national-signals-past-demand" / "public"
    demand = pl.concat(
        [
            pl.read_csv(
                public / f"demanddata_{year}.csv",
                columns=[
                    "SETTLEMENT_DATE",
                    "ND",
                    "EMBEDDED_SOLAR_GENERATION",
                    "EMBEDDED_WIND_GENERATION",
                ],
            ).with_columns(
                local_date=pl.col("SETTLEMENT_DATE")
                .str.to_date("%d-%b-%Y", strict=False)
                .fill_null(pl.col("SETTLEMENT_DATE").str.to_date("%Y-%m-%d", strict=False))
            )
            for year in (2024, 2025, 2026)
        ]
    )
    daily = demand.group_by("local_date").agg(
        pl.col("ND").mean().alias("national_demand_prev_day"),
        pl.col("EMBEDDED_SOLAR_GENERATION").mean().alias("embedded_solar_prev_day"),
        pl.col("EMBEDDED_WIND_GENERATION").mean().alias("embedded_wind_prev_day"),
    )
    dfs = (
        pl.read_csv(
            d / "00177-dp-worst-days-other-data-sources" / "public" / "dfs_service_requirement.csv"
        )
        .select(local_date=pl.col("Delivery Date").str.to_date("%d/%m/%Y"))
        .unique()
        .with_columns(dfs_prev_day=pl.lit(1.0))
    )
    table = prices.join(daily, on="local_date", how="full", coalesce=True).join(
        dfs, on="local_date", how="full", coalesce=True
    )
    table = table.with_columns(pl.col("dfs_prev_day").fill_null(0.0))
    names = (
        "price_prev_day_mean",
        "price_prev_day_max",
        "price_prev_day_evening",
        "dfs_prev_day",
        "national_demand_prev_day",
        "embedded_solar_prev_day",
        "embedded_wind_prev_day",
    )
    first, dense = daily_dense(frame=table, columns=names)
    start = (date(2024, 3, 1) - date(1970, 1, 1)).days - first
    stop = (date(2026, 7, 1) - date(1970, 1, 1)).days - first
    for name, values in dense.items():
        gaps = int((~np.isfinite(values[start:stop])).sum())
        if gaps:
            print(
                f"national signal {name}: {gaps} missing dates between 2024-03-01 and 2026-06-30, filled with the last earlier value"
            )
        position = np.maximum.accumulate(np.where(np.isfinite(values), np.arange(len(values)), 0))
        dense[name] = values[position]
    return first, dense


def cams_previous_day(*, site: str) -> dict[int, tuple[float, float]]:
    """Return, per UTC date code, the CAMS day-mean irradiance and its ratio to clear-sky.

    The hourly stamps are period ends, so a UTC day holds the stamps 01:00 of that day to 00:00 of
    the next. The ratio is the day's summed irradiance over its summed clear-sky irradiance.
    """
    cams = (
        pl.read_parquet(prep418.CAMS_PATH)
        .filter(pl.col("site") == site)
        .with_columns(day=pl.col("time").dt.offset_by("-1h").dt.date().cast(pl.Int32))
        .group_by("day")
        .agg(
            pl.col("ghi_w_m2").mean().alias("ghi"),
            pl.col("ghi_w_m2").sum().alias("ghi_sum"),
            pl.col("clear_sky_ghi_w_m2").sum().alias("clear_sum"),
            pl.len().alias("n"),
        )
        .filter(pl.col("n") == 24)
    )
    return {
        int(d): (float(g), float(s / c) if c > 0 else float("nan"))
        for d, g, s, c in cams.select("day", "ghi", "ghi_sum", "clear_sum").iter_rows()
    }


def site_table(*, row: dict, local_date: np.ndarray, daily: dict) -> pl.DataFrame:  # noqa: PLR0915
    """Return one Primary's input and diagnostic columns for every (run, valid time) of its summary."""
    summary = (
        prep.load_summary(series_id=int(row["time_series_id"]))
        .select(
            origin=pl.col("power_fcst_init_time").dt.epoch("us"),
            valid=pl.col("valid_time").dt.epoch("us"),
            temperature=pl.col("mean_temperature_2m").cast(pl.Float64),
            shortwave=pl.col("mean_shortwave").cast(pl.Float64),
        )
        .sort("origin", "valid")
    )
    origin_us, valid_us = summary["origin"].to_numpy(), summary["valid"].to_numpy()
    temperature, shortwave = summary["temperature"].to_numpy(), summary["shortwave"].to_numpy()
    oi, vi = grid_of(us=origin_us), grid_of(us=valid_us)
    size = common.grid_times().len()
    history_t, history_s = np.full(size, np.nan), np.full(size, np.nan)
    first_day = (vi - oi >= 1) & (vi - oi <= STEPS_PER_DAY)
    if len(np.unique(vi[first_day])) != int(first_day.sum()):
        msg = "two runs' first days overlap on the grid"
        raise ValueError(msg)
    history_t[vi[first_day]] = temperature[first_day]
    history_s[vi[first_day]] = shortwave[first_day]
    clear = clear_sky_on_grid(latitude=float(row["latitude"]), longitude=float(row["longitude"]))
    cams = cams_previous_day(site=row["site"])
    n_rows = len(summary)
    out = {c: np.full(n_rows, np.nan) for c in (*INPUT_COLUMNS, *DIAGNOSTIC_COLUMNS)}
    boundaries = np.flatnonzero(np.r_[True, oi[1:] != oi[:-1], True])
    for start, stop in pairwise(boundaries):
        rows = np.arange(start, stop)
        origin_index = int(oi[start])
        lo = origin_index - LOOKBACK_STEPS
        hi = int(vi[rows].max()) + 1
        t_all, s_all = history_t[lo:hi].copy(), history_s[lo:hi].copy()
        after = origin_index + 1 - lo
        # Nothing after the origin may come from a later run.
        t_all[after:], s_all[after:] = np.nan, np.nan
        p = vi[rows] - lo
        t_all[p], s_all[p] = temperature[rows], shortwave[rows]
        positions = np.arange(len(t_all))
        for n_steps, name in ((6, "3h"), (12, "6h"), (STEPS_PER_DAY, "24h")):
            out[f"t_mean_{name}"][rows], _ = trailing_mean(a=t_all, n=n_steps, at=p)
            sw_mean, found = trailing_mean(a=s_all, n=n_steps, at=p)
            out[f"sw_mean_{name}"][rows] = sw_mean
            if name == "24h":
                out["diag_short_window_24h"][rows] = found < n_steps
        for n_steps, name in ((12, "6h"), (STEPS_PER_DAY, "24h")):
            ahead_t, share = forward_mean(a=t_all, n=n_steps, at=p)
            ahead_s, _ = forward_mean(a=s_all, n=n_steps, at=p)
            out[f"f_t_mean_{name}"][rows] = np.where(np.isfinite(ahead_t), ahead_t, t_all[p])
            out[f"f_sw_mean_{name}"][rows] = np.where(np.isfinite(ahead_s), ahead_s, s_all[p])
            out[f"f_cover_{name}"][rows] = share
        out["t_change_3h"][rows] = np.nan_to_num(t_all[p] - t_all[p - 6], nan=0.0)
        out["t_change_6h"][rows] = np.nan_to_num(t_all[p] - t_all[p - 12], nan=0.0)
        codes = local_date[lo:hi]
        new_day = np.r_[False, codes[1:] != codes[:-1]]
        group_starts = np.flatnonzero(np.r_[True, new_day[1:]])
        group = np.cumsum(new_day)
        out["t_min_day"][rows] = np.fmin.reduceat(t_all, group_starts)[group[p]]
        out["t_max_day"][rows] = np.fmax.reduceat(t_all, group_starts)[group[p]]
        mean_24h, _ = trailing_mean(a=t_all, n=STEPS_PER_DAY, at=positions)
        warm = np.nan_to_num(mean_24h, nan=-99.0) > COLD_SNAP_THRESHOLD_C
        last_warm = np.maximum.accumulate(np.where(warm, positions, -1))
        days = np.where(last_warm >= 0, (positions - last_warm) / STEPS_PER_DAY, COLD_SNAP_CAP_DAYS)
        out["t_cold_snap_days"][rows] = np.minimum(days, COLD_SNAP_CAP_DAYS)[p]
        clear_window = clear[lo:hi]
        csi = np.where(
            (clear_window > MIN_CLEAR_SKY_W_M2) & np.isfinite(s_all),
            np.clip(s_all / np.maximum(clear_window, 1.0), 0.0, MAX_CSI),
            np.nan,
        )
        held = csi[np.maximum.accumulate(np.where(np.isfinite(csi), positions, 0))]
        out["csi_change_3h"][rows] = np.nan_to_num(held[p] - held[p - 6], nan=0.0)
        out["diag_csi"][rows] = csi[p]
        out["diag_missing_history"][rows] = ~np.isfinite(
            history_t[origin_index - STEPS_PER_DAY + 1 : origin_index + 1]
        ).all()
        origin_day = int((origin_us[start] // (86_400 * 1_000_000)) + 0)  # days since the epoch
        out["cams_prev_day_mean"][rows], out["cams_prev_day_csi"][rows] = cams.get(
            origin_day - 1, (np.nan, np.nan)
        )
    minutes = (valid_us // US_PER_MINUTE) % 1440
    out["economy_7"] = (
        (minutes >= ECONOMY_7_FIRST_MINUTE) & (minutes <= ECONOMY_7_LAST_MINUTE)
    ).astype(np.float64)
    out["economy_7_cold"] = out["economy_7"] * np.maximum(HEATING_BASE_C - temperature, 0.0)
    out["diag_temperature"] = temperature
    valid_codes = local_date[vi]
    first, dense = daily["school"]
    for name, values in dense.items():
        out[name] = lookup(first=first, dense=values, codes=valid_codes)
    first, dense = daily["national"]
    previous_codes = local_date[oi] - 1
    for name, values in dense.items():
        filled = np.where(np.isfinite(values), values, np.nan)
        out[name] = lookup(first=first, dense=filled, codes=previous_codes)
    return pl.DataFrame(
        {"origin_us": origin_us, "valid_us": valid_us}
        | {c: v.astype(np.float32) for c, v in out.items()}
    )


def dense_arrays(*, row: dict, local_date: np.ndarray) -> dict[str, np.ndarray]:
    """Return a Primary's pretraining columns on the whole grid, from the dense ERA5-and-CAMS weather.

    The pretraining rows of experiment 418 have no ensemble runs, so the trailing and forward
    windows come straight from the dense weather (ERA5 temperature and CAMS shortwave, the same
    stand-ins the pretraining rows use for the valid-time weather). Only the T, S, and F columns
    exist for pretraining. A forward window is truncated by the pseudo-run's end in
    `pretrain_extras`, not here: the arrays `forward_sum_t`, `forward_sum_s`, `forward_count_t`,
    and `forward_count_s` hold the cumulative sums and counts that function differences.
    """
    dense = prep418.dense_weather(series_id=int(row["time_series_id"]), site=row["site"])
    temperature = dense[:, prep418.WEATHER.index("temperature_2m")].astype(np.float64)
    shortwave = dense[:, prep418.WEATHER.index("shortwave")].astype(np.float64)
    positions = np.arange(len(temperature))
    out: dict[str, np.ndarray] = {}
    for n_steps, name in ((6, "3h"), (12, "6h"), (STEPS_PER_DAY, "24h")):
        out[f"t_mean_{name}"], _ = trailing_mean(a=temperature, n=n_steps, at=positions)
        out[f"sw_mean_{name}"], _ = trailing_mean(a=shortwave, n=n_steps, at=positions)
    for steps, name in ((6, "3h"), (12, "6h")):
        out[f"t_change_{name}"] = np.nan_to_num(temperature - np.roll(temperature, steps), nan=0.0)
    new_day = np.r_[False, local_date[1:] != local_date[:-1]]
    group_starts = np.flatnonzero(np.r_[True, new_day[1:]])
    group = np.cumsum(new_day)
    out["t_min_day"] = np.fmin.reduceat(temperature, group_starts)[group]
    out["t_max_day"] = np.fmax.reduceat(temperature, group_starts)[group]
    warm = np.nan_to_num(out["t_mean_24h"], nan=-99.0) > COLD_SNAP_THRESHOLD_C
    last_warm = np.maximum.accumulate(np.where(warm, positions, -1))
    days = np.where(last_warm >= 0, (positions - last_warm) / STEPS_PER_DAY, COLD_SNAP_CAP_DAYS)
    out["t_cold_snap_days"] = np.minimum(days, COLD_SNAP_CAP_DAYS)
    clear = clear_sky_on_grid(latitude=float(row["latitude"]), longitude=float(row["longitude"]))
    csi = np.where(
        (clear > MIN_CLEAR_SKY_W_M2) & np.isfinite(shortwave),
        np.clip(shortwave / np.maximum(clear, 1.0), 0.0, MAX_CSI),
        np.nan,
    )
    held = csi[np.maximum.accumulate(np.where(np.isfinite(csi), positions, 0))]
    out["csi_change_3h"] = np.nan_to_num(held - np.roll(held, 6), nan=0.0)
    for name, series in (("t", temperature), ("s", shortwave)):
        finite = np.isfinite(series)
        out[f"forward_sum_{name}"] = np.concatenate(
            [[0.0], np.cumsum(np.where(finite, series, 0.0))]
        )
        out[f"forward_count_{name}"] = np.concatenate([[0], np.cumsum(finite)]).astype(np.float64)
    out["valid_temperature"] = temperature
    out["valid_shortwave"] = shortwave
    return out


def pretrain_extras(
    *,
    row: dict,
    dense: dict[str, np.ndarray],
    columns: tuple[str, ...],
    valid_us: np.ndarray,
    origin_us: np.ndarray,
    max_steps: int,
) -> np.ndarray:
    """Return the T, S, and F columns of pretraining rows, shape (rows, len(columns)).

    Args:
        row: The Primary's series row.
        dense: The Primary's `dense_arrays`.
        columns: The wanted columns, in order; none may be a C column.
        valid_us: Each row's valid time in microseconds.
        origin_us: Each row's pseudo-run origin in microseconds.
        max_steps: The pseudo-run's length in half-hours; a forward window ends there.

    Returns:
        The columns as float32.

    Raises:
        ValueError: If a C column is requested.
    """
    vi, oi = grid_of(us=valid_us), grid_of(us=origin_us)
    cams = cams_previous_day(site=row["site"])
    origin_day = origin_us // (86_400 * 1_000_000)
    out = np.empty((len(vi), len(columns)), dtype=np.float32)
    for k, name in enumerate(columns):
        if name in BUNDLES["C"]:
            msg = f"{name} is a C column; the pretraining rows have none"
            raise ValueError(msg)
        if name.startswith(("f_cover_", "f_t_mean_", "f_sw_mean_")):
            n_steps = 12 if name.endswith("6h") else STEPS_PER_DAY
            end = np.minimum(vi + n_steps, oi + max_steps)
            kind = "s" if "sw" in name else "t"
            found = dense[f"forward_count_{kind}"][end + 1] - dense[f"forward_count_{kind}"][vi + 1]
            total = dense[f"forward_sum_{kind}"][end + 1] - dense[f"forward_sum_{kind}"][vi + 1]
            if name.startswith("f_cover_"):
                column = found / n_steps
            else:
                valid_value = dense["valid_shortwave" if kind == "s" else "valid_temperature"][vi]
                column = np.where(found > 0, total / np.maximum(found, 1), valid_value)
        elif name in ("cams_prev_day_mean", "cams_prev_day_csi"):
            slot = 0 if name.endswith("mean") else 1
            column = np.array([cams.get(int(d) - 1, (np.nan, np.nan))[slot] for d in origin_day])
        else:
            column = dense[name][vi]
        out[:, k] = column
    return out


def table_path(*, site: str) -> Path:
    """Return where a Primary's table is stored."""
    return TRAILING_DIR / f"{site}.parquet"


def load_table(*, site: str) -> pl.DataFrame:
    """Return a Primary's stored table of columns, keyed by `origin_us` and `valid_us`."""
    return pl.read_parquet(table_path(site=site))


def build_all() -> None:
    """Write the table of every Primary that has none."""
    local_date = local_date_codes()
    daily = {"school": school_columns(), "national": national_daily()}
    TRAILING_DIR.mkdir(parents=True, exist_ok=True)
    for row in prep.primaries().iter_rows(named=True):
        target = table_path(site=row["site"])
        if target.exists():
            print(f"{row['site']}: exists", flush=True)
            continue
        started = time.monotonic()
        table = site_table(row=row, local_date=local_date, daily=daily)
        nulls = {c: table[c].null_count() + int(table[c].is_nan().sum()) for c in INPUT_COLUMNS}
        bad = {c: n for c, n in nulls.items() if n}
        if bad:
            msg = f"{row['site']}: input columns with missing values: {bad}"
            raise ValueError(msg)
        temporary = target.with_name(target.name + ".tmp")
        table.write_parquet(temporary)
        temporary.rename(target)
        print(
            f"{row['site']}: {table.height:,} rows, {time.monotonic() - started:.0f} s", flush=True
        )


def check(*, site: str = "D2") -> None:
    """Print range checks, and recompute a sample of runs with every later run deleted."""
    row = next(r for r in prep.primaries().iter_rows(named=True) if r["site"] == site)
    table = load_table(site=site)
    print(table.select(*INPUT_COLUMNS, *DIAGNOSTIC_COLUMNS).describe())
    short = int(table["diag_short_window_24h"].sum())
    print(f"rows with a 24 h window shorter than 48 half-hours: {short:,} of {table.height:,}")
    print(
        "rows whose origin has a gap in the previous day's history:",
        int(table["diag_missing_history"].sum()),
    )
    # Causality: rebuild with each later run removed from the summary, compare the sampled run.
    full = prep.load_summary(series_id=int(row["time_series_id"]))
    origins = full["power_fcst_init_time"].unique().sort()
    rng = np.random.default_rng(0)
    for pick in rng.choice(len(origins) - 3, size=3, replace=False) + 2:
        cut_origin = origins[int(pick)]
        truncated = full.filter(pl.col("power_fcst_init_time") <= cut_origin)
        original = prep.load_summary
        prep.load_summary = lambda *, series_id, t=truncated: t
        try:
            rebuilt = site_table(
                row=row,
                local_date=local_date_codes(),
                daily={"school": school_columns(), "national": national_daily()},
            )
        finally:
            prep.load_summary = original
        key = int(cut_origin.timestamp() * 1_000_000)
        a = table.filter(pl.col("origin_us") == key).sort("valid_us")
        b = rebuilt.filter(pl.col("origin_us") == key).sort("valid_us")
        worst = max(
            float(np.nanmax(np.abs(a[c].to_numpy() - b[c].to_numpy()))) for c in INPUT_COLUMNS
        )
        print(f"run {cut_origin}: largest difference with all later runs deleted: {worst:.2e}")
        if worst > 1e-5:
            msg = "a column depends on a later run"
            raise AssertionError(msg)


def main() -> None:
    """Build the tables or run the checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "check"))
    parser.add_argument("--site", default="D2")
    args = parser.parse_args()
    if args.command == "build":
        build_all()
    else:
        check(site=args.site)


if __name__ == "__main__":
    main()
