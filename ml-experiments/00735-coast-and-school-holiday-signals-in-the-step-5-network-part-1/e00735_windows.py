# ruff: noqa: E501
"""Experiment 735: window, weather-split, and group masks for the scorer (CPU only, no torch).

README. This module turns local dates into the pre-registered windows, splits holiday Primary-days by
realised weather, and names the Primary groups. It relies on `e00735_columns` (the school calendar,
the local-date convention, the coast distance), `e00418_prep` (ERA5 and CAMS weather on the dense
half-hourly grid), `e00560_data` (the expected-weather table, which stands in after ERA5 ends),
`e00397_prep` (the Primary table), and `studies.eval_protocol.slices` (the group labels).
PYTHONPATH: source `ml-experiments/00524*/env.sh` from the worktree root.

Windows are read from the local date of the valid time (Europe/London, stamp minus 15 minutes), the
same convention as the new input columns.

- School windows, per block: `7 to 4 days before`, `3 to 1 days before`, `first weekend`, `rest of
  block`, `first 3 days after`, and `4 to 7 days after` the day that term resumes. `school window` is
  the union, 7 days before to 7 days after.
- Bank-holiday windows, per bank holiday: `day before`, `the day`, `day after`, and `extended
  weekend` (the Saturday and Sunday that adjoin a Monday or Friday bank holiday).
- Weather split of a Primary-day, on realised weather: `warm and sunny` when the daily maximum 2 m
  temperature and the daily mean shortwave both reach the median of the same calendar month over
  2019-09 to 2025-12 at that Primary, `cold or wet` when the daily maximum is below that median or
  the daily precipitation is at least 1 mm, and `mixed` otherwise. ERA5 ends on 2026-07-31, so later
  days read the expected-weather table (the ENS analysis proxy) and are labelled so.

Precipitation unit: ERA5 `precipitation` in the pilot file is a rate in kg per square metre per
second (equal to millimetres per second; its mean of about 2e-5 gives about 700 mm a year), so a
daily total in millimetres is the sum of the half-hourly rates times 1,800 seconds. The expected-weather
table's column is assumed to use the same unit (experiment 418's check compares both after `log1p`).
"""

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Final

import e00735_columns as cols  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.eval_protocol.slices import MVA_ONLY_LABELS, SHORT_HISTORY_LABELS, SIGNED_MW_LABELS
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SCHOOL_WINDOWS: Final = (
    "7 to 4 days before",
    "3 to 1 days before",
    "first weekend",
    "rest of block",
    "first 3 days after",
    "4 to 7 days after",
)
BANK_WINDOWS: Final = ("day before", "the day", "day after", "extended weekend")
SCHOOL_WINDOW: Final = "school window"
ORDINARY: Final = "ordinary days"
WEATHER_KINDS: Final = ("warm and sunny", "cold or wet", "mixed")
WET_MM: Final = 1.0
MEDIAN_FIRST: Final = date(2019, 9, 1)
MEDIAN_LAST: Final = date(2025, 12, 31)
ERA5_LAST: Final = date(2026, 7, 31)
MIN_HALF_HOURS: Final = 40
"""The fewest valid half-hours a day needs before its daily maximum or mean is used."""
COASTAL_KM: Final = 3.0
SIX: Final = tuple(SIGNED_MW_LABELS)
EIGHT: Final = tuple(MVA_ONLY_LABELS)
COASTAL: Final = ("D9", "D2", "D4", "D19")
INLAND: Final = ("D11", "D10", "D22")
REPORTED_ONLY: Final = tuple(SHORT_HISTORY_LABELS)
SECONDS_PER_HALF_HOUR: Final = 1800.0
EXPECTED_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00560-modular-lag-encoder-capped-lags-and-block-dropout"
    / "work"
    / "expected"
)
"""The cache of the expected-weather table that experiment 560 wrote (read only)."""


# ----------------------------------------------------------------------------- groups


def sites() -> pl.DataFrame:
    """Return one row per Primary, in Primary-index order: `primary`, `time_series_id`, `d_label`, `coast_km`."""
    import e00397_prep as prep  # ty: ignore[unresolved-import]
    import e00735_diagnostic as diag  # ty: ignore[unresolved-import]

    table = diag.coast_distances(table=diag.primaries()).select(
        "time_series_id", "d_label", "coast_km"
    )
    out = (
        prep.primaries()
        .select("primary", "time_series_id")
        .with_columns(pl.col("time_series_id").cast(pl.Int64))
        .sort("primary")
        .join(table, on="time_series_id", how="left", validate="1:1")
    )
    assert out["d_label"].null_count() == 0
    return out


def group_indices(*, labels: list[str]) -> dict[str, list[int]]:
    """Return each reported group as the Primary indices it holds, for labels in Primary-index order."""
    index = {lab: i for i, lab in enumerate(labels)}
    return {
        "six signed-MW": [index[x] for x in SIX],
        "eight MVA-only": [index[x] for x in EIGHT],
        "4 coastal": [index[x] for x in COASTAL],
        "3 inland placebo": [index[x] for x in INLAND],
        "all 16": list(range(len(labels))),
        **{f"Primary {lab}": [index[lab]] for lab in labels},
    }


def coastal_by_distance(*, table: pl.DataFrame) -> list[str]:
    """Return the labels of the Primaries within `COASTAL_KM` of the coast (a check on `COASTAL`)."""
    return table.filter(pl.col("coast_km") <= COASTAL_KM)["d_label"].to_list()


# ----------------------------------------------------------------------------- windows


@dataclass(frozen=True)
class DayWindows:
    """Window membership per day code over the calendar's span, one boolean array per window."""

    first_code: int
    school: dict[str, np.ndarray]
    bank: dict[str, np.ndarray]
    school_block: np.ndarray
    """Index into `cols.blocks()` of the block whose school window holds the day, else -1."""
    bank_holiday: np.ndarray
    """The day code of the bank holiday whose window holds the day, else 0 (`the day` has its own)."""

    def n_days(self) -> int:
        """Return the number of days the arrays span."""
        return len(self.school_block)


_WINDOWS: list[DayWindows] = []


def day_windows() -> DayWindows:
    """Return the windows over the dense calendar's span (built once)."""
    if _WINDOWS:
        return _WINDOWS[0]
    cal = cols.dense_calendar()
    n = cal.last_code - cal.first_code + 1
    school = {name: np.zeros(n, dtype=bool) for name in SCHOOL_WINDOWS}
    block_of = np.full(n, -1, dtype=np.int32)

    def put(name: str, *, first: date, last: date, block: int) -> None:
        lo = max(cols.code_of(day=first) - cal.first_code, 0)
        hi = min(cols.code_of(day=last) - cal.first_code, n - 1)
        if lo > hi:
            return
        school[name][lo : hi + 1] = True
        block_of[lo : hi + 1] = block

    for b, block in enumerate(cols.blocks()):
        one = timedelta(days=1)
        resume = block.next_term_first
        weekend = set(block.first_weekend)
        put("7 to 4 days before", first=block.first - 7 * one, last=block.first - 4 * one, block=b)
        put("3 to 1 days before", first=block.first - 3 * one, last=block.first - one, block=b)
        for d in block.first_weekend:
            put("first weekend", first=d, last=d, block=b)
        day = block.first
        while day <= block.last:
            if day not in weekend:
                put("rest of block", first=day, last=day, block=b)
            day += one
        put("first 3 days after", first=resume, last=resume + 2 * one, block=b)
        put("4 to 7 days after", first=resume + 3 * one, last=resume + 6 * one, block=b)
    bank = {name: np.zeros(n, dtype=bool) for name in BANK_WINDOWS}
    holiday_of = np.zeros(n, dtype=np.int32)
    import holidays

    table = holidays.country_holidays("GB", subdiv="ENG", years=range(2021, 2028))
    for h in sorted(table):
        code = cols.code_of(day=h) - cal.first_code
        if not 0 <= code < n:
            continue
        spans = {
            "day before": [code - 1],
            "the day": [code],
            "day after": [code + 1],
            "extended weekend": [],
        }
        weekday = h.weekday()
        if weekday == 0:  # a Monday: the weekend before
            spans["extended weekend"] = [code - 2, code - 1]
        elif weekday == 4:  # a Friday: the weekend after
            spans["extended weekend"] = [code + 1, code + 2]
        for name, days in spans.items():
            for d in days:
                if 0 <= d < n:
                    bank[name][d] = True
                    holiday_of[d] = cal.first_code + code
    return _store(
        DayWindows(
            first_code=cal.first_code,
            school=school,
            bank=bank,
            school_block=block_of,
            bank_holiday=holiday_of,
        )
    )


def _store(windows: DayWindows) -> DayWindows:
    """Keep the windows for later calls and return them."""
    _WINDOWS.append(windows)
    return windows


def window_masks(*, codes: np.ndarray) -> dict[str, np.ndarray]:
    """Return a boolean mask over rows for every window, from the rows' local day codes.

    Rows outside the calendar's span are in no window and so count as ordinary days. The keys are
    `school: <window>` for the six school windows, `school window` for their union, `bank: <window>`
    for the four bank-holiday windows, `bank window` for their union, and `ordinary days` for rows in
    no school window and no bank-holiday window.
    """
    w = day_windows()
    codes = np.asarray(codes, dtype=np.int64)
    idx = codes - w.first_code
    inside = (idx >= 0) & (idx < w.n_days())
    safe = np.where(inside, idx, 0)
    out: dict[str, np.ndarray] = {}
    union_school = np.zeros(len(codes), dtype=bool)
    for name, flags in w.school.items():
        m = inside & flags[safe]
        out[f"school: {name}"] = m
        union_school |= m
    out[SCHOOL_WINDOW] = union_school
    union_bank = np.zeros(len(codes), dtype=bool)
    for name, flags in w.bank.items():
        m = inside & flags[safe]
        out[f"bank: {name}"] = m
        union_bank |= m
    out["bank window"] = union_bank
    out[ORDINARY] = ~union_school & ~union_bank
    return out


def block_index(*, codes: np.ndarray) -> np.ndarray:
    """Return the index (into `cols.blocks()`) of the school block whose window holds each row, else -1."""
    w = day_windows()
    idx = np.asarray(codes, dtype=np.int64) - w.first_code
    inside = (idx >= 0) & (idx < w.n_days())
    return np.where(inside, w.school_block[np.where(inside, idx, 0)], -1)


def bank_holiday_code(*, codes: np.ndarray) -> np.ndarray:
    """Return the day code of the bank holiday whose window holds each row, else 0."""
    w = day_windows()
    idx = np.asarray(codes, dtype=np.int64) - w.first_code
    inside = (idx >= 0) & (idx < w.n_days())
    return np.where(inside, w.bank_holiday[np.where(inside, idx, 0)], 0)


# ----------------------------------------------------------------------------- weather split


@dataclass
class DailyWeather:
    """Daily weather per Primary and local day code, on ERA5 and (after ERA5 ends) the expected table."""

    first_code: int
    tmax: np.ndarray
    """(Primaries, days) daily maximum 2 m temperature in degrees Celsius, NaN where unknown."""
    shortwave: np.ndarray
    """(Primaries, days) daily mean shortwave in watts per square metre."""
    precipitation_mm: np.ndarray
    """(Primaries, days) daily precipitation in millimetres."""
    proxy: np.ndarray
    """(days,) True where a day's values come from the expected-weather table, not ERA5."""


def _daily(
    *, half_hourly: np.ndarray, grid_us: np.ndarray, first_code: int, n_days: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (tmax, mean shortwave, precipitation mm) per Primary and day from (Primaries, grid, 3).

    The last axis holds temperature, shortwave, and precipitation rate in that order.
    """
    codes = cols.local_dates(stamp_us=grid_us).code.astype(np.int64) - first_code
    n_p = half_hourly.shape[0]
    tmax = np.full((n_p, n_days), np.nan, dtype=np.float32)
    sw = np.full((n_p, n_days), np.nan, dtype=np.float32)
    rain = np.full((n_p, n_days), np.nan, dtype=np.float32)
    starts = np.flatnonzero(np.r_[True, np.diff(codes) != 0])
    day = codes[starts]
    keep = (day >= 0) & (day < n_days)
    for p in range(n_p):
        a = np.asarray(half_hourly[p], dtype=np.float64)
        finite = np.isfinite(a)
        count = np.add.reduceat(finite[:, 0].astype(np.int64), starts)
        good = count >= MIN_HALF_HOURS
        top = np.fmax.reduceat(np.where(finite[:, 0], a[:, 0], -np.inf), starts)
        mean_sw = np.add.reduceat(np.where(finite[:, 1], a[:, 1], 0.0), starts) / np.maximum(
            np.add.reduceat(finite[:, 1].astype(np.int64), starts), 1
        )
        total = np.add.reduceat(np.where(finite[:, 2], a[:, 2], 0.0), starts) * (
            SECONDS_PER_HALF_HOUR
        )
        sel = good & keep
        tmax[p, day[sel]] = top[sel]
        sw[p, day[sel]] = mean_sw[sel]
        rain[p, day[sel]] = total[sel]
    return tmax, sw, rain


def daily_weather(*, expected_dir: Path, cache: Path | None = None) -> DailyWeather:
    """Return the daily weather of every Primary from 2019-09-01, ERA5 first and the proxy after ERA5 ends.

    Args:
        expected_dir: The cache folder of the expected-weather table (as `RowBuilder` uses).
        cache: An optional `.npz` file that stores the result.

    Returns:
        The daily temperature maximum, shortwave mean, and precipitation total.
    """
    if cache is not None and cache.exists():
        z = np.load(cache)
        return DailyWeather(
            first_code=int(z["first_code"]),
            tmax=z["tmax"],
            shortwave=z["shortwave"],
            precipitation_mm=z["precipitation_mm"],
            proxy=z["proxy"],
        )
    import e00397_prep as prep  # ty: ignore[unresolved-import]
    import e00418_prep as prep418  # ty: ignore[unresolved-import]
    import e00422_prep as prep422  # ty: ignore[unresolved-import]
    import e00560_data as d560  # ty: ignore[unresolved-import]

    table = prep.primaries().sort("primary")
    names = list(prep418.WEATHER)
    columns = [
        names.index("temperature_2m"),
        names.index("shortwave"),
        names.index("precipitation"),
    ]
    dense = prep418.era5_dense_cache()
    era5 = np.stack([dense[s][:, columns] for s in table["site"].to_list()])
    size = era5.shape[1]
    grid_us = prep422.GRID_START_US + np.arange(size, dtype=np.int64) * prep422.STEP_US
    first_code = int(cols.local_dates(stamp_us=grid_us[:1]).code[0])
    last_code = int(cols.local_dates(stamp_us=grid_us[-1:]).code[0])
    n_days = last_code - first_code + 1
    tmax, sw, rain = _daily(half_hourly=era5, grid_us=grid_us, first_code=first_code, n_days=n_days)
    proxy_cols, _ = d560.raw_table(source="ft", cache=expected_dir)
    expected = np.stack(
        [
            np.asarray(proxy_cols[:, :, 2]),
            np.asarray(proxy_cols[:, :, 0]),
            np.asarray(proxy_cols[:, :, 5]),
        ],
        axis=2,
    )
    e_grid = prep422.GRID_START_US + np.arange(expected.shape[1], dtype=np.int64) * prep422.STEP_US
    e_n = int(cols.local_dates(stamp_us=e_grid[-1:]).code[0]) - first_code + 1
    n_total = max(n_days, e_n)
    t2, s2, r2 = _daily(half_hourly=expected, grid_us=e_grid, first_code=first_code, n_days=n_total)
    pad = n_total - n_days
    tmax, sw, rain = (
        np.pad(a, ((0, 0), (0, pad)), constant_values=np.nan) for a in (tmax, sw, rain)
    )
    proxy = np.zeros(n_total, dtype=bool)
    after = cols.code_of(day=ERA5_LAST) - first_code
    use = np.arange(n_total) > after
    for era, alt in ((tmax, t2), (sw, s2), (rain, r2)):
        era[:, use] = alt[:, use]
    proxy[use] = True
    out = DailyWeather(
        first_code=first_code, tmax=tmax, shortwave=sw, precipitation_mm=rain, proxy=proxy
    )
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache,
            first_code=first_code,
            tmax=tmax,
            shortwave=sw,
            precipitation_mm=rain,
            proxy=proxy,
        )
    return out


def monthly_medians(*, weather: DailyWeather) -> np.ndarray:
    """Return (Primaries, 12 months, 2): the 2019-09 to 2025-12 median of the daily maximum and mean shortwave."""
    n_p, n_days = weather.tmax.shape
    codes = weather.first_code + np.arange(n_days)
    stamps = (codes.astype(np.int64) * 86_400_000_000) + 12 * 3_600_000_000
    month = cols.local_dates(stamp_us=stamps).month.astype(np.int64)
    lo = cols.code_of(day=MEDIAN_FIRST)
    hi = cols.code_of(day=MEDIAN_LAST)
    inside = (codes >= lo) & (codes <= hi)
    out = np.full((n_p, 12, 2), np.nan)
    for p in range(n_p):
        for m in range(12):
            sel = inside & (month == m + 1)
            for j, source in enumerate((weather.tmax[p], weather.shortwave[p])):
                v = source[sel]
                v = v[np.isfinite(v)]
                if len(v):
                    out[p, m, j] = np.median(v)
    return out


def weather_split(
    *, weather: DailyWeather, medians: np.ndarray, primary: np.ndarray, codes: np.ndarray
) -> np.ndarray:
    """Return the weather kind of each (Primary, day) row: 0 warm and sunny, 1 cold or wet, 2 mixed, -1 unknown.

    Args:
        weather: The daily weather.
        medians: The output of `monthly_medians`.
        primary: Primary index of each row.
        codes: Local day code of each row.

    Returns:
        An int8 array indexing `WEATHER_KINDS`, with -1 where the day has no weather.
    """
    codes = np.asarray(codes, dtype=np.int64)
    idx = codes - weather.first_code
    inside = (idx >= 0) & (idx < weather.tmax.shape[1])
    safe = np.where(inside, idx, 0)
    stamps = codes * 86_400_000_000 + 12 * 3_600_000_000
    month = cols.local_dates(stamp_us=stamps).month.astype(np.int64) - 1
    t = weather.tmax[primary, safe]
    s = weather.shortwave[primary, safe]
    r = weather.precipitation_mm[primary, safe]
    t_med = medians[primary, month, 0]
    s_med = medians[primary, month, 1]
    known = inside & np.isfinite(t) & np.isfinite(s) & np.isfinite(r) & np.isfinite(t_med)
    warm = known & (t >= t_med) & (s >= s_med) & (r < WET_MM)
    cold_or_wet = known & ((t < t_med) | (r >= WET_MM))
    out = np.full(len(codes), -1, dtype=np.int8)
    out[known] = 2
    out[cold_or_wet] = 1
    out[warm] = 0
    return out


def warm_sunny_wet_overlap_note() -> str:
    """Return the sentence that states how a day that is warm and sunny but wet is labelled."""
    return (
        "A day with a maximum and a mean shortwave at or above the month's median but at least 1 mm "
        "of rain is labelled cold or wet, because the plan's rule names precipitation of 1 mm or more as wet."
    )


def split_counts(*, kinds: np.ndarray, masks: dict[str, np.ndarray]) -> pl.DataFrame:
    """Return the number of Primary-days of each weather kind in each window."""
    rows = []
    for name, m in masks.items():
        row = {"window": name, "Primary-days": int(m.sum())}
        for i, kind in enumerate(WEATHER_KINDS):
            row[kind] = int((m & (kinds == i)).sum())
        row["unknown"] = int((m & (kinds == -1)).sum())
        rows.append(row)
    return pl.DataFrame(rows)
