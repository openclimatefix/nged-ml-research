# ruff: noqa: E501
"""Smoke test of every real file that experiment 729 reads. Read-only and silent about values.

It opens each file once and prints one line per input: a column-presence flag and a row-count flag.
It prints no power value, no date, no farm or site name, no engine size, and no series id. It is
not a pytest module. Run it once by hand, only when the code review's checklist says so:

    SLOT_CPU=<n> ml-experiments/00729-*/run.sh --go tests/smoke_real_files.py

Exit status 0 means every input opened with the columns the code reads.
"""

import sys
from collections.abc import Callable

import e00729_paths

# isort: split
import e00729_core as core
import e00729_data as data
import numpy as np
import polars as pl

e00729_paths.pin_threads()


def check_frame_39() -> str:
    """Check that the experiment 39 frame holds the farm and D8 columns the loader reads."""
    schema = pl.scan_parquet(data.FRAME_39).collect_schema()
    wanted = set(data.FARM_COLUMNS) | set(data.D8_COLUMNS)
    rows = pl.scan_parquet(data.FRAME_39).select(pl.len()).collect().item()
    return f"columns present: {wanted <= set(schema.names())}, has rows: {rows > 0}"


def check_era5_weather() -> str:
    """Check D8's ERA5 wind speed and precipitation: columns, plausible ranges, and base-year coverage."""
    import datetime as dt

    frame = (
        pl.scan_parquet(data.ERA5_FEATURES)
        .filter(
            (pl.col("time_series_id") == core.D8_SERIES_ID)
            & (pl.col("valid_time").dt.minute() == 0)
        )
        .select("valid_time", "wind_speed_10m", "precipitation")
        .collect()
    )
    first = dt.datetime(2019, 9, 1, tzinfo=dt.UTC)
    last = dt.datetime.fromisoformat(core.REAL_UP_TO + "T23:00:00+00:00")
    wanted = int((last - first).total_seconds() // 3600) + 1
    span = frame.filter((pl.col("valid_time") >= first) & (pl.col("valid_time") <= last))
    return (
        f"has rows: {frame.height > 0}, every whole hour of the base years present: {span.height == wanted}, "
        f"no null: {frame.null_count().sum_horizontal().item() == 0}, "
        f"wind speed within 0 to 40 m/s: {bool(frame['wind_speed_10m'].is_between(0.0, 40.0).all())}, "
        f"precipitation never negative: {bool((frame['precipitation'] >= 0.0).all())}"
    )


def check_era_frame() -> str:
    """Check that experiment 697's era frame holds `start` and `era` and the era labels."""
    frame = pl.scan_parquet(data.ERA_FRAME).select("start", "era").collect()
    labels = set(frame["era"].unique().to_list())
    return f"has rows: {frame.height > 0}, net era present: {data.ERA_NET in labels}"


def check_power() -> str:
    """Check that the cleaned power store has D8 and the BSP A meter, with rows for each."""
    from studies.power import scan_power

    counts = (
        scan_power()
        .filter(pl.col("time_series_id").is_in([core.D8_SERIES_ID, core.BSP_SERIES_ID]))
        .group_by("time_series_id")
        .agg(n=pl.len())
        .collect()
    )
    return f"both series have rows: {counts.height == 2 and bool((counts['n'] > 0).all())}"


def check_record() -> str:
    """Check the metered engine record: its columns, a `units` column, and more than a month of rows."""
    frame = data.metered_record()
    has_columns = {"time", "power", "units"} <= set(frame.columns)
    return (
        f"columns present: {has_columns}, more than a month of half-hours: {frame.height > 48 * 31}"
    )


def check_clear_sky_peak() -> str:
    """Check that D8's clear-sky irradiance peaks within 11:00 to 13:00 UTC (the period-end shift)."""
    frame = pl.scan_parquet(data.FRAME_39).filter(pl.col("site") == data.D8_SITE)
    by_hour = (
        frame.group_by(pl.col("time").dt.hour().alias("hour"))
        .agg(pl.col("ghi_u_clear").mean())
        .collect()
        .sort("ghi_u_clear")
    )
    peak = int(by_hour["hour"][-1])
    return f"clear-sky peak hour is within 11 to 13 UTC: {11 <= peak <= 13}"


def check_bank_holidays() -> str:
    """Check that the bank-holiday table is non-empty."""
    return f"has rows: {len(data.bank_holiday_dates()) > 0}"


def check_private_constants() -> str:
    """Check that the private constants file loads and that the private cut can be computed."""
    cut = data.private_cut()
    return f"cut computed: {cut is not None}"


def split_flags(*, closure_days: frozenset) -> tuple[bool, bool, bool]:
    """Return whether the derived split equals the private file, whether a farm joins, and the base."""
    import e00729_stages as stages

    table = data.load_day_table(closure_days=closure_days)
    cal = data.select_days(cal=table.calendar, mask=table.is_base)
    months, n_days, bsp_days = data.month_table(cal=cal)
    derived = core.month_splits(months=months, n_days=n_days, bsp_days=bsp_days)
    private = e00729_paths.data_root() / "PRIVATE" / "splits.md"
    same = private.exists() and stages.parse_private_splits(path=private) == derived
    return same, bool(table.calendar.farms.power_ok.any()), bool(table.is_base.any())


def check_day_table() -> str:
    """Check the day table and print the split flags without and with the closure days (booleans)."""
    without, farm_joined, has_base = split_flags(closure_days=frozenset())
    days, status = data.load_closure_days()
    table = data.load_day_table(closure_days=days)
    on_base = bool(
        np.isin(
            table.calendar.dates[table.is_base], np.array(sorted(days), dtype="datetime64[D]")
        ).any()
    )
    with_closure, _, _ = split_flags(closure_days=days)
    return (
        f"has base days: {has_base}, a farm joins with data: {farm_joined}, "
        f"code split equals the private file without closure days: {without}, "
        f"closure log loaded: {status == 'loaded'}, a closure day falls on a base day: {on_base}, "
        f"code split equals the private file with closure days: {with_closure}"
    )


CHECKS: dict[str, Callable[[], str]] = {
    "frame_39": check_frame_39,
    "era5_weather": check_era5_weather,
    "era_frame": check_era_frame,
    "power": check_power,
    "record": check_record,
    "clear_sky_peak": check_clear_sky_peak,
    "bank_holidays": check_bank_holidays,
    "private_constants": check_private_constants,
    "day_table": check_day_table,
}


def main() -> int:
    """Run every check, print one line each, and return the exit status."""
    failed = 0
    for name, check in CHECKS.items():
        try:
            print(f"{name}: {check()}")
        except Exception as error:  # noqa: BLE001
            failed += 1
            print(f"{name}: FAILED with {type(error).__name__}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
