# ruff: noqa: E501
"""Smoke test of every real file that experiment 729 reads. Read-only and silent about values.

It opens each file once and prints one line per input: a column-presence flag and a row-count flag.
It prints no power value, no date, no farm or site name, no engine size, and no series id. It is
not a pytest module. Run it once by hand, only when the code review's checklist says so:

    SLOT_CPU=<n> ml-experiments/00729-*/run.sh --go tests/smoke_real_files.py

Exit status 0 means every input opened with the columns the code reads.
"""

import sys
import warnings
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
        f"precipitation never negative: {bool((frame['precipitation'] >= 0.0).all())}, "
        f"precipitation unit is kg per square metre per second (mean annual total of 300 to 1,500 mm): "
        f"{300.0 <= float(frame['precipitation'].cast(pl.Float64).to_numpy().mean()) * data.MM_PER_HOUR_PER_KG_M2_S * 8766.0 <= 1500.0}"
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


CLEAR_DAY_CSI = 0.7
"""A base day is clear when D8's mean CAMS clear-sky index over its sun-up half-hours is at least this."""
MIN_S2_OUTPUT_SHARE = 0.99
"""At least this share of clear base days in S2 solar-world segments must hold solar output.

The expected share is 1.0: the sky is present on every clear day, so a day of source S2 has zero
output only if the sky rule breaks. 0.99 fails a rule that silences even a few percent of days."""
N_S2_DRAWS = 8
MAX_NO_SKY_SHARE = 0.02
"""At most this share of all base days may have no farm with a satellite sky."""


def clear_days(*, cal: object) -> np.ndarray:
    """Return a mask of the base days whose mean CAMS clear-sky index over sun-up half-hours is high."""
    sun = cal.cosz > 0  # ty: ignore[unresolved-attribute]
    csi = np.where(sun & np.isfinite(cal.csi), cal.csi, np.nan)  # ty: ignore[unresolved-attribute]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(csi, axis=1)
    return np.isfinite(mean) & (mean >= CLEAR_DAY_CSI)


def check_s2_solar_on_clear_days() -> str:
    """Draw source S2 solar on the real base days and fail if a clear day has zero output.

    Returns:
        A line of flags (booleans only).

    Raises:
        AssertionError: When a clear base day has no farm with a satellite sky, or an S2 draw leaves
            a clear base day with zero output.
    """
    import e00729_sim as sim

    days, _ = data.load_closure_days()
    table = data.load_day_table(closure_days=days)
    cal = data.select_days(cal=table.calendar, mask=table.is_base)
    clear = clear_days(cal=cal)
    sky_any = cal.farms.sky_ok.any(axis=0)
    assert sky_any[clear].all(), "a clear base day has no farm with a satellite sky"
    no_sky_share = float(1.0 - sky_any.mean())
    assert no_sky_share <= MAX_NO_SKY_SHARE, "too many base days have no farm with a sky"
    drawn = zero_clear = 0
    for seed in range(200):
        out, _, source, _ = sim.draw_solar(
            rng=np.random.default_rng(seed),
            farms=cal.farms,
            cal_days=np.arange(len(cal.dates)),
            record={},
        )
        if source != "S2":
            continue
        drawn += 1
        zero_clear += int((out.max(axis=1)[clear] == 0).sum())
        if drawn >= N_S2_DRAWS:
            break
    assert drawn == N_S2_DRAWS, "too few S2 draws"
    assert zero_clear == 0, "an S2 draw left a clear base day with zero solar output"
    return (
        f"has clear base days: {bool(clear.any())}, share of base days with no farm sky is at most {MAX_NO_SKY_SHARE}: {no_sky_share <= MAX_NO_SKY_SHARE}, every clear base day has a farm with a sky: {bool(sky_any[clear].all())}, "
        f"no S2 draw has a clear base day with zero output: {zero_clear == 0}"
    )


def check_bank_s2_solar() -> str:
    """Check the stored test bank: clear base days of S2 solar-world segments hold solar output.

    Prints a flag that the bank is absent before the first bank build, and fails when a bank
    exists and the share of such days with output is under `MIN_S2_OUTPUT_SHARE`. Run it once
    before `banks` (after moving the first run's outputs away) and again after `banks`.

    Returns:
        A line of flags (booleans only).

    Raises:
        AssertionError: When the share of days is under `MIN_S2_OUTPUT_SHARE`.
    """
    import e00729_banks as banks
    import e00729_stages as stages

    path = stages.bank_path(kind="test")
    if not path.exists():
        return "test bank present: False"
    bank = banks.load_bank(path=path)
    with_output = total = 0
    for seg in bank.segments:
        if seg.source != "S2" or seg.world not in ("solar", "both"):
            continue
        days = (seg.cal_index >= 0) & clear_days(cal=seg)
        total += int(days.sum())
        with_output += int((seg.solar[days].max(axis=1) > 0).sum())
    assert total, "no clear base day in an S2 solar-world segment"
    share = with_output / total
    assert share >= MIN_S2_OUTPUT_SHARE, "S2 solar-world days are silent"
    return (
        "test bank present: True, share of clear S2 solar-world base days with output is at least "
        f"{MIN_S2_OUTPUT_SHARE}: {share >= MIN_S2_OUTPUT_SHARE}"
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
    "s2_solar_clear_days": check_s2_solar_on_clear_days,
    "bank_s2_solar": check_bank_s2_solar,
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
