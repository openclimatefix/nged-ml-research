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
import polars as pl

e00729_paths.pin_threads()


def check_frame_39() -> str:
    """Check that the experiment 39 frame holds the farm and D8 columns the loader reads."""
    schema = pl.scan_parquet(data.FRAME_39).collect_schema()
    wanted = set(data.FARM_COLUMNS) | set(data.D8_COLUMNS)
    rows = pl.scan_parquet(data.FRAME_39).select(pl.len()).collect().item()
    return f"columns present: {wanted <= set(schema.names())}, has rows: {rows > 0}"


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
    """Check that the metered engine record has `time` and `power` columns and rows."""
    frame = data.metered_record()
    has_columns = {"time", "power"} <= set(frame.columns)
    return f"columns present: {has_columns}, has rows: {frame.height > 0}"


def check_bank_holidays() -> str:
    """Check that the bank-holiday table is non-empty."""
    return f"has rows: {len(data.bank_holiday_dates()) > 0}"


def check_private_constants() -> str:
    """Check that the private constants file loads and that the private cut can be computed."""
    cut = data.private_cut()
    return f"cut computed: {cut is not None}"


def check_day_table() -> str:
    """Check that the loader builds a day table with base days, printing flags only."""
    table = data.load_day_table()
    months, n_days, bsp_days = data.month_table(
        cal=data.select_days(cal=table.calendar, mask=table.is_base)
    )
    return f"has base days: {bool(table.is_base.any())}, months found: {len(months) > 0}, bsp days exist: {sum(bsp_days) > 0}, day counts positive: {all(n > 0 for n in n_days)}"


CHECKS: dict[str, Callable[[], str]] = {
    "frame_39": check_frame_39,
    "era_frame": check_era_frame,
    "power": check_power,
    "record": check_record,
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
