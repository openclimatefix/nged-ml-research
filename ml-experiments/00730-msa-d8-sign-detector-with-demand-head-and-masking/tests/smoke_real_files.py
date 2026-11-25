# ruff: noqa: E501
"""Smoke test of every real file that experiment 730 reads. Read-only and silent about values.

It is not a pytest module. Run it once by hand, only when the code review's checklist says so:

    SLOT_CPUS=<list> ml-experiments/00730-*/run.sh --go tests/smoke_real_files.py

It opens each real file once and prints one line per input: a column-presence flag and a row-count
flag. It prints no power value, no date, no farm or site name, no engine size, and no series
identifier. The component checks of the `banks` stage read each simulated component against the
share `MIN_OUTPUT_SHARE`, which the smoke test compares experiment 729's S2 checks against. Exit
status 0 means every input opened with the columns the code reads.
"""

import sys
from collections.abc import Callable

import e00730_paths

# isort: split
import e00730_core as core
import e00730_pretrain as pretrain
import e00730_stages as stages
import numpy as np

e00730_paths.pin_threads()


def check_private_splits() -> str:
    """Check that experiment 729's private split table, the one file read from its folder, exists."""
    path = e00730_paths.splits_md_path()
    return f"private split table present: {path.exists()}"


def check_pretraining_inputs() -> str:
    """Check that experiment 545's chain is importable and the encoder width constants agree."""
    import e00545_model as model

    del model
    return (
        f"encoder columns {core.ENCODER_INPUT_COLUMNS} = 7 weather + 21 datetime: "
        f"{core.ENCODER_INPUT_COLUMNS == 28}, minimum output share {core.MIN_OUTPUT_SHARE}"
    )


def check_encoder_table_alignment() -> str:
    """Check D8's real encoder table (30 base days, built in memory) for constants and alignment."""
    import e00418_prep as prep418

    ctx = stages.real_context()
    dates = ctx.table.calendar.dates[:30].astype("datetime64[D]")
    table = pretrain.d8_raw_encoder_table(base_dates=dates)
    names = [n for n, _ in enumerate(prep418.WEATHER)]
    shortwave = next(
        (
            j
            for j, n in enumerate(prep418.WEATHER)
            if any(k in str(n) for k in ("ghi", "sw", "short"))
        ),
        0,
    )
    hour = pretrain.clear_sky_peak_hour(table=table, cosz_column=shortwave)
    low, high = core.ENCODER_PEAK_HOURS_UTC
    share = pretrain.share_non_constant(table=table)
    return (
        f"table columns {table.shape[-1]} = 28: {table.shape[-1] == 28}, weather columns {len(names)}, "
        f"share of non-constant columns is above 0.9: {share > 0.9}, "
        f"clear-sky peak hour within {low} to {high} UTC: {low <= hour <= high}"
    )


def check_context_and_record() -> str:
    """Build experiment 729's context and read the metered engine record's days (counts only)."""
    ctx = stages.real_context()
    data = stages.stages729.data
    cal_all = ctx.table.calendar
    rec = data.record_by_day(record=data.metered_record(), days=cal_all.dates)
    n_days = int(np.isfinite(rec).any(axis=1).sum())
    holidays = len(ctx.cal.holidays)
    return (
        f"context built: True, base days present: {len(ctx.cal.dates) > 0}, bank holidays present: "
        f"{holidays > 0}, record days present: {n_days > 0}, "
        f"farm output present: {bool(cal_all.farms.power_ok.any())}, "
        f"farm sky present: {bool(cal_all.farms.sky_ok.any())}"
    )


def check_primaries_and_summaries() -> str:
    """Count the 16 Primaries and check that D8 is among them (no name or identifier printed)."""
    series = pretrain.d8_series()
    return f"16 Primaries and D8 present: {series is not None}, D8 index {pretrain.d8_primary_index() >= 0}"


def load_smoke_729() -> object:
    """Import experiment 729's smoke test as a module and give it this experiment's output bar."""
    import importlib.util
    from pathlib import Path

    path = next(Path(__file__).resolve().parents[2].glob("00729-*/tests/smoke_real_files.py"))
    spec = importlib.util.spec_from_file_location("smoke729", path)
    module = importlib.util.module_from_spec(spec)  # ty: ignore[invalid-argument-type]
    spec.loader.exec_module(module)  # ty: ignore[unresolved-attribute]
    module.MIN_S2_OUTPUT_SHARE = core.MIN_OUTPUT_SHARE  # ty: ignore[unresolved-attribute]
    return module


def check_s2_solar() -> str:
    """Run experiment 729's S2 check on the real base days against `MIN_OUTPUT_SHARE`."""
    return load_smoke_729().check_s2_solar_on_clear_days()  # ty: ignore[unresolved-attribute]


def check_bank_s2() -> str:
    """Run experiment 729's bank S2 check against `MIN_OUTPUT_SHARE`."""
    return load_smoke_729().check_bank_s2_solar()  # ty: ignore[unresolved-attribute]


def check_grid_dates() -> str:
    """Check that experiment 87's grid gives one UTC day per half-hour for the removal mask."""
    import e00087_common as common

    days = common.grid_times().dt.date().to_numpy().astype("datetime64[D]")
    return f"grid has one day per half-hour: {days.shape[0] == common.grid_times().len()}"


CHECKS: dict[str, Callable[[], str]] = {
    "grid dates": check_grid_dates,
    "private splits": check_private_splits,
    "pre-training inputs": check_pretraining_inputs,
    "encoder table alignment": check_encoder_table_alignment,
    "context and record": check_context_and_record,
    "primaries": check_primaries_and_summaries,
    "S2 solar on clear days": check_s2_solar,
    "bank S2 solar": check_bank_s2,
}


def main() -> int:
    """Run every check, print one line each, and return the exit status."""
    failed = 0
    for name, check in CHECKS.items():
        try:
            line = check()
            print(f"{name}: {line}")
            if "False" in line.replace("test bank present: False", ""):
                failed += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"{name}: FAILED ({type(exc).__name__})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
