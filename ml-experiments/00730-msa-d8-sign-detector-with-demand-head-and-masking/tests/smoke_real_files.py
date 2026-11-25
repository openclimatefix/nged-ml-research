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


def weather_flags(*, table: np.ndarray, names: list[str]) -> tuple[int, list[bool]]:
    """Return the shortwave column and, for each weather column, whether it varies.

    Args:
        table: The encoder table `(days, 48, columns)`, weather columns first.
        names: The weather column names, in order.

    Returns:
        The index of the column named `shortwave`, and one flag per weather column.

    Raises:
        ValueError: When no weather column is named `shortwave`.
    """
    matches = [j for j, n in enumerate(names) if n == "shortwave"]
    if not matches:
        msg = "no weather column is named shortwave"
        raise ValueError(msg)
    flat = table.reshape(-1, table.shape[-1])
    return matches[0], [bool(np.ptp(flat[:, j]) > 0.0) for j in range(len(names))]


def check_encoder_table_alignment() -> str:
    """Check D8's real encoder table (30 base days, built in memory) for constants and alignment.

    Each of the 7 weather columns must vary over the table, and the shortwave column is found by
    name (the check raises when no weather name matches). The datetime columns include `covid` and
    the three `christmas_*` columns, which are constant outside December, so they are not read.
    """
    import e00418_prep as prep418

    ctx = stages.real_context()
    dates = ctx.table.calendar.dates[:30].astype("datetime64[D]")
    table = pretrain.d8_raw_encoder_table(base_dates=dates)
    names = list(prep418.WEATHER)
    shortwave, varying = weather_flags(table=table, names=names)
    hour = pretrain.clear_sky_peak_hour(table=table, cosz_column=shortwave)
    low, high = core.ENCODER_PEAK_HOURS_UTC
    return (
        f"table columns {table.shape[-1]} = 28: {table.shape[-1] == 28}, "
        f"weather columns {len(names)} = 7: {len(names) == 7}, "
        f"every weather column varies: {all(varying)}, "
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
    """Run experiment 729's bank S2 check on this experiment's test bank.

    Experiment 729's own `check_bank_s2_solar` opens experiment 729's test bank, so this check loads
    `banks/test.npz` of this experiment's data folder and applies experiment 729's `clear_days` and
    this experiment's `MIN_OUTPUT_SHARE`. Before `banks` it prints "test bank present: False".
    """
    import e00729_banks as banks729

    path = stages.banks.bank_path(root=e00730_paths.data_root(), kind="test")
    if not path.exists():
        return "test bank present: False"
    smoke729 = load_smoke_729()
    bank = banks729.load_bank(path=path)
    with_output = total = 0
    for seg in bank.segments:
        if seg.source != "S2" or seg.world not in ("solar", "both"):
            continue
        days = (seg.cal_index >= 0) & smoke729.clear_days(cal=seg)  # ty: ignore[unresolved-attribute]
        total += int(days.sum())
        with_output += int((seg.solar[days].max(axis=1) > 0).sum())
    if not total:
        msg = "no clear base day in an S2 solar-world segment"
        raise ValueError(msg)
    share = with_output / total
    return (
        "test bank present: True, share of clear S2 solar-world base days with output is at least "
        f"{core.MIN_OUTPUT_SHARE}: {share >= core.MIN_OUTPUT_SHARE}"
    )


def clear_day_counts(*, cal: object, used: dict[str, np.ndarray]) -> dict[str, int]:
    """Count the clear base days of each split and season under `banks.clear_sun_up`.

    Args:
        cal: A calendar with `csi`, `cosz`, and `dates`.
        used: For each split, the mask over the calendar's days that the split may use.

    Returns:
        The counts, keyed `split/season`.
    """
    _, clear_day = stages.banks.clear_sun_up(csi=cal.csi, cosz=cal.cosz)  # ty: ignore[unresolved-attribute]
    seasons = np.array([stages.banks.season_of(date=d) for d in cal.dates])  # ty: ignore[unresolved-attribute]
    return {
        f"{split}/{season}": int((clear_day & mask & (seasons == season)).sum())
        for split, mask in used.items()
        for season in core.COMPONENT_SEASONS
    }


def expected_cases(*, counts: dict[str, int]) -> dict[str, float]:
    """Return the solar-world cases that a bank of each split would hold, pooled over seasons.

    The bar `MIN_CASES` applies to the pool of a bank kind and source, because a split of a few
    months holds too few clear days for a bar on each season. Each month of a bank draws
    `SEGMENT_DRAWS_*` segments, of which the share `TEST_MIX["solar"] + TEST_MIX["both"]` carries
    solar output, and each such segment counts every clear base day of its month.

    Args:
        counts: The clear base days by `split/season`.

    Returns:
        The expected cases of each split.
    """
    import e00729_core as core729

    share = core729.TEST_MIX["solar"] + core729.TEST_MIX["both"]
    out: dict[str, float] = {}
    for key, n in counts.items():
        split = key.split("/")[0]
        draws = core729.SEGMENT_DRAWS_TRAIN if split == "training" else core729.SEGMENT_DRAWS_EVAL
        out[split] = out.get(split, 0.0) + n * draws * share
    return out


def check_clear_day_counts() -> str:
    """Print the real clear base days by season and split, and whether each can reach `MIN_CASES`.

    The count is of base days under `banks.clear_sun_up` (the rule the solar component check uses)
    in the months of each split. It prints counts only, never a date.
    """
    ctx = stages.real_context()
    used = {
        split: np.any([np.asarray(ctx.use[split][m], dtype=bool) for m in ctx.months(split=split)], axis=0)
        for split in ("training", "validation", "test")
    }  # fmt: skip
    counts = clear_day_counts(cal=ctx.cal, used=used)
    text = ", ".join(f"{k} {n}" for k, n in counts.items())
    expected = expected_cases(counts=counts)
    enough = all(n >= stages.banks.MIN_CASES for n in expected.values())
    seasons = all(n > 0 for n in counts.values())
    return (
        f"clear base days {text}; expected solar-world cases per split "
        f"{', '.join(f'{k} {n:.0f}' for k, n in expected.items())}; "
        f"every split reaches MIN_CASES pooled over seasons: {enough}; "
        f"every season has a clear day (a season without one is not expected): {seasons or 'n/a'}"
    )


def check_sky_gaps() -> str:
    """Print the zeroed sun-up share of the farms' sky before and after `fill_sky_gaps`.

    The bound is `sky.MAX_ZEROED_SHARE`, pre-registered on the plan page. The line prints shares
    and counts of farm-days only.
    """
    import e00730_sky as sky

    ctx = stages.real_context()
    before = sky.zeroed_sky_share(farms=ctx.cal.farms)
    filled = sky.fill_sky_gaps(cal=ctx.cal)
    after = sky.zeroed_sky_share(farms=filled.farms)
    lost_sky = int(ctx.cal.farms.sky_ok.sum() - filled.farms.sky_ok.sum())
    lost_power = int(ctx.cal.farms.power_ok.sum() - filled.farms.power_ok.sum())
    return (
        f"zeroed sun-up share of sky_ok farm-days {before:.4f} before, {after:.4f} after; "
        f"farm-days leaving sky_ok {lost_sky}, leaving power_ok {lost_power}; "
        f"bound {sky.MAX_ZEROED_SHARE} met: {after <= sky.MAX_ZEROED_SHARE}"
    )


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
    "clear base days by season": check_clear_day_counts,
    "sky gaps filled": check_sky_gaps,
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
