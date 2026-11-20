# ruff: noqa: E501
"""Experiment 713: print the weather look-ahead statement for one origin and one Primary (gate 4).

Usage: e00713_weather_lookahead.py [--origin 2026-03-10] [--primary D5] [--check-data]

For one contract origin (12:00 UTC on the given day) and one Primary, the script prints, for each
row of the 15 x 48 matrix, the weather source, the issue time of the weather against the forecast
origin, and the span of valid times, and it prints the newest power timestamp against the origin. The
layout is computed from `e00713_layout` alone, so the statement needs no fit and reads no power.

With `--check-data` the script also reads the `nwp_init_time` and `power_fcst_init_time` columns of the
ensemble summary file of the origin's run (one small file; the first 5 rows of the Primary) and
prints them next to the statement.
The origin-time assertion of the design is `nwp_init_time <= o - 9 h`
with `o` the native origin.
"""

import argparse
import sys
from datetime import UTC, datetime, timedelta

import e00713_layout as lay
import numpy as np

ERA5_LATENCY_NOTE = (
    "ERA5 and CAMS analysis valid at the token's own time; no issue time (a reanalysis is published "
    "about 5 days after its valid time, so this source is a lab shortcut for ages 2 to 5)"
)


def stamp(*, contract: datetime, steps: int) -> datetime:
    """Return the instant `steps` half-hours after `contract`."""
    return contract + timedelta(minutes=30 * steps)


def span(*, contract: datetime, first_step: int) -> str:
    """Return the first and last valid time of a row whose column 0 is `first_step` half-hours after `contract`."""
    lo = stamp(contract=contract, steps=first_step)
    hi = stamp(contract=contract, steps=first_step + lay.STEPS_PER_DAY - 1)
    return f"{lo:%Y-%m-%d %H:%M} to {hi:%Y-%m-%d %H:%M} UTC"


def statement(*, contract: datetime, primary: str, lead_day: int = 0) -> list[str]:
    """Return the lines of the statement for one contract origin, Primary, and lead day (zero-based)."""
    native = contract - timedelta(hours=3)
    run = contract.replace(hour=0)
    lines = [
        f"Contract origin {contract:%Y-%m-%d %H:%M} UTC; native origin {native:%Y-%m-%d %H:%M} UTC; Primary {primary}; lead day {lead_day + 1}",
        f"Ensemble run used: {run:%Y-%m-%d %H:%M} UTC (00Z), issued {native - run} before the native origin and {contract - run} before the contract origin",
        "",
        "row | role | valid times | weather source | issue time against the forecast origin",
    ]
    lines.append(
        f"  0 | target, lead day {lead_day + 1} | {span(contract=contract, first_step=lay.STEPS_PER_DAY * lead_day)} | "
        f"ECMWF ENS mean and spread, run {run:%Y-%m-%d} 00Z; B1 columns from trailing windows of the same run and CAMS of the last complete UTC day before the origin | "
        f"issued {native - run} before the native origin (before: yes)"
    )
    newest_visible = lay.newest_visible_index(contract_idx=np.array([0]))[0]
    for slot, age in enumerate(lay.history_ages(), start=1):
        first = -lay.STEPS_PER_DAY * int(age)
        valid_end = stamp(contract=contract, steps=first + lay.STEPS_PER_DAY - 1)
        available = valid_end + timedelta(days=5)
        live = (
            "could be built in a live service"
            if available <= contract
            else "could NOT be built in a live service (lab shortcut)"
        )
        lines.append(
            f"{slot:3d} | history, age {age} | {span(contract=contract, first_step=first)} | {ERA5_LATENCY_NOTE.split(';', maxsplit=1)[0]} | "
            f"valid {contract - valid_end} before the origin; {live}"
        )
    newest = stamp(contract=contract, steps=int(newest_visible))
    masked = lay.masked_columns_of_newest_row().tolist()
    lines += [
        "",
        f"Newest power timestamp read anywhere in the matrix: {newest:%Y-%m-%d %H:%M} UTC, {contract - newest} before the contract origin and {native - newest} before the native origin (the 24 h cap).",
        f"In the age-2 row the visible columns are 0 to {masked[0] - 1} and the masked columns are {masked[0]} to {masked[-1]}; rows of ages 3 to 15 are fully visible.",
        "The target row carries no power at all (the learned masked token).",
        "No history token carries a weather value valid after the contract origin: the newest history token is valid 1 day before it.",
    ]
    return lines


def check_data(*, contract: datetime, primary: str) -> list[str]:
    """Read the run's summary file and return the stored `nwp_init_time` and `power_fcst_init_time` values."""
    import e00397_prep as prep  # ty: ignore[unresolved-import]
    import e00578_common as k578  # ty: ignore[unresolved-import]
    import e00667_extend as ext  # ty: ignore[unresolved-import]
    import polars as pl

    ext.redirect()
    run = contract.replace(hour=0)
    path = prep.SUMMARY_DIR / f"init={run:%Y-%m-%d}.parquet"
    sites = k578.sites()
    ts = sites.filter(pl.col("label") == primary)["time_series_id"].item()
    frame = pl.scan_parquet(path).filter(pl.col("time_series_id") == ts).head(5).collect()
    cols = [
        c for c in ("nwp_init_time", "power_fcst_init_time", "valid_time") if c in frame.columns
    ]
    return [f"Stored file {path.name}, first rows of {primary}:", str(frame.select(cols))]


def main() -> int:
    """Print the statement."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin", default="2026-03-10")
    parser.add_argument("--primary", default="D5")
    parser.add_argument("--lead-day", type=int, default=1)
    parser.add_argument("--check-data", action="store_true")
    args = parser.parse_args()
    day = datetime.fromisoformat(args.origin).replace(tzinfo=UTC)
    contract = day.replace(hour=12)
    print("\n".join(statement(contract=contract, primary=args.primary, lead_day=args.lead_day - 1)))
    if args.check_data:
        print("\n".join(check_data(contract=contract, primary=args.primary)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
