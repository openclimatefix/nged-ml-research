# ruff: noqa: E501
"""Experiment 715: print the weather look-ahead statement for F4's matrix and check it against stored data (KN12).

Usage: e00715_weather_lookahead.py [--origin 2026-03-10] [--primary D5] [--check-data]

For one contract origin (12:00 UTC on the given day) and one Primary, the script prints the statement of
experiment 713's host matrix of 15 rows by 48 half-hours (the target row and 14 history rows), then the
statement for the neighbour rows: up to 6 neighbour slots of 14 rows each, 84 rows at most. A neighbour
row holds the neighbour's power at the same 14 ages as the host's history rows (2 to 15 days before the
origin), the origin-time-visibility mask on the age 2 row, and no weather at all: a neighbour row carries
the neighbour's power and its signed-megawatt flag, never an ensemble or an analysis value. The host's
history rows hold weather valid at most 1 day before the contract origin, as in experiment 713.

With `--check-data` the script reads the stored run frame that the Candidate reads for the native origin
and prints one PASS or FAIL line, computed from that data: PASS when the latest `nwp_init_time` is at least 9
hours before the native origin and `power_fcst_init_time` is the native origin. The exit status is 1 on FAIL.
Experiment 713's `check_data` is the one implementation, so F4 and M* share the check.
"""

import argparse
import sys
from datetime import UTC, datetime, timedelta

import e00713_layout as lay
import e00713_weather_lookahead as w713
import e00715_neighbours as nb
import numpy as np

NEIGHBOUR_ROWS_PER_SLOT = 14


def neighbour_statement(*, contract: datetime, primary: str) -> list[str]:
    """Return the lines that state what a neighbour row holds and when it was recorded."""
    native = contract - timedelta(hours=3)
    newest_visible = int(lay.newest_visible_index(contract_idx=np.array([0]))[0])
    newest = w713.stamp(contract=contract, steps=newest_visible)
    ages = [int(a) for a in lay.history_ages()]
    lines = [
        "",
        f"Neighbour rows of {primary}: up to {nb.MAX_SLOTS} slots of {NEIGHBOUR_ROWS_PER_SLOT} rows, at most {nb.MAX_SLOTS * NEIGHBOUR_ROWS_PER_SLOT} rows, at the ages {ages[0]} to {ages[-1]} days.",
        "row | role | valid times | weather source | issue time against the forecast origin",
    ]
    for slot, age in enumerate(ages[:3], start=1):
        first = -lay.STEPS_PER_DAY * age
        lines.append(
            f"n{slot:2d} | neighbour power, age {age} | {w713.span(contract=contract, first_step=first)} | none: the row carries the neighbour's power, its signed-megawatt flag, and its relation to the host | "
            f"the newest reading is {contract - newest} before the contract origin"
        )
    lines += [
        "... | (the remaining ages of each slot follow the same pattern; each slot uses the host's ages)",
        f"Newest neighbour power timestamp read anywhere: {newest:%Y-%m-%d %H:%M} UTC, {contract - newest} before the contract origin and {native - newest} before the native origin; the age 2 row masks the same columns as the host's age 2 row.",
        "No neighbour row carries weather, a target-day power value, or any value recorded after the contract origin.",
    ]
    return lines


def main() -> int:
    """Print the statements and, with `--check-data`, the data check."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin", default="2026-03-10")
    parser.add_argument("--primary", default="D5")
    parser.add_argument("--lead-day", type=int, default=1)
    parser.add_argument("--check-data", action="store_true")
    args = parser.parse_args()
    day = datetime.fromisoformat(args.origin).replace(tzinfo=UTC)
    contract = day.replace(hour=12)
    print(
        "\n".join(
            w713.statement(contract=contract, primary=args.primary, lead_day=args.lead_day - 1)
        )
    )
    print("\n".join(neighbour_statement(contract=contract, primary=args.primary)))
    if args.check_data:
        ok, lines = w713.check_data(contract=contract)
        print("\n".join(lines))
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
