# ruff: noqa: E501
"""Experiment 718: print the retrieval's weather look-ahead statement (KR8), and check a row table against it.

Usage: e00718_weather_lookahead.py [--origin 2026-03-10] [--primary D5] [--arm U] [--self-test]

The statement is experiment 713's statement for the 14 fixed rows, followed by one line for each
retrieval measure of the arm: what the measure reads, the issue time or valid time of each input
against the forecast origin, and the oldest and newest age it can choose. The retrieval reads two
kinds of weather. The target row's 14 days come from the ensemble mean of the 00Z run of the origin
day, which was issued before the native origin, and the pool's weather is the analysis valid at each
history block's own times, so a block of age 2 or more is valid before the contract origin.

`check_table` reads a row table and the retrieval arrays, and returns one PASS or FAIL line per
table. It passes when every chosen age is at least 2 (no block valid at or after the contract
origin), at most the pool's oldest age, and eligible under the visibility rule at its contract
origin, which is the rule that no power reading newer than 54 half-hours before the contract origin
is read. `--self-test` builds a tiny synthetic table and prints the statement and the PASS line.
`check_table` takes the finite power flags, not the builder's arrays, so that it stays independent of the builder.
"""

import argparse
import sys
from datetime import UTC, datetime, timedelta

import e00713_weather_lookahead as w713  # ty: ignore[unresolved-import]
import e00718_layout as lay
import e00718_retrieval as ret
import numpy as np

MEASURE_LINES = {
    "Z": [],
    "R": ["weather: the 14 nearest blocks of ages 2 to 91 by the weather distance"],
    "R-A": [
        "placebo: 14 random blocks of the same age classes as the weather rows of arm R (no weather read)"
    ],
    "P90": ["placebo: 14 uniformly random blocks of ages 2 to 91 (no weather read)"],
    "U": [
        "weather: 5 nearest blocks of ages 2 to 379 by the weather distance",
        "recency: the 5 youngest eligible blocks not yet chosen",
        "weekday: 4 youngest eligible blocks of the same weekday not yet chosen",
    ],
    "U-A": [
        "placebo: the age classes of arm U's rows, each drawn at random from the same age class"
    ],
    "U27": [
        "fixed: the 14 blocks of ages 2 to 15",
        "weather: 7 nearest blocks of ages 16 to 379",
        "weekday: 6 youngest blocks of the same weekday and ages 16 to 379",
    ],
    "U27-A": [
        "placebo: the age classes of arm U27's rows, each drawn at random from the same age class"
    ],
    "UX": [
        "calendar: the nearest blocks by day type and day offset from 25 December, from earlier winters (targets 20 December to 5 January)"
    ],
}

TARGET_WEATHER_LINE = (
    "Target-row weather of the distance: stages 2 and 3 and the forecast read the ensemble mean of the "
    "00Z run of the origin day, issued at 00:00 UTC, 9 h before the 09:00 native origin; stage 1 reads ERA5 and CAMS at "
    "the target half-hours, which is the lab shortcut of experiment 713's stage 1"
)
POOL_WEATHER_LINE = (
    "Pool weather of the distance: the analysis valid at each block's own half-hours. A block of age a "
    "ends 48 (a - 1) + 1 half-hours before the contract origin (49 for age 2), so every pool block, for ages 2 and up, "
    "is valid before the contract origin."
)


def statement(*, contract: datetime, primary: str, arm: str, lead_day: int = 0) -> list[str]:
    """Return the statement of experiment 713, then the lines of `arm`'s retrieval."""
    lines = w713.statement(contract=contract, primary=primary, lead_day=lead_day)
    lines += ["", f"Retrieval of arm {arm} ({lay.ROWS_PER_ARM[arm]} history rows):"]
    lines += [f"  - {m}" for m in MEASURE_LINES[arm]] or [
        "  - none: the 14 fixed rows of ages 2 to 15 (experiment 713's layout)"
    ]
    if arm != "Z":
        lines += [f"  - {TARGET_WEATHER_LINE}", f"  - {POOL_WEATHER_LINE}"]
    lines.append(
        "  - Every row's block must hold at least 24 half-hours of power that are visible (the reading index is at most the contract origin minus 54); the eligibility is read from the finite flags only."
    )
    return lines


def check_table(*, table: ret.RowTable, finite: np.ndarray) -> str:
    """Return the PASS or FAIL line for `table`, read against `finite` (P, G) bool, the grid's finite power flags.

    The check does not call the builder's eligibility function. It builds its own cumulative count of
    the flags and applies the rule of design section 2 with the constants of the independent
    recomputation: a row's age is at least 2 and at most the pool's oldest age, and every row that is
    not one of the fixed rows holds at least 24 readings at grid indices up to 54 half-hours before
    the contract origin. The fixed rows (ages 2 to 15 of `Z`, `U27`, and `U27-A`) are exempt from
    eligibility, as in design section 2.1.
    """
    import e00718_retrieval_reference as ref

    cs = np.concatenate(
        [np.zeros((finite.shape[0], 1), np.int64), np.cumsum(finite, axis=1, dtype=np.int64)],
        axis=1,
    )
    long_pool = {
        "U": 379,
        "U-A": 379,
        "U27": 379,
        "U27-A": 379,
        "UX": 10**9,
        "Z": lay.NEWEST_FIXED_MAX_AGE,
    }
    bad = n_rows = 0
    for pi, p in enumerate(table.primaries):
        for di, day in enumerate(table.days):
            c = ref.contract(day=int(day))
            max_age = min(long_pool.get(table.arm, 91), int(day))
            ages = table.ages[pi, di].reshape(-1).astype(np.int64)
            fixed = (table.measure[pi, di].reshape(-1) == lay.MEASURE_FIXED) | (table.arm == "Z")
            keep = ages != lay.ABSENT
            ages, fixed = ages[keep], fixed[keep]
            n_rows += len(ages)
            bad += int(((ages < 2) | (ages > max_age)).sum())
            lo = c - ref.STEPS * ages
            hi = np.minimum(lo + ref.STEPS, c - ref.LAST_VISIBLE_BEFORE + 1)
            count = cs[int(p)][np.clip(hi, 0, None)] - cs[int(p)][np.clip(lo, 0, None)]
            eligible = (lo >= 0) & (count >= ref.MIN_VISIBLE)
            bad += int((~eligible & ~fixed & (ages >= 2) & (ages <= max_age)).sum())
    verdict = "PASS" if bad == 0 else "FAIL"
    return f"{verdict}: arm {table.arm}: {n_rows:,} chosen rows, {bad} with an age below 2, beyond the pool, or not visible at the contract origin"


def self_test() -> list[str]:
    """Build the synthetic table of arm `U`, and return its statement and PASS line; a corrupted table must FAIL."""
    import e00718_check_batch as cb

    source = cb.synthetic_source()
    tables = cb.make_tables(source=source, arm="U")
    finite = np.isfinite(source.power_z.cpu().numpy())
    table = tables["era5"]
    line = check_table(table=table, finite=finite)
    assert line.startswith("PASS"), line
    broken = ret.RowTable(**{**vars(table), "ages": table.ages.copy()})  # ty: ignore[invalid-argument-type]
    broken.ages[0, 0, 0, 0] = 1
    assert check_table(table=broken, finite=finite).startswith("FAIL"), "an age of 1 was accepted"
    lines = [line, "negative control: a table with an age of 1 fails"]
    for arm in (
        "U27",
        "U27-A",
        "Z",
        "UX",
    ):  # regression: the fixed rows of ages 2 to 15 are exempt from eligibility
        arm_tables = cb.make_tables(source=source, arm=arm)
        for version, arm_table in arm_tables.items():
            result = check_table(table=arm_table, finite=finite)
            assert result.startswith("PASS"), (version, result)
        lines.append(
            f"regression: arm {arm}'s tables PASS, with fixed rows on blocks that are not eligible"
        )
    empty = finite.copy()
    empty[table.primaries[0]] = False
    assert check_table(table=table, finite=empty).startswith("FAIL"), (
        "rows on blocks without readings were accepted"
    )
    lines.append("negative control: a table whose blocks hold no readings fails")
    return lines


def main() -> int:
    """Print the statement."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin", default="2026-03-10")
    parser.add_argument("--primary", default="D5")
    parser.add_argument("--arm", default="U", choices=sorted(lay.ROWS_PER_ARM))
    parser.add_argument("--lead-day", type=int, default=1)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    contract = datetime.fromisoformat(args.origin).replace(tzinfo=UTC).replace(hour=12)
    assert contract - timedelta(hours=3) < contract
    print(
        "\n".join(
            statement(
                contract=contract, primary=args.primary, arm=args.arm, lead_day=args.lead_day - 1
            )
        )
    )
    if args.self_test:
        print("\n".join(self_test()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
