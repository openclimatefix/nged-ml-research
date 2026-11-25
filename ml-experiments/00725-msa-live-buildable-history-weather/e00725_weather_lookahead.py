# ruff: noqa: E501
"""Experiment 725: print the weather look-ahead statement for one origin (check KL8).

Usage: e00725_weather_lookahead.py [--origin 2026-03-10] [--rule L|Z|E] [--check-data]

For one contract origin (12:00 UTC on the given day), the script prints one line for each run of
columns that read one source in a row of the 14 x 48 history matrix: the age, the columns, the
weather group, the source, the span of valid times, and the margin in hours between the latest value
of the span and the time at which a live service holds that value (a positive margin means the value
was in hand at the native origin). The lines come from `e00725_rule.reference_table`, which uses
`datetime` arithmetic alone. With `--check-data` the script also runs experiment 713's check of the
stored run frame that the Candidate reads, and prints PASS or FAIL for it. The exit status is 1 on a
negative margin or a FAIL.
"""

import argparse
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta

import e00725_rule as rule

GROUPS: dict[str, tuple[int, ...]] = {
    "shortwave and trailing irradiance (columns 0, 15 to 18)": rule.CAMS_COLUMNS,
    "other weather and trailing temperature (columns 1 to 14)": rule.ERA5_COLUMNS,
    "CAMS day summary (columns 19, 20)": rule.DAY_COLUMNS,
}
SOURCE_NAMES = {
    rule.SRC_ERA5: "ERA5 analysis",
    rule.SRC_CAMS: "CAMS analysis",
    rule.SRC_ENS: "ENS value of the token's own half-hour",
    rule.SRC_DAY: "CAMS day summary of the day before the stamp day",
    rule.SRC_DAY_D3: "CAMS day summary of UTC day D - 3",
}


@dataclass(frozen=True)
class Line:
    """One run of columns of one row that read one source, with its margin."""

    age: int
    first: int
    last: int
    group: str
    source: int
    first_stamp: datetime
    last_stamp: datetime
    margin_hours: float
    text: str


def _latest_origin_at_or_before(*, stamp: datetime) -> datetime:
    """Return the forecast origin (09:00 UTC) of the newest daily 00Z run whose origin is at or before `stamp`."""
    origin = datetime.combine(stamp.date(), time(9), tzinfo=UTC)
    return origin if origin <= stamp else origin - timedelta(days=1)


def lines_for(*, contract: datetime, rule_name: rule.RuleName) -> list[Line]:
    """Return the statement lines of one contract origin and rule, with the margin of each run of columns."""
    table = rule.reference_table(contract=contract, rule=rule_name)
    native = contract - timedelta(hours=3)
    day_d = contract.date()
    out: list[Line] = []
    for slot in range(rule.N_ROWS):
        age = rule.FIRST_AGE + slot
        stamps = [contract - timedelta(days=age) + timedelta(minutes=30 * j) for j in range(48)]
        for group, columns in GROUPS.items():
            codes = table[slot][:, columns[0]]
            start = 0
            for j in range(1, 49):
                if j < 48 and codes[j] == codes[start]:
                    continue
                code = int(codes[start])
                first, last = stamps[start], stamps[j - 1]
                if code == rule.SRC_ENS:
                    origin = _latest_origin_at_or_before(stamp=last)
                    margin = (native - origin).total_seconds() / 3600
                    basis = (
                        f"run issued {origin - timedelta(hours=9):%Y-%m-%d %H:%M} UTC, forecast origin "
                        f"{origin:%Y-%m-%d %H:%M} UTC, {margin:.1f} h before the native origin"
                    )
                elif code == rule.SRC_ERA5:
                    limit = native - timedelta(hours=rule.ERA5_LAG_HOURS)
                    latest = max(
                        last, rule._local_day_end(stamp=last)
                    )  # the daily extremes end at local midnight
                    margin = (limit - latest).total_seconds() / 3600
                    basis = f"latest ERA5 value {margin:.1f} h older than the 120 h limit"
                elif code == rule.SRC_CAMS:
                    cams_last = last if last.minute == 0 else last + timedelta(minutes=30)
                    limit = datetime.combine(day_d - timedelta(days=2), time(0), tzinfo=UTC)
                    margin = (limit - cams_last).total_seconds() / 3600
                    basis = f"latest CAMS source stamp {margin:.1f} h before 00:00 UTC on D - 2"
                elif code == rule.SRC_DAY:
                    read = last.date() - timedelta(days=1)
                    margin = float((day_d - timedelta(days=3) - read).days * 24)
                    basis = f"summary of UTC day {read}, {margin:.0f} h inside the D - 3 limit"
                else:
                    margin = 0.0
                    basis = f"summary of UTC day {day_d - timedelta(days=3)} (the D - 3 limit)"
                text = (
                    f"age {age:2d} | columns {start:2d}-{j - 1:2d} | {group} | {SOURCE_NAMES[code]} | "
                    f"valid {first:%Y-%m-%d %H:%M} to {last:%Y-%m-%d %H:%M} UTC | {basis}"
                )
                out.append(Line(age, start, j - 1, group, code, first, last, margin, text))
                start = j
    return out


def statement(*, contract: datetime, rule_name: rule.RuleName) -> list[str]:
    """Return the printed statement of one contract origin and rule."""
    native = contract - timedelta(hours=3)
    lines = [
        f"Contract origin {contract:%Y-%m-%d %H:%M} UTC; native origin {native:%Y-%m-%d %H:%M} UTC; rule {rule_name}",
        (
            "The target row reads the ensemble run of day D (issued 00:00 UTC, 9 h before the native origin) and "
            "the CAMS day summary of UTC day D - 1 in every arm; the power rows follow experiment 713's 27 h rule."
        ),
        "",
    ]
    lines += [line.text for line in lines_for(contract=contract, rule_name=rule_name)]
    return lines


def check_data(*, contract: datetime) -> tuple[bool, list[str]]:
    """Run experiment 713's check of the stored run frame of the native origin (needs the data folder)."""
    import e00713_weather_lookahead as w713  # ty: ignore[unresolved-import]

    return w713.check_data(contract=contract)


def main() -> int:
    """Print the statement."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin", default="2026-03-10")
    parser.add_argument("--rule", default="L", choices=rule.RULES)
    parser.add_argument("--check-data", action="store_true")
    args = parser.parse_args()
    contract = datetime.fromisoformat(args.origin).replace(hour=12, tzinfo=UTC)
    print("\n".join(statement(contract=contract, rule_name=args.rule)))
    negative = [x for x in lines_for(contract=contract, rule_name=args.rule) if x.margin_hours < 0]
    for line in negative:
        print("NEGATIVE MARGIN:", line.text)
    ok = not negative
    if args.check_data:
        passed, text = check_data(contract=contract)
        print("\n".join(text))
        ok = ok and passed
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
