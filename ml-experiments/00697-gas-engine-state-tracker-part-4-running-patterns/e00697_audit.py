r"""Experiment 697's data audit, run before any plot or statistic.

    uv run --no-sync python $E/00697-*/e00697_audit.py

Each step prints its coverage (first, last, expected count, missing count, longest gap), its
duplicates, its nulls, and its stamp convention, then asserts the figures the design quotes. The
report is written to `printed/audit.txt`. **The report holds no Gas A date and no Gas A series.**
Gas A's dates are compared with the expected values in code and only the outcome is printed. A
failed check is printed and makes the script exit with status 1 after every step has run, so one
failure does not hide the next.

The steps follow the design: (1) Gas A, (2) D8's reading and local weather, (3) the tracker's
labels, (4) the system price, (5) the gas price, (6) the day-ahead price, (7) national wind,
(8) bank holidays, (9) the red band, (10) the stored false-alarm pass.
"""

import argparse
import pickle
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from e00694_common import EXPECTED_UNIT_COUNTS, N_GRID  # ty: ignore[unresolved-import]
from e00695_common import ERA1_START  # ty: ignore[unresolved-import]
from e00697_common import (  # ty: ignore[unresolved-import]
    BLOCK_NAMES,
    EVENING_FIRST,
    EVENING_LAST,
    EXPECTED_ERA_LENGTHS,
    FALSE_ALARM_PATH,
    PIPELINE_OFFSET,
    PRINTED_DIR,
    PROBABILITY_PATH,
    UNIT_MW,
    add_local_fields,
    half_hour_starts,
    load_d8_eras,
    load_panel,
    panel_array,
    quantile_bins,
)
from e00697_drivers import (  # ty: ignore[unresolved-import]
    EXPECTED_DROPS,
    EXPECTED_SYSTEM_PRICE_ROWS,
    GAS_BAD_VALUES,
    bank_holiday_dates,
    day_ahead_prices,
    drops,
    gas_prices,
    national_wind,
    system_prices,
)
from e00697_engine import record_days, record_frame  # ty: ignore[unresolved-import]
from e00697_frame import DriverTables, add_drivers, load_drivers  # ty: ignore[unresolved-import]
from studies.power import scan_power

D8_SERIES_ID: Final[int] = 26
N_STAMP_CHECK: Final[int] = 1000
EXPECTED_GAS_A_ROWS: Final[int] = 2054
EXPECTED_GAS_A_COMPLETE_DAYS: Final[int] = 40
EXPECTED_GAS_A_WEEKEND_DAYS: Final[int] = 12
EXPECTED_GAS_A_WEEKEND_RUNNING_DAYS: Final[int] = 3
EXPECTED_GAS_A_INCOMPLETE_DAYS: Final[int] = 4
EXPECTED_GAS_A_OVERLAP_WITH_TA: Final[int] = 22
"""Half-hours of Gas A whose end stamp falls on or after the first day of experiment 695's labels."""
GAS_A_LAST_END: Final[datetime] = ERA1_START + timedelta(hours=10, minutes=30)
"""Gas A's last end stamp, as read from the stored record. It is compared in code and never printed."""
EXPECTED_SHARES: Final[dict[str, float]] = {"net": 0.268, "solar": 0.384, "fade": 0.262}
EXPECTED_UNITS_MW: Final[dict[str, float]] = {"net": 1.75, "solar": 2.25, "fade": 3.5}
EXPECTED_TA_AGREEMENT: Final[dict[str, float]] = {"net": 0.999, "solar": 0.989, "fade": 0.867}
EXPECTED_PRICE_FIFTHS: Final[tuple[float, ...]] = (0.057, 0.135, 0.151, 0.225, 0.281)
"""Gas A's running share by system-price fifth, cheapest to dearest, joined at the half-hour start."""
EXPERIMENT_194_PRICE_FIFTHS: Final[tuple[float, ...]] = (0.089, 0.152, 0.151, 0.198, 0.259)
"""The same shares as experiment 194 printed them, with the price of the following half-hour."""
FIFTH_TOLERANCE: Final[float] = 0.002
SYSTEM_PRICE_FIRST: Final[datetime] = datetime(2023, 5, 31, 23, 0, tzinfo=UTC)
SYSTEM_PRICE_LAST: Final[datetime] = datetime(2026, 7, 1, 22, 30, tzinfo=UTC)
EXPECTED_GAS_DAYS: Final[int] = 1122
EXPECTED_FA_SEGMENTS: Final[int] = 44
EXPECTED_FA_HALF_HOURS: Final[int] = 253440
HOLIDAY_SPOT_CHECKS: Final[dict[date, bool]] = {
    date(2024, 12, 25): True,
    date(2024, 12, 26): True,
    date(2024, 5, 27): True,
    date(2024, 8, 26): True,
    date(2024, 8, 5): False,
}
"""England's bank holidays: the Scottish summer holiday of 5 August 2024 must not appear."""
BLOCK_START_FRACTION: Final[float] = 0.85
"""The evening block starts at the first half-hour from 15:00 that reaches this share of the peak."""
CONTROL_PRIMARIES: Final[tuple[str, ...]] = ("D8", "D2", "D3", "D4", "D11", "D14", "D16", "D10")

LINES: list[str] = []
FAILURES: list[str] = []


def say(text: str = "") -> None:
    """Print a line and keep it for the report."""
    print(text, flush=True)
    LINES.append(text)


def check(*, name: str, ok: bool, detail: str = "") -> None:
    """Print a pass or a failure line, and remember a failure."""
    status = "ok  " if ok else "FAIL"
    say(f"  [{status}] {name}" + (f": {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def coverage(*, name: str, starts: pl.Series, step_minutes: int = 30, show_dates: bool = True) -> None:
    """Print a table's first and last stamp, expected and missing counts, longest gap, duplicates."""
    ordered = starts.drop_nulls().sort()
    n = len(ordered)
    span = (ordered[-1] - ordered[0]).total_seconds() / 60 / step_minutes
    expected = int(round(span)) + 1
    gaps = ordered.diff().drop_nulls().dt.total_minutes() / step_minutes - 1
    longest = int(gaps.max()) if len(gaps) and gaps.max() is not None else 0
    first_last = f"{ordered[0]:%Y-%m-%d %H:%M} to {ordered[-1]:%Y-%m-%d %H:%M}, " if show_dates else ""
    say(
        f"  {name}: {first_last}{n} stamps, {expected} expected, {expected - n} missing, "
        f"longest gap {longest} steps, {n - ordered.n_unique()} duplicates, "
        f"{starts.null_count()} null stamps"
    )


def step_1_gas_a() -> pl.DataFrame:
    """Gas A: row count, unit counts, last stamp, overlap with the labels, complete local days."""
    say("1. Gas A (aggregates and counts only; no date is printed)")
    frame = record_frame()
    check(name="2,054 rows", ok=frame.height == EXPECTED_GAS_A_ROWS, detail=str(frame.height))
    counts = tuple(frame.group_by("units").len().sort("units")["len"].to_list())
    check(name="unit counts 1,704 / 137 / 133 / 80", ok=counts == EXPECTED_UNIT_COUNTS, detail=str(counts))
    coverage(name="Gas A start stamps", starts=frame["start"], show_dates=False)
    end_stamp_last = frame["start"].max() + timedelta(minutes=30)
    check(name="last end stamp is the expected one", ok=end_stamp_last == GAS_A_LAST_END)
    ends = frame["start"] + timedelta(minutes=30)
    overlap_ta = int((ends >= ERA1_START).sum())
    check(
        name="22 half-hours overlap experiment 695's first day of labels",
        ok=overlap_ta == EXPECTED_GAS_A_OVERLAP_WITH_TA,
        detail=str(overlap_ta),
    )
    d8_starts = half_hour_starts(index=np.concatenate([
        int(e["start"]) + np.arange(len(e["y"])) for e in load_d8_eras()
    ]))
    shared = len(set(frame["start"].to_list()) & set(d8_starts.to_list()))
    check(name="no half-hour shared with experiment 694's labels", ok=shared == 0, detail=str(shared))
    per_day = frame.group_by("local_date").len()
    incomplete = int((per_day["len"] < 48).sum())
    check(
        name="4 incomplete local days",
        ok=incomplete == EXPECTED_GAS_A_INCOMPLETE_DAYS,
        detail=str(incomplete),
    )
    days = record_days()
    n_days, n_weekend = len(days.output), int(days.weekend.sum())
    weekend_running = int((days.output[days.weekend] >= UNIT_MW / 2).any(axis=1).sum())
    check(name="40 complete local days", ok=n_days == EXPECTED_GAS_A_COMPLETE_DAYS, detail=str(n_days))
    check(name="12 weekend days", ok=n_weekend == EXPECTED_GAS_A_WEEKEND_DAYS, detail=str(n_weekend))
    check(
        name="3 weekend days with running",
        ok=weekend_running == EXPECTED_GAS_A_WEEKEND_RUNNING_DAYS,
        detail=str(weekend_running),
    )
    units_from_power = np.rint(np.clip(frame["power"].to_numpy(), 0.0, None) / UNIT_MW).astype(int)
    mismatch = int((units_from_power != frame["units"].to_numpy()).sum())
    say(f"  levels from power differ from the stored `units` column on {mismatch} half-hours")
    nulls = {c: frame[c].null_count() for c in ("power", "units", "file_price")}
    say(f"  nulls: {nulls}")
    return frame


def step_2_panel() -> dict[int, dict]:
    """D8's reading and local weather: panel coverage, and a match with `scan_power`."""
    say("2. D8's reading and local weather (experiment 196's panel)")
    panel = load_panel()
    sites = {v["site"] for v in panel.values()}
    check(name="panel holds D8 and the control Primaries", ok=set(CONTROL_PRIMARIES) <= sites)
    for label in CONTROL_PRIMARIES:
        power = panel_array(panel=panel, label=label, key="power")
        check(name=f"{label}: {N_GRID} half-hours", ok=len(power) == N_GRID)
        say(f"  {label}: {np.isfinite(power).mean():.3f} of half-hours finite")
    for key in ("power", "expected", "ghi", "temp", "wind100"):
        values = panel_array(panel=panel, label="D8", key=key)
        say(f"  D8 {key}: {np.isfinite(values).mean():.3f} finite; range {np.nanmin(values):.2f} to {np.nanmax(values):.2f}")
    d8 = panel_array(panel=panel, label="D8", key="power")
    stored = (
        scan_power()
        .filter(pl.col("time_series_id") == D8_SERIES_ID)
        .select("time", "power")
        .collect()
        .drop_nulls()
    )
    index = (
        stored.select(
            index=(pl.col("time") - pl.lit(datetime(2022, 1, 1, tzinfo=UTC))).dt.total_minutes() // 30
        )["index"]
        .to_numpy()
    )
    inside = (index >= 0) & (index < N_GRID)
    rng = np.random.default_rng(0)
    pick = rng.choice(np.flatnonzero(inside), size=N_STAMP_CHECK, replace=False)
    difference = np.abs(stored["power"].to_numpy()[pick] - d8[index[pick]])
    finite = np.isfinite(difference)
    check(
        name="panel power equals scan_power for series 26 on 1,000 random end stamps",
        ok=bool(finite.all() and difference.max() < 1e-3),
        detail=f"{int(finite.sum())} finite, largest difference {np.nanmax(difference):.2e}",
    )
    return panel


def step_3_labels() -> pl.DataFrame:
    """The tracker's labels: era lengths, shares, units, agreement, and the evening block's start."""
    say("3. Tracker labels")
    eras = load_d8_eras()
    ta = 1.0 - np.load(PROBABILITY_PATH)["pk"][:, 0].astype(float)
    parts = []
    for era in eras:
        label, n = era["label"], len(era["y"])
        check(name=f"{label}: {EXPECTED_ERA_LENGTHS[label]} half-hours", ok=n == EXPECTED_ERA_LENGTHS[label])
        share = float(np.mean(era["M2"]["flag"]))
        check(
            name=f"{label}: flagged share {EXPECTED_SHARES[label]}",
            ok=abs(share - EXPECTED_SHARES[label]) < 0.0006,
            detail=f"{share:.4f}",
        )
        check(
            name=f"{label}: fitted unit {EXPECTED_UNITS_MW[label]} MW",
            ok=float(era["M2"]["u"]) == EXPECTED_UNITS_MW[label],
            detail=str(era["M2"]["u"]),
        )
        index = int(era["start"]) + np.arange(n)
        agree = float(np.mean(np.asarray(era["M2"]["flag"]) == (ta[index + PIPELINE_OFFSET] > 0.5)))
        check(
            name=f"{label}: agreement of Ts with Ta is {EXPECTED_TA_AGREEMENT[label]}",
            ok=abs(agree - EXPECTED_TA_AGREEMENT[label]) < 0.0006,
            detail=f"{agree:.4f}",
        )
        parts.append(
            pl.DataFrame({"era": label, "start": half_hour_starts(index=index), "ts": era["M2"]["flag"]})
        )
        say(f"  {label}: K = {era['M2']['k']}, unit {era['M2']['u']} MW")
    frame = add_local_fields(frame=pl.concat(parts))
    weekday = frame.filter(~pl.col("weekend") & ~pl.col("clock_change_day"))
    profile = (
        weekday.group_by("era", "local_half_hour")
        .agg(share=pl.col("ts").mean())
        .sort("era", "local_half_hour")
    )
    say("  Weekday running share by local half-hour (start of the half-hour), 15:00 to 19:30:")
    for era in ("net", "solar", "fade"):
        rows = profile.filter((pl.col("era") == era) & pl.col("local_half_hour").is_between(30, 39))
        shares = " ".join(f"{h // 2:02d}:{30 * (h % 2):02d}={s:.2f}" for h, s in rows.select("local_half_hour", "share").iter_rows())
        evening = profile.filter((pl.col("era") == era) & (pl.col("local_half_hour") >= 30))
        peak = evening.filter(pl.col("local_half_hour") <= 39)["share"].max()
        first = evening.filter(pl.col("share") >= BLOCK_START_FRACTION * peak)
        start = first["local_half_hour"].min()
        label = f"{start // 2:02d}:{30 * (start % 2):02d}" if start is not None else "none"
        say(f"  {era}: block starts at {label} local; {shares}")
    say(
        f"  The evening window of Q1 is local half-hours {EVENING_FIRST} to {EVENING_LAST} "
        f"(16:00 to 19:00); the day blocks are {', '.join(BLOCK_NAMES)}."
    )
    return frame


def fifth_shares(*, joined: pl.DataFrame) -> list[float]:
    """Return Gas A's running share in each fifth of the joined system price, cheapest first."""
    bins = quantile_bins(values=joined["system_price"].to_numpy(), n_bins=5)
    running = (joined["units"].to_numpy() > 0).astype(float)
    return [float(running[bins == k].mean()) for k in range(5)]


def step_4_system_price(*, record: pl.DataFrame) -> None:
    """The system price: size, span, and Gas A's price-fifth shares at the correct join."""
    say("4. System price (Elexon, settlement-period start stamps)")
    prices = system_prices()
    coverage(name="system price", starts=prices["start"])
    check(
        name="54,096 half-hours",
        ok=prices.height == EXPECTED_SYSTEM_PRICE_ROWS,
        detail=str(prices.height),
    )
    check(
        name="spans 2023-05-31T23:00Z to 2026-07-01T22:30Z",
        ok=(prices["start"].min(), prices["start"].max()) == (SYSTEM_PRICE_FIRST, SYSTEM_PRICE_LAST),
    )
    say(f"  nulls: {prices['system_price'].null_count()}; range {prices['system_price'].min():.1f} to {prices['system_price'].max():.1f} GBP per MWh")
    shares_by_shift = {}
    for shift in (-30, 0, 30, 60):
        joined = (
            record.with_columns(key=pl.col("start") + pl.duration(minutes=shift))
            .join(prices.rename({"start": "key"}), on="key", how="left", maintain_order="left")
        )
        shares_by_shift[shift] = [round(v, 3) for v in fifth_shares(joined=joined)]
    say("  Gas A's running share by price fifth, cheapest to dearest, by the minutes added to the start:")
    for shift, shares in shares_by_shift.items():
        say(f"    {shift:+d} min: {shares}")
    correct = shares_by_shift[0]
    check(
        name="fifths at the start join are 0.057, 0.135, 0.151, 0.225, 0.281",
        ok=all(abs(a - b) <= FIFTH_TOLERANCE for a, b in zip(correct, EXPECTED_PRICE_FIFTHS, strict=True)),
        detail=str(correct),
    )
    check(
        name="the end-stamp join reproduces experiment 194's fifths (0.089 ... 0.259)",
        ok=all(
            abs(a - b) <= FIFTH_TOLERANCE
            for a, b in zip(shares_by_shift[30], EXPERIMENT_194_PRICE_FIFTHS, strict=True)
        ),
        detail=str(shares_by_shift[30]),
    )
    equal_at_end = float(
        (
            record.with_columns(key=pl.col("start") + pl.duration(minutes=30))
            .join(prices.rename({"start": "key"}), on="key", how="left", maintain_order="left")
            .select(same=pl.col("system_price") == pl.col("file_price"))["same"]
            .mean()
        )
    )
    check(
        name="the stored file's price column is the price at the end stamp",
        ok=equal_at_end == 1.0,
        detail=f"share equal {equal_at_end:.3f}",
    )


def step_5_gas() -> None:
    """The gas price: size, bad values, and the gas-day join."""
    say("5. Gas price (National Gas SAP, pounds per MWh of gas)")
    gas = gas_prices()
    coverage(name="gas price days", starts=gas["gas_day"].cast(pl.Datetime("us")), step_minutes=1440)
    check(name="1,122 days after dropping 4 bad values", ok=gas.height == EXPECTED_GAS_DAYS, detail=str(gas.height))
    say(f"  dropped values (pence per kWh): {GAS_BAD_VALUES}; kept range {gas['gas_price_gbp_mwh'].min() / 10:.2f} to {gas['gas_price_gbp_mwh'].max() / 10:.2f}")
    say("  The gas day for date D covers 05:00 on D to 05:00 on D + 1, so the join is on (local start - 5 h).date().")


def step_6_to_9(*, tables: DriverTables) -> None:
    """Day-ahead price, national wind, bank holidays, and the red band."""
    say("6. Day-ahead price (NESO N2EX, hourly, from 2025-09-01; nothing earlier is stored)")
    day_ahead = day_ahead_prices()
    coverage(name="day-ahead price", starts=day_ahead["hour"], step_minutes=60)
    say(f"  nulls {day_ahead['day_ahead_price'].null_count()}; earlier periods are skipped and nothing is downloaded")
    say("7. National wind (transmission wind from 2024-07-01)")
    wind = national_wind()
    coverage(name="national wind", starts=wind["start"])
    say("8. Bank holidays (England, from the `holidays` package, plus 24 December to 1 January)")
    found = bank_holiday_dates()
    for day, expected in HOLIDAY_SPOT_CHECKS.items():
        check(name=f"{day} is {'' if expected else 'not '}a bank holiday", ok=(day in found) == expected)
    say(f"  {len([d for d in found if d.year in (2024, 2025)])} bank holidays in 2024 and 2025")
    say("9. Red band")
    say(
        "  Weekdays 16:00 to 19:00 local, from the private note `predictability.md` of the engine-state "
        "follow-on. No charging statement is stored and no download is allowed, so whether the band "
        "applies on bank holidays is not checked."
    )
    d8_probe = add_drivers(
        frame=add_local_fields(frame=pl.DataFrame({"start": half_hour_starts(index=np.arange(29184, 29184 + 48 * 400))})),
        tables=tables,
    )
    say(
        f"  join check on 400 days of D8's net era: system price {d8_probe['system_price'].null_count()} nulls, "
        f"gas {d8_probe['gas_price_gbp_mwh'].null_count()}, day-ahead {d8_probe['day_ahead_price'].null_count()}, "
        f"wind {d8_probe['national_wind_mw'].null_count()}"
    )
    say(f"  drops of experiment 159 from 2023-07: {drops().height}")
    check(name="28 multi-day drops", ok=drops().height == EXPECTED_DROPS)


def step_10_false_alarms() -> None:
    """The stored false-alarm pass: 44 segments and no half-hour flagged by M2."""
    say("10. Stored false-alarm pass (experiment 694's fa.pkl)")
    segments = pickle.loads(FALSE_ALARM_PATH.read_bytes())
    n = sum(s["arms"]["M2"]["n"] for s in segments)
    flagged = sum(s["arms"]["M2"]["flagged"] for s in segments)
    hosts = sorted({s["host"] for s in segments})
    check(name="44 segments", ok=len(segments) == EXPECTED_FA_SEGMENTS, detail=str(len(segments)))
    check(name="253,440 half-hours", ok=n == EXPECTED_FA_HALF_HOURS, detail=str(n))
    check(name="0 half-hours flagged by M2", ok=flagged == 0, detail=str(flagged))
    say(f"  hosts: {hosts}")


def main() -> None:
    """Run the ten steps, write the report, and exit with status 1 if any check failed."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PRINTED_DIR / "audit.txt")
    args = parser.parse_args()
    say("Experiment 697: data audit")
    record = step_1_gas_a()
    step_2_panel()
    step_3_labels()
    step_4_system_price(record=record)
    step_5_gas()
    step_6_to_9(tables=load_drivers())
    step_10_false_alarms()
    say("")
    say(f"{len(FAILURES)} failed checks" + (f": {FAILURES}" if FAILURES else ""))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(LINES) + "\n")
    if FAILURES:
        sys.exit(1)


if __name__ == "__main__":
    main()
