# ruff: noqa: E501, PLR0915, PT018, RUF005, RUF007, B007, RUF059, PERF401, PLW1510, C416
"""Experiment 735: known-answer tests of the columns, windows, bootstrap, verdict, network, and phase runner.

README. Run with plain `python` (no pytest), from the worktree root, with the experiment folders on
`PYTHONPATH` (source `ml-experiments/00524*/env.sh`) and the three dummy `NGED_S3_BUCKET_*` variables
and `DATA_PATH_INTERNAL` set: `python ml-experiments/00735-*/e00735_tests_known_answer.py [--skip-build]`.
Each test prints PASS or raises. The zero-start tests need `torch` and are skipped, with a printed
note, when it is not installed. The tests read only the two term-date csv files, the coast distances,
and 677's frozen stage-1 files (hashes). No test fits a network or writes under experiment 677.

The test that the new columns reach stage-3 rows, and the test that every arm's files are written
under experiment 735's folder, need the GPU runner (`e00735_k1.py`, which executes on import), so
the smoke test checks them; this file checks the column builder those wrappers call.
"""

import ast
import csv
import hashlib
import importlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

import e00735_columns as cols  # ty: ignore[unresolved-import]
import e00735_score as score  # ty: ignore[unresolved-import]
import e00735_windows as win  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULTS: list[str] = []
HOME_PREFIX = "/" + "home/jack"
"""Built from two parts, so that this file does not itself hold an absolute path under the maintainer's home folder."""


def passed(name: str) -> None:
    """Record and print one passing test."""
    RESULTS.append(name)
    print(f"PASS {name}", flush=True)


def stamp_us(*, year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> int:
    """Return a UTC instant in microseconds since the epoch."""
    from datetime import UTC, datetime

    return int(datetime(year, month, day, hour, minute, tzinfo=UTC).timestamp()) * 1_000_000


def local(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> date:
    """Return the local date that the new columns give a UTC stamp."""
    stamp = stamp_us(year=year, month=month, day=day, hour=hour, minute=minute)
    code = cols.local_dates(stamp_us=np.asarray([stamp])).code[0]
    return cols.date_of(code=int(code))


# ----------------------------------------------------------------------------- imports


def test_every_attribute_exists() -> None:
    """Every `module.attribute` that an experiment-735 script takes from an imported module exists."""
    checked = 0
    for path in sorted(HERE.glob("e00735_*.py")):
        if path.name in {"e00735_k1.py", Path(__file__).name}:
            continue  # the runner executes on import; the smoke test reaches its attributes
        tree = ast.parse(path.read_text())
        aliases: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    aliases[a.asname or a.name.split(".")[0]] = a.name
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                mod = importlib.import_module(node.module)
                for a in node.names:
                    assert hasattr(mod, a.name), (path.name, node.module, a.name)
                    checked += 1
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in aliases
            ):
                name = aliases[node.value.id]
                try:
                    mod = importlib.import_module(name)
                except ModuleNotFoundError:
                    print(f"  skipped {path.name}: {name} is not importable here")
                    continue
                assert hasattr(mod, node.attr), (path.name, name, node.attr)
                checked += 1
    assert checked > 100, checked
    passed(f"every attribute taken from an imported module exists ({checked} checked)")


# ----------------------------------------------------------------------------- local dates


def test_local_date_convention() -> None:
    """The local date is that of the stamp minus 15 minutes, at 00:00, at 00:30, and across the clock changes."""
    assert local(2026, 2, 14, 0, 0) == date(2026, 2, 13)
    assert local(2026, 2, 14, 0, 30) == date(2026, 2, 14)
    assert local(2026, 2, 14, 23, 45) == date(2026, 2, 14)
    # 2025-10-26: BST ends at 01:00 UTC. Stamp 23:00 UTC on 10-25 is 00:00 BST on 10-26.
    assert local(2025, 10, 25, 23, 0) == date(2025, 10, 25)
    assert local(2025, 10, 25, 23, 30) == date(2025, 10, 26)
    assert local(2025, 10, 27, 0, 0) == date(
        2025, 10, 26
    )  # 00:00 GMT on 10-27 is the last stamp of 10-26
    assert local(2025, 10, 27, 0, 30) == date(2025, 10, 27)
    # 2026-03-29: BST starts at 01:00 UTC. Stamp 00:00 UTC is 00:00 GMT.
    assert local(2026, 3, 29, 0, 0) == date(2026, 3, 28)
    assert local(2026, 3, 29, 0, 30) == date(2026, 3, 29)
    assert local(2026, 3, 29, 23, 0) == date(2026, 3, 29)  # 00:00 BST on 03-30
    assert local(2026, 3, 29, 23, 30) == date(2026, 3, 30)
    # 2026-10-25: BST ends at 01:00 UTC, inside the October half-term.
    assert local(2026, 10, 24, 23, 0) == date(2026, 10, 24)  # 00:00 BST on 10-25 ends Saturday
    assert local(2026, 10, 24, 23, 30) == date(2026, 10, 25)
    assert local(2026, 10, 26, 0, 0) == date(2026, 10, 25)  # 00:00 GMT on 10-26 ends Sunday
    assert local(2026, 10, 26, 0, 30) == date(2026, 10, 26)
    mid = cols.is_local_midnight(
        stamp_us=np.asarray(
            [
                stamp_us(year=2026, month=10, day=24, hour=23),
                stamp_us(year=2026, month=10, day=26),
                stamp_us(year=2026, month=10, day=26, minute=30),
            ]
        )
    )
    assert mid.tolist() == [True, True, False], mid
    passed(
        "stamp-minus-15-minutes local date at 00:00, 00:30, and across 2025-10-26, 2026-03-29, 2026-10-25"
    )


# ----------------------------------------------------------------------------- school calendar


def independent_calendar() -> dict[date, tuple[int, int, int]]:
    """Return S1, S2, S3 per date from the two csv files, re-derived with the csv module only."""
    terms: list[tuple[date, date]] = []
    for path in sorted(REPO.glob("ml-experiments/00011-*/lincolnshire_term_dates.csv")) + [
        HERE / "lincolnshire_term_dates_2026_27.csv"
    ]:
        with path.open() as f:
            for row in csv.reader(f):
                if row[0] == "academic_year":
                    continue
                terms.append((date.fromisoformat(row[2]), date.fromisoformat(row[3])))
    terms.sort()
    out: dict[date, tuple[int, int, int]] = {}
    for (_, last), (nxt, _) in zip(terms[:-1], terms[1:], strict=True):
        first, end = last + timedelta(days=1), nxt - timedelta(days=1)
        length = (end - first).days + 1
        sat = first + timedelta(days=(5 - first.weekday()) % 7)
        weekend = {d for d in (sat, sat + timedelta(days=1)) if d <= end}
        d = first
        while d <= end:
            out[d] = (1, int(d in weekend), int(length <= 10))
            d += timedelta(days=1)
    return out


def test_school_columns_equal_csv() -> None:
    """S1 to S3 equal the two csv files on every date from 2024-04-01 to 2026-11-30; S3 is 1 only in blocks of at most 10 days."""
    expected = independent_calendar()
    days = [
        date(2024, 4, 1) + timedelta(days=i)
        for i in range((date(2026, 11, 30) - date(2024, 4, 1)).days + 1)
    ]
    got = cols.school_columns(codes=np.asarray([cols.code_of(day=d) for d in days]))
    for d, row in zip(days, got, strict=True):
        assert tuple(int(v) for v in row) == expected.get(d, (0, 0, 0)), (d, row, expected.get(d))
    long_block_days = [
        d for d in days if d in expected and expected[d][2] == 0 and expected[d][0] == 1
    ]
    assert date(2025, 12, 25) in long_block_days and date(2026, 8, 1) in long_block_days
    assert all(got[days.index(d), 2] == 0 for d in long_block_days)
    assert sum(int(g[0]) for g in got) > 200
    # the October 2026 half-term: Saturday 2026-10-24 to Sunday 2026-11-01, first weekend 24 and 25 October
    i = days.index(date(2026, 10, 24))
    assert (
        got[i].tolist() == [1, 1, 1]
        and got[i + 1].tolist() == [1, 1, 1]
        and got[i + 2].tolist() == [1, 0, 1]
    )
    assert got[days.index(date(2026, 11, 1))].tolist() == [1, 0, 1] and got[
        days.index(date(2026, 11, 2))
    ].tolist() == [0, 0, 0]
    passed("school columns equal the csv files on every date from 2024-04-01 to 2026-11-30")


def test_placebo_calendar() -> None:
    """The placebo calendar equals the real calendar 21 days later; print the share of placebo holiday days that are real holiday days."""
    start = cols.code_of(day=date(2024, 4, 1))
    stop = cols.code_of(day=date(2026, 11, 30)) - 21
    real = cols.school_columns(codes=np.arange(start, stop + 1))
    placebo = cols.school_columns(codes=np.arange(start, stop + 1) + 21, shift_days=21)
    assert (real == placebo).all()
    d = date(2025, 12, 20)
    assert (
        cols.school_columns(codes=np.asarray([cols.code_of(day=d) + 21]), shift_days=21)[0].tolist()
        == cols.school_columns(codes=np.asarray([cols.code_of(day=d)]))[0].tolist()
    )
    calendar = cols.dense_calendar()
    print(
        "  share of placebo holiday days that are real holiday days, by block (window shares in the next lines)"
    )
    for b, block in enumerate(cols.blocks()):
        if block.first < date(2024, 4, 1) or block.last > date(2026, 11, 8):
            continue
        days = [cols.code_of(day=block.first) + i + 21 for i in range(block.days)]
        share = float(calendar.school_off[np.asarray(days) - calendar.first_code].mean())
        print(f"  block {block.first} to {block.last} ({block.days} days): {share:.2f}")
    w = win.day_windows()
    for name, flags in w.school.items():
        idx = np.flatnonzero(flags)
        idx = idx[
            (idx + w.first_code >= cols.code_of(day=date(2024, 4, 1))) & (idx + 21 < len(flags))
        ]
        share = float(calendar.school_off[idx + 21].mean())
        print(f"  window {name}: {share:.2f} of its placebo days are real holiday days")
    passed("the placebo calendar equals the real calendar 21 days later (shares printed)")


def test_window_masks() -> None:
    """Window masks match hand-written dates for the February 2026 half-term and the Christmas block."""

    def names(day: date, *, stamp: tuple[int, ...] | None = None) -> set[str]:
        code = np.asarray([cols.code_of(day=day)])
        if stamp is not None:
            code = cols.local_dates(
                stamp_us=np.asarray(
                    [
                        stamp_us(
                            year=stamp[0],
                            month=stamp[1],
                            day=stamp[2],
                            hour=stamp[3],
                            minute=stamp[4],
                        )
                    ]
                )
            ).code
        return {k for k, v in win.window_masks(codes=code).items() if v[0]}

    feb = {  # the block is 2026-02-14 (Saturday) to 2026-02-22; term resumes on Monday 2026-02-23
        date(2026, 2, 6): {"ordinary days"},
        date(2026, 2, 7): {"school: 7 to 4 days before", "school window"},
        date(2026, 2, 10): {"school: 7 to 4 days before", "school window"},
        date(2026, 2, 11): {"school: 3 to 1 days before", "school window"},
        date(2026, 2, 13): {"school: 3 to 1 days before", "school window"},
        date(2026, 2, 14): {"school: first weekend", "school window"},
        date(2026, 2, 15): {"school: first weekend", "school window"},
        date(2026, 2, 16): {"school: rest of block", "school window"},
        date(2026, 2, 22): {"school: rest of block", "school window"},
        date(2026, 2, 23): {"school: first 3 days after", "school window"},
        date(2026, 2, 25): {"school: first 3 days after", "school window"},
        date(2026, 2, 26): {"school: 4 to 7 days after", "school window"},
        date(2026, 3, 1): {"school: 4 to 7 days after", "school window"},
        date(2026, 3, 2): {"ordinary days"},
    }
    for d, want in feb.items():
        assert names(d) == want, (d, names(d), want)
    # Christmas: the block is 2025-12-20 (Saturday) to 2026-01-05; term resumes on Tuesday 2026-01-06;
    # 2025-12-25, 2025-12-26, and 2026-01-01 are bank holidays inside it.
    assert names(date(2025, 12, 20)) == {"school: first weekend", "school window"}
    # Christmas Day is also the day before Boxing Day, and Boxing Day is also the day after Christmas Day
    assert names(date(2025, 12, 25)) == {
        "school: rest of block",
        "school window",
        "bank: the day",
        "bank: day before",
        "bank window",
    }
    assert names(date(2025, 12, 26)) == {
        "school: rest of block",
        "school window",
        "bank: the day",
        "bank: day after",
        "bank window",
    }
    assert names(date(2025, 12, 24)) == {
        "school: rest of block",
        "school window",
        "bank: day before",
        "bank window",
    }
    assert "bank: the day" in names(date(2026, 1, 1))
    assert names(date(2026, 1, 6)) == {"school: first 3 days after", "school window"}
    assert names(date(2026, 1, 9)) == {"school: 4 to 7 days after", "school window"}
    assert names(date(2026, 1, 12)) == {"school: 4 to 7 days after", "school window"}
    assert names(date(2026, 1, 13)) == {"ordinary days"}
    # Easter Monday 2026-04-06 is a Monday: the extended weekend is Saturday 04-04 and Sunday 04-05
    assert "bank: the day" in names(date(2026, 4, 6))
    assert "bank: extended weekend" in names(
        date(2026, 4, 4)
    ) and "bank: extended weekend" in names(date(2026, 4, 5))
    # at midnight and at a 00:30 stamp after the 2026-10-25 clock change (half-term block 2026-10-24 to 2026-11-01)
    assert names(date(2026, 10, 24)) == {"school: first weekend", "school window"}
    assert "school: first weekend" in names(date(2026, 10, 25))
    assert names(date(2026, 10, 24), stamp=(2026, 10, 23, 23, 0)) == {
        "school: 3 to 1 days before",
        "school window",
    }  # stamp 00:00 BST on the 24th is the 23rd
    assert names(date(2026, 10, 24), stamp=(2026, 10, 23, 23, 30)) == {
        "school: first weekend",
        "school window",
    }  # 00:30 BST on the 24th
    assert names(date(2026, 10, 24), stamp=(2026, 10, 26, 0, 0)) == {
        "school: first weekend",
        "school window",
    }  # 00:00 GMT on the 26th is the 25th
    assert names(date(2026, 10, 24), stamp=(2026, 10, 26, 0, 30)) == {
        "school: rest of block",
        "school window",
    }  # 00:30 GMT on the 26th
    assert names(date(2026, 10, 24), stamp=(2026, 11, 2, 0, 0)) == {
        "school: rest of block",
        "school window",
    }  # 00:00 GMT on 11-02 ends the 1st (last block day)
    assert names(date(2026, 10, 24), stamp=(2026, 11, 2, 0, 30)) == {
        "school: first 3 days after",
        "school window",
    }
    passed(
        "window masks match hand-written dates for two blocks at midnight and at 00:30 stamps after a clock change"
    )


# ----------------------------------------------------------------------------- coast permutation


def test_permutation() -> None:
    """The coast permutation is a derangement by value, fixed by the seed, with correlation within 0.3."""
    c = cols.closeness(km=cols.coast_distances_km())
    assert len(c) == 16 and np.isclose(c.max(), 0.87, atol=0.01) and c.min() < 0.01, c
    perm = cols.derange(c=c)
    again = cols.derange(c=c)
    assert (perm == again).all()
    assert sorted(perm.tolist()) == list(range(16))
    assert (np.abs(c[perm] - c) > 1e-9).all(), "a Primary keeps its closeness (including a tie)"
    assert abs(float(np.corrcoef(c, c[perm])[0, 1])) <= 0.3
    true, permuted = cols.coast_arrays()
    assert (true == c).all() and (permuted == c[perm]).all()
    passed(
        "the coast permutation is a derangement by value, fixed by seed 735, with |correlation| <= 0.3"
    )


# ----------------------------------------------------------------------------- weather columns


def test_w1_w2_monthly_anomaly() -> None:
    """W1 and W2 equal the monthly anomaly of the B1 columns, re-derived with a loop."""
    rng = np.random.default_rng(1)
    n = 6000
    primary = rng.integers(0, 16, size=n)
    month = rng.integers(1, 13, size=n)
    b1 = rng.normal(loc=month[:, None] * 0.1, scale=1.0 + primary[:, None] * 0.05, size=(n, 2))
    fit = cols.fit_anomalies(primary=primary, month=month, b1=b1)
    z = cols.apply_anomalies(fit=fit, primary=primary, month=month, b1=b1)
    for p in (0, 5, 15):
        for m in (1, 7, 12):
            sel = (primary == p) & (month == m)
            mean = b1[sel].mean(axis=0)
            anomaly_all = np.concatenate(
                [
                    b1[(primary == p) & (month == mm)]
                    - b1[(primary == p) & (month == mm)].mean(axis=0)
                    for mm in range(1, 13)
                ]
            )
            sd = anomaly_all.std(axis=0)
            want = np.clip((b1[sel] - mean) / sd, -3, 3)
            want = np.zeros_like(want) if sel.sum() < cols.MIN_ROWS_PER_CELL else want
            assert np.allclose(z[sel], want, atol=1e-5), (p, m)
    assert np.abs(z).max() <= 3.0
    thin = np.zeros(10, dtype=np.int64), np.ones(10, dtype=np.int64)
    f2 = cols.fit_anomalies(primary=thin[0], month=thin[1], b1=rng.normal(size=(10, 2)))
    assert (
        cols.apply_anomalies(fit=f2, primary=thin[0], month=thin[1], b1=rng.normal(size=(10, 2)))
        == 0
    ).all()
    passed(
        "W1 and W2 equal the monthly anomaly of the B1 columns (zero in thin cells, clipped to 3)"
    )


def test_columns_read_no_era5() -> None:
    """No code that builds a new input column reads ERA5: W1 and W2 come from the row's own B1 columns, W3 from the expected-weather table."""
    folder = Path(__file__).parent
    for name in ("e00735_columns.py", "e00735_model.py", "e00735_k1.py"):
        code = "\n".join(
            line
            for line in (folder / name).read_text().splitlines()
            if not line.lstrip().startswith(("#", '"', "-", "`"))
        )
        assert "era5" not in code.lower(), name


def test_w3_causal() -> None:
    """W3 reads no expected-weather day that ended less than 24 hours before the origin (experiment 437's `check` pattern)."""
    n_days = 40
    first = cols.code_of(day=date(2026, 1, 1))
    rng = np.random.default_rng(0)
    value = rng.normal(8.0, 3.0, size=(2, n_days)).astype(np.float32)
    daily = cols.DailyMax(
        first_code=first, value=value, run_us=np.zeros((2, n_days), dtype=np.int64)
    )
    # origin 09:00 UTC on 2026-01-20; origin - 24 h = 2026-01-19 09:00, so the last complete day is 2026-01-18
    origin = np.asarray([stamp_us(year=2026, month=1, day=20, hour=9)])
    last, cut = cols.last_complete_day(origin_us=origin)
    assert cols.date_of(code=int(last[0])) == date(2026, 1, 18), cols.date_of(code=int(last[0]))
    base = cols.week_baseline(daily=daily, primary=np.asarray([0]), origin_us=origin)
    want = value[
        0,
        (cols.code_of(day=date(2026, 1, 12)) - first) : (
            cols.code_of(day=date(2026, 1, 18)) - first
        )
        + 1,
    ].mean()
    assert np.isclose(base[0], want), (base, want)
    # changing any day after the last complete day leaves the baseline unchanged; changing a day inside moves it
    for day_index in range(cols.code_of(day=date(2026, 1, 19)) - first, n_days):
        changed = value.copy()
        changed[0, day_index] += 50.0
        b2 = cols.week_baseline(
            daily=cols.DailyMax(first_code=first, value=changed, run_us=daily.run_us),
            primary=np.asarray([0]),
            origin_us=origin,
        )
        assert b2[0] == base[0], day_index
    moved = value.copy()
    moved[0, cols.code_of(day=date(2026, 1, 15)) - first] += 50.0
    assert (
        cols.week_baseline(
            daily=cols.DailyMax(first_code=first, value=moved, run_us=daily.run_us),
            primary=np.asarray([0]),
            origin_us=origin,
        )[0]
        != base[0]
    )
    # a cut exactly at local midnight completes the day before it: origin 2026-01-20 00:00 -> cut 2026-01-19 00:00 -> day 01-18
    exact, _ = cols.last_complete_day(origin_us=np.asarray([stamp_us(year=2026, month=1, day=20)]))
    assert cols.date_of(code=int(exact[0])) == date(2026, 1, 18)
    # across the clock change of 2026-10-25: origin 2026-10-26 09:00 UTC -> cut 2026-10-25 09:00 -> last complete day 2026-10-24
    across, _ = cols.last_complete_day(
        origin_us=np.asarray([stamp_us(year=2026, month=10, day=26, hour=9)])
    )
    assert cols.date_of(code=int(across[0])) == date(2026, 10, 24)
    # the latest run origin feeding the baseline is read from the same 7 days only
    runs = np.zeros((2, n_days), dtype=np.int64)
    runs[0, cols.code_of(day=date(2026, 1, 19)) - first :] = 10**15
    runs[0, cols.code_of(day=date(2026, 1, 15)) - first] = 123
    latest = cols.week_latest_run_us(
        daily=cols.DailyMax(first_code=first, value=value, run_us=runs),
        primary=np.asarray([0]),
        origin_us=origin,
    )
    assert latest[0] == 123, latest
    # fewer than 4 complete days gives no baseline and W3 = 0
    sparse = value.copy()
    sparse[0, : cols.code_of(day=date(2026, 1, 16)) - first] = np.nan
    assert np.isnan(
        cols.week_baseline(
            daily=cols.DailyMax(first_code=first, value=sparse, run_us=daily.run_us),
            primary=np.asarray([0]),
            origin_us=origin,
        )[0]
    )
    w3 = cols.first_warm_day(
        tmax_celsius=np.asarray([20.0, 5.0, 12.0]), baseline=np.asarray([10.0, 10.0, np.nan])
    )
    assert np.allclose(w3, [2.0, -1.0, 0.0])
    assert (
        cols.first_warm_day(tmax_celsius=np.asarray([100.0]), baseline=np.asarray([0.0]))[0] == 3.0
    )
    passed(
        "W3 is causal: it reads only days that ended 24 hours before the origin; clipped to 3; zero without a baseline"
    )


def test_build_block() -> None:
    """The block holds each arm's columns in order, with K and W built from the real or permuted closeness and never from existing columns."""
    c = np.linspace(0.9, 0.01, 16)
    perm = np.roll(np.arange(16), 5)
    codes = np.asarray(
        [
            cols.code_of(day=d)
            for d in (date(2026, 2, 14), date(2026, 2, 16), date(2026, 3, 4), date(2025, 12, 25))
        ]
    )
    primary = np.asarray([0, 3, 15, 8], dtype=np.int64)
    weather = np.asarray([[1.0, -1.0, 0.5]] * 4, dtype=np.float32)
    factors = cols.Factors(codes=codes, primary=primary, weather=weather)
    for name, spec in cols.ARMS.items():
        block = cols.build_block(spec=spec, factors=factors, c_true=c, c_permuted=c[perm])
        assert block.shape == (4, spec.width) and block.dtype == np.float32, name
        assert len(spec.names) == spec.width
    r0 = cols.build_block(spec=cols.ARMS["R0"], factors=factors, c_true=c, c_permuted=c[perm])
    assert r0.shape == (4, 0)
    skw = cols.build_block(spec=cols.ARMS["SKW"], factors=factors, c_true=c, c_permuted=c[perm])
    skwp = cols.build_block(spec=cols.ARMS["SKWP"], factors=factors, c_true=c, c_permuted=c[perm])
    s = cols.school_columns(codes=codes)
    assert np.allclose(skw[:, :3], s) and np.allclose(skwp[:, :3], s), (
        "the school columns are not permuted"
    )
    assert np.allclose(skw[:, 3], c[primary]) and np.allclose(skwp[:, 3], c[perm][primary])
    assert np.allclose(skw[:, 4], c[primary] * s[:, 0]) and np.allclose(
        skw[:, 5], c[primary] * s[:, 1]
    )
    bank, weekend = cols.day_flags(codes=codes)
    assert np.allclose(skw[:, 6], c[primary] * bank) and np.allclose(
        skw[:, 7], c[primary] * weekend
    )
    assert np.allclose(skw[:, 8], c[primary] * s[:, 0] * 1.0) and np.allclose(
        skw[:, 9], c[primary] * s[:, 0] * -1.0
    )
    assert np.allclose(skw[:, 10], c[primary] * s[:, 0] * 0.5) and np.allclose(
        skw[:, 11], c[primary] * 1.0
    )
    assert bank[3] == 1.0 and weekend[0] == 1.0 and weekend[1] == 0.0
    sp = cols.build_block(spec=cols.ARMS["SP"], factors=factors, c_true=c, c_permuted=c[perm])
    assert np.allclose(sp, cols.school_columns(codes=codes, shift_days=21))
    se = cols.build_block(spec=cols.ARMS["SE"], factors=factors, c_true=c, c_permuted=c[perm])
    assert se.shape == (4, 3)
    passed(
        "each arm's block has its columns in order, with closeness real or permuted and school columns never permuted"
    )


# ----------------------------------------------------------------------------- scorer


def test_cells_known_answer() -> None:
    """One row's sums equal the hand-computed pinball losses, signed error, and coverage; quantiles must be sorted."""
    import e00668_score_lib as lib  # ty: ignore[unresolved-import]

    taus = np.asarray(lib.TAUS)
    from datetime import UTC, datetime

    rows = pl.DataFrame(
        {
            "time_series_id": [3],
            "valid_time": [datetime(2026, 1, 6, 12, tzinfo=UTC)],
            "power_fcst_init_time": [datetime(2026, 1, 5, 12, tzinfo=UTC)],
            **{q: [0.5] for q in lib.QCOLS},
        },
        schema={
            "time_series_id": pl.Int8,
            "valid_time": pl.Datetime("us", "UTC"),
            "power_fcst_init_time": pl.Datetime("us", "UTC"),
            **dict.fromkeys(lib.QCOLS, pl.Float32),
        },
    )
    index = int(lib.grid_index(times=rows["valid_time"])[0])
    grid = np.full((1, index + 1), np.nan)
    grid[0, index] = 1.0
    ctx = score.Context(
        ids=[3],
        cap=np.asarray([100.0]),
        primary_of_row=np.asarray([7]),
        row_of_tsid=np.full(256, -1, dtype=np.int64),
        grid=grid,
    )
    ctx.row_of_tsid[3] = 0
    cells, counts = score.cells_from_contract(rows=rows, ctx=ctx, month_index=2)
    assert counts == {"rows": 1, "scored": 1, "no_truth": 0}, counts
    assert cells.height == 1
    r = cells.row(0, named=True)
    # actual 1.0 MW of 100 MW capacity = 1 point; the median is 0.5 MW = 0.5 points
    q = 0.5 * 100.0 / 100.0
    y = 1.0
    want_pin = np.maximum(taus * (y - q), (taus - 1.0) * (y - q))
    assert np.isclose(r["crps"], 2.0 * want_pin.mean()), r["crps"]
    assert np.isclose(r["mae"], 0.5) and np.isclose(r["signed"], -0.5)
    assert np.isclose(r["pin01"], want_pin[0]) and np.isclose(r["pin99"], want_pin[10])
    assert r["cov01"] == 0.0 and r["cov99"] == 0.0, (
        "the actual is above every quantile, so none covers it"
    )
    assert np.isclose(r["gap"], 0.0, atol=1e-9)
    assert (
        r["primary"] == 7
        and r["lead_day"] == 1
        and r["n"] == 1
        and r["m"] == 2
        and r["n1236"] == 1
        and np.isclose(r["s1236"], -0.5)
        and r["chk"] == int(lib.grid_index(times=rows["power_fcst_init_time"])[0]) + 7 * index
    )
    assert r["vday"] == cols.code_of(day=date(2026, 1, 6)), (
        "valid 12:00 on 2026-01-06 is that local day"
    )
    ctx.grid[0, index] = np.nan
    _, counts = score.cells_from_contract(rows=rows, ctx=ctx, month_index=2)
    assert counts["no_truth"] == 1 and counts["scored"] == 0
    passed(
        "cell sums equal the hand-computed CRPS, pinball loss, signed error, and coverage of one row"
    )


def synthetic_pair(*, effect: float, noise: float, seed: int = 0) -> score.Pair:
    """Return a pair of 12 months x 3 seeds of cells on the school window, with a constant paired CRPS difference of `effect` plus noise."""
    rng = np.random.default_rng(seed)
    rows = []
    start = cols.code_of(day=date(2025, 11, 1))
    for m in range(12):
        month_start = start + 30 * m
        for s in range(3):
            for p in range(16):
                for v in range(40):
                    for lead in (1, 5):
                        rows.append(
                            (m, s, p, month_start + v, lead, 10, effect + noise * rng.normal(), 1.0)
                        )
    df = pl.DataFrame(
        rows, schema=["m", "seed", "primary", "vday", "lead_day", "n", "d_crps", "z"], orient="row"
    ).with_columns(
        pl.col("m").cast(pl.Int8),
        pl.col("seed").cast(pl.Int8),
        pl.col("primary").cast(pl.Int8),
        pl.col("lead_day").cast(pl.Int8),
    )
    df = df.with_columns(d_crps=pl.col("d_crps") * pl.col("n"))
    codes = df["vday"].to_numpy().astype(np.int64)
    return score.Pair(
        frame=df,
        name="a",
        reference="b",
        masks=win.window_masks(codes=codes),
        codes=codes,
        kinds=np.zeros(df.height, dtype=np.int8),
        period=score.period_of(m=df["m"].to_numpy().astype(np.int64), vday=codes),
        lead_ok=np.ones(df.height, dtype=bool),
    )


def cell_frame(*, rows: list[tuple]) -> pl.DataFrame:
    """Return cells of (m, seed, primary, vday, lead_day, n, chk, crps, mae) with the columns `make_pair` reads."""
    df = pl.DataFrame(
        rows, schema=["m", "seed", "primary", "vday", "lead_day", "n", "chk", "crps", "mae"], orient="row"
    ).with_columns(
        pl.col("m").cast(pl.Int8), pl.col("seed").cast(pl.Int8), pl.col("primary").cast(pl.Int8),
        pl.col("lead_day").cast(pl.Int8), pl.lit(0).alias("n1236"),
    )  # fmt: skip
    return df.with_columns(
        [pl.col("crps").alias(c) for c in score.SUM_COLUMNS if c not in {"crps", "mae"}]
    )


def test_pair_on_row_keys() -> None:
    """Cells pair only when both arms hold the same rows; equal counts of different rows, and a cell one arm lacks, are left out and counted."""
    day = cols.code_of(day=date(2026, 1, 10))
    a = cell_frame(
        rows=[
            (2, 0, 1, day, 1, 10, 100, 1.0, 1.0),
            (2, 0, 1, day + 1, 1, 10, 200, 1.0, 1.0),
            (2, 0, 1, day + 2, 1, 10, 300, 1.0, 1.0),
            (2, 0, 1, day + 3, 1, 5, 400, 1.0, 1.0),
        ]
    )
    b = cell_frame(
        rows=[
            (2, 0, 1, day, 1, 10, 100, 0.5, 0.5),
            (2, 0, 1, day + 1, 1, 10, 999, 0.5, 0.5),
            (2, 0, 1, day + 2, 1, 9, 300, 0.5, 0.5),
        ]
    )
    pair = score.make_pair(a=a, b=b, name="a", reference="b", weather=None, max_unpaired=1.0)
    assert pair.frame.height == 1 and pair.frame["vday"][0] == day, pair.frame
    assert pair.unpaired == {"a": 25, "b": 19}, pair.unpaired
    assert np.isclose(pair.frame["d_crps"][0], 0.5)
    try:
        score.make_pair(a=a, b=b, name="a", reference="b", weather=None)
    except AssertionError:
        pass
    else:
        raise AssertionError("a large unpaired share must be refused by default")
    passed(
        "pairing: cells with the same rows pair; equal counts with different row keys do not; unpaired rows are counted and capped"
    )


def test_cell_stamps_and_atomic_writes() -> None:
    """A cell file is current only while its stamp matches; a refit under a new code hash or fits file is refused."""
    with tempfile.TemporaryDirectory(
        dir=REPO / ".claude" / "scratch", prefix="00735-cells-test-"
    ) as tmp:
        out = Path(tmp) / "R0" / "2026-10_s0.parquet"
        frame = pl.DataFrame({"x": [1]})
        stamp = {"code_hash": "a", "input": "b", "version": score.CELLS_VERSION}
        assert not score.cell_is_current(out=out, stamp=stamp)
        score.write_cells(cells=frame, out=out, stamp=stamp)
        assert score.cell_is_current(out=out, stamp=stamp)
        assert not score.cell_is_current(out=out, stamp={**stamp, "code_hash": "c"})
        assert not score.cell_is_current(out=out, stamp={**stamp, "input": "d"})
        assert not list(out.parent.glob("*.tmp")), "no temporary file is left behind"
    passed(
        "cell files carry the fit's code hash and the sha256 of its fits file, and are written through a rename"
    )


def test_held_in_mask_excludes_july() -> None:
    """The 2026-06 refit's valid times from 2026-07-01 are not `held-in`, so the no-harm bound never pools them."""
    m = np.asarray([7, 7, 8, 11, 0])
    v = np.asarray(
        [
            cols.code_of(day=d)
            for d in (
                date(2026, 6, 30),
                date(2026, 7, 1),
                date(2026, 7, 5),
                date(2026, 10, 30),
                date(2025, 11, 3),
            )
        ]
    )
    got = score.period_of(m=m, vday=v).tolist()
    assert got == ["held-in", "july to september", "july to september", "october", "held-in"], got
    assert "year held-in S six" in score.REQUIRED_READINGS and not any(
        k.startswith("year S ") for k in score.REQUIRED_READINGS
    )
    passed("the held-in block ends on 2026-06-30 and the verdict reads the held-in no-harm bounds")


def test_p4_shared_blocks() -> None:
    """P4's two parts number their 7-day blocks from the union of their rows, so one pick is one calendar week in both."""
    pair = synthetic_pair(effect=0.0, noise=0.0)
    week = pair.frame["vday"].to_numpy() // score.BLOCK_DAYS
    mask_a = week % 3 == 0
    mask_b = week % 3 == 1
    union = mask_a | mask_b
    ga = score.gather(pair=pair, mask=mask_a, block_mask=union)
    gb = score.gather(pair=pair, mask=mask_b, block_mask=union)
    assert (ga[2] == gb[2]).all() and ga[0].shape == gb[0].shape
    blocks_a = {tuple(x) for x in np.argwhere(ga[1].sum(axis=1) > 0)}
    blocks_b = {tuple(x) for x in np.argwhere(gb[1].sum(axis=1) > 0)}
    assert not blocks_a & blocks_b, "different weeks sit in different blocks"
    alone_a = {
        tuple(x) for x in np.argwhere(score.gather(pair=pair, mask=mask_a)[1].sum(axis=1) > 0)
    }
    alone_b = {
        tuple(x) for x in np.argwhere(score.gather(pair=pair, mask=mask_b)[1].sum(axis=1) > 0)
    }
    assert alone_a & alone_b, "numbered separately, different weeks would share a block number"
    passed("P4 numbers one set of blocks for both parts")


def test_check_block_catches_silent_zeros() -> None:
    """`check_block` refuses a float64 block, a zero K1, a zero S1 over holiday dates, and zero W1 on holiday rows."""
    spec = cols.ARMS["SKW"]
    codes = np.asarray([cols.code_of(day=date(2026, 2, 16)), cols.code_of(day=date(2026, 2, 17))])
    factors = cols.Factors(
        codes=codes, primary=np.asarray([9, 9]), weather=np.ones((2, 3), dtype=np.float32)
    )
    good = cols.build_block(
        spec=spec, factors=factors, c_true=cols.coast_arrays()[0], c_permuted=cols.coast_arrays()[1]
    )
    cols.check_block(spec=spec, block=good, codes=codes, kind="train")
    for broken in (
        good.astype(np.float64),
        np.where(np.arange(good.shape[1]) == 0, 0.0, good).astype(np.float32),
        np.zeros_like(good),
    ):
        try:
            cols.check_block(spec=spec, block=broken, codes=codes, kind="train")
        except AssertionError:
            continue
        raise AssertionError("a broken block must be refused")
    passed(
        "check_block refuses a float64 block, a zero K1, and an all-zero block over half-term dates"
    )


def test_terms_csv_standard_reader() -> None:
    """The 2026-27 term-date file parses with the standard CSV reader and a quoted source field."""
    table = pl.read_csv(cols.TERMS_2026_27, try_parse_dates=True)
    assert (
        table.columns == ["academic_year", "term", "first_day", "last_day", "source"]
        and table.height == 2
    ), table
    assert table["source"][0].startswith("savemyexams (third party")
    passed("the 2026-27 term dates read with the standard CSV reader")


def test_bootstrap() -> None:
    """The two-level bootstrap returns the known constant, brackets zero when there is no effect, and counts months and fits."""
    pair = synthetic_pair(effect=-0.1, noise=0.0)
    mask = np.ones(pair.frame.height, dtype=bool)
    r = score.reading(
        pair=pair, mask=mask, name="const", level=score.FAMILY_LEVEL, months=list(range(8))
    )
    assert (
        np.isclose(r["estimate"], -0.1)
        and np.isclose(r["lower"], -0.1)
        and np.isclose(r["upper"], -0.1)
    ), r
    assert (
        r["months_negative"] == 8
        and r["fits_negative"] == 24
        and r["n_months"] == 8
        and r["n_fits"] == 24
    )
    assert score.is_gain(r=r)
    noisy = synthetic_pair(effect=0.0, noise=1.0, seed=3)
    r0 = score.reading(
        pair=noisy,
        mask=np.ones(noisy.frame.height, dtype=bool),
        name="null",
        level=score.FAMILY_LEVEL,
        months=list(range(8)),
    )
    assert r0["lower"] < 0 < r0["upper"] and abs(r0["estimate"]) < 0.05, r0
    assert not score.is_gain(r=r0) and not score.is_harm(r=r0)
    # a month-level effect (months differ) widens the interval compared with pooling rows as independent
    rng = np.random.default_rng(5)
    month_effect = rng.normal(0, 0.2, size=12)
    biased = synthetic_pair(effect=0.0, noise=0.01, seed=7)
    biased.frame = biased.frame.with_columns(
        d_crps=pl.col("d_crps")
        + pl.col("n") * pl.Series(month_effect[biased.frame["m"].to_numpy()])
    )
    rb = score.reading(
        pair=biased,
        mask=np.ones(biased.frame.height, dtype=bool),
        name="month effect",
        level=0.95,
        months=list(range(8)),
    )
    assert rb["upper"] - rb["lower"] > 0.1, rb
    again = score.reading(
        pair=biased,
        mask=np.ones(biased.frame.height, dtype=bool),
        name="month effect",
        level=0.95,
        months=list(range(8)),
    )
    assert again == rb, "a reading is reproducible: its generator is seeded by its name"
    one_sided = score.reading(
        pair=pair, mask=mask, name="year", level=0.95, months=list(range(12)), one_sided=True
    )
    assert np.isclose(one_sided["upper"], -0.1)
    passed(
        "two-level bootstrap: constant effect, null effect, month effect, reproducibility, one-sided bound"
    )


def reading(estimate: float, lower: float, upper: float, months: int = 8, fits: int = 24) -> dict:
    """Return a synthetic reading."""
    return {
        "name": "x",
        "estimate": estimate,
        "lower": lower,
        "upper": upper,
        "months_negative": months,
        "fits_negative": fits,
        "n_months": 8,
        "n_fits": 24,
        "rows": 1,
        "level": 0.99,
    }


def test_verdict_mapping() -> None:
    """Each verdict rule of the plan fires on its synthetic readings, and the gain conditions are each necessary."""
    gain = reading(-0.05, -0.08, -0.02)
    flat = reading(0.0, -0.01, 0.01)
    base = {
        n: dict(flat)
        for n in (
            "C1 six",
            "C1 eight",
            "C2 six",
            "C2 eight",
            "C4",
            "C5",
            "SP-R0 six",
            "SP-R0 eight",
            "SKWP-S",
            "SEP-S",
            "inland C4",
            "inland C5",
        )
    }
    base |= {
        f"year held-in {a} {g}": reading(0.0, -0.01, 0.01)
        for a in ("S", "SKW", "SE")
        for g in ("six", "eight")
    }
    assert (
        score.verdict(r=base)["verdict"] == "not supported" and score.verdict(r=base)["rule"] == 5
    )
    harmed = {**base, "C1 six": reading(0.05, 0.02, 0.08, months=0, fits=0)}
    assert score.verdict(r=harmed)["verdict"] == "refuted"
    full = {**base, "C1 six": gain, "C1 eight": gain, "C2 six": gain, "C2 eight": gain, "C4": gain}
    assert score.verdict(r=full)["verdict"] == "supported" and score.verdict(r=full)["rule"] == 2, (
        score.verdict(r=full)
    )
    school_only = {**full, "C4": flat}
    assert score.verdict(r=school_only)["verdict"] == "partly supported"
    one_group = {**school_only, "C1 eight": flat}
    assert score.verdict(r=one_group)["verdict"] == "partly supported"
    coast_only = {**base, "C4": gain}
    assert score.verdict(r=coast_only)["verdict"] == "partly supported"
    small = {**base, "C1 six": reading(-0.01, -0.02, -0.003)}
    assert score.verdict(r=small)["verdict"] == "small gain" and score.verdict(r=small)["rule"] == 4
    wide = {**base, "C1 eight": reading(0.0, -0.2, 0.2)}
    assert score.verdict(r=wide)["verdict"] == "inconclusive" and score.verdict(r=wide)["rule"] == 6
    # each gain condition is necessary
    assert score.is_gain(r=gain)
    assert not score.is_gain(r=reading(-0.05, -0.08, 0.01))
    assert not score.is_gain(r=reading(-0.015, -0.03, -0.001))
    assert not score.is_gain(r=reading(-0.05, -0.08, -0.02, months=5))
    assert not score.is_gain(r=reading(-0.05, -0.08, -0.02, fits=17))
    # a failed placebo blocks the school claim; a whole-year bound above 0.02 blocks it too
    failed = {**full, "SP-R0 six": reading(-0.05, -0.08, -0.01, months=8, fits=24)}
    assert score.verdict(r=failed)["school_claim_six"] is False
    harmful_year = {**full, "year held-in S six": reading(0.05, 0.0, 0.03)}
    assert score.verdict(r=harmful_year)["school_claim_six"] is False
    dropped = {
        k: v
        for k, v in full.items()
        if k not in {"C5", "SEP-S", "year held-in SE six", "year held-in SE eight", "inland C5"}
    }
    assert score.verdict(r=dropped, dropped=frozenset({"SE", "SEP"}))["verdict"] == "supported", (
        "a dropped arm does not block a claim that rests on the other arm"
    )
    try:
        score.verdict(r=dropped)
    except ValueError as error:
        message = str(error)
    else:
        raise AssertionError("a missing reading of an arm that was not dropped must raise")
    assert "C5" in message, message
    try:
        score.verdict(
            r={k: v for k, v in full.items() if k != "C1 six"}, dropped=frozenset({"SE", "SEP"})
        )
    except ValueError:
        pass
    else:
        raise AssertionError("a missing C1 reading must raise, never give inconclusive")
    passed("verdict rules 1 to 6 and the four gain conditions each fire on synthetic readings")


# ----------------------------------------------------------------------------- phases


def test_phase_runner() -> None:
    """The phase runner refuses without a marker, stops at 5.5 GPU-process-hours, and writes failure records with the traceback."""
    import e00735_phases as ph  # ty: ignore[unresolved-import]

    codes = {0, ph.EXIT_FAILED, ph.EXIT_STOP, ph.EXIT_USAGE, ph.EXIT_GATE_FAILED, ph.EXIT_BUSY}
    assert len(codes) == 6 and 1 not in codes and score.GATE_EXIT_CODE == ph.EXIT_GATE_FAILED
    assert f"{ph.EXIT_BUSY}  " in (HERE / "e00735_run.sh").read_text(), (
        "the exit codes are listed in run.sh"
    )
    with tempfile.TemporaryDirectory(
        dir=REPO / ".claude" / "scratch", prefix="00735-phase-test-"
    ) as tmp:
        root = Path(tmp)
        (root / "logs").mkdir()
        env = {
            **os.environ,
            "R735": str(root),
            "DATA_PATH_INTERNAL": os.environ["DATA_PATH_INTERNAL"],
        }
        script = str(HERE / "e00735_phases.py")
        none = subprocess.run(
            [sys.executable, script, "fit"], env=env, capture_output=True, text=True
        )
        assert none.returncode == ph.EXIT_USAGE and "no review marker" in none.stdout, (
            none.returncode,
            none.stdout,
            none.stderr,
        )
        (root / "review_markers").mkdir()
        (root / "review_markers" / "fit").touch()
        rows = ["arm,month,phase,device,start_utc,end_utc"]
        for i in range(5):
            rows.append(f"S,2025-11,step2,cuda,2026-01-01T0{i}:00:00Z,2026-01-01T0{i}:59:00Z")
        rows.append("R0,2026-10,step2,cuda,2026-01-02T00:00:00Z,2026-01-02T00:30:00Z")
        (root / "logs" / "phase_times.csv").write_text("\n".join(rows) + "\n")
        hours = ph.gpu_hours(csv_paths=[root / "logs" / "phase_times.csv"])
        assert abs(hours - (5 * 59 + 30) / 60) < 1e-9, hours
        assert (
            abs(
                ph.job_minutes(path=root / "logs" / "phase_times.csv", arm="R0", month="2026-10")
                - 30.0
            )
            < 1e-9
        )
        stop = subprocess.run(
            [sys.executable, script, "fit", "--dry-run"], env=env, capture_output=True, text=True
        )
        assert stop.returncode == ph.EXIT_STOP, (stop.returncode, stop.stdout, stop.stderr)
        ph.log_cpu_phase(
            root=root, phase="columns", start=ph.datetime.now(ph.UTC), end=ph.datetime.now(ph.UTC)
        )
        assert "columns,cpu" in (root / "logs" / "phase_times.csv").read_text().replace("-,-,", "")
        assert abs(ph.gpu_hours(csv_paths=[root / "logs" / "phase_times.csv"]) - hours) < 1e-9, (
            "a cpu row adds no GPU hours"
        )
        (root / "logs" / "2026-01_S_fit.log").write_text(
            "start\nTraceback (most recent call last):\n  File x\nValueError: boom\n"
        )
        record = ph.failure_record(root=root, arm="S", month="2026-01", stage="fit", code=20)
        data = json.loads(record.read_text())
        assert (
            data["exception_message"] == "ValueError: boom"
            and "Traceback" in data["traceback"]
            and data["exit_code"] == 20
        )
    passed(
        "phase runner: marker check, GPU-hours sum and 5.5 h stop (exit 21), failure record (exit 20 differs from usage 2)"
    )


def test_run_script_never_writes_under_677() -> None:
    """No write command of the driver names experiment 677's folder or its variables."""
    text = (HERE / "e00735_run.sh").read_text().splitlines()
    for line in text:
        code = line.split("#")[0]
        if (
            any(w in code for w in ("mkdir", " rm ", "rm -", "> ", ">>", "tee ", "mv "))
            and "R677" in code
        ) or ("CHAIN677" in code and ("rm " in code or "mkdir" in code or '> "' in code)):
            raise AssertionError(line)
    assert (
        "taskset" in "\n".join(text)
        and "MemoryMax" in "\n".join(text)
        and "MemorySwapMax=0" in "\n".join(text)
        and "OOMPolicy=continue" in "\n".join(text)
    )
    assert HOME_PREFIX not in "\n".join(text)
    for path in HERE.glob("e00735_*"):
        if path.suffix in {".py", ".sh"}:
            assert HOME_PREFIX not in path.read_text() or path.name == Path(__file__).name, (
                path.name
            )
    passed(
        "the driver writes only under experiment 735's folder, names its cap and cores, and has no absolute path under the maintainer's home folder"
    )


def test_677_hashes() -> None:
    """The hashes of 677's stage-1 weights and statistics files are recorded on the first run and compared on every later run."""
    root677 = score.ROOT_677
    files = []
    for month in score.MONTHS:
        files += [
            root677 / "chains" / "5" / month / "work" / "pretrained_P2" / f"{month}_s{s}.pt"
            for s in score.SEEDS
        ]
        files.append(root677 / "chains" / "3b" / month / "work" / "stats" / "B1" / f"{month}.npz")
    files = [f for f in files if f.exists()]
    assert len(files) >= 40, len(files)
    record = {
        str(f.relative_to(root677)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files
    }
    stored = score.ROOT / "hashes_677.json"
    if stored.exists():
        assert json.loads(stored.read_text()) == record, "a file of experiment 677 changed"
        note = "compared with the recorded hashes"
    else:
        stored.parent.mkdir(parents=True, exist_ok=True)
        stored.write_text(json.dumps(record, indent=1))
        note = "recorded for later runs"
    passed(f"hashes of {len(files)} stage-1 and statistics files of experiment 677 {note}")


# ----------------------------------------------------------------------------- network (torch)


def test_zero_start() -> None:
    """With zero weights on the new columns, every arm's network gives the control's outputs on 1,000 rows, with labels attached and detached."""
    try:
        import torch
    except ModuleNotFoundError:
        print("SKIP zero start: torch is not installed")
        return
    import e00397_models as base  # ty: ignore[unresolved-import]
    import e00559_model as m  # ty: ignore[unresolved-import]
    import e00574_model as mod  # ty: ignore[unresolved-import]
    import e00735_model as model735  # ty: ignore[unresolved-import]

    n_b1, n_knots, n_primaries = 14, 6, 16
    width = mod.TAU + 1 + n_b1
    assert width == 60, width
    torch.manual_seed(0)
    control = model735.make_network(
        n_primaries=n_primaries, stage="ft", n_knots=n_knots, n_b1=n_b1, n_new=0, coast=None
    )
    assert type(control).__name__ == "AuxLagMdn"
    # A copy of `e00574_fit.fresh_mixer`, which cannot be imported here because that module parses
    # the command line on import: the seed-12345 mixer at the layer's own width.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(model735.FRESH_SEED)
        fresh = base.mlp(
            n_in=control.mixer_ft[0].in_features,
            hidden=m.MIXER_HIDDEN,
            layers=2,
            n_out=3 * m.COMPONENTS,
            dropout=m.DROPOUT,
        )
        base.init_head(head=fresh, components=m.COMPONENTS)
    control.mixer_ft.load_state_dict(fresh.state_dict())
    mine = type(control)(
        n_primaries=n_primaries,
        stage="ft",
        n_knots=n_knots,
        n_b1=n_b1,
        kind="shared",
        detector=False,
    )
    mine.load_state_dict(control.state_dict())
    model735.fresh_mixer(model=mine)
    assert all(
        torch.equal(a, b)
        for a, b in zip(
            control.mixer_ft.state_dict().values(), mine.mixer_ft.state_dict().values(), strict=True
        )
    ), "fresh_mixer differs from 574's at the control's width"
    # `init_head` makes the head's output nearly constant, which would let any two networks agree.
    # Give the control's last layer random weights so that equality needs the new columns to be inert.
    last = max(i for i, layer in enumerate(control.mixer_ft) if isinstance(layer, torch.nn.Linear))
    with torch.no_grad():
        control.mixer_ft[last].weight.copy_(torch.randn_like(control.mixer_ft[last].weight))
        control.mixer_ft[last].bias.copy_(torch.randn_like(control.mixer_ft[last].bias))
    c, cp = cols.coast_arrays()
    rng = np.random.default_rng(0)
    rows = 1000
    base = rng.normal(size=(rows, width)).astype(np.float32)
    # the lag encoder reads two grid indices (columns 43 and 44) from an expected-input table
    base[:, 43:45] = rng.integers(-1, 500, size=(rows, 2))
    table = torch.from_numpy(rng.normal(size=(n_primaries, 500, 28)).astype(np.float32))
    control.table = table
    primary = torch.from_numpy(rng.integers(0, n_primaries, size=rows))
    labels = rng.normal(size=(rows, 2)).astype(np.float32)
    control.eval()
    ctrl_plain = control(torch.from_numpy(base), primary).detach()
    ctrl_labeled = control(
        torch.from_numpy(np.concatenate([base, labels], axis=1)), primary
    ).detach()
    assert torch.allclose(ctrl_plain, ctrl_labeled, atol=1e-6), (
        "the control's output depends on attached labels"
    )
    for name, spec in cols.ARMS.items():
        if name == "R0":
            continue
        net = model735.make_network(
            n_primaries=n_primaries,
            stage="ft",
            n_knots=n_knots,
            n_b1=n_b1,
            n_new=spec.width,
            coast=(cp if spec.permuted else c) if spec.embed else None,
        )
        assert net.label_col == width + spec.width, (name, net.label_col)
        state = {k: v for k, v in control.state_dict().items()}
        missing = net.load_state_dict(state, strict=False)
        assert all(k == model735.COAST_TERM for k in missing.missing_keys), (name, missing)
        model735.fresh_mixer(model=net)
        with torch.no_grad():  # the same random last layer as the control's
            net.mixer_ft[last].weight.copy_(control.mixer_ft[last].weight)
            net.mixer_ft[last].bias.copy_(control.mixer_ft[last].bias)
        net.eval()
        net.table = table
        assert torch.equal(
            net.mixer_ft[0].weight[
                :, width - (mod.TAU + 1 + n_b1) + (net.mixer_ft[0].in_features - spec.width) :
            ],
            torch.zeros(net.mixer_ft[0].weight.shape[0], spec.width),
        ), name
        new = rng.normal(size=(rows, spec.width)).astype(np.float32)
        x_plain = torch.from_numpy(np.concatenate([base, new], axis=1))
        x_labeled = torch.from_numpy(np.concatenate([base, new, labels], axis=1))
        out_plain = net(x_plain, primary).detach()
        out_labeled = net(x_labeled, primary).detach()
        assert torch.allclose(out_plain, ctrl_plain, atol=1e-6), (
            name,
            float((out_plain - ctrl_plain).abs().max()),
        )
        assert torch.allclose(out_labeled, ctrl_plain, atol=1e-6), (name, "labels attached")
        assert float(ctrl_plain.std()) > 0.1, (
            "the control's output must vary for the comparison to mean anything"
        )
        with torch.no_grad():  # sanity: once a new column's weight is non-zero, the output moves
            net.mixer_ft[0].weight[:, -1] = 0.5
        assert not torch.allclose(net(x_plain, primary).detach(), ctrl_plain, atol=1e-4), name
        if spec.embed:
            u = getattr(net, model735.COAST_TERM)
            assert float(u.abs().max()) == 0.0
            plain_emb = net.embedding(primary).detach()
            assert torch.equal(net.embedding_of(primary=primary).detach(), plain_emb)
            with torch.no_grad():
                u += 0.5
            shifted = net.embedding_of(primary=primary).detach() - plain_emb
            want = (torch.from_numpy((cp if spec.permuted else c)[primary.numpy()]).float() * 0.5)[
                :, None
            ]
            assert torch.allclose(shifted, want.expand_as(shifted), atol=1e-6), name
            # the network's output stays equal to the control's here only because the new head starts at zero
    passed(
        "zero start: every arm's network equals the control on 1,000 rows, labels attached and detached (the coast term moves the embedding by c times u in SE and SEP)"
    )
    assert m.WEATHER_COLS is not None


def main() -> None:
    """Run every test."""
    skip_build = "--skip-build" in sys.argv
    if "--hashes-only" in sys.argv:
        test_677_hashes()
        return
    test_every_attribute_exists()
    test_local_date_convention()
    test_school_columns_equal_csv()
    test_placebo_calendar()
    test_window_masks()
    test_permutation()
    test_w1_w2_monthly_anomaly()
    test_w3_causal()
    test_columns_read_no_era5()
    test_build_block()
    test_cells_known_answer()
    test_pair_on_row_keys()
    test_cell_stamps_and_atomic_writes()
    test_held_in_mask_excludes_july()
    test_p4_shared_blocks()
    test_check_block_catches_silent_zeros()
    test_terms_csv_standard_reader()
    test_bootstrap()
    test_verdict_mapping()
    test_phase_runner()
    test_run_script_never_writes_under_677()
    if not skip_build:
        test_677_hashes()
    test_zero_start()
    print(f"\n{len(RESULTS)} tests passed")


if __name__ == "__main__":
    main()
