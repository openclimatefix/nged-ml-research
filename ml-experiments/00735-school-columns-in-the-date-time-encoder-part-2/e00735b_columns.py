# ruff: noqa: E501, PT018
"""Experiment 735 part 2: the 8 school and coast columns of the date-time encoder (CPU only, no torch).

README. This module builds the 8 columns that the arms SKE, SKE-S21, and SKE-CP append to the
date-time encoder's input, as a function of the valid time (microseconds since the epoch) and the
Primary index alone. It relies on experiment 735 part 1 (`e00735_columns`: the local-date convention,
`school_columns`' column definitions, `closeness`, `derange`, `coast_arrays`, and the dataclasses
`Block` and `DenseCalendar`) and on experiment 11 (`e00011_term_dates`, through part 1). PYTHONPATH:
source `ml-experiments/00524*/env.sh` from the worktree root.

The columns, with `c` the coast closeness of the row's Primary and every date the local date
(Europe/London) of the stamp minus 15 minutes:

- S1 `school_off`, S2 `school_first_weekend`, S3 `school_half_term` (blocks of at most 10 days);
- K1 `c`, K2 `c * S1`, K3 `c * S2`, K4 `c * bank_holiday`, K5 `c * weekend`.

The arms differ in three options. `school_shift_days` reads the school calendar that many days
earlier in S1 to S3 and in the school products K2 and K3 (SKE-S21 uses 21), and leaves K1, K4, and
K5 real. `permuted` replaces `c` by the closeness of another Primary in K1 to K5 (part 1's
permutation, seed 735). The columns are not standardised.

The school calendar is part 1's calendar extended backwards with
`lincolnshire_term_dates_2019_21.csv`, so that it starts on 2019-09-03 and every valid time from
2019-12-01 (the first ERA5 pre-training row) is covered, also when it is read 21 days earlier. A time
outside the calendar raises `ValueError`, except in `grid_table`, which fills such times with NaN so
that a lookup at an uncovered time cannot go unnoticed.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Final

import e00735_columns as p1  # ty: ignore[unresolved-import]
import holidays
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final = "00735-school-columns-in-the-date-time-encoder-part-2"
NAME_PART_1: Final = p1.NAME
HERE: Final[Path] = Path(__file__).resolve().parent
TERMS_2019_21: Final[Path] = HERE / "lincolnshire_term_dates_2019_21.csv"
PLACEBO_SHIFT_DAYS: Final = p1.PLACEBO_SHIFT_DAYS
"""The shifted school calendar reads the real calendar this many days earlier (21)."""
N_COLUMNS: Final = 8
COLUMN_NAMES: Final[tuple[str, ...]] = (
    "S1_school_off",
    "S2_school_first_weekend",
    "S3_school_half_term",
    "K1_c",
    "K2_c_school_off",
    "K3_c_first_weekend",
    "K4_c_bank_holiday",
    "K5_c_weekend",
)
SCHOOL_SLICE: Final = slice(0, 3)
COAST_SLICE: Final = slice(3, 8)
FIRST_PRETRAIN_DATE: Final = date(2019, 12, 1)
"""The first valid date of a stage-1 row; the calendar must cover it, also 21 days earlier."""
BANK_HOLIDAY_YEARS: Final = range(2019, 2028)
N_PRIMARIES: Final = p1.N_PRIMARIES


def check_output_root(*, root: str | Path) -> None:
    """Raise `AssertionError` unless `root` is this experiment's own folder or its smoke folder.

    The path must contain `school-columns-in-the-date-time-encoder-part-2` or `00735-smoke`, and
    must contain neither `coast-and-school` (part 1's folder) nor `00677` (experiment 677's).
    """
    text = str(root)
    assert "school-columns-in-the-date-time-encoder-part-2" in text or "00735-smoke" in text, text
    assert "coast-and-school" not in text, ("part 1's folder is immutable", text)
    assert "00677" not in text, ("experiment 677's folder is immutable", text)


def data_root() -> Path:
    """Return experiment 735 part 2's private data folder (`R735` when set, else the default)."""
    import os

    root = Path(os.environ.get("R735", ML_EXPERIMENTS_DATA_DIR / NAME))
    check_output_root(root=root)
    return root


@dataclass(frozen=True)
class ArmB:
    """What one arm appends to the date-time encoder's input."""

    name: str
    columns: bool
    school_shift_days: int = 0
    permuted: bool = False

    @property
    def width(self) -> int:
        """Return the number of columns the arm appends (8, or 0 for R0 and R1)."""
        return N_COLUMNS if self.columns else 0

    @property
    def names(self) -> tuple[str, ...]:
        """Return the appended columns' names in order."""
        return COLUMN_NAMES if self.columns else ()


ARMS: Final[dict[str, ArmB]] = {
    spec.name: spec
    for spec in (
        ArmB("R0", columns=False),
        ArmB("R1", columns=False),
        ArmB("SKE", columns=True),
        ArmB("SKE-S21", columns=True, school_shift_days=PLACEBO_SHIFT_DAYS),
        ArmB("SKE-CP", columns=True, permuted=True),
    )
}
"""R0 is experiment 677's stored control (no code runs for it); R1 re-runs it through this code."""
NEW_ARMS: Final = ("R1", "SKE", "SKE-S21", "SKE-CP")
"""The four arms whose three stages run in this experiment."""


# ----------------------------------------------------------------------------- the calendar


def terms() -> pl.DataFrame:
    """Return the term dates of 2019-20 to 2020-21, then part 1's table (2021-22 to 2026-27)."""
    rest = p1.terms()
    first = pl.read_csv(TERMS_2019_21, try_parse_dates=True)
    return pl.concat([first.select(rest.columns), rest]).sort("first_day")


def blocks() -> list[p1.Block]:
    """Return every school holiday block between two consecutive terms, after `validate_terms`."""
    table = terms()
    p1.validate_terms(table=table)
    firsts, lasts = table["first_day"].to_list(), table["last_day"].to_list()
    return [
        p1.Block(
            first=lasts[i] + timedelta(days=1),
            last=firsts[i + 1] - timedelta(days=1),
            next_term_first=firsts[i + 1],
        )
        for i in range(len(firsts) - 1)
    ]


_CALENDAR: list[p1.DenseCalendar] = []


def calendar() -> p1.DenseCalendar:
    """Return the dense calendar from 2019-09-03 to part 1's last day (built once per process).

    The block rules are part 1's (`Block`, half-term of at most 10 days, first weekend). The bank
    holidays are those of England from 2019, where part 1's own calendar starts them in 2021.
    """
    if _CALENDAR:
        return _CALENDAR[0]
    table = terms()
    first_day = table["first_day"].min()
    assert isinstance(first_day, date), first_day
    first_code = p1.code_of(day=first_day)
    last_code = p1.code_of(day=p1.CALENDAR_LAST)
    n = last_code - first_code + 1
    off = np.zeros(n, dtype=np.float32)
    first_weekend = np.zeros(n, dtype=np.float32)
    half_term = np.zeros(n, dtype=np.float32)
    block_index = np.full(n, -1, dtype=np.int32)
    for b, block in enumerate(blocks()):
        lo, hi = p1.code_of(day=block.first) - first_code, p1.code_of(day=block.last) - first_code
        hi = min(hi, n - 1)
        if lo > hi:
            continue
        off[lo : hi + 1] = 1.0
        block_index[lo : hi + 1] = b
        if block.days <= p1.HALF_TERM_MAX_DAYS:
            half_term[lo : hi + 1] = 1.0
        for d in block.first_weekend:
            if p1.code_of(day=d) - first_code <= n - 1:
                first_weekend[p1.code_of(day=d) - first_code] = 1.0
    bank = holidays.country_holidays("GB", subdiv="ENG", years=BANK_HOLIDAY_YEARS)
    bank_flag = np.zeros(n, dtype=np.float32)
    weekend = np.zeros(n, dtype=np.float32)
    for i in range(n):
        d = p1.date_of(code=first_code + i)
        bank_flag[i] = 1.0 if d in bank else 0.0
        weekend[i] = 1.0 if d.weekday() >= 5 else 0.0
    cal = p1.DenseCalendar(
        first_code=first_code,
        last_code=last_code,
        school_off=off,
        first_weekend=first_weekend,
        half_term=half_term,
        bank_holiday=bank_flag,
        weekend=weekend,
        block_index=block_index,
    )
    _CALENDAR.append(cal)
    return cal


# ----------------------------------------------------------------------------- the columns


def _factors(
    *, codes: np.ndarray, shift_days: int, strict: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return S1 to S3 (rows, 3), the factors of K1 to K5 before `c` (rows, 5), and the covered mask.

    The K factors are 1, S1, S2, the bank-holiday flag, and the weekend flag; S1 and S2 read the
    calendar `shift_days` earlier and the two flags read the row's own date. With `strict` a day
    outside the calendar (the row's own date or the shifted date) raises; without it such rows
    are NaN and not covered.
    """
    cal = calendar()
    codes = np.asarray(codes, dtype=np.int64)
    covered = (
        (codes - shift_days >= cal.first_code)
        & (codes >= cal.first_code)
        & (codes <= cal.last_code)
    )
    if strict:
        cal.index(codes=codes)
        cal.index(codes=codes - shift_days)
    safe = np.where(covered, codes, cal.first_code + shift_days)
    shifted = cal.index(codes=safe - shift_days)
    own = cal.index(codes=safe)
    school = np.stack(
        [cal.school_off[shifted], cal.first_weekend[shifted], cal.half_term[shifted]], axis=1
    )
    ones = np.ones(len(safe), dtype=np.float32)
    factors = np.stack(
        [ones, school[:, 0], school[:, 1], cal.bank_holiday[own], cal.weekend[own]], axis=1
    )
    school = np.where(covered[:, None], school, np.nan).astype(np.float32)
    factors = np.where(covered[:, None], factors, np.nan).astype(np.float32)
    return school, factors, covered


def coast_arrays() -> tuple[np.ndarray, np.ndarray]:
    """Return the true and the permuted closeness per Primary index (part 1's permutation, seed 735)."""
    c, permuted = p1.coast_arrays(save=False)
    assert len(c) == N_PRIMARIES, len(c)
    return c.astype(np.float32), permuted.astype(np.float32)


def extra_columns(
    *,
    valid_us: np.ndarray,
    primary: np.ndarray,
    arm: ArmB,
    c_true: np.ndarray,
    c_permuted: np.ndarray,
) -> np.ndarray:
    """Return the arm's 8 appended columns (rows, 8) as float32, from the valid time and the Primary.

    Args:
        valid_us: Period-ending UTC stamps of the rows' valid times, microseconds since the epoch.
        primary: The Primary index of each row.
        arm: The arm; an arm with no columns gets an empty (rows, 0) block.
        c_true: Closeness per Primary index.
        c_permuted: The placebo closeness per Primary index.

    Returns:
        The columns S1 to S3 and K1 to K5, in `COLUMN_NAMES` order.

    Raises:
        ValueError: If a valid time (or, for a shifted arm, the date 21 days earlier) lies outside
            the school calendar.
    """
    n = len(valid_us)
    if not arm.columns:
        return np.zeros((n, 0), dtype=np.float32)
    codes = p1.local_dates(stamp_us=valid_us).code.astype(np.int64)
    school, factors, _ = _factors(codes=codes, shift_days=arm.school_shift_days, strict=True)
    c = (c_permuted if arm.permuted else c_true)[np.asarray(primary, dtype=np.int64)]
    block = np.concatenate([school, factors * c.astype(np.float32)[:, None]], axis=1)
    assert block.shape == (n, N_COLUMNS) and block.dtype == np.float32, block.shape
    return block


def grid_table(
    *,
    arm: ArmB,
    first_us: int,
    step_us: int,
    n_grid: int,
    c_true: np.ndarray,
    c_permuted: np.ndarray,
) -> np.ndarray:
    """Return the arm's columns at every grid time and Primary: (Primaries, grid, 8) float32.

    The lag path of the network reads this table at each lag's grid index, so a lag row's columns
    equal the row path's columns at the same valid time and Primary. A grid time outside the
    calendar holds NaN.

    Args:
        arm: The arm with columns.
        first_us: The first grid time in microseconds since the epoch.
        step_us: The grid step in microseconds.
        n_grid: The number of grid times.
        c_true: Closeness per Primary index.
        c_permuted: The placebo closeness per Primary index.

    Returns:
        The table.
    """
    assert arm.columns
    stamps = first_us + np.arange(n_grid, dtype=np.int64) * step_us
    codes = p1.local_dates(stamp_us=stamps).code.astype(np.int64)
    school, factors, _ = _factors(codes=codes, shift_days=arm.school_shift_days, strict=False)
    c = (c_permuted if arm.permuted else c_true).astype(np.float32)
    n_primaries = len(c)
    out = np.empty((n_primaries, n_grid, N_COLUMNS), dtype=np.float32)
    for p in range(n_primaries):
        out[p, :, SCHOOL_SLICE] = school
        out[p, :, COAST_SLICE] = factors * c[p]
    return out


def check_block(*, arm: ArmB, block: np.ndarray, valid_us: np.ndarray, kind: str) -> None:
    """Raise `AssertionError` if an appended block would be silently zero, NaN, or the wrong type.

    The checks: float32 with the arm's width and no NaN; K1 is non-zero on every row; and, for
    training rows, each of S1, S2, and S3 is 0 on some row and 1 on some row (a test month is
    short, so for `test` rows only S1 is checked, and only when the month holds a holiday date).

    Args:
        arm: The arm.
        block: The appended columns.
        valid_us: The rows' valid times.
        kind: `train` or `test`.
    """
    assert block.dtype == np.float32, (kind, block.dtype)
    assert block.shape == (len(valid_us), arm.width), (kind, block.shape, arm.width)
    if not arm.columns or not len(valid_us):
        return
    assert np.isfinite(block).all(), (kind, "a column holds NaN or infinity")
    assert (block[:, 3] != 0).all(), (kind, "K1 is zero on some row")
    if kind == "train":
        for j in range(3):
            assert block[:, j].min() == 0.0 and block[:, j].max() == 1.0, (
                kind,
                COLUMN_NAMES[j],
                "is constant over the training rows",
            )
    else:
        codes = np.unique(p1.local_dates(stamp_us=valid_us).code.astype(np.int64))
        cal = calendar()
        shifted = cal.index(codes=codes - arm.school_shift_days)
        if cal.school_off[shifted].any():
            assert (block[:, 0] != 0).any(), (kind, "S1 is zero though holiday dates are present")
