# ruff: noqa: E501, PT018
"""Experiment 735: the new input columns of the step-5 network (CPU only, no torch).

README. This module builds every column that experiment 735 adds to the step-5 mixer, and the arm
table that says which columns each arm gets. It relies on experiment 11 (`e00011_term_dates`: the
term dates of 2021-22 to 2025-26), experiment 735's own `lincolnshire_term_dates_2026_27.csv`,
`e00735_diagnostic` (the coast distance of each Primary), `e00397_prep` (the Primary table),
`e00087_common` (the half-hourly grid), `e00422_prep` (the grid origin), and `e00418_prep` is not
needed here. PYTHONPATH: source `ml-experiments/00524*/env.sh` from the worktree root.

One local-date convention holds for every new column: the local date (Europe/London) of the stamp
minus 15 minutes. Stamps mark the end of a half-hour, so the stamp 00:00 belongs to the day that
just ended and the stamp 00:30 to the day that has just begun. The existing `bank_holiday` and
`day_of_week_local` columns of the network use the stamp itself and so label the 00:00 stamp as the
next day. The new columns recompute the bank-holiday and weekend flags from the shifted date and
never multiply by an existing column.

The columns, with `c` the coast closeness of the row's Primary:

- S1 `school_off`, S2 `school_first_weekend`, S3 `school_half_term` (blocks of at most 10 days);
- K1 `c`, K2 `c * S1`, K3 `c * S2`, K4 `c * bank_holiday`, K5 `c * weekend`;
- W1 `c * S1 * warm_for_the_month`, W2 `c * S1 * sunny_for_the_month`, W3 `c * S1 * first_warm_day`,
  W4 `c * warm_for_the_month`.

Weather note: the B1 column `t_max_day` (input index 52) is in degrees Celsius, not kelvin, and so
is the expected-weather table's temperature. W3 is a difference of two temperatures, so the unit
cancels.
"""

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Final, Literal

import e00011_term_dates as term_dates  # ty: ignore[unresolved-import]
import holidays
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final = "00735-coast-and-school-holiday-signals-in-the-step-5-network-part-1"
HERE: Final[Path] = Path(__file__).resolve().parent
ROOT: Final[Path] = ML_EXPERIMENTS_DATA_DIR / NAME
"""The private data folder of this experiment."""
TERMS_2026_27: Final[Path] = HERE / "lincolnshire_term_dates_2026_27.csv"
EPOCH: Final = date(1970, 1, 1)
CALENDAR_LAST: Final = date(2026, 11, 30)
"""The school calendar ends here, because the dates after 2026-12-18 are unknown."""
PERMUTATION_SEED: Final = 735
PLACEBO_SHIFT_DAYS: Final = 21
"""The placebo school calendar reads the real calendar this many days earlier."""
HALF_TERM_MAX_DAYS: Final = 10
COAST_SCALE_KM: Final = 5.0
MAX_PERMUTED_CORRELATION: Final = 0.3
N_PRIMARIES: Final = 16
ANOMALY_CLIP: Final = 3.0
W3_SCALE_K: Final = 5.0
W3_DAYS: Final = 7
W3_MIN_DAYS: Final = 4
"""The fewest complete days that W3's seven-day baseline needs; fewer gives W3 = 0."""
MIN_ROWS_PER_CELL: Final = 200
"""The fewest training rows in a Primary-month cell for W1 and W2 to use that cell's mean."""
HOUR_US: Final = 3_600_000_000
STAMP_OFFSET_US: Final = 15 * 60 * 1_000_000
B1_FIRST_COLUMN: Final = 46
B1_TMAX_COLUMN: Final = 52
B1_SW_COLUMN: Final = 56
B1_TMAX_POSITION: Final = B1_TMAX_COLUMN - B1_FIRST_COLUMN
B1_SW_POSITION: Final = B1_SW_COLUMN - B1_FIRST_COLUMN
EXPECTED_TEMPERATURE_COLUMN: Final = 2
"""`mean_temperature_2m` in the expected-weather table's weather columns (0 to 6)."""

SchoolMode = Literal["none", "real", "shifted"]


@dataclass(frozen=True)
class ArmSpec:
    """What one arm adds to the mixer and to the embedding."""

    name: str
    school: SchoolMode
    coast_columns: bool = False
    weather_columns: bool = False
    permuted: bool = False
    embed: bool = False

    @property
    def width(self) -> int:
        """Return the number of columns the arm appends to the mixer input."""
        return (
            (0 if self.school == "none" else 3)
            + (5 if self.coast_columns else 0)
            + (4 if self.weather_columns else 0)
        )

    @property
    def names(self) -> list[str]:
        """Return the column names in the order the block holds them."""
        out: list[str] = []
        if self.school != "none":
            out += ["S1_school_off", "S2_school_first_weekend", "S3_school_half_term"]
        if self.coast_columns:
            out += [
                "K1_c",
                "K2_c_school_off",
                "K3_c_first_weekend",
                "K4_c_bank_holiday",
                "K5_c_weekend",
            ]
        if self.weather_columns:
            out += ["W1_c_school_warm", "W2_c_school_sunny", "W3_c_school_first_warm", "W4_c_warm"]
        return out


ARMS: Final[dict[str, ArmSpec]] = {
    spec.name: spec
    for spec in (
        ArmSpec("R0", "none"),
        ArmSpec("S", "real"),
        ArmSpec("SP", "shifted"),
        ArmSpec("SKW", "real", coast_columns=True, weather_columns=True),
        ArmSpec("SKWP", "real", coast_columns=True, weather_columns=True, permuted=True),
        ArmSpec("SE", "real", embed=True),
        ArmSpec("SEP", "real", embed=True, permuted=True),
    )
}
"""The seven arms. R0 adds nothing; SP, SKWP, and SEP are the placebo twins of S, SKW, and SE."""
ARM_ORDER: Final = tuple(ARMS)


# ----------------------------------------------------------------------------- local dates


@dataclass(frozen=True)
class LocalDates:
    """The local calendar of a set of stamps, by the stamp-minus-15-minutes convention."""

    code: np.ndarray
    """Days since 1970-01-01 of the local date (int32)."""
    month: np.ndarray
    """Local calendar month, 1 to 12 (int8)."""
    weekday: np.ndarray
    """Local weekday, Monday 0 to Sunday 6 (int8)."""


def local_dates(*, stamp_us: np.ndarray) -> LocalDates:
    """Return the local date of each stamp (Europe/London, stamp minus 15 minutes).

    Args:
        stamp_us: Period-ending UTC stamps in microseconds since the epoch.

    Returns:
        The day code, month, and weekday of each stamp's local date.
    """
    shifted = (
        pl.Series("t", np.asarray(stamp_us, dtype=np.int64))
        .cast(pl.Datetime("us"))
        .dt.replace_time_zone("UTC")
        .dt.offset_by("-15m")
        .dt.convert_time_zone("Europe/London")
    )
    return LocalDates(
        code=shifted.dt.date().cast(pl.Int32).to_numpy(),
        month=shifted.dt.month().cast(pl.Int8).to_numpy(),
        weekday=(shifted.dt.weekday() - 1).cast(pl.Int8).to_numpy(),
    )


def is_local_midnight(*, stamp_us: np.ndarray) -> np.ndarray:
    """Return True where the stamp is exactly 00:00 Europe/London, the last stamp of a local day."""
    local = (
        pl.Series("t", np.asarray(stamp_us, dtype=np.int64))
        .cast(pl.Datetime("us"))
        .dt.replace_time_zone("UTC")
        .dt.convert_time_zone("Europe/London")
    )
    return ((local.dt.hour() == 0) & (local.dt.minute() == 0)).to_numpy()


def code_of(*, day: date) -> int:
    """Return a date as days since 1970-01-01."""
    return (day - EPOCH).days


def date_of(*, code: int) -> date:
    """Return days since 1970-01-01 as a date."""
    return EPOCH + timedelta(days=int(code))


# ----------------------------------------------------------------------------- school calendar


def terms() -> pl.DataFrame:
    """Return the term dates of experiment 11's csv and of the 2026-27 csv, in date order."""
    first = term_dates.read_terms()
    # The source column of this csv holds an unquoted comma, so ragged lines are truncated.
    second = pl.read_csv(TERMS_2026_27, try_parse_dates=True, truncate_ragged_lines=True)
    return pl.concat([first.select(second.columns), second]).sort("first_day")


def validate_terms(*, table: pl.DataFrame) -> None:
    """Raise `ValueError` if the combined term dates are out of order, overlap, or look implausible.

    The checks: terms are listed in date order and never overlap; every term starts and ends on a
    weekday; every term lasts 4 to 8 weeks; the gap before a term 1 is 30 to 60 days and every
    other gap is 6 to 20 days. Experiment 11's own `validate` needs six terms per academic year, so
    the two-term 2026-27 file fails it.
    """
    problems: list[str] = []
    firsts, lasts = table["first_day"].to_list(), table["last_day"].to_list()
    for i, (a, b) in enumerate(zip(firsts, lasts, strict=True)):
        label = f"{table['academic_year'][i]} term {table['term'][i]}"
        if a.weekday() >= 5 or b.weekday() >= 5:
            problems.append(f"{label} starts or ends on a weekend")
        if not 4 <= ((b - a).days + 1) / 7 <= 8:
            problems.append(f"{label} lasts {(b - a).days + 1} days")
        if i:
            if a <= lasts[i - 1]:
                problems.append(f"{label} overlaps or precedes the term before it")
            gap = (a - lasts[i - 1]).days - 1
            lo, hi = (30, 60) if table["term"][i] == 1 else (6, 20)
            if not lo <= gap <= hi:
                problems.append(f"gap before {label} is {gap} days")
    if problems:
        raise ValueError("; ".join(problems))


@dataclass(frozen=True)
class Block:
    """A school holiday block: the dates from the day after a term's last day to the day before the next term."""

    first: date
    last: date
    next_term_first: date

    @property
    def days(self) -> int:
        """Return the length of the block in days, weekends included."""
        return (self.last - self.first).days + 1

    @property
    def first_weekend(self) -> tuple[date, ...]:
        """Return the first Saturday and Sunday of the block that lie inside it."""
        saturday = self.first + timedelta(days=(5 - self.first.weekday()) % 7)
        return tuple(d for d in (saturday, saturday + timedelta(days=1)) if d <= self.last)


def blocks() -> list[Block]:
    """Return every school holiday block between two consecutive terms of the combined table."""
    table = terms()
    validate_terms(table=table)
    firsts, lasts = table["first_day"].to_list(), table["last_day"].to_list()
    return [
        Block(
            first=lasts[i] + timedelta(days=1),
            last=firsts[i + 1] - timedelta(days=1),
            next_term_first=firsts[i + 1],
        )
        for i in range(len(firsts) - 1)
    ]


@dataclass(frozen=True)
class DenseCalendar:
    """The calendar as arrays over `first_code` to `last_code`, indexed by day code minus `first_code`."""

    first_code: int
    last_code: int
    school_off: np.ndarray
    first_weekend: np.ndarray
    half_term: np.ndarray
    bank_holiday: np.ndarray
    weekend: np.ndarray
    block_index: np.ndarray
    """Index into `blocks()` of the block that holds the date, or -1 in term time."""

    def index(self, *, codes: np.ndarray) -> np.ndarray:
        """Return array positions of day codes, raising `ValueError` outside the calendar."""
        idx = np.asarray(codes, dtype=np.int64) - self.first_code
        if idx.size and (idx.min() < 0 or idx.max() > self.last_code - self.first_code):
            msg = (
                f"day codes {int(codes.min())} to {int(codes.max())} fall outside the school"
                f" calendar {self.first_code} to {self.last_code}"
            )
            raise ValueError(msg)
        return idx


_CALENDAR: list[DenseCalendar] = []


def dense_calendar() -> DenseCalendar:
    """Return the calendar from the first term day to `CALENDAR_LAST` (built once per process)."""
    if _CALENDAR:
        return _CALENDAR[0]
    table = terms()
    first_code = code_of(day=table["first_day"].min())
    last_code = code_of(day=CALENDAR_LAST)
    n = last_code - first_code + 1
    off = np.zeros(n, dtype=np.float32)
    first_weekend = np.zeros(n, dtype=np.float32)
    half_term = np.zeros(n, dtype=np.float32)
    block_index = np.full(n, -1, dtype=np.int32)
    for b, block in enumerate(blocks()):
        lo, hi = code_of(day=block.first) - first_code, code_of(day=block.last) - first_code
        hi = min(hi, n - 1)
        if lo > hi:
            continue
        off[lo : hi + 1] = 1.0
        block_index[lo : hi + 1] = b
        if block.days <= HALF_TERM_MAX_DAYS:
            half_term[lo : hi + 1] = 1.0
        for d in block.first_weekend:
            if code_of(day=d) - first_code <= n - 1:
                first_weekend[code_of(day=d) - first_code] = 1.0
    bank = holidays.country_holidays("GB", subdiv="ENG", years=range(2021, 2028))
    bank_flag = np.zeros(n, dtype=np.float32)
    weekend = np.zeros(n, dtype=np.float32)
    for i in range(n):
        d = date_of(code=first_code + i)
        bank_flag[i] = 1.0 if d in bank else 0.0
        weekend[i] = 1.0 if d.weekday() >= 5 else 0.0
    cal = DenseCalendar(
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


def school_columns(*, codes: np.ndarray, shift_days: int = 0) -> np.ndarray:
    """Return S1 to S3 (rows, 3) for day codes; `shift_days` reads the real calendar that many days earlier.

    Args:
        codes: Local day codes of the rows.
        shift_days: 0 for the real calendar, `PLACEBO_SHIFT_DAYS` for the placebo.

    Returns:
        A float32 array of the three school columns.
    """
    cal = dense_calendar()
    idx = cal.index(codes=np.asarray(codes) - shift_days)
    return np.stack([cal.school_off[idx], cal.first_weekend[idx], cal.half_term[idx]], axis=1)


def day_flags(*, codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the bank-holiday and weekend flags of day codes (float32), from the shifted local date."""
    cal = dense_calendar()
    idx = cal.index(codes=codes)
    return cal.bank_holiday[idx], cal.weekend[idx]


# ----------------------------------------------------------------------------- coast closeness


def coast_distances_km() -> np.ndarray:
    """Return each Primary's distance to the coast in km, in Primary-index order (the diagnostic's)."""
    import e00397_prep as prep  # ty: ignore[unresolved-import]
    import e00735_diagnostic as diag  # ty: ignore[unresolved-import]

    table = diag.coast_distances(table=diag.primaries())
    sites = prep.primaries().select("primary", "site", "time_series_id").sort("primary")
    joined = sites.join(
        table.select("time_series_id", "coast_km"), on="time_series_id", how="left", validate="1:1"
    )
    assert joined["coast_km"].null_count() == 0
    return joined["coast_km"].to_numpy().astype(np.float64)


def closeness(*, km: np.ndarray) -> np.ndarray:
    """Return `c = exp(-d / 5 km)` for distances in km."""
    return np.exp(-np.asarray(km, dtype=np.float64) / COAST_SCALE_KM)


def derange(*, c: np.ndarray, seed: int = PERMUTATION_SEED) -> np.ndarray:
    """Return a permutation of the Primary indices that no Primary keeps its closeness under.

    The draws come from `numpy.random.default_rng(seed)`, one permutation after another, until no
    Primary receives a closeness equal to its own (two Primaries that share a value count as
    keeping it) and the correlation between permuted and true closeness lies within +-0.3.

    Args:
        c: The true closeness of each Primary.
        seed: The generator seed.

    Returns:
        An index array `perm` so that the placebo closeness of Primary i is `c[perm[i]]`.
    """
    rng = np.random.default_rng(seed)
    for _ in range(100_000):
        perm = rng.permutation(len(c))
        permuted = c[perm]
        if np.any(np.abs(permuted - c) < 1e-9):
            continue
        if abs(float(np.corrcoef(c, permuted)[0, 1])) > MAX_PERMUTED_CORRELATION:
            continue
        return perm
    msg = "no permutation met the rules in 100,000 draws"
    raise RuntimeError(msg)


def coast_arrays(*, save: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return the true and permuted closeness per Primary index, with the permutation saved once.

    The permutation lives in `permutation.json` under the experiment's data folder. When the file
    exists, the recomputed permutation must equal it.
    """
    c = closeness(km=coast_distances_km())
    perm = derange(c=c)
    path = ROOT / "permutation.json"
    if path.exists():
        stored = json.loads(path.read_text())
        assert stored["perm"] == [int(v) for v in perm], "the saved permutation differs"
    elif save:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {"seed": PERMUTATION_SEED, "perm": [int(v) for v in perm], "c": c.tolist()},
                indent=1,
            )
        )
        tmp.rename(path)
    return c, c[perm]


# ----------------------------------------------------------------------------- weather columns


@dataclass
class AnomalyFit:
    """The calendar-month mean of the two B1 columns per Primary, and the scale of their anomalies."""

    mean: np.ndarray
    """(Primaries, 12 months, 2): the mean of the standardised `t_max_day` and `sw_mean_24h`."""
    count: np.ndarray
    """(Primaries, 12 months): the training rows in each cell."""
    sd: np.ndarray
    """(Primaries, 2): the standard deviation of the anomalies over a Primary's training rows."""


def fit_anomalies(*, primary: np.ndarray, month: np.ndarray, b1: np.ndarray) -> AnomalyFit:
    """Fit the monthly means and the anomaly scale on training rows.

    Args:
        primary: Primary index of each training row.
        month: Local calendar month (1 to 12) of each row's valid date.
        b1: (rows, 2) standardised `t_max_day` and `sw_mean_24h`.

    Returns:
        The fit.
    """
    mean = np.zeros((N_PRIMARIES, 12, 2))
    count = np.zeros((N_PRIMARIES, 12), dtype=np.int64)
    cell = primary.astype(np.int64) * 12 + (month.astype(np.int64) - 1)
    count_flat = np.bincount(cell, minlength=N_PRIMARIES * 12)
    for j in range(2):
        sums = np.bincount(cell, weights=b1[:, j].astype(np.float64), minlength=N_PRIMARIES * 12)
        mean[:, :, j] = (sums / np.maximum(count_flat, 1)).reshape(N_PRIMARIES, 12)
    count[:] = count_flat.reshape(N_PRIMARIES, 12)
    anomaly = b1.astype(np.float64) - mean[primary, month - 1]
    sd = np.ones((N_PRIMARIES, 2))
    for p in range(N_PRIMARIES):
        sel = primary == p
        if sel.sum() > 1:
            sd[p] = np.maximum(anomaly[sel].std(axis=0), 1e-6)
    return AnomalyFit(mean=mean, count=count, sd=sd)


def apply_anomalies(
    *, fit: AnomalyFit, primary: np.ndarray, month: np.ndarray, b1: np.ndarray
) -> np.ndarray:
    """Return W1 and W2's weather factors (rows, 2): the anomaly over its scale, clipped to +-3.

    A Primary-month cell with fewer than `MIN_ROWS_PER_CELL` training rows gives 0.
    """
    cell_mean = fit.mean[primary, month - 1]
    z = (b1.astype(np.float64) - cell_mean) / fit.sd[primary]
    z = np.clip(z, -ANOMALY_CLIP, ANOMALY_CLIP)
    z[fit.count[primary, month - 1] < MIN_ROWS_PER_CELL] = 0.0
    return z.astype(np.float32)


@dataclass
class DailyMax:
    """Daily maximum 2 m temperature per Primary from the expected-weather table, by local day."""

    first_code: int
    value: np.ndarray
    """(Primaries, days) in degrees Celsius, NaN where the table has no half-hour of the day."""
    run_us: np.ndarray
    """(Primaries, days) the latest weather-run origin among the day's half-hours, 0 if none."""


def daily_max(
    *, temperature: np.ndarray, run_origin_us: np.ndarray, grid_us: np.ndarray
) -> DailyMax:
    """Reduce the expected-weather temperature to a daily maximum per Primary and local day.

    Args:
        temperature: (Primaries, grid) expected 2 m temperature, NaN where the table has no row.
        run_origin_us: (Primaries, grid) the origin of the run each half-hour was read from.
        grid_us: The stamps of the grid in microseconds.

    Returns:
        The daily maxima and, per day, the latest run origin used.
    """
    codes = local_dates(stamp_us=grid_us).code
    starts = np.flatnonzero(np.r_[True, np.diff(codes) != 0])
    first_code = int(codes[0])
    n_days = int(codes[-1]) - first_code + 1
    day_of = codes[starts] - first_code
    value = np.full((temperature.shape[0], n_days), np.nan, dtype=np.float32)
    run = np.zeros((temperature.shape[0], n_days), dtype=np.int64)
    for p in range(temperature.shape[0]):
        t = np.asarray(temperature[p], dtype=np.float32)
        top = np.fmax.reduceat(np.where(np.isfinite(t), t, -np.inf), starts)
        top = np.where(np.isfinite(top), top, np.nan)
        value[p, day_of] = top
        run[p, day_of] = np.maximum.reduceat(np.asarray(run_origin_us[p], dtype=np.int64), starts)
    return DailyMax(first_code=first_code, value=value, run_us=run)


def last_complete_day(*, origin_us: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the day code of the last local day that ended at least 24 hours before each origin.

    A local day's last stamp is the next 00:00, so the day is complete at the cut `origin - 24 h`
    only when its last stamp is at or before the cut. The cut lies inside a day unless it is
    exactly 00:00 Europe/London.

    Args:
        origin_us: Forecast origins in microseconds since the epoch.

    Returns:
        The last complete day's code per row, and the cut `origin - 24 h` in microseconds.
    """
    cut = np.asarray(origin_us, dtype=np.int64) - 24 * HOUR_US
    code = local_dates(stamp_us=cut).code
    at_midnight = is_local_midnight(stamp_us=cut)
    return np.where(at_midnight, code, code - 1).astype(np.int64), cut


def week_baseline(*, daily: DailyMax, primary: np.ndarray, origin_us: np.ndarray) -> np.ndarray:
    """Return the mean daily maximum over the 7 complete local days ending 24 h before the origin.

    NaN where fewer than `W3_MIN_DAYS` of the 7 days have a value.
    """
    last, _ = last_complete_day(origin_us=origin_us)
    end = last - daily.first_code
    out = np.full(len(primary), np.nan)
    finite = np.isfinite(daily.value)
    csum = np.concatenate(
        [
            np.zeros((daily.value.shape[0], 1)),
            np.cumsum(np.where(finite, daily.value, 0.0), axis=1),
        ],
        axis=1,
    )
    ccnt = np.concatenate(
        [np.zeros((daily.value.shape[0], 1)), np.cumsum(finite.astype(np.float64), axis=1)], axis=1
    )
    lo = np.clip(end - W3_DAYS + 1, 0, daily.value.shape[1])
    hi = np.clip(end + 1, 0, daily.value.shape[1])
    n = ccnt[primary, hi] - ccnt[primary, lo]
    s = csum[primary, hi] - csum[primary, lo]
    good = n >= W3_MIN_DAYS
    out[good] = s[good] / n[good]
    return out


def week_latest_run_us(
    *, daily: DailyMax, primary: np.ndarray, origin_us: np.ndarray
) -> np.ndarray:
    """Return the latest weather-run origin among the days that `week_baseline` reads, per row."""
    last, _ = last_complete_day(origin_us=origin_us)
    end = last - daily.first_code
    out = np.zeros(len(primary), dtype=np.int64)
    for k in range(W3_DAYS):
        idx = np.clip(end - k, 0, daily.run_us.shape[1] - 1)
        out = np.maximum(out, daily.run_us[primary, idx])
    return out


def first_warm_day(*, tmax_celsius: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Return W3's weather factor: (valid day's maximum - the week's mean maximum) / 5 K, clipped to +-3."""
    z = np.clip((tmax_celsius - baseline) / W3_SCALE_K, -ANOMALY_CLIP, ANOMALY_CLIP)
    return np.where(np.isfinite(z), z, 0.0).astype(np.float32)


# ----------------------------------------------------------------------------- the block


@dataclass
class Factors:
    """Everything `build_block` reads for a set of rows, computed once per month and source."""

    codes: np.ndarray
    primary: np.ndarray
    weather: np.ndarray | None
    """(rows, 3): the weather factors of W1 (month, warm), W2 (month, sunny), and W3, or None."""


def build_block(
    *, spec: ArmSpec, factors: Factors, c_true: np.ndarray, c_permuted: np.ndarray
) -> np.ndarray:
    """Return the arm's appended columns (rows, spec.width) as float32.

    Args:
        spec: The arm.
        factors: The rows' day codes, Primaries, and weather factors.
        c_true: Closeness per Primary index.
        c_permuted: The placebo closeness per Primary index.

    Returns:
        The columns in the order of `spec.names`. An arm with no columns gets an empty block.
    """
    n = len(factors.codes)
    parts: list[np.ndarray] = []
    shift = {"none": None, "real": 0, "shifted": PLACEBO_SHIFT_DAYS}[spec.school]
    school = school_columns(codes=factors.codes, shift_days=shift) if shift is not None else None
    if school is not None:
        parts.append(school)
    c = (c_permuted if spec.permuted else c_true)[factors.primary].astype(np.float32)
    if spec.coast_columns or spec.weather_columns:
        assert school is not None
        bank, weekend = day_flags(codes=factors.codes)
    if spec.coast_columns:
        assert school is not None
        parts.append(
            np.stack([c, c * school[:, 0], c * school[:, 1], c * bank, c * weekend], axis=1).astype(
                np.float32
            )
        )
    if spec.weather_columns:
        assert school is not None and factors.weather is not None
        w = factors.weather
        cs = c * school[:, 0]
        parts.append(
            np.stack([cs * w[:, 0], cs * w[:, 1], cs * w[:, 2], c * w[:, 0]], axis=1).astype(
                np.float32
            )
        )
    block = np.concatenate(parts, axis=1) if parts else np.zeros((n, 0), dtype=np.float32)
    assert block.shape == (n, spec.width), (block.shape, spec.width)
    return block


# ----------------------------------------------------------------------------- the row builder


class RowBuilder:
    """Build an arm's appended columns for a month's training rows and its test rows.

    Training rows fit the weather anomalies (the calendar-month mean of the standardised B1
    columns at each Primary, over the rows that are not held out); test rows reuse that fit, so
    nothing from a test row enters the scale of its own column. Arms without weather columns skip
    the weather work.
    """

    def __init__(self, *, spec: ArmSpec, stats_file, expected_dir: Path) -> None:  # noqa: ANN001
        """Set the arm, a function that returns a month's B1 statistics file, and the table folder."""
        self.spec = spec
        self.stats_file = stats_file
        self.expected_dir = expected_dir
        self.c_true, self.c_permuted = coast_arrays()
        self.fits: dict[str, AnomalyFit] = {}
        self._daily: DailyMax | None = None
        self.last_report: dict[str, float] = {}

    def daily(self) -> DailyMax:
        """Return the expected-weather table's daily maximum temperature (built once per process)."""
        if self._daily is None:
            import e00422_prep as prep422  # ty: ignore[unresolved-import]
            import e00560_data as d560  # ty: ignore[unresolved-import]

            cols, origin = d560.raw_table(source="ft", cache=self.expected_dir)
            grid_us = prep422.GRID_START_US + np.arange(cols.shape[1], dtype=np.int64) * (
                prep422.STEP_US
            )
            self._daily = daily_max(
                temperature=np.asarray(cols[:, :, EXPECTED_TEMPERATURE_COLUMN]),
                run_origin_us=np.asarray(origin),
                grid_us=grid_us,
            )
        return self._daily

    def _b1(self, *, month: str, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the standardised (t_max_day, sw_mean_24h) columns and the raw t_max_day in Celsius."""
        stats = np.load(self.stats_file(month=month))
        two = np.stack([x[:, B1_TMAX_COLUMN], x[:, B1_SW_COLUMN]], axis=1).astype(np.float32)
        raw_tmax = (
            x[:, B1_TMAX_COLUMN].astype(np.float64) * stats["std"][B1_TMAX_POSITION]
            + (stats["mean"][B1_TMAX_POSITION])
        )
        return two, raw_tmax

    def _factors(
        self, *, month: str, primary: np.ndarray, valid_us: np.ndarray, origin_us: np.ndarray,
        x: np.ndarray,
    ) -> Factors:  # fmt: skip
        """Return the day codes and, for weather arms, the three weather factors of the rows."""
        local = local_dates(stamp_us=valid_us)
        if not self.spec.weather_columns:
            return Factors(codes=local.code, primary=primary, weather=None)
        two, raw_tmax = self._b1(month=month, x=x)
        w12 = apply_anomalies(fit=self.fits[month], primary=primary, month=local.month, b1=two)
        baseline = week_baseline(daily=self.daily(), primary=primary, origin_us=origin_us)
        w3 = first_warm_day(tmax_celsius=raw_tmax, baseline=baseline)
        self.last_report = {
            "w3_rows_without_baseline": float(np.isnan(baseline).mean()),
            "w12_rows_in_thin_cell": float(
                (self.fits[month].count[primary, local.month - 1] < MIN_ROWS_PER_CELL).mean()
            ),
        }
        return Factors(
            codes=local.code, primary=primary, weather=np.concatenate([w12, w3[:, None]], axis=1)
        )

    def month_block(self, *, month: str, mr) -> np.ndarray:  # noqa: ANN001
        """Return the appended columns of every row of a `MonthRows`, fitting the month's anomalies first."""
        if self.spec.weather_columns:
            base_rows = np.flatnonzero(~mr.is_val[: mr.n_base])
            local = local_dates(stamp_us=mr.valid_us[base_rows])
            two = np.stack([mr.x[base_rows, B1_TMAX_COLUMN], mr.x[base_rows, B1_SW_COLUMN]], axis=1)
            self.fits[month] = fit_anomalies(
                primary=mr.primary[base_rows], month=local.month, b1=two
            )
        factors = self._factors(
            month=month, primary=mr.primary, valid_us=mr.valid_us, origin_us=mr.origin_us, x=mr.x
        )
        return build_block(
            spec=self.spec, factors=factors, c_true=self.c_true, c_permuted=self.c_permuted
        )

    def test_block(self, *, month: str, test) -> np.ndarray:  # noqa: ANN001
        """Return the appended columns of a month's test rows, on the month's fitted anomalies."""
        factors = self._factors(
            month=month, primary=test.primary, valid_us=test.valid, origin_us=test.origin, x=test.x
        )
        return build_block(
            spec=self.spec, factors=factors, c_true=self.c_true, c_permuted=self.c_permuted
        )
