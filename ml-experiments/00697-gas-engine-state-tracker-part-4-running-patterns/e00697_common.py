"""Paths, constants, local-time helpers, week-block intervals, and spells for experiment 697.

D8 is a substation and may be named. The engine record is called "Gas A" everywhere that is
printed or drawn, and it appears only as aggregates over the whole record, never with a date.

**Time conventions.** A power stamp marks the end of its half-hour, as the data-processing page
states, and a price stamp marks the start. Every stamp that comes from a power series (the panel's
grid index, the tracker's grid index, Gas A's `time`) is moved to the start of its half-hour with
`START_SHIFT` before any join on a price, any gas-day join, and any conversion to local time.
Weather columns of the panel are carried by grid index and never re-joined by time.
"""

import pickle
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Final, Literal
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
from e00694_common import (  # ty: ignore[unresolved-import]
    GRID_START,
    Q_FLOOR,
    causal_spread,
    inject_folded,
    load_panel,
    load_record,
    panel_power,
    quantise_spread,
)
from e00694_trackers import grid_track  # ty: ignore[unresolved-import]
from e00695_common import grid_index  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 697
SLUG: Final[str] = "gas-engine-state-tracker-part-4-running-patterns"
FOLDER_NAME: Final[str] = f"{EXPERIMENT_ID:05d}-{SLUG}"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / FOLDER_NAME
"""The experiment's private data folder: frames, control runs, statistics, private figures."""
DOCS_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / FOLDER_NAME
"""Where the published figures and the pages are written."""
PRINTED_DIR: Final[Path] = Path(__file__).resolve().parent / "printed"
"""Printed reports, committed with the scripts. They hold no Gas A date and no Gas A series."""
FRAMES_DIR: Final[Path] = DATA_DIR / "frames"
CONTROLS_DIR: Final[Path] = DATA_DIR / "controls"
PRIVATE_FIGURES_DIR: Final[Path] = DATA_DIR / "private_figures"
STATS_PATH: Final[Path] = DATA_DIR / "stats.json"

E694_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00694-gas-engine-state-tracker-part-1"
E695_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00695-gas-engine-state-tracker-part-2-cleaned-target-and-climatology-add-back"
)
D8_STATES_PATH: Final[Path] = E694_DIR / "d8.pkl"
"""Experiment 694's M2 labels of D8, three eras, on the panel grid."""
FALSE_ALARM_PATH: Final[Path] = E694_DIR / "fa.pkl"
"""Experiment 694's stored false-alarm pass: 44 segments, no engine injected."""
PROBABILITY_PATH: Final[Path] = E695_DIR / "states" / "AH" / "all.npz"
"""Experiment 695's hindsight state probabilities `pk` on the pipeline grid."""

PIPELINE_OFFSET: Final[int] = grid_index(when=GRID_START)
"""Half-hours from the pipeline grid's start to the panel grid's start (40,944)."""
START_SHIFT: Final[timedelta] = timedelta(minutes=30)
"""Subtracted from a power end stamp to give the start of its half-hour."""
LOCAL_TZ: Final[str] = "Europe/London"
DAY: Final[int] = 48
UNIT_MW: Final[float] = 1.97
"""One engine unit, from the record's output levels (experiment 196)."""
ERA_NAMES: Final[tuple[str, ...]] = ("net", "solar", "fade")
VERDICT_ERAS: Final[tuple[str, ...]] = ("net", "solar")
"""The eras that carry the verdict. The fade era is shown and never supports a pattern."""
EXPECTED_ERA_LENGTHS: Final[dict[str, int]] = {"net": 26256, "solar": 16416, "fade": 5856}

EVENING_FIRST: Final[int] = 32
EVENING_LAST: Final[int] = 37
"""Local half-hours 32 to 37 start at 16:00 to 18:30, so they cover 16:00 to 19:00 local time."""
BLOCK_EDGES: Final[tuple[int, ...]] = (0, 10, 18, 32, 38, 48)
BLOCK_NAMES: Final[tuple[str, ...]] = ("night", "morning", "day", "evening", "late")
"""Blocks of the local day: night 00:00 to 05:00, morning to 09:00, day to 16:00, evening to 19:00,
late to 24:00."""
NIGHT_LAST: Final[int] = 9
"""The last local half-hour of the night block (starts 04:30)."""

N_RESAMPLES: Final[int] = 2000
CONFIDENCE: Final[float] = 0.95
BOOTSTRAP_SEED: Final[int] = 697
MIN_DAYS_PER_BIN: Final[int] = 20
"""A bin that holds fewer distinct local days is left out of every test and every plot."""
RUNNING_DAY_MIN_HALF_HOURS: Final[int] = 4
"""A local day counts as a running day when it has this many running half-hours or more."""
MAX_SPELL_LENGTH: Final[int] = 48
Q1_MIN_DIFFERENCE: Final[float] = 0.10

WeekdayClassType = Literal["weekday", "weekend"]
TruthKindType = Literal["null", "weekday"]
"""How the injected days are drawn: ignoring the weekday, or within the weekday class."""


def _clock_change_dates(*, first_year: int, last_year: int) -> tuple[date, ...]:
    """Return the local dates whose UTC offset at midnight differs from the next day's."""
    zone = ZoneInfo(LOCAL_TZ)
    out: list[date] = []
    day = date(first_year, 1, 1)
    while day.year <= last_year:
        nxt = day + timedelta(days=1)
        before = datetime(day.year, day.month, day.day, tzinfo=zone).utcoffset()
        after = datetime(nxt.year, nxt.month, nxt.day, tzinfo=zone).utcoffset()
        if before != after:
            out.append(day)
        day = nxt
    return tuple(out)


CLOCK_CHANGE_DATES: Final[tuple[date, ...]] = _clock_change_dates(first_year=2021, last_year=2027)
"""The local dates with 46 or 50 half-hours."""
WEEK_ORIGIN: Final[date] = date(2000, 1, 3)
"""A Monday, so that `(local_date - WEEK_ORIGIN) // 7` numbers the ISO weeks."""


def half_hour_starts(*, index: np.ndarray) -> pl.Series:
    """Return the UTC start of the half-hour whose end stamp is panel-grid index `index`.

    Args:
        index: Panel-grid indices. Index `i` has the end stamp `GRID_START + i * 30 min`.

    Returns:
        A Series named `start` of UTC datetimes, `START_SHIFT` before each end stamp.
    """
    base = np.datetime64(GRID_START.replace(tzinfo=None), "us")
    steps = (np.asarray(index, dtype=np.int64) - 1) * np.timedelta64(1800, "s")
    return pl.Series("start", (base + steps).astype("datetime64[us]")).dt.replace_time_zone("UTC")


def add_local_fields(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add the local-time columns to a frame that has a UTC half-hour `start`.

    Args:
        frame: A frame with a `start` column (UTC, the start of each half-hour).

    Returns:
        The frame with `local_date`, `local_half_hour` (the clock reading, 0 to 47), `weekday` (1 is
        Monday), `weekend`, `month`, `summer_half` (British Summer Time), `clock_change_day`, and
        `week` (the ISO week number counted from `WEEK_ORIGIN`).
    """
    local = pl.col("start").dt.convert_time_zone(LOCAL_TZ)
    return frame.with_columns(
        local_date=local.dt.date(),
        local_half_hour=(local.dt.hour() * 2 + local.dt.minute() // 30).cast(pl.Int16),
        summer_half=local.dt.dst_offset().dt.total_minutes() > 0,
    ).with_columns(
        weekday=pl.col("local_date").dt.weekday().cast(pl.Int8),
        month=pl.col("local_date").dt.month().cast(pl.Int8),
        clock_change_day=pl.col("local_date").is_in(pl.Series(list(CLOCK_CHANGE_DATES))),
        week=((pl.col("local_date") - pl.lit(WEEK_ORIGIN)).dt.total_days() // 7).cast(pl.Int32),
    ).with_columns(weekend=pl.col("weekday") >= 6)


def add_segments(*, frame: pl.DataFrame, group_columns: tuple[str, ...] = ()) -> pl.DataFrame:
    """Number the runs of consecutive half-hours, restarting at any gap or change of a group column.

    The frame must be sorted by `group_columns` and then by `start`.
    """
    new_run = (pl.col("start").diff() != pl.duration(minutes=30)).fill_null(value=True)
    for column in group_columns:
        new_run = new_run | (pl.col(column) != pl.col(column).shift(1)).fill_null(value=True)
    return frame.with_columns(segment=new_run.cum_sum().cast(pl.Int32))


def group_within(
    *, value: str, n_groups: int, within: tuple[str, ...] = ()
) -> pl.Expr:
    """Return an expression for the group (0 to `n_groups - 1`) of `value` among its peers.

    Peers are the rows that share the `within` columns, and a group is a rank interval of equal
    size, so 3 groups are thirds and 5 are fifths. Null values stay null. Ties are broken by row
    order, which matters little for a continuous driver.
    """
    rank = pl.col(value).rank(method="ordinal")
    count = pl.col(value).count()
    if within:
        rank, count = rank.over(*within), count.over(*within)
    return ((rank - 1) * n_groups // count).cast(pl.Int8)


def quantile_bins(*, values: np.ndarray, n_bins: int) -> np.ndarray:
    """Return the equal-count bin (0 to `n_bins - 1`) of each value, NaN-safe, from value quantiles.

    Edges are the `1/n_bins, 2/n_bins, ...` quantiles of the finite values, and a value equal to an
    edge goes to the upper bin. A NaN gets -1. Experiment 194's price fifths are these bins.
    """
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    edges = np.quantile(values[finite], np.arange(1, n_bins) / n_bins)
    bins = np.searchsorted(edges, values, side="right")
    return np.where(finite, bins, -1)


def day_block(*, half_hour: np.ndarray) -> np.ndarray:
    """Return the index into `BLOCK_NAMES` of each local half-hour (0 to 47)."""
    return np.digitize(np.asarray(half_hour), BLOCK_EDGES[1:-1])


GAS_DAY_START_HOUR: Final[int] = 5
"""The gas day for date D runs from 05:00 local on D to 05:00 local on D + 1."""


def gas_day_expression() -> pl.Expr:
    """Return the gas day (a date) of each half-hour, from its UTC `start`.

    The gas price for gas day D covers 05:00 on D to 05:00 on D + 1, so a half-hour belongs to the
    gas day of its local start time minus 5 hours. Joining on the local date instead would give 10
    of the 48 half-hours of each day the wrong day's gas price.
    """
    local = pl.col("start").dt.convert_time_zone(LOCAL_TZ)
    return (local - pl.duration(hours=GAS_DAY_START_HOUR)).dt.date()


# --------------------------------------------------------------------------- the panel and the eras


def load_d8_eras() -> list[dict]:
    """Return experiment 694's three era records for D8, checked against the stored lengths."""
    eras = pickle.loads(D8_STATES_PATH.read_bytes())
    lengths = {era["label"]: len(era["y"]) for era in eras}
    if lengths != EXPECTED_ERA_LENGTHS:
        msg = f"era lengths {lengths} differ from {EXPECTED_ERA_LENGTHS}"
        raise ValueError(msg)
    return eras


def era_bounds() -> dict[str, tuple[int, int]]:
    """Return each era's (first panel index, one past the last) as stored by experiment 694."""
    return {era["label"]: (int(era["start"]), int(era["start"]) + len(era["y"])) for era in load_d8_eras()}


def panel_array(*, panel: dict[int, dict], label: str, key: str) -> np.ndarray:
    """Return one array of a site in experiment 196's panel (`power`, `temp`, `wind100`, `ghi`)."""
    entry = next(v for v in panel.values() if v["site"] == label)
    return np.asarray(entry[key], dtype=float)


# --------------------------------------------------------------------------- tracking a span


@dataclass(frozen=True)
class SpanTrack:
    """The M2 labels of one tracked span."""

    flag: np.ndarray
    filtered: np.ndarray
    k_path: np.ndarray
    p_any: np.ndarray
    k_max: int
    unit: float


def track_span(*, y_all: np.ndarray, first: int, last: int) -> SpanTrack:
    """Run experiment 694's M2 grid filter on one span of a series.

    This copies the recipe of `d8_era` in `e00694_run.py`, which is never imported: the spread level
    comes from the whole series, the span is sliced, and `grid_track` fits K and the unit size
    without truth and returns the smoothed and filtered flags.

    Args:
        y_all: The whole reading (MVA, NaN where dropped), on the panel grid.
        first: The first panel index of the span.
        last: One past the last panel index of the span.

    Returns:
        The smoothed flag (the 0.5 threshold on the smoothed probability), the filtered flag, the
        smoothed most-probable unit count, the smoothed probability of any unit, K, and the unit.
    """
    lvl_all = quantise_spread(spread=causal_spread(y=y_all))
    result = grid_track(y=y_all[first:last], lvl=lvl_all[first:last])
    return SpanTrack(
        flag=result.p_any_smooth > 0.5,
        filtered=result.p_any_filt > 0.5,
        k_path=np.asarray(result.k_smooth, dtype=np.int64),
        p_any=np.asarray(result.p_any_smooth, dtype=np.float32),
        k_max=int(result.k_max),
        unit=float(result.unit),
    )


def inject_span(*, y_all: np.ndarray, g: np.ndarray, first: int, last: int) -> np.ndarray:
    """Subtract the output `g` from the real power behind the reading on `[first, last)`.

    Experiment 694's folded rule is applied only on the injected span, so the reading elsewhere is
    untouched and the trailing 28-day spread of the injected span's first day reads injected days.
    """
    out = y_all.copy()
    out[first:last] = inject_folded(y=y_all[first:last], g=g[first:last], q=Q_FLOOR)
    return out


# --------------------------------------------------------------------------- week-block intervals


@dataclass(frozen=True)
class Estimate:
    """A statistic with its percentile interval from resampling whole ISO weeks."""

    point: float
    lo: float
    hi: float
    n_weeks: int
    draws: np.ndarray = field(repr=False)

    @property
    def excludes_zero(self) -> bool:
        """Whether the interval has one sign."""
        return bool(np.isfinite(self.lo) and (self.lo > 0 or self.hi < 0))

    def to_dict(self) -> dict[str, float | int]:
        """Return the point, the interval, and the number of weeks, without the draws."""
        return {"point": self.point, "lo": self.lo, "hi": self.hi, "n_weeks": self.n_weeks}


def ratio(*, numerator: np.ndarray | float, denominator: np.ndarray | float) -> np.ndarray:
    """Return `numerator / denominator`, NaN where the denominator is not positive."""
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    out = np.full(np.broadcast(num, den).shape, np.nan)
    np.divide(num, den, out=out, where=den > 0)
    return out


def resample_week_sums(
    *, week: np.ndarray, columns: np.ndarray, seed: int, n_resamples: int = N_RESAMPLES
) -> tuple[np.ndarray, np.ndarray]:
    """Return the column sums and their sums over `n_resamples` resamples of whole weeks.

    Args:
        week: The week number of each row.
        columns: Row-level quantities, shape (rows, columns). Rows of one week move together.
        seed: The seed of the resampling.
        n_resamples: The number of resamples.

    Returns:
        The point sums, shape (columns,), and the resampled sums, shape (n_resamples, columns).
    """
    weeks, inverse = np.unique(np.asarray(week), return_inverse=True)
    per_week = np.stack(
        [
            np.bincount(inverse, weights=columns[:, j], minlength=len(weeks))
            for j in range(columns.shape[1])
        ],
        axis=1,
    )
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(weeks), np.full(len(weeks), 1.0 / len(weeks)), size=n_resamples)
    return per_week.sum(axis=0), counts @ per_week


def percentile_interval(*, draws: np.ndarray, level: float = CONFIDENCE) -> tuple[float, float]:
    """Return the percentile interval of the finite draws, or NaN if there are none."""
    finite = draws[np.isfinite(draws)]
    if len(finite) == 0:
        return float("nan"), float("nan")
    tail = 100.0 * (1.0 - level) / 2.0
    lo, hi = np.percentile(finite, [tail, 100.0 - tail])
    return float(lo), float(hi)


def week_estimate(
    *,
    week: np.ndarray,
    columns: np.ndarray,
    statistic: Callable[[np.ndarray], np.ndarray],
    seed: int,
    n_resamples: int = N_RESAMPLES,
) -> Estimate | None:
    """Return a statistic of column sums, with an interval from resampling whole weeks.

    Args:
        week: The week number of each row.
        columns: Row-level quantities, shape (rows, columns).
        statistic: A function of sums with the columns on the last axis, such as
            `lambda s: ratio(numerator=s[..., 0], denominator=s[..., 1])`.
        seed: The seed of the resampling.
        n_resamples: The number of resamples.

    Returns:
        The estimate, or None when there are no rows.
    """
    if len(week) == 0:
        return None
    point, draws = resample_week_sums(
        week=week, columns=columns, seed=seed, n_resamples=n_resamples
    )
    stat_draws = np.asarray(statistic(draws), dtype=float)
    lo, hi = percentile_interval(draws=stat_draws)
    return Estimate(
        point=float(statistic(point)),
        lo=lo,
        hi=hi,
        n_weeks=int(len(np.unique(week))),
        draws=stat_draws,
    )


def difference_estimate(*, first: Estimate, second: Estimate) -> Estimate:
    """Return `first - second` for two estimates resampled independently, with its interval."""
    draws = first.draws - second.draws
    lo, hi = percentile_interval(draws=draws)
    return Estimate(
        point=first.point - second.point,
        lo=lo,
        hi=hi,
        n_weeks=min(first.n_weeks, second.n_weeks),
        draws=draws,
    )


def share_columns(*, state: np.ndarray, mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    """Return the four columns of a difference of two shares: (state in a, a, state in b, b)."""
    s = np.asarray(state, dtype=float)
    a = np.asarray(mask_a, dtype=float)
    b = np.asarray(mask_b, dtype=float)
    return np.stack([s * a, a, s * b, b], axis=1)


def share_difference_statistic(sums: np.ndarray) -> np.ndarray:
    """Return share(a) - share(b) from the sums of `share_columns`."""
    return ratio(numerator=sums[..., 0], denominator=sums[..., 1]) - ratio(
        numerator=sums[..., 2], denominator=sums[..., 3]
    )


# --------------------------------------------------------------------------- spells and hazards


@dataclass(frozen=True)
class Spells:
    """The complete runs of a state: its value, its length in half-hours, and its first row."""

    value: np.ndarray
    length: np.ndarray
    first_row: np.ndarray


def extract_spells(*, state: np.ndarray, valid: np.ndarray, segment: np.ndarray) -> Spells:
    """Return the complete spells of a boolean state.

    A spell is a maximal run of one value inside one segment. A run that touches the first or the
    last half-hour of its segment is cut off and is dropped, and so is a run that covers a
    half-hour that is not `valid` (a dropped reading, where the label is only interpolated).
    """
    state = np.asarray(state, dtype=bool)
    n = len(state)
    if n == 0:
        empty = np.zeros(0, dtype=int)
        return Spells(value=np.zeros(0, dtype=bool), length=empty, first_row=empty)
    change = np.ones(n, dtype=bool)
    change[1:] = (state[1:] != state[:-1]) | (segment[1:] != segment[:-1])
    starts = np.flatnonzero(change)
    stops = np.append(starts[1:], n)
    seg_start = np.ones(n, dtype=bool)
    seg_start[1:] = segment[1:] != segment[:-1]
    seg_end = np.ones(n, dtype=bool)
    seg_end[:-1] = segment[1:] != segment[:-1]
    complete = ~seg_start[starts] & ~seg_end[stops - 1]
    complete &= np.logical_and.reduceat(np.asarray(valid, dtype=bool), starts)
    return Spells(
        value=state[starts][complete],
        length=(stops - starts)[complete],
        first_row=starts[complete],
    )


def hazard_columns(*, lengths: np.ndarray, max_length: int = MAX_SPELL_LENGTH) -> np.ndarray:
    """Return per-spell columns for hazards: `length == n` then `length >= n`, for n = 1 to max.

    Shape (spells, 2 * max_length). The hazard at n is the sum of the first block's column n over
    the sum of the second block's column n, and the survival is that sum over the spell count.
    """
    ns = np.arange(1, max_length + 1)
    lengths = np.asarray(lengths)[:, None]
    return np.concatenate([(lengths == ns).astype(float), (lengths >= ns).astype(float)], axis=1)


def pooled_hazard(*, sums: np.ndarray, first: int, last: int, max_length: int = MAX_SPELL_LENGTH) -> np.ndarray:
    """Return the hazard over lengths `first` to `last`: spells that end there per spell-step at risk."""
    ends = sums[..., first - 1 : last].sum(axis=-1)
    risk = sums[..., max_length + first - 1 : max_length + last].sum(axis=-1)
    return ratio(numerator=ends, denominator=risk)


def spell_quantiles(*, lengths: np.ndarray) -> dict[str, float]:
    """Return the count, mean, and quantiles of spell lengths in half-hours."""
    if len(lengths) == 0:
        return {"n": 0}
    q = np.percentile(lengths, [10, 25, 50, 75, 90])
    return {
        "n": int(len(lengths)),
        "mean": float(np.mean(lengths)),
        "p10": float(q[0]),
        "p25": float(q[1]),
        "median": float(q[2]),
        "p75": float(q[3]),
        "p90": float(q[4]),
        "max": float(np.max(lengths)),
    }
