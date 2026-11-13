"""The questions, spells, and bins of experiment 697, shared by the statistics and the figures.

A **view** is a frame (D8's rows of one era, a control's rows of one span, or Gas A's record) with a
`state` column added: the label as a Float64, a 0 or 1 for a flag and a probability for `p_any_a`.
Every question takes a view and a seed and returns an `Estimate` (or None when a bin holds fewer
than `MIN_DAYS_PER_BIN` distinct local days). Intervals resample whole ISO weeks of local dates, and
a spell belongs to the week in which it starts.

Gas A's frame is passed through the same functions with `min_days=0`: its intervals are rough
and decide nothing. A bin of 12 weekend days is below the 20-day floor that the verdict's
statistical tests need.
"""

import zlib
from dataclasses import dataclass
from typing import Final

import numpy as np
import polars as pl
from e00697_common import (  # ty: ignore[unresolved-import]
    BLOCK_NAMES,
    BOOTSTRAP_SEED,
    EVENING_FIRST,
    EVENING_LAST,
    MAX_SPELL_LENGTH,
    MIN_DAYS_PER_BIN,
    NIGHT_LAST,
    RUNNING_DAY_MIN_HALF_HOURS,
    UNIT_MW,
    Estimate,
    Spells,
    day_block,
    extract_spells,
    group_within,
    hazard_columns,
    pooled_hazard,
    quantile_bins,
    ratio,
    resample_week_sums,
    share_columns,
    share_difference_statistic,
    week_estimate,
)

HEATMAP_MIN_DAYS: Final[int] = 15
"""A heat-map cell is one weekday and half-hour over half a year, so it holds fewer than 20 days
in a short half. A heat map describes and does not test, so its floor is lower."""
MIN_SPELLS: Final[int] = 20
"""Fewer complete spells than this give no estimate of a hazard."""
CELL_COLUMNS: Final[tuple[str, ...]] = ("local_half_hour", "weekend", "month")
POST_WINDOW: Final[tuple[str, str]] = ("2025-03-24", "2025-09-23")
PRE_WINDOW: Final[tuple[str, str]] = ("2024-03-24", "2024-09-23")
"""Q5 compares the 6 months after the solar start with the same calendar span a year earlier."""
PART_LOAD_MW: Final[float] = 0.15
"""A running half-hour within this many MW of a whole number of units is at a full level."""
EDGE_WINDOW: Final[int] = 3
LabelColumn = str


def stable_seed(*key: str) -> int:
    """Return a seed that depends only on the text of `key`."""
    return BOOTSTRAP_SEED + zlib.crc32("|".join(key).encode()) % 1_000_003


def with_state(*, frame: pl.DataFrame, column: LabelColumn) -> pl.DataFrame:
    """Return the frame with its label `column` as the Float64 column `state`."""
    return frame.with_columns(state=pl.col(column).cast(pl.Float64))


def usable(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Keep half-hours whose reading is present and that are not on a clock-change day."""
    return frame.filter(pl.col("observed"), ~pl.col("clock_change_day"))


def distinct_days(*, frame: pl.DataFrame, mask: np.ndarray) -> int:
    """Count the distinct local dates of the rows where `mask` holds."""
    return int(frame.filter(pl.Series(mask))["local_date"].n_unique()) if mask.any() else 0


def two_group_share(
    *,
    rows: pl.DataFrame,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    seed: int,
    min_days: int,
) -> Estimate | None:
    """Return share(a) - share(b) of `state`, or None when either group is under `min_days`."""
    if rows.height == 0 or distinct_days(frame=rows, mask=mask_a) < max(min_days, 1):
        return None
    if distinct_days(frame=rows, mask=mask_b) < max(min_days, 1):
        return None
    columns = share_columns(state=rows["state"].to_numpy(), mask_a=mask_a, mask_b=mask_b)
    return week_estimate(
        week=rows["week"].to_numpy(),
        columns=columns,
        statistic=share_difference_statistic,
        seed=seed,
    )


def q1(*, view: pl.DataFrame, seed: int, min_days: int = MIN_DAYS_PER_BIN) -> Estimate | None:
    """Q1: weekday share minus weekend share of the state, 16:00 to 19:00 local."""
    rows = usable(frame=view).filter(
        pl.col("local_half_hour").is_between(EVENING_FIRST, EVENING_LAST)
    )
    weekend = rows["weekend"].to_numpy()
    return two_group_share(rows=rows, mask_a=~weekend, mask_b=weekend, seed=seed, min_days=min_days)


def add_price_thirds(*, view: pl.DataFrame, by_quarter: bool = False) -> pl.DataFrame:
    """Add `third` (0 cheapest, 2 dearest), the system-price third within each cell.

    A cell is one local half-hour in one weekday class in one calendar month (or quarter). A cell
    with fewer than 3 priced half-hours is dropped.
    """
    cells = ("local_half_hour", "weekend", "quarter" if by_quarter else "month")
    rows = (
        usable(frame=view)
        .filter(pl.col("system_price").is_not_null())
        .with_columns(quarter=((pl.col("month") - 1) // 3).cast(pl.Int8))
        .filter(pl.len().over(*cells) >= 3)
    )
    return rows.with_columns(third=group_within(value="system_price", n_groups=3, within=cells))


def q2(
    *, view: pl.DataFrame, seed: int, min_days: int = MIN_DAYS_PER_BIN, by_quarter: bool = False
) -> Estimate | None:
    """Q2: share in the dearest price third minus the cheapest, within cell."""
    rows = add_price_thirds(view=view, by_quarter=by_quarter)
    third = rows["third"].to_numpy()
    return two_group_share(
        rows=rows, mask_a=third == 2, mask_b=third == 0, seed=seed, min_days=min_days
    )


def spells_of(*, view: pl.DataFrame) -> Spells:
    """Return the complete spells of the view's state (above 0.5), with the view's row numbers."""
    return extract_spells(
        state=view["state"].to_numpy() > 0.5,
        valid=view["observed"].to_numpy(),
        segment=view["segment"].to_numpy(),
    )


def spell_table(*, view: pl.DataFrame, kind: bool) -> pl.DataFrame:
    """Return the complete on-spells (`kind` True) or off-spells, with length and start fields."""
    spells = spells_of(view=view)
    keep = spells.value == kind
    rows = spells.first_row[keep]
    return pl.DataFrame(
        {
            "length": spells.length[keep],
            "week": view["week"].to_numpy()[rows],
            "start_half_hour": view["local_half_hour"].to_numpy()[rows],
            "summer_half": view["summer_half"].to_numpy()[rows],
            "weekend": view["weekend"].to_numpy()[rows],
            "era": view["era"].to_numpy()[rows],
        }
    )


def q3(
    *, view: pl.DataFrame, seed: int, min_spells: int = MIN_SPELLS
) -> tuple[Estimate | None, Estimate | None]:
    """Q3: the hazard at lengths 1 to 2 minus at 3 to 8, and the hazard at length 1."""
    spells = spell_table(view=view, kind=True)
    if spells.height < min_spells:
        return None, None
    columns = hazard_columns(lengths=spells["length"].to_numpy())
    week = spells["week"].to_numpy()
    contrast = week_estimate(
        week=week,
        columns=columns,
        statistic=lambda s: (
            pooled_hazard(sums=s, first=1, last=2) - pooled_hazard(sums=s, first=3, last=8)
        ),
        seed=seed,
    )
    single = week_estimate(
        week=week,
        columns=columns,
        statistic=lambda s: pooled_hazard(sums=s, first=1, last=1),
        seed=seed + 1,
    )
    return contrast, single


def q5(*, view: pl.DataFrame, seed: int, min_days: int = MIN_DAYS_PER_BIN) -> Estimate | None:
    """Q5: night share (00:00 to 05:00) in the 6 months after 2025-03-24 minus a year earlier."""
    rows = usable(frame=view).filter(pl.col("local_half_hour") <= NIGHT_LAST)
    dates = rows["local_date"].cast(pl.String)
    post = ((dates >= POST_WINDOW[0]) & (dates <= POST_WINDOW[1])).to_numpy()
    pre = ((dates >= PRE_WINDOW[0]) & (dates <= PRE_WINDOW[1])).to_numpy()
    return two_group_share(rows=rows, mask_a=post, mask_b=pre, seed=seed, min_days=min_days)


def q6(*, view: pl.DataFrame, seed: int, min_days: int = MIN_DAYS_PER_BIN) -> Estimate | None:
    """Q6: weekday evening share in the winter half (GMT) minus the summer half (BST)."""
    rows = usable(frame=view).filter(
        ~pl.col("weekend"), pl.col("local_half_hour").is_between(EVENING_FIRST, EVENING_LAST)
    )
    summer = rows["summer_half"].to_numpy()
    return two_group_share(rows=rows, mask_a=~summer, mask_b=summer, seed=seed, min_days=min_days)


def day_table(*, view: pl.DataFrame) -> pl.DataFrame:
    """Return one row per complete local day: its count of running half-hours and its weekday.

    A day is complete when it has 48 half-hours with a present reading each, on a day without a
    clock change. `running_day` is true at `RUNNING_DAY_MIN_HALF_HOURS` running half-hours or more.
    """
    keys = [c for c in ("seed", "era") if c in view.columns]
    return (
        view.group_by(*keys, "local_date", maintain_order=True)
        .agg(
            n=pl.len(),
            n_observed=pl.col("observed").sum(),
            n_on=(pl.col("state") > 0.5).sum(),
            weekday=pl.col("weekday").first(),
            week=pl.col("week").first(),
            clock_change_day=pl.col("clock_change_day").first(),
        )
        .filter(pl.col("n") == 48, pl.col("n_observed") == 48, ~pl.col("clock_change_day"))
        .with_columns(running_day=pl.col("n_on") >= RUNNING_DAY_MIN_HALF_HOURS)
    )


def q7(*, view: pl.DataFrame, seed: int, min_days: int = MIN_DAYS_PER_BIN) -> Estimate | None:
    """Q7: P(running day | previous day ran) minus P(running day | previous day idle), weekdays.

    Both days are Monday to Friday, so the current day is Tuesday to Friday, and both are complete
    days in one run of the same era (and seed).
    """
    days = day_table(view=view)
    keys = [c for c in ("seed", "era") if c in days.columns]
    pairs = days.with_columns(
        previous=pl.col("running_day").shift(1).over(*keys)
        if keys
        else pl.col("running_day").shift(1),
        gap=(pl.col("local_date") - pl.col("local_date").shift(1)).dt.total_days(),
    ).filter(
        pl.col("gap") == 1, pl.col("weekday").is_between(2, 5), pl.col("previous").is_not_null()
    )
    if pairs.height == 0:
        return None
    previous = pairs["previous"].to_numpy().astype(bool)
    rows = pairs.with_columns(state=pl.col("running_day").cast(pl.Float64))
    return two_group_share(
        rows=rows, mask_a=previous, mask_b=~previous, seed=seed, min_days=min_days
    )


def q8(*, view: pl.DataFrame, seed: int) -> dict[str, float | int | bool] | None:
    """Q8: the most common local half-hour of on-spell starts in each half, and whether they agree.

    Returns the two modes (local half-hours 0 to 47), whether they are equal, and the share of
    week resamples in which they are equal.
    """
    spells = spell_table(view=view, kind=True)
    if spells.height < MIN_SPELLS:
        return None
    half = spells["summer_half"].to_numpy().astype(int)
    columns = np.zeros((spells.height, 96))
    columns[np.arange(spells.height), spells["start_half_hour"].to_numpy() + 48 * half] = 1.0
    point, draws = resample_week_sums(week=spells["week"].to_numpy(), columns=columns, seed=seed)
    if point[:48].sum() == 0 or point[48:].sum() == 0:
        return None
    same = np.argmax(draws[:, :48], axis=1) == np.argmax(draws[:, 48:], axis=1)
    return {
        "winter_mode": int(np.argmax(point[:48])),
        "summer_mode": int(np.argmax(point[48:])),
        "same": bool(np.argmax(point[:48]) == np.argmax(point[48:])),
        "share_of_resamples_same": float(same.mean()),
    }


# --------------------------------------------------------------------------- bins and shares


@dataclass(frozen=True)
class BinnedShare:
    """The running share in each bin, with the bin's half-hours and distinct days."""

    label: str
    bin: int
    n: int
    days: int
    share: float


def bin_shares(
    *, view: pl.DataFrame, bins: np.ndarray, label: str, min_days: int
) -> list[BinnedShare]:
    """Return the share of `state` in each bin (a non-negative integer array), without thin bins."""
    out = []
    for value in sorted({int(b) for b in bins if b >= 0}):
        mask = bins == value
        days = distinct_days(frame=view, mask=mask)
        if days < min_days:
            continue
        out.append(
            BinnedShare(
                label=label,
                bin=value,
                n=int(mask.sum()),
                days=days,
                share=float(view["state"].to_numpy()[mask].mean()),
            )
        )
    return out


def overall_bins(*, view: pl.DataFrame, column: str, n_bins: int) -> np.ndarray:
    """Return the overall equal-count bin of a driver column (-1 where the driver is missing)."""
    return quantile_bins(values=view[column].to_numpy().astype(float), n_bins=n_bins)


def cell_bins(*, view: pl.DataFrame, column: str, n_bins: int) -> np.ndarray:
    """Return the bin of a driver within its cell (half-hour, weekday class, month); -1 if null."""
    cells = CELL_COLUMNS
    out = view.with_columns(
        bin=group_within(value=column, n_groups=n_bins, within=cells),
        n_cell=pl.col(column).count().over(*cells),
    ).with_columns(bin=pl.when(pl.col("n_cell") >= n_bins).then(pl.col("bin")).otherwise(-1))
    return out["bin"].fill_null(-1).to_numpy().astype(int)


def heat_map(*, view: pl.DataFrame, min_days: int = HEATMAP_MIN_DAYS) -> pl.DataFrame:
    """Return the mean state for each weekday and local half-hour, with the distinct days behind it.

    Cells with fewer than `min_days` days are dropped. Clock-change days are left out.
    """
    return (
        usable(frame=view)
        .group_by("weekday", "local_half_hour")
        .agg(share=pl.col("state").mean(), days=pl.col("local_date").n_unique(), n=pl.len())
        .filter(pl.col("days") >= min_days)
        .sort("weekday", "local_half_hour")
    )


def block_shares(*, view: pl.DataFrame) -> pl.DataFrame:
    """Return the mean state by weekday class and block of the local day (night to late)."""
    blocks = pl.Series(
        "block", [BLOCK_NAMES[b] for b in day_block(half_hour=view["local_half_hour"].to_numpy())]
    )
    return (
        usable(frame=view.with_columns(blocks))
        .group_by("weekend", "block")
        .agg(share=pl.col("state").mean(), n=pl.len(), days=pl.col("local_date").n_unique())
        .sort("weekend", "block")
    )


def quantile_row(*, lengths: np.ndarray) -> dict[str, float]:
    """Return the share of length 1, length 1 to 2, and the hazard at lengths 1 to 12."""
    out: dict[str, float] = {"n": float(len(lengths))}
    if len(lengths) == 0:
        return out
    sums = hazard_columns(lengths=lengths).sum(axis=0)
    out["share_len1"] = float(np.mean(lengths == 1))
    out["share_len_le2"] = float(np.mean(lengths <= 2))
    for n in range(1, 13):
        out[f"hazard_{n}"] = float(pooled_hazard(sums=sums, first=n, last=n))
    out["survival_12"] = float(
        ratio(numerator=sums[MAX_SPELL_LENGTH + 11], denominator=len(lengths))
    )
    return out


# ---------------------------------------------------------------- Gas A's levels and edges


def record_levels(*, record: pl.DataFrame) -> dict[str, float]:
    """Return the share of running half-hours at 1, 2, and 3 units and at a full level."""
    power = np.clip(record["power"].to_numpy(), 0.0, None)
    units = np.rint(power / UNIT_MW).astype(int)
    running = units >= 1
    full = np.abs(power - units * UNIT_MW) <= PART_LOAD_MW
    out = {f"share_of_running_at_{k}_units": float((units[running] == k).mean()) for k in (1, 2, 3)}
    out["share_of_running_at_full_level"] = float(full[running].mean())
    out["running_half_hours"] = float(running.sum())
    return out


def record_edges(*, record: pl.DataFrame) -> dict[str, object]:
    """Return Gas A's ramp shape and the clock times of spell starts and stops, as aggregates only.

    A spell start is the first running half-hour of a complete on-spell and a stop is its first idle
    half-hour. The level reached is the largest output in the 4 half-hours from the start (or the 4
    before the stop). The profile is the output as a share of that level from `EDGE_WINDOW`
    half-hours before to `EDGE_WINDOW` after, averaged over spells. A spell reaches its first full
    level within one half-hour when its first output is within `PART_LOAD_MW` of the level.
    """
    power = np.clip(record["power"].to_numpy(), 0.0, None)
    state = np.rint(power / UNIT_MW) >= 1
    segment = record["segment"].to_numpy()
    spells = extract_spells(state=state, valid=np.ones(len(state), dtype=bool), segment=segment)
    hh = record["local_half_hour"].to_numpy()
    offsets = np.arange(-EDGE_WINDOW, EDGE_WINDOW + 1)
    starts, stops = [], []
    start_hh, stop_hh = [], []
    within_one, within_two, n_edges = 0, 0, 0
    for value, length, first in zip(spells.value, spells.length, spells.first_row, strict=True):
        if not value:
            continue
        stop = first + length
        if first - EDGE_WINDOW < 0 or stop + EDGE_WINDOW >= len(power):
            continue
        level = float(power[first : first + 4].max())
        end_level = float(power[stop - 4 : stop].max())
        if level <= 0 or end_level <= 0:
            continue
        starts.append(power[first + offsets] / level)
        stops.append(power[stop + offsets] / end_level)
        start_hh.append(int(hh[first]))
        stop_hh.append(int(hh[stop]))
        n_edges += 1
        within_one += int(power[first] >= level - PART_LOAD_MW)
        within_two += int(
            power[first] >= level - PART_LOAD_MW or power[first + 1] >= level - PART_LOAD_MW
        )
    return {
        "offsets": offsets.tolist(),
        "start_profile": np.mean(starts, axis=0).tolist() if starts else [],
        "stop_profile": np.mean(stops, axis=0).tolist() if stops else [],
        "n_spells": n_edges,
        "share_full_level_within_1": within_one / max(n_edges, 1),
        "share_full_level_within_2": within_two / max(n_edges, 1),
        "start_half_hours": np.bincount(start_hh, minlength=48).tolist(),
        "stop_half_hours": np.bincount(stop_hh, minlength=48).tolist(),
    }
