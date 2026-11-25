# ruff: noqa: E501
"""Experiment 728: the event-level scorer, and the two oracle forecasters of gate checks GH1 and GH2.

**The row layout** follows experiment 713's layout (`e00713_layout`). A forecast origin is the
contract origin at 12:00 UTC of grid day `d`, which is grid index `48 d + 24`. The target row of lead
day `L` (1 to 14) holds the 48 half-hours `48 d + 24 + 48 (L - 1) + j` for `j` in 0 to 47. The newest
power an origin may read is at grid index `c - 54`. A forecast frame holds one row per (event,
scored Primary, origin, lead day, target half-hour) with the columns of `ROW_COLUMNS`, and the
quantiles `q01` to `q99`.

**What is scored.** A temporary move is scored against the unswitched truth `clean_share` on the
counted slice, the target half-hours from the onset of the move to 7 days after the move ends
(design section 0, `counted_slice`). The slice is cut on the target half-hour, as the design text
says, and not on the origin day as experiment 715's `event_crps` cuts its slices. A permanent change
is scored against the injected truth `truth_share` (guard P). Each event has equal weight: the value
of an event is the mean over its scored Primaries (the donor and every metered recipient) of the mean
quantile score over the Primary's rows in the slice.

**The oracles** are forecasters with no network: `truth_oracle` returns the correct truth of each
event (design GH1), and `persistence_oracle` carries the power of the same half-hour of the newest
visible day forward (design GH2). Each is run on the injected world and on the real world, and the
follow share is the mean median difference over the target rows, over the mean injected change
(`visible_change`), as experiment 715's `follow_share` computes it.

**Points that the reviewer should know.**

- The slice is cut on the target half-hour. If the design meant the origin day, the counts of GH4
  change, and the gate prints the counts of both cuts.
- **The grid end.** Experiment 709's grid ends on 2026-06-30, so for fold 2026-06 a target half-hour
  of a late origin can lie past the grid (and a stored `g2` pass holds injected power only to 45 days
  after the first day of the fold month). `take` returns NaN for an index past the array. A row
  whose truth is NaN is not scorable: `row_scores` gives it a NaN score, the counted reading and
  guard P drop it, and `follow_shares` drops it. The gate counts the nominal rows, the rows past the
  grid, the rows with a NaN truth, and the scorable rows of every counted move, and prints them.
"""

from dataclasses import dataclass
from typing import Final

import e00713_layout as lay  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.eval_protocol.metrics import quantile_score

QCOLS: Final = ("q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99")
ORIGIN_DAYS: Final = (7, 10, 14, 21)
"""Days after the onset at which the scored origins sit (experiment 715's origins)."""
LEAD_DAYS: Final = tuple(range(1, lay.N_LEAD_DAYS + 1))
FOLLOW_LEAD_DAYS: Final = (2, 10)
"""The first and last lead day of the follow share (experiment 715)."""
PER_DAY: Final = lay.STEPS_PER_DAY
CONTRACT_OFFSET: Final = 24
"""The contract origin is at half-hour 24 of its day (12:00 UTC)."""
SLICE_AFTER_END_DAYS: Final = 7
NEVER_ENDS: Final = 10**6
"""The `end_day` stand-in of a move that does not end inside the grid."""
MIN_CHANGE: Final = 1e-6
UNITS: Final = 100.0
"""Scores are in points of capacity: shares are multiplied by this number."""
ROW_COLUMNS: Final = (
    "event_id", "primary", "kind", "shape", "mirror_state", "onset_day", "end_day", "days_after_onset",
    "lead_day", "valid_hh", "truth_share", "clean_share", "change", "in_force",
    "persist_injected", "persist_real",
)  # fmt: skip


def contract_index(*, origin_day: int) -> int:
    """Return the grid index of the contract origin of grid day `origin_day`."""
    return origin_day * PER_DAY + CONTRACT_OFFSET


def target_index(*, origin_day: int, lead_day: int) -> np.ndarray:
    """Return the 48 grid indices of the target row of `lead_day` (1 to 14) at an origin."""
    return contract_index(origin_day=origin_day) + PER_DAY * (lead_day - 1) + np.arange(PER_DAY)


def newest_visible(*, origin_day: int) -> int:
    """Return the newest grid index that the origin of `origin_day` may read."""
    return contract_index(origin_day=origin_day) - lay.NEWEST_VISIBLE_BEFORE_CONTRACT


def counted_slice(*, onset_day: int, end_day: int) -> tuple[int, int]:
    """Return the counted slice `[lo, hi)` of a temporary move, in grid half-hours.

    The slice runs from the onset to `SLICE_AFTER_END_DAYS` days after the end of the move. A move
    that does not end (`end_day` negative) runs to `NEVER_ENDS`.
    """
    end = end_day if end_day >= 0 else NEVER_ENDS
    return onset_day * PER_DAY, (end + SLICE_AFTER_END_DAYS) * PER_DAY


@dataclass(frozen=True)
class SliceRows:
    """The rows of one scored Primary of a temporary move at the scored origins.

    Attributes:
        nominal: Rows whose target half-hour is in the counted slice.
        by_origin: Rows whose origin day is in the slice (experiment 715's cut), for comparison.
        past_grid: Nominal rows whose target half-hour lies past the end of the grid.
        no_truth: Nominal rows inside the grid whose unswitched truth is NaN.
        scorable: Nominal rows with a finite unswitched truth.
        slice_1: Scorable rows while the move is in force.
        slice_2: Scorable rows in the 7 days after the move ends.
    """

    nominal: int
    by_origin: int
    past_grid: int
    no_truth: int
    scorable: int
    slice_1: int
    slice_2: int


def slice_rows(*, clean: np.ndarray, onset_day: int, end_day: int) -> SliceRows:
    """Count the rows of one scored Primary of a temporary move in the counted slice.

    Args:
        clean: (G,) the Primary's unswitched power over capacity on the grid.
        onset_day: Grid day of the onset.
        end_day: Grid day of the end of the move.

    Returns:
        The counts of `SliceRows`. The slice is cut on the target half-hour, as the design says.
    """
    lo, hi = counted_slice(onset_day=onset_day, end_day=end_day)
    mid = (end_day if end_day >= 0 else NEVER_ENDS) * PER_DAY
    nominal = by_origin = past = no_truth = slice_1 = slice_2 = 0
    for k in ORIGIN_DAYS:
        origin = onset_day + k
        for lead in LEAD_DAYS:
            idx = target_index(origin_day=origin, lead_day=lead)
            inside = (idx >= lo) & (idx < hi)
            by_origin += PER_DAY if lo <= origin * PER_DAY < hi else 0
            truth = take(array=clean, index=idx)
            past_row = inside & (idx >= len(clean))
            missing = inside & ~past_row & ~np.isfinite(truth)
            good = inside & np.isfinite(truth)
            nominal += int(inside.sum())
            past += int(past_row.sum())
            no_truth += int(missing.sum())
            slice_1 += int((good & (idx < mid)).sum())
            slice_2 += int((good & (idx >= mid)).sum())
    return SliceRows(
        nominal=nominal, by_origin=by_origin, past_grid=past, no_truth=no_truth,
        scorable=slice_1 + slice_2, slice_1=slice_1, slice_2=slice_2,
    )  # fmt: skip


def take(*, array: np.ndarray, index: np.ndarray) -> np.ndarray:
    """Return `array[index]`, with NaN where the index lies past the end of the array."""
    out = np.full(index.shape, np.nan)
    inside = index < len(array)
    out[inside] = array[index[inside]]
    return out


def visible_change(
    *, injected: np.ndarray, clean: np.ndarray, onset_day: int, origin_day: int
) -> float:
    """Return the mean injected change over the half-hours the origin can read since the onset.

    The change is `injected - clean` over `[48 onset_day, newest visible]`, NaN where no finite
    half-hour exists (experiment 715's `visible_change`).
    """
    newest = newest_visible(origin_day=origin_day)
    lo = onset_day * PER_DAY
    if newest < lo:
        return float("nan")
    d = injected[lo : newest + 1] - clean[lo : newest + 1]
    d = d[np.isfinite(d)]
    return float(d.mean()) if d.size else float("nan")


def persistence_index(*, valid: np.ndarray, origin_day: int) -> np.ndarray:
    """Return, for each target half-hour, the grid index of the same half-hour on the newest visible day.

    That is `valid - 48 m` for the smallest `m` that puts the index at or before the newest visible
    half-hour.
    """
    newest = newest_visible(origin_day=origin_day)
    m = np.maximum(1, -((newest - valid) // PER_DAY))
    return valid - PER_DAY * m


def rows_for_event(
    *, event_id: str, primary: int, event: dict, injected: np.ndarray, clean: np.ndarray,
) -> pl.DataFrame:  # fmt: skip
    """Return the rows of one scored Primary of one event, without quantiles.

    Args:
        event_id: The event's identifier (month, pass, and row of the truth table).
        primary: The scored Primary's index.
        event: The event's row of the truth table: `truth` (`era` or `switch`), `kind` (the shape),
            `mirror_state`, `onset_day`, `end_day`.
        injected: (G,) the Primary's injected power over capacity on the grid, NaN outside its stored
            window.
        clean: (G,) the Primary's unswitched power over capacity on the grid.

    Returns:
        A frame with the columns `ROW_COLUMNS` except the quantiles. `in_force` is True where the
        change is still in force on the newest visible day of the origin. `persist_injected` and
        `persist_real` hold the persistence oracle's forecast on the injected and the real world.
    """
    onset, end = int(event["onset_day"]), event["end_day"]
    end_day = int(end) if end is not None and end >= 0 else -1
    pieces = []
    for k in ORIGIN_DAYS:
        origin = onset + k
        change = visible_change(injected=injected, clean=clean, onset_day=onset, origin_day=origin)
        newest = newest_visible(origin_day=origin)
        in_force = newest >= onset * PER_DAY and (end_day < 0 or newest < end_day * PER_DAY)
        for lead in LEAD_DAYS:
            valid = target_index(origin_day=origin, lead_day=lead)
            back = persistence_index(valid=valid, origin_day=origin)
            pieces.append(
                pl.DataFrame(
                    {
                        "days_after_onset": np.full(PER_DAY, k),
                        "lead_day": np.full(PER_DAY, lead),
                        "valid_hh": valid,
                        "truth_share": take(array=injected, index=valid),
                        "clean_share": take(array=clean, index=valid),
                        "change": np.full(PER_DAY, change),
                        "in_force": np.full(PER_DAY, in_force),
                        "persist_injected": injected[back],
                        "persist_real": clean[back],
                    }
                )
            )
    return pl.concat(pieces).with_columns(
        event_id=pl.lit(event_id), primary=pl.lit(primary, dtype=pl.Int64),
        kind=pl.lit(str(event["truth"])), shape=pl.lit(str(event["kind"])),
        mirror_state=pl.lit(str(event["mirror_state"])),
        onset_day=pl.lit(onset, dtype=pl.Int64), end_day=pl.lit(end_day, dtype=pl.Int64),
    ).select(*ROW_COLUMNS)  # fmt: skip


def with_constant_quantiles(*, rows: pl.DataFrame, level: pl.Expr) -> pl.DataFrame:
    """Return `rows` with every quantile column equal to the expression `level` (an oracle forecast)."""
    return rows.with_columns(**dict.fromkeys(QCOLS, level))


def truth_oracle(*, rows: pl.DataFrame, world: str) -> pl.DataFrame:
    """Return the forecast of the truth oracle (design GH1) on `world`, `injected` or `real`.

    On the real world the oracle forecasts the clean power. On the injected world it forecasts the
    injected power for a permanent change and the clean power for a temporary move, which is what a
    forecast of the unswitched truth returns.
    """
    assert world in {"injected", "real"}, world
    if world == "real":
        return with_constant_quantiles(rows=rows, level=pl.col("clean_share"))
    level = (
        pl.when(pl.col("kind") == "era")
        .then(pl.col("truth_share"))
        .otherwise(pl.col("clean_share"))
    )
    return with_constant_quantiles(rows=rows, level=level)


def persistence_oracle(*, rows: pl.DataFrame, world: str) -> pl.DataFrame:
    """Return the forecast of the persistence oracle (design GH2) on `world`.

    The forecast of a target half-hour is the power of the same half-hour on the newest visible day,
    of the injected panel or of the clean panel.
    """
    assert world in {"injected", "real"}, world
    column = "persist_injected" if world == "injected" else "persist_real"
    return with_constant_quantiles(rows=rows, level=pl.col(column))


def row_scores(*, forecast: pl.DataFrame) -> np.ndarray:
    """Return the quantile score of every row, in points of capacity (NaN where the truth is NaN).

    A row of a permanent change is scored against `truth_share` and a row of a temporary move
    against `clean_share`.
    """
    actual = (
        forecast.select(
            pl.when(pl.col("kind") == "era")
            .then(pl.col("truth_share"))
            .otherwise(pl.col("clean_share"))
        )
        .to_series()
        .to_numpy()
        .astype(np.float64)
    )
    q = forecast.select(list(QCOLS)).to_numpy().astype(np.float64)
    out = np.full(len(actual), np.nan)
    ok = np.isfinite(actual) & np.isfinite(q).all(axis=1)
    out[ok] = quantile_score(actual=actual[ok] * UNITS, quantiles=q[ok] * UNITS)
    return out


def scoring_rows(*, forecast: pl.DataFrame, what: str) -> pl.DataFrame:
    """Return the rows that enter a reading, with their score in a column `crps`.

    `what` is `counted` (the target half-hours of temporary moves in the counted slice) or `guard_p`
    (all scored rows of permanent changes, origins 7 to 21 days after the onset).
    """
    frame = forecast.with_columns(crps=pl.Series(row_scores(forecast=forecast)))
    if what == "guard_p":
        return frame.filter((pl.col("kind") == "era") & pl.col("crps").is_finite())
    assert what == "counted", what
    end = pl.when(pl.col("end_day") >= 0).then(pl.col("end_day")).otherwise(NEVER_ENDS)
    return frame.filter(
        (pl.col("kind") == "switch")
        & (pl.col("valid_hh") >= pl.col("onset_day") * PER_DAY)
        & (pl.col("valid_hh") < (end + SLICE_AFTER_END_DAYS) * PER_DAY)
        & pl.col("crps").is_finite()
    )


def event_values(*, scored: pl.DataFrame) -> pl.DataFrame:
    """Return one value per event: the mean over its scored Primaries of the Primary's mean score.

    Each event has equal weight whatever its number of rows or scored Primaries. The columns are
    `event_id`, `crps`, `n_rows`, and `n_primaries`.
    """
    per_primary = scored.group_by("event_id", "primary").agg(
        pl.col("crps").mean(), pl.len().alias("n_rows")
    )
    return (
        per_primary.group_by("event_id")
        .agg(pl.col("crps").mean(), pl.col("n_rows").sum(), pl.len().alias("n_primaries"))
        .sort("event_id")
    )


def follow_shares(*, injected: pl.DataFrame, real: pl.DataFrame) -> pl.DataFrame:
    """Return the follow share of every (event, Primary, origin).

    The follow share is the mean median difference over the target rows of lead days 2 to 10, over
    the mean injected change (experiment 715's reading).

    Args:
        injected: A forecast frame on the injected world.
        real: The same forecast frame on the real world.

    Returns:
        A frame with `event_id`, `primary`, `kind`, `shape`, `days_after_onset`, `in_force`, `change`, and
        `follow`. A row with no finite change has a null `follow`. Rows with a NaN median are dropped before
        the mean, so a NaN truth never gives a NaN follow share.
    """
    keys = ["event_id", "primary", "days_after_onset", "lead_day", "valid_hh"]
    lo, hi = FOLLOW_LEAD_DAYS
    a = injected.filter(pl.col("lead_day").is_between(lo, hi))
    b = real.filter(pl.col("lead_day").is_between(lo, hi)).select(
        *keys, pl.col("q50").alias("q50_real")
    )
    joined = a.join(b, on=keys, how="inner")
    assert joined.height == a.height, "the injected and the real forecasts cover different rows"
    finite = joined.filter(pl.col("q50").is_finite() & pl.col("q50_real").is_finite())
    return (
        finite.group_by("event_id", "primary", "kind", "shape", "days_after_onset")
        .agg(
            (pl.col("q50") - pl.col("q50_real")).mean().alias("shift"),
            pl.col("change").first(),
            pl.col("in_force").first(),
        )
        .with_columns(
            follow=pl.when(pl.col("change").is_finite() & (pl.col("change").abs() > MIN_CHANGE))
            .then(pl.col("shift") / pl.col("change"))
            .otherwise(None)
        )
        .sort("event_id", "primary", "days_after_onset")
    )


@dataclass(frozen=True)
class OracleReading:
    """What an oracle scored on a set of events.

    Attributes:
        max_abs_score: The largest absolute row score on the counted slice and on guard P.
        mean_counted_crps: The mean over events of the counted CRPS (temporary moves).
        follow_permanent: The mean follow share after permanent changes of shape `flat`.
        follow_permanent_all: The same over all four shapes of permanent change.
        follow_temporary: The mean follow share after temporary moves in force at the origin.
        n_temporary: Temporary moves with a counted score.
        n_permanent: Permanent changes with a follow share.
    """

    max_abs_score: float
    mean_counted_crps: float
    follow_permanent: float
    follow_permanent_all: float
    follow_temporary: float
    n_temporary: int
    n_permanent: int


def _mean(series: pl.Series) -> float:
    """Return the mean of a numeric series as a float."""
    return float(np.mean(series.to_numpy()))


def read_oracle(*, injected: pl.DataFrame, real: pl.DataFrame) -> OracleReading:
    """Score an oracle's forecasts on the injected and the real world, and return the readings."""
    counted = scoring_rows(forecast=injected, what="counted")
    guard = scoring_rows(forecast=injected, what="guard_p")
    scores = np.concatenate([counted["crps"].to_numpy(), guard["crps"].to_numpy()])
    values = event_values(scored=counted)
    follow = follow_shares(injected=injected, real=real).drop_nulls("follow")
    every = follow.filter(pl.col("kind") == "era")
    permanent = every.filter(pl.col("shape") == "flat")["follow"]
    temporary = follow.filter((pl.col("kind") == "switch") & pl.col("in_force"))["follow"]
    return OracleReading(
        max_abs_score=float(np.abs(scores).max()) if scores.size else float("nan"),
        mean_counted_crps=_mean(values["crps"]) if values.height else float("nan"),
        follow_permanent=_mean(permanent) if len(permanent) else float("nan"),
        follow_permanent_all=_mean(every["follow"]) if every.height else float("nan"),
        follow_temporary=_mean(temporary) if len(temporary) else float("nan"),
        n_temporary=values.height,
        n_permanent=len(permanent),
    )
