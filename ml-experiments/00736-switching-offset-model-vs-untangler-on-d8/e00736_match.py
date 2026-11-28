"""The match rule, the sign test, the intervals, and the verdict mapping of experiment 736.

Everything here is a pure function of arrays and numbers, so the known-answer tests run without
private data. A call is a half-hour with a sign (+1 when the detector says the substation gains
load). A transfer is found when a call of the same sign lies within the tolerance of the
transfer's injected start, one call matches at most one transfer, and the larger transfer is
assigned first.
"""

from __future__ import annotations

from math import comb
from typing import Final, Literal, NamedTuple

import numpy as np
import polars as pl
from e00736_common import (
    BLOCK,
    HALF_HOURS_PER_DAY,
    INDEX_OFFSET,
    MERGE_GAP_HH,
    MIN_COUNTED,
    TOLERANCE_HH,
)

PORTING_THRESHOLD: Final[float] = 0.15
"""Smallest gain of the offset model's share over the step head's share that justifies porting."""
SIGN_ALPHA: Final[float] = 0.05
"""A contrast counts when the two-sided sign test gives a p-value below this."""
V1_RATIO: Final[float] = 1.10
"""Veto V1 passes when the offset model's false-call rate is at most this multiple of the head's."""
N_RESAMPLES: Final[int] = 10_000
INTERVAL_PERCENTILES: Final[tuple[float, float]] = (2.5, 97.5)
MAX_EMPTY_MONTHS: Final[int] = 2
"""More months than this without a discordant transfer make the month-resampled interval fail."""
Verdict = Literal[
    "supported",
    "partly supported",
    "refuted",
    "small gain",
    "not supported",
    "inconclusive",
    "invalid",
]
VERDICTS: Final[tuple[str, ...]] = (
    "supported",
    "partly supported",
    "refuted",
    "small gain",
    "not supported",
    "inconclusive",
    "invalid",
)


def panel_index(*, grid_day: int, half_hour: int) -> int:
    """Return the offset model's panel index of a bank half-hour.

    The bank's half-hour `half_hour` of the calendar day `grid_day` (days since the panel's first
    day) sits at `grid_day * 48 + half_hour + 1` in the panel. A transfer that starts at 13:20 lies
    in the bank half-hour 26 (13:00 to 13:30), so its panel index is 27 plus the day's offset, and
    a transfer that starts at 00:10 lies in half-hour 0, panel index 1.

    Args:
        grid_day: The calendar day, in days since the panel's first day.
        half_hour: The bank's half-hour of the day, 0 to 47.

    Returns:
        The panel index.
    """
    return grid_day * HALF_HOURS_PER_DAY + half_hour + INDEX_OFFSET


def half_hour_of_tick(*, tick: int, ticks_per_half_hour: int = 6) -> int:
    """Return the half-hour (of a dense axis) that holds a 5-minute tick.

    Args:
        tick: The tick, counted from the axis's first tick.
        ticks_per_half_hour: Ticks in one half-hour.

    Returns:
        The half-hour index.
    """
    return tick // ticks_per_half_hour


def merge_calls(
    *, times: np.ndarray, signs: np.ndarray, gap: int = MERGE_GAP_HH
) -> tuple[np.ndarray, np.ndarray]:
    """Merge calls of one sign that lie less than `gap` half-hours apart.

    A chain of such calls becomes one call placed at its first half-hour. Calls of opposite signs
    never merge.

    Args:
        times: The calls' half-hours.
        signs: The calls' signs, +1 or -1.
        gap: Calls closer than this (in half-hours) to the previous call of the same sign merge.

    Returns:
        The merged calls' times and signs, sorted by time.
    """
    out_t: list[int] = []
    out_s: list[int] = []
    for sign in (1, -1):
        picked = np.sort(times[signs == sign])
        last = None
        for t in picked.tolist():
            if last is None or t - last >= gap:
                out_t.append(int(t))
                out_s.append(sign)
            last = t
    order = np.argsort(np.asarray(out_t, dtype=np.int64), kind="stable")
    return (
        np.asarray(out_t, dtype=np.int64)[order],
        np.asarray(out_s, dtype=np.int64)[order],
    )


def match_edges(
    *,
    truth_times: np.ndarray,
    truth_signs: np.ndarray,
    truth_sizes: np.ndarray,
    call_times: np.ndarray,
    call_signs: np.ndarray,
    tolerance: int = TOLERANCE_HH,
) -> np.ndarray:
    """Match each injected edge to at most one call; the larger transfer is assigned first.

    Each transfer, from the largest absolute size down (ties by position), takes the nearest
    unused call of its sign within `tolerance` half-hours (ties by the earlier call). A call
    matches at most one transfer.

    Args:
        truth_times: The injected edges' half-hours.
        truth_signs: The sign the matching call must have.
        truth_sizes: The transfers' sizes (absolute values decide the order).
        call_times: The calls' half-hours.
        call_signs: The calls' signs.
        tolerance: The largest distance in half-hours.

    Returns:
        For each transfer the index of its matching call, or -1.
    """
    taken = np.zeros(len(call_times), dtype=bool)
    out = np.full(len(truth_times), -1, dtype=np.int64)
    order = sorted(range(len(truth_times)), key=lambda i: (-abs(float(truth_sizes[i])), i))
    for i in order:
        dist = np.abs(call_times - truth_times[i])
        ok = (~taken) & (call_signs == truth_signs[i]) & (dist <= tolerance)
        if ok.any():
            cand = np.flatnonzero(ok)
            best = cand[np.lexsort((call_times[cand], dist[cand]))[0]]
            out[i] = best
            taken[best] = True
    return out


def match_by_segment(
    *,
    truth: pl.DataFrame,
    calls: pl.DataFrame,
    tolerance: int = TOLERANCE_HH,
    time_col: str = "start_hh",
    end: bool = False,
) -> np.ndarray:
    """Match each transfer's start (or end) edge to a call of the same segment.

    The match is made segment by segment with `match_edges`. For an end edge the call must have the
    opposite sign of the transfer.

    Args:
        truth: The transfer table, with `seg`, `sign`, `size_points`, and the time column.
        calls: The calls, with `seg`, `time`, and `sign`.
        tolerance: The start tolerance in half-hours.
        time_col: The truth column that holds the edge's half-hour on the segment's dense axis.
        end: Whether to match the end edge, with the opposite sign.

    Returns:
        For each transfer, in the order of `truth`, the row of `calls` that matches it, or -1.
    """
    out = np.full(truth.height, -1, dtype=np.int64)
    seg_of = truth["seg"].to_numpy()
    call_seg = calls["seg"].to_numpy()
    for seg in np.unique(seg_of).tolist():
        rows = np.flatnonzero(seg_of == seg)
        call_rows = np.flatnonzero(call_seg == seg)
        if len(call_rows) == 0:
            continue
        sign = truth["sign"].to_numpy()[rows]
        m = match_edges(
            truth_times=truth[time_col].to_numpy()[rows],
            truth_signs=-sign if end else sign,
            truth_sizes=truth["size_points"].to_numpy()[rows],
            call_times=calls["time"].to_numpy()[call_rows],
            call_signs=calls["sign"].to_numpy()[call_rows],
            tolerance=tolerance,
        )
        out[rows] = np.where(m >= 0, call_rows[np.maximum(m, 0)], -1)
    return out


def sign_test_p(*, b: int, c: int) -> float:
    """Return the exact two-sided sign test p-value of `b` against `c` discordant transfers.

    Args:
        b: Transfers found only by the first detector.
        c: Transfers found only by the second detector.

    Returns:
        The p-value, 1.0 when there are no discordant transfers.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2.0**n
    return min(1.0, 2.0 * tail)


class PairedCounts(NamedTuple):
    """The table of paired counts: both, only the head, only the offset model, neither."""

    both: int
    only_head: int
    only_offset: int
    neither: int

    @property
    def n(self) -> int:
        """Number of transfers in the table."""
        return self.both + self.only_head + self.only_offset + self.neither

    @property
    def c1(self) -> float:
        """The contrast C1, the offset model's share minus the head's share."""
        return (self.only_offset - self.only_head) / self.n if self.n else float("nan")


def paired_counts(*, found_head: np.ndarray, found_offset: np.ndarray) -> PairedCounts:
    """Count the four cells of the paired table.

    Args:
        found_head: Whether the step head found each transfer.
        found_offset: Whether the offset model found each transfer.

    Returns:
        The paired counts.
    """
    h = np.asarray(found_head, dtype=bool)
    o = np.asarray(found_offset, dtype=bool)
    return PairedCounts(
        both=int((h & o).sum()),
        only_head=int((h & ~o).sum()),
        only_offset=int((~h & o).sum()),
        neither=int((~h & ~o).sum()),
    )


def group_interval(
    *,
    found_head: np.ndarray,
    found_offset: np.ndarray,
    groups: np.ndarray,
    n_resamples: int = N_RESAMPLES,
    seed: int = 736,
) -> tuple[float, float]:
    """Return the 95% percentile interval of C1 when whole groups are resampled.

    Each resample draws as many groups as there are with replacement and takes C1 over the
    transfers of the drawn groups. Resamples with no transfer are skipped.

    Args:
        found_head: Whether the step head found each transfer.
        found_offset: Whether the offset model found each transfer.
        groups: The group (world or month) of each transfer.
        n_resamples: Number of resamples.
        seed: Seed of the random generator.

    Returns:
        The lower and upper ends, or NaN twice when no resample holds a transfer.
    """
    labels = np.unique(groups)
    diff = np.asarray(found_offset, dtype=np.int64) - np.asarray(found_head, dtype=np.int64)
    n_g = np.array([(groups == g).sum() for g in labels], dtype=np.float64)
    d_g = np.array([diff[groups == g].sum() for g in labels], dtype=np.float64)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(labels), size=(n_resamples, len(labels)))
    n_tot = n_g[picks].sum(axis=1)
    d_tot = d_g[picks].sum(axis=1)
    ok = n_tot > 0
    if not ok.any():
        return float("nan"), float("nan")
    vals = d_tot[ok] / n_tot[ok]
    lo, hi = np.percentile(vals, INTERVAL_PERCENTILES)
    return float(lo), float(hi)


def months_without_discordant(
    *, found_head: np.ndarray, found_offset: np.ndarray, months: np.ndarray, all_months: np.ndarray
) -> int:
    """Count the months that hold no discordant transfer.

    Args:
        found_head: Whether the step head found each transfer.
        found_offset: Whether the offset model found each transfer.
        months: The month of each transfer.
        all_months: Every month of the experiment (a month with no transfer counts as empty).

    Returns:
        The number of months with no transfer found by only one detector.
    """
    discordant = np.asarray(found_head, dtype=bool) != np.asarray(found_offset, dtype=bool)
    return int(sum(not discordant[months == m].any() for m in all_months))


def verdict(
    *,
    gates_passed: bool,
    counts: PairedCounts,
    interval_world: tuple[float, float],
    interval_month: tuple[float, float],
    empty_months: int,
    v1_passes: bool,
) -> Verdict:
    """Map the counted contrast to a verdict row.

    The rows are applied in this order. `invalid` when a gate failed. `inconclusive` when the
    counted cell holds fewer than 20 transfers, or more than 2 months hold no discordant
    transfer. Otherwise the sign test decides whether C1 counts: with `only_offset` above
    `only_head`, `supported` when V1 passes and C1 reaches the porting threshold, `small gain`
    when V1 passes and C1 falls short of it, and `partly supported` when V1 fails; with
    `only_head` above `only_offset`, `refuted`. When C1 does not count, `not supported` when the
    upper end of the wider interval is below the porting threshold, and `inconclusive` otherwise.

    Args:
        gates_passed: Whether gates G1 to G4 passed.
        counts: The paired counts of the counted cell.
        interval_world: The world-resampled interval of C1.
        interval_month: The month-resampled interval of C1.
        empty_months: Months without a discordant transfer.
        v1_passes: Whether the offset model's false-call rate is within the veto.

    Returns:
        The verdict row.
    """
    if not gates_passed:
        return "invalid"
    if counts.n < MIN_COUNTED or empty_months > MAX_EMPTY_MONTHS:
        return "inconclusive"
    p = sign_test_p(b=counts.only_head, c=counts.only_offset)
    if p < SIGN_ALPHA:
        if counts.only_offset > counts.only_head:
            if not v1_passes:
                return "partly supported"
            return "supported" if counts.c1 >= PORTING_THRESHOLD else "small gain"
        return "refuted"
    widths = [interval_world[1] - interval_world[0], interval_month[1] - interval_month[0]]
    upper = (interval_world, interval_month)[int(np.argmax(widths))][1]
    return "not supported" if upper < PORTING_THRESHOLD else "inconclusive"


def v1_passes(*, offset_false_per_100: float, head_false_per_100: float) -> bool:
    """Return whether veto V1 passes.

    Args:
        offset_false_per_100: The offset model's false edge calls per 100 scored base days.
        head_false_per_100: The step head's false edge calls per 100 scored base days.

    Returns:
        True when the offset model's rate is at most 1.10 times the head's.
    """
    return offset_false_per_100 <= V1_RATIO * head_false_per_100


def false_calls(
    *,
    call_times: np.ndarray,
    call_signs: np.ndarray,
    start_times: np.ndarray,
    start_signs: np.ndarray,
    end_times: np.ndarray,
    end_signs: np.ndarray,
    tolerance: int = TOLERANCE_HH,
) -> np.ndarray:
    """Flag the calls that match no injected start or end edge within the tolerance.

    A call is near a start edge when it has the start's sign, and near an end edge when it has the
    end's sign (the opposite of the transfer's sign). The test is per call, so one edge can excuse
    several calls.

    Args:
        call_times: The calls' half-hours.
        call_signs: The calls' signs.
        start_times: The injected starts' half-hours.
        start_signs: The transfers' signs.
        end_times: The injected ends' half-hours.
        end_signs: The sign a call at an end edge has (minus the transfer's sign).
        tolerance: The largest distance in half-hours.

    Returns:
        A boolean array, True for a false call.
    """
    flag = np.ones(len(call_times), dtype=bool)
    for times, signs in ((start_times, start_signs), (end_times, end_signs)):
        for t, s in zip(times.tolist(), signs.tolist(), strict=True):
            flag &= ~((np.abs(call_times - t) <= tolerance) & (call_signs == s))
    return flag


def block_of(*, half_hour: int) -> int:
    """Return the index of the 6-hour block that holds a half-hour of the panel.

    Args:
        half_hour: The panel's half-hour index.

    Returns:
        The block index.
    """
    return half_hour // BLOCK


__all__ = [
    "MIN_COUNTED",
    "VERDICTS",
    "PairedCounts",
    "block_of",
    "false_calls",
    "group_interval",
    "half_hour_of_tick",
    "match_by_segment",
    "match_edges",
    "merge_calls",
    "months_without_discordant",
    "paired_counts",
    "panel_index",
    "sign_test_p",
    "v1_passes",
    "verdict",
]
