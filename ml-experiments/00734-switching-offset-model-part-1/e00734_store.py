"""Per-world outputs of experiment 734 that scoring reads without decoding again.

Everything here is a pure function of arrays and tables, so it is tested on synthetic inputs. The
tasks in `e00734_tasks.py` call these functions on the decoded posteriors and write the results to
files under `OUT`; the report in `e00734_score.py` reads only those files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import e00734_events as ev
import e00734_hmm as hmm
import numpy as np
import polars as pl

BLOCK: Final[int] = 12
"""Half-hours per 6-hour block."""
DAY: Final[int] = 48


def blocks_to_half_hours(*, level: np.ndarray) -> np.ndarray:
    """Linearly interpolate a block-grid level of any length to the half-hour grid.

    Experiment 653's `to_half_hours` assumes the full record's `N_BLOCKS` blocks, so it fails on
    the shorter arrays of a tuning world. This version takes the block count from `level`, and
    gives the same values as experiment 653's function when `level` has `N_BLOCKS` entries.
    """
    n = len(level)
    centres = (np.arange(n) + 0.5) * BLOCK
    return np.interp(np.arange(n * BLOCK), centres, level)


AFTER_DAYS: Final[int] = 7
"""Days after the end of a called event that form the fourth scenario."""
SCENARIOS: Final[tuple[str, ...]] = (
    "all rows",
    "quiet rows",
    "inside called events",
    "7 days after",
)
PERMANENT_KINDS: Final[tuple[str, ...]] = ("permanent_shift", "permanent_transfer")
Q_COLS: Final[tuple[str, ...]] = tuple(f"q{int(q * 100):02d}" for q in ev.QUANTILES)
S_COLS: Final[tuple[str, ...]] = tuple(f"s{int(q * 100):02d}" for q in ev.QUANTILES)
"""Quantiles of the offset plus the permanent level, in the order of `ev.QUANTILES`."""


@dataclass(frozen=True)
class SiteBlocks:
    """One Primary's posterior summaries on the blocks of a world."""

    quantiles: np.ndarray
    """`(n_blocks, 7)` quantiles of the switched megawatts, in the order of `ev.QUANTILES`."""
    mean_mw: np.ndarray
    """`(n_blocks,)` posterior mean of the switched megawatts."""
    p_zero: np.ndarray
    """`(n_blocks,)` posterior probability that the offset is zero."""
    o_marg: np.ndarray
    """`(n_blocks, 41)` posterior of the offset level."""
    l_marg: np.ndarray
    """`(n_blocks, 21)` posterior of the permanent level."""
    sigma: float
    """The noise unit of the input in megawatts."""


def site_blocks(*, post: hmm.Posterior, sigma: float) -> SiteBlocks:
    """Summaries of one Primary's posterior.

    Args:
        post: The posterior of the Primary.
        sigma: The noise unit of the input in megawatts.

    Returns:
        The block quantiles, the posterior mean, and the probability of zero offset, with the
        marginals they came from.
    """
    levels = np.array(hmm.FULL_GRID.o_levels, dtype=float)
    o_marg = post.o_marg.astype(np.float64)
    return SiteBlocks(
        quantiles=ev.block_quantiles(post=post, sigma=sigma),
        mean_mw=(o_marg @ levels) * sigma,
        p_zero=o_marg[:, hmm.FULL_GRID.zero],
        o_marg=post.o_marg,
        l_marg=post.l_marg,
        sigma=sigma,
    )


def sum_quantiles(*, o_marg: np.ndarray, l_marg: np.ndarray, sigma: float) -> np.ndarray:
    """Quantiles of the offset plus the permanent level, treating the two marginals as independent.

    The offset grid has a step of 1 noise unit and the permanent grid a step of 2, so the sum lives
    on the 1-unit grid from -40 to 40.

    Args:
        o_marg: `(n, 41)` posterior of the offset level.
        l_marg: `(n, 21)` posterior of the permanent level.
        sigma: The noise unit in megawatts.

    Returns:
        `(n, 7)` quantiles in megawatts, each the first level whose cumulative probability reaches
        the quantile.
    """
    o_lv = np.array(hmm.FULL_GRID.o_levels)
    l_lv = np.array(hmm.FULL_GRID.l_levels)
    lo = int(o_lv.min() + l_lv.min())
    n_out = int(o_lv.max() + l_lv.max()) - lo + 1
    total = np.zeros((o_marg.shape[0], n_out))
    for j, lv in enumerate(l_lv):
        shift = int(lv - l_lv.min())
        total[:, shift : shift + len(o_lv)] += l_marg[:, j, None] * o_marg
    cum = np.cumsum(total, axis=1)
    levels = lo + np.arange(n_out)
    out = np.zeros((o_marg.shape[0], len(ev.QUANTILES)))
    for j, q in enumerate(ev.QUANTILES):
        out[:, j] = levels[np.argmax(cum >= q - 1e-6, axis=1)] * sigma
    return out


def block_slice(*, start: int, end: int | None, n_blocks: int) -> slice:
    """Blocks that overlap the half-hour span `[start, end)` (an open end runs to the last block).

    Args:
        start: First half-hour.
        end: One past the last half-hour, or None or a negative number for no end.
        n_blocks: Number of blocks in the world.

    Returns:
        A slice of at least one block, inside `[0, n_blocks)`.
    """
    first = min(max(start // BLOCK, 0), n_blocks - 1)
    stop = n_blocks if end is None or end < 0 else -(-end // BLOCK)
    return slice(first, min(max(stop, first + 1), n_blocks))


def block_mean_of(*, delta: np.ndarray, window: slice) -> np.ndarray:
    """Mean of a half-hourly series over each block of `window`."""
    seg = delta[window.start * BLOCK : window.stop * BLOCK]
    return seg.reshape(-1, BLOCK).mean(axis=1)


def window_table(
    *,
    world: int,
    arm: str,
    truths: list[dict],
    blocks: dict[str, SiteBlocks],
) -> pl.DataFrame:
    """Posterior quantiles inside every truth window, one row per truth event and block.

    Permanent shifts also get the permanent-level marginal and the quantiles of the offset plus
    the permanent level, because the model is meant to put a permanent shift in the level.

    Args:
        world: The world's seed.
        arm: The arm's name.
        truths: One dict per truth event with `site`, `start`, `end` (None or negative for no
            end), `kind`, and `truth_mw`, the injected megawatts at each block of the window
            (`block_slice` of the truth's span).
        blocks: The posterior summaries of each Primary.

    Returns:
        A table with `world`, `arm`, `site`, `start`, `kind`, `block`, `truth_mw`, the seven
        `q..` columns, `p_zero`, `mean_mw`, and for permanent shifts the `s..` columns and
        `l_marg`; those columns are null for other kinds.
    """
    rows: list[dict] = []
    for t in truths:
        b = blocks.get(t["site"])
        if b is None:
            continue
        n = b.quantiles.shape[0]
        win = block_slice(start=t["start"], end=t["end"], n_blocks=n)
        idx = np.arange(win.start, win.stop)
        truth_mw = np.asarray(t["truth_mw"], dtype=float)
        assert len(truth_mw) == len(idx), (len(truth_mw), len(idx))
        permanent = t["kind"] in PERMANENT_KINDS
        sums = (
            sum_quantiles(o_marg=b.o_marg[idx], l_marg=b.l_marg[idx], sigma=b.sigma)
            if permanent
            else None
        )
        for k, blk in enumerate(idx):
            row: dict = {
                "world": world,
                "arm": arm,
                "site": t["site"],
                "start": int(t["start"]),
                "kind": t["kind"],
                "block": int(blk),
                "truth_mw": float(truth_mw[k]),
                "p_zero": float(b.p_zero[blk]),
                "mean_mw": float(b.mean_mw[blk]),
            }
            row.update({c: float(b.quantiles[blk, j]) for j, c in enumerate(Q_COLS)})
            row.update(
                {c: None if sums is None else float(sums[k, j]) for j, c in enumerate(S_COLS)}
            )
            row["l_marg"] = b.l_marg[blk].astype(float).tolist() if permanent else None
            rows.append(row)
    schema = {
        "world": pl.Int64,
        "arm": pl.String,
        "site": pl.String,
        "start": pl.Int64,
        "kind": pl.String,
        "block": pl.Int64,
        "truth_mw": pl.Float64,
        "p_zero": pl.Float64,
        "mean_mw": pl.Float64,
        **dict.fromkeys(Q_COLS, pl.Float64),
        **dict.fromkeys(S_COLS, pl.Float64),
        "l_marg": pl.List(pl.Float64),
    }
    return pl.DataFrame(rows, schema=schema)


def uninjected_table(*, blocks: dict[str, SiteBlocks]) -> pl.DataFrame:
    """Block quantiles, mean, zero probability, and median of offset plus level, per Primary."""
    frames = []
    for site, b in blocks.items():
        n = b.quantiles.shape[0]
        both = sum_quantiles(o_marg=b.o_marg, l_marg=b.l_marg, sigma=b.sigma)
        frames.append(
            pl.DataFrame(
                {
                    "site": [site] * n,
                    "block": np.arange(n),
                    **{c: b.quantiles[:, j] for j, c in enumerate(Q_COLS)},
                    "mean_mw": b.mean_mw,
                    "p_zero": b.p_zero,
                    "s50": both[:, 3],
                }
            )
        )
    return pl.concat(frames)


def window_mean_shift(*, windows: list[dict], blocks: dict[str, SiteBlocks]) -> pl.DataFrame:
    """Mean posterior shift over each window, such as a hard negative's.

    Args:
        windows: Dicts with `site`, `start`, `end` (None or negative for no end), and any
            descriptive keys, which are kept.
        blocks: The posterior summaries of each Primary.

    Returns:
        One row per window, with `mean_shift_mw` (mean of the posterior mean over the window's
        blocks) and `mean_abs_shift_mw` (mean of its absolute value).
    """
    rows = []
    for w in windows:
        b = blocks.get(w["site"])
        if b is None:
            continue
        win = block_slice(start=w["start"], end=w["end"], n_blocks=len(b.mean_mw))
        seg = b.mean_mw[win]
        rows.append(
            {
                **w,
                "mean_shift_mw": float(seg.mean()),
                "mean_abs_shift_mw": float(np.abs(seg).mean()),
            }
        )
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def call_shift_columns(
    *, called: list, blocks: dict[str, SiteBlocks], is_mva: dict[str, bool]
) -> dict[str, list]:
    """Posterior shift of each called event's Primary and of its partners over the event.

    Args:
        called: The called events (`Event` objects).
        blocks: The posterior summaries of each Primary.
        is_mva: Whether each Primary reports apparent power.

    Returns:
        `post_mean_mw` (the Primary's posterior mean over the event's blocks),
        `partner_post_mean_mw` (the sum of the partners' posterior means over the same blocks,
        null without partners), and `signed_pair` (True when the Primary and every partner report
        signed megawatts).
    """
    own, partner, signed = [], [], []
    for e in called:
        b = blocks.get(e.primary)
        if b is None:
            own.append(None)
            partner.append(None)
            signed.append(False)
            continue
        win = block_slice(start=e.start, end=e.end, n_blocks=len(b.mean_mw))
        own.append(float(b.mean_mw[win].mean()))
        parts = [p for p in e.partners if p in blocks]
        partner.append(float(sum(blocks[p].mean_mw[win].mean() for p in parts)) if parts else None)
        signed.append(
            bool(parts)
            and not is_mva.get(e.primary, False)
            and not any(is_mva.get(p, False) for p in parts)
        )
    return {"post_mean_mw": own, "partner_post_mean_mw": partner, "signed_pair": signed}


def scenario_masks(
    *, length: int, called: list[tuple[int, int | None]], lo: int
) -> dict[str, np.ndarray]:
    """Half-hour masks of the four scenarios at one Primary, from the scored years `lo` on.

    Args:
        length: Half-hours in the world.
        called: `(start, end)` of the called events at the Primary (an open end is None or
            negative).
        lo: The first scored half-hour.

    Returns:
        Boolean masks for `all rows`, `quiet rows` (outside every called event and not within
        7 days after one), `inside called events`, and `7 days after` (after an event's end and
        outside every called event).
    """
    inside = np.zeros(length, dtype=bool)
    after = np.zeros(length, dtype=bool)
    for start, end in called:
        stop = length if end is None or end < 0 else min(end, length)
        inside[max(start, 0) : stop] = True
        if stop < length:
            after[stop : min(stop + AFTER_DAYS * DAY, length)] = True
    scored = np.arange(length) >= lo
    after &= ~inside
    return {
        "all rows": scored,
        "quiet rows": scored & ~inside & ~after,
        "inside called events": scored & inside,
        "7 days after": scored & after,
    }


def scenario_rows(
    *,
    world: int,
    arm: str,
    power: dict[str, np.ndarray],
    forecast: dict[str, np.ndarray],
    capacity: dict[str, float],
    shift_mw: dict[str, np.ndarray],
    called: dict[str, list[tuple[int, int | None]]],
    lo: int,
) -> pl.DataFrame:
    """Normalised mean absolute error of the lag-free forecast, before and after the shift.

    The error before is `|power - forecast| / capacity`. The error after subtracts the posterior
    mean of the switched megawatts, repeated over the half-hours of each block, in hindsight.
    Half-hours without a reading, or without a forecast, are skipped. The row of `inside called
    events` falls by construction, because the shift is fitted to the data inside the events.

    Args:
        world: The world's seed.
        arm: The arm's name.
        power: Half-hourly power of each Primary.
        forecast: Half-hourly lag-free forecast of each Primary.
        capacity: Capacity of each Primary in megawatts.
        shift_mw: Half-hourly posterior mean of the switched megawatts of each Primary.
        called: The called events of each Primary as `(start, end)`.
        lo: The first scored half-hour.

    Returns:
        One row per scenario with the number of half-hours pooled over Primaries and both errors.
    """
    acc = {s: [0, 0.0, 0.0] for s in SCENARIOS}
    for site, p in power.items():
        length = len(p)
        masks = scenario_masks(length=length, called=called.get(site, []), lo=lo)
        sh = np.zeros(length)
        got = shift_mw.get(site)
        if got is not None:
            sh[: min(length, len(got))] = got[:length]
        err_before = np.abs(p - forecast[site]) / capacity[site]
        err_after = np.abs(p - forecast[site] - sh) / capacity[site]
        ok = np.isfinite(err_before)
        for s, m in masks.items():
            use = m & ok
            acc[s][0] += int(use.sum())
            acc[s][1] += float(err_before[use].sum())
            acc[s][2] += float(err_after[use].sum())
    rows = [
        {
            "world": world,
            "arm": arm,
            "scenario": s,
            "n": n,
            "sum_before": b,
            "sum_after": a,
            "nmae_before": b / n if n else float("nan"),
            "nmae_after": a / n if n else float("nan"),
        }
        for s, (n, b, a) in acc.items()
    ]
    return pl.DataFrame(rows)


MIN_IOU: Final[float] = 0.2
"""Overlap with an event of the uninjected record under which a called event is a false event."""


def overlap_iou(*, a: tuple[int, int | None], b: tuple[int, int | None], length: int) -> float:
    """Intersection over union of two spans, an open end (None or negative) running to `length`."""
    a_end = length if a[1] is None or a[1] < 0 else a[1]
    b_end = length if b[1] is None or b[1] < 0 else b[1]
    inter = max(0, min(a_end, b_end) - max(a[0], b[0]))
    union = (a_end - a[0]) + (b_end - b[0]) - inter
    return inter / union if union else 0.0


def false_events(
    *, after: list, before: pl.DataFrame, grown: list[str], lo: int, length: int
) -> list:
    """Called events at grown Primaries, from `lo` on, that overlap no uninjected-record event.

    Args:
        after: The called events of the world with growth or bias (`Event` objects).
        before: The called events of the uninjected record (an events table with `primary`,
            `start`, and `end`, where -1 means no end).
        grown: The Primaries that received the growth or bias.
        lo: The first scored half-hour.
        length: Half-hours in the world.

    Returns:
        The false events.
    """
    spans: dict[str, list[tuple[int, int]]] = {}
    for r in before.iter_rows(named=True):
        spans.setdefault(r["primary"], []).append((int(r["start"]), int(r["end"])))
    return [
        e
        for e in after
        if e.primary in grown
        and e.start >= lo
        and not any(
            overlap_iou(a=(e.start, e.end), b=s, length=length) >= MIN_IOU
            for s in spans.get(e.primary, [])
        )
    ]


SHARE_WINDOW_BLOCKS: Final[int] = 112
"""A 28-day window in 6-hour blocks, the window of the neighbourhood-share baseline."""


def neighbourhood_share_residual(
    *, own: np.ndarray, others: list[np.ndarray], window: int = SHARE_WINDOW_BLOCKS
) -> np.ndarray:
    """Residual of the neighbourhood-share baseline, in the units of `own`.

    The baseline at a block is the Primary's share of its neighbourhood's total, taken as the
    median of the share over a centred window, times the total at that block. The neighbourhood is
    the Primary and its neighbours, and a block with a missing reading anywhere has no total.

    Args:
        own: Block means of the Primary's power, NaN where missing.
        others: Block means of each neighbour's power, the same length as `own`.
        window: Window of the median in blocks (odd or even; the window is centred).

    Returns:
        `own` minus the baseline, NaN where either is missing.
    """
    total = own + (np.sum(others, axis=0) if others else 0.0)
    share = own / np.where(total != 0, total, np.nan)
    half = window // 2
    med = np.full(len(own), np.nan)
    for t in range(len(own)):
        seg = share[max(0, t - half) : t + half + 1]
        if np.isfinite(seg).any():
            med[t] = np.nanmedian(seg)
    return own - med * total


def event_key_mismatches(*, mine: list[tuple[str, int, int]], stored: pl.DataFrame) -> dict:
    """Compare reproduced events with a stored table on `(primary, start, end)`.

    Args:
        mine: `(primary, start, end)` of the reproduced events, `end` of -1 for no end.
        stored: The stored table with `primary`, `start`, and `end` (-1 for no end).

    Returns:
        The counts of both, and the keys found only in one of them (at most 10 of each).
    """
    theirs = [
        (str(r["primary"]), int(r["start"]), int(r["end"])) for r in stored.iter_rows(named=True)
    ]
    only_mine = sorted(set(mine) - set(theirs))
    only_stored = sorted(set(theirs) - set(mine))
    return {
        "n_mine": len(mine),
        "n_stored": len(theirs),
        "only_mine": only_mine[:10],
        "only_stored": only_stored[:10],
        "equal": sorted(mine) == sorted(theirs),
    }


def cascaded_keys(*, table: pl.DataFrame) -> set[tuple[str, int]]:
    """`(site, start)` of the truth events that are cascades (the earliest step of each cascade)."""
    if "cascade_id" not in table.columns:
        return set()
    casc = table.filter(pl.col("cascade_id").is_not_null())
    if casc.height == 0:
        return set()
    firsts = casc.group_by("cascade_id", "site").agg(start=pl.col("start").min())
    return {(r["site"], int(r["start"])) for r in firsts.iter_rows(named=True)}
