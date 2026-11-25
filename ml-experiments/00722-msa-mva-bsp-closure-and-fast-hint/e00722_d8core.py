# ruff: noqa: E501
"""Experiment 722: the event-study statistics of the stored BSP run, copied from its private folder.

The code is a copy of `d8_core.py` in the private `d8_bsp_residual/` folder, with type hints and a
shared seed argument. It is used by job 1 for the era-split table and the co-step by generator
type. Each event is a change in the engine state; the null pool of an event is the set of quiet
half-hours with the same matched key; the bootstrap resamples the events and every key's pool.
"""

from typing import Any, Final

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

PRE: Final[int] = 4
"""Half-hours before an event in its step (2 hours)."""

POST: Final[int] = 4
"""Half-hours from an event on in its step (2 hours)."""

PAD: Final[int] = 6
"""Half-hours kept clear around a gap or an engine change (as in the stored BSP run)."""

MIN_POOL: Final[int] = 5
"""An event whose matched null pool has fewer candidates is dropped and counted."""


def step_vec(*, x: np.ndarray) -> np.ndarray:
    """Return `mean(x[i..i+3]) - mean(x[i-4..i-1])`, NaN unless all 8 values are finite."""
    w = sliding_window_view(x, PRE + POST)
    out = np.full(len(x), np.nan)
    out[PRE : PRE + len(w)] = w[:, PRE:].mean(1) - w[:, :PRE].mean(1)
    return out


def event_pools(
    *,
    step: np.ndarray,
    events: np.ndarray,
    step_at_events: np.ndarray,
    key_of: np.ndarray,
    cand_by_key: dict[int, np.ndarray],
) -> tuple[list[np.ndarray | None], int]:
    """Return each event's null pool (or None), and the count dropped for a missing or small pool."""
    out: list[np.ndarray | None] = []
    n_dropped = 0
    for j, h in enumerate(events):
        if not np.isfinite(step_at_events[j]):
            out.append(None)
            continue
        c = cand_by_key.get(int(key_of[h]), np.array([], dtype=int))
        c = c[np.isfinite(step[c])]
        if len(c) >= MIN_POOL:
            out.append(c)
        else:
            out.append(None)
            n_dropped += 1
    return out, n_dropped


def study_core(
    *,
    step: np.ndarray,
    events: np.ndarray,
    sign: np.ndarray,
    scale: bool,
    key_of: np.ndarray,
    cand_by_key: dict[int, np.ndarray],
    rng: np.random.Generator,
    n_boot: int,
    m_null: int,
    chunk: int = 100,
) -> dict[str, Any] | None:
    """Event study of the step series at the events against matched null pools.

    The null pool of each key is resampled once per replicate and shared by every event with that
    key, so the sampling error of the finite pools enters the interval and the detection limit.
    Events are resampled as well.
    """
    s_ev = step[events] / (sign if scale else 1.0)
    all_pools, n_drop = event_pools(
        step=step, events=events, step_at_events=s_ev, key_of=key_of, cand_by_key=cand_by_key
    )
    keep = np.array([j for j, p in enumerate(all_pools) if p is not None], dtype=int)
    if len(keep) < MIN_POOL:
        return None
    sv = s_ev[keep]
    sc = sign[keep] if scale else np.ones(len(keep))
    n = len(keep)
    ukeys, kid = np.unique(key_of[events][keep], return_inverse=True)
    pool_of = {int(key_of[events[j]]): all_pools[j] for j in keep}
    plist = [pool_of[int(u)] for u in ukeys]
    plen = np.array([len(p) for p in plist])
    koff = np.concatenate([[0], np.cumsum(plen)[:-1]])
    flat = np.concatenate(plist)
    pos_len = np.repeat(plen, plen)
    pos_off = np.repeat(koff, plen)

    def draw(ix: np.ndarray, m: int) -> np.ndarray:
        out = np.empty(ix.shape)
        for a in range(0, ix.shape[0], chunk):
            ixc = ix[a : a + chunk]
            r = ixc.shape[0]
            res = pos_off[None, :] + (rng.random((r, len(flat))) * pos_len[None, :]).astype(int)
            resampled = step[flat[res]]
            ku = kid[ixc]
            col = koff[ku][:, :, None] + (
                rng.random((*ixc.shape, m)) * plen[ku][:, :, None]
            ).astype(int)
            vals = np.take_along_axis(resampled, col.reshape(r, -1), axis=1).reshape(r, n, m)
            out[a : a + r] = (vals / sc[ixc][:, :, None]).mean(2)
        return out

    mean = float(sv.mean())
    null_mean = float(
        np.mean(np.array([step[p].mean() for p in (all_pools[j] for j in keep)]) / sc)
    )
    ix = rng.integers(0, n, size=(n_boot, n))
    exc_b = (sv[ix] - draw(ix, m_null)).mean(1)
    excess = mean - null_mean
    lo, hi = np.percentile(exc_b, [2.5, 97.5])
    return {
        "n_events": n,
        "n_dropped_small_pool": n_drop,
        "excess": excess,
        "excess_lo": float(lo),
        "excess_hi": float(hi),
        "se_excess": float(exc_b.std()),
    }


def any_between(*, flag: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Return True where `flag` holds at some row in `[i + lo, i + hi]` (inclusive)."""
    n = len(flag)
    cs = np.concatenate([[0], np.cumsum(flag.astype(np.int64))])
    idx = np.arange(n)
    a = np.clip(idx + lo, 0, n)
    b = np.clip(idx + hi + 1, 0, n)
    return (cs[b] - cs[a]) > 0


def isolated_events(
    *,
    change: np.ndarray,
    k_path: np.ndarray,
    era: np.ndarray,
    has_row: np.ndarray,
    in_drop: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the isolated engine events: row, signed unit change, era, and a null-candidate mask.

    An event is isolated when it is the only change in the window from 5 half-hours before to 3
    after. Events whose window touches a gap or a multi-day drop are dropped, and the same
    half-hours are removed from the null candidates, as in the stored BSP run.
    """
    n = len(k_path)
    prev = np.concatenate([[np.nan], k_path[:-1]])
    h_all = np.flatnonzero(change)
    h_all = h_all[(h_all >= PRE + 1) & (h_all < n - POST)]
    cs = np.concatenate([[0], np.cumsum(change.astype(np.int64))])
    count = cs[np.clip(h_all + POST, 0, n)] - cs[np.clip(h_all - PRE - 1, 0, n)]
    bad = in_drop | any_between(flag=~has_row, lo=-PAD, hi=PAD)
    bad_win = any_between(flag=bad, lo=-PRE - 1, hi=POST - 1)
    keep = (count == 1) & ~bad_win[h_all]
    h = h_all[keep]
    dk = (k_path - prev)[h]
    near = any_between(flag=change, lo=-PAD, hi=PAD)
    return h, dk, era[h], ~near & has_row & ~bad_win


def candidates_by_key(*, key: np.ndarray, ok: np.ndarray) -> dict[int, np.ndarray]:
    """Return the null candidates (row indices) of each matched key."""
    return {int(k): np.flatnonzero(ok & (key == k)) for k in np.unique(key[ok & (key >= 0)])}
