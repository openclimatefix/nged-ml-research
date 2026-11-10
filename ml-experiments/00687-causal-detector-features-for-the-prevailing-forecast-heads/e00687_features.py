# ruff: noqa: E501, D103
"""Experiment 687: causal detector features of a lag, in noise units (pure numpy).

Time is the half-hour grid index of experiment 578 (48 per UTC day); power is a share of effective
capacity. The *gap* is the power minus the lag-free median `T` of experiment 598. A feature at the
cutoff index `t` reads only gap values at indices `<= t`.

Per cutoff index the features are (names in `FEATURE_NAMES`):

- `lvl_<w>`: D1, the w-half-hour block mean of the gap minus the median of that block mean over all
  Primaries, in noise units (w = 12, 48, 144: 6 hours, 1 day, 3 days).
- `step_<w>`: D4, the block mean ending at `t` minus the block mean ending w half-hours earlier, in
  noise units.
- `cstep_<w>`: the same step with the all-Primaries median step removed.
- `mir_12`, `mir_48`, `mir_dt`: D2, the largest value over the electrical neighbours of minus the sign
  of the Primary's own step times the neighbour's step (neighbour signal read from `gap_sig`), and the
  hours between the Primary's last 2-noise-unit step and that neighbour's.
- `cu_depth`, `cu_age`, `cu_ratio`, `cu_ret`: D5 and D6, a two-sided cumulative-sum detector over
  the previous 24 six-hour blocks of the common-mode-removed level: the depth at the excursion's
  extreme, hours since the excursion started, the sum over its threshold, and the return towards the
  earlier level (0 at the extreme, -1 when the level is back to where the excursion began).
- `rate`: D3, the share of the previous 60 days whose common-mode-removed 6-hour step exceeded 3
  noise units.
- `missing`: 1 where the block means needed were absent.

`cu_flag` (the explicit "inside a suspected event" flag) is returned separately.
"""

from dataclasses import dataclass
from typing import Final

import e00598_signals as s598  # ty: ignore[unresolved-import]
import numpy as np

W: Final[tuple[int, ...]] = (12, 48, 144)
BLOCK: Final[int] = 12
MIN_FRAC: Final[float] = 0.5
CU_BLOCKS: Final[int] = 24
CU_DRIFT: Final[float] = 1.5
CU_THRESHOLD: Final[float] = 6.0
STEP_FLAG: Final[float] = 2.0
ROUTINE_NU: Final[float] = 3.0
ROUTINE_WINDOW: Final[int] = 60 * 48
CLIP: Final[float] = 20.0
FEATURE_NAMES: Final[tuple[str, ...]] = (
    *(f"lvl_{w}" for w in W), *(f"step_{w}" for w in W), *(f"cstep_{w}" for w in W),
    "mir_12", "mir_48", "mir_dt", "cu_depth", "cu_age", "cu_ratio", "cu_ret", "rate", "missing",
)  # fmt: skip
N_FEATURES: Final[int] = len(FEATURE_NAMES)


def block_unit(*, gaps: list[np.ndarray], w: int) -> np.ndarray:
    """Return per Primary 1.4826 x the median absolute deviation of consecutive non-overlapping w-block mean differences.

    `gaps` holds one (Primary, time) array per development month; differences are pooled over months.
    """
    n_p = gaps[0].shape[0]
    pooled: list[list[np.ndarray]] = [[] for _ in range(n_p)]
    for gap in gaps:
        nb = gap.shape[1] // w
        blocks = gap[:, : nb * w].reshape(n_p, nb, w)
        cnt = np.isfinite(blocks).sum(axis=2)
        with np.errstate(all="ignore"):
            mean = np.where(cnt >= MIN_FRAC * w, np.nanmean(np.where(np.isfinite(blocks), blocks, np.nan), axis=2), np.nan)  # fmt: skip
        d = np.diff(mean, axis=1)
        for p in range(n_p):
            pooled[p].append(d[p][np.isfinite(d[p])])
    out = np.empty(n_p)
    for p in range(n_p):
        v = np.concatenate(pooled[p])
        out[p] = 1.4826 * np.median(np.abs(v - np.median(v))) if len(v) >= 20 else np.nan
    return out


@dataclass
class Grids:
    """Per-Primary block means, steps, and crossing times on a column window of the grid."""

    start: int
    lvl: dict[int, np.ndarray]
    cm_lvl: dict[int, np.ndarray]
    step: dict[int, np.ndarray]
    cstep: dict[int, np.ndarray]
    sig_step: dict[int, np.ndarray]
    last_cross: np.ndarray
    sig_last_cross: np.ndarray
    routine_cs: np.ndarray


def _block_mean(*, gap: np.ndarray, w: int) -> np.ndarray:
    cs, cn = s598._prefix(res=gap)
    n = gap.shape[1]
    hi = np.arange(1, n + 1)
    lo = np.clip(hi - w, 0, None)
    cnt = cn[:, hi] - cn[:, lo]
    tot = cs[:, hi] - cs[:, lo]
    ok = (cnt >= MIN_FRAC * w) & (hi - w >= 0)[None, :]
    return np.where(ok, tot / np.maximum(cnt, 1), np.nan)


def _shift(*, a: np.ndarray, k: int) -> np.ndarray:
    out = np.full_like(a, np.nan)
    out[:, k:] = a[:, :-k]
    return out


def build_grids(*, gap: np.ndarray, gap_sig: np.ndarray, gap_ref: np.ndarray, units: dict[int, np.ndarray], start: int) -> Grids:
    """Return the grids for the column window `gap[:, :]`, whose column 0 is grid index `start`.

    `gap_ref` is the uninjected gap; the all-Primaries median is taken over it, so that only the
    scored Primary carries its shift (real operation has few concurrent shifts).
    """
    lvl, cm_lvl, step, cstep, sig_step = {}, {}, {}, {}, {}
    for w in W:
        u = units[w][:, None]
        bm = _block_mean(gap=gap, w=w) / u
        lvl[w] = bm
        ref = _block_mean(gap=gap_ref, w=w) / u
        with np.errstate(all="ignore"):
            med = np.nanmedian(ref, axis=0, keepdims=True)
        cm_lvl[w] = bm - med
        step[w] = bm - _shift(a=bm, k=w)
        cstep[w] = cm_lvl[w] - _shift(a=cm_lvl[w], k=w)
        if w in (12, 48):
            sb = _block_mean(gap=gap_sig, w=w) / u
            sig_step[w] = sb - _shift(a=sb, k=w)
    big = np.arange(gap.shape[1])[None, :]
    last_cross = np.maximum.accumulate(np.where(np.abs(np.nan_to_num(step[12])) >= STEP_FLAG, big, -1), axis=1)  # fmt: skip
    sig_last_cross = np.maximum.accumulate(np.where(np.abs(np.nan_to_num(sig_step[12])) >= STEP_FLAG, big, -1), axis=1)  # fmt: skip
    routine = (np.abs(np.nan_to_num(cstep[12])) > ROUTINE_NU).astype(np.float64)
    routine_cs = np.concatenate([np.zeros((gap.shape[0], 1)), np.cumsum(routine, axis=1)], axis=1)
    return Grids(start=start, lvl=lvl, cm_lvl=cm_lvl, step=step, cstep=cstep, sig_step=sig_step,
                 last_cross=last_cross, sig_last_cross=sig_last_cross, routine_cs=routine_cs)  # fmt: skip


def features_at(
    *, g: Grids, prim: np.ndarray, t: np.ndarray, neighbours: dict[int, list[int]]
) -> tuple[np.ndarray, np.ndarray]:
    """Return ((n, N_FEATURES) features, (n,) explicit flag) at cutoff grid indices `t` (absolute); reads indices <= t only."""
    i = (t - g.start).astype(np.int64)
    assert (i >= 0).all()
    n = len(prim)
    out = np.zeros((n, N_FEATURES), np.float32)
    miss = np.zeros(n, bool)

    def put(name: str, v: np.ndarray) -> None:
        nonlocal miss
        bad = ~np.isfinite(v)
        miss |= bad
        out[:, FEATURE_NAMES.index(name)] = np.clip(np.where(bad, 0.0, v), -CLIP, CLIP)

    for w in W:
        put(f"lvl_{w}", g.cm_lvl[w][prim, i])
        put(f"step_{w}", g.step[w][prim, i])
        put(f"cstep_{w}", g.cstep[w][prim, i])
    # D2: opposite-sign neighbour step
    for w in (12, 48):
        own = g.step[w][prim, i]
        best = np.full(n, np.nan)
        best_nb = np.full(n, -1)
        for p in np.unique(prim):
            rows = np.flatnonzero(prim == p)
            nbs = neighbours.get(int(p), [])
            if not nbs:
                continue
            nbv = g.sig_step[w][np.ix_(nbs, i[rows])]
            val = -np.sign(own[rows])[None, :] * nbv
            val = np.where(np.isfinite(val), val, -np.inf)
            k = val.argmax(axis=0)
            top = val[k, np.arange(len(rows))]
            best[rows] = np.where(np.isfinite(top), top, np.nan)
            best_nb[rows] = np.where(np.isfinite(top), np.array(nbs)[k], -1)
        put(f"mir_{w}", np.where(best_nb >= 0, best, 0.0))
        if w == 12:
            ok = best_nb >= 0
            own_t = g.last_cross[prim, i]
            nb_t = g.sig_last_cross[np.where(ok, best_nb, 0), i]
            dt = np.where(ok & (own_t >= 0) & (nb_t >= 0), (own_t - nb_t) / 2.0, 0.0)
            out[:, FEATURE_NAMES.index("mir_dt")] = np.clip(dt, -72.0, 72.0) / 12.0
    # D5, D6: two-sided cumulative sum on the 6-hour level, over the previous 24 blocks
    z = np.stack([g.cm_lvl[12][prim, np.clip(i - BLOCK * j, 0, None)] for j in range(CU_BLOCKS - 1, -1, -1)], axis=1)  # fmt: skip
    z = np.where((i[:, None] - BLOCK * np.arange(CU_BLOCKS - 1, -1, -1)[None, :]) >= 0, z, np.nan)
    zf = np.nan_to_num(z)
    best_s = np.zeros(n)
    flag = np.zeros(n, bool)
    depth = np.zeros(n)
    age = np.zeros(n)
    ret = np.zeros(n)
    for sgn in (1.0, -1.0):
        s = np.zeros(n)
        start_j = np.zeros(n, np.int64)
        for j in range(CU_BLOCKS):
            new = np.maximum(0.0, s + sgn * zf[:, j] - CU_DRIFT)
            start_j = np.where((s == 0.0) & (new > 0.0), j, start_j)
            s = new
        better = s > best_s
        if better.any():
            rows = np.flatnonzero(better)
            sj = start_j[rows]
            sl = np.arange(CU_BLOCKS)[None, :] >= sj[:, None]
            ext = np.where(sl, sgn * zf[rows], -np.inf).max(axis=1)
            pre = np.where(sj > 0, sgn * zf[rows, np.clip(sj - 1, 0, None)], 0.0)
            now = sgn * zf[rows, -1]
            span = np.maximum(ext - pre, 1.0)
            best_s[rows] = s[rows]
            depth[rows] = sgn * (ext - pre)
            age[rows] = (CU_BLOCKS - 1 - sj) * BLOCK / 2.0
            ret[rows] = (now - ext) / span
            flag[rows] = s[rows] > CU_THRESHOLD
    out[:, FEATURE_NAMES.index("cu_depth")] = np.clip(depth, -CLIP, CLIP)
    out[:, FEATURE_NAMES.index("cu_age")] = age / 144.0
    out[:, FEATURE_NAMES.index("cu_ratio")] = np.clip(best_s / CU_THRESHOLD, 0.0, CLIP)
    out[:, FEATURE_NAMES.index("cu_ret")] = np.clip(ret, -CLIP, CLIP)
    # D3: routine step rate over the previous 60 days, ending 6 hours before the cutoff (no overlap with the lag)
    hi = np.clip(i - BLOCK, 0, None)
    lo = np.clip(hi - ROUTINE_WINDOW, 0, None)
    rate = (g.routine_cs[prim, hi + 1] - g.routine_cs[prim, lo + 1]) / np.maximum(hi - lo, 1)
    out[:, FEATURE_NAMES.index("rate")] = rate * 10.0
    out[:, FEATURE_NAMES.index("missing")] = miss
    return out, flag
