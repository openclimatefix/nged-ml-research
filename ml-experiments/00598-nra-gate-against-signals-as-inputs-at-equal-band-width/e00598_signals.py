# ruff: noqa: E501, D205
"""Experiment 598: the gate's signals as per-row network inputs.

Everything here is numpy except `masked_quantiles`. Time is the half-hour grid index of experiment
578 (`e00422_prep.GRID_START_US`, 48 per UTC day); power is a share of effective capacity. A forecast
origin is at 09:00 UTC, so `origin_idx % 48 == 18`.

The lag-free table `T` (`LagFreeTable`) holds, for each (Primary, grid index), the median of the
fold's frozen A1 step-2 network run with its lags masked, taken from the freshest forecast origin
that lies before the grid index and whose lead is at most 24 h. A signal is then a lookup on `T`
plus the observed (or synthetically shifted) power, as the G4 gate does with experiment 565's E2:

- `dis_day`, `dis_week`: the lag's power minus `T` at the lag's own time (zero where the lag or `T`
  is missing, or where the lag is masked).
- `sustain_day`, `sustain_week`: how many of the 3 days before the lag showed a same-signed
  disagreement above 5% of capacity (trailing 24 h mean of power minus `T`), divided by 3. Only
  times before the lag are read, so only earlier origins' information is used.
- `fresh`, `fresh_missing`: the mean of power minus `T` over the 40 half-hours from the origin minus
  24 h up to the origin minus 4.0 h (experiment 592's G3 reading; 36 of 40 half-hours are needed). It
  is read from power directly, never through a lag.
- `nb_fresh`, `nb_dis_day`, `nb_present`: the mirrored reading of the electrical neighbour whose
  fresh reading is largest in megawatts (its fresh reading and its disagreement at the Primary's day
  lag time, each scaled to the Primary's capacity and negated). `nb_present` is 1 where the Primary
  has a neighbour.
- `dis_day_missing`, `dis_week_missing`, `nb_fresh_missing`, `nb_dis_day_missing`: 1 where the matching
  signal is 0 only because `T` or the power is absent (for the lag signals, only where the lag
  itself is present), so the network can tell "absent" from "agrees with `T`".
- `q05_width`, `q95_width`: the lags-masked network's q05 and q95 for the row, minus its median.

The network adds the two lag ages (columns 39 and 42 of its input) to the signal columns itself.
Share-valued columns are multiplied by `SCALE` so that typical values are of order 1.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np

PER_DAY: Final[int] = 48
ORIGIN_OFFSET: Final[int] = 18
"""Half-hours from 00:00 UTC to the 09:00 UTC forecast origin."""
FRESH_START: Final[int] = 48
FRESH_END: Final[int] = 8
FRESH_MIN_COUNT: Final[int] = 36
"""The fresh window is `[origin - 48, origin - 8)` half-hours, 40 readings, of which 36 are needed."""
MAX_FRESH_LEAD: Final[int] = 48
"""`T` uses forecasts whose lead is at most 24 h (48 half-hours)."""
DAY_MIN_COUNT: Final[int] = 36
SUSTAIN_DAYS: Final[int] = 3
SUSTAIN_THETA: Final[float] = 0.05
"""A day counts toward the sustain count when its mean disagreement exceeds 5% of capacity (the G4 gate's floor)."""
SCALE: Final[float] = 10.0
CLIP: Final[float] = 10.0
SIGNAL_NAMES: Final[tuple[str, ...]] = (
    "dis_day", "dis_week", "sustain_day", "sustain_week", "fresh", "fresh_missing",
    "nb_fresh", "nb_dis_day", "nb_present", "q05_width", "q95_width",
    "dis_day_missing", "dis_week_missing", "nb_fresh_missing", "nb_dis_day_missing",
)  # fmt: skip
N_SIGNALS: Final[int] = len(SIGNAL_NAMES)
COL: Final[dict[str, int]] = {n: i for i, n in enumerate(SIGNAL_NAMES)}
LAG_COLUMNS: Final[dict[str, tuple[str, ...]]] = {
    "day": ("dis_day", "sustain_day", "nb_dis_day", "dis_day_missing", "nb_dis_day_missing"),
    "week": ("dis_week", "sustain_week", "dis_week_missing"),
}
"""Signal columns that come from a lag, so a masked lag zeroes them."""
NOISED: Final[tuple[str, ...]] = ("dis_day", "dis_week", "fresh")
"""Columns that get training noise when the training-to-test standard deviation ratio is below `RATIO_LIMIT`."""
RATIO_LIMIT: Final[float] = 0.8


# ---------------------------------------------------------------- the lag-free table T


@dataclass
class LagFreeTable:
    """The lag-free median and the origin of the forecast it came from, both (16, grid)."""

    median: np.ndarray
    origin_idx: np.ndarray

    def check_no_lookahead(self) -> int:
        """Assert every entry comes from a forecast issued before the grid index it covers.

        Returns the number of filled entries.
        """
        filled = np.isfinite(self.median)
        cols = np.broadcast_to(np.arange(self.median.shape[1]), self.median.shape)
        assert (self.origin_idx[filled] >= 0).all()
        assert (self.origin_idx[filled] < cols[filled]).all(), "a T entry comes from a later origin"
        assert (cols[filled] - self.origin_idx[filled] <= MAX_FRESH_LEAD).all()
        return int(filled.sum())


def build_lag_free_table(
    *, prim: np.ndarray, origin_idx: np.ndarray, target_idx: np.ndarray, median: np.ndarray,
    n_primaries: int, n_grid: int,
) -> LagFreeTable:  # fmt: skip
    """Return `T`: per (Primary, grid index), the lags-masked median of the freshest origin that
    precedes the grid index, among rows with a lead of at most 24 h.
    """
    lead = target_idx - origin_idx
    keep = (lead > 0) & (lead <= MAX_FRESH_LEAD) & np.isfinite(median)
    p, o, t, m = prim[keep], origin_idx[keep], target_idx[keep], median[keep]
    key = p.astype(np.int64) * n_grid + t
    order = np.lexsort((o, key))
    key, p, o, t, m = key[order], p[order], o[order], t[order], m[order]
    last = np.r_[key[1:] != key[:-1], True]
    med = np.full((n_primaries, n_grid), np.nan, np.float32)
    org = np.full((n_primaries, n_grid), -1, np.int64)
    med[p[last], t[last]] = m[last]
    org[p[last], t[last]] = o[last]
    table = LagFreeTable(median=med, origin_idx=org)
    table.check_no_lookahead()
    return table


# ---------------------------------------------------------------- window means


def _prefix(*, res: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return zero-padded cumulative sums of the finite values and of their count, (16, grid + 1)."""
    fin = np.isfinite(res)
    v = np.where(fin, res, 0.0).astype(np.float64)
    zero = np.zeros((res.shape[0], 1))
    return (
        np.concatenate([zero, np.cumsum(v, axis=1)], axis=1),
        np.concatenate([zero, np.cumsum(fin, axis=1)], axis=1),
    )


def window_mean(
    *, res: np.ndarray, prim: np.ndarray, end_idx: np.ndarray, width: int, min_count: int,
    prefix: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:  # fmt: skip
    """Return, per row, the mean of `res[prim, end_idx - width + 1 : end_idx + 1]` over its finite
    values (NaN where fewer than `min_count` are finite or the window starts before the grid).
    """
    cs, cn = prefix if prefix is not None else _prefix(res=res)
    lo = end_idx - width + 1
    ok = lo >= 0
    hi_c = np.clip(end_idx + 1, 0, res.shape[1])
    lo_c = np.clip(lo, 0, res.shape[1])
    s = cs[prim, hi_c] - cs[prim, lo_c]
    n = cn[prim, hi_c] - cn[prim, lo_c]
    out = np.where(ok & (n >= min_count), s / np.maximum(n, 1), np.nan)
    return out.astype(np.float32)


def daily_mean_array(*, res: np.ndarray) -> np.ndarray:
    """Return (16, grid): the mean of `res` over the 48 half-hours ending at each index (NaN if fewer than 36)."""
    n_p, n_g = res.shape
    prim = np.repeat(np.arange(n_p), n_g)
    end = np.tile(np.arange(n_g), n_p)
    return window_mean(
        res=res, prim=prim, end_idx=end, width=PER_DAY, min_count=DAY_MIN_COUNT
    ).reshape(n_p, n_g)


def sustain_array(*, daily: np.ndarray) -> np.ndarray:
    """Return (16, grid): the share of the `SUSTAIN_DAYS` days before each index whose daily mean has
    the sign of the index's own daily mean and exceeds `SUSTAIN_THETA`, counting back without a break.

    Zero where the index's own daily mean is missing or below the floor. Reads only indices at or
    before the index itself.
    """
    n_g = daily.shape[1]
    own = daily
    ok0 = np.isfinite(own) & (np.abs(own) > SUSTAIN_THETA)
    alive = ok0.copy()
    count = np.zeros(daily.shape, np.float32)
    sgn = np.sign(own)
    for k in range(1, SUSTAIN_DAYS + 1):
        prev = np.full(daily.shape, np.nan, np.float32)
        prev[:, PER_DAY * k :] = daily[:, : n_g - PER_DAY * k]
        same = alive & np.isfinite(prev) & (np.sign(prev) == sgn) & (np.abs(prev) > SUSTAIN_THETA)
        count += same
        alive = same
    return np.where(ok0, count / SUSTAIN_DAYS, 0.0).astype(np.float32)


# ---------------------------------------------------------------- the per-row signal block


@dataclass
class Neighbours:
    """Each Primary's electrical neighbours and every Primary's capacity in megawatts."""

    of: dict[int, list[int]]
    capacity: np.ndarray


def fresh_reading(
    *,
    res: np.ndarray,
    prim: np.ndarray,
    origin_idx: np.ndarray,
    prefix: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Return the fresh reading per row: the mean of `res` over `[origin - 48, origin - 8)` (NaN if short)."""
    return window_mean(
        res=res, prim=prim, end_idx=origin_idx - FRESH_END - 1, width=FRESH_START - FRESH_END,
        min_count=FRESH_MIN_COUNT, prefix=prefix,
    )  # fmt: skip


def signal_block(
    *, prim: np.ndarray, origin_idx: np.ndarray, day_idx: np.ndarray, week_idx: np.ndarray,
    day_missing: np.ndarray, week_missing: np.ndarray, power_lag: np.ndarray,
    power_sig: np.ndarray, table: LagFreeTable, widths: np.ndarray, nb: Neighbours,
) -> np.ndarray:  # fmt: skip
    """Return the (rows, `N_SIGNALS`) float32 signal block.

    Args:
        prim: Primary index per row.
        origin_idx: Forecast origin grid index per row.
        day_idx: Grid index of the row's day lag time.
        week_idx: Grid index of the row's week lag time.
        day_missing: True where the day lag is missing or masked.
        week_missing: True where the week lag is missing or masked.
        power_lag: (16, grid) power behind the lag columns (own disagreement and sustain count).
        power_sig: (16, grid) power behind the fresh and neighbour readings. Equal to `power_lag`
            except in training and in the injected bank, where it also carries the shifts mirrored
            onto neighbours.
        table: The lag-free table `T`.
        widths: (rows, 2) lags-masked q05 minus median and q95 minus median, share of capacity.
        nb: The neighbours and capacities.
    """
    n = len(prim)
    assert (day_idx[~day_missing] <= origin_idx[~day_missing] - 48).all()
    assert (week_idx[~week_missing] <= origin_idx[~week_missing] - 48).all()
    out = np.zeros((n, N_SIGNALS), np.float32)
    res_lag = (power_lag - table.median).astype(np.float32)
    res_sig = (power_sig - table.median).astype(np.float32)
    daily = daily_mean_array(res=res_lag)
    sustain = sustain_array(daily=daily)
    for name, idx, miss in (("day", day_idx, day_missing), ("week", week_idx, week_missing)):
        i = np.clip(idx, 0, None)
        dis = np.where(miss | (idx < 0), np.nan, res_lag[prim, i])
        out[:, COL[f"dis_{name}"]] = np.nan_to_num(dis) * SCALE
        out[:, COL[f"dis_{name}_missing"]] = ~np.isfinite(dis) & ~miss & (idx >= 0)
        out[:, COL[f"sustain_{name}"]] = np.where(miss | (idx < 0), 0.0, sustain[prim, i])
        used = ~miss & (idx >= 0)
        assert (table.origin_idx[prim[used], i[used]] < origin_idx[used]).all(), (
            "a lag-time T entry comes from a later origin than the row"
        )
    pre = _prefix(res=res_sig)
    fresh = fresh_reading(res=res_sig, prim=prim, origin_idx=origin_idx, prefix=pre)
    out[:, COL["fresh"]] = np.nan_to_num(fresh) * SCALE
    out[:, COL["fresh_missing"]] = ~np.isfinite(fresh)
    assert ((origin_idx - FRESH_END - 1) < origin_idx).all()
    fok = np.isfinite(fresh)
    assert (table.origin_idx[prim[fok], origin_idx[fok] - FRESH_END - 1] < origin_idx[fok]).all()
    # neighbours: for each Primary, the neighbour with the largest fresh reading in MW
    best_mw = np.zeros(n, np.float32)
    nb_seen = np.zeros(n, bool)
    for p, qs in nb.of.items():
        rows = np.flatnonzero(prim == p)
        if len(rows) == 0 or not qs:
            continue
        out[rows, COL["nb_present"]] = 1.0
        best_abs = np.full(len(rows), -1.0, np.float32)
        for q in qs:
            qp = np.full(len(rows), q)
            f = fresh_reading(res=res_sig, prim=qp, origin_idx=origin_idx[rows], prefix=pre)
            mw = np.nan_to_num(f) * nb.capacity[q]
            nb_seen[rows] |= np.isfinite(f)
            better = np.abs(mw) > best_abs
            best_abs = np.where(better, np.abs(mw), best_abs)
            best_mw[rows] = np.where(better, mw, best_mw[rows])
            i = np.clip(day_idx[rows], 0, None)
            lag_here = ~(day_missing[rows] | (day_idx[rows] < 0))
            d = np.where(lag_here, np.nan_to_num(res_sig[q, i]), 0.0)
            cur = out[rows, COL["nb_dis_day"]]
            out[rows, COL["nb_dis_day"]] = np.where(
                better, -d * nb.capacity[q] / nb.capacity[p] * SCALE, cur
            )
            cur_m = out[rows, COL["nb_dis_day_missing"]]
            out[rows, COL["nb_dis_day_missing"]] = np.where(
                better, lag_here & ~np.isfinite(res_sig[q, i]), cur_m
            )
        out[rows, COL["nb_fresh"]] = -best_mw[rows] / nb.capacity[p] * SCALE
        out[rows, COL["nb_fresh_missing"]] = ~nb_seen[rows]
    out[:, COL["q05_width"]] = np.nan_to_num(widths[:, 0]) * SCALE
    out[:, COL["q95_width"]] = np.nan_to_num(widths[:, 1]) * SCALE
    np.clip(out, -CLIP, CLIP, out=out)
    return out


def mask_lag_columns(*, block: np.ndarray, day_masked: np.ndarray, week_masked: np.ndarray) -> None:
    """Zero, in place, the signal columns that come from a lag wherever that lag is masked."""
    for name, mask in (("day", day_masked), ("week", week_masked)):
        for col in LAG_COLUMNS[name]:
            block[mask, COL[col]] = 0.0


# ---------------------------------------------------------------- synthetic shifts mirrored at a neighbour


def mirror_choice(*, neighbours: dict[int, list[int]], month: str) -> dict[int, int]:
    """Return each Primary's mirrored neighbour for a fold month: neighbour number `(month + Primary) % count`."""
    seed = int(month.replace("-", ""))
    return {p: qs[(seed + p) % len(qs)] for p, qs in neighbours.items() if qs}


def mirrored_power(
    *, base: np.ndarray, shift: np.ndarray, choice: dict[int, int], capacity: np.ndarray
) -> np.ndarray:
    """Return `base` plus, on each Primary's mirrored neighbour, minus that Primary's shift in megawatts
    divided by the neighbour's capacity. `shift` is (16, grid), the shift added to each Primary's power.
    """
    out = base.copy()
    for p, q in choice.items():
        add = -shift[p] * capacity[p] / capacity[q]
        out[q] = np.where(np.isfinite(out[q]), out[q] + add, out[q])
    return out


# ---------------------------------------------------------------- the pre-fit check and training noise


def train_test_sd(
    *, blocks: dict[str, np.ndarray], prim: dict[str, np.ndarray], n_primaries: int
) -> dict[str, np.ndarray]:
    """Return per-Primary standard deviations of `NOISED` columns for the 'train' and 'test' blocks.

    Rows whose column is exactly zero (missing) are left out. Shape (len(NOISED), n_primaries) each.
    """
    out = {}
    for which in ("train", "test"):
        sd = np.full((len(NOISED), n_primaries), np.nan)
        for j, name in enumerate(NOISED):
            col = blocks[which][:, COL[name]]
            for p in range(n_primaries):
                v = col[(prim[which] == p) & (col != 0.0)]
                if len(v) > 100:
                    sd[j, p] = float(v.std())
        out[which] = sd
    return out


def noise_sd(*, sd: dict[str, np.ndarray]) -> np.ndarray:
    """Return the standard deviation of Gaussian noise to add to training-row signals, (len(NOISED), n_primaries).

    Where the training-to-test ratio of a column's standard deviation is below `RATIO_LIMIT` the
    noise brings the training variance up to the test variance; elsewhere it is zero.
    """
    ratio = sd["train"] / sd["test"]
    extra = np.sqrt(np.maximum(sd["test"] ** 2 - sd["train"] ** 2, 0.0))
    return np.where(np.isfinite(ratio) & (ratio < RATIO_LIMIT), extra, 0.0)


def add_training_noise(*, block: np.ndarray, prim: np.ndarray, sd: np.ndarray, seed: int) -> None:
    """Add Gaussian noise (per column and Primary standard deviation `sd`) to `block`, in place."""
    rng = np.random.default_rng(seed)
    for j, name in enumerate(NOISED):
        col = COL[name]
        s = sd[j][prim].astype(np.float32)
        nz = rng.standard_normal(len(prim)).astype(np.float32)
        live = block[:, col] != 0.0
        block[:, col] += np.where(live, nz * s, 0.0)
    np.clip(block, -CLIP, CLIP, out=block)


def placebo_block(*, source: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Return `n` rows (arm S-p) in which each column is resampled independently, with replacement,
    from the same column of `source` (the training block).

    Every column keeps its marginal distribution (a binary flag stays binary) and its variance, and
    the dependence between columns and with the row is broken.
    """
    rng = np.random.default_rng(seed)
    out = np.empty((n, source.shape[1]), np.float32)
    for j in range(source.shape[1]):
        out[:, j] = source[rng.integers(0, len(source), size=n), j]
    return out


# ---------------------------------------------------------------- G4 on T


def fresh_gap_by_day(*, res_sig: np.ndarray, n_days: int) -> np.ndarray:
    """Return (16, days): the fresh reading at each origin day's 09:00 UTC origin (NaN if short)."""
    n_p = res_sig.shape[0]
    days = np.arange(n_days)
    out = np.full((n_p, n_days), np.nan, np.float32)
    pre = _prefix(res=res_sig)
    origin = days * PER_DAY + ORIGIN_OFFSET
    ok = origin < res_sig.shape[1]
    for p in range(n_p):
        out[p, ok] = fresh_reading(
            res=res_sig, prim=np.full(ok.sum(), p), origin_idx=origin[ok], prefix=pre
        )
    return out


def g4_wide(
    *, gap: np.ndarray, thr: np.ndarray, origin_days: list[int], neighbours: dict[int, list[int]],
    hold_days: int = 7,
) -> np.ndarray:  # fmt: skip
    """Return (16, days) True where the G4 band is wide: the gate is on, or within `hold_days` days
    after it turned off. Experiment 592's `gate_fresh` and experiment 583's `hold_from`.
    """
    import e00583_common as c  # ty: ignore[unresolved-import]
    import e00592_common as g  # ty: ignore[unresolved-import]

    on, snap = g.gate_fresh(gap=gap, thr=thr, origin_days=origin_days, neighbours=neighbours)
    return c.hold_from(snap=snap, days=hold_days) | on


# ---------------------------------------------------------------- masked network quantiles (torch)


def masked_quantiles(
    *, model, x: np.ndarray, primary: np.ndarray, target_mean: np.ndarray, target_std: np.ndarray,  # noqa: ANN001
    device: str, batch: int = 262_144,
) -> np.ndarray:  # fmt: skip
    """Return (rows, 3) q05, q50, q95 (share of capacity) of `model` with both lags masked.

    The masked copy of each batch sets the day and week lag columns to the absent encoding
    (experiment 572's `absent`). `x` is not changed.
    """
    import e00510_common as c  # ty: ignore[unresolved-import]
    import e00572_lags as lg  # ty: ignore[unresolved-import]
    import torch  # ty: ignore[unresolved-import]

    amax = lg.age_max(cap_steps=lg.CAP_24H)
    mu = torch.from_numpy(target_mean).to(device)
    sd = torch.from_numpy(target_std).to(device)
    out = []
    model.eval()
    for a in range(0, len(x), batch):
        xb = torch.from_numpy(x[a : a + batch]).to(device)
        for c0, age in ((37, amax[0]), (40, amax[1])):
            xb[:, c0], xb[:, c0 + 1], xb[:, c0 + 2] = 0.0, 1.0, float(age)
        pb = torch.from_numpy(primary[a : a + batch]).to(device)
        q = torch.sort(c.serving_quantiles(model=model, x=xb, p=pb, outlier=False), dim=1).values
        q = q * sd[pb][:, None] + mu[pb][:, None]
        out.append(q[:, [1, 5, 9]].float().cpu().numpy())
    return np.concatenate(out)
