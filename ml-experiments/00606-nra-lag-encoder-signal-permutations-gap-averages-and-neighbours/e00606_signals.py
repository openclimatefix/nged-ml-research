# ruff: noqa: E501
"""Experiment 606: the lag encoder's extra inputs, built from the gap between a lag and the lag-free forecast.

Time is the half-hour grid index of experiment 578 (48 per UTC day); power is a share of effective
capacity. A forecast origin is at 09:00 UTC. The lag-free table `T` is experiment 598's
(`e00598_signals.LagFreeTable`: the lags-masked median of the freshest forecast issued before the
grid index, lead at most 24 h).

The *gap* at a grid index is the power minus `T` at that index. The network reads, for each of its two
lags (the day lag and the week lag), these column groups:

- `gap`: the gap at the lag's own time, plus a flag that is 1 where the gap is absent (`T` or the lag
  missing).
- `sustain`: experiment 598's sustain count, the share of the 3 days before the lag whose daily mean
  gap
  had the sign of the lag's own daily mean gap and exceeded 5% of capacity.
- `ew`: exponentially weighted averages of the gap over the days up to the lag's own time, at
  half-lives of
  1, 3, and 7 days, each with an absent flag. A causal filter: the value at index `i` reads indices
  up to `i`.
- `exp`: one flag per lag, 1 where the stored weather table has no row for the lag's valid time (the
  expected-embedding input is then the zero fill, which is also a valid embedding value).
- `nb`: the summaries over all electrical neighbours (sum, mean, minimum, maximum, each signed and
  scaled to the Primary's capacity) of the gap and of the three weighted averages, at each lag's time,
  plus the sum of the Primary's own value and the neighbours' sum, the neighbour count, and a flag that
  is 1 where the Primary has no neighbour with a gap.
- `bn`: the comparison arm's single best opposite-sign neighbour (the one whose day-lag gap, scaled
  to the
  Primary's capacity, has the largest size and the opposite sign to the Primary's own day-lag gap), its
  values of the same 8 quantities, the sum of own and neighbour for each, the count, and the none flag.

Share-valued columns are multiplied by `SCALE` and clipped, as in experiment 598.
"""

from dataclasses import dataclass
from typing import Final

import e00598_signals as s598  # ty: ignore[unresolved-import]
import numpy as np
from scipy.signal import lfilter

PER_DAY: Final[int] = s598.PER_DAY
SCALE: Final[float] = s598.SCALE
CLIP: Final[float] = s598.CLIP
LAGS: Final[tuple[str, ...]] = ("day", "week")
HALF_LIVES_DAYS: Final[tuple[int, ...]] = (1, 3, 7)
EW_MIN_WEIGHT: Final[float] = 0.2
"""A weighted average needs at least this share of the weight of a full, gap-free history."""
STATS: Final[tuple[str, ...]] = ("sum", "mean", "min", "max")

QUANTITIES: Final[tuple[str, ...]] = tuple(
    f"{name}_{lag}" for lag in LAGS for name in ("gap", *(f"ew{h}" for h in HALF_LIVES_DAYS))
)
"""The 8 per-lag values that the neighbour summaries cover."""

OWN_NAMES: Final[tuple[str, ...]] = tuple(
    n
    for lag in LAGS
    for n in (
        f"gap_{lag}", f"gap_{lag}_missing",
        f"sustain_{lag}",
        *(f"ew{h}_{lag}" for h in HALF_LIVES_DAYS),
        *(f"ew{h}_{lag}_missing" for h in HALF_LIVES_DAYS),
        f"exp_missing_{lag}",
    )
)  # fmt: skip
NB_NAMES: Final[tuple[str, ...]] = (
    *(f"nb_{st}_{q}" for q in QUANTITIES for st in STATS),
    *(f"own_plus_nb_{q}" for q in QUANTITIES),
    "nb_count", "nb_none",
)  # fmt: skip
BN_NAMES: Final[tuple[str, ...]] = (
    *(f"bn_{q}" for q in QUANTITIES),
    *(f"own_plus_bn_{q}" for q in QUANTITIES),
    "bn_count", "bn_none",
)  # fmt: skip
ALL_NAMES: Final[tuple[str, ...]] = (*OWN_NAMES, *NB_NAMES, *BN_NAMES)
N_ALL: Final[int] = len(ALL_NAMES)
COL: Final[dict[str, int]] = {n: i for i, n in enumerate(ALL_NAMES)}

GROUPS: Final[dict[str, tuple[str, ...]]] = {
    "gap": tuple(f"gap_{lag}{suf}" for lag in LAGS for suf in ("", "_missing")),
    "sustain": tuple(f"sustain_{lag}" for lag in LAGS),
    "ew": tuple(
        f"ew{h}_{lag}{suf}" for lag in LAGS for h in HALF_LIVES_DAYS for suf in ("", "_missing")
    ),
    "exp": tuple(f"exp_missing_{lag}" for lag in LAGS),
    "nb": NB_NAMES,
    "nb_nosum": tuple(n for n in NB_NAMES if not n.startswith("own_plus_")),
    "bn": BN_NAMES,
}
"""Named column groups; an arm's extra inputs are the union of some groups."""

FLAG_OR_COUNT: Final[frozenset[str]] = frozenset(
    n
    for n in ALL_NAMES
    if n.endswith("_missing")
    or n in {"nb_count", "nb_none", "bn_count", "bn_none"}
    or n.startswith("sustain_")
)
"""Columns that never get training noise: flags, counts, and the sustain share."""


def columns_of(*, groups: tuple[str, ...]) -> list[int]:
    """Return the sorted column indices of the union of the named groups."""
    names = {n for g in groups for n in GROUPS[g]}
    return sorted(COL[n] for n in names)


# ---------------------------------------------------------------- weighted averages


def ew_average(*, res: np.ndarray, half_life_days: int) -> np.ndarray:
    """Return (16, grid): the exponentially weighted average of `res` up to and including each index.

    The weight of a reading `k` half-hours old is `0.5 ** (k / (48 * half_life_days))`. Missing readings
    carry no weight. NaN where the weights in the window fall under `EW_MIN_WEIGHT` of a full history.
    """
    alpha = 0.5 ** (1.0 / (PER_DAY * half_life_days))
    fin = np.isfinite(res)
    v = np.where(fin, res, 0.0).astype(np.float64)
    a = [1.0, -alpha]
    num = lfilter([1.0], a, v, axis=1)
    den = lfilter([1.0], a, fin.astype(np.float64), axis=1)
    full = 1.0 / (1.0 - alpha)
    ok = den >= EW_MIN_WEIGHT * full
    return np.where(ok, num / np.maximum(den, 1e-12), np.nan).astype(np.float32)


def quantity_arrays(*, res: np.ndarray) -> dict[str, np.ndarray]:
    """Return, per quantity name without the lag suffix ('gap', 'ew1', ...), the (16, grid) array."""
    out = {"gap": res.astype(np.float32)}
    for h in HALF_LIVES_DAYS:
        out[f"ew{h}"] = ew_average(res=res, half_life_days=h)
    return out


# ---------------------------------------------------------------- the block


def _fill_own(
    *, out: np.ndarray, prim: np.ndarray, lag: str, idx: np.ndarray, miss: np.ndarray,
    arrays: dict[str, np.ndarray], sustain: np.ndarray,
) -> dict[str, np.ndarray]:  # fmt: skip
    """Fill the Primary's own columns of one lag; return the per-quantity own values (NaN if absent)."""
    absent = miss | (idx < 0)
    i = np.clip(idx, 0, None)
    own: dict[str, np.ndarray] = {}
    for name, arr in arrays.items():
        val = np.where(absent, np.nan, arr[prim, i])
        own[f"{name}_{lag}"] = val
        out[:, COL[f"{name}_{lag}"]] = np.nan_to_num(val) * SCALE
        out[:, COL[f"{name}_{lag}_missing"]] = ~np.isfinite(val) & ~absent
    out[:, COL[f"sustain_{lag}"]] = np.where(absent, 0.0, sustain[prim, i])
    return own


CHUNK: Final[int] = 400_000


def signal_columns(
    *, prim: np.ndarray, origin_idx: np.ndarray, day_idx: np.ndarray, week_idx: np.ndarray,
    day_missing: np.ndarray, week_missing: np.ndarray, power_lag: np.ndarray,
    power_sig: np.ndarray, table: s598.LagFreeTable, nb: s598.Neighbours,
    weather_origin: np.ndarray, columns: list[int],
) -> np.ndarray:  # fmt: skip
    """Return the (rows, len(columns)) float32 block of the chosen columns, built in chunks of rows.

    The gap arrays are built once; the neighbour summaries are the costly part, so a chunk is only
    built at full width, and then cut down to `columns`.
    """
    res_lag = (power_lag - table.median).astype(np.float32)
    res_sig = (power_sig - table.median).astype(np.float32)
    arr_lag, arr_sig = quantity_arrays(res=res_lag), quantity_arrays(res=res_sig)
    sustain = s598.sustain_array(daily=s598.daily_mean_array(res=res_lag))
    out = np.empty((len(prim), len(columns)), np.float32)
    for a in range(0, len(prim), CHUNK):
        sl = slice(a, a + CHUNK)
        full = _signal_chunk(
            prim=prim[sl], origin_idx=origin_idx[sl], day_idx=day_idx[sl], week_idx=week_idx[sl],
            day_missing=day_missing[sl], week_missing=week_missing[sl], arr_lag=arr_lag,
            arr_sig=arr_sig, sustain=sustain, table=table, nb=nb, weather_origin=weather_origin,
        )  # fmt: skip
        out[sl] = full[:, columns]
    return out


def _signal_chunk(
    *, prim: np.ndarray, origin_idx: np.ndarray, day_idx: np.ndarray, week_idx: np.ndarray,
    day_missing: np.ndarray, week_missing: np.ndarray, arr_lag: dict[str, np.ndarray],
    arr_sig: dict[str, np.ndarray], sustain: np.ndarray, table: s598.LagFreeTable,
    nb: s598.Neighbours, weather_origin: np.ndarray,
) -> np.ndarray:  # fmt: skip
    """Return the (rows, `N_ALL`) float32 block of every column for one chunk of rows.

    Args:
        prim: Primary index per row.
        origin_idx: Forecast origin grid index per row.
        day_idx: Grid index of the row's day lag time.
        week_idx: Grid index of the row's week lag time.
        day_missing: True where the day lag is missing or masked.
        week_missing: True where the week lag is missing or masked.
        arr_lag: Gap and weighted-gap arrays of the Primary's own power (the lags, shifted in training).
        arr_sig: The same arrays for the neighbour columns' power (equal to `arr_lag` except in training
            and the injected bank, where the power also carries the shifts mirrored onto neighbours).
        sustain: The sustain share array of the own power.
        table: The lag-free table `T`.
        nb: The neighbours and capacities.
        weather_origin: (16, grid) origin (microseconds) of the stored weather run behind each grid
            time's expected input, 0 where none exists.

    Returns:
        The block. Every gap read is of a lag time at least 24 h before the origin.
    """
    n = len(prim)
    for idx, miss in ((day_idx, day_missing), (week_idx, week_missing)):
        assert (idx[~miss] <= origin_idx[~miss] - 48).all(), "a lag time is under 24 h old"
    out = np.zeros((n, N_ALL), np.float32)
    lag_info = (("day", day_idx, day_missing), ("week", week_idx, week_missing))
    own_vals: dict[str, np.ndarray] = {}
    for lag, idx, miss in lag_info:
        own_vals |= _fill_own(
            out=out, prim=prim, lag=lag, idx=idx, miss=miss, arrays=arr_lag, sustain=sustain
        )
        used = ~miss & (idx >= 0)
        i = np.clip(idx, 0, None)
        assert (table.origin_idx[prim[used], i[used]] < origin_idx[used]).all(), (
            "a lag-time T entry comes from a later origin than the row"
        )
        out[:, COL[f"exp_missing_{lag}"]] = (idx >= 0) & (weather_origin[prim, i] <= 0)
    _fill_neighbours(
        out=out, prim=prim, lag_info=lag_info, arr_sig=arr_sig, own_vals=own_vals, nb=nb
    )
    np.clip(out, -CLIP, CLIP, out=out)
    return out


def _fill_neighbours(
    *, out: np.ndarray, prim: np.ndarray, lag_info: tuple, arr_sig: dict[str, np.ndarray],
    own_vals: dict[str, np.ndarray], nb: s598.Neighbours,
) -> None:  # fmt: skip
    """Fill the all-neighbour summaries (`nb`) and the best-opposite-sign neighbour (`bn`) columns."""
    n_q = len(QUANTITIES)
    for p, qs in nb.of.items():
        rows = np.flatnonzero(prim == p)
        if len(rows) == 0 or not qs:
            out[rows, COL["nb_none"]] = 1.0
            out[rows, COL["bn_none"]] = 1.0
            continue
        # vals[k, j, r]: neighbour k's scaled value of quantity j for row r (NaN if absent)
        vals = np.full((len(qs), n_q, len(rows)), np.nan, np.float32)
        for k, q in enumerate(qs):
            scale = nb.capacity[q] / nb.capacity[p]
            for lag, idx, miss in lag_info:
                absent = (miss | (idx < 0))[rows]
                i = np.clip(idx[rows], 0, None)
                for name, arr in arr_sig.items():
                    j = QUANTITIES.index(f"{name}_{lag}")
                    vals[k, j] = np.where(absent, np.nan, arr[q, i] * scale)
        own = np.stack([own_vals[q][rows] for q in QUANTITIES])  # (n_q, rows)
        fin = np.isfinite(vals)
        gap_day = QUANTITIES.index("gap_day")
        have = fin[:, gap_day, :]  # (k, rows): neighbour has a day-lag gap
        count = have.sum(axis=0)
        none = count == 0
        out[rows, COL["nb_count"]] = count
        out[rows, COL["nb_none"]] = none
        z = np.where(fin, vals, 0.0)
        nfin = fin.sum(axis=0)  # (n_q, rows)
        summ = z.sum(axis=0)
        mean = np.where(nfin > 0, summ / np.maximum(nfin, 1), 0.0)
        vmin = np.where(nfin > 0, np.where(fin, vals, np.inf).min(axis=0), 0.0)
        vmax = np.where(nfin > 0, np.where(fin, vals, -np.inf).max(axis=0), 0.0)
        own0 = np.nan_to_num(own)
        for j, q in enumerate(QUANTITIES):
            for st, arr in zip(STATS, (summ, mean, vmin, vmax), strict=True):
                out[rows, COL[f"nb_{st}_{q}"]] = arr[j] * SCALE
            out[rows, COL[f"own_plus_nb_{q}"]] = (own0[j] + summ[j]) * SCALE
        # the single best opposite-sign neighbour, chosen separately for each lag by that lag's gap
        for lag in LAGS:
            idx_q = [
                QUANTITIES.index(f"{n}_{lag}")
                for n in ("gap", *(f"ew{h}" for h in HALF_LIVES_DAYS))
            ]
            gj = QUANTITIES.index(f"gap_{lag}")
            have_l = fin[:, gj, :]
            gd = np.where(have_l, vals[:, gj, :], np.nan)
            sign_own = np.sign(np.nan_to_num(own[gj]))
            opposite = np.where(have_l, -sign_own[None, :] * gd, -np.inf)
            best = opposite.argmax(axis=0)
            top = np.take_along_axis(opposite, best[None, :], axis=0)[0]
            # with no opposite-sign neighbour the strongest neighbour by size is used
            by_size = np.where(have_l, np.abs(gd), -np.inf).argmax(axis=0)
            best = np.where(top > 0, best, by_size)
            pick = np.take_along_axis(vals, best[None, None, :], axis=0)[0]
            pick0 = np.where(have_l.any(axis=0)[None, :], np.nan_to_num(pick), 0.0)
            for j in idx_q:
                q = QUANTITIES[j]
                out[rows, COL[f"bn_{q}"]] = pick0[j] * SCALE
                out[rows, COL[f"own_plus_bn_{q}"]] = (own0[j] + pick0[j]) * SCALE
        out[rows, COL["bn_count"]] = count
        out[rows, COL["bn_none"]] = none


def mask_lag_columns(
    *, block: np.ndarray, columns: list[int], day_masked: np.ndarray, week_masked: np.ndarray
) -> None:
    """Zero, in place, every column that comes from a lag wherever that lag is masked.

    `block` holds the columns `columns` (indices into `ALL_NAMES`). The expected-missing flags depend on
    the lag's time and not on its value, so they stay.
    """
    pos = {ALL_NAMES[c]: j for j, c in enumerate(columns)}
    for name, value in (("nb_count", 0.0), ("bn_count", 0.0), ("nb_none", 1.0), ("bn_none", 1.0)):
        if name in pos:
            block[day_masked, pos[name]] = value
    for lag, mask in (("day", day_masked), ("week", week_masked)):
        for j, c in enumerate(columns):
            name = ALL_NAMES[c]
            if not name.startswith("exp_missing") and (
                name.endswith((f"_{lag}", f"_{lag}_missing"))
            ):
                block[mask, j] = 0.0


# ---------------------------------------------------------------- the pre-fit check and training noise


@dataclass
class NoisePlan:
    """Per column and Primary, the standard deviation of the training noise (0 where none)."""

    sd: np.ndarray
    columns: list[int]
    ratio: np.ndarray


def noise_plan(
    *, train: np.ndarray, train_prim: np.ndarray, test: np.ndarray, test_prim: np.ndarray,
    n_primaries: int, names: list[str], ratio_limit: float = s598.RATIO_LIMIT,
) -> NoisePlan:  # fmt: skip
    """Return the noise plan for a block whose columns are called `names`.

    Where a continuous column's training-to-test standard deviation ratio (per Primary, over rows whose
    value is not exactly zero) is under `ratio_limit`, noise brings the training variance up to the
    test variance. Flags, counts, and the sustain share get none.
    """
    cols = [j for j, n in enumerate(names) if n not in FLAG_OR_COUNT]
    sd = np.zeros((len(cols), n_primaries))
    ratio = np.full((len(cols), n_primaries), np.nan)
    for j, c in enumerate(cols):
        for p in range(n_primaries):
            a = train[(train_prim == p) & (train[:, c] != 0.0), c]
            b = test[(test_prim == p) & (test[:, c] != 0.0), c]
            if len(a) > 100 and len(b) > 100:
                sa, sb = float(a.std()), float(b.std())
                if sb > 0:
                    ratio[j, p] = sa / sb
                    if sa / sb < ratio_limit:
                        sd[j, p] = float(np.sqrt(max(sb**2 - sa**2, 0.0)))
    return NoisePlan(sd=sd, columns=cols, ratio=ratio)


def add_training_noise(*, block: np.ndarray, prim: np.ndarray, plan: NoisePlan, seed: int) -> None:
    """Add Gaussian noise to the planned columns of `block`, in place, on rows where the value is not zero."""
    rng = np.random.default_rng(seed)
    for j, c in enumerate(plan.columns):
        s = plan.sd[j][prim].astype(np.float32)
        if not s.any():
            continue
        nz = rng.standard_normal(len(prim)).astype(np.float32)
        live = block[:, c] != 0.0
        block[:, c] += np.where(live, nz * s, 0.0)
    np.clip(block, -CLIP, CLIP, out=block)
