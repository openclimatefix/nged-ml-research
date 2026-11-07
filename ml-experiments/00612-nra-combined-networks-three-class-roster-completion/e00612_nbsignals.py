# ruff: noqa: E501
"""Experiment 612 (copy of 610): the lag encoder's extra inputs, built from the gap between a lag and the lag-free forecast.

Copied from experiment 606's `e00606_signals.py`. What changed: the span of the summaries is a choice
(exponentially weighted averages `ew1`, `ew3`, `ew7`, and trailing box means `bx1` to `bx28`); the
neighbour summaries can be compact (`nbc`) and can add the share of opposite-sign neighbours
(`nbopp`); a group can be restricted to the week lag; a block row can be set to the "no information"
value; and the single-best-neighbour columns are gone.

Time is the half-hour grid index of experiment 578 (48 per UTC day); power is a share of effective
capacity. A forecast origin is at 09:00 UTC. The lag-free table `T` is experiment 598's
(`e00598_signals.LagFreeTable`: the lags-masked median of the freshest forecast issued before the
grid index, lead at most 24 h).

The *gap* at a grid index is the power minus `T` at that index. A *span* quantity is a causal
average of the gap up to a grid index. The network reads, for each of its two lags (the day lag and
the week lag), these column groups, each chosen by a token (see `columns_of`):

- `gap`: the gap at the lag's own time, plus a flag that is 1 where the gap is absent.
- `sustain`: experiment 598's sustain count.
- `own:<spans>`: the chosen span quantities of the Primary's own gap, each with an absent flag.
- `exp`: one flag per lag, 1 where the stored weather table has no row for the lag's valid time.
- `nb:<quantities>`: summaries over all electrical neighbours (sum, mean, minimum, maximum, each
  signed and scaled to the Primary's capacity) of the chosen quantities at each lag's time, the sum of
  the Primary's own value and the neighbours' sum, the neighbour count, and a flag that is 1 where the
  Primary has no neighbour with a gap.
- `nbc:<quantities>`: the compact form: only the neighbours' mean and the own-plus-neighbours sum.
- `nbopp:<quantities>`: the share of neighbours whose value has the opposite sign to the Primary's own
  gap at that lag and exceeds `OPP_THRESHOLD` of the Primary's capacity.

A token may end in `@week` or `@day` to keep only that lag's columns. Share-valued columns are
multiplied by `SCALE` and clipped, as in experiment 598.
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
BOX_DAYS: Final[tuple[int, ...]] = (1, 3, 7, 14, 28)
SPANS: Final[tuple[str, ...]] = (
    *(f"ew{h}" for h in HALF_LIVES_DAYS),
    *(f"bx{d}" for d in BOX_DAYS),
)
"""Every span: an exponentially weighted average (half-life in days) or a trailing box mean (window in days)."""
EW_MIN_WEIGHT: Final[float] = 0.2
"""A span needs at least this share of the weight (or of the window) of a full, gap-free history."""
OPP_THRESHOLD: Final[float] = 0.05
"""A neighbour is opposite-sign when its value has the opposite sign to the Primary's and exceeds this share of capacity."""
STATS: Final[tuple[str, ...]] = ("sum", "mean", "min", "max")

QUANTITY_BASES: Final[tuple[str, ...]] = ("gap", *SPANS)
QUANTITIES: Final[tuple[str, ...]] = tuple(f"{b}_{lag}" for lag in LAGS for b in QUANTITY_BASES)
"""The per-lag values that the neighbour summaries can cover."""

OWN_NAMES: Final[tuple[str, ...]] = tuple(
    n
    for lag in LAGS
    for n in (
        f"gap_{lag}", f"gap_{lag}_missing",
        f"sustain_{lag}",
        *(f"{sp}_{lag}" for sp in SPANS),
        *(f"{sp}_{lag}_missing" for sp in SPANS),
        f"exp_missing_{lag}",
    )
)  # fmt: skip
NB_NAMES: Final[tuple[str, ...]] = (
    *(f"nb_{st}_{q}" for q in QUANTITIES for st in STATS),
    *(f"own_plus_nb_{q}" for q in QUANTITIES),
    *(f"nbopp_{q}" for q in QUANTITIES),
    "nb_count", "nb_none",
)  # fmt: skip
ALL_NAMES: Final[tuple[str, ...]] = (*OWN_NAMES, *NB_NAMES)
N_ALL: Final[int] = len(ALL_NAMES)
COL: Final[dict[str, int]] = {n: i for i, n in enumerate(ALL_NAMES)}

FLAG_OR_COUNT: Final[frozenset[str]] = frozenset(
    n
    for n in ALL_NAMES
    if n.endswith("_missing")
    or n in {"nb_count", "nb_none"}
    or n.startswith(("sustain_", "nbopp_"))
)
"""Columns that never get training noise: flags, counts, shares, and the sustain share."""


def _names_of_token(*, token: str) -> list[str]:
    """Return the column names of one group token such as `own:bx3,bx7@week`."""
    body, _, lagpart = token.partition("@")
    kind, _, qs = body.partition(":")
    bases = qs.split(",") if qs else []
    lags = (lagpart,) if lagpart else LAGS
    assert all(lg in LAGS for lg in lags), token
    out: list[str] = []
    if kind == "gap":
        out = [f"gap_{lg}{suf}" for lg in lags for suf in ("", "_missing")]
    elif kind == "sustain":
        out = [f"sustain_{lg}" for lg in lags]
    elif kind == "exp":
        out = [f"exp_missing_{lg}" for lg in lags]
    elif kind == "own":
        assert all(b in SPANS for b in bases), token
        out = [f"{b}_{lg}{suf}" for lg in lags for b in bases for suf in ("", "_missing")]
    elif kind in {"nb", "nbc", "nbopp"}:
        assert all(b in QUANTITY_BASES for b in bases), token
        qs_full = [f"{b}_{lg}" for lg in lags for b in bases]
        if kind == "nb":
            out = [f"nb_{st}_{q}" for q in qs_full for st in STATS]
            out += [f"own_plus_nb_{q}" for q in qs_full]
            out += ["nb_count", "nb_none"]
        elif kind == "nbc":
            out = [f"nb_mean_{q}" for q in qs_full] + [f"own_plus_nb_{q}" for q in qs_full]
            out += ["nb_count", "nb_none"]
        else:
            out = [f"nbopp_{q}" for q in qs_full]
    else:
        msg = f"unknown group token {token!r}"
        raise ValueError(msg)
    return out


def columns_of(*, groups: tuple[str, ...]) -> list[int]:
    """Return the sorted column indices of the union of the group tokens (see the module docstring)."""
    names = {n for g in groups for n in _names_of_token(token=g)}
    return sorted(COL[n] for n in names)


def quantities_needed(*, columns: list[int], neighbours_only: bool) -> set[str]:
    """Return the per-lag quantity names (`gap_day`, `bx3_week`, ...) that the chosen columns read.

    With `neighbours_only`, only the quantities read by a neighbour summary or share are returned.
    """
    need: set[str] = set()
    for c in columns:
        n = ALL_NAMES[c]
        is_nb = n.startswith(("nb_", "own_plus_nb_", "nbopp_"))
        for q in QUANTITIES:
            if n.endswith(q) and (is_nb or (n == q and not neighbours_only)):
                need.add(q)
    return need


def no_information_row(*, columns: list[int]) -> np.ndarray:
    """Return the "no information" value of each chosen column: zero, but 1 for the absent and "none" flags.

    The expected-missing flags describe the lag's time and not the signals, so they are left as NaN
    here, and `drop_signals` leaves them alone.
    """
    out = np.zeros(len(columns), np.float32)
    for j, c in enumerate(columns):
        n = ALL_NAMES[c]
        if n.startswith("exp_missing"):
            out[j] = np.nan
        elif n.endswith("_missing") or n == "nb_none":
            out[j] = 1.0
    return out


def drop_signals(
    *, block: np.ndarray, columns: list[int], rows: np.ndarray, only: list[int] | None = None
) -> None:
    """Set, in place, the signal columns of the rows `rows` (boolean) to the "no information" value.

    `only` limits the columns (indices into `ALL_NAMES`) that are dropped; the default is every chosen
    column except the expected-missing flags.
    """
    base = no_information_row(columns=columns)
    for j, c in enumerate(columns):
        if np.isnan(base[j]) or (only is not None and c not in only):
            continue
        block[rows, j] = base[j]


# ---------------------------------------------------------------- spans


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


def box_average(*, res: np.ndarray, days: int) -> np.ndarray:
    """Return (16, grid): the mean of `res` over the `days` days up to and including each index.

    Missing readings are left out. NaN where fewer than `EW_MIN_WEIGHT` of the window's readings exist.
    """
    width = PER_DAY * days
    fin = np.isfinite(res)
    cs_v = np.cumsum(np.where(fin, res, 0.0).astype(np.float64), axis=1)
    cs_n = np.cumsum(fin.astype(np.float64), axis=1)
    pad = np.zeros((res.shape[0], 1))
    cs_v, cs_n = np.concatenate([pad, cs_v], axis=1), np.concatenate([pad, cs_n], axis=1)
    hi = np.arange(1, res.shape[1] + 1)
    lo = np.clip(hi - width, 0, None)
    num = cs_v[:, hi] - cs_v[:, lo]
    den = cs_n[:, hi] - cs_n[:, lo]
    ok = den >= EW_MIN_WEIGHT * width
    return np.where(ok, num / np.maximum(den, 1e-12), np.nan).astype(np.float32)


def quantity_arrays(*, res: np.ndarray, bases: set[str]) -> dict[str, np.ndarray]:
    """Return, per quantity base ('gap', 'ew1', 'bx7', ...) in `bases`, the (16, grid) array."""
    out: dict[str, np.ndarray] = {}
    for b in bases:
        if b == "gap":
            out[b] = res.astype(np.float32)
        elif b.startswith("ew"):
            out[b] = ew_average(res=res, half_life_days=int(b[2:]))
        else:
            out[b] = box_average(res=res, days=int(b[2:]))
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
        if f"{name}_{lag}_missing" in COL:
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

    Only the quantities that the chosen columns read are built.
    """
    need = quantities_needed(columns=columns, neighbours_only=True)
    bases = {q.rsplit("_", 1)[0] for q in quantities_needed(columns=columns, neighbours_only=False)}
    bases |= {"gap"}
    res_lag = (power_lag - table.median).astype(np.float32)
    res_sig = (power_sig - table.median).astype(np.float32)
    arr_lag = quantity_arrays(res=res_lag, bases=bases)
    arr_sig = quantity_arrays(res=res_sig, bases=bases)
    sustain = s598.sustain_array(daily=s598.daily_mean_array(res=res_lag))
    out = np.empty((len(prim), len(columns)), np.float32)
    for a in range(0, len(prim), CHUNK):
        sl = slice(a, a + CHUNK)
        full = _signal_chunk(
            prim=prim[sl], origin_idx=origin_idx[sl], day_idx=day_idx[sl], week_idx=week_idx[sl],
            day_missing=day_missing[sl], week_missing=week_missing[sl], arr_lag=arr_lag,
            arr_sig=arr_sig, sustain=sustain, table=table, nb=nb, weather_origin=weather_origin,
            need=need,
        )  # fmt: skip
        out[sl] = full[:, columns]
    return out


def _signal_chunk(
    *, prim: np.ndarray, origin_idx: np.ndarray, day_idx: np.ndarray, week_idx: np.ndarray,
    day_missing: np.ndarray, week_missing: np.ndarray, arr_lag: dict[str, np.ndarray],
    arr_sig: dict[str, np.ndarray], sustain: np.ndarray, table: s598.LagFreeTable,
    nb: s598.Neighbours, weather_origin: np.ndarray, need: set[str],
) -> np.ndarray:  # fmt: skip
    """Return the (rows, `N_ALL`) float32 block for one chunk of rows (unneeded columns stay zero).

    Args:
        prim: Primary index per row.
        origin_idx: Forecast origin grid index per row.
        day_idx: Grid index of the row's day lag time.
        week_idx: Grid index of the row's week lag time.
        day_missing: True where the day lag is missing or masked.
        week_missing: True where the week lag is missing or masked.
        arr_lag: Gap and span arrays of the Primary's own power (the lags, shifted in training).
        arr_sig: The same arrays for the neighbour columns' power (equal to `arr_lag` except in training
            and the injected bank, where the power also carries the shifts mirrored onto neighbours).
        sustain: The sustain share array of the own power.
        table: The lag-free table `T`.
        nb: The neighbours and capacities.
        weather_origin: (16, grid) origin (microseconds) of the stored weather run behind each grid
            time's expected input, 0 where none exists.
        need: The per-lag quantity names whose neighbour summaries are built.

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
        out=out, prim=prim, lag_info=lag_info, arr_sig=arr_sig, own_vals=own_vals, nb=nb, need=need
    )
    np.clip(out, -CLIP, CLIP, out=out)
    return out


def _fill_neighbours(
    *, out: np.ndarray, prim: np.ndarray, lag_info: tuple, arr_sig: dict[str, np.ndarray],
    own_vals: dict[str, np.ndarray], nb: s598.Neighbours, need: set[str],
) -> None:  # fmt: skip
    """Fill the all-neighbour summaries and the opposite-sign shares of the quantities in `need`."""
    qlist = [q for q in QUANTITIES if q in need]
    n_q = len(qlist)
    for p, qs in nb.of.items():
        rows = np.flatnonzero(prim == p)
        if len(rows) == 0 or not qs or n_q == 0:
            out[rows, COL["nb_none"]] = 1.0
            continue
        # vals[k, j, r]: neighbour k's scaled value of quantity j for row r (NaN if absent)
        vals = np.full((len(qs), n_q, len(rows)), np.nan, np.float32)
        for k, q in enumerate(qs):
            scale = nb.capacity[q] / nb.capacity[p]
            for lag, idx, miss in lag_info:
                absent = (miss | (idx < 0))[rows]
                i = np.clip(idx[rows], 0, None)
                for name, arr in arr_sig.items():
                    key = f"{name}_{lag}"
                    if key in need:
                        j = qlist.index(key)
                        vals[k, j] = np.where(absent, np.nan, arr[q, i] * scale)
        own = np.stack([own_vals[q][rows] for q in qlist])  # (n_q, rows)
        fin = np.isfinite(vals)
        # the neighbour count follows the day lag's gap, which every arm that has neighbours builds
        have = np.isfinite(
            np.stack(
                [
                    np.where(
                        (lag_info[0][2] | (lag_info[0][1] < 0))[rows],
                        np.nan,
                        arr_sig["gap"][q, np.clip(lag_info[0][1][rows], 0, None)],
                    )
                    for q in qs
                ]
            )
        )
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
        sign_own = np.sign(own0)
        opposite = fin & (np.sign(vals) == -sign_own[None]) & (np.abs(vals) > OPP_THRESHOLD)
        share = np.where(nfin > 0, opposite.sum(axis=0) / np.maximum(nfin, 1), 0.0)
        for j, q in enumerate(qlist):
            for st, arr in zip(STATS, (summ, mean, vmin, vmax), strict=True):
                out[rows, COL[f"nb_{st}_{q}"]] = arr[j] * SCALE
            out[rows, COL[f"own_plus_nb_{q}"]] = (own0[j] + summ[j]) * SCALE
            out[rows, COL[f"nbopp_{q}"]] = share[j] * SCALE


def mask_lag_columns(
    *, block: np.ndarray, columns: list[int], day_masked: np.ndarray, week_masked: np.ndarray
) -> None:
    """Zero, in place, every column that comes from a lag wherever that lag is masked.

    `block` holds the columns `columns` (indices into `ALL_NAMES`). The expected-missing flags depend on
    the lag's time and not on its value, so they stay.
    """
    pos = {ALL_NAMES[c]: j for j, c in enumerate(columns)}
    if "nb_count" in pos:
        block[day_masked, pos["nb_count"]] = 0.0
    if "nb_none" in pos:
        block[day_masked, pos["nb_none"]] = 1.0
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
