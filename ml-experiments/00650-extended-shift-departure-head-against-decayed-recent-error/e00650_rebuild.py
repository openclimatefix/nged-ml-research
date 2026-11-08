# ruff: noqa: E501, E402, ANN001, D103
"""Experiment 650, step 2: rebuild K1 on the extended-shift bank (CPU).

K1 is the seed mean of five seeds, experiment 613's tail constants (fitted on earlier months' event-free
rows), and the G4 gated band, all through experiment 613's own `e00613_score` machinery (imported, not
edited) as `e00647_k1_rebuild` did. This script differs from experiment 647's rebuild in three ways:
the forecasts come from `inject/` (step 1), the shifts are this experiment's `own` and `mirror` arrays
(any shape, not only a constant block), and the gate's gap is the daily mean of those arrays added
to the fresh and stored gaps (experiment 613 added a constant to each day of a block).

`--validate` runs the same code on experiment 581's passes and experiment 613's stored forecasts and
compares the rebuilt K1 with experiment 647's stored `k1_bank` rows, so a difference between the two
rebuilds shows up before any new pass is read.

Writes `k1_bank/rows_<month>_r<pass>.npz` (same keys as experiment 647's) and `k1_bank/events_<month>.parquet`.

Usage: python e00650_rebuild.py --months 2025-10 2025-12 --passes 0 1 ... | --validate
"""

import argparse
import sys
import time

import e00650_common as c650
import numpy as np
import polars as pl

_p = argparse.ArgumentParser()
_p.add_argument("--months", nargs="+", default=["2025-10"])
_p.add_argument("--passes", nargs="+", type=int, default=list(c650.ALL_PASSES))
_p.add_argument("--validate", action="store_true")
MINE = _p.parse_args()
OUT = c650.ROWS
OUT.mkdir(parents=True, exist_ok=True)
sys.argv = ["e00613_score.py", str(OUT / "scratch"), "--arm", "k1_d5"]
import e00581_bank as bank  # ty: ignore[unresolved-import]
import e00613_score as S  # ty: ignore[unresolved-import]  # noqa: N812
import e00647_common as c647  # ty: ignore[unresolved-import]

PER = 48
QCOLS = S.c.QCOLS
VALID_TOL = 1e-4


def load_seed_mean(*, t, month: str, r: int, rows, order, sorted_keys, inject_dir) -> np.ndarray:
    """Return the five-seed mean of the injected quantiles of pass `r` for the table rows `rows` (NaN where absent)."""
    parts = []
    for d in ("d0", "d14"):
        seeds = []
        for s in S.SEEDS:
            f = inject_dir / f"{month}_r{r:02d}_{d}_s{s}.parquet"
            if not f.exists():
                raise FileNotFoundError(f)
            seeds.append(pl.read_parquet(f, columns=[*S.c.KEYS, *QCOLS]))
        parts.append(pl.concat(seeds).group_by(S.c.KEYS).agg([pl.col(q).mean() for q in QCOLS]))
    sm = S.c.k578.sites()
    prim_of = dict(zip(sm["time_series_id"].to_list(), sm["primary"].to_list(), strict=True))
    df = pl.concat(parts).with_columns(
        prim=pl.col("time_series_id").replace_strict(prim_of, return_dtype=pl.Int64),
        idx=((pl.col("valid_time").dt.epoch("us") - S.c.k578.prep.GRID_START_US) // S.c.k578.prep.STEP_US).cast(pl.Int64),
        oidx=((pl.col("power_fcst_init_time").dt.epoch("us") - S.c.k578.prep.GRID_START_US) // S.c.k578.prep.STEP_US).cast(pl.Int64),
    )  # fmt: skip
    kk = S.s.key(df["prim"].to_numpy(), df["oidx"].to_numpy(), df["idx"].to_numpy())
    pos = np.clip(np.searchsorted(sorted_keys, kk), 0, len(sorted_keys) - 1)
    ok = sorted_keys[pos] == kk
    slot = np.full(t.n, -1, np.int64)
    slot[order[pos[ok]]] = np.flatnonzero(ok)
    q = df.select(QCOLS).to_numpy().astype(np.float32)
    out = np.full((len(rows), 11), np.nan, np.float32)
    s_ = slot[rows]
    has = s_ >= 0
    out[has] = np.sort(q[s_[has]], axis=1)
    return out


def fresh_gap_from_delta(*, fresh, delta: np.ndarray) -> np.ndarray:
    """Return the fresh gap (16, days) with `delta` (16, grid) added to the observed power it averages."""
    total = fresh.sum.copy()
    np.add.at(total, (fresh.p, fresh.day), delta[fresh.p, fresh.idx])
    return np.where(fresh.cnt >= S.g.MIN_FRESH_HALF_HOURS, total / np.maximum(fresh.cnt, 1), np.nan)


def daily_mean(*, delta: np.ndarray, n_days: int) -> np.ndarray:
    """Return the UTC-day mean of `delta` (16, grid) as (16, n_days)."""
    d = delta[:, : n_days * PER].reshape(delta.shape[0], n_days, PER)
    return np.nanmean(d, axis=2)


def one_pass(*, inp, t, m: str, r: int, own: np.ndarray, mirror: np.ndarray, events: list[dict],
             lower, upper, order, sorted_keys, inject_dir, gi, days_m, origin_days_m, real_wide_all):  # fmt: skip
    """Return the rebuilt rows of pass `r` as a dict of arrays, or None if there are none."""
    in_m = t.month == m
    lo_p = {e["primary"]: e["onset_idx"] - 7 * PER for e in events}
    hi_p = {e["primary"]: e["end_idx"] + 14 * PER for e in events}
    lo = np.array([lo_p.get(p, 1 << 40) for p in range(16)])[t.prim]
    hi = np.array([hi_p.get(p, -1) for p in range(16)])[t.prim]
    rows = np.flatnonzero(in_m & t.free & (t.idx % 2 == 0) & (t.idx >= lo) & (t.idx < hi))
    vin = load_seed_mean(t=t, month=m, r=r, rows=rows, order=order, sorted_keys=sorted_keys, inject_dir=inject_dir)  # fmt: skip
    ok = np.isfinite(vin).all(axis=1)
    rows, vin = rows[ok], vin[ok]
    if len(rows) == 0:
        return None
    n_days = gi["gap"].shape[1]
    sl = slice(days_m.min(), days_m.max() + 1)
    both = (own + mirror)[:, : n_days * PER]
    dm = daily_mean(delta=np.nan_to_num(both), n_days=n_days)
    old = np.full_like(gi["gap"], np.nan)
    old[:, sl] = gi["gap"][:, sl] + dm[:, sl]
    fresh = fresh_gap_from_delta(fresh=inp.fresh, delta=np.nan_to_num(both))
    fresh[:, : origin_days_m[0] + 1] = np.nan
    fresh[:, origin_days_m[-1] + 1 :] = np.nan
    states = S.k592.g.all_gates(old_gap=old, old_thr=gi["thr"], fresh_gap=fresh, fresh_thr=inp.fresh_thr,
                                origin_days=origin_days_m, neighbours=inp.nb)  # fmt: skip
    wide_inj = S.k592.g.arm_masks(states=states)["G4"][0]
    wide_rows = wide_inj[t.prim[rows], t.origin_day[rows]]
    v_clean, m_q = t.arms["V"][rows], t.arms["M"][rows]
    lw, up = lower[rows], upper[rows]
    wf = S.month_factor(month_col=t.month[rows])
    q_inj = S.systems(v=vin, m=m_q, wide=wide_rows, lower=lw, upper=up, wf=wf)
    q_twin = S.systems(v=v_clean, m=m_q, wide=real_wide_all[rows], lower=lw, upper=up, wf=wf)
    return {
        "prim": t.prim[rows].astype(np.int8), "origin_idx": t.origin_idx[rows].astype(np.int32),
        "idx": t.idx[rows].astype(np.int32), "lead": t.lead[rows].astype(np.float32), "obs": t.obs[rows],
        "vin": vin, "k1_inj": q_inj["K1"], "k1t_inj": q_inj["K1+t"], "k1t_clean": q_twin["K1+t"],
        "k2_inj": q_inj["K2"],
    }  # fmt: skip


def old_bank_shifts(*, clean, month: str, r: int, nb: dict[int, list[int]], capacity):  # noqa: ANN201
    """Return experiment 647's own and mirrored shifts of experiment 581's pass `r` (for `--validate`)."""
    ms = c650.month_start_idx(month=month)
    fixed, ev = bank.contaminate(clean=clean, month_start_idx=ms, r=r)
    own = np.where(np.isfinite(fixed) & np.isfinite(clean), fixed - clean, 0.0).astype(np.float32)
    mirror = np.zeros_like(own)
    for p, qs in nb.items():
        if qs:
            q = qs[(r + p) % len(qs)]
            mirror[q] -= own[p] * capacity[p] / capacity[q]
    return own, mirror, ev


def main() -> None:
    t0 = time.time()
    S.k600.WORK = S.OUT
    inp = S.k600.Inputs(arm="k1_d5")
    t = inp.t
    print(f"inputs built in {time.time() - t0:.0f}s; {t.n} rows", flush=True)
    lower, upper, consts = S.row_factors(t=t)
    consts.write_csv(OUT / "tail_constants.csv")
    kk_all = S.s.key(t.prim, t.origin_idx, t.idx)
    order = np.argsort(kk_all)
    sorted_keys = kk_all[order]
    gi = inp.gi
    clean = S.c.k578.clean()
    nb_of, capacity, _ = c650.world()
    real_wide_all = inp.mode[t.prim, t.origin_day]
    inject_dir = (S.ROOT / "arms" / "k1_d5" / "work" / "inject") if MINE.validate else c650.INJECT
    worst = 0.0
    for m in MINE.months:
        in_m = t.month == m
        days_m = np.unique(t.idx[in_m] // PER)
        origin_days_m = sorted({int(d) for d in np.unique(t.origin_day[in_m])})
        passes = [0, 1, 2] if MINE.validate else MINE.passes
        for r in passes:
            if MINE.validate:
                own, mirror, ev = old_bank_shifts(
                    clean=clean, month=m, r=r, nb=nb_of, capacity=capacity
                )
                events = [{"primary": p, "onset_idx": ev[p]["onset_idx"], "end_idx": ev[p]["end_idx"]} for p in range(16)]  # fmt: skip
            else:
                own, mirror, evs = c650.shifts_for(clean=clean, month=m, r=r)
                events = [e.row() for e in evs]
                pl.DataFrame(events).write_parquet(OUT / f"events_{m}_r{r:02d}.parquet")
            out = one_pass(inp=inp, t=t, m=m, r=r, own=own, mirror=mirror, events=events, lower=lower,
                           upper=upper, order=order, sorted_keys=sorted_keys, inject_dir=inject_dir,
                           gi=gi, days_m=days_m, origin_days_m=origin_days_m, real_wide_all=real_wide_all)  # fmt: skip
            if out is None:
                continue
            if MINE.validate:
                z = np.load(c647.WORK / "k1_bank" / f"rows_{m}_r{r:02d}.npz")
                key_new = S.s.key(out["prim"].astype(np.int64), out["origin_idx"].astype(np.int64), out["idx"].astype(np.int64))  # fmt: skip
                key_old = S.s.key(z["prim"].astype(np.int64), z["origin_idx"].astype(np.int64), z["idx"].astype(np.int64))  # fmt: skip
                common, i_new, i_old = np.intersect1d(key_new, key_old, return_indices=True)
                d = np.abs(out["k1_inj"][i_new] - z["k1_inj"][i_old]).max()
                share = float((out["k1_inj"][i_new] != z["k1_inj"][i_old]).any(axis=1).mean())
                worst = max(worst, float(d))
                print(f"validate {m} r{r:02d}: {len(common)} common rows of {len(key_new)} new and {len(key_old)} old; max |dK1| {d:.2e}; share of rows differing {share:.4f}", flush=True)  # fmt: skip
            else:
                np.savez_compressed(OUT / f"rows_{m}_r{r:02d}.npz", **out)
                print(f"rows {m} r{r:02d}: {len(out['prim'])}", flush=True)
    print(f"done in {time.time() - t0:.0f}s; worst validate difference {worst:.2e}", flush=True)


if __name__ == "__main__":
    main()
