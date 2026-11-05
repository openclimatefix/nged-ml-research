# ruff: noqa: E501, PLR0915, PLR0912, D205
"""Experiment 582: tail coverage of every arm on experiment 581's lag-only injected event bank.

Experiment 581 forecast v3+D5 with lags contaminated by a constant shift (the lag inputs only,
never the targets) in 24 passes: Primary `p` in pass `r` gets depth in {0.15, 0.30, 0.50} of
capacity and length in {3, 7, 15, 30} days, ending on day 14 of the fold month, with a positive
sign in passes 0 to 11 and a negative sign in passes 12 to 23 (every Primary meets every
combination with both signs). Its stored forecasts are read here, read-only. The uninjected power
is the NRA (the observed power is unchanged), and the uninjected v3+D5 forecast is the reference
of the tail follow ratio.

Arms built from the injected v3+D5 forecast: v3+D5, R (the recalibration factors of the fold
month, fitted on earlier months without injection), the union band with the lags-masked twin, the
50/50 mixture with the lag-free D5 network and the per-lead mixture, each with and without the
factors. Arms that do not read the lags (lags masked, week lag masked, lag-free D5) are the
stored forecasts. Gates are not scored: the observed power is unchanged under a lag-only
injection, so the gate sees no gap. Every second half-hour of valid time is used (the other half
is dropped to save time). Writes `bank_event_phase.csv` and `bank_summary.csv` to the work
folder.
"""

import e00582_arms as a  # ty: ignore[unresolved-import]
import e00582_common as k  # ty: ignore[unresolved-import]
import e00582_score as sc  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

INJECT = ML_EXPERIMENTS_DATA_DIR / "00581-robust-multiweek-lag-encoder-x3/arms/v3d5/work/inject"
BANK = next(
    (ML_EXPERIMENTS_DATA_DIR / "00581-robust-multiweek-lag-encoder-x3").glob(
        "bank_events_v3d5_*.parquet"
    )
)
IQ = a.IQ
BANK_ARMS = (
    "v3+D5",
    "R",
    "Union",
    "Mix50",
    "Mix50-R",
    "MixPL",
    "MixPL-R",
    "Masked",
    "Week-masked",
    "Lag-free D5",
)
STRIDE = 2


def key(prim: np.ndarray, origin_idx: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Return an integer key of (Primary, origin, valid time)."""
    return (prim.astype(np.int64) * 200_000 + origin_idx) * 200_000 + idx


def load_injected(
    *,
    t: a.Table,
    month: str,
    r: int,
    sel_month: np.ndarray,
    order: np.ndarray,
    sorted_keys: np.ndarray,
) -> np.ndarray:
    """Return the seed-mean injected quantiles of pass `r`, aligned to the table's rows of
    `month` (NaN where absent).
    """
    sm = a.k578.sites()
    prim_of = dict(zip(sm["time_series_id"].to_list(), sm["primary"].to_list(), strict=True))
    parts = []
    for d in ("d0", "d14"):
        seeds = []
        for s in (0, 1):
            f = INJECT / f"{month}_r{r:02d}_{d}_s{s}.parquet"
            if not f.exists():
                continue
            seeds.append(pl.read_parquet(f, columns=[*a.KEYS, *a.QCOLS]))
        if seeds:
            parts.append(pl.concat(seeds).group_by(a.KEYS).agg([pl.col(q).mean() for q in a.QCOLS]))
    df = pl.concat(parts).with_columns(
        prim=pl.col("time_series_id").replace_strict(prim_of, return_dtype=pl.Int64),
        idx=(
            (pl.col("valid_time").dt.epoch("us") - a.k578.prep.GRID_START_US) // a.k578.prep.STEP_US
        ).cast(pl.Int64),
        oidx=(
            (pl.col("power_fcst_init_time").dt.epoch("us") - a.k578.prep.GRID_START_US)
            // a.k578.prep.STEP_US
        ).cast(pl.Int64),
    )
    kk = key(df["prim"].to_numpy(), df["oidx"].to_numpy(), df["idx"].to_numpy())
    pos = np.searchsorted(sorted_keys, kk)
    pos = np.clip(pos, 0, len(sorted_keys) - 1)
    ok = sorted_keys[pos] == kk
    out = np.full((t.n, 11), np.nan, np.float32)
    q = df.select(a.QCOLS).to_numpy().astype(np.float32)
    out[order[pos[ok]]] = np.sort(q[ok], axis=1)
    return out


def main() -> None:
    """Score the injected bank and write the event and summary tables."""
    t = a.Table()
    gates = a.run_gates(t=t)
    _arms, common, _ = sc.build_arms(t=t, gates=gates)
    stores = sc.STORES
    bank = pl.read_parquet(BANK)
    kk_all = key(t.prim, t.origin_idx, t.idx)
    order = np.argsort(kk_all)
    sorted_keys = kk_all[order]
    base_sel = common & t.free & (t.idx % STRIDE == 0)
    months = [
        m for m in sorted(bank["month"].unique().to_list()) if (common & (t.month == m)).any()
    ]
    print("bank months scored", months, "rows", int(base_sel.sum()))
    rows: list[dict] = []
    for m in months:
        for r in range(24):
            vin = load_injected(
                t=t, month=m, r=r, sel_month=t.month == m, order=order, sorted_keys=sorted_keys
            )
            ev = bank.filter((pl.col("month") == m) & (pl.col("pass") == r))
            for e in ev.iter_rows(named=True):
                p = e["primary"]
                grp = str(t.group[p])
                if grp == "D15 and D18":
                    continue
                sgn = float(e["sign"])
                depth = float(e["depth"])
                for phase in sc.PHASES:
                    lo, hi = sc.window(onset=e["onset_idx"], end=e["end_idx"], phase=phase)
                    for band in (0, 1):
                        sel = (
                            base_sel
                            & (t.month == m)
                            & (t.prim == p)
                            & (t.idx >= lo)
                            & (t.idx < hi)
                            & (t.band == band)
                            & np.isfinite(vin[:, 0])
                        )
                        n = int(sel.sum())
                        if n < 60:
                            continue
                        idx = np.flatnonzero(sel)
                        y = t.obs[idx]
                        V0 = t.arms["V"][idx]
                        Vi = vin[idx]
                        qs: dict[str, np.ndarray] = {
                            "v3+D5": Vi,
                            "Masked": t.arms["M"][idx],
                            "Week-masked": t.arms["W"][idx],
                            "Lag-free D5": t.arms["E"][idx],
                        }
                        s_r = stores["R"].get((m, grp, band))
                        if s_r is not None:
                            qs["R"] = a.scale_quantiles(q=Vi, s=s_r)
                        qs["Union"] = a.union_band(lag=Vi, masked=t.arms["M"][idx])
                        E = t.arms["E"][idx]
                        if np.isfinite(E).all():
                            w50 = np.full(n, 0.5, np.float32)
                            mix50 = a.mixture(a=Vi, b=E, w=w50)
                            wpl = np.full(n, stores["MixPL_w"].get((m, band), 0.5), np.float32)
                            mixpl = a.mixture(a=Vi, b=E, w=wpl)
                            qs["Mix50"], qs["MixPL"] = mix50, mixpl
                            for nm, q in (("Mix50", mix50), ("MixPL", mixpl)):
                                s_m = stores[f"{nm}-R"].get((m, grp, band))
                                if s_m is not None:
                                    qs[f"{nm}-R"] = a.scale_quantiles(q=q, s=s_m)
                        for name, q in qs.items():
                            rec = {
                                "month": m,
                                "pass": r,
                                "primary": p,
                                "group": grp,
                                "sign": int(sgn),
                                "depth": depth,
                                "length_days": e["length_days"],
                                "phase": phase,
                                "band": sc.BANDS[band],
                                "arm": name,
                                "n_rows": n,
                            }
                            rec["cover90"] = float(
                                np.mean((q[:, IQ[0.05]] <= y) & (y <= q[:, IQ[0.95]]))
                            )
                            rec["cover98"] = float(
                                np.mean((q[:, IQ[0.01]] <= y) & (y <= q[:, IQ[0.99]]))
                            )
                            # the forecast follows the lag shift: for sign +1 it moves up, so the NRA lies below it (lower tail is the NRA side)
                            if sgn > 0:
                                rec["nra_side_cover95"] = float(np.mean(y >= q[:, IQ[0.05]]))
                                rec["nra_side_cover99"] = float(np.mean(y >= q[:, IQ[0.01]]))
                                side = IQ[0.05]
                            else:
                                rec["nra_side_cover95"] = float(np.mean(y <= q[:, IQ[0.95]]))
                                rec["nra_side_cover99"] = float(np.mean(y <= q[:, IQ[0.99]]))
                                side = IQ[0.95]
                            rec["follow_q50"] = float(
                                np.mean((q[:, IQ[0.5]] - V0[:, IQ[0.5]]) * sgn) / depth
                            )
                            rec["follow_safety"] = float(
                                np.mean((q[:, side] - V0[:, side]) * sgn) / depth
                            )
                            rows.append(rec)
        print("done", m, flush=True)
    tab = pl.DataFrame(rows)
    tab.write_csv(k.WORK / "bank_event_phase.csv")
    summ = []
    metrics = [
        "cover90",
        "cover98",
        "nra_side_cover95",
        "nra_side_cover99",
        "follow_q50",
        "follow_safety",
    ]
    for (grp, depth, length, phase, band, arm), g in tab.group_by(
        "group", "depth", "length_days", "phase", "band", "arm"
    ):
        rec = {
            "group": grp,
            "depth": depth,
            "length_days": length,
            "phase": phase,
            "band": band,
            "arm": arm,
            "n_events": g.height,
        }
        for mname in metrics:
            mean, lo, hi = sc._boot_mean(vals=g[mname].to_numpy())
            rec[mname], rec[f"{mname}_lo"], rec[f"{mname}_hi"] = mean, lo, hi
            loo = [
                float(np.nanmean(g.filter(pl.col("primary") != pp)[mname].to_numpy()))
                for pp in g["primary"].unique().to_list()
                if g.filter(pl.col("primary") != pp).height
            ]
            rec[f"{mname}_loo_min"], rec[f"{mname}_loo_max"] = (
                (min(loo), max(loo)) if loo else (np.nan, np.nan)
            )
        summ.append(rec)
    pl.DataFrame(summ).write_csv(k.WORK / "bank_summary.csv")
    print(tab.height, "rows")


if __name__ == "__main__":
    main()
