# ruff: noqa: D103, E501, ANN001, ANN201
"""Experiment 647, stage 3: E3's NRA output against K1 (CPU).

E3's NRA branch (and the E3 placebo's) is compared with K1's raw per-seed forecasts of experiment 613
(the same two K1 seeds, ungated, no tail constants), seed by seed and as the two-seed mean, on:

- N2: event-free, unflagged real rows: CRPS proxy and MAE in points of capacity;
- N1: injected-bank rows of the 15 and 30 day shifts from day 4 of the shift to its end: q05-q95
  coverage of the observed power, the median's follow and the q05/q95 safe-side follow (the mean move
  of the quantile caused by the injected lags, in the direction of the shift, as a share of the shift
  depth; the clean-lag twin is the same network's forecast on the real lags of the same row), and the
  safe-side share;
- real days: on the flagged days and the verified D5/D16 pair, MAE, and the share of K1's miss that
  E3's median closes (mean of sign(observed - K1 median) x (E3 median - K1 median) over the mean
  absolute miss).

Usage: python e00647_enc_nra.py MONTH VARIANT
"""

import sys

import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00510_common as c510  # ty: ignore[unresolved-import]
import e00581_bank as bank  # ty: ignore[unresolved-import]
import e00647_common as c647  # ty: ignore[unresolved-import]
import e00647_data as dat  # ty: ignore[unresolved-import]
import e00647_fit as fit  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

month, variant = sys.argv[1], sys.argv[2]
STORED = ML_EXPERIMENTS_DATA_DIR / "00613-nra-roster-contrasts-with-intervals-five-seeds/arms/k1_d5/work"
COLS = ["q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99"]
step, t0 = c510.prep422.STEP_US, c510.prep422.GRID_START_US
tbl = prep.primaries().select("primary", "time_series_id").with_columns(pl.col("time_series_id").cast(pl.Int32))
PER = 48


def stored(*, kind: str, z: dict, idx: np.ndarray, seed: int) -> np.ndarray:
    """K1's raw stored forecast of `seed` at the head-data rows `idx`."""
    base = pl.DataFrame({
        "primary": z["prim"][idx].astype(np.int64), "o": t0 + z["oi"][idx].astype(np.int64) * step, "v": t0 + z["vi"][idx].astype(np.int64) * step,
        "row": np.arange(len(idx)), "r": z["pass_"][idx].astype(np.int64) if kind == "bank" else np.zeros(len(idx), np.int64),
    }).with_columns(pl.col("o").cast(pl.Datetime("us", "UTC")), pl.col("v").cast(pl.Datetime("us", "UTC"))).join(tbl, on="primary", how="left")
    out = np.full((len(idx), 11), np.nan, np.float32)
    keys = ["time_series_id", "o", "v"]
    for r in sorted(base["r"].unique().to_list()):
        sub = base.filter(pl.col("r") == r)
        st = pl.concat([
            pl.read_parquet(STORED / ("inject" if kind == "bank" else "fits/P5") / (f"{month}_r{r:02d}_d{dd}_s{seed}.parquet" if kind == "bank" else f"{month}_d{dd}_s{seed}.parquet"))
            for dd in (0, 14)
        ]).rename({"power_fcst_init_time": "o", "valid_time": "v"})
        j = sub.join(st, on=keys, how="inner")
        out[j["row"].to_numpy()] = j.select(COLS).to_numpy()
    return out


def crps(*, y, q):
    return fit.metrics(y=y, q=q)["crps"]


def main() -> None:
    d = fit.load(month=month)
    zr, zb = d["real"], d["bank"]
    ir, ib = np.flatnonzero(zr["usable"]), np.flatnonzero(zb["usable"])
    rows = np.load(fit.data_dir() / "fits" / f"{{arm}}nrarows_{month}_{variant}.npz".replace("{arm}", "E3"))
    prow = np.load(fit.data_dir() / "fits" / f"E3pnrarows_{month}_{variant}.npz")
    seeds = (0, 1)
    k1 = {kind: {s: stored(kind=kind, z=z, idx=i, seed=s) for s in seeds} for kind, z, i in (("real", zr, ir), ("bank", zb, ib))}
    for kind in k1:
        assert all(np.isfinite(k1[kind][s]).all() for s in seeds), f"{kind}: stored K1 rows missing"
    systems = {"K1": {kind: np.mean([k1[kind][s] for s in seeds], axis=0) for kind in k1}}
    for name, src in (("E3", rows), ("E3p", prow)):
        systems[name] = {kind: np.mean([src[f"{kind}_{s}"] for s in seeds], axis=0) for kind in k1}
    for s in seeds:
        systems[f"K1 seed {s}"] = {kind: k1[kind][s] for kind in k1}
        systems[f"E3 seed {s}"] = {kind: rows[f"{kind}_{s}"] for kind in k1}
    out: list[dict] = []
    # N2 and real-day sets
    y = zr["y"][ir].astype(np.float32)
    ef = (zr["event_free"][ir] & ~zr["flagged"][ir]) & np.isfinite(zr["res"][ir]) & np.isfinite(zr["res_star"][ir])
    sets = {"N2 event-free, unflagged": ef, "flagged days": zr["flagged"][ir], "verified D5/D16 pair": zr["verified"][ir]}
    for sn, m in sets.items():
        for name, sysq in systems.items():
            q = sysq["real"]
            miss_k1 = y - systems["K1"]["real"][:, 5]
            out.append({
                "set": sn, "system": name, "rows": int(m.sum()), "crps_proxy": float(crps(y=y[m], q=q[m]).mean()), "mae": float((np.abs(y[m] - q[m, 5]) * 100).mean()),
                "cover90": float(((y[m] >= q[m, 1]) & (y[m] <= q[m, 9])).mean()),
                "share_of_K1_miss_closed": float((np.sign(miss_k1[m]) * (q[m, 5] - systems["K1"]["real"][m, 5])).mean() / np.abs(miss_k1[m]).mean()) if m.any() else float("nan"),
            })
    # N1: injected bank
    prim, oi, vi, ps = zb["prim"][ib].astype(np.int64), zb["oi"][ib].astype(np.int64), zb["vi"][ib].astype(np.int64), zb["pass_"][ib].astype(np.int64)
    yb = zb["y"][ib].astype(np.float32)
    ms = dat.month_start_idx(month=month)
    end = ms + bank.END_DAY * PER
    ev = np.array([bank.event(primary=int(p), r=int(r)) for p, r in zip(prim, ps, strict=True)])
    signed, length, sgn = ev[:, 0], ev[:, 1].astype(int), ev[:, 2]
    on = end - length * PER
    in_set = (vi >= on + 4 * PER) & (vi < end) & np.isin(length, (15, 30))
    key = lambda p, o, v: (p * 10_000_000 + o) * 10_000_000 + v  # noqa: E731
    rk = key(zr["prim"][ir].astype(np.int64), zr["oi"][ir].astype(np.int64), zr["vi"][ir].astype(np.int64))
    order = np.argsort(rk)
    pos = np.searchsorted(rk[order], key(prim, oi, vi))
    pos = np.clip(pos, 0, len(rk) - 1)
    has_twin = rk[order][pos] == key(prim, oi, vi)
    twin_ix = order[pos]
    m = in_set & has_twin
    for name, sysq in systems.items():
        qi, qt = sysq["bank"], sysq["real"][twin_ix]
        sg = sgn.astype(np.float32)
        dep = np.abs(signed).astype(np.float32)
        side = np.where(sg > 0, yb >= qi[:, 1], yb <= qi[:, 9])
        out.append({
            "set": "N1 bank 15 and 30 d, day 4 to end", "system": name, "rows": int(m.sum()),
            "cover90": float(((yb[m] >= qi[m, 1]) & (yb[m] <= qi[m, 9])).mean()), "safe_side_share": float(side[m].mean()),
            "follow_q50": float((((qi[m, 5] - qt[m, 5]) * sg[m]) / dep[m]).mean()),
            "follow_safe_edge": float((((np.where(sg[m] > 0, qi[m, 1] - qt[m, 1], qi[m, 9] - qt[m, 9])) * sg[m]) / dep[m]).mean()),
            "crps_proxy": float(crps(y=yb[m], q=qi[m]).mean()), "mae": float((np.abs(yb[m] - qi[m, 5]) * 100).mean()),
        })
    tab = pl.DataFrame(out, infer_schema_length=None)
    path = c647.WORK / "score" / f"enc_nra_{month}_{variant}.csv"
    path.parent.mkdir(exist_ok=True)
    tab.write_csv(path)
    with pl.Config(tbl_rows=80, tbl_cols=12, fmt_str_lengths=40):
        print(tab)


main()
