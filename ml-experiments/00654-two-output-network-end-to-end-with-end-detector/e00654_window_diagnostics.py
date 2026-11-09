# ruff: noqa: D103, E501, PLR0915, D102, D107, ANN201, ANN202, PT018, F841, PERF401, B007, RUF100
"""Experiment 654: diagnostics on the already-scored window (CPU, reads no new window data).

Answers: how long are the gate-on spells; which lags does the manual heuristic use and how often
is a lag half-hour itself gate-on; what do the prevailing and NRA outputs of J2 do on gate-on
rows by whether the observed power really departs from K1's median (false alarms).
"""

import json
import os

import e00583_common as c  # ty: ignore[unresolved-import]
import e00620_extend as ext  # ty: ignore[unresolved-import]
import e00620_score as s  # ty: ignore[unresolved-import]
import e00654_common as c654  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

assert os.environ.get("E00620_PHASE") == "predict"
F = c654.WORK / "final"
START_US, STEP_US = c.k578.prep.GRID_START_US, c.k578.prep.STEP_US
EXPORT = next(ML_EXPERIMENTS_DATA_DIR.glob("00620-*/export/K1_2026-07.parquet"))


def main() -> None:
    ext.redirect()
    sub = pl.read_csv(s.SUBSTATIONS)
    cap = dict(zip(sub["time_series_id"].to_list(), sub["capacity"].to_list(), strict=True))
    sm = c.k578.sites()
    prim_of = dict(zip(sm["time_series_id"].to_list(), sm["primary"].to_list(), strict=True))
    clean = c.k578.clean()
    base = pl.read_parquet(EXPORT)
    tid = base["time_series_id"].to_numpy()
    prim = np.array([prim_of[int(t)] for t in tid])
    capv = np.array([cap[int(t)] for t in tid], float)
    oidx = (
        ((base["power_fcst_init_time"].dt.epoch("us") - START_US) // STEP_US)
        .to_numpy()
        .astype(np.int64)
    )
    tidx = ((base["valid_time"].dt.epoch("us") - START_US) // STEP_US).to_numpy().astype(np.int64)
    lead = base["lead_hours"].to_numpy().astype(float)
    y = base["observed_power"].to_numpy().astype(float) / capv
    gate = base["gate_on"].fill_null(False).to_numpy()
    k1 = np.sort(base.select(s.QCOLS).to_numpy().astype(float) / capv[:, None], axis=1)
    key = (prim.astype(np.int64) * 4_000_000 + oidx) * 1024 + (tidx - oidx)
    d = pl.read_parquet(F / "forecasts" / "J2_s0.parquet")
    parts = [pl.read_parquet(F / "forecasts" / f"J2_s{sd}.parquet") for sd in (0, 1)]
    k2 = (
        d["primary"].to_numpy().astype(np.int64) * 4_000_000 + d["origin_idx"].to_numpy()
    ) * 1024 + (d["target_idx"].to_numpy() - d["origin_idx"].to_numpy())
    o = np.argsort(k2)
    pos = np.clip(np.searchsorted(k2[o], key), 0, len(k2) - 1)
    nra = np.sort(
        np.mean([p.select([f"nra_{i}" for i in range(11)]).to_numpy() for p in parts], axis=0),
        axis=1,
    )[o][pos]
    prv = np.sort(
        np.mean(
            [p.select([f"prevailing_{i}" for i in range(11)]).to_numpy() for p in parts], axis=0
        ),
        axis=1,
    )[o][pos]
    surv = np.mean([p["survival_probability"].to_numpy() for p in parts], axis=0)[o][pos]
    man = s.manual_heuristic(clean=clean, prim=prim, idx=tidx, lead_h=lead)
    ok = np.isfinite(y) & (lead >= 24) & (lead <= 240) & np.isfinite(man).all(axis=1)
    out: dict = {}
    # 1. gate-on spells per Primary on the target grid
    n = clean.shape[1]
    on = np.zeros((16, n), bool)
    seen = np.zeros((16, n), bool)
    on[prim[gate], tidx[gate]] = True
    seen[prim, tidx] = True
    lens = []
    for p in range(16):
        run = 0
        for t in np.flatnonzero(seen[p]):
            if on[p, t]:
                run += 1
            elif run:
                lens.append(run / 48.0)
                run = 0
        if run:
            lens.append(run / 48.0)
    lens = np.array(lens)
    out["spells"] = {
        "n": len(lens),
        "quantiles_days_5_25_50_75_95": [
            float(x) for x in np.quantile(lens, [0.05, 0.25, 0.5, 0.75, 0.95])
        ],
        "share_under_1_day": float((lens < 1).mean()),
        "share_over_14_days": float((lens > 14).mean()),
        "max_days": float(lens.max()),
    }
    # 2. manual heuristic members that are themselves gate-on, on gate-on rows, by lag
    sel = np.flatnonzero(ok & gate)
    share = {}
    for h in s.LAGS_H:
        j = tidx[sel] - 2 * h
        valid = (lead[sel] < h) & (j >= 0) & (j < n)
        valid2 = valid.copy()
        valid2[valid] = seen[prim[sel][valid], j[valid]]
        share[str(h)] = {
            "member_exists": float(valid.mean()),
            "member_gate_on_given_known": float(on[prim[sel][valid2], j[valid2]].mean())
            if valid2.any()
            else None,
        }
    out["manual_lag_gate_on_share"] = share
    # 3. departure and the carried depth
    dep = np.abs(y - k1[:, 5])
    depth = prv[:, 5] - nra[:, 5]
    for name, m in (("gate_on", gate), ("gate_off", ~gate)):
        r = np.flatnonzero(ok & m)
        out[name] = {
            "rows": len(r),
            "mean_abs_prevailing_minus_nra_median_pts": float(np.abs(depth[r]).mean() * 100),
            "mean_survival": float(surv[r].mean()),
            "crps_prevailing": float(2 * s.pinball(y=y[r], q=prv[r]).mean() * 100),
            "crps_nra": float(2 * s.pinball(y=y[r], q=nra[r]).mean() * 100),
            "crps_manual": float(2 * s.pinball(y=y[r], q=man[r]).mean() * 100),
            "crps_k1": float(2 * s.pinball(y=y[r], q=k1[r]).mean() * 100),
        }
        for lab, mm in (("departs_ge_3pts", dep[r] >= 0.03), ("departs_lt_3pts", dep[r] < 0.03)):
            rr = r[mm]
            if len(rr) > 100:
                out[name][lab] = {
                    "rows": len(rr),
                    "crps_prevailing": float(2 * s.pinball(y=y[rr], q=prv[rr]).mean() * 100),
                    "crps_nra": float(2 * s.pinball(y=y[rr], q=nra[rr]).mean() * 100),
                    "crps_manual": float(2 * s.pinball(y=y[rr], q=man[rr]).mean() * 100),
                    "mean_abs_depth_pts": float(np.abs(depth[rr]).mean() * 100),
                }
    # by lead day on gate-on rows
    for dd in (1, 3, 5, 9):
        r = np.flatnonzero(ok & gate & ((lead // 24).astype(int) == dd))
        if len(r):
            out.setdefault("gate_on_by_lead_day", {})[dd + 1] = {
                "prevailing": float(2 * s.pinball(y=y[r], q=prv[r]).mean() * 100),
                "nra": float(2 * s.pinball(y=y[r], q=nra[r]).mean() * 100),
                "manual": float(2 * s.pinball(y=y[r], q=man[r]).mean() * 100),
            }
    (F / "score" / "diagnostics.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


main()
