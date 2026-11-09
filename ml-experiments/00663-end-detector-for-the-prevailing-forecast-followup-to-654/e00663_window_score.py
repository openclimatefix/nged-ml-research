# ruff: noqa: D103, E501, PLR0912, PLR0915, ANN001, D102, D107, ANN201, ANN202, PT018, F841, PERF401, B007, RUF100
"""Experiment 663: score the frozen candidates (the experiment 654 arm J2 is read from its stored forecasts) on the final-test window, once (CPU).

Reads the frozen forecasts of `final/forecasts/<arm>_s<seed>.parquet` (seed-mean of the
quantiles), the experiment 620 export of K1 for the 2026-07 fold (row keys, observed power, event
flags), and regenerates the manual heuristic from the cleaned power with the same code as
experiment 620. Writes `final/score/window.json`. Origins are those of the experiment 620 export
(2026-07-01 to 2026-10-31); lead 24 to 240 hours; scores in points of capacity; intervals blocked
by origin month.

Scenarios, by the state of the target half-hour at the row's Primary: `all`; `outside` (no event
in force and at least 7 days since the last event ended); `inside`; `after7` (outside, and under
7 days since an event ended). Events are the half-hours on which the hand-built G4 gate of
experiment 620 is on (the export's labelled and verified event columns are empty in the window).
The counts are lower bounds.
"""

import json
import os

import e00583_common as c  # ty: ignore[unresolved-import]
import e00601_common as k601  # ty: ignore[unresolved-import]
import e00620_extend as ext  # ty: ignore[unresolved-import]
import e00620_score as s  # ty: ignore[unresolved-import]
import e00663_common as c663
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

assert os.environ.get("E00620_PHASE") == "predict"
ARMS = ("J2",)  # experiment 654's frozen forecasts, not rescored
F = c663.E654 / "final"
OUTD = c663.WORK / "final"
BOOT_SEED = 663
START_US, STEP_US = c.k578.prep.GRID_START_US, c.k578.prep.STEP_US
EXPORT = next((ML_EXPERIMENTS_DATA_DIR).glob("00620-*/export/K1_2026-07.parquet"))
BOUNDS = [(d - 1) * 24 for d in range(1, 11)]  # lead-day edges 0,24,..


def boot(*, grp, vals, seed=BOOT_SEED, n=2000):
    ug = np.unique(grp)
    sm = np.array([vals[grp == g].sum() for g in ug])
    ct = np.array([(grp == g).sum() for g in ug], float)
    pick = np.random.default_rng(seed).integers(0, len(ug), size=(n, len(ug)))
    d = sm[pick].sum(axis=1) / ct[pick].sum(axis=1)
    return float(vals.mean()), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def main() -> None:
    ext.redirect()
    sub = pl.read_csv(s.SUBSTATIONS)
    cap = dict(zip(sub["time_series_id"].to_list(), sub["capacity"].to_list(), strict=True))
    grp_of = dict(zip(sub["time_series_id"].to_list(), sub["group"].to_list(), strict=True))
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
    key = (prim.astype(np.int64) * 4_000_000 + oidx) * 1024 + (tidx - oidx)
    q: dict[str, np.ndarray] = {
        "K1": np.sort(base.select(s.QCOLS).to_numpy().astype(float) / capv[:, None], axis=1)
    }
    q["Manual heuristic"] = s.manual_heuristic(clean=clean, prim=prim, idx=tidx, lead_h=lead)
    for arm in ARMS:
        parts = [pl.read_parquet(F / "forecasts" / f"{arm}_s{sd}.parquet") for sd in (0, 1)]
        d = parts[0]
        k2 = (
            d["primary"].to_numpy().astype(np.int64) * 4_000_000 + d["origin_idx"].to_numpy()
        ) * 1024 + (d["target_idx"].to_numpy() - d["origin_idx"].to_numpy())
        o = np.argsort(k2)
        pos = np.searchsorted(k2[o], key)
        pos = np.clip(pos, 0, len(k2) - 1)
        hit = k2[o][pos] == key
        print(arm, "matched share", hit.mean(), flush=True)
        for kind in ("nra", "prevailing"):
            m = np.mean(
                [p.select([f"{kind}_{i}" for i in range(11)]).to_numpy() for p in parts], axis=0
            )
            m = np.sort(m, axis=1)
            full = np.full((len(key), 11), np.nan)
            full[hit] = m[o][pos[hit]]
            q[arm if kind == "prevailing" else f"{arm}_nra"] = full
    # experiment 663's arms: final-fold E0 and E0d (`fits_e0/<arm>_final_s<seed>.npz`) and J2d
    # (`fits_j2/J2d_final_s<seed>.npz`); each stores the window rows' prevailing quantiles
    zw = np.load(c663.WORK / "window" / "real_window.npz")
    kw = (zw["prim"].astype(np.int64) * 4_000_000 + zw["oi"]) * 1024 + (zw["vi"] - zw["oi"])
    ow = np.argsort(kw)
    posw = np.clip(np.searchsorted(kw[ow], key), 0, len(kw) - 1)
    hitw = kw[ow][posw] == key
    print("window rows matched", hitw.mean(), flush=True)
    for arm663, folder in (("E0", "fits_e0"), ("E0d", "fits_e0"), ("J2d", "fits_j2")):
        fs = [np.load(c663.WORK / folder / f"{arm663}_final_s{sd}.npz") for sd in (0, 1)]
        for f in fs:
            assert np.array_equal(f["window_idx"], fs[0]["window_idx"])
        idx = fs[0]["window_idx"]
        for kind, fld in (("", "q"), ("_nra", "nra")):
            if kind == "_nra" and arm663 != "J2d":
                m = zw["q"][idx]
            else:
                m = np.mean([f[f"window_{fld}"] for f in fs], axis=0)
            m = np.sort(m, axis=1)
            full_w = np.full((len(kw), 11), np.nan)
            full_w[idx] = m
            full = np.full((len(key), 11), np.nan)
            full[hitw] = full_w[ow][posw[hitw]]
            q[arm663 + kind] = full
    # event state of each (Primary, grid index) from the export's target half-hours the export holds
    # no labelled or verified event in the window (all three flag columns are empty), so the
    # detected state is the hand-built G4 gate
    ev = base["gate_on"].fill_null(False).to_numpy()
    n_idx = clean.shape[1]
    inev = np.zeros((16, n_idx), bool)
    seen = np.zeros((16, n_idx), bool)
    inev[prim[ev], tidx[ev]] = True
    seen[prim, tidx] = True
    since = np.full((16, n_idx), np.inf)
    for p in range(16):
        last = -np.inf
        for t in np.flatnonzero(seen[p]):
            if inev[p, t]:
                last = t
            since[p, t] = (t - last) / 48.0 if np.isfinite(last) else np.inf
    state_in = inev[prim, tidx]
    d_since = since[prim, tidx]
    scen = {
        "all": np.ones(len(y), bool),
        "outside": (~state_in) & (d_since >= 7),
        "inside": state_in,
        "after7": (~state_in) & (d_since < 7),
    }
    ok = np.isfinite(y) & (lead >= 24) & (lead <= 240)
    for n, v in q.items():
        ok &= np.isfinite(v).all(axis=1)
    month = ((base["power_fcst_init_time"].dt.month()).to_numpy()).astype(int)
    day_of_lead = np.minimum((lead // 24).astype(int), 9)  # 1..10 -> lead day 1 = 24-48 h
    peak = np.nanmax(
        clean[
            :,
            : int(
                (np.datetime64("2026-07-01") - np.datetime64("2019-09-01"))
                / np.timedelta64(30, "m")
            ),
        ],
        axis=1,
    )
    out: dict = {"rows_total": len(y), "rows_scored": int(ok.sum())}
    thr_idx = int(
        (np.datetime64("2026-07-01") - np.datetime64("2019-09-01")) / np.timedelta64(30, "m")
    )
    thr_p = np.array([np.nanquantile(clean[p, :thr_idx], 0.9) for p in range(16)])
    for sname, mask in scen.items():
        sel = np.flatnonzero(ok & mask)
        res: dict = {"rows": len(sel), "event_half_hours_primary_pairs": int(mask.sum())}
        if len(sel) < 500:
            out[sname] = res
            continue
        yy, mm = y[sel], month[sel]
        for n, v in q.items():
            vv = v[sel]
            pin = s.pinball(y=yy, q=vv)
            crps = 2 * pin.mean(axis=1) * 100
            r = {
                "crps": boot(grp=mm, vals=crps),
                "nmae": boot(grp=mm, vals=np.abs(yy - vv[:, 5]) * 100),
                "cover90": boot(grp=mm, vals=((vv[:, 1] <= yy) & (yy <= vv[:, 9])).astype(float)),
                "twcrps": boot(grp=mm, vals=s.twcrps(y=yy, q=vv, thr=thr_p[prim[sel]]) * 100),
                "crps_by_lead_day": {
                    int(dd + 1): float(crps[day_of_lead[sel] == dd].mean())
                    for dd in range(1, 10)
                    if (day_of_lead[sel] == dd).any()
                },
            }
            cost = []
            for p in range(16):
                t0 = next((t for t in prim_of if prim_of[t] == p), None)
                if t0 is None or grp_of[t0] != "six MW Primaries":
                    continue
                sp = (
                    (prim[sel] == p)
                    & (lead[sel] >= k601.LEAD_COST[0])
                    & (lead[sel] < k601.LEAD_COST[1])
                )
                if sp.sum() < 100:
                    continue
                cp = float(cap[t0])
                lim = np.full(int(sp.sum()), 0.9 * peak[p] * cp)
                cost.append(
                    float(
                        k601.avoidable_cost(u=vv[sp][:, 9] * cp, y=yy[sp] * cp, limit=lim).mean()
                        * 48
                    )
                )
            r["curtailment_cost_gbp_per_day_f0.9_mean_over_primaries"] = (
                float(np.mean(cost)) if cost else None
            )
            nra_q = q.get(n + "_nra") if not n.endswith("_nra") else None
            if nra_q is not None:
                dep = np.abs(vv[:, 5] - nra_q[sel][:, 5]) >= 0.03
                r["departure_share"] = boot(grp=mm, vals=dep.astype(float))
                r["crps_on_departure_rows"] = float(crps[dep].mean()) if dep.any() else None
                r["nra_crps_on_departure_rows"] = (
                    float((2 * s.pinball(y=yy, q=nra_q[sel]).mean(axis=1) * 100)[dep].mean())
                    if dep.any()
                    else None
                )
            res[n] = r
        out[sname] = res
    (OUTD / "score").mkdir(parents=True, exist_ok=True)
    (OUTD / "score" / "window.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: {"rows": v.get("rows")} for k, v in out.items() if isinstance(v, dict)}))


main()
