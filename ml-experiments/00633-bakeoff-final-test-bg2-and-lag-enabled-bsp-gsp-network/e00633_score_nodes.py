# ruff: noqa: E501, D103, PLR0915
"""Experiment 633, Part 2: score the lag-enabled network, BG2, the manual heuristic, and the climatology at the bulk and grid supply points, once (CPU).

Scoring follows experiment 620's `e00620_score.py` (copied into `e00633_score.py`): scores are shares of capacity, in percent
where printed; the interval of every mean is a day-block bootstrap over forecast-origin days; every contrast prints its
Bonferroni count. All systems are scored on the rows on which every system has a forecast and the series has a finite,
unflagged reading. Only the series with readings in the window are scored (the count is printed).

Usage: python e00633_score_nodes.py OUT_DIR
"""

import json
import sys
from pathlib import Path

import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_fit_demand as fd  # ty: ignore[unresolved-import]
import e00633_common as k
import e00633_score as s
import numpy as np
import polars as pl

NODE_SITES = ("D13", "D17", "D1", "D6")
GROUP_OF = {"D13": "BSP", "D17": "BSP", "D1": "GSP", "D6": "GSP"}
GROUP_DIR = {"BSP": "bsp", "GSP": "gsp"}
TAG = "J_LG_B1"
SEEDS = (0, 1, 2, 3, 4)
SYSTEMS = ("Lag-enabled network", "BG2", "Manual heuristic", "Climatology")
SETS = {"days 1 to 3": (0, 72), "days 4 to 10": (72, 240)}
KEYS = ["time_series_id", "power_fcst_init_time", "valid_time"]


def network(*, group: str) -> pl.DataFrame:
    files = [k.ROOT / "work" / GROUP_DIR[group] / "seed_preds" / TAG / f"2026-07_s{seed}.parquet" for seed in SEEDS]
    assert all(f.exists() for f in files), files
    frame = pl.concat([pl.read_parquet(f) for f in files]).group_by(KEYS).agg(*[pl.col(c).mean() for c in s.QCOLS]).sort(KEYS)
    values = np.sort(frame.select(s.QCOLS).to_numpy(), axis=1)
    return frame.with_columns(*[pl.Series(c, values[:, i]) for i, c in enumerate(s.QCOLS)])


def bg2(*, site: str, tid: int) -> pl.DataFrame:
    f = pl.read_parquet(k.ROOT / "preds" / "bg2_single_fit" / f"{site}.parquet").select(*KEYS, *s.QCOLS)
    arr = np.sort(f.select(s.QCOLS).to_numpy().astype(np.float64), axis=1)
    median = arr[:, [5]]
    arr = median + k.SPREAD_FACTOR * (arr - median)
    return f.select(KEYS).with_columns(*[pl.Series(c, arr[:, i]) for i, c in enumerate(s.QCOLS)])


def main(out: Path) -> None:
    assert k.PHASE == "predict", "set E00633_PHASE=predict"
    out.mkdir(parents=True, exist_ok=True)
    k.redirect()
    series, arrays = fd._series_and_arrays()
    rows = {r["site"]: r for r in series.iter_rows(named=True) if r["site"] in NODE_SITES}
    cutoff_idx = int(common.grid_index(times=k.CUTOFF))
    window_end = int(common.grid_index(times=k.GRID_END))
    readings = {site: int(np.isfinite(arrays[site].power[cutoff_idx : window_end + 1]).sum()) for site in NODE_SITES}
    scored_sites = [site for site in NODE_SITES if readings[site] > 0]
    (out / "series_with_readings.json").write_text(json.dumps({"readings_in_window": readings, "scored": scored_sites}, indent=1))
    print("half-hours with a reading in the window:", readings, "scored:", scored_sites, flush=True)
    clean = np.stack([arrays[site].power for site in scored_sites]).astype(np.float32)
    parts = {}
    for group in ("BSP", "GSP"):
        net = network(group=group)
        for site in scored_sites:
            if GROUP_OF[site] != group:
                continue
            tid = int(rows[site]["time_series_id"])
            n = net.filter(pl.col("time_series_id") == tid)
            b = bg2(site=site, tid=tid)
            both = n.join(b, on=KEYS, how="left", suffix="_bg2", validate="1:1")
            parts[site] = both
    frames = pl.concat(list(parts.values()))
    base = frames.select(KEYS)
    prim_of = {int(rows[site]["time_series_id"]): i for i, site in enumerate(scored_sites)}
    tid = base["time_series_id"].to_numpy()
    prim = np.array([prim_of[int(t)] for t in tid])
    site_of = np.array([scored_sites[p] for p in prim])
    idx = ((base["valid_time"].dt.epoch("us") - s.START_US) // s.STEP_US).to_numpy().astype(np.int64)
    oidx = ((base["power_fcst_init_time"].dt.epoch("us") - s.START_US) // s.STEP_US).to_numpy().astype(np.int64)
    lead = (idx - oidx) / 2.0
    y = clean[prim, idx].astype(np.float64)
    q = {
        "Lag-enabled network": np.sort(frames.select(s.QCOLS).to_numpy().astype(np.float64), axis=1),
        "BG2": np.sort(frames.select([f"{c}_bg2" for c in s.QCOLS]).to_numpy().astype(np.float64), axis=1),
        "Manual heuristic": s.manual_heuristic(clean=clean, prim=prim, idx=idx, lead_h=lead),
        "Climatology": s.climatology(clean=clean, prim=prim, idx=idx, cutoff_idx=cutoff_idx),
    }
    ok = np.isfinite(y)
    dropped = {"no observed reading": int((~ok).sum())}
    for n in SYSTEMS:
        miss = ok & ~np.isfinite(q[n]).all(axis=1)
        dropped[f"{n}: no forecast"] = int(miss.sum())
        ok &= ~miss
    print("rows", len(y), "common scored rows", int(ok.sum()), dropped, flush=True)
    pl.DataFrame({"what": list(dropped), "rows": list(dropped.values())}).write_csv(out / "dropped_rows.csv")
    sel = np.flatnonzero(ok & (lead < 240))
    y, lead, prim, site_of, oidx, idx = y[sel], lead[sel], prim[sel], site_of[sel], oidx[sel], idx[sel]
    q = {n: v[sel] for n, v in q.items()}
    group = np.array([GROUP_OF[x] for x in site_of])
    day = oidx // 48
    thr_p = np.array([np.nanquantile(clean[p, :cutoff_idx], 0.9) for p in range(len(scored_sites))])
    thr = thr_p[prim]
    stats = {}
    for n in SYSTEMS:
        pin = s.pinball(y=y, q=q[n])
        stats[n] = {
            "crps": 2.0 * pin.mean(axis=1) * 100,
            "mae": np.abs(y - q[n][:, 5]) * 100,
            "twcrps": s.twcrps(y=y, q=q[n], thr=thr) * 100,
            "cover90": ((q[n][:, 1] <= y) & (y <= q[n][:, 9])).astype(float),
            "cover98": ((q[n][:, 0] <= y) & (y <= q[n][:, 10])).astype(float),
            "ge05": (y >= q[n][:, 1]).astype(float),
            "le95": (y <= q[n][:, 9]).astype(float),
            "pin01": pin[:, 0] * 100, "pin05": pin[:, 1] * 100, "pin95": pin[:, 9] * 100, "pin99": pin[:, 10] * 100,
            "width90": (q[n][:, 9] - q[n][:, 1]) * 100,
        }  # fmt: skip
    slices = {"all nodes": np.ones(len(y), bool)}
    for g in ("BSP", "GSP"):
        slices[g] = group == g
    for site in scored_sites:
        slices[site] = site_of == site
    rows_out = []
    for sl, m_ in slices.items():
        for set_name, (lo, hi) in {"days 1 to 10": (0, 240), **SETS}.items():
            mm = m_ & (lead >= lo) & (lead < hi)
            if not mm.any():
                continue
            for n in SYSTEMS:
                for metric, v in stats[n].items():
                    mean, lo_, hi_, _ = s.day_boot(day=day[mm], vals=v[mm], seed=633)
                    rows_out.append({"slice": sl, "set": set_name, "system": n, "measure": metric, "mean": mean, "lo": lo_, "hi": hi_, "rows": int(mm.sum())})
    pl.DataFrame(rows_out).write_csv(out / "headline.csv")
    band = np.digitize(lead, [24, 48, 72, 96, 120, 144, 168, 192, 216], right=False)
    rows_b = []
    for sl, m_ in slices.items():
        for b in range(10):
            mm = m_ & (band == b)
            if mm.any():
                for n in SYSTEMS:
                    rows_b.append({"slice": sl, "day": b + 1, "system": n, "rows": int(mm.sum()), **{key: float(v[mm].mean()) for key, v in stats[n].items()}})
    pl.DataFrame(rows_b).write_csv(out / "by_day.csv")
    pairs = [("Lag-enabled network", "BG2"), ("Lag-enabled network", "Manual heuristic"), ("Lag-enabled network", "Climatology"), ("BG2", "Manual heuristic"), ("BG2", "Climatology")]
    metrics = ("crps", "mae", "twcrps")
    contrast_slices = [(sl, set_name) for sl in ("BSP", "GSP") for set_name in SETS]
    n_tests = len(pairs) * len(metrics) * len(contrast_slices)
    alpha = 0.05 / n_tests
    rows_c = []
    for sl, set_name in contrast_slices:
        lo, hi = SETS[set_name]
        mm = slices[sl] & (lead >= lo) & (lead < hi)
        if not mm.any():
            continue
        for a_, b_ in pairs:
            for metric in metrics:
                mean, lo95, hi95, draws = s.day_boot(day=day[mm], vals=stats[a_][metric][mm] - stats[b_][metric][mm], seed=633)
                blo, bhi = s.interval(draws, alpha)
                rows_c.append({"slice": sl, "set": set_name, "a": a_, "b": b_, "measure": metric, "diff_a_minus_b": mean, "lo95": lo95, "hi95": hi95, "lo_bonferroni": blo, "hi_bonferroni": bhi, "n_tests": n_tests, "excludes_zero_bonferroni": bool(blo > 0 or bhi < 0)})
    pl.DataFrame(rows_c).write_csv(out / "contrasts.csv")
    print("Bonferroni count:", n_tests, flush=True)
    np.savez_compressed(out / "rows.npz", site=site_of, lead=lead, day=day, idx=idx, y=y, **{f"crps_{n}": stats[n]["crps"] for n in SYSTEMS}, **{f"q50_{n}": q[n][:, 5] for n in SYSTEMS})
    print("done", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
