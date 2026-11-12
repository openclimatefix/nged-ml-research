# ruff: noqa: ANN001, ANN201, B023, B905, C408, C416, D103, E501, E731, F841, PLR0912, PLR0915, PT018, PTH123, RET504, RUF015, SIM115
"""Audit of experiment 695: re-derive contrasts from raw stored fits with independent scoring code.

Reads only stored files. Writes nothing except stdout (redirected by the caller) and a pickle cache
of the per-row losses under the audit folder. The add-back method (e00695_addback) is imported only to
rebuild systems that have no stored predictions (A1n, the bounds, WT); every score, filter, pairing
and interval below is the auditor's own code.
"""

import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

ROOT = Path(os.environ["DATA_PATH_INTERNAL"]) / "ml-experiments"
E = ROOT / "00695-gas-engine-state-tracker-part-2-cleaned-target-and-climatology-add-back"
OUT = ROOT / "briefs" / "e695_audit"
SHARED = ROOT / "backtest_year_2025-11_to_2026-10"
E677 = ROOT / "00677-building-up-the-candidate-demand-network-one-component-at-a-time"
Q = ["q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99"]
LV = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99])
MONTHS = ["2025-11", "2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06"]
D8 = 26
CAP_D8 = 10.379
GRID0 = np.datetime64("2019-09-01T00:00:00", "us")
rng_seed = 12345


def fits(arm):
    fr = []
    for m in MONTHS:
        for s in (0, 1, 2):
            p = E / "runs" / arm / "arms/k1_5w_sp1/work/fits/P5" / f"{m}_d0_s{s}.parquet"
            fr.append(pl.read_parquet(p).with_columns(seed=pl.lit(s), fold=pl.lit(m)))
    f = pl.concat(fr)
    return f.with_columns(
        time_series_id=pl.col("time_series_id").cast(pl.Int64),
        origin=pl.col("power_fcst_init_time") + pl.duration(hours=3),
    )


def window(f):
    lead = (pl.col("valid_time") - pl.col("origin")).dt.total_minutes() / 60
    return f.filter(
        pl.col("origin") >= pl.datetime(2025, 11, 1, time_zone="UTC"),
        pl.col("origin") < pl.datetime(2026, 7, 1, time_zone="UTC"),
        pl.col("valid_time") < pl.datetime(2026, 7, 1, time_zone="UTC"),
        lead >= 24,
        lead <= 240,
    )


def crps_rows(actual, q):
    q = np.sort(q, axis=1)
    e = actual[:, None] - q
    pin = np.maximum(LV * e, (LV - 1) * e)
    return 2 * pin.mean(axis=1), np.abs(actual - q[:, 5]), q


def score(frame, truth, cap):
    """frame: origin, valid_time, seed, quantiles in MVA. truth: valid_time, power (MVA)."""
    j = frame.join(truth, on="valid_time", how="inner").filter(pl.col("power").is_not_null())
    a = j["power"].to_numpy().astype(float) * 100 / cap
    q = j.select(Q).to_numpy().astype(float) * 100 / cap
    c, m, qs = crps_rows(a, q)
    out = j.select("origin", "valid_time", pl.col("seed").cast(pl.Int64)).with_columns(
        crps=pl.Series(c),
        mae=pl.Series(m),
        med_err=pl.Series(qs[:, 5] - a),
        pit_lo05=pl.Series((a < qs[:, 1]).astype(float)),
        pit_hi95=pl.Series((a > qs[:, 9]).astype(float)),
        pit_lo01=pl.Series((a < qs[:, 0]).astype(float)),
        below_med=pl.Series((a < qs[:, 5]).astype(float)),
        width90=pl.Series(qs[:, 9] - qs[:, 1]),
        month=pl.col("origin").dt.strftime("%Y-%m"),
    )
    return out


def paired(la, lb, metric="crps"):
    k = ["origin", "valid_time", "seed"]
    j = la.select(*k, "month", a=pl.col(metric)).join(
        lb.select(*k, b=pl.col(metric)), on=k, how="full", coalesce=True
    )
    assert j["a"].null_count() == 0 and j["b"].null_count() == 0, "unpaired rows"
    return j.with_columns(d=pl.col("a") - pl.col("b"))


def contrast(la, lb, level, metric="crps", n=20000):
    j = paired(la, lb, metric)
    est = j["d"].mean()
    seeds = j.group_by("seed").agg(pl.col("d").mean()).sort("seed")["d"].to_list()
    # seed-averaged per row, then month blocks
    row = j.group_by("origin", "valid_time", "month").agg(d=pl.col("d").mean())
    mon = row.group_by("month").agg(s=pl.col("d").sum(), n=pl.len()).sort("month")
    s, cnt = mon["s"].to_numpy(), mon["n"].to_numpy()
    monthly = s / cnt
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, len(s), size=(n, len(s)))
    boot = s[idx].sum(1) / cnt[idx].sum(1)
    t = (100 - level) / 2
    lo, hi = np.percentile(boot, [t, 100 - t])
    # seed-and-month version (one seed drawn per resample, as the library does)
    ms = j.group_by("seed", "month").agg(s=pl.col("d").sum(), n=pl.len()).sort("seed", "month")
    ns = ms["seed"].n_unique()
    S = ms["s"].to_numpy().reshape(ns, -1)
    C = ms["n"].to_numpy().reshape(ns, -1)
    sd = rng.integers(0, ns, size=n)
    boot2 = np.take_along_axis(S[sd], idx, 1).sum(1) / np.take_along_axis(C[sd], idx, 1).sum(1)
    lo2, hi2 = np.percentile(boot2, [t, 100 - t])
    tl, th = stats.t.interval(
        level / 100, df=len(monthly) - 1, loc=monthly.mean(), scale=stats.sem(monthly)
    )
    return dict(
        est=est,
        seeds=seeds,
        boot=(lo, hi),
        boot_seed=(lo2, hi2),
        t=(tl, th),
        monthly=monthly.round(3).tolist(),
        sd_month=float(monthly.std(ddof=1)),
        n_rows=row.height,
    )


def fmt(name, r):
    return (
        f"{name}: est {r['est']:+.3f}  boot(seed-mean) [{r['boot'][0]:+.3f},{r['boot'][1]:+.3f}]"
        f"  boot(seed-draw) [{r['boot_seed'][0]:+.3f},{r['boot_seed'][1]:+.3f}]"
        f"  t [{r['t'][0]:+.3f},{r['t'][1]:+.3f}]  seeds {[round(x, 3) for x in r['seeds']]}"
        f"  monthly {r['monthly']} sd {r['sd_month']:.3f} rows {r['n_rows']}"
    )


def d8_truth():
    return (
        pl.read_parquet(SHARED / "ground_truth_primaries.parquet")
        .filter(pl.col("time_series_id") == D8)
        .select("valid_time", "power")
    )


def grid_idx(vt):
    return ((vt.dt.replace_time_zone(None).to_numpy() - GRID0) // np.timedelta64(30, "m")).astype(
        int
    )


def addback_job(args):
    name, arm, states_arm, variant, signed, series, cap = args
    sys.path.insert(0, str(Path(sys.argv[1])))
    import e00695_addback as ab

    f = window(fits(arm).filter(pl.col("time_series_id") == series))
    f = f.with_columns(*[pl.col(q) * cap for q in Q])
    n_grid = int(np.load(E / "stacks/W0/2025-11.npz")["values"].shape[0])
    cal = ab.build_calendar(n_grid=n_grid)
    out = []
    for m in MONTHS:
        part = f.filter(pl.col("origin").dt.strftime("%Y-%m") == m)
        if states_arm == "truth":
            import e00695_engine as en

            g, _ = en.drive_series(n_grid=n_grid)
            bins = en.output_bins(g=g)
            mass = np.zeros((n_grid, bins.max() + 1))
            mass[np.arange(n_grid), bins] = 1
            outputs = en.OUTPUT_BIN_MW * np.arange(mass.shape[1])
        else:
            p = E / "states" / states_arm / f"{m}.npz"
            d = np.load(p if p.exists() else E / "states" / states_arm / "all.npz")
            mass = d["pk"].astype(float)
            outputs = float(d["unit"]) * np.arange(mass.shape[1])
        fc = part.with_columns(power_fcst_init_time=pl.col("origin"))
        r = ab.add_back(
            forecast=fc, mass=mass, outputs=outputs, calendar=cal, variant=variant, signed=signed
        )
        out.append(r.drop("power_fcst_init_time"))
    res = pl.concat(out)
    res.write_parquet(OUT / f"cache_{name.replace('+', 'p')}.parquet")
    return name


def main():
    expdir = sys.argv[1]
    sys.path.insert(0, expdir)
    truth = d8_truth()
    a0 = window(fits("A0"))
    a1raw = window(fits("A1"))
    # --- pairing checks
    keys = ["time_series_id", "origin", "valid_time", "seed"]
    for arm in ("A1", "A1n", "AH", "W0", "W1"):
        o = window(fits(arm)).select(keys).sort(keys)
        print(
            f"pairing {arm} vs A0: keys equal = {o.equals(a0.select(keys).sort(keys))}, rows {o.height}"
        )
    print(
        "origin hours (contract):",
        a0["origin"].dt.hour().unique().to_list(),
        "fold==origin month:",
        (a0["fold"] == a0["origin"].dt.strftime("%Y-%m")).all(),
        "fold==native origin month:",
        (a0["fold"] == a0["power_fcst_init_time"].dt.strftime("%Y-%m")).all(),
    )
    # do A1 and A0 fits differ at other Primaries? (network refits with D8's target changed)
    j = a0.join(a1raw, on=keys, suffix="_1")
    dif = (
        j.with_columns(dq=(pl.col("q50") - pl.col("q50_1")).abs())
        .group_by("time_series_id")
        .agg(pl.col("dq").mean())
        .sort("time_series_id")
    )
    print(
        "mean |q50 A0 - q50 A1raw| (fraction of capacity) per series:",
        dict(zip(dif["time_series_id"], dif["dq"].round(4))),
    )
    # --- stored predictions vs raw fits
    pa0 = pl.read_parquet(E / "predictions/a0-d8/predictions.parquet")
    chk = window(fits("A0").filter(pl.col("time_series_id") == D8)).with_columns(
        *[pl.col(q) * CAP_D8 for q in Q]
    )
    jj = pa0.rename({"power_fcst_init_time": "origin"}).join(
        chk, on=["origin", "valid_time", "seed"], suffix="_r"
    )
    print(
        "a0-d8 stored vs raw fits*capacity: rows",
        jj.height,
        "of",
        pa0.height,
        "max |diff| MVA",
        float(
            np.max(
                np.abs(
                    np.sort(jj.select(Q).to_numpy(), 1)
                    - np.sort(jj.select([q + "_r" for q in Q]).to_numpy(), 1)
                )
            )
        ),
    )
    cap_tab = pl.read_parquet(SHARED / "series_capacity.parquet")
    print(
        "series_capacity D8:",
        cap_tab.filter(pl.col("time_series_id") == D8).to_dicts(),
        "fixed",
        CAP_D8,
    )
    # --- rebuild add-back systems with no stored predictions
    host_id = None
    lab = pl.read_parquet(
        E677 / "scores/protocol_all/rows.parquet", columns=["time_series_id", "label"]
    ).unique()
    label_of = dict(zip(lab["time_series_id"].cast(pl.Int64), lab["label"]))
    host = json.loads((E / "host.json").read_text())["host"]
    host_id = [k for k, v in label_of.items() if v == host][0]
    cap_host = float(cap_tab.filter(pl.col("time_series_id") == host_id)["capacity"][0])
    print("host", host, host_id, "capacity", cap_host)
    jobs = [
        ("A1", "A1", "A1", "climatology", False, D8, CAP_D8),
        ("A1n", "A1n", "A1n", "climatology", True, D8, CAP_D8),
        ("A1+state", "A1", "A1", "state", False, D8, CAP_D8),
        ("AH+state", "AH", "AH", "state", False, D8, CAP_D8),
        ("WT", "A0", "truth", "climatology", False, host_id, cap_host),
        ("WT+truth", "A0", "truth", "truth", False, host_id, cap_host),
    ]
    todo = [jb for jb in jobs if not (OUT / f"cache_{jb[0].replace('+', 'p')}.parquet").exists()]
    if todo:
        with ProcessPoolExecutor(4) as ex:
            for nm in ex.map(addback_job, [(*jb,) for jb in todo]):
                print("rebuilt", nm, flush=True)
    sysf = {}
    for nm in ("A1n", "A1+state", "AH+state", "WT", "WT+truth", "A1"):
        sysf[nm] = pl.read_parquet(OUT / f"cache_{nm.replace('+', 'p')}.parquet")
    # check stored a1-d8 equals rebuilt A1
    pa1 = pl.read_parquet(E / "predictions/a1-d8/predictions.parquet").rename(
        {"power_fcst_init_time": "origin"}
    )
    jj = pa1.join(sysf["A1"], on=["origin", "valid_time", "seed"], suffix="_r")
    print(
        "a1-d8 stored vs rebuilt: rows",
        jj.height,
        "max |diff| MVA",
        float(
            np.max(np.abs(jj.select(Q).to_numpy() - jj.select([q + "_r" for q in Q]).to_numpy()))
        ),
    )
    # --- truth for host: injected reading from stacks/W0
    vals = np.load(E / "stacks/W0/2025-11.npz")["values"].astype(float) * cap_host
    n = len(vals)
    vt = pl.Series(GRID0 + np.arange(n) * np.timedelta64(1800, "s")).dt.replace_time_zone("UTC")
    host_truth = pl.DataFrame(
        {"valid_time": vt, "power": pl.Series(vals, nan_to_null=True)}
    ).filter(pl.col("valid_time") < pl.datetime(2026, 7, 1, time_zone="UTC"))
    # --- losses
    L = {}
    d8 = lambda f: f.filter(pl.col("time_series_id") == D8).with_columns(
        *[pl.col(q) * CAP_D8 for q in Q]
    )
    L["A0"] = score(d8(a0), truth, CAP_D8)
    L["A1raw(no add-back)"] = score(d8(a1raw), truth, CAP_D8)
    L["A1"] = score(pa1, truth, CAP_D8)
    L["AH"] = score(
        pl.read_parquet(E / "predictions/ah-d8/predictions.parquet").rename(
            {"power_fcst_init_time": "origin"}
        ),
        truth,
        CAP_D8,
    )
    for nm in ("A1n", "A1+state", "AH+state"):
        L[nm] = score(sysf[nm], truth, CAP_D8)
    hw = lambda arm: (
        window(fits(arm))
        .filter(pl.col("time_series_id") == host_id)
        .with_columns(*[pl.col(q) * cap_host for q in Q])
    )
    L["W0"] = score(hw("W0"), host_truth, cap_host)
    L["W1"] = score(
        pl.read_parquet(E / "predictions/w1-host/predictions.parquet").rename(
            {"power_fcst_init_time": "origin"}
        ),
        host_truth,
        cap_host,
    )
    for nm in ("WT", "WT+truth"):
        L[nm] = score(sysf[nm], host_truth, cap_host)
    pw0 = pl.read_parquet(E / "predictions/w0-host/predictions.parquet").rename(
        {"power_fcst_init_time": "origin"}
    )
    print(
        "w0 stored crps equals raw-fit crps:",
        abs(score(pw0, host_truth, cap_host)["crps"].mean() - L["W0"]["crps"].mean()) < 1e-9,
    )
    print("\nMEAN CRPS / MAE per system (rows per seed)")
    for k, v in L.items():
        print(
            f"  {k}: CRPS {v['crps'].mean():.3f} MAE {v['mae'].mean():.3f} rows/seed {v.height // 3}"
            f" cov<q01 {v['pit_lo01'].mean():.3f} cov<q05 {v['pit_lo05'].mean():.3f} >q95 {v['pit_hi95'].mean():.3f}"
            f" below median {v['below_med'].mean():.3f} median err {v['med_err'].mean():+.3f} width90 {v['width90'].mean():.2f}"
        )
    F = 100 * (1 - 0.05 / 3)
    print("\nCONTRASTS")
    for nm, a, b, lv in (
        ("C1 A1-A0", "A1", "A0", F),
        ("C4 W1-W0", "W1", "W0", F),
        ("C5 WT-W0", "WT", "W0", F),
        ("MAE A1-A0", "A1", "A0", 95),
        ("A1n-A1", "A1n", "A1", 95),
        ("AH-A1", "AH", "A1", 95),
        ("A1+state-A0", "A1+state", "A0", 95),
        ("AH+state-A0", "AH+state", "A0", 95),
        ("WT+truth-W0", "WT+truth", "W0", 95),
        ("W1-WT", "W1", "WT", 95),
        ("D8 per-Primary row: A1raw-A0", "A1raw(no add-back)", "A0", 95),
    ):
        print(fmt(nm, contrast(L[a], L[b], lv, "mae" if nm.startswith("MAE") else "crps")))
    # C2/C3 by origin month
    for nm, ms in (("C2 engine months", MONTHS[:4]), ("C3 fade months", MONTHS[4:])):
        f = lambda x: x.filter(pl.col("month").is_in(ms))
        print(fmt(nm, contrast(f(L["A1"]), f(L["A0"]), 95)))
    # --- null noise: seed vs seed within A0, and 677 sister arms on the same rows
    print("\nNULL / SISTER WIDTHS (98.3%)")
    for s1, s2 in ((0, 1), (1, 2), (0, 2)):
        a = L["A0"].filter(pl.col("seed") == s1).with_columns(seed=pl.lit(0))
        b = L["A0"].filter(pl.col("seed") == s2).with_columns(seed=pl.lit(0))
        r = contrast(a, b, F)
        print(
            f"  A0 seed{s1}-seed{s2}: est {r['est']:+.3f} boot [{r['boot'][0]:+.3f},{r['boot'][1]:+.3f}] monthly sd {r['sd_month']:.3f}"
        )
    for nm, fn in (("5w", "step5w_with_injected_sp1"), ("5wo", "step5wo_without_injected_sp1")):
        st = pl.read_parquet(E677 / "backtests" / f"{fn}.parquet").filter(
            pl.col("time_series_id") == D8, pl.col("seed").is_in([0, 1, 2])
        )
        st = st.rename({"power_fcst_init_time": "origin"})
        L["677_" + nm] = score(
            window(
                st.with_columns(power_fcst_init_time=pl.col("origin")).with_columns(
                    origin=pl.col("origin")
                )
            )
            if False
            else st.filter(
                pl.col("origin") >= pl.datetime(2025, 11, 1, time_zone="UTC"),
                pl.col("origin") < pl.datetime(2026, 7, 1, time_zone="UTC"),
                pl.col("valid_time") < pl.datetime(2026, 7, 1, time_zone="UTC"),
                ((pl.col("valid_time") - pl.col("origin")).dt.total_minutes() / 60).is_between(
                    24, 240
                ),
            ),
            truth,
            CAP_D8,
        )
    print(
        "  677 5w rows/seed", L["677_5w"].height // 3, "CRPS", round(L["677_5w"]["crps"].mean(), 3)
    )
    print(fmt("  677 5w-5wo (sister, D8)", contrast(L["677_5w"], L["677_5wo"], F)))
    try:
        print(fmt("  A0 - 677 5w (G0 paired)", contrast(L["A0"], L["677_5w"], F)))
    except AssertionError as e:
        print("  G0 paired:", e)
    # --- state-sliced diagnostics with 694 hindsight flags and own climatology weights
    sys.path.insert(0, expdir)
    import e00695_common as cm

    flags = cm.e694_flags().select("valid_time", "flag")
    print(
        "\nBY HINDSIGHT M2 FLAG (valid time): CRPS, MAE, median err (pts), <q05, >q95, below median, width90"
    )
    for k in ("A0", "A1", "A1raw(no add-back)", "AH", "A1n", "A1+state"):
        x = L[k].join(flags, on="valid_time", how="left")
        for fl in (True, False):
            p = x.filter(pl.col("flag") == fl)
            print(
                f"  {k:20s} on={fl!s:5s} rows/seed {p.height // 3:6d} CRPS {p['crps'].mean():.3f} MAE {p['mae'].mean():.3f} "
                f"med_err {p['med_err'].mean():+.3f} <q05 {p['pit_lo05'].mean():.3f} >q95 {p['pit_hi95'].mean():.3f} "
                f"belowmed {p['below_med'].mean():.3f} w90 {p['width90'].mean():.2f}"
            )
    # by month: A1 vs A0 CRPS and hindsight on-share, and A1's climatology P(k>=1)
    import e00695_addback as ab

    n_grid = n
    cal = ab.build_calendar(n_grid=n_grid)
    rows = []
    a1f = d8(a1raw).filter(pl.col("seed") == 0)
    for m in MONTHS:
        part = a1f.filter(pl.col("origin").dt.strftime("%Y-%m") == m).with_columns(
            power_fcst_init_time=pl.col("origin")
        )
        mass = np.load(E / "states/A1" / f"{m}.npz")["pk"].astype(float)
        w = ab.row_weights(forecast=part, mass=mass, calendar=cal, variant="climatology")
        gi = grid_idx(part["valid_time"])
        state_now = mass[gi][:, 1:].sum(1)
        fl = part.select("valid_time").join(flags, on="valid_time", how="left")["flag"].to_numpy()
        rows.append(
            (
                m,
                float(w[:, 1:].sum(1).mean()),
                float(np.nanmean(state_now)),
                float(np.nanmean(fl.astype(float))),
            )
        )
    print(
        "\nMONTH: climatology P(k>=1) used | causal fixed-lag state P(k>=1) at valid time | hindsight flag share"
    )
    for r in rows:
        print(f"  {r[0]}: {r[1]:.3f} | {r[2]:.3f} | {r[3]:.3f}")
    # monthly C1
    c1 = paired(L["A1"], L["A0"]).group_by("month").agg(pl.col("d").mean()).sort("month")
    print("C1 by origin month:", dict(zip(c1["month"], c1["d"].round(3))))
    # sign of cleaned series: share of running rows where cleaned level < k*u (export branch)
    print(
        "\nFOLD BRANCH: era-2 rows with khat>=1, scored months, share where cleaned level < khat*u (export)"
    )
    raw = np.load(E / "stacks/W0/2025-11.npz")  # just for n
    for arm in ("A1", "A1n"):
        for m in ("2025-11", "2026-03"):
            st = np.load(E / "states" / arm / f"{m}.npz")
            stk = np.load(E / "stacks" / arm / f"{m}.npz")["values"].astype(float) * CAP_D8
            kh = st["pk"].argmax(1)
            i0, i1 = (
                int(
                    (np.datetime64("2025-03-24") - np.datetime64("2019-09-01"))
                    / np.timedelta64(30, "m")
                ),
                int(
                    (np.datetime64("2026-07-01") - np.datetime64("2019-09-01"))
                    / np.timedelta64(30, "m")
                ),
            )
            sl = slice(i0, i1)
            run = (kh[sl] >= 1) & np.isfinite(stk[sl])
            lvl = st["lmean"].astype(float)[sl]
            ku = kh[sl] * float(st["unit"])
            print(
                f"  {arm} fold {m}: running rows {run.sum()}, level<k*u share {np.mean(lvl[run] < ku[run]):.3f}, "
                f"mean cleaned on running rows {np.nanmean(stk[sl][run]):.2f} MVA, mean level off rows {np.nanmean(lvl[(kh[sl] == 0)]):.2f}"
            )
    # power
    print("\nPOWER: t half-width at 98.3% from monthly sd of C1; resolvable gain")
    r = contrast(L["A1"], L["A0"], F)
    hwid = stats.t.ppf(1 - (1 - F / 100) / 2, 7) * r["sd_month"] / np.sqrt(8)
    print(
        f"  monthly sd {r['sd_month']:.3f}, t half-width {hwid:.3f}, boot half-width {(r['boot'][1] - r['boot'][0]) / 2:.3f}"
    )
    pickle.dump({k: v for k, v in L.items()}, open(OUT / "losses.pkl", "wb"))
    # --- MC check of the mixture on a sample of A1 rows
    rng = np.random.default_rng(0)
    part = (
        d8(a1raw)
        .filter(pl.col("seed") == 0, pl.col("origin").dt.strftime("%Y-%m") == "2025-12")
        .sample(200, seed=1)
        .with_columns(power_fcst_init_time=pl.col("origin"))
    )
    st = np.load(E / "states/A1/2025-12.npz")
    mass = st["pk"].astype(float)
    outs = float(st["unit"]) * np.arange(mass.shape[1])
    w = ab.row_weights(forecast=part, mass=mass, calendar=cal, variant="climatology")
    cq = np.sort(part.select(Q).to_numpy().astype(float), 1)
    mq = ab.mixture_quantiles(clean_q=cq, outputs=outs, weights=w)
    gaps = []
    levels13 = np.concatenate([[0.001], LV, [0.999]])
    for i in range(len(cq)):
        lo = cq[i, 0] - (cq[i, 1] - cq[i, 0]) / 0.04 * 0.009
        hi = cq[i, -1] + (cq[i, -1] - cq[i, -2]) / 0.04 * 0.009
        pts = np.concatenate([[lo], cq[i], [hi]])
        u = rng.uniform(0.001, 0.999, 200000)
        c = np.interp(u, levels13, pts)
        k = rng.choice(len(outs), size=len(u), p=w[i] / w[i].sum())
        o = outs[k]
        y = np.where(o > 0, np.sqrt((c - o) ** 2 + 0.14**2), c)
        # the CDF has mass 0.002 outside [0.001,0.999] in the method; renormalise levels accordingly
        gaps.append(np.abs(np.quantile(y, (LV - 0.001) / 0.998) - mq[i]).max())
    print(
        f"\nMIXTURE MC CHECK (200 rows): median max |gap| {np.median(gaps):.3f} MVA, 95th pct {np.percentile(gaps, 95):.3f} MVA"
    )


if __name__ == "__main__":
    main()
