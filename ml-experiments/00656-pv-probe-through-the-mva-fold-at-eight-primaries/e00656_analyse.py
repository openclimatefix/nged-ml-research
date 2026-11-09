# ruff: noqa: E501
"""Experiment 656: recoveries, leave-one-Primary-out factors, and the Tier A and Tier B bars.

Reads the probe parquets of `e00656_probe.py`. Recovery is `(C_condition - C_twin) / (L x m)` in MW of
AC capacity per injected MW, with `C_twin` the capacity of the uninjected network of the same cutoff,
seed, site, and reactive-power arm, and `m` the Primary's median sunny-midday reading. Every bar is
printed with its number. Nothing is tuned after a bar fails.

Usage: python e00656_analyse.py [--cutoffs 2026-06 2026-03]
"""

import argparse
import json

import e00621_common as k621  # ty: ignore[unresolved-import]
import e00656_common as k
import numpy as np
import polars as pl

METHODS = ("L0", "L1", "F4", "F1", "FA")
OUT = k.WORK / "analysis"


def load(*, cutoffs: tuple[str, ...]) -> pl.DataFrame:
    """Load every probe parquet of the given cutoffs with parsed condition fields."""
    frames = [pl.read_parquet(p) for c in cutoffs for p in sorted((k.WORK / "probes" / c).glob("*.parquet"))]
    return pl.concat(frames, how="diagonal_relaxed")


def injected_mw(*, cutoff: str, site: str, level: float) -> float:
    """Return the injected AC capacity (MW) of a Primary at a level."""
    meta = json.loads((k.WORK / "world" / f"{cutoff}.json").read_text())["median_midday"]
    idx = k.index_of(sites=(site,))[site]
    return level * float(meta[str(idx)]) * float(k621.effective_capacity_mw()[idx])


def recoveries(*, res: pl.DataFrame) -> pl.DataFrame:
    """Return one row per (cutoff, condition, seed, site, method) with the capacity difference and the recovery."""
    rows = []
    twins = {
        (r["cutoff"], r["condition"], r["seed"], r["site"]): r for r in res.filter(pl.col("condition").str.ends_with("_twin")).iter_rows(named=True)
    }
    for r in res.iter_rows(named=True):
        cond = k.parse(name=r["condition"])
        if cond.kind == "twin":
            continue
        twin = twins.get((r["cutoff"], f"{cond.arm}_twin", r["seed"], r["site"]))
        if twin is None:
            continue
        level = k.LEVELS[k.PLACEBO_LEVEL] if cond.kind in {"clim", "boxcar", "cloud"} else (k.LEVELS["l6"] if cond.kind == "sunload" else cond.level)
        inj = injected_mw(cutoff=r["cutoff"], site=r["site"], level=level)
        for meth in METHODS:
            d = r[meth] - twin[meth]
            rec = {"cutoff": r["cutoff"], "condition": r["condition"], "arm": cond.arm, "kind": cond.kind, "level": cond.level, "rot": cond.rotation, "seed": r["seed"], "site": r["site"], "method": meth, "diff_mw": d, "inj_mw": inj, "rec": d / inj}
            boot, tboot = r.get(f"{meth}_boot"), twin.get(f"{meth}_boot")
            if boot is not None and tboot is not None:
                rec["diff_boot"] = (np.array(boot) - np.array(tboot)).tolist()
            rows.append(rec)
    return pl.DataFrame(rows, infer_schema_length=None)


def med(*, rec: pl.DataFrame, condition: str | list[str], method: str, sites: tuple[str, ...]) -> dict[str, float]:
    """Return the median recovery per site over cutoffs, seeds, and the named conditions."""
    conds = [condition] if isinstance(condition, str) else condition
    sub = rec.filter(pl.col("condition").is_in(conds) & (pl.col("method") == method) & pl.col("site").is_in(sites))
    g = sub.group_by("site").agg(r=pl.col("rec").median())
    return dict(zip(g["site"].to_list(), g["r"].to_list(), strict=True))


def loo_factor(*, rec: pl.DataFrame, conditions: list[str], method: str) -> dict[str, float]:
    """Return each of the six Primaries' leave-one-out factor: the median recovery at the other five."""
    per = med(rec=rec, condition=conditions, method=method, sites=k.SIX)
    return {s: float(np.median([v for t, v in per.items() if t != s])) for s in per}


def count_in(*, values: dict[str, float], lo: float, hi: float) -> int:
    """Count sites with a value in [lo, hi]."""
    return sum(lo <= v <= hi for v in values.values())


def coverage(*, rec: pl.DataFrame, conditions: list[str], method: str, sites: tuple[str, ...], factors: dict[str, float]) -> float:
    """Return the share of injections whose calibrated, widened 80% interval covers the injected capacity."""
    sub = rec.filter(pl.col("condition").is_in(conditions) & (pl.col("method") == method) & pl.col("site").is_in(sites) & pl.col("diff_boot").is_not_null())
    f_lo, f_hi = min(factors.values()), max(factors.values())
    hit = 0
    n = 0
    for r in sub.iter_rows(named=True):
        lo, hi = np.percentile(r["diff_boot"], [10, 90])
        n += 1
        hit += (lo / (f_hi if lo > 0 else f_lo)) <= r["inj_mw"] <= (hi / (f_lo if hi > 0 else f_hi))
    return hit / max(n, 1)


def bars(*, rec: pl.DataFrame, res: pl.DataFrame, cutoffs: tuple[str, ...]) -> dict[str, dict]:
    """Evaluate every Tier A and Tier B bar and return {bar: {pass, detail}}."""
    out: dict[str, dict] = {}
    six, nat = k.SIX, k.NATIVE
    pf05, pf1, pf2, pf6 = ["pf_fleet_l05_r0"], ["pf_fleet_l1_r0", "pf_fleet_l1_r1"], ["pf_fleet_l2_r0", "pf_fleet_l2_r1"], ["pf_fleet_l6_r0", "pf_fleet_l6_r1"]
    # A1
    a1 = {}
    for meth in ("L0", "F4"):
        pf = med(rec=rec, condition=pf05, method=meth, sites=six)
        sg = med(rec=rec, condition="sig_fleet_l1_r0", method=meth, sites=six)
        a1[meth] = {s: pf[s] / sg[s] for s in pf if s in sg and abs(sg[s]) > 1e-6}
    ok1 = all(count_in(values=v, lo=0.9, hi=1.1) >= 5 for v in a1.values())
    out["A1"] = {"pass": ok1, "detail": {m: {s: round(x, 2) for s, x in v.items()} for m, v in a1.items()}, "note": "denominator is the signed arm at L1 (no signed L0.5 condition is run)"}
    # A2
    pf_six = med(rec=rec, condition=pf05, method="L0", sites=six)
    pf_nat = med(rec=rec, condition=pf05, method="L0", sites=nat)
    centre = float(np.median(list(pf_six.values())))
    n_ok = sum(abs(v - centre) <= 0.12 for v in pf_nat.values())
    out["A2"] = {"pass": n_ok >= 4, "detail": {"six_median": round(centre, 3), "native": {s: round(v, 3) for s, v in pf_nat.items()}}}
    # A3
    a3 = {}
    for kind in ("clim", "boxcar"):
        for grp, sites, need in (("six", six, 5), ("native", nat, 4)):
            good = 0
            for s in sites:
                real = rec.filter((pl.col("condition") == "pf_fleet_l05_r0") & (pl.col("method") == "L0") & (pl.col("site") == s))["diff_mw"].median()
                p95 = rec.filter((pl.col("condition") == f"pf_{kind}") & (pl.col("method") == "L0") & (pl.col("site") == s))["diff_mw"].quantile(0.95)
                good += p95 < 0.25 * real
            a3[f"{kind}_{grp}"] = (good, need)
    out["A3"] = {"pass": all(g >= n for g, n in a3.values()), "detail": a3}
    # A4
    f05 = loo_factor(rec=rec, conditions=pf05, method="L0")
    out["A4"] = {"pass": count_in(values=f05, lo=0.55, hi=0.95) >= 5 and (max(f05.values()) - min(f05.values())) < 0.15, "detail": {s: round(v, 3) for s, v in f05.items()}}
    # A5
    cov_a = coverage(rec=rec, conditions=pf05, method="L0", sites=(*six, *nat), factors=f05)
    out["A5"] = {"pass": cov_a >= 0.6, "detail": round(cov_a, 3)}
    # Tier B
    fa: dict[str, dict[str, float]] = {}
    for arm, lv_name, conds in (("pf", "l1", pf1), ("pf", "l2", pf2), ("pf", "l6", pf6), ("floor", "l1", ["floor_fleet_l1_r0"]), ("floor", "l2", ["floor_fleet_l2_r0"])):
        raw = med(rec=rec, condition=conds, method="FA", sites=six)
        fac = loo_factor(rec=rec, conditions=conds, method="FA")
        fa[f"{arm}_{lv_name}"] = {s: raw[s] / fac[s] for s in raw if abs(fac.get(s, 0)) > 1e-6}
    ok_b1 = all(count_in(values=fa[key], lo=0.75, hi=1.25) >= 4 for key in ("pf_l1", "pf_l2", "floor_l1", "floor_l2"))
    out["B1"] = {"pass": ok_b1, "detail": {key: {s: round(v, 2) for s, v in d.items()} for key, d in fa.items() if key != "pf_l6"}}
    out["B2"] = {"pass": count_in(values=fa["pf_l6"], lo=0.75, hi=1.25) >= 4, "detail": {s: round(v, 2) for s, v in fa["pf_l6"].items()}}
    l1_cal = {}
    raw = med(rec=rec, condition=pf2, method="L1", sites=six)
    fac = loo_factor(rec=rec, conditions=pf2, method="L1")
    l1_cal = {s: raw[s] / fac[s] for s in raw if abs(fac.get(s, 0)) > 1e-6 and np.isfinite(raw[s])}
    m_fa = float(np.median([abs(v - 1) for v in fa["pf_l2"].values()]))
    m_l1 = float(np.median([abs(v - 1) for v in l1_cal.values()])) if l1_cal else float("nan")
    out["B3"] = {"pass": bool(m_l1 - m_fa >= 0.10), "detail": {"FA_median_abs_error": round(m_fa, 3), "L1_median_abs_error": round(m_l1, 3)}}
    # B4
    sun = rec.filter((pl.col("condition") == "pf_sunload") & (pl.col("method") == "FA") & pl.col("site").is_in(six))
    real6 = rec.filter(pl.col("condition").is_in(pf6) & (pl.col("method") == "FA") & pl.col("site").is_in(six))
    n_b4 = 0
    for s in six:
        a = sun.filter(pl.col("site") == s)["diff_mw"].median()
        b = real6.filter(pl.col("site") == s)["diff_mw"].median()
        n_b4 += a < 0.25 * b
    sun_res = res.filter((pl.col("condition") == "pf_sunload") & pl.col("site").is_in(six))
    flags = [f for r in sun_res.iter_rows(named=True) for f in r["FA_cross"]]
    cross_share = float(np.mean(flags)) if flags else float("nan")
    out["B4"] = {"pass": bool(n_b4 >= 5 and cross_share < 0.10), "detail": {"sites_under_0.25": int(n_b4), "crossing_flag_share": round(cross_share, 3)}, "note": "point-estimate crossing flags stand in for the bootstrap crossing posterior"}
    # B5
    cov_b = {}
    for key, conds in (("l1", pf1), ("l2", pf2)):
        fac = loo_factor(rec=rec, conditions=conds, method="FA")
        cov_b[key] = round(coverage(rec=rec, conditions=conds, method="FA", sites=six, factors=fac), 3)
    out["B5"] = {"pass": all(v >= 0.6 for v in cov_b.values()), "detail": cov_b}
    # B6
    bal = {}
    for key, level, conds in (("l1", 1.0, pf1), ("l2", 2.0, pf2)):
        tp = tn = fp = fn = 0
        for r in res.filter(pl.col("condition").is_in(conds) & pl.col("site").is_in(six)).iter_rows(named=True):
            inj = injected_mw(cutoff=r["cutoff"], site=r["site"], level=level)
            truth = np.array(r["cell_p_mw"]) < inj * np.array(r["cell_umax"])
            pred = np.array(r["FA_cross"])
            tp += int((truth & pred).sum())
            tn += int((~truth & ~pred).sum())
            fp += int((~truth & pred).sum())
            fn += int((truth & ~pred).sum())
        bal[key] = round(0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1)), 3)
    fp05 = []
    for r in res.filter((pl.col("condition") == "pf_fleet_l05_r0") & pl.col("site").is_in(six)).iter_rows(named=True):
        inj = injected_mw(cutoff=r["cutoff"], site=r["site"], level=0.5)
        truth = np.array(r["cell_p_mw"]) < inj * np.array(r["cell_umax"])
        fp05.append(float((~truth & np.array(r["FA_cross"])).mean()))
    fp_share = float(np.mean(fp05)) if fp05 else float("nan")
    out["B6"] = {"pass": bool(all(v >= 0.8 for v in bal.values()) and fp_share <= 0.10), "detail": {"balanced_accuracy": bal, "false_crossing_share_l05": round(fp_share, 3)}}
    return out


def main() -> None:
    """Print every bar and write the recovery table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoffs", nargs="+", default=list(k.CUTOFFS))
    args = parser.parse_args()
    cutoffs = tuple(args.cutoffs)
    res = load(cutoffs=cutoffs)
    rec = recoveries(res=res)
    OUT.mkdir(parents=True, exist_ok=True)
    rec.drop("diff_boot", strict=False).write_csv(OUT / f"recoveries_{'_'.join(cutoffs)}.csv")
    result = bars(rec=rec, res=res, cutoffs=cutoffs)
    for name, v in result.items():
        print(f"{name}: {'PASS' if v['pass'] else 'FAIL'}  {v['detail']}" + (f"  [{v['note']}]" if "note" in v else ""))
    (OUT / f"bars_{'_'.join(cutoffs)}.json").write_text(json.dumps(result, default=str, indent=1))


if __name__ == "__main__":
    main()
