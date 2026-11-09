# ruff: noqa: ANN001, D103, E501, PLR0912, PLR0915, ANN201, C901
"""Experiment 663, step 5: score every arm by origin state and lead day (CPU).

Arms: `K1`, `R0-168`, `Bhaz` (experiment 652's no-fit references, recomputed), the E0 arms of
`fits_e0/`, the stored J2 of experiment 654 (`J2`), and the J2 detector arms of `fits_j2/`.
Truth is the observed power plus the injected shift (bank rows) or the observed power (real rows).
The NRA output of a detector arm is the base arm's NRA output; it is scored against the unshifted
base on bank rows.

Equal width scaling: one factor per arm, month, set, and seed, fitted on all the month's scored
rows pooled across strata (experiment 652's rule). Per row: CRPS (scaled and unscaled), the
absolute error of the median, q05-q95 and q01-q99 coverage, the pinball losses at 0.01, 0.05,
0.95, and 0.99, and the false-alarm flag (the median of the prevailing output departs from the
median of the NRA output by 3 points of capacity or more) with the CRPS of the prevailing and of the
NRA output on the flagged rows.

Output `score/agg.parquet`: weighted sums by month, set, arm, seed, stratum, band, and lead day.

Usage: python e00663_score.py [--months ...]
"""

import argparse

import e00650_common as c650  # ty: ignore[unresolved-import]
import e00650_fit as f650  # ty: ignore[unresolved-import]
import e00650_score as s650  # ty: ignore[unresolved-import]
import e00652_fit as f652  # ty: ignore[unresolved-import]
import e00652_score as s652  # ty: ignore[unresolved-import]
import e00663_common as c663
import numpy as np
import polars as pl

SEEDS = (0, 1)
DEP = 0.03
OUT = c663.WORK / "score"
E0_ARMS = ("E0", "E0d", "E0d_rw0", "E0d_rw1", "E0d_rw2", "E0d_noneigh")
J2_ARMS = ("J2d", "J2d_rw0", "J2d_rw1", "J2d_rw2", "J2d_noneigh")


def j2_rows(*, month: str, seed: int, setname: str) -> dict[str, np.ndarray]:
    parts = [np.load(c663.E654 / "fits" / f"J2_{month}_d{d}_s{seed}.npz") for d in (0, 14)]
    idx = np.concatenate([p[f"{setname}_idx"] for p in parts])
    o = np.argsort(idx)
    return {
        "idx": idx[o],
        "q": np.concatenate([p[f"{setname}_q"] for p in parts])[o],
        "nra": np.concatenate([p[f"{setname}_nra"] for p in parts])[o],
        "p": np.concatenate([p[f"{setname}_p"] for p in parts])[o],
    }


def arm_rows(*, arm: str, month: str, seed: int, setname: str, z: dict) -> dict[str, np.ndarray]:
    if arm == "J2":
        return j2_rows(month=month, seed=seed, setname=setname)
    if arm in E0_ARMS:
        f = np.load(c663.WORK / "fits_e0" / f"{arm}_{month}_s{seed}.npz")
        idx = f[f"{setname}_idx"]
        return {"idx": idx, "q": f[f"{setname}_q"], "nra": z["q"][idx], "p": f[f"{setname}_p"]}
    f = np.load(c663.WORK / "fits_j2" / f"{arm}_{month}_s{seed}.npz")
    return {k: f[f"{setname}_{k}"] for k in ("idx", "q", "nra", "p")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="+", default=list(c663.MONTHS))
    ap.add_argument("--arms", nargs="+", default=["J2", *E0_ARMS, *J2_ARMS])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    agg = []
    for m in a.months:
        src = f650.sources(month=m, bank="new")
        f652.with_state(z=src["bank"], month=m)
        train_months = [x for x in c650.MONTHS if x != m]
        curve = s652.survival_curve(months=train_months, mixture="m50")
        for setname in ("fam", "held", "real"):
            z = src["real" if setname == "real" else "bank"]
            ref = np.load(c663.WORK / "fits_e0" / f"E0_{m}_s0.npz")
            idx = ref[f"{setname}_idx"]
            lead, k1, res = z["lead"][idx], z["q"][idx], z["res"][idx]
            assert z["vi"][idx].max() < c650.WINDOW_START_IDX
            if setname == "real":
                y = base = z["y"][idx]
                stratum = np.where(z["verified"][idx], "verified", np.where(z["event_free"][idx] & ~z["flagged"][idx], "real_free", "real_other"))
                band = np.full(len(idx), "real")
                far14 = np.zeros(len(idx), bool)
                ended24 = np.zeros(len(idx), bool)
            else:
                base = z["y"][idx]
                y = (base + z["shift"][idx]).astype(np.float32)
                stratum = z["stratum"][idx]
                band = np.where(setname == "held", "held", np.where(z["short"][idx], "short", "long"))
                oi = z["oi"][idx].astype(np.int64)
                far14 = (stratum == "free") & ((oi < z["onset"][idx] - 14 * 48) | (oi >= z["end"][idx] + 14 * 48))
                ended24 = (stratum == "just_ended") & (oi - z["end"][idx] <= 48)
            keep = np.isfinite(res) & np.isfinite(y)
            age = z["age"][z["org"][idx]].astype(np.float32)
            lead_day = np.clip((lead // 24).astype(int), 1, 10)
            arms: dict[str, list[tuple[int, np.ndarray, np.ndarray, np.ndarray | None]]] = {
                "K1": [(0, k1, k1, None)],
                "R0-168": [(0, np.sort(k1 + (res * np.exp(-lead / 168.0))[:, None], axis=1).astype(np.float32), k1, None)],
                "Bhaz": [(0, np.sort(k1 + (res * s652.cond_survival(curve=curve, age_days=age, lead_h=lead))[:, None], axis=1).astype(np.float32), k1, None)],
            }
            for arm in a.arms:
                lst = []
                for s in SEEDS:
                    try:
                        r = arm_rows(arm=arm, month=m, seed=s, setname=setname, z=z)
                    except FileNotFoundError:
                        break
                    assert np.array_equal(r["idx"], idx), (arm, m, setname)
                    lst.append((s, np.sort(r["q"], axis=1), np.sort(r["nra"], axis=1), r["p"]))
                if lst:
                    arms[arm] = lst
            scale_rows = keep & ~np.isin(stratum, ["censored", "real_other"])
            for arm, lst in arms.items():
                for seed, q, nra, _p in lst:
                    _qs, fac = s650.scale_to_nominal(y=y[scale_rows], q=q[scale_rows])
                    qsc = (q[:, 5:6] + fac * (q - q[:, 5:6])).astype(np.float32)
                    p = s650.pin(y=y, q=q)
                    crps, crps_s = 2 * p.mean(1), s650.crps(y=y, q=qsc)
                    crps_nra = 2 * s650.pin(y=base if setname != "real" else y, q=nra).mean(1)
                    dep = np.abs(q[:, 5] - nra[:, 5]) >= DEP
                    df = pl.DataFrame({
                        "stratum": stratum, "band": band, "lead_day": lead_day, "far14": far14, "ended24": ended24,
                        "crps": crps, "crps_s": crps_s, "nmae": np.abs(y - q[:, 5]),
                        "cov90": ((y >= q[:, 1]) & (y <= q[:, 9])).astype(float),
                        "cov98": ((y >= q[:, 0]) & (y <= q[:, 10])).astype(float),
                        "pin01": p[:, 0], "pin05": p[:, 1], "pin95": p[:, 9], "pin99": p[:, 10],
                        "dep": dep.astype(float), "dep_crps": np.where(dep, crps, 0.0), "dep_crps_nra": np.where(dep, crps_nra, 0.0),
                        "crps_nra": crps_nra,
                    }).filter(pl.Series(keep))
                    df = pl.concat([df, df.filter(pl.col("stratum").is_in(["far", "middle", "near"])).with_columns(stratum=pl.lit("inforce"))])
                    df = pl.concat([df, df.filter(pl.col("far14")).with_columns(stratum=pl.lit("free_far14")), df.filter(pl.col("ended24")).with_columns(stratum=pl.lit("ended24")), df.with_columns(stratum=pl.lit("all"))])
                    cols = [c for c in df.columns if c not in ("stratum", "band", "lead_day", "far14", "ended24")]
                    g = df.group_by("stratum", "band", "lead_day").agg(pl.len().alias("n"), *[pl.col(c).sum() for c in cols])
                    agg.append(g.with_columns(month=pl.lit(m), set=pl.lit(setname), arm=pl.lit(arm), seed=pl.lit(seed), factor=pl.lit(float(fac))))
            print(m, setname, "scored", flush=True)
    pl.concat(agg).write_parquet(OUT / "agg.parquet")


if __name__ == "__main__":
    main()
