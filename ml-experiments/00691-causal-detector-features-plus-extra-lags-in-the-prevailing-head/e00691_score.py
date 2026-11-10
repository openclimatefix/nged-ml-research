# ruff: noqa: E501, D103, PLR0915, F841
"""Experiment 691, step 2: score B0, L1, F1, and FL1 as experiment 687 scored its arms (CPU).

This is experiment 687's `e00687_score.py` with the arm list cut to `ARMS` and the FL1 fits read
from experiment 691's folder; the rest is unchanged.

Arms: `K1` (the NRA output), `R0-168`, `R652` (experiment 652's stored R-haz m50 fit, the gate),
and `ARMS`. Equal width scaling is that of experiment 652 (one factor per
arm, month, set, and seed, fitted on all scored rows pooled across strata). Lead bands: `d1` (24
to 48 h), `d2_4` (48 to 120 h), `d5_10` (120 to 240 h).

Writes to `score/`: `agg.parquet` (weighted sums by month, set, arm, seed, stratum, band: CRPS
scaled and unscaled, absolute error of the median, coverage, tail pinball loss, false-alarm
count), `reaction.parquet` (per event, arm, seed: start and end reaction in origin days,
censored), and `cost.csv`.

Usage: python e00687_score.py
"""

import e00578_common as k578  # ty: ignore[unresolved-import]
import e00601_bank as b601  # ty: ignore[unresolved-import]
import e00601_common as k601  # ty: ignore[unresolved-import]
import e00650_common as c650  # ty: ignore[unresolved-import]
import e00650_fit as f650  # ty: ignore[unresolved-import]
import e00650_score as s650  # ty: ignore[unresolved-import]
import e00652_common as c652  # ty: ignore[unresolved-import]
import e00652_fit as f652  # ty: ignore[unresolved-import]
import e00687_fit as f687  # ty: ignore[unresolved-import]
import e00687_score as s687  # ty: ignore[unresolved-import]
import e00691_fit as f691
import numpy as np
import polars as pl

OUT = f691.WORK / "score"
ARMS = ("B0", "L1", "F1", "FL1")
TAU_H = 168.0
BANDS = (("d1", 24.0, 48.0), ("d2_4", 48.0, 120.0), ("d5_10", 120.0, 240.0))
FALSE_ALARM = 0.03
REACT_FRACTION = 0.5
REACT_CAP_DAYS = 14
LEAD = (12.0, 36.0)
FRACTION = 0.9
SETS = ("fam", "held", "real")


band_of = s687.band_of
reaction_rows = s687.reaction_rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    clean = k578.clean()
    cap = k578.sites()["capacity_mw"].to_numpy()
    six = np.array([str(g).startswith("signed-MW") for g in k578.sites()["group"].to_list()])
    lim = FRACTION * np.nanmax(clean[:, : b601.CUTOFF_IDX], axis=1) * cap
    agg, react, cost = [], [], []
    for m in c650.MONTHS:
        src = f650.sources(month=m, bank="new")
        f652.with_state(z=src["bank"], month=m)
        events = {
            (int(r["pass"]), int(r["primary"])): r
            for r in c652.events_table(month=m).iter_rows(named=True)
        }
        for setname in SETS:
            z = src["real" if setname == "real" else "bank"]
            fits = {
                a: [
                    np.load((f691.OUT if a == "FL1" else f687.OUT) / f"{a}_{m}_s{s}.npz")
                    for s in c652.SEEDS
                ]
                for a in ARMS
            }
            fits["R652"] = [np.load(f652.OUT / f"Rhaz_m50_{m}_s{s}.npz") for s in c652.SEEDS]
            idx = fits["B0"][0][f"{setname}_idx"]
            for fs in fits.values():
                for f in fs:
                    assert np.array_equal(f[f"{setname}_idx"], idx), "index sets differ"
            lead, k1, res = z["lead"][idx], z["q"][idx], z["res"][idx]
            assert z["vi"][idx].max() < c650.WINDOW_START_IDX, "a valid time on or after 2026-07-01"
            if setname == "real":
                y = base = z["y"][idx]
                stratum = np.where(z["verified"][idx], "verified", np.where(z["event_free"][idx] & ~z["flagged"][idx], "real_free", "real_other"))  # fmt: skip
            else:
                base = z["y"][idx]
                y = (base + z["shift"][idx]).astype(np.float32)
                stratum = z["stratum"][idx]
            keep = np.isfinite(res) & np.isfinite(y)
            r0 = np.sort(k1 + (res * np.exp(-lead / TAU_H))[:, None], axis=1).astype(np.float32)
            band = band_of(lead)
            free = np.isin(stratum, ["free", "real_free"])
            arms = {"K1": [(0, k1)], "R0-168": [(0, r0)]}
            arms |= {
                a: [(s, np.sort(f[f"{setname}_q"], axis=1)) for s, f in enumerate(fs)]
                for a, fs in fits.items()
            }
            scale_rows = keep & ~np.isin(stratum, ["censored", "real_other"])
            for arm, lst in arms.items():
                for seed, q in lst:
                    _qs, fac = s650.scale_to_nominal(y=y[scale_rows], q=q[scale_rows])
                    qsc = (q[:, 5:6] + fac * (q - q[:, 5:6])).astype(np.float32)
                    p = s650.pin(y=y, q=q)
                    far = (np.abs(q[:, 5] - k1[:, 5]) >= FALSE_ALARM).astype(float)
                    df = pl.DataFrame({
                        "stratum": stratum, "band": band, "crps": 2 * p.mean(1), "crps_s": s650.crps(y=y, q=qsc),
                        "ae_med": np.abs(y - q[:, 5]), "cov90": ((y >= q[:, 1]) & (y <= q[:, 9])).astype(float),
                        "pin01": p[:, 0], "pin05": p[:, 1], "pin95": p[:, 9], "pin99": p[:, 10], "false_alarm": far,
                    }).filter(pl.Series(keep))  # fmt: skip
                    df = pl.concat([df, df.filter(pl.col("stratum").is_in(["far", "middle", "near"])).with_columns(stratum=pl.lit("inforce"))])  # fmt: skip
                    cols = [
                        "crps",
                        "crps_s",
                        "ae_med",
                        "cov90",
                        "pin01",
                        "pin05",
                        "pin95",
                        "pin99",
                        "false_alarm",
                    ]
                    g = df.group_by("stratum", "band").agg(
                        pl.len().alias("n"), *[pl.col(c).sum() for c in cols]
                    )
                    agg.append(g.with_columns(month=pl.lit(m), set=pl.lit(setname), arm=pl.lit(arm), seed=pl.lit(seed), factor=pl.lit(float(fac))))  # fmt: skip
                    if setname == "held" and arm != "K1":
                        react += reaction_rows(
                            z=z,
                            idx=idx,
                            q50_head=q[:, 5],
                            events=events,
                            month=m,
                            arm=arm,
                            seed=seed,
                        )
                    if setname == "held":
                        pr = z["prim"][idx].astype(int)
                        ev = [
                            events.get((int(a), int(b)))
                            for a, b in zip(z["pass_"][idx], pr, strict=True)
                        ]
                        on = np.array([e["onset_idx"] for e in ev])  # ty: ignore[not-subscriptable]
                        en = np.array([e["end_idx"] for e in ev])  # ty: ignore[not-subscriptable]
                        vi = z["vi"][idx].astype(int)
                        okc = six[pr] & keep & (lead >= LEAD[0]) & (lead < LEAD[1])
                        for sname, s0 in (("inside", (vi >= on) & (vi < en)), ("7 days after the end", (vi >= en) & (vi < en + 7 * 48)), ("all rows", np.ones(len(vi), bool))):  # fmt: skip
                            sel = okc & s0
                            if sel.sum() < 100:
                                continue
                            c = k601.avoidable_cost(u=q[sel, 9] * cap[pr[sel]], y=y[sel] * cap[pr[sel]], limit=lim[pr[sel]]).mean() * 48  # fmt: skip
                            cost.append({"month": m, "set": sname, "arm": arm, "seed": seed, "cost": float(c), "n": int(sel.sum())})  # fmt: skip
            # the NRA output is K1 for every arm (heads sit on frozen quantiles): identity check
            assert np.array_equal(k1, z["q"][idx])
            print(m, setname, "scored", flush=True)
    pl.concat(agg).write_parquet(OUT / "agg.parquet")
    pl.DataFrame(react).write_parquet(OUT / "reaction.parquet")
    pl.DataFrame(cost).write_csv(OUT / "cost.csv")


if __name__ == "__main__":
    main()
