# ruff: noqa: E501, ANN001, D103, PLR0912, PLR0915, D102, D107, ANN201, ANN202, PT018, F841, PERF401, B007, RUF100
"""Experiment 654, step 2: score every arm by origin state and lead day (CPU).

Reuses experiment 652's scorer parts (`survival_curve`, `cond_survival`, the strata of
`row_state`, equal width scaling of experiment 650). Arms:

- prevailing forecasts, scored against the observed power (bank rows: base plus the injected
  shift): `K1`, `R0-168`, `Bhaz`, `E0` (experiment 652's R-haz, mixture m50, seed mean not taken:
  both seeds scored), `J1`, `J2`, and `C0` (its NRA output, the matched-compute control);
- NRA forecasts, scored against the unshifted base on bank rows and the observed power on real
  rows: `K1_nra`, `C0_nra`, `J1_nra`, `J2_nra`.

Strata add three to experiment 652's: `all` (every stratum), `ended24` (the shift ended in the 24
hours before the origin), and `free14` (event-free and more than 14 days from the event). The
four scenarios of the ml-experiment skill are `all`, `free` (outside events and more than 7 days
after one), `inforce` plus `onset_ahead` (inside events), and `just_ended` (7 days after an event
ends).

Output (data folder `score/`): `agg.parquet` (weighted sums, with the absolute error of the
median as `mae`), `reliability.parquet`, `dropped.csv`.

Usage: python e00654_score.py [--arms J1 J2 C0 ...] [--months ...]
"""

import argparse

import e00650_common as c650  # ty: ignore[unresolved-import]
import e00650_fit as f650  # ty: ignore[unresolved-import]
import e00650_score as s650  # ty: ignore[unresolved-import]
import e00652_common as c652  # ty: ignore[unresolved-import]
import e00652_fit as f652  # ty: ignore[unresolved-import]
import e00652_score as sc652  # ty: ignore[unresolved-import]
import e00654_common as c654  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

OUT = c654.WORK / "score"
FITS = c654.WORK / "fits"
SEEDS = (0, 1)
MIX = "m50"
NEW_ARMS = ("C0", "J1", "J2")
COLS = ["crps", "crps_s", "cov90", "pin01", "pin05", "pin95", "pin99", "cov98", "mae"]


def load_arm(*, arm: str, month: str, seed: int) -> dict[str, np.ndarray]:
    """Return the arm's rows of the month, both cuts joined, as {set_idx, set_nra, set_q, set_p}."""
    parts = [np.load(FITS / f"{arm}_{month}_d{d}_s{seed}.npz") for d in (0, 14)]
    out = {}
    for s in ("real", "held", "fam"):
        idx = np.concatenate([p[f"{s}_idx"] for p in parts])
        o = np.argsort(idx)
        out[f"{s}_idx"] = idx[o]
        for k in ("nra", "q", "p"):
            out[f"{s}_{k}"] = np.concatenate([p[f"{s}_{k}"] for p in parts])[o]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(NEW_ARMS))
    ap.add_argument("--months", nargs="+", default=list(c650.MONTHS))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    agg, rel, dropped = [], [], []
    for m in a.months:
        src = f650.sources(month=m, bank="new")
        f652.with_state(z=src["bank"], month=m)
        train_months = [x for x in c650.MONTHS if x != m]
        curve = sc652.survival_curve(months=train_months, mixture=MIX)
        mine = {(arm, s): load_arm(arm=arm, month=m, seed=s) for arm in a.arms for s in SEEDS}
        e0 = {s: np.load(f652.OUT / f"Rhaz_{MIX}_{m}_s{s}.npz") for s in SEEDS}
        for setname in ("fam", "held", "real"):
            z = src["real" if setname == "real" else "bank"]
            idx = e0[0][f"{setname}_idx"]
            for d in [*mine.values(), *e0.values()]:
                idx = np.intersect1d(idx, d[f"{setname}_idx"])

            def take(d, key, idx=idx, setname=setname):
                have = d[f"{setname}_idx"]
                return d[f"{setname}_{key}"][np.searchsorted(have, idx)]

            lead, k1, res = z["lead"][idx], z["q"][idx], z["res"][idx]
            assert z["vi"][idx].max() < c650.WINDOW_START_IDX
            if setname == "real":
                y = base = z["y"][idx]
                stratum = np.where(
                    z["verified"][idx],
                    "verified",
                    np.where(z["event_free"][idx] & ~z["flagged"][idx], "real_free", "real_other"),
                )
                short = np.ones(len(idx), bool)
                label = in_force = np.zeros(len(idx), bool)
            else:
                base = z["y"][idx]
                y = (base + z["shift"][idx]).astype(np.float32)
                stratum, short = z["stratum"][idx].astype("<U12"), z["short"][idx]
                label, in_force = (
                    z["still_on"][idx] & ~z["censored"][idx],
                    z["in_force"][idx] & ~z["censored"][idx],
                )
                oi, on, en = z["oi"][idx].astype(np.int64), z["onset"][idx], z["end"][idx]
                stratum = np.where((stratum == "just_ended") & (oi - en <= 48), "ended24", stratum)
                stratum = np.where(
                    (stratum == "free") & ((oi - en > 14 * 48) | (oi < on - 14 * 48)),
                    "free14",
                    stratum,
                )
            keep = np.isfinite(res) & np.isfinite(y)
            for st in sorted(set(stratum)):
                dropped.append(
                    {
                        "month": m,
                        "set": setname,
                        "stratum": st,
                        "rows": int((stratum == st).sum()),
                        "kept": int(((stratum == st) & keep).sum()),
                    }
                )
            age = z["age"][z["org"][idx]].astype(np.float32)
            r0 = np.sort(k1 + (res * np.exp(-lead / sc652.TAU_H))[:, None], axis=1).astype(
                np.float32
            )
            p_b = sc652.cond_survival(curve=curve, age_days=age, lead_h=lead)
            bhaz = np.sort(k1 + (res * p_b)[:, None], axis=1).astype(np.float32)
            lead_day = np.clip((lead // 24).astype(int), 1, 10)
            w_row = (
                np.ones(len(idx))
                if setname == "real"
                else c652.class_weight(short=short, mixture=MIX)
            )
            arms: dict[str, list[tuple[int, np.ndarray, np.ndarray]]] = {
                "K1": [(0, k1, y)], "R0-168": [(0, r0, y)], "Bhaz": [(0, bhaz, y)],
                "E0": [(s, np.sort(take(e0[s], "q"), axis=1), y) for s in SEEDS],
            }  # fmt: skip
            if setname != "real":
                arms["K1_nra"] = [(0, k1, base)]
            for arm in a.arms:
                if arm == "C0":
                    arms["C0"] = [
                        (s, np.sort(take(mine[(arm, s)], "nra"), axis=1), y) for s in SEEDS
                    ]
                else:
                    arms[arm] = [(s, np.sort(take(mine[(arm, s)], "q"), axis=1), y) for s in SEEDS]
                if setname != "real":
                    arms[f"{arm}_nra"] = [
                        (s, np.sort(take(mine[(arm, s)], "nra"), axis=1), base) for s in SEEDS
                    ]
            scale_rows = keep & ~np.isin(stratum, ["censored", "real_other"])
            for arm, lst in arms.items():
                for seed, q, truth in lst:
                    _qs, fac = s650.scale_to_nominal(y=truth[scale_rows], q=q[scale_rows])
                    qsc = (q[:, 5:6] + fac * (q - q[:, 5:6])).astype(np.float32)
                    p = s650.pin(y=truth, q=q)
                    df = pl.DataFrame({
                        "stratum": stratum, "lead_day": lead_day, "w": w_row,
                        "crps": 2 * p.mean(1), "crps_s": s650.crps(y=truth, q=qsc),
                        "cov90": ((truth >= q[:, 1]) & (truth <= q[:, 9])).astype(float),
                        "pin01": p[:, 0], "pin05": p[:, 1], "pin95": p[:, 9], "pin99": p[:, 10],
                        "cov98": ((truth >= q[:, 0]) & (truth <= q[:, 10])).astype(float),
                        "mae": np.abs(truth - q[:, 5]),
                    }).filter(pl.Series(keep))  # fmt: skip
                    extra = [
                        df.filter(pl.col("stratum").is_in(["far", "middle", "near"])).with_columns(
                            stratum=pl.lit("inforce")
                        ),
                        df.with_columns(stratum=pl.lit("all")),
                        df.filter(pl.col("stratum").is_in(["free", "free14"])).with_columns(
                            stratum=pl.lit("free_any")
                        ),
                    ]
                    df = pl.concat([df, *extra])
                    g = df.group_by("stratum", "lead_day").agg(
                        pl.col("w").sum().alias("sw"), pl.len().alias("n"),
                        *[(pl.col(c) * pl.col("w")).sum().alias(c) for c in COLS],
                    )  # fmt: skip
                    agg.append(
                        g.with_columns(
                            month=pl.lit(m),
                            set=pl.lit(setname),
                            mixture=pl.lit(MIX),
                            arm=pl.lit(arm),
                            seed=pl.lit(seed),
                            factor=pl.lit(float(fac)),
                        )
                    )
            if setname != "real":
                sel = keep & in_force & (age >= 0.5)
                band = np.where(lead < 96, "d1-3", np.where(lead < 192, "d4-7", "d8-10"))
                probs = {"Bhaz": [(0, p_b)], "E0": [(s, take(e0[s], "p")) for s in SEEDS]}
                for arm in a.arms:
                    if arm != "C0":
                        probs[arm] = [(s, take(mine[(arm, s)], "p")) for s in SEEDS]
                for arm, lst in probs.items():
                    for seed, p_ in lst:
                        d = pl.DataFrame(
                            {
                                "p": np.asarray(p_, dtype=float),
                                "label": label.astype(float),
                                "w": w_row,
                                "band": band,
                            }
                        ).filter(pl.Series(sel))
                        d = d.with_columns(bin=(pl.col("p") * 10).floor().clip(0, 9))
                        rel.append(
                            d.group_by("bin", "band").agg(
                                pl.col("w").sum().alias("sw"), (pl.col("w") * pl.col("p")).sum().alias("sp"),
                                (pl.col("w") * pl.col("label")).sum().alias("sl"), pl.len().alias("n"),
                            ).with_columns(month=pl.lit(m), set=pl.lit(setname), arm=pl.lit(arm), seed=pl.lit(seed))
                        )  # fmt: skip
            print(m, setname, len(idx), "rows scored", flush=True)
    pl.concat(agg).write_parquet(OUT / "agg.parquet")
    pl.concat(rel).write_parquet(OUT / "reliability.parquet")
    pl.DataFrame(dropped).write_csv(OUT / "dropped.csv")


if __name__ == "__main__":
    main()
