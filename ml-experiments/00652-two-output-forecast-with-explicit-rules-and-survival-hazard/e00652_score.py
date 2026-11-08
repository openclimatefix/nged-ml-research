# ruff: noqa: E501, ANN001, D103, PERF401, PLR0912, PLR0915
"""Experiment 652, step 2: score every arm by origin state and lead day (CPU).

Arms: `K1` (the NRA output, scored against the observed-plus-shift truth), `K1_nra` (K1 scored
against the unshifted base, bank rows only), `R0-168`, `Bhaz` (K1 plus the same residual times the
conditional survival of the best single curve fitted to the training durations), `Rold` and `Rnew`
(experiment 650's five-month fits), `Rhaz` (this experiment's survival head for the mixture), and
`Placebo`. Mixtures: `m50`, `mshort`, `mlong` (event-class weights on rows; `Rhaz` and `Bhaz` are
the arms trained or fitted under that mixture).

Equal width scaling is fitted on unweighted rows, so the sensitivity mixtures reuse the unweighted
factors. One factor per arm, month, set, and seed, fitted on all the month's scored rows pooled
across strata (not on the shift's future end), so that the q05-q95 band covers 0.90.

The stratum `inforce` pools `far`, `middle`, and `near` and duplicates their rows in `agg`: never
sum over strata. The learned survival of `Rhaz` is measured from the origin; `Rprior` (experiment
647's fixed prior) is measured from the newest lag, one day earlier.

Output (data folder `score/`): `agg.parquet` (weighted sums by month, set, mixture, arm, seed,
stratum, lead day), `reliability.parquet`, `dropped.csv`, `survival_curves.csv`.

Usage: python e00652_score.py
"""

import e00650_common as c650  # ty: ignore[unresolved-import]
import e00650_fit as f650  # ty: ignore[unresolved-import]
import e00650_score as s650  # ty: ignore[unresolved-import]
import e00652_common as c652  # ty: ignore[unresolved-import]
import e00652_fit as f652  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from scipy import optimize, stats

OUT = c652.WORK / "score"
FITS5 = c650.WORK / "fits5"
TAU_H = 168.0
PRIOR_SHORT = 0.5
DISTS = {
    "expon": stats.expon,
    "weibull": stats.weibull_min,
    "lognorm": stats.lognorm,
    "loglogistic": stats.fisk,
}


def survival_curve(*, months: list[str], mixture: str) -> tuple[str, tuple, float]:
    """Return the best of four survival curves fitted to the training durations.

    The curves are exponential, Weibull, log-normal, and log-logistic.
    """
    days, w = [], []
    for m in months:
        ev = c652.events_table(month=m).filter(
            (pl.col("pass") < 200) & (pl.col("end_idx") < c650.WINDOW_START_IDX)
        )
        d = ev["length_days"].to_numpy().astype(float)
        days.append(d)
        w.append(c652.class_weight(short=d <= c652.SHORT_MAX_DAYS, mixture=mixture))
    d, w = np.concatenate(days), np.concatenate(w)
    best = None
    for name, dist in DISTS.items():
        n_par = 0 if name == "expon" else 1

        def nll(theta, dist=dist, n_par=n_par) -> float:
            shape = np.exp(theta[:n_par]) if n_par else ()
            return -float((w * dist.logpdf(d, *shape, 0.0, np.exp(theta[n_par]))).sum())

        r = optimize.minimize(
            nll, x0=np.r_[np.zeros(n_par), np.log(d.mean())], method="Nelder-Mead"
        )
        par = (*(np.exp(r.x[:n_par])), 0.0, float(np.exp(r.x[n_par])))
        if best is None or r.fun < best[2]:
            best = (name, par, float(r.fun))
    assert best is not None
    return best


def cond_survival(
    *, curve: tuple[str, tuple, float], age_days: np.ndarray, lead_h: np.ndarray
) -> np.ndarray:
    """Return S(age + 1 + lead / 24) / S(age + 1) for the fitted curve.

    The age is measured at the newest lag, 1 day before the origin. The result is zero where no
    departure was detected (age below 0.5 days).
    """
    dist, par = DISTS[curve[0]], curve[1]
    a = np.clip(age_days, 0.5, 100.0)
    p = dist.sf(a + 1.0 + lead_h / 24.0, *par) / np.maximum(dist.sf(a + 1.0, *par), 1e-3)
    return np.where(age_days < 0.5, 0.0, np.clip(p, 0.0, 1.0))


def prior_survival(*, age_days: np.ndarray, lead_h: np.ndarray) -> np.ndarray:
    import e00647_head as h  # ty: ignore[unresolved-import]
    import torch  # ty: ignore[unresolved-import]

    return h.survival_probability(
        age_days=torch.from_numpy(age_days.astype(np.float32)),
        lead_h=torch.from_numpy(lead_h.astype(np.float32)),
        short_weight=PRIOR_SHORT,
    ).numpy()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    agg, rel, dropped, curves = [], [], [], []
    for m in c650.MONTHS:
        src = f650.sources(month=m, bank="new")
        f652.with_state(z=src["bank"], month=m)
        train_months = [x for x in c650.MONTHS if x != m]
        curve = {mix: survival_curve(months=train_months, mixture=mix) for mix in c652.MIXTURES}
        curves += [
            {
                "month": m,
                "mixture": mix,
                "curve": c[0],
                "params": str(np.round(c[1], 4).tolist()),
                "nll": c[2],
            }
            for mix, c in curve.items()
        ]
        for setname in ("fam", "held", "real"):
            z = src["real" if setname == "real" else "bank"]
            r5 = {
                a: [np.load(FITS5 / f"{a}_{m}_s{s}.npz") for s in c652.SEEDS]
                for a in ("Rnew", "Rold")
            }
            rh = {
                a: [np.load(f652.OUT / f"{a}_{m}_s{s}.npz") for s in c652.SEEDS] for a in f652.ARMS
            }
            idx = r5["Rnew"][0][f"{setname}_idx"]
            for fs in [*r5.values(), *rh.values()]:
                for f in fs:
                    assert np.array_equal(f[f"{setname}_idx"], idx), "index sets differ"
            lead, k1, res = z["lead"][idx], z["q"][idx], z["res"][idx]
            assert z["vi"][idx].max() < c650.WINDOW_START_IDX, "a valid time on or after 2026-07-01"
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
                stratum, short = z["stratum"][idx], z["short"][idx]
                label, in_force = (
                    z["still_on"][idx] & ~z["censored"][idx],
                    z["in_force"][idx] & ~z["censored"][idx],
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
            r0 = np.sort(k1 + (res * np.exp(-lead / TAU_H))[:, None], axis=1).astype(np.float32)
            lead_day = np.clip((lead // 24).astype(int), 1, 10)
            for mix in c652.MIXTURES:
                w_row = (
                    np.ones(len(idx))
                    if setname == "real"
                    else c652.class_weight(short=short, mixture=mix)
                )
                p_b = cond_survival(curve=curve[mix], age_days=age, lead_h=lead)
                bhaz = np.sort(k1 + (res * p_b)[:, None], axis=1).astype(np.float32)
                arms: dict[
                    str, list[tuple[int, np.ndarray, np.ndarray]]
                ] = {  # name -> [(seed, q, truth)]
                    "K1": [(0, k1, y)],
                    "R0-168": [(0, r0, y)],
                    "Bhaz": [(0, bhaz, y)],
                    **({"K1_nra": [(0, k1, base)]} if setname != "real" else {}),
                }
                for a in ("Rold", "Rnew"):
                    arms[a] = [
                        (s, np.sort(f[f"{setname}_q"], axis=1), y) for s, f in enumerate(r5[a])
                    ]
                arms["Rhaz"] = [
                    (s, np.sort(f[f"{setname}_q"], axis=1), y)
                    for s, f in enumerate(rh[f"Rhaz_{mix}"])
                ]
                arms["Placebo"] = [
                    (s, np.sort(f[f"{setname}_q"], axis=1), y) for s, f in enumerate(rh["Placebo"])
                ]
                scale_rows = keep & ~np.isin(stratum, ["censored", "real_other"])
                for arm, lst in arms.items():
                    for seed, q, truth in lst:
                        _qs, fac = s650.scale_to_nominal(y=truth[scale_rows], q=q[scale_rows])
                        qsc = (q[:, 5:6] + fac * (q - q[:, 5:6])).astype(np.float32)
                        p = s650.pin(y=truth, q=q)
                        crps, crps_s = 2 * p.mean(1), s650.crps(y=truth, q=qsc)
                        cov = ((truth >= q[:, 1]) & (truth <= q[:, 9])).astype(float)
                        df = pl.DataFrame(
                            {
                                "stratum": stratum,
                                "lead_day": lead_day,
                                "w": w_row,
                                "crps": crps,
                                "crps_s": crps_s,
                                "cov90": cov,
                                "pin01": p[:, 0],
                                "pin05": p[:, 1],
                                "pin95": p[:, 9],
                                "pin99": p[:, 10],
                                "cov98": ((truth >= q[:, 0]) & (truth <= q[:, 10])).astype(float),
                            }
                        ).filter(pl.Series(keep))
                        df = pl.concat(
                            [
                                df,
                                df.filter(
                                    pl.col("stratum").is_in(["far", "middle", "near"])
                                ).with_columns(stratum=pl.lit("inforce")),
                            ]
                        )
                        cols = [
                            "crps",
                            "crps_s",
                            "cov90",
                            "pin01",
                            "pin05",
                            "pin95",
                            "pin99",
                            "cov98",
                        ]
                        g = df.group_by("stratum", "lead_day").agg(
                            pl.col("w").sum().alias("sw"),
                            pl.len().alias("n"),
                            *[(pl.col(c) * pl.col("w")).sum().alias(c) for c in cols],
                        )
                        agg.append(
                            g.with_columns(
                                month=pl.lit(m),
                                set=pl.lit(setname),
                                mixture=pl.lit(mix),
                                arm=pl.lit(arm),
                                seed=pl.lit(seed),
                                factor=pl.lit(float(fac)),
                            )
                        )
                if setname != "real":  # survival reliability on in-force rows with a valid label
                    sel = keep & in_force & (age >= 0.5)
                    probs = {"Bhaz": [(0, p_b)], "Rprior": [(0, prior_survival(age_days=age, lead_h=lead))],
                             "Rhaz": [(s, f[f"{setname}_p"]) for s, f in enumerate(rh[f"Rhaz_{mix}"])],
                             "Placebo": [(s, f[f"{setname}_p"]) for s, f in enumerate(rh["Placebo"])]}  # fmt: skip
                    band = np.where(lead < 96, "d1-3", np.where(lead < 192, "d4-7", "d8-10"))
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
                                d.group_by("bin", "band")
                                .agg(
                                    pl.col("w").sum().alias("sw"),
                                    (pl.col("w") * pl.col("p")).sum().alias("sp"),
                                    (pl.col("w") * pl.col("label")).sum().alias("sl"),
                                    pl.len().alias("n"),
                                )
                                .with_columns(
                                    month=pl.lit(m),
                                    set=pl.lit(setname),
                                    mixture=pl.lit(mix),
                                    arm=pl.lit(arm),
                                    seed=pl.lit(seed),
                                )
                            )
            print(m, setname, "scored", flush=True)
    pl.concat(agg).write_parquet(OUT / "agg.parquet")
    pl.concat(rel).write_parquet(OUT / "reliability.parquet")
    pl.DataFrame(dropped).write_csv(OUT / "dropped.csv")
    pl.DataFrame(curves).write_csv(OUT / "survival_curves.csv")


if __name__ == "__main__":
    main()
