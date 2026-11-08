# ruff: noqa: E501, ANN001, ANN201, D103, D205
"""Experiment 650, step 9: the audit checks on the five-month fits (CPU): median minus mean forecast, the
median's error beside the quantile contrasts, on the decision rows and on event-free rows. Writes
`score5/checks.csv`.
"""

import e00650_common as c650
import e00650_fit as f650
import numpy as np
import polars as pl

TAUS = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99])


def mean_fc(q):
    return (
        q[:, 0] * 0.01
        + q[:, -1] * 0.01
        + (0.5 * (q[:, 1:] + q[:, :-1]) * np.diff(TAUS)[None, :]).sum(1)
    )


def main() -> None:
    rows = []
    for m in c650.MONTHS:
        src = f650.sources(month=m, bank="new")
        events = pl.concat(
            [
                pl.read_parquet(c650.ROWS / f"events_{m}_r{r:02d}.parquet")
                for r in c650.HELD_OUT_PASSES
            ]
        )
        ev = {(int(r["pass"]), int(r["primary"])): r for r in events.iter_rows(named=True)}
        fits = {
            a: [np.load(c650.WORK / "fits5" / f"{a}_{m}_s{s}.npz") for s in (0, 1)]
            for a in ("Rnew", "Rold", "Placebo")
        }
        for name, z in (("held", src["bank"]), ("real", src["real"])):
            idx = fits["Rnew"][0][f"{name}_idx"]
            if name == "held":
                pa, pr, vi = (
                    z["pass_"][idx].astype(int),
                    z["prim"][idx].astype(int),
                    z["vi"][idx].astype(int),
                )
                on = np.array([ev[(a, b)]["onset_idx"] for a, b in zip(pa, pr, strict=True)])
                en = np.array([ev[(a, b)]["end_idx"] for a, b in zip(pa, pr, strict=True)])
                sel = np.isfinite(z["res"][idx]) & (vi >= on) & (vi < en)
                y = z["y"][idx] + z["shift"][idx]
            else:
                sel = (
                    z["event_free"][idx]
                    & ~z["flagged"][idx]
                    & ~z["verified"][idx]
                    & np.isfinite(z["res"][idx])
                )
                y = z["y"][idx]
            lead = z["lead"][idx]
            k1 = z["q"][idx]
            qs = {
                "K1": k1,
                "R0-168": np.sort(k1 + (z["res"][idx] * np.exp(-lead / 168.0))[:, None], axis=1),
            }
            for a, fs in fits.items():
                qs[a] = np.sort(np.mean([f[f"{name}_q"] for f in fs], axis=0), axis=1)
            for a, q in qs.items():
                err_med = np.abs(q[sel, 5] - y[sel]).mean() * 100
                mf = mean_fc(q[sel])
                rows.append({"month": m, "set": name, "arm": a, "mae_median": err_med, "mae_mean": np.abs(mf - y[sel]).mean() * 100, "median_minus_mean": ((q[sel, 5] - mf) * 100).mean(), "n": int(sel.sum())})  # fmt: skip
    d = pl.DataFrame(rows)
    out = (
        d.group_by("set", "arm")
        .agg(
            pl.col("mae_median").mean().round(2),
            pl.col("mae_mean").mean().round(2),
            pl.col("median_minus_mean").mean().round(3),
        )
        .sort("set", "arm")
    )
    out.write_csv(c650.WORK / "score5" / "checks.csv")
    print(out)


if __name__ == "__main__":
    main()
