# ruff: noqa: D103
"""Experiment 650, step 8: avoidable-curtailment cost on the six MW Primaries (CPU).

Experiment 601's `avoidable_cost` with limit 0.9 times the Primary's peak before experiment 601's
cut-off, q95 as the planned level, leads 12 to 36 h, in pounds per Primary-day (mean over rows x
48), by arm, over three row sets of the held-out bank: inside the shift, in the 7 days after it
ends, and all rows. Seeds averaged; the interval is a percentile bootstrap over months. Unscaled
forecasts. Writes `score5/cost.csv`.
"""

import e00578_common as k578  # ty: ignore[unresolved-import]
import e00601_bank as b601  # ty: ignore[unresolved-import]
import e00601_common as k601  # ty: ignore[unresolved-import]
import e00650_common as c650
import e00650_fit as f650
import numpy as np
import polars as pl

LEAD = (12.0, 36.0)
FRACTION = 0.9


def main() -> None:
    clean = k578.clean()
    sm = k578.sites()
    cap = sm["capacity_mw"].to_numpy()
    six = np.array([str(g).startswith("signed-MW") for g in sm["group"].to_list()])
    peak = np.nanmax(clean[:, : b601.CUTOFF_IDX], axis=1) * cap
    lim = FRACTION * peak
    rows = []
    for m in c650.MONTHS:
        z = f650.sources(month=m, bank="new")["bank"]
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
        idx = fits["Rnew"][0]["held_idx"]
        pa, pr, vi = (
            z["pass_"][idx].astype(int),
            z["prim"][idx].astype(int),
            z["vi"][idx].astype(int),
        )
        end = np.array([ev[(a, b)]["end_idx"] for a, b in zip(pa, pr, strict=True)])
        on = np.array([ev[(a, b)]["onset_idx"] for a, b in zip(pa, pr, strict=True)])
        lead = z["lead"][idx]
        y = (z["y"][idx] + z["shift"][idx]).astype(np.float64)
        ok = six[pr] & np.isfinite(z["res"][idx]) & (lead >= LEAD[0]) & (lead < LEAD[1])
        k1 = z["q"][idx]
        r0 = k1 + (z["res"][idx] * np.exp(-lead / 168.0))[:, None]
        q95 = {
            "K1": [k1[:, 9]],
            "R0-168": [np.sort(r0, axis=1)[:, 9]],
            **{a: [np.sort(f["held_q"], axis=1)[:, 9] for f in fs] for a, fs in fits.items()},
        }
        sets = {
            "inside": (vi >= on) & (vi < end),
            "7 days after the end": (vi >= end) & (vi < end + 7 * 48),
            "all rows": np.ones(len(vi), bool),
        }
        for sname, sel0 in sets.items():
            sel = ok & sel0
            if sel.sum() < 100:
                continue
            for arm, qs in q95.items():
                c = np.mean(
                    [
                        k601.avoidable_cost(
                            u=q[sel] * cap[pr[sel]], y=y[sel] * cap[pr[sel]], limit=lim[pr[sel]]
                        ).mean()
                        * 48
                        for q in qs
                    ]
                )
                rows.append(
                    {
                        "month": m,
                        "set": sname,
                        "arm": arm,
                        "cost_gbp_per_primary_day": float(c),
                        "n": int(sel.sum()),
                    }
                )
    d = pl.DataFrame(rows)
    d.write_csv(c650.WORK / "score5" / "cost_by_month.csv")
    out = (
        d.group_by("set", "arm")
        .agg(
            pl.col("cost_gbp_per_primary_day").mean().round(1),
            pl.col("cost_gbp_per_primary_day").min().round(1).alias("min_month"),
            pl.col("cost_gbp_per_primary_day").max().round(1).alias("max_month"),
        )
        .sort("set", "arm")
    )
    out.write_csv(c650.WORK / "score5" / "cost.csv")
    print(out)


if __name__ == "__main__":
    main()
