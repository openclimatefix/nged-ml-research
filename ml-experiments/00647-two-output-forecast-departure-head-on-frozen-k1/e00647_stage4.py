# ruff: noqa: E731, PERF401
# ruff: noqa: D103, E501
"""Experiment 647, stage 4: intervals and tables from the five-month leave-one-month-out scorecard (CPU).

Reads `score5/scorecard.csv` (written by `e00647_score.py all5`) and writes, beside it:

- `stage4_contrasts.csv`: pooled differences of the CRPS proxy (points of capacity) with 95% intervals
  that resample the five origin months and the five seeds (K1 seeds for E2, head seeds for H1; the
  same sampled seeds for both arms of a contrast; K1 and the R0 baselines have no seed);
- `stage4_levels.csv`: each arm's pooled CRPS proxy with the same interval;
- `stage4_groups.csv`, `stage4_primaries.csv`, `stage4_tails.csv`, `stage4_medmean.csv`, `stage4_cost.csv`;
- `stage4_tables.md`: all of them as markdown.

Usage: python e00647_stage4.py
"""

import numpy as np
import polars as pl
from e00647_common import MONTHS, WORK

SC = WORK / "score5"
N_BOOT = 4000
SETS = {
    "P1 event-free": "P1/P5 event-free",
    "P2 inside shift": "P2 inside shift",
    "P3 after end": "P3 after end",
    "P4 flagged or verified": "P4 flagged or verified",
}
LENGTHS = (3, 7, 15, 30)
FITTED = ("H1", "H2", "H1p", "H2p", "AG", "AS", "E1", "E2")


def arm_matrix(sc: pl.DataFrame, *, arm: str, setname: str, cut: str, metric: str) -> np.ndarray:
    """Return (months, seeds) values of the metric; baselines repeat across seeds."""
    out = np.full((len(MONTHS), 5), np.nan)
    d = sc.filter(
        (pl.col("arm") == arm)
        & (pl.col("set") == setname)
        & (pl.col("cut") == cut)
        & pl.col("month").is_in(MONTHS)
    )
    for r in d.iter_rows(named=True):
        mi = MONTHS.index(r["month"])
        if r["seed"] < 0:
            out[mi, :] = r[metric]
        else:
            out[mi, r["seed"]] = r[metric]
    return out


def weights(sc: pl.DataFrame, *, setname: str, cut: str) -> np.ndarray:
    d = sc.filter(
        (pl.col("arm") == "K1")
        & (pl.col("set") == setname)
        & (pl.col("cut") == cut)
        & pl.col("month").is_in(MONTHS)
    )
    w = np.zeros(len(MONTHS))
    for r in d.iter_rows(named=True):
        w[MONTHS.index(r["month"])] = r["rows"]
    return w


def pooled(v: np.ndarray, w: np.ndarray, mi: np.ndarray, si: np.ndarray) -> float:
    x = np.nanmean(v[np.ix_(mi, si)], axis=1)
    ww = np.where(np.isfinite(x), w[mi], 0.0)
    return (
        float(np.nansum(np.where(np.isfinite(x), x, 0.0) * ww) / ww.sum())
        if ww.sum() > 0
        else float("nan")
    )


def interval(
    sc: pl.DataFrame, *, a: str, b: str | None, setname: str, cut: str, metric: str = "crps_proxy"
) -> dict:
    va = arm_matrix(sc, arm=a, setname=setname, cut=cut, metric=metric)
    vb = arm_matrix(sc, arm=b, setname=setname, cut=cut, metric=metric) if b else np.zeros_like(va)
    w = weights(sc, setname=setname, cut=cut)
    rng = np.random.default_rng(647)
    full_m, full_s = np.arange(len(MONTHS)), np.arange(5)
    point = pooled(va - vb, w, full_m, full_s)
    reps = []
    for _ in range(N_BOOT):
        reps.append(
            pooled(va - vb, w, rng.integers(0, len(MONTHS), len(MONTHS)), rng.integers(0, 5, 5))
        )
    lo, hi = np.nanpercentile(reps, [2.5, 97.5])
    return {"point": point, "lo": float(lo), "hi": float(hi), "rows": int(w.sum())}


def md(df: pl.DataFrame, digits: int = 3) -> str:
    cols = df.columns
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in df.iter_rows():
        lines.append(
            "| "
            + " | ".join(f"{v:.{digits}f}" if isinstance(v, float) else str(v) for v in r)
            + " |"
        )
    return "\n".join(lines)


def main() -> None:
    sc = pl.read_csv(SC / "scorecard.csv").filter(pl.col("variant") == "lomo")
    # the seed-averaged encoder arms are scored once as seed -2 and once more as seed 0; drop the duplicate
    sc = sc.filter(~(pl.col("arm").is_in(["E1a", "E2a", "E3a", "E3pa"]) & (pl.col("seed") == 0)))
    text: list[str] = []
    # 1. levels and contrasts
    levels, contrasts = [], []
    arms = ("K1", "R0_72", "R0s_72", "H1", "H1p", "H2", "H2p", "AG", "AS", "E1", "E2")
    cuts = [("all", "all")] + [(f"length {n} d", f"length {n} d") for n in LENGTHS]
    for sname, sset in SETS.items():
        for cname, cut in cuts:
            if cname != "all" and not (sname.startswith(("P2", "P3"))):
                continue
            for arm in arms:
                levels.append(
                    {
                        "set": sname,
                        "cut": cname,
                        "arm": arm,
                        **interval(sc, a=arm, b=None, setname=sset, cut=cut),
                    }
                )
            for a, b in (
                ("H1", "K1"),
                ("H1", "R0_72"),
                ("H1", "R0s_72"),
                ("H1", "H1p"),
                ("H2", "H2p"),
                ("E2", "H1"),
                ("E2", "K1"),
                ("E1", "H1"),
            ):
                contrasts.append(
                    {
                        "set": sname,
                        "cut": cname,
                        "contrast": f"{a} minus {b}",
                        **interval(sc, a=a, b=b, setname=sset, cut=cut),
                    }
                )
    lv, ct = pl.DataFrame(levels), pl.DataFrame(contrasts)
    lv.write_csv(SC / "stage4_levels.csv")
    ct.write_csv(SC / "stage4_contrasts.csv")
    text += [
        "## Contrasts (CRPS proxy, points of capacity; negative favours the first arm; 95% interval resampling 5 months and 5 seeds)",
        md(ct.with_columns(pl.col("point", "lo", "hi").round(3))),
    ]
    text += [
        "## Levels (pooled CRPS proxy with 95% interval)",
        md(lv.with_columns(pl.col("point", "lo", "hi").round(3))),
    ]

    # 2. per group (real sets) and per Primary
    a5 = sc.filter(pl.col("month") == "all5")
    seedmean = lambda d: d.filter((pl.col("seed") < 0) | (pl.col("seed") == -2))
    g = seedmean(a5).filter(
        pl.col("cut").str.starts_with("group")
        & ~pl.col("cut").str.contains(r"\|")
        & pl.col("arm").is_in(["K1", "R0_72", "R0s_72", "H1", "H1p", "E2"])
    )
    g = g.select(
        "set", "cut", "arm", "rows", pl.col("crps_proxy").round(3), pl.col("cover90").round(3)
    )
    gp = g.pivot(on="arm", index=["set", "cut", "rows"], values="crps_proxy").sort("set", "cut")
    gp.write_csv(SC / "stage4_groups.csv")
    text += [
        "## CRPS proxy by metering group (real-row sets; pooled over five months; seed means)",
        md(gp),
    ]
    pr = seedmean(a5).filter(
        pl.col("cut").str.starts_with("prim ")
        & pl.col("arm").is_in(["K1", "R0_72", "R0s_72", "H1", "H1p", "E2a"])
    )
    prp = pr.pivot(on="arm", index=["set", "cut", "rows"], values="crps_proxy").sort("set", "cut")
    prp.write_csv(SC / "stage4_primaries.csv")
    text += [
        "## CRPS proxy per Primary (P1, P2, P3, P4a; pooled over five months; seed means; E2a is the seed-averaged output)",
        md(prp.with_columns(pl.col(pl.Float64).round(3))),
    ]
    # 3. median minus mean per site
    mm = (
        pr.filter(pl.col("set") == "P1/P5 event-free")
        .pivot(on="arm", index=["cut"], values="medmean")
        .sort("cut")
    )
    mm.write_csv(SC / "stage4_medmean.csv")
    text += [
        "## Median minus mean forecast per Primary, event-free rows (points of capacity; mean of the quantile function, flat beyond q01 and q99)",
        md(mm.with_columns(pl.col(pl.Float64).round(3))),
    ]
    # 4. tails
    tails = seedmean(a5).filter(
        (pl.col("cut") == "all") & pl.col("arm").is_in(["K1", "R0_72", "R0s_72", "H1", "H1p", "E2"])
    )
    tails = tails.select(
        "set",
        "arm",
        "pin01",
        "pin05",
        "pin95",
        "pin99",
        pl.col("low01").alias("below_q01"),
        pl.col("low").alias("below_q05"),
        pl.col("high").alias("above_q95"),
        pl.col("high99").alias("above_q99"),
        "width",
    ).sort("set", "arm")
    tails.write_csv(SC / "stage4_tails.csv")
    text += [
        "## Tails (pinball loss at q01, q05, q95, q99 in points of capacity; share of rows below q01, below q05, above q95, above q99; q05-q95 width)",
        md(tails.with_columns(pl.col(pl.Float64).round(4))),
    ]
    # 5. curtailment cost
    cost = (
        seedmean(a5)
        .filter(
            (pl.col("cut") == "all")
            & pl.col("arm").is_in(["K1", "R0_72", "R0s_72", "H1", "H1p", "E2"])
        )
        .select("set", "arm", "cost_0.9", "cost_0.75", "cost_0.5")
        .sort("set", "arm")
    )
    cost.write_csv(SC / "stage4_cost.csv")
    text += [
        "## Avoidable-curtailment cost proxy (six MW Primaries, leads 24 to 36 h; limit fractions 0.9, 0.75, 0.5 of the pre-2025-10 peak)",
        md(cost.with_columns(pl.col(pl.Float64).round(4))),
    ]
    (SC / "stage4_tables.md").write_text("\n\n".join(text) + "\n")
    print("\n\n".join(text[:1]))


if __name__ == "__main__":
    main()
