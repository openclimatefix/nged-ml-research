"""Experiment 636: turn the step statistics into recall, size, false-step, and MDS tables.

Usage: `uv run --no-sync python e00636_analyse.py`. Writes CSVs next to `stats.parquet`.
"""

import e00636_common as c
import numpy as np
import polars as pl

PASS_RECALL = 0.8
Z = 1.96


def wilson(hits: int, n: int) -> tuple[float, float]:
    """Return the Wilson 95% interval of a proportion."""
    if n == 0:
        return float("nan"), float("nan")
    p = hits / n
    den = 1 + Z**2 / n
    centre = (p + Z**2 / (2 * n)) / den
    half = Z * np.sqrt(p * (1 - p) / n + Z**2 / (4 * n**2)) / den
    return float(centre - half), float(centre + half)


def thresholds(stats: pl.DataFrame) -> pl.DataFrame:
    """95th percentile of the absolute uninjected statistic over the 20 standard starts."""
    return (
        stats.filter(
            (pl.col("world") == "none") & pl.col("standard_start") & ~pl.col("stat").is_nan()
        )
        .group_by("farm", "est", "k")
        .agg(thr=pl.col("stat").abs().quantile(0.95), n_thr=pl.len())
    )


def main() -> None:
    """Write the analysis tables."""
    stats = pl.read_parquet(c.DATA_DIR / "stats.parquet")
    thr = thresholds(stats)
    thr.write_csv(c.DATA_DIR / "thresholds.csv")
    base = stats.filter(pl.col("world") == "none").select(
        "farm", "start", "est", "k", placebo="stat"
    )
    inj = (
        stats.filter(pl.col("world") != "none")
        .join(thr, on=["farm", "est", "k"])
        .join(base, on=["farm", "start", "est", "k"])
        .with_columns(
            detected=(pl.col("stat") < -pl.col("thr")) & ~pl.col("stat").is_nan(),
            size=(pl.col("placebo") - pl.col("stat")),
        )
    )
    inj.write_parquet(c.DATA_DIR / "injected_with_thresholds.parquet")

    # Recall, worlds ac and dc (standard starts).
    rows = []
    for keys, g in inj.filter(pl.col("world").is_in(["ac", "dc"])).group_by(
        "world", "farm", "est", "loss", "k"
    ):
        hits, n = int(g["detected"].sum()), g.height
        lo, hi = wilson(hits, n)
        done = g.filter(~pl.col("size").is_nan() & pl.col("size").is_not_null())
        rows.append(
            dict(zip(("world", "farm", "est", "loss", "k"), keys, strict=True))
            | dict(
                recall=hits / n, lo=lo, hi=hi, n=n,
                size_ratio=float(done["size"].median() / keys[3]) if done.height else np.nan,
                size_ratio_detected=float(
                    g.filter(pl.col("detected"))["size"].median() / keys[3]
                ) if hits else np.nan,
                thr=float(g["thr"][0]),
            )
        )  # fmt: skip
    recall = pl.DataFrame(rows).sort("world", "est", "farm", "k", "loss")
    recall.write_csv(c.DATA_DIR / "recall.csv")

    # Minimum detectable step: smallest loss with recall >= 0.8, linear interpolation.
    mds = []
    for keys, g in recall.group_by("world", "est", "farm", "k"):
        g = g.sort("loss")
        x, y = g["loss"].to_numpy(), g["recall"].to_numpy()
        val = np.nan
        for i in range(len(x)):
            if y[i] >= PASS_RECALL:
                val = (
                    x[i]
                    if i == 0
                    else x[i - 1] + (PASS_RECALL - y[i - 1]) * (x[i] - x[i - 1]) / (y[i] - y[i - 1])
                )
                break
        mds.append(dict(zip(("world", "est", "farm", "k"), keys, strict=True)) | {"mds": val})
    pl.DataFrame(mds).sort("world", "est", "farm", "k").write_csv(c.DATA_DIR / "mds.csv")

    # False-step rates: uninjected statistic below -threshold.
    fs = base.join(thr, on=["farm", "est", "k"]).with_columns(
        false_step=pl.col("placebo") < -pl.col("thr"), has=~pl.col("placebo").is_nan()
    )
    farms = {f: c.load_farms()[f] for f in c.TESTABLE}
    cloudy = {f: set(c.cloudiest_months(farms[f], n=10)) for f in c.TESTABLE}
    cloudy3 = {f: set(c.cloudiest_months(farms[f], n=3)) for f in c.TESTABLE}
    standard = {c.month_index(y, m) for y, m in c.START_MONTHS}
    fs = fs.with_columns(
        group=pl.struct("farm", "start").map_elements(
            lambda r: (
                "cloudy3"
                if r["start"] in cloudy3[r["farm"]]
                else (
                    "cloudy10"
                    if r["start"] in cloudy[r["farm"]]
                    else ("standard" if r["start"] in standard else "other")
                )
            ),
            return_dtype=pl.String,
        )
    )
    out = []
    for grp in ("standard", "cloudy3", "cloudy10"):
        sub = fs.filter(pl.col("has"))
        sub = (
            sub.filter(pl.col("start").is_in(list(standard)))
            if grp == "standard"
            else sub.filter(
                pl.struct("farm", "start").map_elements(
                    lambda r, grp=grp: (
                        r["start"] in (cloudy3 if grp == "cloudy3" else cloudy)[r["farm"]]
                    ),
                    return_dtype=pl.Boolean,
                )
            )
        )
        for keys, g in sub.group_by("est", "k"):
            hits, n = int(g["false_step"].sum()), g.height
            lo, hi = wilson(hits, n)
            out.append(dict(group=grp, est=keys[0], k=keys[1], rate=hits / n, lo=lo, hi=hi, n=n))
    pl.DataFrame(out).sort("group", "est", "k").write_csv(c.DATA_DIR / "false_step.csv")

    # Curtailment placebo: detection at each horizon, by farm and pooled.
    cur = inj.filter(pl.col("world") == "curtail")
    out = []
    for keys, g in cur.group_by("est", "k", "farm"):
        hits, n = int(g["detected"].sum()), g.height
        lo, hi = wilson(hits, n)
        out.append(dict(est=keys[0], k=keys[1], farm=keys[2], rate=hits / n, lo=lo, hi=hi, n=n))
    for keys, g in cur.group_by("est", "k"):
        hits, n = int(g["detected"].sum()), g.height
        lo, hi = wilson(hits, n)
        out.append(dict(est=keys[0], k=keys[1], farm="all", rate=hits / n, lo=lo, hi=hi, n=n))
    pl.DataFrame(out).sort("est", "k", "farm").write_csv(c.DATA_DIR / "curtail.csv")

    # Cloudy-month injection recall.
    cl = inj.filter(pl.col("world").str.ends_with("_cloudy"))
    out = []
    for keys, g in cl.group_by("world", "est", "loss", "k"):
        hits, n = int(g["detected"].sum()), g.height
        lo, hi = wilson(hits, n)
        out.append(
            dict(zip(("world", "est", "loss", "k"), keys, strict=True))
            | dict(recall=hits / n, lo=lo, hi=hi, n=n)
        )
    pl.DataFrame(out).sort("world", "est", "loss", "k").write_csv(c.DATA_DIR / "cloudy_recall.csv")

    # Stop rule: farms missing a 10% step at 3 months (recall < 0.8), per world, slope.
    stop = recall.filter((pl.col("loss") == 0.10) & (pl.col("k") == 3))
    stop.write_csv(c.DATA_DIR / "stop_rule.csv")
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(20)
    print(
        "THRESHOLDS",
        thr.sort("est", "k", "farm")
        .with_columns(pl.col("thr").round(3))
        .pivot(on="farm", index=["est", "k"], values="thr"),
    )
    print(
        "STOP RULE (10% at 3 months)\n",
        stop.select("world", "est", "farm", "recall", "lo", "hi", "size_ratio", "thr")
        .with_columns(pl.col(pl.Float64).round(2))
        .sort("world", "est", "farm"),
    )
    print(
        "RECALL slope/envelope, 10% by k\n",
        recall.filter(pl.col("loss") == 0.10)
        .pivot(on="farm", index=["world", "est", "k"], values="recall")
        .with_columns(pl.col(pl.Float64).round(2))
        .sort("world", "est", "k"),
    )
    print(
        "MDS\n",
        pl.DataFrame(mds)
        .pivot(on="farm", index=["world", "est", "k"], values="mds")
        .with_columns(pl.col(pl.Float64).round(3))
        .sort("world", "est", "k"),
    )
    print(
        "FALSE STEP\n",
        pl.read_csv(c.DATA_DIR / "false_step.csv").with_columns(pl.col(pl.Float64).round(2)),
    )
    print(
        "CURTAIL (pooled)\n",
        pl.read_csv(c.DATA_DIR / "curtail.csv")
        .filter(pl.col("farm") == "all")
        .with_columns(pl.col(pl.Float64).round(2)),
    )
    print(
        "CLOUDY INJECTION\n",
        pl.read_csv(c.DATA_DIR / "cloudy_recall.csv")
        .filter(pl.col("k").is_in([1, 3]))
        .with_columns(pl.col(pl.Float64).round(2)),
    )


if __name__ == "__main__":
    main()
