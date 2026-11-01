"""Experiment 458: pool the cells into contrasts against B-mix, weight tables, and tail tables.

```bash
DATA_PATH_INTERNAL=<private data folder> uv run --no-sync python ml-experiments/00458-*/e00458_report.py
```

Months are the unit of resampling. Seeds are averaged within a month first. The warm-up month
(2025-10, scored at 50/50 for every arm) is left out of every pooled number, B-mix's included.
"""

from typing import Final

import e00458_core as core  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

REFERENCE: Final[str] = "B-mix"
CANDIDATES: Final[tuple[str, ...]] = ("b-A0", "b-S1", "c-A0", "c-S1", "d")
CONTROLS: Final[tuple[str, ...]] = (
    "b2-A0", "b2-S1", "b3-A0", "b3-S1", "half-A0", "half-S1", "lagged-A0", "lagged-S1", "lagless",
)  # fmt: skip
CONTRAST_COUNT: Final[int] = 20
"""5 candidate arms x 2 row sets (real-event rows, rows with no labelled event) x 2 groups."""
RESAMPLES: Final[int] = 20000
MDE_MULTIPLIER: Final[float] = 2.8
SCORED: Final[tuple[str, ...]] = tuple(m for m in core.MONTHS if m != core.WARM_UP)


def boot(*, values: np.ndarray, level: float) -> tuple[float, float]:
    """Return the percentile interval of the mean, resampling the entries (months)."""
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(0)
    means = values[rng.integers(0, len(values), size=(RESAMPLES, len(values)))].mean(1)
    tail = (1.0 - level) / 2.0
    return float(np.quantile(means, tail)), float(np.quantile(means, 1.0 - tail))


def month_means(*, summary: pl.DataFrame, group: str, rowset: str, band: str, col: str) -> pl.DataFrame:
    """Return arm x month means (seeds averaged) of `col` on scored months."""
    return (
        summary.filter(
            (pl.col("group") == group) & (pl.col("rowset") == rowset) & (pl.col("band") == band)
            & pl.col("month").is_in(SCORED)
        )  # fmt: skip
        .group_by("arm", "month")
        .agg(pl.col(col).mean().alias("v"), pl.col("n").mean().alias("n"))
    )


def contrasts(*, summary: pl.DataFrame, reference: str = REFERENCE) -> pl.DataFrame:
    """Return arm-minus-B-mix contrasts on each group and row set, with intervals and the MDE."""
    rows = []
    level = 1.0 - 0.05 / CONTRAST_COUNT
    for group in ("signed_six", "mva_eight", "report_only"):
        for rowset, col in (
            ("event", "crps_nra1.0"), ("event", "crps"), ("event", "crps_nra0.75"),
            ("event", "crps_nra1.25"), ("nolabel", "crps"), ("harm", "crps"), ("post", "crps_nra1.0"),
            ("event", "pin_q99_raw"), ("nolabel", "pin_q99_raw"),
        ):  # fmt: skip
            mm = month_means(summary=summary, group=group, rowset=rowset, band="all", col=col)
            if mm.is_empty():
                continue
            wide = mm.pivot(on="arm", index="month", values="v").sort("month")
            ref = wide[reference].to_numpy()
            lagged = wide["lagged-A0"].to_numpy()
            gap = ref - lagged
            mde = MDE_MULTIPLIER * gap.std(ddof=1) / np.sqrt(len(gap)) if len(gap) > 1 else np.nan
            for arm in (*CANDIDATES, *CONTROLS):
                d = wide[arm].to_numpy() - ref
                lo, hi = boot(values=d, level=0.95)
                lob, hib = boot(values=d, level=level)
                loo = [float(np.delete(d, i).mean()) for i in range(len(d))]
                rows.append(
                    {
                        "group": group, "rowset": rowset, "col": col, "arm": arm,
                        "months": len(d), "arm_score": float(wide[arm].mean()),
                        "bmix_score": float(ref.mean()), "reference": reference, "diff": float(d.mean()),
                        "low95": lo, "high95": hi, "low_bonf": lob, "high_bonf": hib,
                        "negative_months": int((d < 0).sum()), "loo_min": min(loo), "loo_max": max(loo),
                        "mde": float(mde), "candidate_arm": arm in CANDIDATES,
                    }
                )  # fmt: skip
    return pl.DataFrame(rows)


def drop_largest(*, events: pl.DataFrame) -> pl.DataFrame:
    """Return arm-minus-B-mix on event rows with the largest event of each month dropped."""
    out = []
    key = ["month", "seed", "group", "primary", "target", "origin"]
    ref = events.filter(pl.col("arm") == REFERENCE).select([*key, "cluster", ref_col := "crps_nra1.0"])
    ref = ref.rename({ref_col: "ref"})
    for arm in (*CANDIDATES, *CONTROLS):
        arm_df = events.filter(pl.col("arm") == arm).select([*key, "crps_nra1.0"])
        both = ref.join(arm_df, on=key, how="inner").with_columns(diff=pl.col("crps_nra1.0") - pl.col("ref"))
        for group in (0, 1):
            part = both.filter((pl.col("group") == group) & pl.col("month").is_in(SCORED))
            vals = []
            for _, mpart in part.group_by("month", "seed"):
                tot = mpart.group_by("cluster").agg(pl.col("diff").sum())
                worst = tot.sort(tot["diff"].abs(), descending=True)["cluster"][0]
                kept = mpart.filter(pl.col("cluster") != worst)
                if kept.height and mpart["cluster"].n_unique() > 1:
                    vals.append((mpart["month"][0], float(kept["diff"].mean())))
            if not vals:
                continue
            per_month = pl.DataFrame(vals, schema=["month", "v"], orient="row").group_by("month").agg(pl.col("v").mean())
            v = per_month["v"].to_numpy()
            lo, hi = boot(values=v, level=0.95)
            out.append({
                "arm": arm, "group": core.GROUPS[group], "months": len(v), "diff_drop_largest": float(v.mean()),
                "low95": lo, "high95": hi,
            })  # fmt: skip
    return pl.DataFrame(out)


def main() -> None:
    """Write the contrast, drop-largest, weight, and tail tables and print them."""
    summary = pl.read_parquet(core.OUT_DIR / "summary.parquet")
    events = pl.read_parquet(core.OUT_DIR / "events.parquet")
    pl.Config.set_tbl_rows(400)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_fmt_str_lengths(40)
    table = contrasts(summary=summary)
    table.write_csv(core.OUT_DIR / "contrasts.csv", float_precision=3)
    # Same-algebra reference: the stored B-mix is a Gaussian-mixture quantile set, so its tails differ
    # slightly from the piecewise-linear mixtures of the arms; "half-A0" is the 50/50 mixture in the arms' algebra.
    contrasts(summary=summary, reference="half-A0").write_csv(
        core.OUT_DIR / "contrasts_vs_half_a0.csv", float_precision=3
    )
    contrasts(summary=summary, reference="half-S1").write_csv(
        core.OUT_DIR / "contrasts_vs_half_s1.csv", float_precision=3
    )
    dropped = drop_largest(events=events)
    dropped.write_csv(core.OUT_DIR / "contrasts_drop_largest.csv", float_precision=3)
    weights = pl.read_parquet(core.OUT_DIR / "weights.parquet")
    weights.write_csv(core.OUT_DIR / "weights.csv", float_precision=3)
    nolabel = (
        summary.filter(
            (pl.col("rowset") == "nolabel") & (pl.col("band") == "all") & pl.col("month").is_in(SCORED)
        )
        .group_by("arm", "group")
        .agg(
            pl.col("crps").mean(), pl.col("n").sum(),
            *[pl.col(f"{k}_{tag}").mean() for k in ("cov_q01", "cov_q05", "cov_q95", "cov_q99", "pin_q95", "pin_q99") for tag in ("raw", "p")],
        )  # fmt: skip
        .sort("group", "arm")
    )
    nolabel.write_csv(core.OUT_DIR / "tails_nolabel.csv", float_precision=3)
    bands = (
        summary.filter(
            (pl.col("rowset") == "nolabel") & (pl.col("band") != "all") & pl.col("month").is_in(SCORED)
        )
        .group_by("arm", "group", "band")
        .agg(pl.col("crps").mean(), pl.col("n").sum())
        .sort("group", "band", "arm")
    )
    bands.write_csv(core.OUT_DIR / "bands_nolabel.csv", float_precision=3)
    print(table.filter(pl.col("col") == "crps_nra1.0"))
    print(table.filter((pl.col("rowset") == "nolabel") & (pl.col("col") == "crps")))
    print(dropped)
    print(weights.filter(pl.col("rule").str.starts_with("b-")).sort("rule", "group", "band", "month"))


if __name__ == "__main__":
    main()
