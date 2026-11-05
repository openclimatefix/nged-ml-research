# ruff: noqa: E501, D103
"""Experiment 581: print every table the write-up quotes (CRPS, contrasts, injected bank, real events, calibration)."""

import e00581_summary as sm_
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

W = ML_EXPERIMENTS_DATA_DIR / "00581-robust-multiweek-lag-encoder-x3"
ARMS = ["v3d5", "x3med", "placebo", "x335", "x3syn", "v3syn"]
PHASES = sm_.PHASE_ORDER


def show(title: str, df: pl.DataFrame) -> None:
    print(f"\n## {title}")
    with pl.Config(tbl_rows=100, tbl_cols=30, tbl_width_chars=250):
        print(df.with_columns(pl.col(pl.Float64, pl.Float32).round(3)))


def main() -> None:
    a = pl.read_csv(W / "work/scores/part1_absolute.csv").filter(
        (pl.col("slice") == "all rows")
        & pl.col("set").is_in(["eight", "six"])
        & (pl.col("band") != "all")
    )
    show(
        "Mean CRPS (points of capacity), all rows, by group and lead band",
        a.pivot(on="arm", index=["set", "band"], values="mean_single_seed_crps"),
    )
    c = pl.read_csv(W / "work/scores/part1_contrasts.csv").filter(
        (pl.col("slice") == "all rows")
        & pl.col("set").is_in(["eight", "six"])
        & pl.col("band").is_in(["0-24h", "24-120h"])
        & (pl.col("b") == "V3D5")
    )
    show(
        "CRPS contrasts against v3+D5 (positive: worse); Bonferroni interval over months",
        c.select(
            "set",
            "band",
            "a",
            "counted",
            "difference",
            "low_bonferroni_month",
            "high_bonferroni_month",
            "mde",
            "folds_positive",
        ).sort("a", "set", "band"),
    )
    show("Injected bank, 24-120 h, events of 3, 7, 15 days, by phase", sm_.injected(arms=ARMS))
    sm = pl.read_csv(sm_.SITE_MAP).select("primary", "group")
    t = (
        pl.concat([pl.read_parquet(W / f"tails/events_{x}.parquet") for x in ARMS])
        .join(sm, on="primary")
        .filter(
            (pl.col("band") == "24-120h")
            & (pl.col("n") >= 20)
            & pl.col("group").is_in(["MVA-only eight", "signed-MW six"])
        )
    )
    for key, ph in (("length_days", "d4-end"), ("length_days", "after0-7"), ("depth", "d4-end")):
        show(
            f"Median follow ratio by {key}, phase {ph}, both groups",
            t.filter(pl.col("phase") == ph)
            .group_by(key, "arm")
            .agg(f=pl.col("follow_q50").mean())
            .pivot(on="arm", index=key, values="f")
            .sort(key),
        )
    show(
        "30-day events: median follow ratio by phase",
        t.filter(pl.col("length_days") == 30)
        .group_by("phase", "arm")
        .agg(f=pl.col("follow_q50").mean())
        .pivot(on="arm", index="phase", values="f"),
    )
    r = pl.concat([pl.read_parquet(W / f"tails/real_{x}.parquet") for x in ARMS]).filter(
        (pl.col("band") == "24-120h") & (pl.col("n") >= 20)
    )
    show(
        "Verified transfer 23 (both Primaries, 2025-12 and 2026-01 folds), 24-120 h",
        r.filter(pl.col("transfer_id") == 23)
        .group_by("arm", "phase")
        .agg(
            n=pl.col("n").sum(),
            follow_q50=pl.col("follow_q50").mean(),
            nra_side95=pl.col("nra_side95").mean(),
            nra_side99=pl.col("nra_side99").mean(),
            cov90=pl.col("cov90").mean(),
            width90=pl.col("width90").mean(),
        )
        .with_columns(o=pl.col("phase").replace_strict(PHASES))
        .sort("o", "arm")
        .drop("o"),
    )
    show(
        "Verified transfers 24-26 (May 2026), 7-14 days after the end, June fold",
        r.filter((pl.col("transfer_id") > 23) & (pl.col("month") == "2026-06"))
        .group_by("arm")
        .agg(
            n=pl.col("n").sum(),
            follow_q50=pl.col("follow_q50").mean(),
            nra_side95=pl.col("nra_side95").mean(),
            cov90=pl.col("cov90").mean(),
        ),
    )
    show(
        "Verified transfers 24-26 in the 2026-05 fold (v3+D5 and X3-med only)",
        r.filter((pl.col("transfer_id") > 23) & (pl.col("month") == "2026-05"))
        .group_by("arm", "phase")
        .agg(
            n=pl.col("n").sum(),
            follow_q50=pl.col("follow_q50").mean(),
            nra_side95=pl.col("nra_side95").mean(),
            cov90=pl.col("cov90").mean(),
        )
        .with_columns(o=pl.col("phase").replace_strict(PHASES))
        .sort("o", "arm")
        .drop("o"),
    )
    f = pl.concat([pl.read_parquet(W / f"tails/free_{x}.parquet") for x in ARMS]).filter(
        (pl.col("band") == "24-120h") & pl.col("group").is_in(["MVA-only eight", "signed-MW six"])
    )
    show(
        "Rows with no labelled or verified event within 7 days, 24-120 h: calibration, pinball, width",
        f.select(
            "arm",
            "group",
            "n",
            "cov90",
            "cov98",
            "width90",
            "below_q01",
            "below_q05",
            "below_q95",
            "below_q99",
            "pin_q01",
            "pin_q05",
            "pin_q95",
            "pin_q99",
        ).sort("group", "arm"),
    )


if __name__ == "__main__":
    main()
