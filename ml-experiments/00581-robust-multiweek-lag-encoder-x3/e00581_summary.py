# ruff: noqa: E501, ANN001
"""Experiment 581: summarise the tail metrics of the injected bank (per arm, group, phase)."""

import sys

import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00581-robust-multiweek-lag-encoder-x3"
SITE_MAP = (
    ML_EXPERIMENTS_DATA_DIR
    / "00521-lag-advantage-by-lead-time-k2-s1-vs-daily-lag-free/site_map.csv"
)
PHASE_ORDER = {"before": 0, "d0-3": 1, "d4-end": 2, "after0-7": 3, "after7-14": 4}


def injected(*, arms: list[str], band: str = "24-120h", lengths=(3.0, 7.0, 15.0)) -> pl.DataFrame:
    """Return the mean of the event-level metrics by arm, metering group, and phase."""
    sm = pl.read_csv(SITE_MAP).select("primary", "group")
    t = pl.concat(
        [
            pl.read_parquet(ML_EXPERIMENTS_DATA_DIR / NAME / "tails" / f"events_{a}.parquet")
            for a in arms
        ]
    ).join(sm, on="primary")
    sel = t.filter(
        (pl.col("band") == band)
        & pl.col("length_days").is_in(list(lengths))
        & (pl.col("n") >= 20)
        & pl.col("group").is_in(["MVA-only eight", "signed-MW six"])
    )
    g = sel.group_by("arm", "group", "phase").agg(
        events=pl.len(), follow_q50=pl.col("follow_q50").mean(), follow_q95=pl.col("follow_q95").mean(),
        follow_q05=pl.col("follow_q05").mean(), follow_q99=pl.col("follow_q99").mean(), follow_q01=pl.col("follow_q01").mean(),
        cov90=pl.col("cov90").mean(), cov98=pl.col("cov98").mean(), nra_side95=pl.col("nra_side95").mean(),
        nra_side99=pl.col("nra_side99").mean(), width90=pl.col("width90").mean(), width90_twin=pl.col("width90_twin").mean(),
    )  # fmt: skip
    return (
        g.with_columns(o=pl.col("phase").replace_strict(PHASE_ORDER))
        .sort("group", "o", "arm")
        .drop("o")
    )


if __name__ == "__main__":
    arms = sys.argv[1:] or ["v3d5", "x3med"]
    with pl.Config(tbl_rows=80, tbl_cols=20, tbl_width_chars=220):
        print(injected(arms=arms).with_columns(pl.col(pl.Float64).round(3)))
