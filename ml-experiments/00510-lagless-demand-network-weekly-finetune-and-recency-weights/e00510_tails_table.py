# ruff: noqa: E501
"""Aggregate experiment 510's per-month tail cells into the tables of the write-up (row-weighted)."""

from pathlib import Path
from typing import Final

import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SCORES: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00510-lagless-demand-network-weekly-finetune-and-recency-weights/work/scores"
)
METRICS: Final[tuple[str, ...]] = (
    "crps",
    "cov_q01",
    "pin_q01",
    "cov_q05",
    "pin_q05",
    "cov_q95",
    "pin_q95",
    "cov_q99",
    "pin_q99",
)


def main() -> None:
    """Print row-weighted means by arm, group, class, basis, and widening variant."""
    t = pl.read_parquet(SCORES / "tails_cells.parquet")
    agg = t.group_by("variant", "group", "cls", "basis", "arm").agg(
        n=pl.col("n").sum(),
        **{m: (pl.col(m) * pl.col("n")).sum() / pl.col("n").sum() for m in METRICS},
    )
    agg.write_csv(SCORES / "tails_by_arm.csv")
    with pl.Config(tbl_rows=200, tbl_width_chars=200):
        head = agg.filter(
            (pl.col("variant") == "w1.15")
            & (pl.col("basis") == "metered")
            & (pl.col("cls") == "clean")
        ).sort("group", "arm")
        print(head.with_columns(pl.col(pl.Float64).round(3)))
        ev = agg.filter(
            (pl.col("variant") == "w1.15")
            & (pl.col("group") == "signed_six")
            & (pl.col("cls") == "event")
        ).sort("basis", "arm")
        print(
            ev.select("basis", "arm", "n", "crps", "cov_q95", "pin_q99", "cov_q99").with_columns(
                pl.col(pl.Float64).round(3)
            )
        )
        sweep = agg.filter(
            (pl.col("cls") == "clean")
            & (pl.col("basis") == "metered")
            & pl.col("arm").is_in(["L-W", "L-Ww", "L-Ww-O"])
        ).sort("group", "arm", "variant")
        print(
            sweep.select(
                "group", "arm", "variant", "crps", "cov_q95", "cov_q99", "pin_q99"
            ).with_columns(pl.col(pl.Float64).round(3))
        )


if __name__ == "__main__":
    main()
