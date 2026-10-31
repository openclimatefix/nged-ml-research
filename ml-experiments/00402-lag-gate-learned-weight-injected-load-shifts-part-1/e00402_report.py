"""Experiment 402 step 4 and 5 report: mean exact CRPS and mean weight per slice and group.

DATA_PATH_INTERNAL=... uv run --no-sync python ml-experiments/00402-*/e00402_report.py \
    --scores gate_scores_r1.parquet
"""

import argparse
from pathlib import Path
from typing import Final

import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00402-lag-gate-learned-weight-injected-load-shifts-part-1" / "work"
)
ARMS: Final[tuple[str, ...]] = (
    "lags",
    "lagless",
    "fixed 50/50",
    "const (injected, per group)",
    "const (real, per group)",
    "gate single (group)",
    "gate own (unit)",
    "gate full (group)",
    "gate flag (group)",
    "gate real (group)",
)
SLICES_INJECTED: Final[dict[str, pl.Expr]] = {
    "harm (a lag touches an injected event, target outside)": (pl.col("lags_in_event") > 0)
    & ~pl.col("target_in_event"),
    "target inside an injected event": pl.col("target_in_event"),
    "clean (no lag and not the target in an injected event)": (pl.col("lags_in_event") == 0)
    & ~pl.col("target_in_event"),
}
SLICES_REAL: Final[dict[str, pl.Expr]] = {
    "harm (a lag touches a labelled event, target outside)": (pl.col("lags_real") > 0)
    & ~pl.col("target_real"),
    "target inside a labelled event": pl.col("target_real"),
    "clean (no lag in a labelled event, target outside, not within 7 days after)": (
        pl.col("lags_real") == 0
    )
    & ~pl.col("target_real")
    & ~pl.col("after_real"),
}


def main() -> None:
    """Print one table per kind, group and slice."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True)
    args = parser.parse_args()
    d = pl.read_parquet(WORK / args.scores)
    for kind, slices in (("injected", SLICES_INJECTED), ("real", SLICES_REAL)):
        k = d.filter(pl.col("kind") == kind)
        for group in ("six", "eight"):
            g = k.filter(pl.col("group") == group)
            for name, expr in slices.items():
                s = g.filter(expr)
                arms = [a for a in ARMS if s[f"crps::{a}"].null_count() < s.height]
                row = {a: round(float(s[f"crps::{a}"].mean()), 3) for a in arms}
                w = {
                    a: round(float(s[f"w::{a}"].mean()), 2)
                    for a in arms
                    if f"w::{a}" in s.columns and a.startswith("gate")
                }
                print(f"\n{kind} | {group} | {name} | n={s.height}")
                print(" CRPS:", row)
                print(" mean weight on lagged forecast:", w)


if __name__ == "__main__":
    main()
