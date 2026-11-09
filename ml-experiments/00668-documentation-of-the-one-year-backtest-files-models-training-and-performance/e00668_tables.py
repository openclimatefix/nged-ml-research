"""Render the scores of one demand node as markdown tables for the experiment 668 page."""

import sys
from pathlib import Path

import polars as pl

FOLDER = Path(
    "/home/jack/dev/nged-substation-forecast/data/ml-experiments/backtest_year_2025-11_to_2026-10"
)


def table(df: pl.DataFrame, slice_name: str, metric: str) -> str:
    """One table: a row per group, columns for the three systems and the two gains."""
    d = df.filter((pl.col("slice") == slice_name) & (pl.col("metric") == metric))
    lines = [
        "| Group | Half-hours | Model | Manual heuristic | Climatology | Gain over heuristic (%) | Gain over climatology (%) |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for g in d["group"].unique(maintain_order=True):
        s = d.filter(pl.col("group") == g)
        m = s.filter(pl.col("system") == "Model").row(0, named=True)
        h = s.filter(pl.col("system") == "Manual heuristic").row(0, named=True)
        c = s.filter(pl.col("system") == "Climatology").row(0, named=True)
        lines.append(
            f"| {g} | {m['rows']:,} | {m['mean']:.2f} | {h['mean']:.2f} | {c['mean']:.2f} "
            f"| {h['gain_pct']:.1f} ({h['gain_lo']:.1f} to {h['gain_hi']:.1f}) "
            f"| {c['gain_pct']:.1f} ({c['gain_lo']:.1f} to {c['gain_hi']:.1f}) |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    name, slice_name, metric = sys.argv[1:4]
    print(table(pl.read_csv(FOLDER / "scores_668" / name / "scores.csv"), slice_name, metric))
