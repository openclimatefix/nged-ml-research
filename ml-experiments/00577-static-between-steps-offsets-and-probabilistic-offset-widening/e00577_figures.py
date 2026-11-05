"""Experiment 577: the headline contrast figure and the example forecasts.

Reads `<tag>_contrasts.csv` (written by `e00577_score.py`) and the assembled forecasts, and writes
`contrasts_vs_d5.svg` and `example_forecasts.svg` into the experiment's docs folder.

Usage: python e00577_figures.py [--tag final]
"""

import argparse
from pathlib import Path
from typing import Final

import e00454_score as score454  # ty: ignore[unresolved-import]
import e00470_build as build470  # ty: ignore[unresolved-import]
import e00565_design6 as d6  # ty: ignore[unresolved-import]
import matplotlib as mpl
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

mpl.use("Agg")
import matplotlib.pyplot as plt

NAME: Final = "00577-static-between-steps-offsets-and-probabilistic-offset-widening"
W: Final[Path] = ML_EXPERIMENTS_DATA_DIR / NAME / "work"
DOCS: Final[Path] = Path(__file__).parents[2] / "docs" / "ml-experiments" / NAME
COLOURS: Final = {"eight": "#306BFF", "six": "#FF4901"}
ORDER: Final = ("O1", "O2", "O3", "Q1", "Q2", "Q3", "Qconst", "Q4-O2", "Q5", "Q5shuf")
EXAMPLE_PRIMARY: Final = 0


def contrasts_figure(*, tag: str) -> None:
    """Draw each arm's CRPS minus D5's at 24-120 h, with the Bonferroni month-block interval."""
    t = pl.read_csv(W / "scores" / f"{tag}_contrasts.csv").filter(
        (pl.col("kind") == "all") & (pl.col("band") == "24-120h") & (pl.col("metric") == "crps")
    )
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for off, group in ((-0.15, "eight"), (0.15, "six")):
        for i, arm in enumerate(a for a in ORDER if a in t["arm"].to_list()):
            r = t.filter((pl.col("set") == group) & (pl.col("arm") == arm)).row(0, named=True)
            ax.errorbar(
                i + off,
                r["difference"],
                yerr=[
                    [r["difference"] - r["low_bonferroni_month"]],
                    [r["high_bonferroni_month"] - r["difference"]],
                ],
                fmt="o",
                color=COLOURS[group],
                capsize=2,
                label=f"{group} Primaries" if i == 0 else None,
            )
    arms = [a for a in ORDER if a in t["arm"].to_list()]
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_xticks(range(len(arms)), arms, fontsize=8)
    ax.set_ylabel("CRPS minus D5's, points of capacity\n(negative is better)", fontsize=8)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(DOCS / "contrasts_vs_d5.svg")


def example_figure() -> None:
    """Draw one Primary's forecasts from one origin for D5 and Q3 against the actual."""
    sites = build470.site_map()
    month = "2025-10"
    base = score454.month_base(month=month).filter(pl.col("primary") == EXAMPLE_PRIMARY)
    origin = int(base["origin"].unique().sort()[10])
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 5.4))
    for arm, colour, folder in (
        ("D5", "#306BFF", d6.W / "seed_preds" / "D5"),
        ("Q3", "#FF4901", W / "seed_preds" / "Q3"),
    ):
        f = build470.pool_stored(folder=folder, month=month, sites=sites, seeds=(0,))
        f = (
            f.join(base.select(*d6.KEYS, "lead_hours", "actual"), on=d6.KEYS, how="inner")
            .filter(pl.col("origin") == origin)
            .sort("target")
        )
        for ax, window in zip(axes, ((24.0, 48.0), (0.0, 1e9)), strict=True):
            g = f.filter((pl.col("lead_hours") >= window[0]) & (pl.col("lead_hours") < window[1]))
            x = g["lead_hours"].to_numpy()
            ax.fill_between(
                x,
                100 * g["q05"].to_numpy(),
                100 * g["q95"].to_numpy(),
                color=colour,
                alpha=0.25,
                lw=0,
                label=f"{arm} 5% to 95%",
            )
            ax.plot(x, 100 * g["q50"].to_numpy(), color=colour, lw=1.2)
            ax.plot(
                x,
                100 * g["actual"].to_numpy(),
                color="black",
                lw=1.0,
                label="actual" if arm == "D5" else None,
            )
    axes[0].set_title("day-ahead (24-48 h lead)", fontsize=9)
    axes[1].set_title("every lead hour stored for this origin", fontsize=9)
    for ax in axes:
        ax.set_ylabel("power, % of capacity", fontsize=8)
        ax.tick_params(labelsize=8)
    axes[1].set_xlabel("lead time, hours", fontsize=8)
    axes[0].legend(frameon=False, fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(DOCS / "example_forecasts.svg")


def main() -> None:
    """Draw the figures."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="final")
    args = parser.parse_args()
    contrasts_figure(tag=args.tag)
    example_figure()


if __name__ == "__main__":
    main()
