"""Draw the recall-by-duration figure of experiment 658 (experiments 653, first, second look)."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from e00658_report import BANDS, SEEDS, band, big, picked, wrec
from e00658_train import WORK

OUT: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00658-edge-state-event-table-neural-network"
)
BLUE, ORANGE, GREY = "#1f4e9c", "#e8730c", "#888888"


def net_rows(prefix: str) -> pl.DataFrame:
    """Default-spec rows of the network at its validation-chosen costs, all seeds."""
    parts = []
    for s in SEEDS:
        p = picked(f"{prefix}_seed{s}")
        parts.append(
            band(pl.read_parquet(WORK / f"injected_{prefix}_seed{s}.parquet")).filter(
                (pl.col("arm") == "net")
                & (pl.col("spec") == "default")
                & (pl.col("on") == p["on"])
                & (pl.col("boost") == p["boost"])
                & (pl.col("cutoff") == p["cutoff"])
            )
        )
    return pl.concat(parts)


def main() -> None:
    """Grouped bars of MW-weighted recall by duration band."""
    base = band(pl.read_parquet(WORK / "baseline_e653.parquet")).filter(pl.col("spec") == "default")
    arms = [
        ("experiment 653 (penalised segmentation)", base, GREY),
        ("network, first look", net_rows("main"), ORANGE),
        ("network, second look", net_rows("v2main"), BLUE),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(len(BANDS))
    for i, (name, df, col) in enumerate(arms):
        v = [wrec(big(df).filter(pl.col("band") == b), "found") for b in BANDS]
        ax.bar(x + (i - 1) * 0.27, np.nan_to_num(v), 0.27, color=col, label=name)
    ax.set_xticks(x, BANDS)
    ax.set_ylabel("Share of injected MW found")
    ax.set_xlabel("Event duration (events of 4 noise units or more)")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "figure_recall_by_band.svg")


if __name__ == "__main__":
    main()
