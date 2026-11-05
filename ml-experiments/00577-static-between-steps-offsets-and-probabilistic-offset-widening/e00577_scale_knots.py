"""Experiment 577: the learnt log-scale knots of Q5, per Primary and over time (descriptive).

Reads the last fold's (2026-06) pretrained Q5 weights, seeds 0 and 1. A scale knot is added to the
log standard deviation of every mixture component, so `exp(knot)` is the factor by which that
Primary's forecast spread is multiplied in that month. Writes `scale_knots.json` (per Primary: the
mean knot over time and over seeds, the standard deviation over time, the first-to-last change, and
the seed-to-seed correlation of the time profile) and `scale_knots.svg`.
"""

import json
from pathlib import Path
from typing import Final

import matplotlib as mpl
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

mpl.use("Agg")
import matplotlib.pyplot as plt

NAME: Final = "00577-static-between-steps-offsets-and-probabilistic-offset-widening"
W: Final[Path] = ML_EXPERIMENTS_DATA_DIR / NAME / "work"
MONTH: Final = "2026-06"
SITES: Final = pl.read_csv(next(ML_EXPERIMENTS_DATA_DIR.glob("00521*/site_map.csv")))
DOCS: Final[Path] = Path(__file__).parents[2] / "docs" / "ml-experiments" / NAME


def main() -> None:
    """Summarise and plot the scale knots."""
    ks = [
        torch.load(W / "pretrained_Q5" / "real" / f"{MONTH}_s{s}.pt", map_location="cpu")[
            "s_knots"
        ][:, :, 0].numpy()
        for s in (0, 1)
    ]
    mean = np.mean(ks, axis=0)
    labels = SITES["label"].to_list()
    rows = {}
    for p, lab in enumerate(labels):
        rows[lab] = {
            "mean_knot": float(mean[p].mean()),
            "sd_over_time": float(mean[p].std()),
            "last_minus_first": float(mean[p, -1] - mean[p, 0]),
            "seed_correlation": float(np.corrcoef(ks[0][p], ks[1][p])[0, 1]),
        }
    (W / "scale_knots.json").write_text(json.dumps(rows, indent=1))
    for lab, r in rows.items():
        print(lab, {k: round(v, 3) for k, v in r.items()})
    fig, ax = plt.subplots(2, 1, figsize=(7.5, 5.6), sharex=True)
    for a, group in zip(ax, ("mva_eight", "signed_six_and_report"), strict=True):
        for p, lab in enumerate(labels):
            is_mva = SITES["group"][p] == "MVA-only eight"
            if is_mva == (group == "mva_eight"):
                a.plot(np.arange(mean.shape[1]), np.exp(mean[p]), lw=1.2, label=lab)
        a.set_ylabel("spread factor, exp(knot)", fontsize=8)
        a.legend(ncol=4, fontsize=7, frameon=False)
        a.tick_params(labelsize=8)
    ax[0].set_title("eight MVA Primaries", fontsize=9)
    ax[1].set_title("six MW Primaries and the two report-only Primaries", fontsize=9)
    ax[1].set_xlabel("months since 2019-09", fontsize=8)
    fig.tight_layout()
    fig.savefig(DOCS / "scale_knots.svg")


if __name__ == "__main__":
    main()
