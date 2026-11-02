"""Horizon-band contrasts and the headline figure of experiment 517 (after `e00517_score.py rows`).

Prints, per group (the eight, the six) and horizon band (day 1, days 2 to 3, days 4 to 10, days 1 to
10), each arm's mean CRPS and each contrast's pooled difference with its per-fold values. Five
folds give a min-to-max range, not a confidence interval.
"""

import e00470_score as score470  # ty: ignore[unresolved-import]
import e00517_score as s
import matplotlib as mpl
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

mpl.use("Agg")
import matplotlib.pyplot as plt

OUT = ML_EXPERIMENTS_DATA_DIR / "00517-lagless-demand-network-lockdown-flag-removal" / "work"
BANDS = {"day 1": (0, 24), "days 2-3": (24, 72), "days 4-10": (72, 241), "days 1-10": (0, 241)}
CONTRASTS = (("C1", "C0"), ("C2", "C0"), ("C2", "C1"))


def main() -> None:
    """Print the band table and save the figure."""
    s.configure()
    table = score470.load_rows().filter(pl.col("main16"))
    table = table.with_columns(lead_h=(pl.col("target") - pl.col("origin")) / 2.0)
    rows = []
    for group, label in (("mva_eight", "eight"), ("signed_six", "six")):
        part = table.filter(pl.col("group") == group)
        if group == "signed_six":
            part = part.filter(pl.col("cls") == "clean")
        for band, (lo, hi) in BANDS.items():
            sel = part.filter((pl.col("lead_h") >= lo) & (pl.col("lead_h") < hi))
            means = {a: sel[f"{a}__crps"].mean() for a in s.ARMS}
            per_fold = {
                a: {m: sel.filter(pl.col("month") == m)[f"{a}__crps"].mean() for m in s.FIVE}
                for a in s.ARMS
            }
            for x, y in CONTRASTS:
                folds = np.array([per_fold[x][m] - per_fold[y][m] for m in s.FIVE])
                rows.append(
                    {"group": label, "band": band, "contrast": f"{x} - {y}",
                     "diff": means[x] - means[y], "fold_min": folds.min(), "fold_max": folds.max(),
                     "n_fold_neg": int((folds < 0).sum()),
                     **{f"crps_{a}": means[a] for a in s.ARMS}}
                )  # fmt: skip
    out = pl.DataFrame(rows)
    out.write_csv(OUT / "scores" / "horizon_bands.csv")
    with pl.Config(tbl_rows=60, tbl_cols=12, float_precision=3):
        print(out)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    colours = {"C1 - C0": "#4c9be8", "C2 - C0": "#f4a261", "C2 - C1": "#2a9d8f"}
    for ax, label in zip(axes, ("eight", "six"), strict=True):
        for i, (x, y) in enumerate(CONTRASTS):
            d = out.filter((pl.col("group") == label) & (pl.col("contrast") == f"{x} - {y}"))
            xs = np.arange(len(BANDS)) + (i - 1) * 0.25
            ax.errorbar(
                xs, d["diff"], yerr=[d["diff"] - d["fold_min"], d["fold_max"] - d["diff"]],
                fmt="o", color=colours[f"{x} - {y}"], label=f"{x} minus {y}", capsize=3,
            )  # fmt: skip
        ax.axhline(0, color="#555555", lw=0.8)
        ax.set_xticks(range(len(BANDS)), list(BANDS), rotation=20)
        ax.set_title(f"The {label}")
    axes[0].set_ylabel("CRPS difference (points of capacity;\nnegative favours the first arm)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.patch.set_facecolor("#fffbf5")
    for ax in axes:
        ax.set_facecolor("#fffbf5")
    fig.tight_layout()
    fig.savefig(OUT / "scores" / "horizon_bands.svg")


if __name__ == "__main__":
    main()
