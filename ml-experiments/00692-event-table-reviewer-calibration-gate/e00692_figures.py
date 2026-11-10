"""Experiment 692: the write-up's figure and the two example figures.

uv run --no-sync python e00692_figures.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl
from e00692_score import OUT, load_answers, wilson  # ty: ignore[unresolved-import]

DOCS = Path(__file__).parents[2] / "docs/ml-experiments/00692-event-table-reviewer-calibration-gate"


def main() -> None:
    """Draw the share of figures the reviewer called a transfer, by kind of candidate."""
    key = pl.read_parquet(OUT / "answer_key_private" / "key.parquet")
    cand = pl.read_parquet(OUT / "answer_key_private" / "candidates.parquet").filter(
        pl.col("kind") == "sym"
    )
    rec = (
        cand.filter(~pl.col("true"))
        .with_columns((pl.col("onset") // 48).alias("day"))
        .group_by(["donor", "recipient", "day"])
        .agg(pl.col("seed").n_unique().alias("worlds"))
        .filter(pl.col("worlds") >= 10)
        .select(["donor", "recipient", "day"])
        .with_columns(pl.lit(True).alias("recurring"))
    )
    cal = (
        key.filter(pl.col("set") == "calibration")
        .join(load_answers(OUT / "answers" / "calibration"), on="figure")
        .with_columns((pl.col("onset") // 48).alias("day"))
        .join(rec, on=["donor", "recipient", "day"], how="left")
        .with_columns(pl.col("recurring").fill_null(False))
    )

    def kind(r: dict) -> str:
        if r["stratum"] == "verified_real":
            return "Verified real transfers"
        if r["stratum"] == "true":
            return "Injected transfers found by the detector"
        if r["recurring"]:
            return "Unlabelled: same event in 10 or more worlds (real record)"
        if r["stratum"] == "hard_negative_injected_placebo":
            return "Injected look-alikes (placebos, solo shifts)"
        return "Other false candidates"

    cal = cal.with_columns(
        pl.Series("kind", [kind(r) for r in cal.to_dicts()]),
        pl.col("verdict").eq("transfer").alias("called"),
    )
    tab = cal.group_by("kind").agg(pl.len().alias("n"), pl.col("called").sum().alias("k"))
    order = [
        "Injected transfers found by the detector",
        "Verified real transfers",
        "Unlabelled: same event in 10 or more worlds (real record)",
        "Other false candidates",
        "Injected look-alikes (placebos, solo shifts)",
    ]
    rows = {r["kind"]: r for r in tab.to_dicts()}
    fig, ax = plt.subplots(figsize=(8.6, 3.4), dpi=110)
    colours = ["#1f77b4", "#1f77b4", "#d95f02", "#7a7a7a", "#7a7a7a"]
    for i, name in enumerate(order):
        r = rows[name]
        lo, hi = wilson(int(r["k"]), int(r["n"]))
        p = r["k"] / r["n"]
        ax.plot([lo, hi], [i, i], color=colours[i], linewidth=1)
        ax.plot([p], [i], "o", color=colours[i], markersize=5)
        ax.text(1.02, i, f"{int(r['k'])} of {int(r['n'])}", va="center", fontsize=8)
    ax.set_yticks(range(len(order)), [f"{o}" for o in order], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel(
        "Share of figures the reviewer called a transfer (point and 95% Wilson interval)",
        fontsize=8,
    )
    ax.grid(axis="x", color="#e5e5e5", linewidth=0.5)
    fig.tight_layout()
    DOCS.mkdir(parents=True, exist_ok=True)
    fig.savefig(DOCS / "call_rate_by_kind.svg")
    plt.close(fig)
    for name, src in (
        ("example_false_called_transfer", "calibration_035"),
        ("example_true_missed", "calibration_019"),
    ):
        shutil.copy(OUT / "figures" / "calibration" / f"{src}.png", DOCS / f"{name}.png")
    print(tab)


if __name__ == "__main__":
    main()
