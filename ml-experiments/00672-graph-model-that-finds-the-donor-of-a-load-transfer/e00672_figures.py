"""Draw the figures of experiment 672 from the stored scored rows.

Run with the environment given in `e00672_run.py`: `uv run --no-sync python e00672_figures.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from e00665_score import CELLS  # ty: ignore[unresolved-import]
from e00672_report import load_670, load_mine, per_world

DOCS: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00672-graph-model-that-finds-the-donor-of-a-load-transfer"
)
BLUE, ORANGE, SKY, PURPLE = "#306BFF", "#FF4901", "#10C5F7", "#B701FF"
WIDTH: Final[float] = 6.8
N_BOOT: Final[int] = 1000


def share_and_interval(
    *, frames: list[pl.DataFrame], cell: str, measure: str, worlds: list[int]
) -> tuple[float, float, float]:
    """Mean over seeds of the megawatt-weighted share, and a 95% interval over whole worlds."""
    tabs = [per_world(df=f, cell=cell, measure=measure, worlds=worlds) for f in frames]
    rng = np.random.default_rng(672)
    n = len(worlds)

    def one(draw: np.ndarray) -> float:
        vals = [t[draw, 0].sum() / t[draw, 1].sum() for t in tabs if t[draw, 1].sum() > 0]
        return float(np.mean(vals)) if vals else float("nan")

    point = one(np.arange(n))
    boot = [one(rng.integers(0, n, n)) for _ in range(N_BOOT)]
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return point, float(lo), float(hi)


def figure(
    *,
    arms: list[tuple[str, str, list[pl.DataFrame]]],
    panels: list[tuple[str, str]],
    name: str,
    note: str,
) -> None:
    """Grouped bars with intervals, one panel per measure, six cells on the x axis."""
    ref = load_670()
    worlds = sorted(ref["world"].unique().to_list())
    fig, axes = plt.subplots(len(panels), 1, figsize=(WIDTH, 3.1 * len(panels) + 0.6))
    axes = np.atleast_1d(axes)
    x = np.arange(len(CELLS))
    k = len(arms)
    for ax, (measure, title) in zip(axes, panels, strict=True):
        for i, (label, colour, frames) in enumerate(arms):
            res = [
                share_and_interval(frames=frames, cell=c, measure=measure, worlds=worlds)
                for c in CELLS
            ]
            v = np.array([r[0] for r in res])
            lo = np.array([r[1] for r in res])
            hi = np.array([r[2] for r in res])
            pos = x + (i - (k - 1) / 2) * 0.8 / k
            ax.bar(pos, v, 0.8 / k, color=colour, label=label)
            ax.errorbar(pos, v, yerr=[v - lo, hi - v], fmt="none", ecolor="k", lw=1)
        ax.set_ylim(0, 1)
        ax.set_title(title + " (larger is better)", fontsize=9, loc="left")
        ax.set_xticks(x, CELLS, fontsize=7.5)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.text(0.01, 0.005, note, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(DOCS / name)
    plt.close(fig)


def main() -> None:
    """Draw both figures."""
    ref = load_670()
    e1 = [ref.filter(pl.col("method") == "E1")]
    e1j = [ref.filter(pl.col("method") == "E1-J")]
    note = (
        "Bars: estimate (G arms: mean of 3 networks); black line: 95% interval from resampling\n"
        "the 40 worlds. Deep: 8 noise units or more. Megawatt-weighted. Default worlds, from 2024."
    )
    figure(
        arms=[("E1", SKY, e1), ("E1-J", BLUE, e1j), ("G", ORANGE, load_mine(arm="G"))],
        panels=[
            ("paired", "Share of injected edge-transfer megawatts paired"),
            ("donor_found", "Share of injected donor megawatts found"),
        ],
        name="figure_paired_and_donor_found.svg",
        note=note,
    )
    figure(
        arms=[
            ("E1-J", BLUE, e1j),
            ("G", ORANGE, load_mine(arm="G")),
            ("G-0 (no links)", PURPLE, load_mine(arm="G0")),
        ],
        panels=[("found", "Share of injected megawatts found (any detection)")],
        name="figure_found_by_arm.svg",
        note=note,
    )


if __name__ == "__main__":
    main()
