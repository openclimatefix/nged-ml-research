"""Draw experiment 680's figure from the stored outputs of `e00680_run.py` and `e00680_report.py`.

Run with the environment given in `e00680_run.py`: `uv run --no-sync python e00680_figures.py`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from e00665_score import CELLS  # ty: ignore[unresolved-import]
from e00680_report import OUT

DOCS: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00680-event-table-reconstruction-from-edge-flows-part-1-symmetric-search"
)
BLUE, ORANGE, GREY = "#306BFF", "#FF4901", "#9AA0A6"


def paired_figure() -> None:
    """Paired share per cell for E1-J, E1-J-sym, and E1-J-sym on a rewired graph."""
    res = json.loads((OUT / "injected_by_cell.json").read_text())["default"]
    arms = (("E1-J", BLUE), ("E1-J-sym", ORANGE), ("E1-J-sym rewired", GREY))
    cells = CELLS[:6]
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    x = np.arange(len(cells))
    for i, (name, colour) in enumerate(arms):
        v = np.array([res[name][c]["paired_mw"] for c in cells])
        lo = np.array([res[name][c]["interval"]["paired"][0] for c in cells])
        hi = np.array([res[name][c]["interval"]["paired"][1] for c in cells])
        ax.bar(x + (i - 1) * 0.27, v, 0.27, color=colour, label=name)
        ax.errorbar(x + (i - 1) * 0.27, v, yerr=[v - lo, hi - v], fmt="none", ecolor="k", lw=1)
    ax.set_xticks(x, [f"{c}\n{int(res['E1-J'][c]['n_edge'])} ends" for c in cells], fontsize=7.5)
    ax.set_ylim(0, 0.8)
    ax.set_title(
        "Searching from the rise as well as the fall pairs more megawatts (larger is better)",
        fontsize=9,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.text(
        0.01,
        0.005,
        "Share of injected edge-transfer megawatts paired, megawatt-weighted; black lines: 95% "
        "interval from resampling the 40 worlds.",
        fontsize=7.5,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(DOCS / "figure_paired_by_cell.svg")
    plt.close(fig)


if __name__ == "__main__":
    paired_figure()
