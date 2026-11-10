"""Draw experiment 682's two figures from the stored outputs of `e00682_report.py`.

Run with the environment given in `e00682_run.py`: `uv run --no-sync python e00682_figures.py`.
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
from e00682_report import OUT

SLUG: Final[str] = "00682-event-table-reconstruction-from-edge-flows-part-2-change-cost-model"
DOCS: Final[Path] = Path(__file__).resolve().parents[2] / "docs/ml-experiments" / SLUG
ARMS: Final[tuple[tuple[str, str], ...]] = (
    ("E1-J", "#306BFF"),
    ("E1-J-sym", "#8FB0FF"),
    ("E1-F", "#FF4901"),
    ("E1-F all pairs", "#333333"),
    ("E1-F rewired", "#9AA0A6"),
)


def paired_figure(*, res: dict) -> None:
    """Paired share per cell for the five arms."""
    cells = CELLS[:6]
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    x = np.arange(len(cells))
    w = 0.16
    for i, (name, colour) in enumerate(ARMS):
        v = np.array([res[name][c]["paired_mw"] for c in cells])
        lo = np.array([res[name][c]["interval"]["paired"][0] for c in cells])
        hi = np.array([res[name][c]["interval"]["paired"][1] for c in cells])
        ax.bar(x + (i - 2) * w, v, w, color=colour, label=name)
        ax.errorbar(x + (i - 2) * w, v, yerr=[v - lo, hi - v], fmt="none", ecolor="k", lw=0.8)
    ax.set_xticks(x, [f"{c}\n{int(res['E1-J'][c]['n_edge'])} ends" for c in cells], fontsize=7.5)
    ax.set_ylim(0, 0.8)
    ax.set_title(
        "Choosing edge flows by cost pairs no more than the symmetric search",
        fontsize=9,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=7, loc="upper right", ncol=2)
    fig.text(
        0.01,
        0.005,
        "Share of injected edge-transfer megawatts paired (larger is better); black lines: 95% "
        "interval from resampling the 40 worlds.",
        fontsize=7,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(DOCS / "figure_paired_by_cell.svg")
    plt.close(fig)


def decoded_figure(*, res: dict) -> None:
    """Decoded transfers and correct ends in 40 worlds, per arm."""
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    x = np.arange(len(ARMS))
    dec = [res[n]["decoded"] for n, _ in ARMS]
    right = [res[n]["right_ends"] for n, _ in ARMS]
    ax.bar(x - 0.2, dec, 0.4, color="#9AA0A6", label="decoded transfers")
    ax.bar(x + 0.2, right, 0.4, color="#FF4901", label="correct ends")
    for i, (d, r) in enumerate(zip(dec, right, strict=True)):
        ax.text(i - 0.2, d, str(d), ha="center", va="bottom", fontsize=7)
        ax.text(i + 0.2, r, str(r), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x, [n for n, _ in ARMS], fontsize=7.5)
    ax.set_title("Edge flows chosen by cost accept most pairs they can", fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(DOCS / "figure_decoded_transfers.svg")
    plt.close(fig)


if __name__ == "__main__":
    result = json.loads((OUT / "by_cell.json").read_text())["default|default"]
    paired_figure(res=result)
    decoded_figure(res=result)
