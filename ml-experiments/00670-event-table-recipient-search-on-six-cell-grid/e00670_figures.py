# ruff: noqa: DTZ001
"""Draw the three figures of experiment 670 from the stored outputs of `e00670_run.py`.

Run with the environment given in `e00670_run.py`: `uv run --no-sync python e00670_figures.py`.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from e00653_common import GRID_START, Panel
from e00665_score import CELLS
from e00670_report import OUT

DOCS: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00670-event-table-recipient-search-on-six-cell-grid"
)
BLUE, ORANGE, GREY = "#306BFF", "#FF4901", "#9AA0A6"
WIDTH: Final[float] = 6.8


def recall_figure() -> None:
    """Paired and found shares per cell for E1, E1-J, and E1-J on a rewired graph."""
    res = json.loads((OUT / "injected_by_cell.json").read_text())["default"]
    arms = (("E1", BLUE), ("E1-J", ORANGE), ("E1-J rewired", GREY))
    fig, axes = plt.subplots(2, 1, figsize=(WIDTH, 6.2))
    x = np.arange(len(CELLS[:6]))
    for ax, key, iv, title in (
        (axes[0], "paired_mw", "paired", "Share of injected edge-transfer megawatts paired"),
        (axes[1], "found_mw", "found", "Share of injected megawatts found (any detection)"),
    ):
        for i, (name, colour) in enumerate(arms):
            v = np.array([res[name][c][key] for c in CELLS[:6]])
            lo = np.array([res[name][c]["interval"][iv][0] for c in CELLS[:6]])
            hi = np.array([res[name][c]["interval"][iv][1] for c in CELLS[:6]])
            ax.bar(x + (i - 1) * 0.27, v, 0.27, color=colour, label=name)
            ax.errorbar(x + (i - 1) * 0.27, v, yerr=[v - lo, hi - v], fmt="none", ecolor="k", lw=1)
        ax.set_ylim(0, 1)
        ax.set_title(title + " (larger is better)", fontsize=9, loc="left")
    pred_f = {"long, deep": 0.85, "medium, deep": 0.80}
    pred_p = {"long, deep": 0.60, "medium, deep": 0.50}
    for ax, pred in ((axes[0], pred_p), (axes[1], pred_f)):
        for c, v in pred.items():
            i = CELLS.index(c)
            ax.hlines(v, i - 0.42, i + 0.42, color="k", ls="--", lw=1)
    axes[0].set_xticks(
        x, [f"{c}\n{int(res['E1'][c]['n_edge'])} ends" for c in CELLS[:6]], fontsize=7.5
    )
    axes[1].set_xticks(
        x, [f"{c}\n{int(res['E1'][c]['n_events'])} events" for c in CELLS[:6]], fontsize=7.5
    )
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.text(
        0.01,
        0.005,
        "Bars: estimate; black line: 95% interval from resampling the 40 worlds. Dashed lines: "
        "pre-registered minimum for E1-J.\nDeep: 8 noise units or more. Megawatt-weighted.",
        fontsize=7.5,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(DOCS / "figure_recall_by_cell.svg")
    plt.close(fig)


def event_series_figure() -> None:
    """D5 and D16 power with the events of E1 and E1-J."""
    panel = Panel()
    ev = {m: pl.read_parquet(OUT / f"real_events_{m}.parquet") for m in ("E1", "E1-J")}
    lo, hi = datetime(2025, 12, 15), datetime(2026, 1, 21)
    base = np.datetime64("2022-01-01T00:00")
    lo_i = int((lo - GRID_START.replace(tzinfo=None)).total_seconds() // 1800)
    hi_i = int((hi - GRID_START.replace(tzinfo=None)).total_seconds() // 1800)
    verified = {
        "D5": (datetime(2025, 12, 29, 1, 0), datetime(2026, 1, 13, 11, 30)),
        "D16": (datetime(2025, 12, 29, 16, 0), datetime(2026, 1, 13, 11, 30)),
    }
    fig, axes = plt.subplots(2, 1, figsize=(WIDTH, 6.2), sharex=True)
    for ax, site in zip(axes, ("D5", "D16"), strict=True):
        t = np.arange(lo_i, hi_i)
        ax.plot(base + t.astype("timedelta64[30m]"), panel.power[site][t], color="k", lw=0.8)
        top = float(np.nanmax(panel.power[site][lo_i:hi_i]))
        bottom = float(np.nanmin(panel.power[site][lo_i:hi_i]))
        ax.set_ylim(bottom - 0.3 * (top - bottom), top + 0.05 * (top - bottom))
        for v in verified[site]:
            ax.axvline(np.datetime64(v), color="k", ls="--", lw=1)
        for m, colour, (y0, y1) in (("E1", BLUE, (0.0, 0.06)), ("E1-J", ORANGE, (0.07, 0.13))):
            for r in ev[m].filter(pl.col("primary") == site).iter_rows(named=True):
                end = hi_i if r["end"] < 0 else r["end"]
                if r["start"] < hi_i and end > lo_i and r["units"] >= 2:
                    a = base + np.timedelta64(30 * max(r["start"], lo_i), "m")
                    b = base + np.timedelta64(30 * min(end, hi_i), "m")
                    ax.axvspan(
                        a, b, ymin=y0, ymax=y1, facecolor=colour, edgecolor="white", alpha=0.9
                    )
                    span = ax.get_ylim()[1] - ax.get_ylim()[0]
                    label = f"{r['amp_mw']:+.1f}" + (
                        f" pairs {r['partners']}" if r["partners"] else ""
                    )
                    ax.text(
                        a + (b - a) / 2,
                        ax.get_ylim()[0] + (y0 + y1) / 2 * span,
                        label,
                        fontsize=7,
                        ha="center",
                        va="center",
                        color="white",
                    )
        ax.set_ylabel(f"{site} power (MW)")
        ax.set_title(f"Primary {site}", fontsize=9, loc="left")
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    handles = [
        plt.Line2D([], [], color="k", lw=0.8, label="observed power"),
        plt.Line2D([], [], color="k", ls="--", lw=1, label="verified start and end"),
        plt.Rectangle((0, 0), 1, 1, color=BLUE, label="E1 events"),
        plt.Rectangle((0, 0), 1, 1, color=ORANGE, label="E1-J events"),
    ]
    fig.legend(handles=handles, frameon=False, fontsize=7, loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(DOCS / "figure_event_series_D5_D16.svg")
    plt.close(fig)


def placebo_figure() -> None:
    """Edge events on the real record: the real graph against 100 rewired graphs."""
    real = json.loads((OUT / "real.json").read_text())
    counts = real["rewired_edge_events"]
    fig, ax = plt.subplots(figsize=(WIDTH, 3.4))
    bins = np.arange(-0.5, max(max(counts), real["real_graph_edge_events"]) + 1.5, 1)
    ax.hist(counts, bins=bins, color=GREY, label="100 rewired graphs")
    ax.axvline(real["real_graph_edge_events"], color=ORANGE, lw=2, label="real graph")
    ax.set_xlabel("Edge events (transfers) of E1-J in 2024-01 to 2026-06")
    ax.set_ylabel("Rewired graphs")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(
        "A rewired graph gives fewer transfers than the real graph", fontsize=9, loc="left"
    )
    fig.tight_layout()
    fig.savefig(DOCS / "figure_rewired_placebo.svg")
    plt.close(fig)


if __name__ == "__main__":
    recall_figure()
    event_series_figure()
    placebo_figure()
