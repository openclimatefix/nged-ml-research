# ruff: noqa: DTZ001
"""Draw the three figures of experiment 665 from the stored JSON of `e00665_score.py`.

Run from this folder with `DATA_PATH_INTERNAL` set to the data folder and `PYTHONPATH` set to
the folder of experiment 653 and this folder, using
`uv run --no-sync python e00665_figures.py`.
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
from e00653_common import GRID_START, WORK, Panel
from e00665_score import CELLS, E658, OUT, SEEDS

DOCS: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00665-event-table-recall-by-length-and-depth"
)
BLUE, ORANGE, SKY, PURPLE = "#306BFF", "#FF4901", "#10C5F7", "#B701FF"
WIDTH: Final[float] = 6.8
"""Figure width in inches; at 100 dpi it is the 680 pixels of the docs text column."""


def recall_figure() -> None:
    """Found and paired shares per cell, with intervals."""
    res = json.loads((OUT / "injected_by_cell.json").read_text())
    fig, axes = plt.subplots(2, 1, figsize=(WIDTH, 6.0), sharex=False)
    x = np.arange(len(CELLS))
    for ax, key, iv_key, title in (
        (axes[0], "found_mw", "found", "Share of injected megawatts found (larger is better)"),
        (
            axes[1],
            "paired_mw",
            "paired",
            "Share of injected edge-transfer ends paired correctly (larger is better)",
        ),
    ):
        for i, (name, colour) in enumerate((("653 default", BLUE), ("658 second look", ORANGE))):
            v = np.array([res[name][c][key] for c in CELLS])
            lo = np.array([res[name][c]["interval"][iv_key][0] for c in CELLS])
            hi = np.array([res[name][c]["interval"][iv_key][1] for c in CELLS])
            ax.bar(
                x + (i - 0.5) * 0.38,
                v,
                0.38,
                color=colour,
                label=f"experiment {name}",
            )
            ax.errorbar(
                x + (i - 0.5) * 0.38, v, yerr=[v - lo, hi - v], fmt="none", ecolor="k", lw=1
            )
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_ylabel("Share")
    top_labels = [f"{c}\nn={int(res['653 default'][c]['n_events'])} events" for c in CELLS]
    bottom_labels = [f"{c}\nn={int(res['653 default'][c]['n_edge'])} ends" for c in CELLS]
    axes[0].set_xticks(x, top_labels, fontsize=7.5)
    axes[1].set_xticks(x, bottom_labels, fontsize=7.5)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.text(
        0.01,
        0.005,
        "Bars: estimate. Black line: 95% interval from resampling the 6 injected worlds "
        "(optimistic).\nDeep: 8 noise units or more. n: injected events or edge-transfer ends.",
        fontsize=7.5,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(DOCS / "figure_recall_by_cell.svg")
    fig.savefig("/tmp/e665_png/recall.png", dpi=100)
    plt.close(fig)


def event_series_figure() -> None:
    """D5 and D16 power with the decoded events of experiments 653 and 658 (seed 0)."""
    panel = Panel()
    ev653 = pl.read_parquet(WORK / "real_events_masked.parquet")
    ev658 = pl.read_parquet(E658 / "real_events_v2main_seed0.parquet")
    lo, hi = datetime(2025, 12, 15), datetime(2026, 1, 21)
    lo_i = int((lo - GRID_START.replace(tzinfo=None)).total_seconds() // 1800)
    hi_i = int((hi - GRID_START.replace(tzinfo=None)).total_seconds() // 1800)
    verified = {
        "D5": (datetime(2025, 12, 29, 1, 0), datetime(2026, 1, 13, 11, 30)),
        "D16": (datetime(2025, 12, 29, 16, 0), datetime(2026, 1, 13, 11, 30)),
    }
    base = np.datetime64("2022-01-01T00:00")
    fig, axes = plt.subplots(2, 1, figsize=(WIDTH, 6.2), sharex=True)
    for ax, site in zip(axes, ("D5", "D16"), strict=True):
        t = np.arange(lo_i, hi_i)
        ax.plot(base + t.astype("timedelta64[30m]"), panel.power[site][t], color="k", lw=0.8)
        top = float(np.nanmax(panel.power[site][lo_i:hi_i]))
        bottom = float(np.nanmin(panel.power[site][lo_i:hi_i]))
        ax.set_ylim(bottom - 0.3 * (top - bottom), top + 0.05 * (top - bottom))
        ax.axvline(np.datetime64(verified[site][0]), color="k", ls="--", lw=1)
        ax.axvline(np.datetime64(verified[site][1]), color="k", ls="--", lw=1)
        for ev, colour, (y0, y1) in ((ev653, BLUE, (0.0, 0.06)), (ev658, ORANGE, (0.07, 0.13))):
            for r in ev.filter(pl.col("primary") == site).iter_rows(named=True):
                end = hi_i if r["end"] < 0 else r["end"]
                if r["start"] < hi_i and end > lo_i and r["units"] >= 2:
                    ax.axvspan(
                        base + np.timedelta64(30 * max(r["start"], lo_i), "m"),
                        base + np.timedelta64(30 * min(end, hi_i), "m"),
                        ymin=y0,
                        ymax=y1,
                        facecolor=colour,
                        edgecolor="white",
                        linewidth=1.5,
                        alpha=0.85,
                    )
                    ax.text(
                        base
                        + np.timedelta64(30 * (max(r["start"], lo_i) + min(end, hi_i)) // 2, "m"),
                        ax.get_ylim()[0] + (y0 + y1) / 2 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                        f"{r['amp_mw']:+.1f}",
                        fontsize=7.5,
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
        plt.Line2D(
            [], [], color="k", ls="--", lw=1, label="verified start and end (experiment 580)"
        ),
        plt.Rectangle((0, 0), 1, 1, color=BLUE, label="experiment 653 decoded events"),
        plt.Rectangle((0, 0), 1, 1, color=ORANGE, label="experiment 658 second look, seed 0"),
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        fontsize=7,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.03),
        ncol=2,
    )
    fig.text(
        0.01,
        0.005,
        "Numbers on the bands: decoded step in MW (events of 2 noise units or more). "
        "Band height carries no meaning.",
        fontsize=7.5,
    )
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(DOCS / "figure_event_series_D5_D16.svg")
    fig.savefig("/tmp/e665_png/series.png", dpi=100)
    plt.close(fig)


def event_free_figure() -> None:
    """Events per Primary-year of event-free time, per cell and method."""
    res = json.loads((OUT / "real_by_cell.json").read_text())["event_free_per_primary_year"]
    methods = {
        "653 default": ("experiment 653 table", BLUE),
        "658": ("experiment 658 network (mean of 3 seeds)", ORANGE),
        "LR_b20": ("label-trained detector LR_b20", SKY),
        "RO-NC": ("label-free detector RO-NC", PURPLE),
    }
    vals = {
        "653 default": res["653 default"],
        "658": {
            c: float(np.mean([res[f"658 second look seed {s}"][c] for s in SEEDS]))
            for c in [*CELLS, "none"]
        },
        "LR_b20": res["LR_b20"],
        "RO-NC": res["RO-NC"],
    }
    groups = [*CELLS, "none"]
    fig, ax = plt.subplots(figsize=(WIDTH, 4.0))
    x = np.arange(len(groups))
    for i, (k, (label, colour)) in enumerate(methods.items()):
        ax.bar(x + (i - 1.5) * 0.2, [vals[k][c] for c in groups], 0.2, color=colour, label=label)
    ax.set_xticks(
        x, [c.replace(", ", ",\n") for c in CELLS] + ["under 2\nnoise units"], fontsize=7.5
    )
    ax.set_ylabel("Events per Primary-year")
    ax.set_title(
        "Events called in event-free time, more than 14 days from any logged or verified event "
        "(smaller is better)",
        fontsize=8,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=8)
    fig.text(
        0.01,
        0.005,
        "27.6 Primary-years of event-free time; one event is 0.04 per Primary-year.\n"
        "NGED's logs miss events, so rates are upper bounds on false events.",
        fontsize=7,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(DOCS / "figure_event_free_stratum.svg")
    fig.savefig("/tmp/e665_png/eventfree.png", dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    Path("/tmp/e665_png").mkdir(exist_ok=True)
    recall_figure()
    event_series_figure()
    event_free_figure()
