# ruff: noqa: E501
"""Figures of experiment 660, drawn from the stored evidence tables (no refitting).

    DATA_PATH_INTERNAL=... PYTHONPATH=<this folder> uv run --no-sync python e00660_figures.py

Each figure is 6.8 inches wide (680 pixels) and uses the OCF data colours.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import e00660_data as data
import e00660_generator as gen
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

OUT: Final[Path] = (
    Path(__file__).parents[2]
    / "docs/ml-experiments/00660-shared-generator-of-injected-load-transfer-events"
)
BLUE, ORANGE, SKY, PURPLE, GREEN = "#306BFF", "#FF4901", "#10C5F7", "#B701FF", "#17E58F"
GREY = "#6b6b6b"
BLOCK = 12
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})


def save(fig: plt.Figure, name: str) -> None:
    """Write the SVG into the docs folder and a PNG preview into /tmp for review."""
    fig.savefig(OUT / f"{name}.svg")
    Path("/tmp/png660").mkdir(exist_ok=True)
    fig.savefig(f"/tmp/png660/{name}.png", dpi=100)


def block(x: np.ndarray) -> np.ndarray:
    """6-hour block means."""
    n = len(x) // BLOCK * BLOCK
    return np.nanmean(x[:n].reshape(-1, BLOCK), axis=1)


def fig_real_vs_injected() -> None:
    """Real D5/D16 transfer beside an injected transfer of the same size on other dates."""
    panel = data.load_panel()
    fc = {s: data.load_forecast(site=s) for s in ("D5", "D16")}
    on = data.idx_of(when=gen.datetime(2025, 12, 29, 1, tzinfo=gen.UTC))
    dur = int(15.4375 * gen.DAY)
    new_on = data.idx_of(when=gen.datetime(2024, 1, 8, 1, tzinfo=gen.UTC))
    rows = []
    for site, role, delta in (("D5", "donor", -1.49), ("D16", "recipient", 2.07)):
        rows.append(
            {
                "event_id": len(rows),
                "group_id": 0,
                "site": site,
                "role": role,
                "event_class": "edge_transfer",
                "label": gen.POSITIVE_EDGE,
                "start": new_on,
                "end": new_on + dur,
                "residual_frac": 0.0,
                "delta": delta,
                "units": abs(delta) / panel.noise[site],
                "shape": "proportional",
                "prop_share": 0.5,
                "shape_source": "D5",
                "ramp_in_hh": 2,
                "ramp_out_hh": 2,
                "drift": 0.0,
                "mid_factor": 1.0,
                "staged_hh": 0,
                "share": 1.39,
                "period_hh": 0.0,
                "phase": 0.0,
                "partners": [],
                "chain_parent": -1,
            }
        )
    injected = gen.apply_events(panel=panel, events=pl.DataFrame(rows, schema=gen.EVENT_COLUMNS))
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 4.6), sharey=True)
    pad = 6 * gen.DAY
    for ax, (title, power, start) in zip(
        axes,
        (
            ("Real transfer (December 2025 to January 2026)", panel.power, on),
            (
                "Injected transfer of the same size (January 2024 to February 2024)",
                injected,
                new_on,
            ),
        ),
        strict=True,
    ):
        for site, colour in (("D5", BLUE), ("D16", ORANGE)):
            r = block(
                power[site][start - pad : start + dur + pad]
                - fc[site][start - pad : start + dur + pad]
            )
            r = r - np.nanmedian(r[: pad // BLOCK])
            ax.plot(
                (np.arange(len(r)) * BLOCK - pad) / gen.DAY, r, color=colour, lw=1.0, label=site
            )
        ax.axvline(0, color=GREY, lw=0.8, ls="--")
        ax.axvline(dur / gen.DAY, color=GREY, lw=0.8, ls="--")
        ax.axhline(0, color=GREY, lw=0.5)
        ax.set_title(title, loc="left", fontsize=8.5)
        ax.set_ylabel("Residual (MW)")
    axes[0].legend(
        loc="upper right",
        frameon=False,
        ncol=2,
        title="Primary (donor D5, recipient D16)",
        title_fontsize=7.5,
    )
    axes[1].set_xlabel("Days since the event began (dashed lines: begin and end)")
    fig.suptitle(
        "Injected load transfer beside a real one at the same two Primaries\n(residual: power minus the lag-free forecast, 6-hour means)",
        x=0.01,
        ha="left",
        fontsize=9.5,
    )
    fig.tight_layout()
    save(fig, "figure_real_vs_injected")
    plt.close(fig)


def fig_edges() -> None:
    """Fitted transition width of each end of the two verified transfers, with intervals."""
    t = pl.read_parquet(data.WORK / "ev1_edges.parquet")
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    labels = [f"{r['primary']} {r['edge']}" for r in t.iter_rows(named=True)]
    y = np.arange(len(t))
    colours = [BLUE if r["pair"] == "D5/D16" else ORANGE for r in t.iter_rows(named=True)]
    ax.hlines(y, t["width_lo"], t["width_hi"], color=colours, lw=2)
    ax.scatter(t["width_hh"], y, color=colours, zorder=3, s=18)
    ax.axvspan(0, 4, color=GREY, alpha=0.12)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Ramp width (half-hours; 0: a step within one half-hour)")
    ax.set_title(
        "Dot: best fit. Line: 90% interval. Shaded: default range 0 to 4. Blue: D5/D16. Orange: D3/D22.",
        loc="left",
        fontsize=8.5,
    )
    fig.tight_layout()
    save(fig, "figure_edge_width")
    plt.close(fig)


def fig_shape() -> None:
    """Share of each step that follows the donor's daily profile."""
    t = pl.read_parquet(data.WORK / "ev1_shapes.parquet").filter(
        pl.col("profile") == "donor_profile"
    )
    fig, ax = plt.subplots(figsize=(6.8, 2.6))
    y = np.arange(len(t))
    colours = [BLUE if r["pair"] == "D5/D16" else ORANGE for r in t.iter_rows(named=True)]
    ax.hlines(y, t["c_lo"], t["c_hi"], color=colours, lw=2)
    ax.scatter(t["c_proportional"], y, color=colours, zorder=3, s=18)
    ax.axvline(0, color=GREY, lw=0.6)
    ax.axvline(1, color=GREY, lw=0.6)
    ax.set_yticks(y, [f"{r['primary']} ({r['pair']})" for r in t.iter_rows(named=True)])
    ax.invert_yaxis()
    ax.set_xlabel(
        "Share of the step that follows the donor's daily profile (0: constant MW; 1: proportional)"
    )
    ax.set_title(
        "Step shape in the two verified transfers. Dot: best fit. Line: 90% interval.",
        loc="left",
        fontsize=8.5,
    )
    fig.tight_layout()
    save(fig, "figure_step_shape")
    plt.close(fig)


def fig_realness() -> None:
    """Realness AUC with the label-shuffle null, and each feature alone."""
    res = json.loads((data.WORK / "ev2_default.json").read_text())
    feats = list(res["only"])
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 4.6), gridspec_kw={"height_ratios": [1, 2.2]})
    ax = axes[0]
    ax.barh(["all features"], [res["all"]["auc_mean"]], color=BLUE, height=0.5)
    ax.errorbar(
        [res["all"]["auc_mean"]],
        [0],
        xerr=[[res["all"]["auc_sd"]], [res["all"]["auc_sd"]]],
        color="k",
        capsize=3,
    )
    ax.axvline(0.5, color=GREY, lw=0.8)
    ax.axvline(0.65, color=ORANGE, lw=1.0, ls="--")
    ax.axvspan(res["all"]["null_mean"], res["all"]["null_p95"], color=GREY, alpha=0.2)
    ax.set_xlim(0.4, 1.0)
    ax.set_title(
        "AUC of real against injected windows (0.5: cannot tell apart)\nBar: mean of 20 repeats; line: spread over repeats. Dashed: 0.65 target.\nGrey: label-shuffle null, up to its 95th percentile.",
        loc="left",
        fontsize=8.5,
    )
    ax.set_xlabel("AUC, all features together")
    ax = axes[1]
    only = [res["only"][f] for f in feats]
    ax.barh(feats, only, color=SKY, height=0.6)
    ax.axvline(0.5, color=GREY, lw=0.8)
    ax.axvline(0.65, color=ORANGE, lw=1.0, ls="--")
    ax.set_xlim(0.4, 1.0)
    ax.invert_yaxis()
    ax.set_xlabel("AUC of each feature used alone")
    fig.tight_layout()
    save(fig, "figure_realness")
    plt.close(fig)


def fig_cross() -> None:
    """Recall of detectors tuned on one configuration and tested on another."""
    res = json.loads((data.WORK / "ev3_cross_injector.json").read_text())
    names = {
        "A_constant_square": "constant-MW square steps",
        "B_shaped_short_heavy": "shaped, short-heavy",
    }
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    width = 0.35
    for i, tuned in enumerate(names):
        vals, los, his = [], [], []
        for tested in names:
            c = next(x for x in res["cells"] if x["tuned_on"] == tuned and x["tested_on"] == tested)
            m, lo, hi = c["recall_big"]
            vals.append(m)
            los.append(m - lo)
            his.append(hi - m)
        x = np.arange(2) + (i - 0.5) * width
        ax.bar(
            x,
            vals,
            width,
            yerr=[los, his],
            capsize=3,
            color=[BLUE, ORANGE][i],
            label=f"tuned on {names[tuned]}",
        )
    ax.set_xticks(np.arange(2), [f"tested on {v}" for v in names.values()])
    ax.set_ylabel("MW-weighted recall")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=2)
    ax.set_title(
        "Events of 4 noise units or more lasting a day. Bar: mean of 8 injected worlds. Line: 95% interval.",
        loc="left",
        fontsize=8,
        y=1.17,
    )
    fig.tight_layout()
    save(fig, "figure_cross_injector")
    plt.close(fig)


def fig_size() -> None:
    """Recall by size: injected default-configuration events against real events."""
    res = json.loads((data.WORK / "ev4_recall_by_size.json").read_text())
    bins = [b for b in res["bins"] if b["injected_recall"]]
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    x = np.arange(len(bins))
    inj = [b["injected_recall"] for b in bins]
    ax.errorbar(
        x - 0.07,
        [i[0] for i in inj],
        yerr=[[i[0] - i[1] for i in inj], [i[2] - i[0] for i in inj]],
        fmt="o",
        color=BLUE,
        capsize=3,
        label="Injected events (8 worlds)",
    )
    real = [b for b in bins if b["real_n"]]
    xr = np.array([i for i, b in enumerate(bins) if b["real_n"]])
    ax.errorbar(
        xr + 0.07,
        [b["real_recall"] for b in real],
        yerr=[
            [b["real_recall"] - b["real_interval"][0] for b in real],
            [b["real_interval"][1] - b["real_recall"] for b in real],
        ],
        fmt="s",
        color=ORANGE,
        capsize=3,
        label="Real events visible in the telemetry",
    )
    for xi, b in zip(xr, real, strict=True):
        ax.annotate(f"n={b['real_n']}", (xi + 0.07, 0.02), ha="center", fontsize=7, color=ORANGE)
    ax.set_xticks(x, [b["bin"] for b in bins])
    ax.set_xlabel("Event size (noise units); events of at least one day")
    ax.set_ylabel("Share of events found")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, "figure_recall_by_size")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (fig_real_vs_injected, fig_edges, fig_shape, fig_realness, fig_cross, fig_size):
        try:
            f()
            print("drew", f.__name__)
        except FileNotFoundError as e:
            print("skipped", f.__name__, e)
