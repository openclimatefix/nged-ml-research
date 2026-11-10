# ruff: noqa: E501
"""Figures of experiment 690, drawn from the stored tables.

    DATA_PATH_INTERNAL=... uv run --no-sync python e00690_figures.py

Each figure is 6.8 inches wide and uses the OCF data colours; panels are at least 280 pixels tall.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DATA: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00690-slow-level-state-on-primaries-13-and-16-july-august"
)
OUT: Final[Path] = (
    Path(__file__).parents[2]
    / "docs/ml-experiments/00690-slow-level-state-on-primaries-13-and-16-july-august"
)
BLUE, ORANGE, SKY, PURPLE, GREY = "#306BFF", "#FF4901", "#10C5F7", "#B701FF", "#6b6b6b"
NAMES: Final[dict[int, str]] = {
    13: "Primary 13 (D18)",
    16: "Primary 16 (D14)",
    26: "Primary 26 (control)",
    17: "Primary 17 (control)",
}
ARM_COLOUR: Final[dict[str, str]] = {"masked": BLUE, "huber": SKY, "rw30": PURPLE}
ARM_LABEL: Final[dict[str, str]] = {
    "masked": "masked frozen level (main)",
    "huber": "Huber, no mask (S1)",
    "rw30": "30-day random walk (S0)",
}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})


def save(fig: plt.Figure, name: str) -> None:
    """Write the SVG into the docs folder and a PNG preview into /tmp for review."""
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    Path("/tmp/png690").mkdir(exist_ok=True)
    fig.savefig(f"/tmp/png690/{name}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def shade(ax: plt.Axes) -> None:
    """Mark the two spans that define the change."""
    spans = (
        (dt.date(2026, 5, 1), dt.date(2026, 6, 15), "base"),
        (dt.date(2026, 7, 15), dt.date(2026, 9, 1), "after"),
    )
    for a, b, t in spans:
        ax.axvspan(a, b, color="#eeeeee", zorder=0)
        ax.text(a, ax.get_ylim()[1], t, va="top", fontsize=7, color=GREY)


def fig_miss(ids: list[int], name: str) -> None:
    """Weekly median of observed minus stored q50, with the weekly 5th to 95th percentile."""
    w = pl.read_csv(DATA / "weekly_miss.csv", try_parse_dates=True)
    fig, axes = plt.subplots(len(ids), 1, figsize=(6.8, 2.9 * len(ids)), sharex=True)
    for ax, s in zip(np.atleast_1d(axes), ids, strict=True):
        d = w.filter(pl.col("time_series_id") == s).sort("week")
        x = d["week"].to_list()
        ax.fill_between(
            x,
            d["q05_resid_mw"],
            d["q95_resid_mw"],
            color=BLUE,
            alpha=0.2,
            lw=0,
            label="weekly 5th to 95th percentile",
        )
        ax.plot(x, d["median_resid_mw"], color=BLUE, lw=1.6, label="weekly median")
        ax.axhline(0, color=GREY, lw=0.6)
        ax.set_ylabel("observed minus q50 (MW)")
        ax.set_title(
            f"{NAMES[s]}: mean power {d['mean_power_mw'].mean():.1f} MW", loc="left", fontsize=8
        )
        ax.legend(frameon=False, fontsize=7, loc="lower left")
    save(fig, name)


def fig_level(ids: list[int], name: str) -> None:
    """The 7-day residual of the stored forecast and the three slow-level estimates."""
    d = pl.read_parquet(DATA / "daily_level_and_residual.parquet")
    fig, axes = plt.subplots(len(ids), 1, figsize=(6.8, 2.9 * len(ids)), sharex=True)
    for ax, s in zip(np.atleast_1d(axes), ids, strict=True):
        m = d.filter((pl.col("time_series_id") == s) & (pl.col("arm") == "masked")).sort("day")
        ax.plot(
            m["day"],
            m["residual_7d"],
            color="black",
            lw=1.6,
            label="stored forecast's miss (7-day mean)",
        )
        for arm in ("masked", "huber", "rw30"):
            a = d.filter((pl.col("time_series_id") == s) & (pl.col("arm") == arm)).sort("day")
            ax.plot(a["day"], a["level"], color=ARM_COLOUR[arm], lw=1.2, label=ARM_LABEL[arm])
        ax.set_ylabel("MW")
        ax.set_title(NAMES[s], loc="left", fontsize=8)
        shade(ax)
        ax.legend(frameon=False, fontsize=7, loc="best")
    save(fig, name)


def fig_error() -> None:
    """Change in median absolute error from adding the level to q50, with 95% intervals."""
    r = json.loads((DATA / "results.json").read_text())
    ids = [int(i) for i in r["per_id"]]
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    for k, arm in enumerate(("masked", "huber", "rw30")):
        for j, s in enumerate(ids):
            v = r["per_id"][str(s)]["diagnostic"][arm]["level"]
            err = [
                [v["mae_change_pct"] - v["mae_change_ci"][0]],
                [v["mae_change_ci"][1] - v["mae_change_pct"]],
            ]
            ax.errorbar(
                j + (k - 1) * 0.22,
                v["mae_change_pct"],
                yerr=err,
                fmt="o",
                color=ARM_COLOUR[arm],
                capsize=3,
                label=ARM_LABEL[arm] if j == 0 else None,
            )
    ax.axhline(0, color=GREY, lw=0.6)
    ax.axhline(-25, color=ORANGE, lw=0.8, ls="--")
    ax.set_xticks(range(len(ids)), [NAMES[s] for s in ids], fontsize=7)
    ax.set_ylabel("change in median absolute error (%)")
    ax.legend(frameon=False, fontsize=7)
    save(fig, "figure_error_change")


if __name__ == "__main__":
    fig_miss([13, 16], "figure_miss")
    fig_level([13, 16], "figure_level_targets")
    fig_level([26, 17], "figure_level_controls")
    fig_error()
