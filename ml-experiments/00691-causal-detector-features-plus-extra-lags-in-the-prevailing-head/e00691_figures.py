# ruff: noqa: E501, D103
"""Experiment 691: figures (SVG, to docs/ml-experiments/00691-*/).

Run `npx svgo@4 --multipass --precision=1 --final-newline` on them after.
"""

from pathlib import Path

import e00687_report as r687  # ty: ignore[unresolved-import]
import e00691_score as s
import matplotlib as mpl
import numpy as np
import polars as pl

mpl.use("svg")
import matplotlib.pyplot as plt

mpl.rcParams.update({"svg.fonttype": "none", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.0})  # fmt: skip
DOCS = Path(__file__).resolve().parents[2] / "docs/ml-experiments/00691-causal-detector-features-plus-extra-lags-in-the-prevailing-head"  # fmt: skip
COL = {"L1": "#B701FF", "F1": "#FF4901", "FL1": "#306BFF"}
LABEL = {"L1": "L1 (extra lags)", "F1": "F1 (detector features)", "FL1": "FL1 (both)"}


def paired(r: pl.DataFrame, a: str, b: str, col: str, rng: np.random.Generator) -> tuple[float, float, float]:
    xa = r.filter(pl.col("arm") == a).sort("month", "seed", "pass", "prim")
    xb = r.filter(pl.col("arm") == b).sort("month", "seed", "pass", "prim")
    key = (xb["month"] + "_" + xb["pass"].cast(str) + "_" + xb["prim"].cast(str)).to_numpy()
    _u, inv = np.unique(key, return_inverse=True)
    ev = np.bincount(inv, weights=xa[col].to_numpy() - xb[col].to_numpy()) / np.bincount(inv)
    bs = ev[rng.integers(0, len(ev), (4000, len(ev)))].mean(axis=1)
    return float(ev.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def fig_reaction() -> None:
    r = pl.read_parquet(s.OUT / "reaction.parquet")
    rng = np.random.default_rng(1)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
    for ax, col, title in zip(
        axes, ("start_days", "end_days"), ("Start of a shift", "End of a shift"), strict=True
    ):
        for i, arm in enumerate(("L1", "F1", "FL1")):
            m, lo, hi = paired(r, arm, "B0", col, rng)
            ax.errorbar(m, i, xerr=[[m - lo], [hi - m]], fmt="o", color=COL[arm], capsize=3, ms=4)
        ax.axvline(0, color="#888888", lw=0.8)
        ax.set_yticks(range(3), [LABEL[a] for a in ("L1", "F1", "FL1")])
        ax.set_xlabel("Days sooner (negative) than B0, injected held-out shifts")
        ax.set_title(title, loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(DOCS / "fig1_reaction.svg")


def fig_contrasts() -> None:
    r = pl.read_parquet(s.OUT / "reaction.parquet")
    rng = np.random.default_rng(1)
    fig, ax = plt.subplots(figsize=(7.4, 2.6))
    for i, (a, b) in enumerate((("FL1", "F1"), ("FL1", "L1"), ("F1", "L1"))):
        m, lo, hi = paired(r, a, b, "start_days", rng)
        ax.errorbar(m, i, xerr=[[m - lo], [hi - m]], fmt="o", color=COL[a], capsize=3, ms=4)
    ax.axvline(0, color="#888888", lw=0.8)
    ax.axvline(-0.5, color="#222222", lw=0.8, ls=":")
    ax.set_yticks(range(3), ["FL1 minus F1", "FL1 minus L1", "F1 minus L1"])
    ax.set_xlabel("Start-of-shift reaction time, days (dotted line: the 0.5-day bar)")
    fig.tight_layout()
    fig.savefig(DOCS / "fig2_paired.svg")


def fig_crps() -> None:
    agg = pl.read_parquet(s.OUT / "agg.parquet")
    rows = (("real", "real_free", "Real rows, event-free"), ("held", "inforce", "Injected, in force"), ("fam", "just_ended", "Injected, just ended"), ("fam", "free", "Injected, event-free"))  # fmt: skip
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 3.0), sharey=True)
    for ax, (set_, st, title) in zip(axes, rows, strict=True):
        pm = r687.per_month(agg=agg, set_=set_, stratum=st, col="crps_s")
        ref = pm.filter(pl.col("arm") == "B0").sort("month")["v"].to_numpy()
        for i, arm in enumerate(("L1", "F1", "FL1")):
            x = pm.filter(pl.col("arm") == arm).sort("month")["v"].to_numpy()
            m, lo, hi = r687.boot(v=x - ref)
            ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="o", color=COL[arm], capsize=3, ms=4)
        ax.axhline(0, color="#888888", lw=0.8)
        ax.set_xticks(range(3), ["L1", "F1", "FL1"])
        ax.set_title(title, fontsize=8, loc="left")
    axes[0].set_ylabel("Scaled CRPS minus B0 (points of capacity)")
    fig.tight_layout()
    fig.savefig(DOCS / "fig3_crps.svg")


if __name__ == "__main__":
    fig_reaction()
    fig_contrasts()
    fig_crps()
