# ruff: noqa: E501
"""Figures of experiment 686, drawn from the stored tables.

    DATA_PATH_INTERNAL=... PYTHONPATH=<folders of 653 and 657>:. uv run --no-sync python e00686_figures.py

Each figure is 6.8 inches wide (680 pixels at 100 dpi) and uses the OCF data colours. Panels are at
least 280 pixels tall, with at most two per figure.
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from e00653_common import DAY, GRID_LENGTH, GRID_START, SCORED_FROM  # ty: ignore[unresolved-import]
from e00653_core import decode_world  # ty: ignore[unresolved-import]
from e00653_inject import GenParams, inject  # ty: ignore[unresolved-import]
from e00653_run import load_params, setup  # ty: ignore[unresolved-import]
from e00653_slow import to_half_hours  # ty: ignore[unresolved-import]
from e00686_classes import CLASSES, WORK  # ty: ignore[unresolved-import]
from e00686_false_events import ARM, MIDDAY, seed_of  # ty: ignore[unresolved-import]

OUT: Final[Path] = (
    Path(__file__).parents[2]
    / "docs/ml-experiments/00686-event-table-transfer-timing-and-false-solar-growth-events"
)
BLUE, ORANGE, SKY, PURPLE, GREEN = "#306BFF", "#FF4901", "#10C5F7", "#B701FF", "#17E58F"
GREY = "#6b6b6b"
CLASS_COLOUR: Final[dict[str, str]] = dict(zip(CLASSES, (ORANGE, BLUE, PURPLE, SKY), strict=True))
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})


def save(fig: plt.Figure, name: str) -> None:
    """Write the SVG into the docs folder and a PNG preview into /tmp for review."""
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.svg")
    Path("/tmp/png686").mkdir(exist_ok=True)
    fig.savefig(f"/tmp/png686/{name}.png", dpi=100)
    plt.close(fig)


def fig_class_shares() -> None:
    """Share of the false events in each class, with 95% Wilson intervals."""
    d = json.loads((WORK / "classes.json").read_text())
    fig, ax = plt.subplots(figsize=(6.8, 2.9))
    for i, c in enumerate(CLASSES):
        v = d["classes"][c]
        ax.barh(i, v["share"], color=CLASS_COLOUR[c], height=0.55)
        ax.plot([v["share_lo"], v["share_hi"]], [i, i], color="black", lw=1)
        ax.text(
            v["share_hi"] + 0.02,
            i,
            f"{v['count']} of {d['n_events']} events; {v['rate']:.2f} per Primary-year",
            va="center",
            fontsize=7,
        )
    ax.set_yticks(range(4), CLASSES)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.45)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("Share of the 333 false events (bar), 95% Wilson interval (line)")
    ax.set_title(
        "Few false solar-growth events are steps in a sunny run; most are slow ramps or short\n"
        "bumps. Experiment 653's masked arm, 24 worlds, growth at 3 of 15 Primaries, 2024 to mid-2026.",
        fontsize=8,
        loc="left",
    )
    fig.tight_layout()
    save(fig, "figure_class_shares")


def reconstruct(*, k: int) -> tuple:
    """Re-inject world `k` and decode it; return what the example plots need."""
    panel, res, fz = setup()
    p = load_params()
    _, base_slow = decode_world(panel=panel, residual=res, frozen=fz, arm=ARM, params=p)
    world, _ = inject(
        panel=panel,
        residual=res,
        sigma=fz.sigma,
        rng=np.random.default_rng(seed_of(k=k)),
        lo=SCORED_FROM,
        hi=GRID_LENGTH,
        gen=GenParams(),
        kinds=("pv_growth",),
    )
    _, slows = decode_world(panel=panel, residual=world, frozen=fz, arm=ARM, params=p)
    return res, world, slows, base_slow


def midday_daily(x: np.ndarray) -> np.ndarray:
    """Daily mean of the 10:00 to 14:30 UTC half-hours."""
    with np.errstate(invalid="ignore"):
        return np.nanmean(x[: len(x) // DAY * DAY].reshape(-1, DAY)[:, MIDDAY], axis=1)


def pick(*, rows: pl.DataFrame, cls: str) -> dict:
    """The event of a class whose size in noise units is nearest the class median."""
    sub = rows.filter(pl.col("cls686") == cls)
    med = sub["units"].median()
    return (
        sub.with_columns((pl.col("units") - med).abs().alias("gap")).sort("gap").row(0, named=True)
    )


def draw_example(*, ax: plt.Axes, ev: dict, title: str) -> None:
    """One event: the midday level of the world, the injected growth, and the slow level."""
    res, world, slows, base_slow = reconstruct(k=int(ev["world"]))
    site, start = ev["primary"], int(ev["start"])
    end = start + int(ev["days"] * DAY) if ev["end"] >= 0 else start + 30 * DAY
    d0, d1 = start // DAY, end // DAY
    lo, hi = max(0, d0 - 45), min(len(world[site]) // DAY, d1 + 45)
    days = np.arange(lo, hi)
    when = [GRID_START + timedelta(days=int(d)) for d in days]
    mid_world = midday_daily(world[site])[lo:hi]
    mid_base = midday_daily(res[site])[lo:hi]
    mid_slow = midday_daily(to_half_hours(slow=slows[site] - base_slow[site]))[lo:hi]
    ax.axvspan(
        GRID_START + timedelta(days=d0),
        GRID_START + timedelta(days=max(d1, d0 + 1)),
        color="#dddddd",
        lw=0,
    )
    ax.plot(when, mid_world - mid_base, color=ORANGE, lw=1, label="injected growth (truth)")
    ax.plot(
        when, mid_slow, color=GREEN, lw=1, label="slow level's change from the un-injected record"
    )
    ax.plot(
        when, mid_world - np.nanmedian(mid_world), color=BLUE, lw=1, label="residual with growth"
    )
    ax.axhline(0, color=GREY, lw=0.5)
    ax.set_ylabel("MW, midday daily mean")
    ax.set_title(title, fontsize=8, loc="left")
    ax.legend(fontsize=6.5, loc="upper left", frameon=False, ncol=1)


def fig_examples() -> None:
    """Two figures of two example events each, one event per class."""
    rows = pl.read_parquet(WORK / "classes.parquet")
    for name, classes in (("figure_examples_1", CLASSES[:2]), ("figure_examples_2", CLASSES[2:])):
        fig, axes = plt.subplots(2, 1, figsize=(6.8, 6.0))
        for ax, c in zip(axes, classes, strict=True):
            ev = pick(rows=rows, cls=c)
            draw_example(
                ax=ax,
                ev=ev,
                title=(
                    f"{c}: {ev['primary']}, world {ev['world']}, {ev['days']:.1f} days, "
                    f"{ev['units']:.1f} noise units, midday clearness {ev['mean_clearness_3d']:.2f}"
                ),
            )
        fig.autofmt_xdate()
        fig.tight_layout()
        save(fig, name)


def fig_gap() -> None:
    """Fitted onset gap (recipient minus donor) at the four edges, against the labelled gap."""
    real = json.loads((WORK / "timing_real.json").read_text())
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    names = []
    for i, x in enumerate(real):
        lo, hi = x["gap_lo"] / 2, x["gap_hi"] / 2
        ax.plot([lo, hi], [i, i], color=BLUE, lw=1)
        ax.plot(
            x["gap_hh"] / 2,
            i,
            "o",
            color=BLUE,
            ms=4,
            label="fitted gap, 90% interval" if i == 0 else None,
        )
        ax.plot(
            x["nominal_gap_hh"] / 2,
            i,
            "D",
            color=ORANGE,
            ms=4,
            label="gap between the labelled times" if i == 0 else None,
        )
        names.append(f"{x['pair']} {x['edge']}\nshared-onset p = {x['p_value']:.2f}")
    ax.axvline(0, color=GREY, lw=0.5)
    ax.set_yticks(range(len(real)), names)
    ax.invert_yaxis()
    ax.set_xlabel("Recipient's step minus donor's step (hours; 0 means simultaneous)")
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.set_title(
        "The fitted steps of donor and recipient coincide within noise at all four edges, although\n"
        "the labelled times differ by up to 18 hours. Verified transfers D5/D16 and D3/D22.",
        fontsize=8,
        loc="left",
    )
    fig.tight_layout()
    save(fig, "figure_gap")


def fig_injected() -> None:
    """Gap the same fit reports for transfers injected with a known gap of 0 or 14.5 hours."""
    inj = pl.read_parquet(WORK / "timing_injected.parquet")
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 6.0), sharex=True)
    rng = np.random.default_rng(1)
    for ax, edge in zip(axes, ("onset", "end"), strict=True):
        for j, pair in enumerate(("D5/D16", "D3/D22")):
            for lag, colour in ((0, BLUE), (29, ORANGE)):
                g = (
                    inj.filter(
                        (pl.col("pair") == pair)
                        & (pl.col("edge") == edge)
                        & (pl.col("true_lag_hh") == lag)
                    )["gap_hh"].to_numpy()
                    / 2
                )
                ax.plot(
                    g,
                    j + (0.18 if lag else -0.18) + rng.uniform(-0.07, 0.07, len(g)),
                    ".",
                    color=colour,
                    ms=3,
                    label=f"true gap {lag / 2:g} hours" if j == 0 else None,
                )
                ax.plot(
                    [np.median(g)] * 2,
                    [j + (0.18 if lag else -0.18) - 0.12, j + (0.18 if lag else -0.18) + 0.12],
                    color="black",
                    lw=1,
                )
        ax.set_yticks([0, 1], ["D5/D16", "D3/D22"])
        ax.set_ylim(1.6, -0.6)
        ax.axvline(0, color=GREY, lw=0.5)
        ax.set_title(
            f"Injected at the {edge} (60 windows per Primary pair and gap; black tick = median)",
            fontsize=8,
            loc="left",
        )
    axes[0].legend(fontsize=7, frameon=False, loc="lower right")
    axes[1].set_xlabel("Gap the fit reports, recipient minus donor (hours)")
    fig.suptitle(
        "A true gap of 0 is usually read back within 2 hours; some windows scatter by 6 hours or more.",
        fontsize=8,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    save(fig, "figure_injected")


def fig_real_edge() -> None:
    """Residual of both Primaries around the D5/D16 onset, with the fitted and labelled onsets."""
    import e00660_data as data  # ty: ignore[unresolved-import]

    real = next(
        x
        for x in json.loads((WORK / "timing_real.json").read_text())
        if x["pair"] == "D5/D16" and x["edge"] == "onset"
    )
    centre = real["centre_idx"]
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 6.0), sharex=True)
    for ax, site, key in zip(
        axes, ("D5", "D16"), ("onset_donor_hh", "onset_recipient_hh"), strict=True
    ):
        power = np.load(data.E327 / f"power/power_{data.series_info()[site]['id']}.npy").astype(
            float
        )
        r = power - data.load_forecast(site=site)
        lo = centre - 5 * DAY
        seg = r[lo : lo + 10 * DAY]
        hours = (np.arange(len(seg)) - 5 * DAY) / 2
        blocks = np.nanmean(seg[: len(seg) // 6 * 6].reshape(-1, 6), axis=1)
        ax.plot(hours[::6] + 1.5, blocks, color=BLUE, lw=1)
        ax.axvline(real[key] / 2, color=ORANGE, lw=1, label="fitted onset (separate fit)")
        ax.axvline(
            real["onset_shared_hh"] / 2,
            color=GREEN,
            lw=1,
            ls="--",
            label="fitted onset (shared fit)",
        )
        ax.set_ylabel(f"{site} residual (MW, 3-hour mean)")
        ax.set_title(f"{site}", fontsize=8, loc="left")
    axes[0].legend(fontsize=7, frameon=False, loc="lower left")
    axes[1].set_xlabel("Hours from the midpoint of the two labelled onsets (December 2025)")
    fig.suptitle(
        "Donor and recipient step together at the D5/D16 onset.", fontsize=8, x=0.01, ha="left"
    )
    fig.tight_layout()
    save(fig, "figure_real_edge")


if __name__ == "__main__":
    fig_class_shares()
    fig_examples()
    fig_gap()
    fig_injected()
    fig_real_edge()
