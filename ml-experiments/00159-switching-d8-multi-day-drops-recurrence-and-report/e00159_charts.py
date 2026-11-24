# ruff: noqa: E501
"""Draw the figures of experiment 159 from the tables that `e00159_drops.py` wrote.

The drops' dates and depths are not drawn: figure 1 is a histogram of durations.

Afterwards optimise each with `npx svgo@4 --multipass --precision=1 --final-newline`.
"""

from __future__ import annotations

import sys
import textwrap
from itertools import pairwise
from pathlib import Path
from typing import Final

import altair as alt
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
from e00159_drops import (
    HALF_HOURS_PER_DAY,
    STEP_FRACTIONS,
    WORK_DIR,
    wilson,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import engine_figure_guard as guard

OUT_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00159-switching-d8-multi-day-drops-recurrence-and-report"
)
WIDTH: Final[int] = 640
WRAP: Final[int] = 95
WEEKDAYS: Final[list[str]] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _title(*, text: str, line: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=text,
        subtitle=textwrap.wrap(line, WRAP),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def figure_history(*, drops: pl.DataFrame) -> alt.TopLevelMixin:
    """Histogram of the drops' durations in day bands; no dates, no power values."""
    edges = [2, 3, 4, 5, 7, 10, 15, 20]
    labels = [f"{lo} to {hi}" for lo, hi in pairwise(edges)]
    durations = drops["duration_days"].to_numpy()
    counts, _ = np.histogram(durations, bins=edges)
    data = pl.DataFrame({"band": labels, "drops": counts.tolist()})
    return (
        alt.Chart(data)
        .mark_bar(color=ocf.DATA_BLUE, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "band:N",
                sort=labels,
                title="Duration of the drop (days)",
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y("drops:Q", title="Drops", axis=alt.Axis(tickMinStep=1)),
        )
        .properties(
            width=WIDTH,
            height=200,
            title=_title(
                text=f"{counts[0]} of D8's {len(durations)} drops last under 3 days, and {counts[-2:].sum()} last 10 days or longer",
                line="Runs of 2 days or longer in which the 24-hour median of metered power minus "
                "the convex model's expected power is below the drop threshold.",
            ),
        )
    )


def figure_recurrence(*, drops: pl.DataFrame) -> alt.TopLevelMixin:
    """Weekday and hour-of-day histograms of the drops' starts, and the spacing between them."""
    weekday = (
        drops.group_by("start_weekday")
        .len()
        .join(pl.DataFrame({"start_weekday": range(7)}), on="start_weekday", how="right")
        .fill_null(0)
        .with_columns(day=pl.col("start_weekday").replace_strict(dict(enumerate(WEEKDAYS))))
    )
    bars = (
        alt.Chart(weekday)
        .mark_bar(color=ocf.DATA_BLUE)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "day:N",
                sort=WEEKDAYS,
                title="Weekday of the drop's start",
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y("len:Q", title="Drops"),
        )
        .properties(width=WIDTH, height=110)
    )
    hours = drops.with_columns(block=(pl.col("start_hour") // 3 * 3).cast(pl.Int64))
    hour_bars = (
        alt.Chart(hours)
        .mark_bar(color=ocf.DATA_BLUE)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "block:Q",
                bin=alt.Bin(step=3, extent=[0, 24]),
                title="Hour of day of the drop's start (local time)",
                axis=alt.Axis(values=[0, 6, 12, 18, 24]),
            ),
            y=alt.Y("count():Q", title="Drops"),
        )
        .properties(width=WIDTH, height=110)
    )
    gaps = (
        drops.sort("first_stamp")
        .with_columns(gap=pl.col("first_stamp").diff() / HALF_HOURS_PER_DAY)
        .drop_nulls("gap")
    )
    gap_bars = (
        alt.Chart(gaps)
        .mark_bar(color=ocf.DATA_BLUE)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "gap:Q",
                bin=alt.Bin(step=5, extent=[0, 65]),
                title="Days from one drop's start to the next (gaps over 65 days not shown)",
            ),
            y=alt.Y("count():Q", title="Drops"),
        )
        .properties(width=WIDTH, height=110)
    )
    return alt.vconcat(bars, hour_bars, gap_bars, spacing=18).properties(
        title=_title(
            text="No weekday, hour, or interval pattern is detectable in the 29 drops",
            line="The 29 drops. No test of the weekday and hour-of-day counts against a flat "
            "pattern reaches p = 0.05, but 29 drops give the tests little power.",
        )
    )


def figure_neighbours(*, drops: pl.DataFrame, placebo: dict[float, float]) -> alt.TopLevelMixin:
    """Share of drops with a neighbour step-up of a given fraction of the depth, against chance."""
    depth = drops["depth_mw"].to_numpy()
    steps = drops["best_neighbour_step_mw"].to_numpy()
    n = len(depth)
    rows = []
    for fraction in STEP_FRACTIONS:
        k = int((steps >= fraction * depth).sum())
        low, high = wilson(k=k, n=n)
        rows.append(
            {
                "fraction": f"{fraction:.2f} x depth",
                "group": "At a drop's start",
                "share": k / n,
                "low": low,
                "high": high,
            }
        )
        rows.append(
            {
                "fraction": f"{fraction:.2f} x depth",
                "group": "At a random time, no drop",
                "share": placebo[fraction],
                "low": placebo[fraction],
                "high": placebo[fraction],
            }
        )
    data = pl.DataFrame(rows)
    colour = alt.Color(
        "group:N",
        scale=alt.Scale(
            domain=["At a drop's start", "At a random time, no drop"],
            range=[ocf.DATA_BLUE, ocf.DATA_BLUE_LIGHT],
        ),
        legend=alt.Legend(title=None, orient="top"),
    )
    base = alt.Chart(data)
    bars = base.mark_bar().encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "fraction:N",
            title="Step-up of the best neighbour, as a share of the drop's depth",
            axis=alt.Axis(labelAngle=0),
        ),
        xOffset="group:N",
        y=alt.Y("share:Q", title="Share of drops", scale=alt.Scale(domain=[0, 0.7])),
        color=colour,
    )
    errors = (
        base.transform_filter(alt.datum.group == "At a drop's start")
        .mark_rule(color="black")
        .encode(x="fraction:N", xOffset="group:N", y="low:Q", y2="high:Q")  # ty: ignore[unresolved-attribute]
    )
    return (bars + errors).properties(
        width=WIDTH,
        height=200,
        title=_title(
            text="A neighbour's step-up at a drop is not clearly above its rate at random times",
            line="Best step-up of D3, D14, and D5 (6-hour median after the drop's start minus "
            "6-hour median before). Whiskers: 95% Wilson interval. The comparison is 5,000 "
            "random times from July 2023 at the time of day of a drop start, at least 2 days "
            "from any drop.",
        ),
    )


def main() -> None:
    """Draw every figure."""
    drops = pl.read_parquet(WORK_DIR / "drops.parquet")
    placebo = {float(k): v for k, v in pl.read_parquet(WORK_DIR / "placebo.parquet").iter_rows()}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figures = {
        "figure_1_history.svg": figure_history(drops=drops),
        "figure_4_recurrence.svg": figure_recurrence(drops=drops),
        "figure_6_neighbours.svg": figure_neighbours(drops=drops, placebo=placebo),
    }
    for name, chart in figures.items():
        guard.save_chart(chart=chart, path=OUT_DIR / name)
    print(f"wrote {len(figures)} figures")


if __name__ == "__main__":
    main()
