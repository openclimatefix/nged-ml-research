# ruff: noqa: E501, PERF401
"""Experiment 644: print every number the write-up quotes and draw its figures.

Usage: python e00644_report.py
"""

import json
import os
from pathlib import Path

import altair as alt
import e00639_analyse as a639  # ty: ignore[unresolved-import]
import e00639_common as k639  # ty: ignore[unresolved-import]
import e00644_analyse as an
import e00644_fit as fit
import numpy as np
import polars as pl

DOCS = Path(__file__).resolve().parents[2] / "docs/ml-experiments" / fit.NAME
W = 560
NAMES = {
    "a": "Free network", "b": "Explicit solar term", "d": "Free + 14-day tracker",
    "e": "Free + training constant", "f": "Explicit, R4 table", "s": "Shuffled-day placebo",
}  # fmt: skip
COLOURS = {
    "a": "#FF4901",
    "b": "#306BFF",
    "d": "#17E58F",
    "e": "#B701FF",
    "f": "#10C5F7",
    "s": "#808080",
}
SEEDS = tuple(k639.SEEDS)


def save(chart: alt.TopLevelMixin, name: str) -> None:
    """Write the SVG, and a PNG for review when `PNG_DIR` is set."""
    chart.save(DOCS / f"{name}.svg")
    if os.environ.get("PNG_DIR"):
        chart.save(Path(os.environ["PNG_DIR"]) / f"{name}.png", scale_factor=1.0)


def by_lead_day(*, world: str) -> pl.DataFrame:
    """Return sunny-midday CRPS by lead day and arm, with day-block 95% intervals."""
    rows = an.load(world=world)
    w = rows["in_window"]
    out = []
    for arm in ("a", "b", "d", "e", "f", "s"):
        sc = an.arm_crps(world=world, arm=arm, rows=rows, seeds=SEEDS).mean(axis=0)
        for day in range(1, 11):
            m = rows["sunny"] & w & (rows["lead_day"] == day)
            mean, lo, hi = a639.bootstrap(diff=sc[m], day=rows["day"][m], level=0.95)
            out.append({"arm": arm, "lead_day": day, "crps": mean, "lo": lo, "hi": hi})
    return pl.DataFrame(out).with_columns(pl.col("arm").replace(NAMES).alias("name"))


def headline(*, world: str, title: str, name: str) -> None:
    """Draw CRPS by lead day for every arm."""
    df = by_lead_day(world=world)
    dom = list(NAMES.values())
    colour = alt.Color(
        "name:N",
        sort=dom,
        scale=alt.Scale(domain=dom, range=[COLOURS[a] for a in NAMES]),
        legend=alt.Legend(title=None, orient="bottom", columns=2),
    )
    base = alt.Chart(df).encode(
        x=alt.X(
            "lead_day:O",
            title="Lead day (from the power-forecast origin)",
            axis=alt.Axis(labelAngle=0),
        )
    )
    band = base.mark_errorband(opacity=0.15).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("lo:Q", title=None), y2="hi:Q", color=colour
    )
    line = base.mark_line(point=True).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y(
            "crps:Q",
            title="CRPS on sunny middays (points of capacity)",
            scale=alt.Scale(zero=False),
        ),
        color=colour,
    )
    save((band + line).properties(width=W, height=280, title=title), name)
    print(
        name,
        df.pivot(on="arm", index="lead_day", values="crps")
        .sort("lead_day")
        .with_columns(pl.exclude("lead_day").round(2)),
    )


def example(*, world: str = "real") -> None:
    """Draw one Primary's sunny day: day-ahead and day-8 forecasts of arms a and b against the metered value."""
    rows = an.load(world=world)
    q = {a: an.arm_quantiles(world=world, arm=a, seed=0, rows=rows) for a in ("a", "b")}
    prim = int(np.unique(rows["primary"])[0])
    sunny_days = np.unique(
        rows["valid_date"][
            (rows["primary"] == prim) & rows["sunny"] & rows["in_window"] & (rows["cutoff"] == 2)
        ]
    )
    day = int(sunny_days[len(sunny_days) // 2])
    frames = []
    for lead_day, label in ((1, "Lead day 1"), (8, "Lead day 8")):
        m = (rows["primary"] == prim) & (rows["valid_date"] == day) & (rows["lead_day"] == lead_day)
        for arm in ("a", "b"):
            frames.append(
                pl.DataFrame(
                    {
                        "hour": (rows["t_us"][m] % 86_400_000_000) / 3.6e9,
                        "q05": q[arm][m, 1] * 100,
                        "q50": q[arm][m, 5] * 100,
                        "q95": q[arm][m, 9] * 100,
                        "actual": rows["y"][m] * 100,
                        "arm": NAMES[arm],
                        "panel": label,
                    }
                )
            )
    df = pl.concat(frames).sort("hour")
    base = alt.Chart(df).encode(x=alt.X("hour:Q", title="Hour of day (UTC)"))
    band = base.mark_area(opacity=0.2).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("q05:Q", title="Net demand (points of capacity)"),
        y2="q95:Q",
        color=alt.Color(
            "arm:N",
            scale=alt.Scale(domain=[NAMES["a"], NAMES["b"]], range=[COLOURS["a"], COLOURS["b"]]),
            legend=alt.Legend(title=None, orient="bottom"),
        ),
    )
    med = base.mark_line().encode(y="q50:Q", color="arm:N")  # ty: ignore[unresolved-attribute]
    act = base.mark_line(color="black", strokeDash=[4, 3]).encode(y="actual:Q")  # ty: ignore[unresolved-attribute]
    save(
        alt.layer(band, med, act)
        .properties(width=W // 2 - 20, height=240)
        .facet(column=alt.Column("panel:N", title=None))
        .properties(
            title="One day at one Primary: q05-q95 band, median, and metered value (dashed)"
        ),
        "example_day",
    )


def main() -> None:
    """Draw every figure and print the contrasts file."""
    DOCS.mkdir(parents=True, exist_ok=True)
    headline(
        world="real",
        title="Real months: CRPS on sunny middays by lead day",
        name="crps_by_lead_real",
    )
    headline(
        world="step",
        title="Step world: CRPS on sunny middays by lead day",
        name="crps_by_lead_step",
    )
    example()
    print(
        json.dumps(
            json.loads((fit.WORK / "analysis" / "contrasts_stage1.json").read_text())["real"][
                "contrasts"
            ]["all_half_hours"],
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
