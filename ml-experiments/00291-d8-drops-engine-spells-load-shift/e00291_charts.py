# ruff: noqa: E501
"""Draw the figures of experiment 291 from the tables that `e00291_analysis.py` wrote.

Afterwards optimise each with `npx svgo@4 --multipass --precision=1 --final-newline`.
"""

import json
import sys
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from e00291_common import DATA_DIR

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import engine_figure_guard as guard

OUT_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00291-d8-drops-engine-spells-load-shift"
)
WIDTH: Final[int] = 640
WRAP: Final[int] = 95


def wrapped(text: str) -> list[str]:
    """Wrap a title or subtitle to the plot area's width."""
    return textwrap.wrap(text, WRAP)


def figure_overlap() -> alt.Chart:
    """Share of drops with engine-running states, against matched placebo windows and controls."""
    s = json.loads((DATA_DIR / "summary.json").read_text())
    rows = [
        ("D8's 28 drops", s["v1_observed"], None, None, "drops"),
        (
            "D8 matched placebo windows",
            s["v1_placebo"],
            s["v1_placebo_lo"],
            s["v1_placebo_hi"],
            "placebo",
        ),
        ("D5, same dates as the drops", s["D5_own"], None, None, "control"),
        ("D10, same dates as the drops", s["D10_own"], None, None, "control"),
        ("D11, same dates as the drops", s["D11_own"], None, None, "control"),
    ]
    frame = pl.DataFrame(rows, schema=["row", "share", "lo", "hi", "kind"], orient="row")
    colours = alt.Scale(
        domain=["drops", "placebo", "control"],
        range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_SKY],
    )
    y = alt.Y("row:N", sort=[r[0] for r in rows], title=None)
    base = alt.Chart(frame)
    bars = base.mark_bar(aria=False).encode(  # ty: ignore[unresolved-attribute]
        y=y,
        x=alt.X(
            "share:Q",
            title="Share of drops with at least half of their half-hours flagged (higher: more engine-like)",
            scale=alt.Scale(domain=[0, 1]),
        ),
        color=alt.Color("kind:N", scale=colours, legend=None),
    )
    interval = base.mark_rule(color=ocf.BLACK_1).encode(y=y, x="lo:Q", x2="hi:Q")  # ty: ignore[unresolved-attribute]
    labels = base.mark_text(align="left", dx=6, color=ocf.BLACK_1, aria=False).encode(  # ty: ignore[unresolved-attribute]
        y=y, x="share:Q", text=alt.Text("share:Q", format=".2f")
    )
    return (bars + interval + labels).properties(
        width=WIDTH - 230,
        height=170,
        title=alt.TitleParams(
            "Engine-running states cover 27 of D8's 28 drops, against about 1 in 5 matched windows",
            subtitle=wrapped(
                "Bar: share of drops (or of placebo windows) whose half-hours are at least half flagged as "
                "engine running by experiment 196's detector. Black line: 95% of 5,000 placebo draws. "
                "Controls: the same detector on D5, D10, and D11 over the dates of D8's drops."
            ),
            anchor="start",
        ),
    )


def figure_depth() -> alt.Chart:
    """Histogram of drop depths in engine units, with the whole numbers marked."""
    table = pl.read_parquet(DATA_DIR / "drops_overlap.parquet").filter(pl.col("in_scope"))
    hist = (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_bar(color=ocf.DATA_BLUE, aria=False)
        .encode(
            x=alt.X(
                "depth_in_units:Q",
                bin=alt.Bin(step=0.25, extent=[0, 3]),
                title="Drop depth in units of the engine's recorded step",
                axis=alt.Axis(values=[0, 1, 2, 3]),
            ),
            y=alt.Y("count()", title="Drops", axis=alt.Axis(tickMinStep=1)),
        )
    )
    rules = (
        alt.Chart(pl.DataFrame({"unit": [1.0, 2.0, 3.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color=ocf.BRAND_ORANGE, strokeDash=[4, 3], aria=False)
        .encode(x="unit:Q")
    )
    return (hist + rules).properties(
        width=WIDTH - 60,
        height=170,
        title=alt.TitleParams(
            "Drop depths spread between whole engine units",
            subtitle=wrapped(
                "Bars: the 28 drops in scope. Dashed lines: 1, 2, and 3 units. If the drops were "
                "whole numbers of units, the bars would stand at the lines."
            ),
            anchor="start",
        ),
    )


def figure_fade() -> alt.VConcatChart:
    """D8's detector flag share and left-tail share, and drops started, pooled over two eras."""
    eras = ["Before the shortfall fades", "After the shortfall fades"]
    m = pl.read_parquet(DATA_DIR / "monthly.parquet").with_columns(
        era=pl.when(pl.col("month") >= "2026-03").then(pl.lit(eras[1])).otherwise(pl.lit(eras[0]))
    )
    pooled = m.group_by("era").agg(
        v1_flag_share=pl.col("v1_flag_share").mean(),
        left_tail_share=pl.col("left_tail_share").mean(),
        drops_started=pl.col("drops_started").mean(),
    )
    long = pooled.unpivot(
        on=["v1_flag_share", "left_tail_share"],
        index="era",
        variable_name="series",
        value_name="share",
    ).with_columns(
        pl.col("series").replace(
            {
                "v1_flag_share": "Half-hours flagged as engine running",
                "left_tail_share": "Half-hours in the left tail",
            }
        )
    )
    names = ["Half-hours flagged as engine running", "Half-hours in the left tail"]
    top = (
        alt.Chart(long)  # ty: ignore[unresolved-attribute]
        .mark_bar(aria=False)
        .encode(
            x=alt.X("era:N", title=None, sort=eras, axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("series:N", sort=names),
            y=alt.Y(
                "share:Q",
                title="Mean monthly share of D8's half-hours",
                scale=alt.Scale(domain=[0, 0.8]),
            ),
            color=alt.Color(
                "series:N",
                sort=names,
                scale=alt.Scale(domain=names, range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]),
                legend=alt.Legend(orient="top", title=None, direction="vertical", labelLimit=500),
            ),
        )
        .properties(width=WIDTH - 60, height=170)
    )
    drops = (
        alt.Chart(pooled)  # ty: ignore[unresolved-attribute]
        .mark_bar(color=ocf.GREY_3, opacity=0.8, aria=False, size=40)
        .encode(
            x=alt.X("era:N", title=None, sort=eras, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("drops_started:Q", title="Drops started per month"),
        )
        .properties(width=WIDTH - 60, height=70)
    )
    return alt.vconcat(top, drops).properties(
        title=alt.TitleParams(
            "D8's shortfall and its drops stop; the detector's flag does not",
            subtitle=wrapped(
                "Means over the months of each era. Grey bars: drops started per month, no drop starts after the shortfall fades."
            ),
            anchor="start",
        )
    )


def main() -> None:
    """Write the three figures as SVG."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, chart in (
        ("figure_1", figure_overlap()),
        ("figure_2", figure_depth()),
        ("figure_3", figure_fade()),
    ):
        guard.save_chart(chart=chart, path=OUT_DIR / f"{name}.svg")
        print(name, flush=True)


if __name__ == "__main__":
    main()
