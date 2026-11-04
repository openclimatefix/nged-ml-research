# ruff: noqa: D103, E501
"""Draw the figures of experiment 543 from the tables of `e00543_score.py` and `e00543_stress.py`."""

from pathlib import Path

import altair as alt
import polars as pl
from plotting import ocf_theme as theme
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SCORES = (
    ML_EXPERIMENTS_DATA_DIR
    / "00543-lag-free-network-auxiliary-recent-level-and-reverse-flow-heads/scores"
)
DOCS = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00543-lag-free-network-auxiliary-recent-level-and-reverse-flow-heads"
)
SETS = {"eight": "MVA-only eight", "six": "Signed-MW six"}
COLOURS = {
    "F1": theme.DATA_BLUE,
    "F2": theme.DATA_GREEN,
    "F3": theme.DATA_PURPLE,
    "L-1d": theme.GREY_1,
    "A-pool": theme.DATA_AMBER,
    "R-DW": theme.DATA_SKY,
}
BANDS = ["0-6h", "6-12h", "12-24h", "24-48h", "48-72h", "72-120h", "120-240h"]


def figure_horizon() -> None:
    a = pl.read_csv(SCORES / "absolute.csv").filter(
        pl.col("band").is_in(BANDS) & pl.col("arm").is_in(list(COLOURS))
    )
    panels = []
    for key, label in SETS.items():
        part = a.filter(pl.col("set") == key).to_pandas()
        panels.append(
            alt.Chart(part)  # ty: ignore[unresolved-attribute]
            .mark_line(point=True)
            .encode(
                x=alt.X("band:N", sort=BANDS, title="Lead time"),
                y=alt.Y("crps:Q", title="CRPS (points of capacity)", scale=alt.Scale(zero=False)),
                color=alt.Color(
                    "arm:N",
                    scale=alt.Scale(domain=list(COLOURS), range=list(COLOURS.values())),
                    title=None,
                ),
            )
            .properties(width=520, height=200, title=alt.TitleParams(label, anchor="start"))
        )
    alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "CRPS by lead band, five folds, rows with no labelled event.",
                "Lower is better; A-pool and R-DW read lagged power, the others do not.",
            ],
            anchor="start",
        )
    ).save(DOCS / "figure_1_crps_by_horizon.svg")


def figure_contrasts() -> None:
    c = pl.read_csv(SCORES / "contrasts.csv").filter(
        pl.col("band") == "24-72h",
        pl.col("contrast").is_in(["F1 - L-1d", "F2 - L-1d", "F3 - L-1d"]),
        pl.col("set").is_in(list(SETS)),
    )
    c = c.with_columns(
        arm=pl.col("contrast").str.replace(" - L-1d", ""), group=pl.col("set").replace(SETS)
    )
    panels = []
    for label in SETS.values():
        part = c.filter(pl.col("group") == label).to_pandas()
        base = alt.Chart(part).encode(y=alt.Y("arm:N", sort=["F1", "F2", "F3"], title=None))
        col = alt.Color(
            "arm:N",
            scale=alt.Scale(domain=list(COLOURS)[:3], range=list(COLOURS.values())[:3]),
            legend=None,
        )
        bar = base.mark_rule(strokeWidth=3).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "low_bonferroni_month:Q",
                title="CRPS minus L-1d (points of capacity; below zero is better)",
            ),
            x2="high_bonferroni_month:Q",
            color=col,
        )
        dot = base.mark_point(filled=True, size=70).encode(x="difference:Q", color=col)  # ty: ignore[unresolved-attribute]
        zero = (
            alt.Chart(pl.DataFrame({"x": [0.0]}).to_pandas())  # ty: ignore[unresolved-attribute]
            .mark_rule(color="black")
            .encode(x="x:Q")
        )
        panels.append(
            alt.layer(zero, bar, dot).properties(
                width=520, height=110, title=alt.TitleParams(label, anchor="start")
            )
        )
    alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "Each head against L-1d, lead 24 to 72 h, month-block interval (5 blocks, near min-to-max).",
                "The gap to A-pool that a head had to close is 0.27 at the eight and 0.08 at the six.",
            ],
            anchor="start",
        )
    ).save(DOCS / "figure_2_head_minus_l1d.svg")


def figure_leak() -> None:
    s = (
        pl.read_csv(SCORES / "stress_leak.csv")
        .filter(pl.col("band").is_in(["day1", "days2_3", "days4_10"]))
        .to_pandas()
    )
    alt.Chart(s).mark_bar().encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "band:N",
            sort=["day1", "days2_3", "days4_10"],
            title="Lead time of the forecast (day 1, days 2-3, days 4-10)",
            axis=alt.Axis(labelAngle=0),
        ),
        xOffset=alt.XOffset("arm:N", sort=list(COLOURS)),
        y=alt.Y("leak:Q", title="Injected-shift leak (points of capacity)"),
        color=alt.Color(
            "arm:N",
            scale=alt.Scale(
                domain=["L-1d", "F1", "F2", "F3"],
                range=[COLOURS[a] for a in ("L-1d", "F1", "F2", "F3")],
            ),
            title=None,
        ),
    ).properties(
        width=520,
        height=220,
        title=alt.TitleParams(
            "Median forecast moved by an injected 7-day shift, mean of five folds", anchor="start"
        ),
    ).save(DOCS / "figure_3_stress_leak.svg")


if __name__ == "__main__":
    figure_horizon()
    figure_contrasts()
    figure_leak()
