# ruff: noqa: E501
"""Experiment 576: the gap against hour of day (sunny and cloudy half-hours), and the gap by Primary.

The gap is E2's CRPS minus the lag arm's, in points of capacity, at 24-120 h; positive means the lag
arm is better. Hours are UTC hours of the target. Intervals: month-block bootstrap, 5 blocks,
uncorrected.
"""

from pathlib import Path

import altair as alt
import polars as pl
from plotting import ocf_theme as theme
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00576-lag-gap-by-half-hour-type-eight-mva-and-six-mw-primaries"
WORK = ML_EXPERIMENTS_DATA_DIR / NAME / "work"
DOCS = Path(__file__).resolve().parents[2] / "docs/ml-experiments" / NAME
GROUPS = {"eight": "The eight MVA Primaries", "six": "The six MW Primaries"}
ARMS = {
    "V3": ("v3 (lags)", theme.DATA_GREEN),
    "V3D5": ("v3+D5 (lags and offset)", theme.DATA_PURPLE),
}


def zero_rule(*, horizontal: bool) -> alt.Chart:
    """Return the dashed zero line."""
    c = alt.Chart(pl.DataFrame({"z": [0.0]})).mark_rule(color="grey", strokeDash=[4, 3])
    return c.encode(y="z:Q") if horizontal else c.encode(x="z:Q")  # ty: ignore[unresolved-attribute]


def hour_figure() -> None:
    """Write the gap by hour of day, split sunny and cloudy, for v3+D5."""
    rows = pl.read_parquet(WORK / "gap_rows.parquet").filter(
        (pl.col("band") == "24-120h") & pl.col("sky").is_in(["sunny", "cloudy"])
    )
    df = (
        rows.group_by("set", "sky", "hour")
        .agg(n=pl.len(), gap=pl.col("gap_V3D5").mean())
        .filter(pl.col("n") >= 300)
        .with_columns(group=pl.col("set").replace(GROUPS), sky=pl.col("sky").str.to_titlecase())
    )
    line = (
        alt.Chart()
        .mark_line(point=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "hour:Q",
                title="Hour of day (UTC) of the forecast target",
                scale=alt.Scale(domain=[0, 23]),
            ),
            y=alt.Y("gap:Q", title="E2 minus v3+D5 gap (points)"),
            color=alt.Color(
                "sky:N",
                title=None,
                scale=alt.Scale(
                    domain=["Sunny", "Cloudy"], range=[theme.BRAND_ORANGE, theme.DATA_BLUE]
                ),
                legend=alt.Legend(orient="top"),
            ),
        )
    )  # ty: ignore[unresolved-attribute]
    chart = (
        alt.layer(zero_rule(horizontal=True), line)
        .properties(width=560, height=170)
        .facet(row=alt.Row("group:N", title=None, sort=list(GROUPS.values())), data=df)
        .properties(
            title=alt.TitleParams(
                [
                    "At the eight MVA Primaries the lag gap is as large at night as in sunny half-hours",
                    "Mean CRPS gap at 24-120 h; above zero means the lag arm is better.",
                    "Sunny: forecast shortwave at least 0.6 of clear sky, with the sun above 5 degrees.",
                    "Hours with fewer than 300 half-hours of a type are left out.",
                ],
                anchor="start",
                fontSize=11,
            )
        )
    )
    chart.save(str(DOCS / "gap_by_hour_sunny_cloudy.svg"))


def primary_figure() -> None:
    """Write the gap by Primary at 24-120 h with month-block intervals."""
    t = pl.read_csv(WORK / "gap_tables.csv").filter(
        (pl.col("dim") == "primary")
        & (pl.col("band") == "24-120h")
        & pl.col("group").is_in(list(GROUPS))
    )
    df = t.with_columns(
        group=pl.col("group").replace(GROUPS),
        arm_label=pl.col("arm").replace({k: v[0] for k, v in ARMS.items()}),
    )
    order = {
        "The eight MVA Primaries": ["D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22"],
        "The six MW Primaries": ["D5", "D9", "D10", "D11", "D14", "D16"],
    }
    panels = []
    for group, labels in order.items():
        d = df.filter(pl.col("group") == group)
        base = alt.Chart(d).encode(
            y=alt.Y("value:N", sort=labels, title=None),
            color=alt.Color(
                "arm_label:N",
                title=None,
                scale=alt.Scale(
                    domain=[v[0] for v in ARMS.values()], range=[v[1] for v in ARMS.values()]
                ),
                legend=alt.Legend(orient="top"),
            ),
            yOffset="arm_label:N",
        )
        bars = base.mark_rule().encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("low_month:Q", title="E2 minus lag arm CRPS (points of capacity)"),
            x2="high_month:Q",
        )
        dots = base.mark_point(filled=True, size=45).encode(x="difference:Q")  # ty: ignore[unresolved-attribute]
        panels.append(
            alt.layer(zero_rule(horizontal=False), bars, dots).properties(
                width=520, height=20 * len(labels) + 40, title=group
            )
        )
    chart = alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "The lag gap at the eight MVA Primaries sits at D4, D12, and D19; D8 goes the other way",
                "CRPS gap at 24-120 h; above zero means the lag arm is better. Dot: estimate (2 seeds).",
                "Line: uncorrected 95% interval from resampling whole months (5 blocks).",
            ],
            anchor="start",
            fontSize=11,
        )
    )
    chart.save(str(DOCS / "gap_by_primary.svg"))


if __name__ == "__main__":
    hour_figure()
    primary_figure()
