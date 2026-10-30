# ruff: noqa: E501
"""Draw the figures of experiment 389's write-up.

The numbers are those printed by `e00389_score.py` (`work/scores/score_final.log`), typed in so the
charts need no re-score: pooled 99% intervals with 20,000 resamples, days 1 to 10.

    uv run --no-sync python ml-experiments/00389-*/e00389_charts.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00389-xgboost-tail-calibration-and-ensemble-spread-columns"
)
WIDTH: Final[int] = 680
WRAP: Final[int] = 110
GROUPS: Final[dict[str, str]] = {"signed-MW six": ocf.DATA_BLUE, "MVA-only eight": ocf.BRAND_ORANGE}
CONTRASTS: Final[list[tuple[str, str, float, float, float]]] = [
    # label, group, difference, lower, upper (CRPS, points of capacity; arm minus its reference)
    ("No widening: factor 1.0 minus 1.15 (weekly)", "signed-MW six", 0.014, -0.009, 0.045),
    ("No widening: factor 1.0 minus 1.15 (weekly)", "MVA-only eight", -0.014, -0.031, 0.008),
    ("Per-Primary factors (T2) minus 1.15 (weekly)", "signed-MW six", 0.025, -0.023, 0.066),
    ("Per-Primary factors (T2) minus 1.15 (weekly)", "MVA-only eight", 0.000, -0.015, 0.015),
    ("Per-Primary offsets (T3) minus 1.15 (weekly)", "signed-MW six", 0.005, -0.015, 0.021),
    ("Per-Primary offsets (T3) minus 1.15 (weekly)", "MVA-only eight", 0.000, -0.007, 0.007),
    ("Spread columns minus stored monthly member", "signed-MW six", -0.047, -0.078, -0.008),
    ("Spread columns minus stored monthly member", "MVA-only eight", -0.035, -0.052, -0.019),
]
COVERAGE: Final[list[tuple[str, str, str, float]]] = [
    # arm, group, quantile, coverage
    ("Weekly member, 1.15 (baseline)", "signed-MW six", "q95", 0.923),
    ("Weekly member, 1.15 (baseline)", "signed-MW six", "q99", 0.973),
    ("Weekly member, 1.15 (baseline)", "MVA-only eight", "q99", 0.995),
    ("Per-Primary factors (T2)", "signed-MW six", "q95", 0.932),
    ("Per-Primary factors (T2)", "signed-MW six", "q99", 0.976),
    ("Per-Primary factors (T2)", "MVA-only eight", "q99", 0.994),
    ("Per-Primary offsets (T3)", "signed-MW six", "q95", 0.932),
    ("Per-Primary offsets (T3)", "signed-MW six", "q99", 0.976),
    ("Per-Primary offsets (T3)", "MVA-only eight", "q99", 0.987),
    ("Monthly stored member, 1.15", "signed-MW six", "q95", 0.932),
    ("Monthly stored member, 1.15", "signed-MW six", "q99", 0.986),
    ("Monthly stored member, 1.15", "MVA-only eight", "q99", 0.995),
    ("Spread columns (S1), 1.15", "signed-MW six", "q95", 0.925),
    ("Spread columns (S1), 1.15", "signed-MW six", "q99", 0.984),
    ("Spread columns (S1), 1.15", "MVA-only eight", "q99", 0.994),
]
COLD: Final[list[tuple[str, str, float]]] = [
    ("Weekly member (baseline)", "signed-MW six", 0.758),
    ("Weekly bundle of experiment 381", "signed-MW six", 0.773),
    ("Monthly stored member", "signed-MW six", 0.775),
    ("Spread columns (S1)", "signed-MW six", 0.768),
    ("BG2", "signed-MW six", 0.799),
    ("Weekly member (baseline)", "MVA-only eight", 0.846),
    ("Weekly bundle of experiment 381", "MVA-only eight", 0.848),
    ("Monthly stored member", "MVA-only eight", 0.851),
    ("Spread columns (S1)", "MVA-only eight", 0.850),
    ("BG2", "MVA-only eight", 0.844),
]


def titled(*, title: str, subtitle: str) -> alt.TitleParams:
    """Return a title and subtitle wrapped to the plot width."""
    return alt.TitleParams(
        text=textwrap.wrap(title, 85), subtitle=textwrap.wrap(subtitle, WRAP), anchor="start"
    )


def colour(*, field: str = "group") -> alt.Color:
    """Return the group colour encoding."""
    return alt.Color(
        f"{field}:N",
        scale=alt.Scale(domain=list(GROUPS), range=list(GROUPS.values())),
        legend=alt.Legend(title=None, orient="top"),
    )


def figure_contrasts() -> None:
    """Write figure 1: the CRPS contrasts with 99% intervals."""
    frame = pl.DataFrame(
        CONTRASTS, schema=["arm", "group", "difference", "lower", "upper"], orient="row"
    )
    order = list(dict.fromkeys(c[0] for c in CONTRASTS))
    base = alt.Chart(frame).encode(
        y=alt.Y("arm:N", sort=order, title=None, axis=alt.Axis(labelLimit=330)),
        yOffset="group:N",
        color=colour(),
    )
    bars = base.mark_rule(size=2).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "lower:Q", title="Arm minus its reference (points of capacity; negative: arm is better)"
        ),
        x2="upper:Q",
    )
    dots = base.mark_point(filled=True, size=50).encode(x="difference:Q")  # ty: ignore[unresolved-attribute]
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))
        .mark_rule(color="#808080", strokeDash=[4, 3])
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )  # ty: ignore[unresolved-attribute]
    chart = (zero + bars + dots).properties(
        width=WIDTH - 400,
        height=230,
        title=titled(
            title="Spread columns lower the CRPS by 0.035 to 0.047 points; tail corrections change it by under 0.03",
            subtitle=(
                "Quantile-score CRPS, days 1 to 10, shared half-hours of 2025-07 to 2026-06. Dot: estimate. "
                "Line: 99% interval from resampling whole months. Dashed line: no change. The first three "
                "rows use weekly refits; the last row uses monthly refits."
            ),
        ),
    )
    chart.save(DOCS_DIR / "figure_1_contrasts.svg")


def figure_coverage() -> None:
    """Write figure 2: the share of outcomes below the tail quantiles, by arm."""
    frame = pl.DataFrame(COVERAGE, schema=["arm", "group", "quantile", "coverage"], orient="row")
    frame = frame.with_columns(
        panel=pl.format("{} {}", pl.col("group"), pl.col("quantile")),
        nominal=pl.when(pl.col("quantile") == "q95").then(0.95).otherwise(0.99),
    )
    order = list(dict.fromkeys(c[0] for c in COVERAGE))
    panels = ["signed-MW six q95", "signed-MW six q99", "MVA-only eight q99"]
    charts = []
    for panel in panels:
        part = frame.filter(pl.col("panel") == panel)
        nominal = float(part["nominal"][0])
        points = (
            alt.Chart(part)
            .mark_point(filled=True, size=70)
            .encode(  # ty: ignore[unresolved-attribute]
                y=alt.Y(
                    "arm:N",
                    sort=order,
                    title=None,
                    axis=alt.Axis(labels=panel == panels[0], labelLimit=240),
                ),
                x=alt.X(
                    "coverage:Q",
                    scale=alt.Scale(domain=[0.9, 1.0]),
                    axis=alt.Axis(values=[0.9, 0.95, 1.0]),
                    title=f"Share below {panel.split()[-1]}",
                ),
                color=colour(),
            )
        )
        rule = (
            alt.Chart(pl.DataFrame({"x": [nominal]}))
            .mark_rule(color="#808080", strokeDash=[4, 3])
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X(
                    "x:Q",
                    scale=alt.Scale(domain=[0.9, 1.0]),
                    axis=alt.Axis(values=[0.9, 0.95, 1.0]),
                )
            )
        )  # ty: ignore[unresolved-attribute]
        charts.append((rule + points).properties(width=(WIDTH - 330) // 3, height=150, title=panel))
    chart = alt.hconcat(*charts).properties(
        title=titled(
            title="No arm brings the upper tail of the signed-MW six to its nominal level",
            subtitle=(
                "Share of outcomes at or below the quantile, days 1 to 10, shared half-hours of 2025-07 to "
                "2026-06. Dashed line: the nominal level. The monthly rows use monthly refits."
            ),
        )
    )
    chart.save(DOCS_DIR / "figure_2_coverage.svg")


def figure_cold() -> None:
    """Write figure 3: the share of cold-night outcomes below q95."""
    frame = pl.DataFrame(COLD, schema=["arm", "group", "coverage"], orient="row")
    order = list(dict.fromkeys(c[0] for c in COLD))
    points = (
        alt.Chart(frame)
        .mark_point(filled=True, size=70)
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("arm:N", sort=order, title=None),
            x=alt.X(
                "coverage:Q",
                scale=alt.Scale(domain=[0.7, 1.0]),
                title="Share of cold-night outcomes below q95",
            ),
            color=colour(),
        )
    )
    rule = (
        alt.Chart(pl.DataFrame({"x": [0.95]}))
        .mark_rule(color="#808080", strokeDash=[4, 3])
        .encode(x=alt.X("x:Q", scale=alt.Scale(domain=[0.7, 1.0])))  # ty: ignore[unresolved-attribute]
    )  # ty: ignore[unresolved-attribute]
    chart = (rule + points).properties(
        width=WIDTH - 250,
        height=170,
        title=titled(
            title="On cold nights every arm covers only 76% to 85% of outcomes with its 95% quantile",
            subtitle=(
                "Share of cold-night half-hours (days 1 to 10) at or below q95, all arms with factor 1.15. "
                "Dashed line: nominal 95%."
            ),
        ),
    )
    chart.save(DOCS_DIR / "figure_3_cold_nights.svg")


if __name__ == "__main__":
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    figure_contrasts()
    figure_coverage()
    figure_cold()
