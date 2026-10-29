"""Draw the two figures of experiment 381's write-up.

The numbers are those printed by `e00381_score.py` (screening contrasts at the 99.86% level with
20,000 resamples; final contrasts at 95% per Primary and 99% pooled), typed in so the chart needs
no re-score.

    uv run --no-sync python ml-experiments/00381-*/e00381_charts.py
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
    / "00381-simple-weekly-xgboost-demand-against-gam-stack"
)
WIDTH: Final[int] = 640
WRAP: Final[int] = 105
GROUP_COLOURS: Final[dict[str, str]] = {
    "signed-MW six": ocf.DATA_BLUE,
    "MVA-only eight": ocf.BRAND_ORANGE,
    "short history": ocf.DATA_PURPLE,
}
HEADLINE_ROWS: Final[list[tuple[str, str, float, float, float]]] = [
    ("Pooled, signed-MW (99%)", "signed-MW six", 0.075, -0.120, 0.266),
    ("Pooled, MVA-only (99%)", "MVA-only eight", -0.361, -0.602, -0.101),
    ("Pooled, no D8 (99%)", "MVA-only eight", -0.197, -0.399, -0.025),
    ("D5", "signed-MW six", -0.253, -0.456, -0.043),
    ("D9", "signed-MW six", -0.004, -0.259, 0.228),
    ("D10", "signed-MW six", 0.164, -0.138, 0.567),
    ("D11", "signed-MW six", 0.148, 0.016, 0.301),
    ("D14", "signed-MW six", 0.301, 0.042, 0.634),
    ("D16", "signed-MW six", 0.090, -0.039, 0.192),
    ("D2", "MVA-only eight", -0.193, -0.426, 0.039),
    ("D3", "MVA-only eight", 0.050, -0.079, 0.197),
    ("D4", "MVA-only eight", -0.025, -0.222, 0.258),
    ("D8", "MVA-only eight", -1.615, -2.810, -0.492),
    ("D12", "MVA-only eight", -0.799, -1.251, -0.366),
    ("D19", "MVA-only eight", -0.124, -0.295, 0.010),
    ("D21", "MVA-only eight", 0.009, -0.106, 0.126),
    ("D22", "MVA-only eight", -0.294, -0.478, -0.086),
    ("D15", "short history", 2.338, 0.143, 4.238),
    ("D18", "short history", 0.066, -0.135, 0.237),
]
SCREEN_ROWS: Final[list[tuple[str, str, float, float, float]]] = [
    ("Solar summary", "signed-MW six", -0.051, -0.086, -0.019),
    ("1-year half-life", "signed-MW six", -0.019, -0.034, -0.006),
    ("Sunny flag", "signed-MW six", -0.002, -0.003, 0.000),
    ("Economy 7 input", "signed-MW six", 0.000, -0.002, 0.002),
    ("Lead-dependent half-life", "signed-MW six", 0.007, -0.016, 0.033),
    ("12-month window", "signed-MW six", 0.043, 0.016, 0.074),
    ("0.25-year half-life", "signed-MW six", 0.072, 0.006, 0.136),
    ("Solar summary ", "MVA-only eight", -0.016, -0.049, 0.023),
    ("1-year half-life ", "MVA-only eight", 0.004, -0.016, 0.021),
    ("Sunny flag ", "MVA-only eight", -0.002, -0.005, 0.000),
    ("Economy 7 input ", "MVA-only eight", 0.000, -0.002, 0.002),
    ("Lead-dependent half-life ", "MVA-only eight", 0.025, 0.008, 0.039),
    ("12-month window ", "MVA-only eight", 0.024, 0.002, 0.072),
    ("0.25-year half-life ", "MVA-only eight", 0.039, -0.021, 0.088),
]


def draw(*, rows: list[tuple[str, str, float, float, float]], title: str, subtitle: str,
         x_title: str, name: str) -> None:  # fmt: skip
    """Write one interval chart to `name`."""
    frame = pl.DataFrame(
        rows, schema=["row", "group", "difference", "lower", "upper"], orient="row"
    )
    order = [r[0] for r in rows]
    base = alt.Chart(frame).encode(
        y=alt.Y("row:N", sort=order, title=None),
        color=alt.Color(
            "group:N",
            scale=alt.Scale(domain=list(GROUP_COLOURS), range=list(GROUP_COLOURS.values())),
            legend=alt.Legend(title=None, orient="top"),
        ),
    )
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))
        .mark_rule(color="#808080", strokeDash=[4, 3])
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    bars = base.mark_rule(size=2).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("lower:Q", title=x_title), x2="upper:Q"
    )
    dots = base.mark_point(filled=True, size=50).encode(x="difference:Q")  # ty: ignore[unresolved-attribute]
    chart = (zero + bars + dots).properties(
        width=WIDTH - 200,
        height=22 * len(rows),
        title=alt.TitleParams(
            text=textwrap.wrap(title, 80),
            subtitle=textwrap.wrap(subtitle, WRAP),
            anchor="start",
        ),
    )
    chart.save(DOCS_DIR / name)


def main() -> None:
    """Write the two SVG files."""
    draw(
        rows=HEADLINE_ROWS,
        title="The simple XGBoost model trails the GAM stack at some signed-MW Primaries and "
        "beats it at most MVA-only ones",
        subtitle="Weekly bundle minus the GAM stack, CRPS at days 1 to 10 on the shared "
        "half-hours of 2025-07 to 2026-06. Negative is better for the XGBoost model. Dot: "
        "estimate. Line: 95% interval per Primary and 99% interval for pooled rows, from "
        "resampling whole months. D15 and D18 are report only.",
        x_title="CRPS of the weekly bundle minus the GAM stack (points of capacity)",
        name="headline.svg",
    )
    draw(
        rows=SCREEN_ROWS,
        title="Of seven changes, a solar summary and a 1-year half-life helped at the signed-MW "
        "Primaries",
        subtitle="Each arm minus the monthly control, CRPS at days 1 to 10, pooled over the "
        "group. Dot: estimate. Line: 99.86% interval (Bonferroni level for seven arms) from "
        "resampling whole months. Negative is better.",
        x_title="CRPS of the arm minus the control (points of capacity)",
        name="screening.svg",
    )


if __name__ == "__main__":
    main()
