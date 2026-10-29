"""Draw the figures of experiment 386's write-up.

The numbers are those printed by `e00386_score_a.py` and `e00386_score_b.py` (per-series 95%
intervals with 2,000 resamples, pooled 99% intervals with 20,000 resamples), typed in so the charts
need no re-score.

    uv run --no-sync python ml-experiments/00386-*/e00386_charts.py
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
    / "00386-bsp-gsp-demand-v1-top-down-bg2-gam-simple-xgboost"
)
WIDTH: Final[int] = 640
WRAP: Final[int] = 105
COLOURS: Final[dict[str, str]] = {"BSP": ocf.DATA_BLUE, "GSP": ocf.BRAND_ORANGE}
ROWS: Final[list[tuple[str, str, float, float, float]]] = [
    ("Pooled BSPs (99%)", "BSP", 1.022, 0.677, 1.427),
    ("Pooled GSPs (99%)", "GSP", 2.010, 1.405, 2.598),
    ("D13", "BSP", 0.554, 0.107, 0.982),
    ("D17", "BSP", 1.491, 1.117, 1.889),
    ("D1", "GSP", 2.536, 1.508, 3.407),
    ("D6", "GSP", 1.509, 1.040, 1.936),
]
SCORES: Final[list[tuple[str, str, float]]] = [
    (series, arm, value)
    for series, values in {
        "D13": (4.676, 5.230, 5.466, 5.624),
        "D17": (7.063, 8.554, 8.394, 8.819),
        "D1": (14.065, 16.600, 16.967, 16.981),
        "D6": (13.682, 15.191, 15.497, 15.552),
    }.items()
    for arm, value in zip(
        ("BG2", "GAM stack", "Manual heuristic", "Climatology"), values, strict=True
    )
]
ARM_COLOURS: Final[dict[str, str]] = {
    "BG2": ocf.DATA_BLUE,
    "GAM stack": ocf.BRAND_ORANGE,
    "Manual heuristic": "#808080",
    "Climatology": "#c0c0c0",
}


def headline() -> None:
    """Write the contrast chart."""
    frame = pl.DataFrame(
        ROWS, schema=["row", "group", "difference", "lower", "upper"], orient="row"
    )
    order = [r[0] for r in ROWS]
    base = alt.Chart(frame).encode(
        y=alt.Y("row:N", sort=order, title=None),
        color=alt.Color(
            "group:N",
            scale=alt.Scale(domain=list(COLOURS), range=list(COLOURS.values())),
            legend=alt.Legend(title=None, orient="top"),
        ),
    )
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))
        .mark_rule(color="#808080", strokeDash=[4, 3])
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    bars = base.mark_rule(size=2).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "lower:Q", title="CRPS of GAM stack minus BG2 (points of capacity; negative is better)"
        ),
        x2="upper:Q",
    )
    dots = base.mark_point(filled=True, size=50).encode(x="difference:Q")  # ty: ignore[unresolved-attribute]
    chart = (zero + bars + dots).properties(
        width=WIDTH - 200,
        height=30 * len(ROWS),
        title=alt.TitleParams(
            text=textwrap.wrap("The GAM stack trails BG2 at all four series", 80),
            subtitle=textwrap.wrap(
                "Days 1 to 10, shared half-hours of 2025-07 to 2026-06. BSPs are bulk supply "
                "points and GSPs are grid supply points. Dot: estimate. Line: 95% interval per "
                "series and 99% interval for pooled rows, from resampling whole months. Dashed "
                "line: same as BG2. The stack is the one of experiment 377, without S below 168 h.",
                WRAP,
            ),
            anchor="start",
        ),
    )
    chart.save(DOCS_DIR / "headline.svg")


def scores() -> None:
    """Write the absolute-score chart."""
    frame = pl.DataFrame(SCORES, schema=["series", "arm", "crps"], orient="row")
    chart = (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            x=alt.X("arm:N", sort=list(ARM_COLOURS), title=None, axis=alt.Axis(labelAngle=-40)),
            y=alt.Y("crps:Q", title="CRPS (points of capacity)"),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=list(ARM_COLOURS), range=list(ARM_COLOURS.values())),
                legend=None,
            ),
            column=alt.Column("series:N", sort=["D13", "D17", "D1", "D6"], title=None),
        )
        .properties(
            width=110,
            height=200,
            title=alt.TitleParams(
                text="BG2 scores best at every series; the GAM stack is close to the references",
                subtitle=textwrap.wrap(
                    "CRPS, days 1 to 10, shared half-hours. D13 and D17 are bulk supply points; "
                    "D1 and D6 are grid supply points. Lower is better.",
                    WRAP,
                ),
                anchor="start",
            ),
        )
    )
    chart.save(DOCS_DIR / "absolute_scores.svg")


if __name__ == "__main__":
    headline()
    scores()
