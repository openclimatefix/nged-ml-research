"""Draw the figures of experiment 380.

The numbers are those printed by `e00380_score.py` and `e00380_jaggedness.py`, typed in so the
charts need no re-score, plus the jaggedness table.

    uv run --no-sync python ml-experiments/00380-*/e00380_charts.py
"""

from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final[str] = "00380-gam-stack-ablation-and-jaggedness-at-16-primaries"
DOCS: Final[Path] = Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / NAME
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / NAME / "work"
WIDTH: Final[int] = 600
COLOURS: Final[dict[str, str]] = {
    "signed-MW six": ocf.DATA_BLUE,
    "MVA-only eight, with D8": ocf.BRAND_ORANGE,
    "MVA-only seven, without D8": ocf.DATA_PURPLE,
}
# (stage, group, difference, lower, upper): removing the stage minus the full stack, days 1 to 10.
STAGES: Final[list[tuple[str, str, float, float, float]]] = [
    ("Kalman filter", "signed-MW six", 0.121, 0.026, 0.219),
    ("Kalman filter", "MVA-only eight, with D8", 0.465, 0.179, 0.821),
    ("Kalman filter", "MVA-only seven, without D8", 0.330, 0.120, 0.604),
    ("Winter night offset", "signed-MW six", -0.002, -0.011, 0.008),
    ("Winter night offset", "MVA-only eight, with D8", -0.049, -0.116, -0.005),
    ("Winter night offset", "MVA-only seven, without D8", -0.003, -0.020, 0.012),
    ("Recipe R", "signed-MW six", 0.027, 0.014, 0.040),
    ("Recipe R", "MVA-only eight, with D8", 0.095, 0.047, 0.149),
    ("Recipe R", "MVA-only seven, without D8", 0.079, 0.043, 0.120),
    ("Per-fold spread", "signed-MW six", -0.001, -0.015, 0.011),
    ("Per-fold spread", "MVA-only eight, with D8", 0.270, 0.138, 0.420),
    ("Per-fold spread", "MVA-only seven, without D8", 0.017, -0.006, 0.044),
]
KALMAN_BY_LEAD: Final[list[tuple[str, str, float, float, float]]] = [
    ("Day 1", "signed-MW six", 0.499, 0.204, 0.939),
    ("Day 1", "MVA-only eight, with D8", 0.646, 0.340, 0.963),
    ("Days 1 to 10", "signed-MW six", 0.121, 0.026, 0.219),
    ("Days 1 to 10", "MVA-only eight, with D8", 0.465, 0.179, 0.821),
    ("Days 8 to 10", "signed-MW six", -0.074, -0.293, 0.115),
    ("Days 8 to 10", "MVA-only eight, with D8", 0.341, 0.029, 0.726),
    ("Days 11 to 14", "signed-MW six", -0.154, -0.455, 0.087),
    ("Days 11 to 14", "MVA-only eight, with D8", 0.303, 0.003, 0.666),
]


def interval_chart(
    *, rows: list, title: list[str], axis_title: str, domain: str, order: list[str]
) -> alt.LayerChart | alt.FacetChart:
    """Return a dot-and-interval chart with a zero rule, one row per `domain` entry and group."""
    frame = pl.DataFrame(
        rows, schema=["row", "group", "difference", "lower", "upper"], orient="row"
    )
    base = alt.Chart(frame).encode(
        y=alt.Y("row:N", sort=order, title=None, axis=alt.Axis(labelLimit=300)),
        yOffset=alt.YOffset("group:N", sort=list(COLOURS)),
        color=alt.Color(
            "group:N",
            scale=alt.Scale(domain=list(COLOURS), range=list(COLOURS.values())),
            legend=alt.Legend(title=None, orient="top"),
        ),
    )
    rule = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))
        .mark_rule(color="#000000", strokeDash=[3, 3])
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    bars = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("lower:Q", title=axis_title), x2="upper:Q"
    )
    dots = base.mark_circle(size=60).encode(x="difference:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(rule, bars, dots).properties(
        width=WIDTH,
        height=260,
        title=alt.TitleParams(title, anchor="start", fontSize=11, subtitleFontSize=10),
    )


def scatter(*, table: pl.DataFrame, measure: str, label: str) -> alt.LayerChart | alt.FacetChart:
    """Return a scatter of the stack-minus-BG2 gap against one regularity measure."""
    points = alt.Chart(table).encode(
        x=alt.X(f"{measure}:Q", title=label),
        y=alt.Y("gap:Q", title="Stack minus BG2 (points; positive: stack worse)"),
        color=alt.Color(
            "group:N",
            scale=alt.Scale(
                domain=["signed-MW", "MVA-only", "short-history"],
                range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_PURPLE],
            ),
            legend=alt.Legend(title=None, orient="top"),
        ),
    )
    return alt.layer(
        points.mark_circle(size=70),  # ty: ignore[invalid-argument-type]
        points.mark_text(dx=9, dy=-6, fontSize=9).encode(text="label:N"),  # ty: ignore[unresolved-attribute]
    ).properties(width=WIDTH, height=220)


def main() -> None:
    """Write the three SVG figures."""
    stage = interval_chart(
        rows=[(s, g, d, lo, hi) for s, g, d, lo, hi in STAGES],
        title=[
            "Cost of removing one stage from the GAM stack, days 1 to 10",
            "Dot: stage removed minus full stack (points of capacity; positive: the stage helps).",
            "Line: 99% interval from resampling whole months.",
        ],
        axis_title="Quantile score, stage removed minus full stack (points of capacity)",
        domain="stage",
        order=["Kalman filter", "Winter night offset", "Recipe R", "Per-fold spread"],
    )
    stage.save(DOCS / "figure_1_stage_costs.svg")
    lead = interval_chart(
        rows=KALMAN_BY_LEAD,
        title=[
            "The Kalman filter helps on day 1 and not beyond day 7 at the signed-MW Primaries",
            "Dot: filter removed minus full stack (points of capacity; positive: filter helps).",
            "Line: 99% interval from resampling whole months.",
        ],
        axis_title="Quantile score, filter removed minus full stack (points of capacity)",
        domain="lead",
        order=["Day 1", "Days 1 to 10", "Days 8 to 10", "Days 11 to 14"],
    )
    lead.save(DOCS / "figure_2_kalman_by_lead.svg")
    table = pl.read_parquet(WORK / "jaggedness_table.parquet")
    top = scatter(
        table=table,
        measure="m1_profile_share",
        label="Share of variance in the weekly profile (training data)",
    )
    bottom = scatter(
        table=table,
        measure="m4_step_rate",
        label="Share of half-hour changes above 5% of capacity (training data)",
    )
    alt.vconcat(top, bottom).properties(
        title=alt.TitleParams(
            [
                "The stack's gap to BG2 against two regularity measures, 16 Primaries",
                "Dot: one Primary. Neither measure shows a clear slope at n = 16.",
            ],
            anchor="start",
            fontSize=11,
            subtitleFontSize=10,
        )
    ).save(DOCS / "figure_3_gap_against_regularity.svg")


if __name__ == "__main__":
    main()
