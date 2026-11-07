# ruff: noqa: E501, D103
"""Experiment 610 figures: follow against event-free CRPS cost, per arm, at each metering group.

Reads `score/arm_summary.csv` written by `e00610_score.py`. Each point is one arm; the vertical bar is
the printed minimum detectable effect of the event-free CRPS contrast against A1, and the dashed
lines (dark grey) mark the keep rule (follow at most 0.05, and cost at most the bar's half-height
above zero).
"""

from pathlib import Path

import altair as alt
import polars as pl
from plotting import ocf_theme as theme
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00610-nra-neighbour-summary-lag-encoder-inputs-iteration"
DOCS = Path(__file__).resolve().parents[2] / "docs/ml-experiments" / NAME
SCORE = ML_EXPERIMENTS_DATA_DIR / NAME / "score"
GROUPS = {
    "MVA-only eight": ("the eight MVA Primaries", "eight_mva"),
    "signed-MW six": ("the six MW Primaries", "six_mw"),
}
LABELLED = ["P7", "P7+D50", "P7+BN", "P7+O3", "B3", "B1-14+D50", "P7+O3+D50", "P7+D50-p"]
"""The arms named on the chart; the rest are in the tables."""
FAMILY = [
    ("P7 (experiment 606)", theme.DATA_BLUE),
    ("Other summary span", theme.DATA_SKY),
    ("One change to P7", theme.DATA_GREEN),
    ("Changes combined", theme.DATA_PURPLE),
    ("Placebo: shuffled neighbours", theme.DATA_AMBER),
]


def family(name: str) -> str:
    if name == "P7":
        return FAMILY[0][0]
    if name.endswith("-p"):
        return FAMILY[4][0]
    if name.startswith(("B", "E")) and "+" not in name:
        return FAMILY[1][0]
    return FAMILY[2][0] if name.count("+") == 1 and name.startswith("P7") else FAMILY[3][0]


def chart(*, group: str, label: str) -> alt.LayerChart | alt.FacetChart:
    s = pl.read_csv(SCORE / "arm_summary.csv", infer_schema_length=None)
    s = s.filter(
        ~pl.col("system").str.ends_with("-rs")
        & pl.col("system").is_in(["A1", "R", "W", "G4T"]).not_()
    )
    cost, mde = f"ef_crps_minus_A1_{group}", f"ef_crps_minus_A1_{group}_mde"
    d = (
        s.select(arm="system", follow="follow_q50_days4-end", cost=cost, mde=mde)
        .drop_nulls()
        .with_columns(
            family=pl.col("arm").map_elements(family, return_dtype=pl.String),
            lo=pl.col("cost") - pl.col("mde"),
            hi=pl.col("cost") + pl.col("mde"),
        )
    )
    scale = alt.Scale(domain=[f for f, _ in FAMILY], range=[c for _, c in FAMILY])
    colour = alt.Color(
        "family:N",
        scale=scale,
        legend=alt.Legend(title=None, orient="bottom", columns=2, labelLimit=400),
    )
    base = alt.Chart(d)
    zone = pl.DataFrame({"x": [0.05], "y": [0.0]})
    layers = [
        alt.Chart(zone).mark_rule(strokeDash=[4, 3], color=theme.BLACK_2).encode(x="x:Q"),  # ty: ignore[unresolved-attribute]
        alt.Chart(zone).mark_rule(strokeDash=[4, 3], color=theme.BLACK_2).encode(y="y:Q"),  # ty: ignore[unresolved-attribute]
        base.mark_rule().encode(x="follow:Q", y="lo:Q", y2="hi:Q", color=colour),  # ty: ignore[unresolved-attribute]
        base.mark_circle(size=70).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "follow:Q",
                title="Follow of the median, day 4 to the end (A1 = 0.119)",
                scale=alt.Scale(domain=[0.0, 0.13]),
            ),
            y=alt.Y(
                "cost:Q",
                title="Event-free CRPS minus A1's (points of capacity)",
                scale=alt.Scale(domain=[-0.03, 0.17]),
            ),
            color=colour,
            tooltip=["arm", "follow", "cost", "mde"],
        ),
        base.transform_filter(alt.FieldOneOfPredicate(field="arm", oneOf=LABELLED))
        .mark_text(dx=8, dy=-8, fontSize=11, align="left")
        .encode(x="follow:Q", y="cost:Q", text="arm:N", color=colour),  # ty: ignore[unresolved-attribute]
    ]
    return alt.layer(*layers).properties(
        width=620,
        height=360,
        title=alt.TitleParams(
            f"Follow against event-free CRPS cost at {label}, 24-120 h ahead",
            subtitle="Dots: arms. Vertical bars: minimum detectable effect. Dashed black lines: follow 0.05 and cost 0.",
        ),
    )


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    for group, (label, tag) in GROUPS.items():
        chart(group=group, label=label).save(str(DOCS / f"follow_vs_cost_{tag}.svg"))


if __name__ == "__main__":
    main()
