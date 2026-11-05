# ruff: noqa: ANN201, D103, E501
"""Figure 1 of experiment 569: B3a school term, real minus placebo per Primary, beside the seed noise of plain M0."""

from pathlib import Path
from typing import Final

import altair as alt
import polars as pl
from plotting import ocf_theme as theme
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final = (
    "00569-modular-network-inputs-through-matching-encoders-in-pretraining-and-fine-tuning-run-1"
)
DOCS: Final[Path] = Path(__file__).resolve().parents[2] / "docs/ml-experiments" / NAME


def main():
    t = pl.read_csv(ML_EXPERIMENTS_DATA_DIR / NAME / "school_per_primary.csv").sort(
        "real_minus_placebo"
    )
    order = t["label"].to_list()
    d = t.to_pandas()
    base = alt.Chart(d).encode(y=alt.Y("label:N", sort=order, title=None))
    x = alt.X(
        "real_minus_placebo_low:Q",
        title="CRPS difference (points of capacity; negative favours the school-term columns)",
        scale=alt.Scale(domain=[-1.0, 1.0]),
    )
    bar = base.mark_rule(color=theme.DATA_BLUE, strokeWidth=3).encode(
        x=x, x2="real_minus_placebo_high:Q"
    )  # ty: ignore[unresolved-attribute]
    dot = base.mark_point(color=theme.BRAND_ORANGE, filled=True, size=70, aria=False).encode(
        x="real_minus_placebo:Q"
    )  # ty: ignore[unresolved-attribute]
    noise = base.mark_point(color="grey", shape="cross", size=70, aria=False).encode(
        x="seed_noise:Q"
    )  # ty: ignore[unresolved-attribute]
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}).to_pandas()).mark_rule(color="black").encode(x="x:Q")
    )  # ty: ignore[unresolved-attribute]
    chart = alt.layer(zero, bar, dot, noise).properties(
        width=470,
        height=380,
        title=alt.TitleParams(
            [
                "The spread of school-term effects across Primaries is smaller than the seed noise.",
                "Orange dot: school-term minus shuffled-day placebo (1 seed each). Bar: 95% 5-month interval.",
                "Grey cross: plain M0 seed 0 minus seed 1 (pure noise). All rows. Interim.",
            ],
            anchor="start",
        ),
    )
    chart.save(DOCS / "figure_1_school_term_per_primary.svg")


LABEL: Final = {
    "B1 - B1p": "B1 thermal lag, solar gain",
    "B2 - B2p": "B2 look-ahead",
    "B3a - B3ap": "B3a school term",
    "B3b - B3bp": "B3b national signals",
    "B4 - B4p": "B4 price shape",
    "B3c - B3a": "B3c school proximity (vs B3a)",
}
SLICES: Final = {
    "all": "All rows",
    "cold_night": "Cold nights",
    "christmas_week": "Christmas week",
    "high_spread": "High spread",
}


def figure_2():
    c = pl.read_csv(ML_EXPERIMENTS_DATA_DIR / NAME / "scores_contrasts.csv").filter(
        (pl.col("kind") == "primary")
        & (pl.col("scope") == "24-120h")
        & pl.col("slice").is_in(list(SLICES))
    )
    c = c.with_columns(
        arm=pl.col("contrast").replace(LABEL),
        slice_label=pl.col("slice").replace(SLICES),
        group=pl.col("set").replace({"eight": "Eight MVA Primaries", "six": "Six MW Primaries"}),
    )
    d = c.to_pandas()
    base = alt.Chart().encode(y=alt.Y("arm:N", sort=list(LABEL.values()), title=None))
    bar = base.mark_rule(color=theme.DATA_BLUE, strokeWidth=3).encode(
        x=alt.X(
            "low_bonferroni_month:Q",
            title="Real minus placebo, CRPS points (negative favours real)",
            scale=alt.Scale(domain=[-2, 2]),
        ),
        x2="high_bonferroni_month:Q",
    )  # ty: ignore[unresolved-attribute]
    dot = base.mark_point(color=theme.BRAND_ORANGE, filled=True, size=50, aria=False).encode(
        x="difference:Q"
    )  # ty: ignore[unresolved-attribute]
    zero = alt.Chart().mark_rule(color="black").encode(x=alt.datum(0))  # ty: ignore[unresolved-attribute]
    chart = (
        alt.layer(zero, bar, dot)
        .properties(width=200, height=130)
        .facet(
            data=d,
            row=alt.Row("group:N", title=None),
            column=alt.Column("slice_label:N", title=None, sort=list(SLICES.values())),
        )
        .properties(
            title=alt.TitleParams(
                [
                    "No input beats its shuffled-day placebo on any keep-rule slice at 24-120 h.",
                    "Dot: difference, 1 seed. Bar: Bonferroni (48 tests) 5-month interval, near min-to-max. Interim.",
                ],
                anchor="start",
            )
        )
    )
    chart.save(DOCS / "figure_2_real_minus_placebo_by_slice.svg")


main()
figure_2()
