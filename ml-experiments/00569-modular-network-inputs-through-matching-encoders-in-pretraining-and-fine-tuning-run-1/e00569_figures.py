# ruff: noqa: ANN201, D103, E501, RUF028
"""Figure 1 of experiment 569: B3a school term, real minus placebo per Primary, beside the seed noise of plain M0."""

from pathlib import Path
from typing import Final

import altair as alt
import polars as pl
from plotting import ocf_theme as theme
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final = "00569-modular-network-inputs-through-matching-encoders-in-pretraining-and-fine-tuning-run-1"
DOCS: Final[Path] = Path(__file__).resolve().parents[2] / "docs/ml-experiments" / NAME


def main():
    t = pl.read_csv(ML_EXPERIMENTS_DATA_DIR / NAME / "school_per_primary.csv").sort("real_minus_placebo")
    order = t["label"].to_list()
    d = t.to_pandas()
    base = alt.Chart(d).encode(y=alt.Y("label:N", sort=order, title=None))
    x = alt.X("real_minus_placebo_low:Q", title="CRPS difference (points of capacity; negative favours the school-term columns)", scale=alt.Scale(domain=[-1.0, 1.0]))  # fmt: skip
    bar = base.mark_rule(color=theme.DATA_BLUE, strokeWidth=3).encode(x=x, x2="real_minus_placebo_high:Q")  # ty: ignore[unresolved-attribute]
    dot = base.mark_point(color=theme.BRAND_ORANGE, filled=True, size=70, aria=False).encode(x="real_minus_placebo:Q")  # ty: ignore[unresolved-attribute]
    noise = base.mark_point(color="grey", shape="cross", size=70, aria=False).encode(x="seed_noise:Q")  # ty: ignore[unresolved-attribute]
    zero = alt.Chart(pl.DataFrame({"x": [0.0]}).to_pandas()).mark_rule(color="black").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    chart = alt.layer(zero, bar, dot, noise).properties(
        width=470, height=380,
        title=alt.TitleParams(
            [
                "The spread of school-term effects across Primaries is smaller than the seed noise.",
                "Orange dot: school-term minus shuffled-day placebo (1 seed each). Bar: 95% 5-month interval.",
                "Grey cross: plain M0 seed 0 minus seed 1 (pure noise). All rows. Interim.",
            ],
            anchor="start",
        ),
    )  # fmt: skip
    chart.save(DOCS / "figure_1_school_term_per_primary.svg")


main()
