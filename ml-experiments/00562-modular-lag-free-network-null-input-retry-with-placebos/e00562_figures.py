# ruff: noqa: ANN201, D103, E501, RUF028
"""Figure 1 of experiment 562: real minus placebo CRPS by arm, with Bonferroni intervals."""

from pathlib import Path
from typing import Final

import altair as alt
import polars as pl
from plotting import ocf_theme as theme
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final = "00562-modular-lag-free-network-null-input-retry-with-placebos"
D: Final[Path] = ML_EXPERIMENTS_DATA_DIR / NAME
DOCS: Final[Path] = Path(__file__).resolve().parents[2] / "docs/ml-experiments" / NAME
LABEL: Final = {
    "B1": "B1 thermal lag, solar gain", "B2": "B2 look-ahead windows", "B3a": "B3a school term",
    "B3b": "B3b national signals", "B4": "B4 price shape",
}  # fmt: skip
PANELS: Final = (
    ("eight", "all", "Eight MVA-only Primaries, all rows"),
    ("eight", "cold_night", "Eight MVA-only Primaries, cold nights (3 months)"),
    ("six", "all", "Six signed-MW Primaries, all rows"),
    ("six", "cold_night", "Six signed-MW Primaries, cold nights (3 months)"),
)


def main():
    c = pl.read_csv(D / "scores_contrasts.csv").filter(pl.col("primary"))
    c = c.with_columns(arm=pl.col("contrast").str.split(" - ").list.get(0).replace(LABEL))
    panels = []
    for group, slice_name, label in PANELS:
        part = c.filter((pl.col("set") == group) & (pl.col("slice") == slice_name)).to_pandas()
        base = alt.Chart(part).encode(y=alt.Y("arm:N", sort=list(LABEL.values()), title=None))
        bar = base.mark_rule(color=theme.DATA_BLUE, strokeWidth=3).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "low_bonferroni_month:Q",
                title="Real minus placebo (CRPS points of capacity; more negative is better)",
                scale=alt.Scale(domain=[-0.4, 0.4]),
            ),  # fmt: skip
            x2="high_bonferroni_month:Q",
        )
        dot = base.mark_point(color=theme.BRAND_ORANGE, filled=True, size=60, aria=False).encode(x="difference:Q")  # ty: ignore[unresolved-attribute]  # fmt: skip
        zero = (
            alt.Chart(pl.DataFrame({"x": [0.0]}).to_pandas())
            .mark_rule(color="black")
            .encode(x="x:Q")
        )  # ty: ignore[unresolved-attribute]
        panels.append(alt.layer(zero, bar, dot).properties(width=470, height=100, title=alt.TitleParams(label, anchor="start")))  # fmt: skip
    chart = alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "Only B1 on cold nights at the six clearly beats its placebo. Dot: difference over",
                "2 seeds (B3a, B4: 1 seed). Bar: Bonferroni (20 tests) month-block interval, near",
                "min-to-max with 5 months (3 for cold nights). Line at 0: same as placebo. Interim.",
            ],
            anchor="start",
        )
    )
    chart.save(DOCS / "figure_1_real_minus_placebo.svg")


main()
