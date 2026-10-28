# ruff: noqa: E501
"""Draw experiment 369's figures from the stored site table.

uv run --no-sync python ml-experiments/00369-*/e00369_charts.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "00369-d8-cross-site-step-rate-against-stack-gap"
DOCS_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / SLUG
WIDTH: Final[int] = 480
HIGHLIGHT: Final[tuple[str, ...]] = ("D8", "D12", "D22")


def _title(text: str, subtitle: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=textwrap.wrap(text, 85), subtitle=textwrap.wrap(subtitle, 110),
        anchor="start", fontSize=14, subtitleFontSize=11,
    )  # fmt: skip


def main() -> None:
    """Write the scatter and the placebo comparison."""
    table = pl.read_parquet(
        ML_EXPERIMENTS_DATA_DIR / SLUG / "work" / "site_table.parquet"
    ).with_columns(
        group=pl.when(pl.col("site").is_in(HIGHLIGHT))
        .then(pl.lit("D8, D12, D22"))
        .otherwise(pl.lit("Other Primaries"))
    )
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    points = alt.Chart(table).encode(
        x=alt.X("rate_per_month:Q", title="Detected steps per month in the scored year (1.5 MW, 6 hours)"),
        y=alt.Y("gap_points:Q", title="Stack minus BG2 (quantile-score points)"),
        color=alt.Color("group:N", scale=alt.Scale(domain=["D8, D12, D22", "Other Primaries"], range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]), legend=alt.Legend(title=None, orient="top")),
    )  # fmt: skip
    chart = (
        points.mark_point(filled=True, size=60)  # ty: ignore[unresolved-attribute]
        + points.mark_text(dx=8, dy=-6, align="left", fontSize=11).encode(text="site:N")  # ty: ignore[unresolved-attribute]
    ).properties(
        width=WIDTH, height=300,
        title=_title("Step rate does not line up with the stack's gap to BG2", "14 scored Primaries; Spearman rank correlation +0.01 (95% interval -0.68 to +0.59); only D8 has both a high rate and a large gap"),
    )  # fmt: skip
    chart.save(DOCS_DIR / "figure_1_rate_against_gap.svg")
    long = table.select(
        "site",
        real="history_rate_per_month",
        day_shuffle="placebo_rate_per_month",
        within_month_shuffle="placebo2_rate_per_month",
    ).unpivot(index="site", variable_name="series", value_name="rate")
    order = table.sort("history_rate_per_month", descending=True)["site"].to_list()
    bars = alt.Chart(long).mark_bar().encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("site:N", sort=order, title=None),
        x=alt.X("rate:Q", title="Detected steps per month over all history"),
        yOffset="series:N",
        color=alt.Color("series:N", sort=["real", "within_month_shuffle", "day_shuffle"], scale=alt.Scale(domain=["real", "within_month_shuffle", "day_shuffle"], range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_GREEN]), legend=alt.Legend(title=None, orient="top")),
    ).properties(width=WIDTH, height=420, title=_title("Shuffled copies of the readings show as many steps as the real readings", "Day-shuffle moves any day anywhere; within-month shuffle (post hoc) moves a day only among days of its month and day type; mean of 20 shuffles"))  # fmt: skip
    bars.save(DOCS_DIR / "figure_2_placebo.svg")
    print("wrote figures")


if __name__ == "__main__":
    main()
