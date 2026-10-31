"""Draw the anonymised figures for experiment 433 from the tables `e00433_look.py` wrote."""

from pathlib import Path
from typing import Final

import altair as alt
import polars as pl
from plotting import ocf_theme as ocf
from studies.charts import CONTENT_WIDTH_PX

SLUG: Final[str] = "00433-solar-idle-noise-data-look"
DOCS_DIR: Final[Path] = Path("docs/ml-experiments") / SLUG
HEIGHT: Final[int] = 150
FARMS: Final[list[str]] = [f"Solar {c}" for c in "ABCDEF"]
FARM_COLOURS: Final[list[str]] = [
    ocf.BRAND_ORANGE,
    ocf.DATA_BLUE,
    ocf.DATA_SKY,
    ocf.DATA_PURPLE,
    ocf.DATA_GREEN,
    ocf.DATA_BLUE_LIGHT,
]


def _save(chart: alt.Chart | alt.VConcatChart, name: str) -> None:
    chart.properties(width=CONTENT_WIDTH_PX - 80).save(DOCS_DIR / name)  # ty: ignore[unresolved-attribute]


def main() -> None:  # noqa: D103
    summary = pl.read_csv(DOCS_DIR / "summary.csv")
    comp = (
        summary.select(
            "site",
            exact_zero=pl.col("zero_share"),
            negative=pl.col("negative_share"),
            positive=1 - pl.col("zero_share") - pl.col("negative_share"),
        )
        .unpivot(index="site", variable_name="reading", value_name="share")
        .with_columns(share=pl.col("share").clip(lower_bound=0) * 100)
    )
    fig1 = (
        alt.Chart(comp)  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            y=alt.Y("site:N", title=None, sort=FARMS),
            x=alt.X(
                "share:Q", title="Share of night half-hours (%)", scale=alt.Scale(domain=[0, 100])
            ),
            color=alt.Color(
                "reading:N",
                title=None,
                scale=alt.Scale(
                    domain=["exact_zero", "negative", "positive"],
                    range=[ocf.DATA_BLUE_LIGHT, ocf.BRAND_ORANGE, ocf.DATA_PURPLE],
                ),
                legend=alt.Legend(orient="top"),
            ),
        )
        .properties(height=HEIGHT)
    )
    _save(fig1, "figure_1.svg")
    by_year = pl.read_csv(DOCS_DIR / "by_year.csv").with_columns(neg=pl.col("negative_share") * 100)
    fig2 = (
        alt.Chart(by_year)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True)
        .encode(
            x=alt.X("year:O", title="Calendar year"),
            y=alt.Y("neg:Q", title="Negative night readings (%)"),
            color=alt.Color(
                "site:N",
                title=None,
                scale=alt.Scale(domain=FARMS, range=FARM_COLOURS),
                legend=alt.Legend(orient="top"),
            ),
        )
        .properties(height=HEIGHT)
    )
    _save(fig2, "figure_2.svg")
    bins = pl.read_csv(DOCS_DIR / "daytime_run_bins.csv")
    order = ["1", "2-3", "4-7", "8-24", "over 24"]
    fig3 = (
        alt.Chart(bins)  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            x=alt.X("site:N", title=None, sort=FARMS),
            y=alt.Y("idle_half_hours:Q", title="Clear-sky daytime idle half-hours"),
            color=alt.Color(
                "bin:N",
                title="Run length (days)",
                sort=order,
                scale=alt.Scale(
                    domain=order,
                    range=[
                        ocf.DATA_BLUE_LIGHT,
                        ocf.DATA_SKY,
                        ocf.DATA_BLUE,
                        ocf.DATA_PURPLE,
                        ocf.BRAND_ORANGE,
                    ],
                ),
                legend=alt.Legend(orient="top"),
            ),
        )
        .properties(height=HEIGHT)
    )
    _save(fig3, "figure_3.svg")


if __name__ == "__main__":
    main()
