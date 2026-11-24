# ruff: noqa: E501
"""Draw the figures for experiment 186 from the tables `e00186_analyse.py` wrote."""

import sys
from pathlib import Path
from typing import Final

import altair as alt
import e00186_common as common
import numpy as np
import polars as pl
from plotting import ocf_theme as ocf
from studies.charts import CONTENT_WIDTH_PX

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import engine_figure_guard as guard  # noqa: E402

DOCS_DIR: Final[Path] = Path(
    "docs/ml-experiments/00186-switching-d8-gas-peaker-spark-spread-and-triads"
)
WIDTH: Final[int] = CONTENT_WIDTH_PX - 80
LABELS: Final[dict[str, str]] = {
    "positive daily spread, efficiency 35%": "Daily spread above 0, 35% efficiency",
    "positive daily spread, efficiency 40%": "Daily spread above 0, 40% efficiency",
    "positive daily spread, efficiency 45%": "Daily spread above 0, 45% efficiency",
    "positive evening spread, efficiency 35%": "Evening (16-19) spread above 0, 35%",
    "positive evening spread, efficiency 40%": "Evening (16-19) spread above 0, 40%",
    "positive evening spread, efficiency 45%": "Evening (16-19) spread above 0, 45%",
    "positive daily spread with the gas price held at its mean, efficiency 40% (price alone)": "Price alone (gas held at mean), 40%",
}


def main() -> None:  # noqa: D103
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    rows = (
        pl.read_parquet(common.WORK_DIR / "step3_results.parquet")
        .filter(pl.col("name").is_in(list(LABELS)))
        .with_columns(label=pl.col("name").replace_strict(LABELS))
    )
    order = [LABELS[k] for k in LABELS]
    base = alt.Chart(rows).encode(
        y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=260))
    )
    fig1 = (
        base.mark_rule(color=ocf.DATA_BLUE, strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "lo:Q",
                title="Drop days minus matched ordinary days (percentage points; 0 = no difference)",
            ),
            x2="hi:Q",
        )
        + base.mark_point(color=ocf.DATA_BLUE, filled=True, size=90).encode(x="diff_points:Q")  # ty: ignore[unresolved-attribute]
        + alt.Chart(pl.DataFrame({"x": [0.0]}))
        .mark_rule(color=ocf.BRAND_ORANGE, strokeDash=[4, 3])
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    ).properties(
        width=WIDTH - 200,
        height=200,
        title=alt.TitleParams(
            "Drop days have a positive spark spread more often than matched ordinary days",
            subtitle=[
                "Dot: estimate. Line: 95% interval from resampling whole drops",
                "(28 drops, 126 drop days).",
            ],
            fontSize=12,
            subtitleFontSize=10,
            anchor="start",
            offset=14,
        ),
    )
    guard.save_chart(chart=fig1, path=DOCS_DIR / "figure_1_spread_share.svg")

    flat = pl.read_parquet(common.WORK_DIR / "flatness.parquet").with_columns(
        kind=pl.col("kind").replace({"control": "Injected 2.8 MW blocks", "drop": "D8's 28 drops"})
    )
    fig2 = (
        alt.Chart(flat)  # ty: ignore[unresolved-attribute]
        .mark_bar(opacity=0.75)
        .encode(
            x=alt.X(
                "rel_iqr:Q",
                bin=alt.Bin(step=0.1, extent=[0, 2.5]),
                title="Relative range of the 3-hour median depth (smaller = flatter block)",
            ),
            y=alt.Y("count():Q", stack=None, title="Share of windows", axis=None),
            color=alt.Color(
                "kind:N", title=None, scale=alt.Scale(range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE])
            ),
        )
        .transform_joinaggregate(total="count()", groupby=["kind"])
        .transform_calculate(share="1 / datum.total")
        .encode(y=alt.Y("sum(share):Q", stack=None, title="Share of windows"))
        .properties(
            width=WIDTH,
            height=150,
            title=alt.TitleParams(
                "The drops are less flat than injected constant blocks",
                subtitle=[
                    "Pre-registered flat-block cutoff: 0.25. The control passes it in 9% of windows."
                ],
                fontSize=12,
                subtitleFontSize=10,
                anchor="start",
                offset=14,
            ),
        )
    )
    guard.save_chart(chart=fig2, path=DOCS_DIR / "figure_2_flatness.svg")

    days = (
        pl.read_parquet(common.WORK_DIR / "d8_day_table.parquet")
        .sort("date")
        .with_columns(
            spread=pl.col("spread_40").rolling_mean(7, center=True, min_samples=4),
            drop=pl.col("drop_share") >= 0.75,
        )
    )
    # Pooled by fifth of the 7-day mean spread: no dates, no engine-state series.
    valid = days.filter(pl.col("spread").is_not_null())
    edges = np.quantile(valid["spread"].to_numpy(), [0.2, 0.4, 0.6, 0.8])
    fifths = (
        valid.with_columns(
            fifth=pl.Series(np.searchsorted(edges, valid["spread"].to_numpy(), side="right") + 1)
        )
        .group_by("fifth")
        .agg(share=pl.col("drop").mean())
        .sort("fifth")
    )
    bars = (
        alt.Chart(fifths)
        .mark_bar(color=ocf.DATA_BLUE, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "fifth:O",
                title="Fifth of the 7-day mean spark spread (1 = lowest)",
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y("share:Q", title="Share of days that are drop days"),
        )
    )
    fig3 = bars.properties(
        width=WIDTH // 2,
        height=170,
        title=alt.TitleParams(
            "Share of days that are drop days, by fifth of the spark spread",
            subtitle=[
                "Spark spread: system price minus gas fuel cost and operating cost,",
                "40% efficiency, 5 pounds/MWh. A higher fifth is a better margin for a gas plant.",
            ],
            fontSize=12,
            subtitleFontSize=10,
            anchor="start",
            offset=14,
        ),
    )
    guard.save_chart(chart=fig3, path=DOCS_DIR / "figure_3_spread_and_drops.svg")

    prof = pl.read_parquet(common.WORK_DIR / "event_profile.parquet").with_columns(
        anchor=pl.col("anchor").replace(
            {"start": "Around the start of a drop", "end": "Around the end of a drop"}
        )
    )
    fig4 = (
        alt.Chart(prof)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "offset:Q",
                title="Days from the drop's first (or last) day",
                axis=alt.Axis(values=[-10, -5, 0, 5, 10]),
            ),
            y=alt.Y("mean_spread:Q", title="Mean spark spread (pounds/MWh)"),
            color=alt.Color(
                "anchor:N", title=None, scale=alt.Scale(range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE])
            ),
        )
        .properties(
            width=WIDTH,
            height=150,
            title=alt.TitleParams(
                "Spark spread before and after the drops' starts and ends",
                subtitle=[
                    "40% efficiency, mean of 28 drops. A gas plant that follows the spread would",
                    "show a rise at the start and a fall at the end.",
                ],
                fontSize=12,
                subtitleFontSize=10,
                anchor="start",
                offset=14,
            ),
        )
    )
    guard.save_chart(chart=fig4, path=DOCS_DIR / "figure_4_event_profile.svg")


if __name__ == "__main__":
    main()
