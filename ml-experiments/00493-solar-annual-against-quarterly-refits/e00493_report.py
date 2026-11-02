# ruff: noqa: E501
"""Figures of experiment 493 from the contrast table `scores/contrasts.csv`, plus example forecasts.

    PYTHONPATH=ml-experiments/00310-metered-farm-long-lead-scoring:ml-experiments/00430-unified-forecaster-solar-and-pretraining-ablations \
        uv run --no-sync python ml-experiments/00493-*/e00493_report.py

Farms appear only as Solar A to F, and the example forecasts are drawn against hours since the run.
"""

from pathlib import Path
from typing import Final

import altair as alt
import e00310_common as c310  # ty: ignore[unresolved-import]
import e00493_common as common
import e00493_score as sc
import plotting.ocf_theme as ocf
import polars as pl

DOCS: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00493-solar-annual-against-quarterly-refits"
)
WIDTH: Final[int] = 440
LABELS: Final[dict[str, str]] = {
    "C1": "One pass: annual vs quarterly",
    "C2": "Encoder: annual vs quarterly",
    "C3": "One pass annual vs encoder quarterly",
}


def rule_layer() -> alt.Chart:
    """Return dashed rules at 0 and at the plus and minus 1% limit."""
    return (
        alt.Chart(alt.Data(values=[{"x": 0}, {"x": 1}, {"x": -1}]))  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[4, 3], color="#999999")
        .encode(x="x:Q")
    )


def interval_chart(
    *, data: pl.DataFrame, row: str, row_title: str | None, height: int
) -> alt.Chart:
    """Return points with intervals, one row per `row` value."""
    base = alt.Chart(alt.Data(values=data.to_dicts())).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y(f"{row}:N", title=row_title, sort=None)
    )
    bars = base.mark_rule(strokeWidth=3, color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("lo:Q", title="Change in quantile score (%, negative is better)"), x2="hi:Q"
    )
    points = base.mark_point(filled=True, size=70, color=ocf.BRAND_ORANGE).encode(x="pct:Q")  # ty: ignore[unresolved-attribute]
    return (rule_layer() + bars + points).properties(width=WIDTH, height=height)


def figures(*, table: pl.DataFrame) -> None:
    """Draw the headline, per-farm, and per-quarter figures."""
    labelled = table.with_columns(contrast=pl.col("id").replace(LABELS))
    head = labelled.filter(pl.col("cut").is_in(["bands_all", "bands_d1"])).with_columns(
        row=pl.col("contrast")
        + pl.when(pl.col("cut") == "bands_all")
        .then(pl.lit(", all bands"))
        .otherwise(pl.lit(", day 1"))
    )
    interval_chart(data=head, row="row", row_title=None, height=200).properties(
        title="Annual against quarterly refits, 98.33% month-block intervals; dashed lines mark 0 and the 1% limit"
    ).configure_title(fontSize=12, anchor="start").save(str(DOCS / "headline.svg"))

    farms = labelled.filter(pl.col("cut").str.starts_with("farm_")).with_columns(
        farm=pl.lit("Solar ") + pl.col("cut").str.slice(5)
    )
    charts = [
        interval_chart(
            data=farms.filter(pl.col("id") == cid), row="farm", row_title=None, height=120
        ).properties(title=LABELS[cid])
        for cid in LABELS
    ]
    alt.vconcat(*charts).configure_title(fontSize=12, anchor="start").save(
        str(DOCS / "per_farm.svg")
    )  # ty: ignore[unresolved-attribute]

    quarters = labelled.filter(pl.col("cut").str.starts_with("quarter_")).with_columns(
        quarter=pl.lit("Months ")
        + pl.col("cut")
        .str.slice(8)
        .cast(pl.Int64)
        .map_elements(lambda k: f"{3 * (k - 1)} to {3 * k} since the refit", return_dtype=pl.String)
    )
    charts = [
        interval_chart(
            data=quarters.filter(pl.col("id") == cid), row="quarter", row_title=None, height=100
        ).properties(title=LABELS[cid])
        for cid in LABELS
    ]
    alt.vconcat(*charts).configure_title(fontSize=12, anchor="start").save(
        str(DOCS / "per_quarter.svg")
    )  # ty: ignore[unresolved-attribute]


def example_figures() -> None:
    """Draw a day-ahead and a 10-day forecast of Solar A, annual and quarterly one-pass networks."""
    frames = {
        n: sc.load(name=n).filter(pl.col("time_series_id") == "A")
        for n in ("annual_g0_onepass", "q_g0_onepass")
    }
    pieces = []
    for season, init in (("summer", "2026-06-15"), ("winter", "2026-01-15")):
        for name, frame in frames.items():
            rows = frame.filter(pl.col("init_time").dt.strftime("%Y-%m-%d") == init)
            pieces.append(
                rows.with_columns(
                    season=pl.lit(season), hours=pl.col("lead_hours"), schedule=pl.lit(name)
                )
            )
    data = pl.concat(pieces)
    for name, bands, title in (
        ("example_day_ahead.svg", ["d1"], "Day ahead (12 to 36 hours), Solar A"),
        ("example_10_day.svg", list(c310.BAND_ORDER), "Days 1 to 10, Solar A"),
    ):
        d = data.filter(pl.col("band").is_in(bands)).sort("hours")
        x = alt.X("hours:Q", title="Hours since the run")
        base = alt.Chart().mark_line().encode(x=x)  # ty: ignore[unresolved-attribute]
        is_annual = alt.datum.schedule == "annual_g0_onepass"
        band90 = (
            base.transform_filter(is_annual)  # ty: ignore[unresolved-attribute]
            .mark_area(opacity=0.25, color=ocf.DATA_BLUE)
            .encode(y=alt.Y("q05:Q", title="Output (share of capacity)"), y2="q95:Q")
        )
        median = base.transform_filter(is_annual).mark_line(color=ocf.DATA_BLUE).encode(y="q50:Q")  # ty: ignore[unresolved-attribute]
        quarterly_median = (
            base.transform_filter(~is_annual)  # ty: ignore[unresolved-attribute]
            .mark_line(color=ocf.DATA_PURPLE, strokeDash=[4, 3])
            .encode(y="q50:Q")
        )
        actual = (
            base.transform_filter(is_annual)  # ty: ignore[unresolved-attribute]
            .mark_point(size=12, color=ocf.BRAND_ORANGE, filled=True)
            .encode(y="actual:Q")
        )
        chart = (
            alt.layer(band90, median, quarterly_median, actual)
            .properties(width=WIDTH, height=150)
            .facet(
                row=alt.Row("season:N", title=None), data=alt.Data(values=d.to_dicts()), title=title
            )
        )
        chart.save(str(DOCS / name))  # ty: ignore[unresolved-attribute]


def main() -> None:
    """Write every figure."""
    DOCS.mkdir(parents=True, exist_ok=True)
    figures(table=pl.read_csv(common.SCORE_DIR / "contrasts.csv"))
    example_figures()


if __name__ == "__main__":
    main()
