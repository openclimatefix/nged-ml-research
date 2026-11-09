# ruff: noqa: E501, D103
"""Experiment 667 figures (CPU): read the score tables and the file; write SVGs for the page.

Usage: python e00667_figures.py OUT_DIR
"""

import sys
from datetime import datetime
from pathlib import Path

import altair as alt
import e00667_export as exp
import polars as pl
from plotting import ocf_theme as theme

COLOUR = alt.Scale(domain=["NRA", "prevailing"], range=[theme.DATA_BLUE, theme.BRAND_ORANGE])
LEGEND = alt.Legend(title=None, orient="top")
W = 640
SHARED = exp.ROOT.parent / "backtest_year_2025-11_to_2026-10"
CAPACITY = pl.read_parquet(SHARED / "series_capacity.parquet")


def scores() -> pl.DataFrame:
    parts = [
        pl.read_csv(exp.ROOT / "score" / f"{n}_by_band_month.csv").with_columns(arm=pl.lit(n))
        for n in ("NRA", "prevailing")
    ]
    return pl.concat(parts)


def weighted(d: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    return (
        d.group_by(by)
        .agg(
            (pl.col("crps_pts") * pl.col("rows")).sum() / pl.col("rows").sum(),
            (pl.col("nmae_median_pts") * pl.col("rows")).sum() / pl.col("rows").sum(),
            (pl.col("coverage_q05_q95") * pl.col("rows")).sum() / pl.col("rows").sum(),
            (pl.col("coverage_q25_q75") * pl.col("rows")).sum() / pl.col("rows").sum(),
        )
        .sort(by)
    )


def crps_by_band(d: pl.DataFrame) -> alt.Chart:
    b = weighted(d, ["arm", "band"])
    return (
        alt.Chart(  # ty: ignore[unresolved-attribute]
            b,
            width=W,
            height=220,
            title=alt.TitleParams(
                [
                    "CRPS by lead band, all 16 Primaries, 365 origins",
                    "Points of capacity; smaller is better",
                ],
                anchor="start",
                fontSize=12,
                subtitleFontSize=11,
            ),
        )
        .mark_bar()
        .encode(
            x=alt.X(
                "band:N",
                title="Lead band (days from the 12:00 UTC origin)",
                axis=alt.Axis(labelAngle=0),
            ),
            xOffset="arm:N",
            y=alt.Y("crps_pts:Q", title="CRPS (points of capacity)"),
            color=alt.Color("arm:N", scale=COLOUR, legend=LEGEND),
        )
    )


def crps_by_month(d: pl.DataFrame) -> alt.Chart:
    m = weighted(d, ["arm", "origin_month"])
    return (
        alt.Chart(  # ty: ignore[unresolved-attribute]
            m,
            width=W,
            height=240,
            title=alt.TitleParams(
                [
                    "CRPS by month of the origin, all lead bands",
                    "Points of capacity; smaller is better. Prevailing is lower in 9 of 12 months. 2025-11 and 2025-12 prevailing rows use the fallback survival curve.",
                ],
                anchor="start",
                fontSize=12,
                subtitleFontSize=11,
            ),
        )
        .mark_line(point=True)
        .encode(
            x=alt.X("origin_month:N", title="Month of the origin", axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("crps_pts:Q", title="CRPS (points of capacity)", scale=alt.Scale(zero=True)),
            color=alt.Color("arm:N", scale=COLOUR, legend=LEGEND),
        )
    )


def coverage_by_month(d: pl.DataFrame) -> alt.Chart:
    m = weighted(d, ["arm", "origin_month"])
    line = (
        alt.Chart(  # ty: ignore[unresolved-attribute]
            m,
            width=W,
            height=240,
            title=alt.TitleParams(
                [
                    "Share of observed power inside the q05 to q95 band, by month of the origin",
                    "Dashed rule: 0.90 nominal. Dotted rules: 0.85 and 0.95, the range that prediction P3 required.",
                    "NRA is scored against metered power, so event windows lower its share.",
                ],
                anchor="start",
                fontSize=12,
                subtitleFontSize=11,
            ),
        )
        .mark_line(point=True)
        .encode(
            x=alt.X("origin_month:N", title="Month of the origin", axis=alt.Axis(labelAngle=-45)),
            y=alt.Y(
                "coverage_q05_q95:Q",
                title="Share inside the band",
                scale=alt.Scale(domain=[0.6, 1.0]),
            ),
            color=alt.Color("arm:N", scale=COLOUR, legend=LEGEND),
        )
    )
    rule = (
        alt.Chart(pl.DataFrame({"y": [0.9]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[4, 4], color="#555")
        .encode(y="y:Q")
    )
    tol = (
        alt.Chart(pl.DataFrame({"y": [0.85, 0.95]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[1, 3], color="#888")
        .encode(y="y:Q")
    )
    return line + tol + rule


def example(*, series: int, origin: str, name: str) -> alt.FacetChart:
    f = pl.read_parquet(SHARED / "primaries.parquet").filter(
        (pl.col("time_series_id") == series)
        & (
            pl.col("power_fcst_init_time")
            == pl.lit(datetime.fromisoformat(origin + "T12:00:00+00:00"))
        )
    )
    t = (
        pl.read_parquet(SHARED / "ground_truth_primaries.parquet")
        .filter(pl.col("time_series_id") == series)
        .select("valid_time", "power")
    )
    f = f.join(t, on="valid_time", how="left").rename({"NRA_or_prevailing": "arm"})
    unit = CAPACITY.filter(pl.col("time_series_id") == series)["power_unit"][0]
    base = alt.Chart(f, width=W - 80, height=280).encode(
        x=alt.X(
            "valid_time:T", title="Valid time (UTC)", axis=alt.Axis(format="%a %d", tickCount=14)
        )
    )
    band = base.mark_area(opacity=0.25).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("q05:Q", title=f"Power ({unit})"),
        y2="q95:Q",
        color=alt.Color("arm:N", scale=COLOUR, legend=LEGEND),
    )
    med = base.mark_line().encode(y="q50:Q", color=alt.Color("arm:N", scale=COLOUR, legend=None))  # ty: ignore[unresolved-attribute]
    obs = base.mark_line(color="#222", strokeWidth=1).encode(y="power:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(band, med, obs).facet(
        row=alt.Row("arm:N", title=None),
        title=alt.TitleParams(
            [
                name,
                "Series 26 (MVA), origin 2026-02-01 12:00 UTC. Coloured line: median. Shade: q05 to q95.",
                "Black line: observed power. The prevailing band falls below zero.",
            ],
            anchor="start",
            fontSize=12,
            subtitleFontSize=11,
        ),
    )


if __name__ == "__main__":
    out = Path(sys.argv[1])
    d = scores()
    crps_by_band(d).save(str(out / "crps_by_band.svg"))
    crps_by_month(d).save(str(out / "crps_by_month.svg"))
    coverage_by_month(d).save(str(out / "coverage_by_month.svg"))
    if len(sys.argv) > 3:
        example(
            series=int(sys.argv[2]),
            origin=sys.argv[3],
            name="Example 14-day forecast at one Primary",
        ).save(str(out / "example_forecast.svg"))
