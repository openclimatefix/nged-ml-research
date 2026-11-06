# ruff: noqa: E501, D103
"""Experiment 593: the 14-day example plot for the export README (D5, December 2025, day-ahead forecasts).

Day-ahead means the forecast issued at 09:00 UTC the day before the valid day. Substations may be named.
"""

import altair as alt
import polars as pl
from plotting.ocf_theme import BRAND_ORANGE, DATA_BLUE, GREY_3
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT = ML_EXPERIMENTS_DATA_DIR / "backtest_export_2026-11-06"
START, DAYS = pl.datetime(2025, 12, 19, time_zone="UTC"), 14


def day_ahead(arm: str) -> pl.DataFrame:
    return (
        pl.read_parquet(OUT / f"{arm}_2025-12.parquet")
        .join(
            pl.read_csv(OUT / "substations.csv").select("time_series_id", "primary_name"),
            on="time_series_id",
        )
        .filter(
            (pl.col("primary_name") == "D5")
            & (pl.col("valid_time") >= START)
            & (pl.col("valid_time") < START + pl.duration(days=DAYS))
            & (pl.col("power_fcst_init_time").dt.hour() == 9)
            & (
                pl.col("power_fcst_init_time").dt.date()
                == (pl.col("valid_time").dt.date() - pl.duration(days=1))
            )
        )
        .with_columns(arm=pl.lit(arm))
    )


def main() -> None:
    df = pl.concat([day_ahead("C0"), day_ahead("C2")]).with_columns(
        day=(pl.col("valid_time") - START).dt.total_minutes() / 1440.0,
        day2=(pl.col("valid_time") - START).dt.total_minutes() / 1440.0 + 1 / 48,
    )
    names = {"C0": "C0 (earlier forecast)", "C2": "C2 (candidate)"}
    df = df.with_columns(arm=pl.col("arm").replace(names))
    scale = alt.Scale(domain=list(names.values()), range=[DATA_BLUE, BRAND_ORANGE])
    base = alt.Chart(df).encode(
        x=alt.X(
            "day:Q",
            title="Days since 19 December 2025, 00:00 UTC",
            axis=alt.Axis(values=list(range(0, 15, 2))),
            scale=alt.Scale(domain=[0, 14]),
        )
    )
    band = base.mark_area(opacity=0.25).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("q05:Q", title="Power (MW; D5 is a signed-MW Primary)"),
        y2="q95:Q",
        color=alt.Color("arm:N", scale=scale, title=None),
    )
    med = base.mark_line(strokeWidth=1.5).encode(  # ty: ignore[unresolved-attribute]
        y="q50:Q", color=alt.Color("arm:N", scale=scale, title=None)
    )
    obs = (
        alt.Chart(df.filter(pl.col("arm") == names["C2"]))  # ty: ignore[unresolved-attribute]
        .mark_line(color="#222222", strokeWidth=1.3)
        .encode(x="day:Q", y="observed_power:Q")
    )
    shade = (
        alt.Chart(df.filter((pl.col("arm") == names["C2"]) & pl.col("in_verified_event")))  # ty: ignore[unresolved-attribute]
        .mark_rect(opacity=0.12, color=GREY_3)
        .encode(x="day:Q", x2="day2:Q")
    )
    chart = (
        alt.layer(shade, band, med, obs)
        .properties(
            width=620,
            height=260,
            title=alt.Title(
                "D5 day-ahead forecasts, December 2025",
                subtitle=[
                    "Black line: observed power. Grey shade: verified load-shift. Coloured line: median forecast. Shaded band: 90% band (q05 to q95).",
                    "Forecast issued 09:00 UTC the day before. The verified load-shift at D5 starts on day 10.",
                ],
            ),
        )
        .configure_view(stroke=None)
        .configure_legend(orient="top")
    )
    chart.save(OUT / "example_forecast_D5_2025-12.svg")
    chart.save(OUT / "example_forecast_D5_2025-12.png", scale_factor=2)


if __name__ == "__main__":
    main()
