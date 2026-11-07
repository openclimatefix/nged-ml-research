# ruff: noqa: D103, B007
"""Experiment 641: figures for the write-up (SVG, 680 px wide, one or two panels each)."""

import os
from pathlib import Path
from typing import Final

import altair as alt
import polars as pl
from plotting import ocf_theme as theme

DATA: Final = Path(os.environ["DATA_PATH_INTERNAL"]) / "ml-experiments"
TABLES: Final = DATA / "00641-d15-d12-error-diagnosis-and-cheapest-fixes/tables"
EXPORT: Final = DATA / "backtest_export_roster_2026-11-07"
DOCS: Final = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00641-d15-d12-error-diagnosis-and-cheapest-fixes"
)
W: Final = 600


def save(chart: alt.TopLevelMixin, name: str) -> None:
    chart.configure_view(strokeWidth=0).save(DOCS / name)


def fig_hour() -> None:
    d = pl.concat(
        [
            pl.read_csv(TABLES / f"{p}_by_hour.csv").with_columns(primary=pl.lit(p))
            for p in ("D15", "D12")
        ]
    )
    save(
        alt.Chart(d.to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "hour:Q", title="Hour of the day (local time)", scale=alt.Scale(domain=[0, 23])
            ),
            y=alt.Y(
                "mae:Q",
                title="Mean absolute error of the median (% of capacity)",
                scale=alt.Scale(domain=[0, 30]),
            ),
            color=alt.Color(
                "primary:N",
                title=None,
                scale=alt.Scale(domain=["D15", "D12"], range=[theme.BRAND_ORANGE, theme.DATA_BLUE]),
                legend=alt.Legend(orient="top"),
            ),
        )
        .properties(width=W, height=240),
        "fig_error_by_hour.svg",
    )


def fig_d15_days() -> None:
    d = pl.read_csv(TABLES / "D15_daily.csv", try_parse_dates=True).select(
        "day", observed=pl.col("obs_med"), forecast=pl.col("fc_med")
    )
    long = d.unpivot(index="day", variable_name="series", value_name="mw")
    save(
        alt.Chart(long.to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_point(size=14, filled=True)
        .encode(
            x=alt.X("day:T", title="Day (five test months)"),
            y=alt.Y("mw:Q", title="Daily median power (MW)"),
            color=alt.Color(
                "series:N",
                title=None,
                scale=alt.Scale(
                    domain=["observed", "forecast"], range=[theme.DATA_BLUE, theme.BRAND_ORANGE]
                ),
                legend=alt.Legend(orient="top"),
            ),
        )
        .properties(width=W, height=240),
        "fig_d15_daily_level.svg",
    )


def fig_fixes() -> None:
    t = pl.read_csv(TABLES / "fix_by_primary.csv")
    long = pl.concat(
        [
            t.select(
                "primary",
                "k1",
                pl.col("f1c_pct").alias("pct"),
                pl.col("f1c_lo").alias("lo"),
                pl.col("f1c_hi").alias("hi"),
                pl.lit("14-day level term").alias("fix"),
            ),
            t.select(
                "primary",
                "k1",
                pl.col("oracle_pct").alias("pct"),
                pl.lit(None, dtype=pl.Float64).alias("lo"),
                pl.lit(None, dtype=pl.Float64).alias("hi"),
                pl.lit("Oracle: daily bias removed").alias("fix"),
            ),
        ]
    ).to_pandas()
    order = t.sort("k1", descending=True)["primary"].to_list()
    base = alt.Chart(long).encode(
        y=alt.Y("primary:N", sort=order, title=None),
        color=alt.Color(
            "fix:N",
            title=None,
            scale=alt.Scale(range=[theme.BRAND_ORANGE, theme.DATA_BLUE]),
            legend=alt.Legend(orient="top"),
        ),
    )
    pts = base.mark_point(filled=True, size=50).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("pct:Q", title="Change in CRPS against K1 (%, negative is better)")
    )
    bars = base.mark_rule().encode(x="lo:Q", x2="hi:Q")  # ty: ignore[unresolved-attribute]
    zero = (
        alt.Chart(pl.DataFrame({"z": [0.0]}).to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[4, 4], color="grey")
        .encode(x="z:Q")
    )
    save((zero + bars + pts).properties(width=W, height=380), "fig_fix_by_primary.svg")


def fig_examples() -> None:
    parts = []
    for name, tsid, month, init_hour_filter, title in (
        ("D15", 12, "2026-03", None, "D15, 10-day forecast issued 2026-03-10"),
        ("D12", 14, "2026-06", None, "D12, 10-day forecast issued 2026-06-10"),
    ):
        df = pl.read_parquet(EXPORT / f"K1_{month}.parquet").filter(
            pl.col("time_series_id") == tsid
        )
        init = df.filter(
            pl.col("power_fcst_init_time").dt.date()
            == pl.datetime(int(month[:4]), int(month[5:]), 10).dt.date()
        )["power_fcst_init_time"].min()
        d = (
            df.filter(pl.col("power_fcst_init_time") == init)
            .select("valid_time", "q05", "q25", "q50", "q75", "q95", "observed_power")
            .to_pandas()
        )
        band = (
            alt.Chart(d)  # ty: ignore[unresolved-attribute]
            .mark_area(opacity=0.2, color=theme.DATA_BLUE)
            .encode(
                x=alt.X("valid_time:T", title=None),
                y=alt.Y("q05:Q", title="Power (MW)"),
                y2="q95:Q",
            )
        )
        band50 = (
            alt.Chart(d)  # ty: ignore[unresolved-attribute]
            .mark_area(opacity=0.3, color=theme.DATA_BLUE)
            .encode(x="valid_time:T", y="q25:Q", y2="q75:Q")
        )
        med = alt.Chart(d).mark_line(color=theme.DATA_BLUE).encode(x="valid_time:T", y="q50:Q")  # ty: ignore[unresolved-attribute]
        obs = alt.Chart(d).mark_line(color="black").encode(x="valid_time:T", y="observed_power:Q")  # ty: ignore[unresolved-attribute]
        parts.append((band + band50 + med + obs).properties(width=W, height=180, title=title))
    save(alt.vconcat(*parts), "fig_examples.svg")


if __name__ == "__main__":
    fig_hour()
    fig_d15_days()
    fig_fixes()
    fig_examples()
