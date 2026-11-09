# ruff: noqa: D103, E501
"""Experiment 669: figures of the scores and example forecasts of the one-year backtest (CPU).

Usage: python e00669_figures.py SCORE_DIR OUT_DIR
"""

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import altair as alt
import e00087_common as common  # ty: ignore[unresolved-import]
import e00669_common as k
import e00669_score as sc
import numpy as np
import polars as pl

GROUP_COLOURS = {"BSP": "#306BFF", "GSP": "#FF4901", "All four series": "#17E58F"}
SITE_NAMES = {
    "D13": "D13 Skegness (BSP)",
    "D17": "D17 Boston (BSP)",
    "D1": "D1 Bicker Fen (GSP)",
    "D6": "D6 Walpole (GSP)",
}


def save(chart: alt.Chart, path: Path) -> None:
    chart.save(str(path))
    print("wrote", path, flush=True)


def lead_chart(*, score: Path) -> alt.Chart:
    g = pl.read_csv(score / "by_group_lead_day.csv").with_columns(series=pl.col("group"))
    a = pl.read_csv(score / "by_lead_day.csv").with_columns(series=pl.lit("All four series"))
    long = pl.concat(
        [
            g.select("series", "lead_day", "crps", "crps_lo", "crps_hi"),
            a.select("series", "lead_day", "crps", "crps_lo", "crps_hi"),
        ]
    ).with_columns(day=pl.col("lead_day") + 1)
    base = alt.Chart(long.to_pandas()).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "day:Q",
            title="Forecast day (days after the 12:00 UTC origin)",
            axis=alt.Axis(values=[1, 3, 5, 7, 9, 11, 13]),
        ),
        color=alt.Color(
            "series:N",
            title=None,
            scale=alt.Scale(domain=list(GROUP_COLOURS), range=list(GROUP_COLOURS.values())),
            legend=alt.Legend(orient="top", columns=3),
        ),
    )
    band = base.mark_area(opacity=0.2).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y(
            "crps_lo:Q",
            title="CRPS (points of capacity; smaller is better)",
            scale=alt.Scale(zero=True),
        ),
        y2="crps_hi:Q",
    )
    line = base.mark_line(point=True, strokeWidth=2).encode(y="crps:Q")  # ty: ignore[unresolved-attribute]
    return (band + line).properties(
        width=620,
        height=300,
        title=alt.TitleParams(
            [
                "CRPS by forecast day, 2025-11-01 to 2026-10-31 origins",
                "Line: mean; band: 95% day-block bootstrap interval",
            ],
            anchor="start",
            fontSize=13,
        ),
    )


def month_chart(*, scored: pl.DataFrame) -> alt.Chart:
    m = scored.group_by("group", "month").agg(crps=pl.col("crps").mean()).to_pandas()  # ty: ignore[unresolved-attribute]
    return (
        alt.Chart(m)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X("month:N", title="Month of the forecast origin", axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("crps:Q", title="CRPS (points of capacity)", scale=alt.Scale(zero=True)),
            color=alt.Color(
                "group:N",
                title=None,
                scale=alt.Scale(
                    domain=["BSP", "GSP"], range=[GROUP_COLOURS["BSP"], GROUP_COLOURS["GSP"]]
                ),
                legend=alt.Legend(orient="top"),
            ),
        )
        .properties(
            width=620,
            height=260,
            title=alt.TitleParams(
                "CRPS by month of the forecast origin, all lead times", anchor="start", fontSize=13
            ),
        )
    )


def forecast_panel(
    *, f: pl.DataFrame, y: pl.DataFrame, title: str, height: int = 200, hours: int = 24
) -> alt.Chart:
    pdf = f.to_pandas()  # ty: ignore[unresolved-attribute]
    x = alt.X("valid_time:T", title=None)
    b90 = (
        alt.Chart(pdf)  # ty: ignore[unresolved-attribute]
        .mark_area(opacity=0.2, color="#306BFF")
        .encode(x=x, y=alt.Y("q05:Q", title="MW"), y2="q95:Q")
    )
    b50 = alt.Chart(pdf).mark_area(opacity=0.35, color="#306BFF").encode(x=x, y="q25:Q", y2="q75:Q")  # ty: ignore[unresolved-attribute]
    med = alt.Chart(pdf).mark_line(color="#306BFF", strokeWidth=1.5).encode(x=x, y="q50:Q")  # ty: ignore[unresolved-attribute]
    act = (
        alt.Chart(y.to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_line(color="#FF4901", strokeWidth=1.5)
        .encode(x=x, y="power:Q")
    )  # ty: ignore[unresolved-attribute]
    _ = hours
    return (b90 + b50 + med + act).properties(
        width=620, height=height, title=alt.TitleParams(title, anchor="start", fontSize=12)
    )


def examples(*, out: Path) -> None:
    gt = {g: pl.read_parquet(k.SHARED / f"ground_truth_{g}.parquet") for g in ("bsps", "gsps")}
    files = {"D13": ("bsps", 2), "D17": ("bsps", 1), "D1": ("gsps", 28), "D6": ("gsps", 27)}
    forecasts = {g: pl.read_parquet(k.SHARED / f"{g}.parquet") for g in ("bsps", "gsps")}

    def actual(site: str) -> pl.DataFrame:
        g, tid = files[site]
        return gt[g].filter(pl.col("time_series_id") == tid).select("valid_time", "power")

    def fc(site: str, origin: datetime) -> pl.DataFrame:
        g, tid = files[site]
        return forecasts[g].filter(
            (pl.col("time_series_id") == tid) & (pl.col("power_fcst_init_time") == origin)
        )

    def window(site: str, origin: datetime, hours: int, title: str, height: int = 200) -> alt.Chart:
        f = fc(site, origin).filter(pl.col("valid_time") < origin + timedelta(hours=hours))
        y = actual(site).filter(
            (pl.col("valid_time") >= origin)
            & (pl.col("valid_time") < origin + timedelta(hours=hours))
        )
        return forecast_panel(f=f, y=y, title=title, height=height)

    # (1) day-ahead forecasts (first 24 h) in a winter and a summer month, (2) the 14-day horizon of one forecast
    wint, summ = datetime(2026, 1, 12, 12, tzinfo=UTC), datetime(2026, 7, 14, 12, tzinfo=UTC)
    for site in ("D13", "D6"):
        panels = [
            window(
                site,
                wint,
                24,
                f"{SITE_NAMES[site]}: day ahead, origin {wint:%Y-%m-%d} 12:00 UTC (winter)",
            ),
            window(
                site,
                summ,
                24,
                f"{SITE_NAMES[site]}: day ahead, origin {summ:%Y-%m-%d} 12:00 UTC (summer)",
            ),
        ]
        save(
            alt.vconcat(*panels).resolve_scale(y="independent"),
            out / f"figure_day_ahead_{site}.svg",
        )
        panels = [
            window(
                site,
                wint,
                336,
                f"{SITE_NAMES[site]}: 14-day horizon, origin {wint:%Y-%m-%d} 12:00 UTC (winter)",
                220,
            ),
            window(
                site,
                summ,
                336,
                f"{SITE_NAMES[site]}: 14-day horizon, origin {summ:%Y-%m-%d} 12:00 UTC (summer)",
                220,
            ),
        ]
        save(alt.vconcat(*panels).resolve_scale(y="independent"), out / f"figure_14_day_{site}.svg")
    # (3) the largest observed demand of the year at each series: forecast issued 2 days before the day
    panels = []
    for site in files:
        a = actual(site).filter(pl.col("power").is_not_null())
        peak = a.sort("power", descending=True).row(0, named=True)
        day = peak["valid_time"].replace(hour=0, minute=0)
        origin = (day - timedelta(days=2)).replace(hour=12)
        panels.append(
            window(
                site,
                origin,
                72,
                f"{SITE_NAMES[site]}: largest observed demand of the year ({peak['power']:.0f} MW at {peak['valid_time']:%Y-%m-%d %H:%M} UTC); origin {origin:%Y-%m-%d} 12:00",
                200,
            )
        )
        print(site, "peak", peak, flush=True)
    save(alt.vconcat(*panels).resolve_scale(y="independent"), out / "figure_largest_demand.svg")


def main(score: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    save(lead_chart(score=score), out / "figure_crps_by_day.svg")
    by_gm = (
        pl.read_csv(score / "by_group_month.csv")
        if (score / "by_group_month.csv").exists()
        else None
    )
    if by_gm is not None:
        save(month_chart(scored=by_gm), out / "figure_crps_by_month.svg")
    examples(out=out)
    _ = (sc, common, np)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
