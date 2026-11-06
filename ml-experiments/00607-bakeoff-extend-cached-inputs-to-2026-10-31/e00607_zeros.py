# ruff: noqa: D103, DTZ001, E501
"""Why the solar and wind series have no exact-zero reading after 2026-03-26, and why Solar E misses midday.

Reads only the raw power table and the downloaded-file list. Draws two figures and writes
`zeros_weekly.parquet` and `solar_e_midday.parquet`. Generators are anonymised: only the group
label and the label `E` appear, with weekly counts and monthly medians, no dated time series.
"""

from datetime import datetime
from pathlib import Path
from typing import Final

import altair as alt
import polars as pl
from plotting import ocf_theme as theme
from studies.pv_dataset import pv_sites, wind_sites
from studies.sources import ML_EXPERIMENTS_DATA_DIR, REPO_DATA_DIR

OUT: Final[Path] = next(ML_EXPERIMENTS_DATA_DIR.glob("00607-*"))
DOCS: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00607-bakeoff-extend-cached-inputs-to-2026-10-31"
)
SWITCH: Final = pl.datetime(2026, 3, 26, time_zone="UTC")
WIDTH: Final[int] = 640


def power() -> pl.DataFrame:
    solar = pv_sites().select(
        "time_series_id", label=pl.concat_str(pl.lit("Solar "), "site"), group=pl.lit("Solar farms")
    )
    wind = wind_sites().select("time_series_id", label=pl.col("site"), group=pl.lit("Wind farms"))
    sites = pl.concat([solar, wind])
    return (
        pl.scan_delta(str(REPO_DATA_DIR / "NGED" / "power_time_series.delta"))
        .filter(pl.col("time_series_id").is_in(sites["time_series_id"].to_list()))
        .filter(pl.col("time") >= pl.datetime(2025, 12, 1, time_zone="UTC"))
        .collect()
        .join(sites, on="time_series_id")
    )


def weekly(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Per group and week: readings per series per day, and the share of readings that are exactly 0."""
    out = (
        frame.filter(pl.col("time") < pl.datetime(2026, 10, 26, time_zone="UTC"))
        .with_columns(week=pl.col("time").dt.truncate("1w"))
        .group_by("group", "week")
        .agg(
            readings_per_series_day=pl.len() / pl.col("label").n_unique() / 7,
            zero_share=100 * (pl.col("power") == 0).mean(),
        )
        .sort("group", "week")
    )
    out.write_parquet(OUT / "zeros_weekly.parquet")
    return out


def figure_weekly(*, table: pl.DataFrame) -> None:
    rule = (
        alt.Chart(pl.DataFrame({"t": [datetime(2026, 3, 26)]}).to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[4, 3], color=theme.DATA_PURPLE)
        .encode(x="t:T")
    )
    colours = alt.Scale(
        domain=["Solar farms", "Wind farms"], range=[theme.BRAND_ORANGE, theme.DATA_BLUE]
    )
    base = alt.Chart(table.to_pandas()).encode(
        x=alt.X("week:T", title="Week starting", axis=alt.Axis(format="%b %Y")),
        color=alt.Color("group:N", scale=colours, legend=alt.Legend(title=None, orient="top")),
    )
    top = (
        base.mark_line(point=True).encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("zero_share:Q", title="Readings exactly 0 (% of readings)")
        )
        + rule
    ).properties(width=WIDTH, height=280)
    bottom = (
        base.mark_line(point=True).encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y(
                "readings_per_series_day:Q", title="Readings per series per day (48 = complete)"
            )
        )
        + rule
    ).properties(width=WIDTH, height=280)
    chart = alt.vconcat(top, bottom).properties(
        title=alt.TitleParams(
            "Zero readings stop on 2026-03-26, and readings per day fall",
            subtitle=[
                "Dashed purple line: 2026-03-26, when the stored files change from the bulk history export to 6-hourly files.",
                "Each point is one week, pooled over 6 solar farms or 3 wind farms.",
            ],
            anchor="start",
        )
    )
    chart.save(DOCS / "zero_readings_vanish.svg")


def solar_e_midday(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Per Solar farm and month: share of midday half-hours with a reading, and the median of the daily midday peak."""
    mid = (
        frame.filter(
            pl.col("group") == "Solar farms",
            pl.col("time") > pl.datetime(2026, 3, 27, time_zone="UTC"),
            pl.col("time") < pl.datetime(2026, 11, 1, time_zone="UTC"),
        )
        .filter(pl.col("time").dt.hour().is_between(10, 13))
        .with_columns(day=pl.col("time").dt.date(), month=pl.col("time").dt.strftime("%Y-%m"))
        .group_by("label", "month", "day")
        .agg(n=pl.len(), peak=pl.col("power").max())
    )
    out = (
        mid.group_by("label", "month")
        .agg(
            present_pct=100 * pl.col("n").sum() / (8 * pl.col("day").n_unique()),
            median_daily_peak_mw=pl.col("peak").median(),
            days_seen=pl.col("day").n_unique(),
        )
        .sort("label", "month")
    )
    out.write_parquet(OUT / "solar_e_midday.parquet")
    return out


def figure_solar_e(*, table: pl.DataFrame) -> None:
    e = table.filter(pl.col("label") == "Solar E").filter(pl.col("month") >= "2026-04")
    base = alt.Chart(e.to_pandas()).encode(
        x=alt.X("month:N", title="Month (2026)", axis=alt.Axis(labelAngle=0))
    )
    top = (
        base.mark_bar(color=theme.DATA_SKY)  # ty: ignore[unresolved-attribute]
        .encode(
            y=alt.Y(
                "present_pct:Q",
                title="Midday half-hours with a reading (%)",
                scale=alt.Scale(domain=[0, 100]),
            )
        )
        .properties(width=WIDTH, height=280)
    )
    bottom = (
        base.mark_bar(color=theme.BRAND_ORANGE)  # ty: ignore[unresolved-attribute]
        .encode(
            y=alt.Y(
                "median_daily_peak_mw:Q",
                title="Median daily midday peak (MW)",
                scale=alt.Scale(domain=[0, 18]),
            )
        )
        .properties(width=WIDTH, height=280)
    )
    alt.vconcat(top, bottom).properties(
        title=alt.TitleParams(
            "Solar E lacks midday readings in the months its peak is near zero",
            subtitle=[
                "Midday is 10:00 to 14:00 UTC. In July and October, when readings are scarce, the median daily peak is near zero,",
                "consistent with absent half-hours being zero-output half-hours that the source no longer sends.",
            ],
            anchor="start",
        )
    ).save(DOCS / "solar_e_midday.svg")


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    frame = power()
    figure_weekly(table=weekly(frame=frame))
    t = solar_e_midday(frame=frame)
    figure_solar_e(table=t)
    pl.Config.set_tbl_rows(40)
    print(t.filter(pl.col("label") == "Solar E"))


if __name__ == "__main__":
    main()
