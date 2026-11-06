# ruff: noqa: E501, DTZ001
"""Draw experiment 605's three figures from the stored check tables. No forecast is drawn."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import altair as alt
import polars as pl
from e00605_power import DEVELOPMENT_MONTHS, MONTHS, OUT_DIR
from plotting import ocf_theme as theme
from studies.sources import REPO_DATA_DIR

DOCS: Final[Path] = (
    Path(__file__).resolve().parents[2] / "docs/ml-experiments/00605-bakeoff-final-test-data-check"
)
WIDTH: Final[int] = 640
BANDS: Final[dict[str, str]] = {
    "under 0.5%": theme.DATA_SKY_LIGHT,
    "0.5% to 2%": "#FFDABC",
    "2% to 5%": "#FFB48F",
    "5% to 10%": theme.BRAND_ORANGE_LIGHT,
    "10% or more": theme.BRAND_ORANGE,
}
LAST_NEEDED: Final[datetime] = datetime(2026, 10, 31, 23, 30, tzinfo=UTC)
TEST_START: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)


def month_label(month: str) -> str:
    """Return the axis label of a month, marking the development months."""
    return f"{month} (dev)" if month in DEVELOPMENT_MONTHS else month


def figure_missing() -> None:
    """Heat map: share of half-hours with no reading, per series and month."""
    t = pl.read_parquet(OUT_DIR / "power_by_series_month.parquet").filter(
        pl.col("group") != "Solar"
    )
    t = t.with_columns(
        missing_pct=100 * pl.col("missing") / pl.col("expected"),
        month_label=pl.col("month").map_elements(month_label, return_dtype=pl.String),
    )
    t = t.with_columns(
        band=pl.when(pl.col("missing_pct") < 0.5)
        .then(pl.lit("under 0.5%"))
        .when(pl.col("missing_pct") < 2)
        .then(pl.lit("0.5% to 2%"))
        .when(pl.col("missing_pct") < 5)
        .then(pl.lit("2% to 5%"))
        .when(pl.col("missing_pct") < 10)
        .then(pl.lit("5% to 10%"))
        .otherwise(pl.lit("10% or more"))
    )
    order = [month_label(m) for m in MONTHS]
    series = t.select("group", "label").unique().sort("group", "label")["label"].to_list()
    chart = (
        alt.Chart(t.to_pandas())
        .mark_rect(opacity=1)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("month_label:N", sort=order, title=None, axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("label:N", sort=series, title=None, axis=alt.Axis(labelLimit=320)),
            color=alt.Color(
                "band:N",
                scale=alt.Scale(domain=list(BANDS), range=list(BANDS.values())),
                legend=alt.Legend(
                    orient="top", title="Half-hours with no reading", symbolType="square"
                ),
            ),
            tooltip=["label", "month", "missing_pct"],
        )
        .properties(
            width=WIDTH,
            height=20 * len(series),
            title=alt.TitleParams(
                [
                    "Most Primaries and grid supply points miss under 2% of half-hours; the wind farms miss up to 10%",
                    "Months marked (dev) are development months. 2026-07 to 2026-10 are the test window.",
                ],
                anchor="start",
            ),
        )
    )
    chart.save(DOCS / "missing_half_hours.svg")


def zero_share() -> pl.DataFrame:
    """Return the share of half-hours that read exactly zero, per generator group and month."""
    ids = {
        "Solar": [21, 22, 23, 29, 30, 31],
        "Wind": [4, 5, 6],
    }
    flat = [(group, sid) for group, sids in ids.items() for sid in sids]
    frame = (
        pl.scan_delta(str(REPO_DATA_DIR / "NGED" / "power_time_series.delta"))
        .filter(pl.col("time_series_id").is_in([s for _, s in flat]))
        .filter(pl.col("time") > datetime(2025, 9, 1, tzinfo=UTC))
        .collect()
        .join(
            pl.DataFrame({"time_series_id": [s for _, s in flat], "group": [g for g, _ in flat]}),
            on="time_series_id",
        )
    )
    out = (
        frame.with_columns(month=(pl.col("time") - pl.duration(minutes=30)).dt.strftime("%Y-%m"))
        .group_by("group", "month")
        .agg(
            zero_pct=100 * (pl.col("power") == 0).mean(),
            rows_per_day=pl.len()
            / pl.col("time").dt.date().n_unique()
            / pl.col("time_series_id").n_unique(),
        )
        .filter(pl.col("month") <= "2026-10")
        .sort("group", "month")
    )
    out.write_parquet(OUT_DIR / "generator_zero_share.parquet")
    return out


def figure_zeros() -> None:
    """Two panels: readings of exactly zero stop in late March 2026, solar above wind."""
    t = zero_share()
    colours = {"Solar": theme.BRAND_ORANGE, "Wind": theme.DATA_BLUE}
    panels = []
    for group, colour in colours.items():
        panels.append(
            alt.Chart(t.filter(pl.col("group") == group).to_pandas())
            .mark_line(point=True, color=colour)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("month:O", title=None, axis=alt.Axis(labelAngle=-45)),
                y=alt.Y("zero_pct:Q", title=f"{group}: half-hours reading 0 (%)"),
            )
            .properties(
                width=WIDTH,
                height=280,
                title=alt.TitleParams(
                    f"{'Six solar farms' if group == 'Solar' else 'Three wind farms'}",
                    anchor="start",
                ),
            )
        )
    chart = alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "Solar and wind farms stop reporting half-hours of exactly zero after March 2026",
                "Each point is the share of that month's stored half-hours that read 0. March ends on the 26th.",
            ],
            anchor="start",
        )
    )
    chart.save(DOCS / "generator_zero_readings.svg")


def figure_ends() -> None:
    """Dot chart: the newest date each input holds, against the test window."""
    inputs = (
        pl.read_parquet(OUT_DIR / "inputs_end.parquet")
        .with_columns(
            newest=pl.coalesce(
                pl.col("newest_run").str.to_datetime("%Y-%m-%d", time_zone="UTC"), pl.col("newest")
            ).dt.cast_time_unit("us"),
            input=pl.when(pl.col("newest_run").is_not_null())
            .then(pl.col("input") + pl.lit(" (newest run)"))
            .otherwise(pl.col("input")),
        )
        .select("input", "newest")
    )
    power_end = pl.read_parquet(OUT_DIR / "series_end.parquet")["last"].max()
    ens_end = pl.read_parquet(OUT_DIR / "ens_runs.parquet")["init_time"].max()
    extra = pl.DataFrame(
        {
            "input": [
                "Power, production table (newest reading)",
                "ENS, production table (newest run)",
            ],
            "newest": [power_end, ens_end],
        }
    ).with_columns(pl.col("newest").dt.cast_time_unit("us"))
    t = pl.concat([extra, inputs], how="vertical_relaxed").with_columns(
        reaches=pl.when(pl.col("newest") >= LAST_NEEDED)
        .then(pl.lit("Reaches 2026-10-31"))
        .otherwise(pl.lit("Stops before 2026-10-31"))
    )
    order = t["input"].to_list()
    colours = {
        "Reaches 2026-10-31": theme.DATA_GREEN,
        "Stops before 2026-10-31": theme.BRAND_ORANGE,
    }
    window = pl.DataFrame({"start": [TEST_START], "end": [LAST_NEEDED]}).to_pandas()
    shade = (
        alt.Chart(window)
        .mark_rect(color=theme.DATA_SKY_LIGHT, opacity=0.6)
        .encode(x="start:T", x2="end:T")  # ty: ignore[unresolved-attribute]
    )
    dots = (
        alt.Chart(t.to_pandas())
        .mark_point(filled=True, size=120)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "newest:T",
                title="Newest timestamp held",
                scale=alt.Scale(domain=[datetime(2024, 1, 1), datetime(2026, 12, 1)]),
                axis=alt.Axis(
                    format="%b %Y",
                    values=[
                        datetime(2024, 1, 1),
                        datetime(2025, 1, 1),
                        datetime(2026, 1, 1),
                        datetime(2026, 7, 1),
                        datetime(2026, 11, 1),
                    ],
                ),
            ),
            y=alt.Y("input:N", sort=order, title=None, axis=alt.Axis(labelLimit=420)),
            color=alt.Color(
                "reaches:N",
                scale=alt.Scale(domain=list(colours), range=list(colours.values())),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
    )
    chart = alt.layer(shade, dots).properties(
        width=WIDTH,
        height=26 * len(order),
        title=alt.TitleParams(
            [
                "The derived ensemble inputs and the reanalysis inputs stop before the test window ends",
                "Blue band: the test window, 2026-07-01 to 2026-10-31. A dot left of the band's right edge marks a gap.",
            ],
            anchor="start",
        ),
    )
    chart.save(DOCS / "newest_date_of_each_input.svg")


if __name__ == "__main__":
    DOCS.mkdir(parents=True, exist_ok=True)
    figure_missing()
    figure_zeros()
    figure_ends()
