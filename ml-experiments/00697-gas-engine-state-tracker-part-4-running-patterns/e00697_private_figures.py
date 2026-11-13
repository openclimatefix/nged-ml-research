r"""Draw experiment 697's dated figures of D8 into the private data folder only.

    uv run --no-sync python $E/00697-*/e00697_private_figures.py

**Nothing here is committed or published.** The figures hold dated output of D8, so they are written
to `PRIVATE_FIGURES_DIR` inside the private data folder and nowhere else. They are:

- `raster_<era>.svg`: a day-by-day raster, local date on the x axis and local half-hour on the y
  axis, coloured by D8's smoothed flag.
- `example_weeks.svg`: D8's reading, the tracker's fitted block, and the flag for four weeks of the
  net and solar eras, chosen as the weeks with the highest and the lowest flagged share in each.
- `g7_night_share_by_month.svg`: the night share (00:00 to 05:00 local) flagged by month, beside the
  share of night half-hours with D8's residual below -2.5 MW, with rules at the solar start and at
  the fade start. G7 is private because it is a dated series of D8.
"""

from datetime import date
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from e00697_analysis import usable, with_state  # ty: ignore[unresolved-import]
from e00697_common import NIGHT_LAST, PRIVATE_FIGURES_DIR  # ty: ignore[unresolved-import]
from e00697_frame import load_frame  # ty: ignore[unresolved-import]

NIGHT_SHORTFALL_MW: Final[float] = -2.5
SOLAR_START: Final[date] = date(2025, 3, 24)
FADE_START: Final[date] = date(2026, 3, 1)
WEEKS_PER_ERA: Final[int] = 2


def save(*, chart: alt.TopLevelMixin, name: str) -> None:
    """Write an SVG into the private figures folder."""
    PRIVATE_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(PRIVATE_FIGURES_DIR / f"{name}.svg")
    print(f"wrote {name}.svg (private)", flush=True)


def rasters(*, d8: pl.DataFrame) -> None:
    """Draw one day-by-half-hour raster of the smoothed flag for each era."""
    for era in ("net", "solar", "fade"):
        rows = d8.filter(pl.col("era") == era, pl.col("observed")).select(
            "local_date", hour=pl.col("local_half_hour") / 2, flag=pl.col("ts").cast(pl.Int8)
        )
        chart = (
            alt.Chart(rows, title=f"D8 smoothed flag, {era} era")  # ty: ignore[unresolved-attribute]
            .mark_rect(aria=False)
            .encode(
                x=alt.X("local_date:T", title="Local date"),
                y=alt.Y(
                    "hour:Q",
                    title="Local time of day (hours)",
                    axis=alt.Axis(values=[0, 6, 12, 18, 24]),
                    scale=alt.Scale(domain=[0, 24], nice=False),
                ),
                color=alt.Color(
                    "flag:Q", scale=alt.Scale(domain=[0, 1], range=["#FFFFFF", ocf.DATA_BLUE])
                ),
            )
            .properties(width=640, height=220)
        )
        save(chart=chart, name=f"raster_{era}")


def example_weeks(*, d8: pl.DataFrame) -> None:
    """Draw D8's reading and the tracker's block for the busiest and the quietest weeks."""
    panels = []
    for era in ("net", "solar"):
        rows = d8.filter(pl.col("era") == era)
        weekly = (
            rows.group_by("week")
            .agg(n=pl.len(), share=pl.col("ts").mean())
            .filter(pl.col("n") == 7 * 48)
            .sort("share")
        )
        chosen = [
            *weekly["week"].head(WEEKS_PER_ERA).to_list(),
            *weekly["week"].tail(WEEKS_PER_ERA).to_list(),
        ]
        for week in chosen:
            part = rows.filter(pl.col("week") == week).with_columns(
                time=pl.col("start")
                .dt.convert_time_zone("Europe/London")
                .dt.replace_time_zone(None),
                flagged_mw=pl.when(pl.col("ts")).then(pl.col("fitted_mw")),
            )
            base = alt.Chart(part, title=f"{era} era, week of {week}")  # ty: ignore[unresolved-attribute]
            panels.append(
                base.mark_line(aria=False, color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
                    x=alt.X("time:T", title=None), y=alt.Y("reading:Q", title="Reading (MVA)")
                )
                + base.mark_line(aria=False, color=ocf.BRAND_ORANGE).encode(  # ty: ignore[unresolved-attribute]
                    x="time:T", y="flagged_mw:Q"
                )
            )
    save(
        chart=alt.vconcat(*[p.properties(width=640, height=110) for p in panels]),
        name="example_weeks",
    )


def night_share_by_month(*, d8: pl.DataFrame) -> None:
    """Draw G7: the night share flagged and the night share with a negative residual, by month."""
    night = usable(frame=with_state(frame=d8, column="ts")).filter(
        pl.col("local_half_hour") <= NIGHT_LAST, pl.col("expected").is_not_null()
    )
    monthly = (
        night.with_columns(month_start=pl.col("local_date").dt.truncate("1mo"))
        .group_by("month_start")
        .agg(
            flagged=pl.col("state").mean(),
            residual_below=((pl.col("reading") - pl.col("expected")) < NIGHT_SHORTFALL_MW).mean(),
        )
        .sort("month_start")
        .unpivot(index="month_start", variable_name="series", value_name="share")
        .with_columns(
            series=pl.col("series").replace_strict(
                {"flagged": "Flagged as running", "residual_below": "Residual below -2.5 MW"}
            )
        )
    )
    rules = pl.DataFrame({"when": [SOLAR_START, FADE_START]})
    chart = (
        alt.Chart(monthly)  # ty: ignore[unresolved-attribute]
        .mark_line(aria=False, point=True)
        .encode(
            x=alt.X("month_start:T", title="Month"),
            y=alt.Y("share:Q", title="Share of night half-hours", axis=alt.Axis(format=".0%")),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(
                    domain=["Flagged as running", "Residual below -2.5 MW"],
                    range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE],
                ),
                legend=alt.Legend(title=None),
            ),
        )
        + alt.Chart(rules).mark_rule(aria=False, color="black").encode(x="when:T")  # ty: ignore[unresolved-attribute]
    ).properties(width=600, height=200)
    save(chart=chart, name="g7_night_share_by_month")


def main() -> None:
    """Draw every private figure of D8."""
    d8 = load_frame(name="d8")
    rasters(d8=d8)
    example_weeks(d8=d8)
    night_share_by_month(d8=d8)


if __name__ == "__main__":
    main()
