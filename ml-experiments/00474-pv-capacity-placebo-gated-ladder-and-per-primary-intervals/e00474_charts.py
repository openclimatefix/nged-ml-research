"""Figures of experiment 474: estimated unmetered PV capacity per Primary, with intervals.

Reads `work/table_<name>.parquet` (the stage 1 table) and writes SVG figures into the experiment's
docs folder. Substations may be named; no metered generator appears.
"""

import argparse
from pathlib import Path
from typing import Final

import altair as alt
import e00474_common as common
import plotting.ocf_theme as ocf
import polars as pl

WIDTH: Final[int] = 680
DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
READ_OUT: Final[int] = 2025
FIRST_SEASON_YEAR: Final[dict[str, int]] = {
    "D9": 2023,
    "D18": 2024,
    "D16": 2020,
    "D14": 2020,
    "D4": 2021,
    "D12": 2020,
    "D8": 2019,
}
"""The first season-year (July to June) with readings; earlier capacities are not estimated."""
DC_LABEL: Final[str] = "DC capacity (assumed ratio)"


def overview(*, table: pl.DataFrame, title: str) -> alt.Chart:
    """One panel: AC and DC capacity range of the latest season-year at every Primary."""
    year = table.filter(pl.col("season_year") == READ_OUT, pl.col("ac_p50").is_not_null())
    order = year.sort("ac_p50", descending=True)["site"].to_list()
    long = pl.concat(
        [
            year.select(
                "site",
                kind=pl.lit("AC capacity"),
                low=pl.col("ac_p10"),
                mid=pl.col("ac_p50"),
                high=pl.col("ac_p90"),
            ),
            year.select(
                "site",
                kind=pl.lit(DC_LABEL),
                low=pl.col("dc_p10"),
                mid=pl.col("dc_p50"),
                high=pl.col("dc_p90"),
            ),
        ]
    )
    scale = alt.Scale(
        domain=["AC capacity", DC_LABEL],
        range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE],
    )
    base = alt.Chart(long.to_pandas() if False else long).encode(
        y=alt.Y("site:N", sort=order, title=None),
        yOffset=alt.YOffset("kind:N"),
        color=alt.Color("kind:N", scale=scale, legend=alt.Legend(orient="top", title=None)),
    )
    bars = base.mark_rule(strokeWidth=4, aria=False).encode(
        x=alt.X("low:Q", title="Estimated unmetered solar capacity (MW)"), x2="high:Q"
    )
    dots = base.mark_point(filled=True, size=50, aria=False).encode(x="mid:Q")
    pinned = (
        alt.Chart(year.filter(pl.col("pinned_mw").is_not_null()))
        .mark_tick(color="#606060", thickness=2, size=18, aria=False)
        .encode(y=alt.Y("site:N", sort=order), x="pinned_mw:Q")
    )
    return alt.layer(bars, dots, pinned).properties(
        width=WIDTH - 90,
        height=26 * len(order) + 30,
        title=alt.TitleParams(
            text=title,
            subtitle=[
                "Season-year July 2025 to June 2026. Dot: median. Line: 10th to 90th percentile",
                "of a month-block bootstrap widened by the spread of known-answer recoveries.",
                "Grey tick: capacity pinned from NGED's register (where one exists).",
            ],
            anchor="start",
        ),
    )


def panels(*, table: pl.DataFrame, title: str) -> alt.Chart:
    """Per-Primary panels: AC and DC capacity by season-year with 10th to 90th percentile bands."""
    data = table.filter(pl.col("ac_p50").is_not_null(), pl.col("season_year") >= 2019).filter(
        pl.col("season_year") >= pl.col("site").replace_strict(FIRST_SEASON_YEAR, default=2019)
    )
    sites = sorted(data["site"].unique().to_list(), key=lambda s: int(s[1:]))
    base = alt.Chart(data).encode(x=alt.X("season_year:Q"))

    def layer(prefix: str, colour: str) -> alt.LayerChart:
        band = base.mark_area(opacity=0.25, color=colour, aria=False).encode(
            x=alt.X(
                "season_year:Q",
                title="Season-year (starts July)",
                axis=alt.Axis(format="d", values=[2019, 2021, 2023, 2025]),
            ),
            y=alt.Y(f"{prefix}_p10:Q", title="MW"),
            y2=f"{prefix}_p90:Q",
        )
        line = base.mark_line(color=colour, aria=False).encode(
            x="season_year:Q", y=f"{prefix}_p50:Q"
        )
        return band + line

    chart = alt.layer(layer("dc", ocf.DATA_BLUE), layer("ac", ocf.BRAND_ORANGE)).properties(
        width=150, height=100
    )
    return (
        chart.facet(facet=alt.Facet("site:N", sort=sites, title=None), columns=4)
        .properties(
            title=alt.TitleParams(
                text=title,
                subtitle=[
                    "Orange: AC capacity. Blue: DC capacity (AC times an assumed ratio of 1.1 to 1.5).",
                    "Line: median. Band: 10th to 90th percentile. Fits use rows before 2026-07-01;",
                    "D18 has data only from 2025-03-01 and is report-only.",
                ],
                anchor="start",
            )
        )
        .resolve_scale(y="independent")
    )


def main() -> None:
    """Write the figures for a table."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", required=True)
    parser.add_argument("--stem", required=True)
    arguments = parser.parse_args()
    table = pl.read_parquet(common.WORK_DIR / f"{arguments.table}.parquet")
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    overview(table=table, title="Unmetered solar capacity: estimated range at every Primary").save(
        DOCS_DIR / f"{arguments.stem}_overview.svg"
    )
    panels(table=table, title="Unmetered solar capacity by season-year at each Primary").save(
        DOCS_DIR / f"{arguments.stem}_panels.svg"
    )
    overview(table=table, title="x").save(
        common.WORK_DIR / f"{arguments.stem}_overview.png", scale_factor=1.5
    )
    panels(table=table, title="x").save(
        common.WORK_DIR / f"{arguments.stem}_panels.png", scale_factor=1.5
    )


if __name__ == "__main__":
    main()
