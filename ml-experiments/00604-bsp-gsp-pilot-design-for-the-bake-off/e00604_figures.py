"""Draw experiment 604's two figures from NGED's cleaned power Delta table (read-only, no model).

Usage: python e00604_figures.py NGED_DIR
"""

import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import altair as alt
import polars as pl
from plotting import ocf_theme as theme

NGED: Final[Path] = Path(sys.argv[1])
DOCS: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00604-bsp-gsp-pilot-design-for-the-bake-off"
)
LABEL: Final[dict[int, str]] = {1: "D17 (bulk)", 2: "D13 (bulk)", 27: "D6 (grid)", 28: "D1 (grid)"}
KIND: Final[dict[int, str]] = {
    1: "Bulk supply points",
    2: "Bulk supply points",
    27: "Grid supply points",
    28: "Grid supply points",
}
COLOUR: Final[dict[str, str]] = {
    "D17 (bulk)": theme.DATA_BLUE,
    "D13 (bulk)": theme.DATA_SKY,
    "D6 (grid)": theme.BRAND_ORANGE,
    "D1 (grid)": theme.DATA_PURPLE,
}

WIDTH: Final = 620


def load() -> pl.DataFrame:
    """Return the four series' cleaned power, with flagged half-hours set to null."""
    return (
        pl.scan_delta(str(NGED / "cleaned_power_time_series.delta"))
        .filter(pl.col("time_series_id").is_in(list(LABEL)))
        .with_columns(
            pl.when(pl.col("drop_reason").is_null()).then(pl.col("power")).alias("power"),
            pl.col("time_series_id").replace_strict(LABEL, return_dtype=pl.String).alias("series"),
            pl.col("time_series_id").replace_strict(KIND, return_dtype=pl.String).alias("kind"),
        )
        .collect()
    )


def figure_history(df: pl.DataFrame) -> None:
    """Weekly mean power of the four series, one panel per kind."""
    weekly = (
        df.sort("time")
        .group_by_dynamic("time", every="1w", group_by=["series", "kind"])
        .agg(pl.col("power").mean().alias("mw"), pl.col("power").is_not_null().sum().alias("n"))
        .filter(pl.col("n") >= 300)
        .with_columns(pl.col("time").dt.replace_time_zone(None))
    )
    panels = []
    for kind in ("Bulk supply points", "Grid supply points"):
        sub = weekly.filter(pl.col("kind") == kind)
        lines = (
            alt.Chart(sub)
            .mark_line(strokeWidth=1.6)
            .encode(
                x=alt.X("time:T", title="Week"),
                y=alt.Y("mw:Q", title="Weekly mean power (MW)"),
                color=alt.Color(
                    "series:N",
                    scale=alt.Scale(
                        domain=sub["series"].unique().sort().to_list(),
                        range=[COLOUR[s] for s in sub["series"].unique().sort().to_list()],
                    ),
                    legend=alt.Legend(title="Series", orient="right"),
                ),
                detail="series:N",
            )
        )
        panels.append(lines.properties(width=WIDTH, height=300, title=kind))
    alt.vconcat(*panels).resolve_scale(color="independent").save(DOCS / "figure_1_history.svg")


def figure_pairs(df: pl.DataFrame) -> None:
    """Draw 14 days of a grid supply point beside the bulk supply point beneath it."""
    start, end = datetime(2026, 6, 8, tzinfo=UTC), datetime(2026, 6, 22, tzinfo=UTC)
    sub = df.filter((pl.col("time") >= start) & (pl.col("time") < end)).with_columns(
        pl.col("time").dt.replace_time_zone(None)
    )
    pairs = {
        "Walpole pair: D6 (grid) above D17 (bulk)": ("D6 (grid)", "D17 (bulk)"),
        "Bicker Fen pair: D1 (grid) above D13 (bulk)": ("D1 (grid)", "D13 (bulk)"),
    }
    panels = []
    for title, members in pairs.items():
        panels.append(
            alt.Chart(sub.filter(pl.col("series").is_in(list(members))))
            .mark_line(strokeWidth=1.2)
            .encode(
                x=alt.X("time:T", title="Time (UTC)"),
                y=alt.Y("power:Q", title="Power (MW)"),
                color=alt.Color(
                    "series:N",
                    scale=alt.Scale(domain=list(members), range=[COLOUR[s] for s in members]),
                    legend=alt.Legend(title="Series", orient="right"),
                ),
            )
            .properties(width=WIDTH, height=300, title=title)
        )
    alt.vconcat(*panels).resolve_scale(color="independent").save(
        DOCS / "figure_2_pairs_june_2026.svg"
    )


if __name__ == "__main__":
    frame = load()
    figure_history(frame)
    figure_pairs(frame)
