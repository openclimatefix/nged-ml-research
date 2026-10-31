"""Draw experiment 431's published-safe figures (shares, areas under the curve) as SVG.

    uv run --no-sync python -I ml-experiments/00431-wind-idle-data-look/e00431_charts.py

Afterwards optimise each with `npx svgo@4 --multipass --precision=1 --final-newline`. No chart shows
a raw farm-day, a calendar date, or a capacity in MW; farms are Farm A, Farm B, and Farm C.
"""

import os
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl

DATA_DIR: Final[Path] = (
    Path(os.environ["DATA_PATH_INTERNAL"]) / "ml-experiments" / "00431-wind-idle-data-look"
)
OUT_DIR: Final[Path] = Path("docs/ml-experiments/00431-wind-idle-data-look")
FARMS: Final[dict[str, str]] = {"W1": "Farm A", "W2": "Farm B", "W3": "Farm C"}
COLOURS: Final[dict[str, str]] = {
    "Farm A": ocf.BRAND_ORANGE,
    "Farm B": ocf.DATA_BLUE,
    "Farm C": ocf.DATA_GREEN,
}
WIDTH: Final[int] = 480


def farm(frame: pl.DataFrame) -> pl.DataFrame:
    """Replace the stored site code with the farm label."""
    return frame.with_columns(farm=pl.col("site").replace_strict(FARMS))


def figure_auc() -> None:
    """Level-matched AUC of each level-free statistic, with year-block intervals."""
    auc = farm(pl.read_parquet(DATA_DIR / "auc.parquet")).filter(pl.col("gate_statistic"))
    auc = auc.filter(pl.col("auc_level_matched").is_not_nan())
    base = alt.Chart(auc).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y(
            "statistic:N", title=None, sort=alt.SortField("auc_level_matched", order="descending")
        ),
        color=alt.Color(
            "farm:N", scale=alt.Scale(domain=list(COLOURS), range=list(COLOURS.values()))
        ),
    )
    chart = (
        base.mark_rule(aria=False).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "matched_lo:Q",
                title="Level-matched AUC (0.5 = no separation)",
                scale=alt.Scale(domain=[0.45, 0.9]),
            ),
            x2="matched_hi:Q",
        )
        + base.mark_point(filled=True, size=40, aria=False).encode(x="auc_level_matched:Q")  # ty: ignore[unresolved-attribute]
        + alt.Chart(pl.DataFrame({"x": [0.65]}))
        .mark_rule(strokeDash=[4, 3], color="grey")
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    ).properties(width=WIDTH, height=280)
    chart.facet(row=alt.Row("farm:N", title=None)).save(OUT_DIR / "figure_1_auc.svg")  # ty: ignore[unresolved-attribute]


def figure_heatmap() -> None:
    """Idle share of high-wind half-hours by weekday and UTC hour."""
    hw = (
        farm(pl.read_parquet(DATA_DIR / "high_wind_idle_used.parquet"))
        .group_by("farm", "weekday", "hour")
        .agg(n=pl.col("n").sum(), n_idle=pl.col("n_idle").sum())
        .with_columns(share=pl.col("n_idle") / pl.col("n"))
        .filter(pl.col("n") >= 30)
    )
    alt.Chart(hw).mark_rect(aria=False).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("hour:O", title="Hour of day (UTC)"),
        y=alt.Y("weekday:O", title="Weekday (1 = Monday)"),
        color=alt.Color("share:Q", title="Idle share", scale=alt.Scale(scheme="blues")),
    ).properties(width=WIDTH, height=130).facet(row=alt.Row("farm:N", title=None)).save(  # ty: ignore[unresolved-attribute]
        OUT_DIR / "figure_2_idle_share_weekday_hour.svg"
    )


def figure_ratio() -> None:
    """Weekday-daytime over other idle share at high wind, with year-block intervals."""
    ratio = farm(pl.read_parquet(DATA_DIR / "maintenance_ratio.parquet"))
    base = alt.Chart(ratio).encode(
        y=alt.Y("farm:N", title=None), color=alt.Color("cut_in:N", title="High-wind speed")
    )  # ty: ignore[unresolved-attribute]
    (
        base.mark_rule(aria=False).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("lo:Q", title="Idle share, weekday daytime over other hours"), x2="hi:Q"
        )
        + base.mark_point(filled=True, aria=False).encode(x="ratio:Q")  # ty: ignore[unresolved-attribute]
        + alt.Chart(pl.DataFrame({"x": [1.5]}))
        .mark_rule(strokeDash=[4, 3], color="grey")
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    ).properties(width=WIDTH, height=140).save(OUT_DIR / "figure_3_weekday_ratio.svg")


def figure_floor() -> None:
    """Median and 95th percentile of the idle floor, as a share of capacity, by year."""
    floor = farm(pl.read_parquet(DATA_DIR / "floor_by_year.parquet")).filter(pl.col("n") >= 200)
    base = alt.Chart(floor).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("year:O", title="Year"),
        color=alt.Color(
            "farm:N", scale=alt.Scale(domain=list(COLOURS), range=list(COLOURS.values()))
        ),
    )
    (
        base.mark_line(point=True, aria=False).encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("median:Q", title="Idle floor median (share of capacity)")
        )
        + base.mark_line(strokeDash=[4, 3], aria=False).encode(y="p95:Q")  # ty: ignore[unresolved-attribute]
    ).properties(width=WIDTH, height=220).save(OUT_DIR / "figure_4_floor_by_year.svg")


def figure_negatives() -> None:
    """Farm B's share of negative readings at ERA5 wind below 2 m/s, by UTC hour."""
    neg = pl.read_parquet(DATA_DIR / "farm_b_negatives_by_hour.parquet")
    alt.Chart(neg).mark_bar(color=ocf.DATA_BLUE, aria=False).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("hour:O", title="Hour of day (UTC)"),
        y=alt.Y(
            "share_negative:Q", title="Share of readings below zero", scale=alt.Scale(domain=[0, 1])
        ),
    ).properties(width=WIDTH, height=200).save(OUT_DIR / "figure_5_farm_b_negatives_by_hour.svg")


def figure_agreement() -> None:
    """Farm B against Farm C: which farm reads idle, by shared-wind class."""
    both = (
        pl.read_parquet(DATA_DIR / "b_vs_c.parquet")
        .group_by("wind_class", "b_idle", "c_idle")
        .agg(n=pl.col("n").sum())
        .with_columns(
            reading=pl.when(pl.col("b_idle") & pl.col("c_idle"))
            .then(pl.lit("both idle"))
            .when(pl.col("b_idle"))
            .then(pl.lit("only Farm B idle"))
            .when(pl.col("c_idle"))
            .then(pl.lit("only Farm C idle"))
            .otherwise(pl.lit("neither idle"))
        )
        .group_by("wind_class", "reading")
        .agg(n=pl.col("n").sum())
        .with_columns(share=pl.col("n") / pl.col("n").sum().over("wind_class"))
    )
    alt.Chart(both).mark_bar(aria=False).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y(
            "wind_class:N", title="Shared ERA5 wind", sort=["below 2 m/s", "between", "high wind"]
        ),
        x=alt.X("share:Q", title="Share of half-hours", stack="normalize"),
        color=alt.Color("reading:N", title=None),
    ).properties(width=WIDTH, height=140).save(OUT_DIR / "figure_6_farm_b_vs_c.svg")


def figure_runs() -> None:
    """Share of high-wind idle half-hours by the length of the run they sit in."""
    kinds = {
        "used cut-in": "Pre-registered: in the idle band, no gaps",
        "post hoc: used cut-in, at or below band top, gaps up to 3": (
            "Post hoc: at or below band top, gaps up to 3"
        ),
    }
    runs = farm(
        pl.read_parquet(DATA_DIR / "run_buckets.parquet").filter(pl.col("kind").is_in(list(kinds)))
    ).with_columns(definition=pl.col("kind").replace_strict(kinds))
    alt.Chart(runs).mark_bar(aria=False).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("farm:N", title=None),
        x=alt.X(
            "share_of_half_hours:Q", title="Share of high-wind idle half-hours", stack="normalize"
        ),
        color=alt.Color(
            "bucket:N",
            title="Run length",
            sort=["up to 3 h", "3 to 12 h", "12 to 24 h", "1 to 7 days", "over 7 days"],
        ),
        row=alt.Row("definition:N", title=None),
    ).properties(width=WIDTH, height=90).save(OUT_DIR / "figure_7_idle_run_lengths.svg")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for draw in (
        figure_auc,
        figure_heatmap,
        figure_ratio,
        figure_floor,
        figure_negatives,
        figure_agreement,
        figure_runs,
    ):
        draw()
