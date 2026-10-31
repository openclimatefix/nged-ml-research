# ruff: noqa: E501
"""Tables and figures of experiment 443 from the stored predictions.

PYTHONPATH=ml-experiments/00310-metered-farm-long-lead-scoring:ml-experiments/00430-unified-forecaster-solar-and-pretraining-ablations:ml-experiments/00443-solar-network-hyperparameter-and-architecture-sweep \
        uv run --no-sync python ml-experiments/00443-*/e00443_report.py
"""

from pathlib import Path
from typing import Final

import altair as alt
import e00310_common as c310  # ty: ignore[unresolved-import]
import e00430_score as s430  # ty: ignore[unresolved-import]
import e00443_common as common
import e00443_score as sc
import plotting.ocf_theme as ocf
import polars as pl

DOCS: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00443-solar-network-hyperparameter-and-architecture-sweep"
)
WIDTH: Final[int] = 440
CHECK_ARMS: Final[tuple[str, ...]] = ("lr004", "w32", "k1")


def seed_arm(*, arm: str, seeds: tuple[int, ...]) -> pl.DataFrame:
    """Return an arm's seed-averaged predictions, from this experiment or experiment 430's skeleton."""
    root = common.E430_PRED_DIR if arm == common.SKELETON else common.PRED_DIR
    return sc.load(root=root, arm=arm, seeds=seeds)


def absolute_and_checks() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return tail scores of the skeleton and the check arms, and their per-farm checks."""
    names = (common.SKELETON, *CHECK_ARMS)
    arms = {n: seed_arm(arm=n, seeds=common.SEEDS_SCREEN) for n in names}
    absolute = s430.absolute_table(arms=arms)
    aligned = s430.common_rows(frames=arms)
    per_farm = []
    for name, frame in aligned.items():
        rows = s430.losses(frame=frame)
        per_farm.append(
            rows.group_by("time_series_id")
            .agg(
                pl.col("median_minus_mean").mean(), pl.col("ae_mean").mean(), pl.col("score").mean()
            )
            .with_columns(arm=pl.lit(name))
        )
    return absolute, pl.concat(per_farm).sort("arm", "time_series_id")


def headline_figure(summary: pl.DataFrame) -> None:
    """Draw the 25 contrasts with 99% intervals, the 0.5% margin, and zero."""
    order = [a for arms in common.FACTORS.values() for a in arms]
    frame = summary.select("arm", "pct_change", "lo99", "hi99").to_dicts()
    base = alt.Chart(alt.Data(values=frame)).encode(y=alt.Y("arm:N", sort=order, title=None))
    bars = base.mark_rule(strokeWidth=3, color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "lo99:Q", title="Change in quantile score against the skeleton (%, negative is better)"
        ),
        x2="hi99:Q",
    )
    points = base.mark_point(filled=True, size=60, color=ocf.BRAND_ORANGE).encode(x="pct_change:Q")  # ty: ignore[unresolved-attribute]
    marks = (
        alt.Chart(alt.Data(values=[{"x": 0}, {"x": -common.MARGIN_PERCENT}]))  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[4, 3], color="#999999")
        .encode(x="x:Q")
    )
    chart = (marks + bars + points).properties(
        width=WIDTH,
        height=520,
        title="Each setting change against the skeleton, 99% intervals; dashed lines mark 0 and the 0.5% margin",
    )
    chart.configure_title(fontSize=12, anchor="start").save(str(DOCS / "headline.svg"))  # ty: ignore[unresolved-attribute]


def example_figures() -> None:
    """Draw a day-ahead and a 10-day forecast of farm A from the skeleton (seed average)."""
    frame = seed_arm(arm=common.SKELETON, seeds=common.SEEDS_SCREEN).filter(
        pl.col("time_series_id") == "A"
    )
    pieces = []
    for season, init in (("summer", "2026-06-15"), ("winter", "2026-01-15")):
        rows = frame.filter(pl.col("init_time").dt.strftime("%Y-%m-%d") == init)
        pieces.append(rows.with_columns(season=pl.lit(season), hours=pl.col("lead_hours")))
    data = pl.concat(pieces)
    for name, bands, title in (
        ("example_day_ahead.svg", ["d1"], "Day ahead (12 to 36 hours), farm A"),
        ("example_10_day.svg", list(c310.BAND_ORDER), "Days 1 to 10, farm A"),
    ):
        d = data.filter(pl.col("band").is_in(bands)).sort("hours")
        base = alt.Chart(alt.Data(values=d.to_dicts())).encode(
            x=alt.X("hours:Q", title="Hours since the run")
        )
        band90 = base.mark_area(opacity=0.25, color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("q05:Q", title="Output (share of capacity)"), y2="q95:Q"
        )
        band50 = base.mark_area(opacity=0.35, color=ocf.DATA_BLUE).encode(y="q25:Q", y2="q75:Q")  # ty: ignore[unresolved-attribute]
        median = base.mark_line(color=ocf.DATA_BLUE).encode(y="q50:Q")  # ty: ignore[unresolved-attribute]
        actual = base.mark_point(size=12, color=ocf.BRAND_ORANGE, filled=True).encode(y="actual:Q")  # ty: ignore[unresolved-attribute]
        chart = (
            alt.layer(band90, band50, median, actual)
            .properties(width=WIDTH, height=150)
            .facet(row=alt.Row("season:N", title=None), title=title)
        )
        chart.save(str(DOCS / name))  # ty: ignore[invalid-argument-type]


def main() -> None:
    """Write every table and figure."""
    pl.Config.set_tbl_rows(60)
    pl.Config.set_tbl_cols(14)
    pl.Config.set_tbl_width_chars(220)
    summary = pl.read_csv(common.SCORE_DIR / "stage1.csv")
    absolute, per_farm = absolute_and_checks()
    absolute.write_csv(common.SCORE_DIR / "tails.csv")
    per_farm.write_csv(common.SCORE_DIR / "checks.csv")
    print(absolute.with_columns(pl.col(pl.Float64).round(4)))
    print(per_farm.with_columns(pl.col(pl.Float64).round(4)))
    headline_figure(summary)
    example_figures()


if __name__ == "__main__":
    main()
