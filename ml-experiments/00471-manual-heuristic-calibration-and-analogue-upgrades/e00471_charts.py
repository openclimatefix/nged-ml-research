# ruff: noqa: E501
"""Draw experiment 471's figures as SVG from the score tables and the stored members.

    source /tmp/e471_env.sh
    uv run --no-sync python ml-experiments/00471-*/e00471_charts.py

Optimise afterwards with `npx svgo@4 --multipass --precision=1 --final-newline f.svg`.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import altair as alt
import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00471_common as common  # ty: ignore[unresolved-import]
import e00471_score as score  # ty: ignore[unresolved-import]
import plotting.ocf_theme as ocf
import polars as pl

DOCS: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00471-manual-heuristic-calibration-and-analogue-upgrades"
)
SCORES: Final[Path] = common.WORK_DIR / "scores"
STAGE0: Final[Path] = common.WORK_DIR / "stage0"
WIDTH: Final[int] = 680
LABELS: Final[dict[str, str]] = {
    "E28": "EMOS, 28-day window", "E56": "EMOS, 56-day window (primary)", "E56m13": "EMOS as 13 members",
    "E56m99": "EMOS as 99 members", "MeanLS56": "Mean shift only, 56 days", "Q56": "Recalibrated levels, 56 days",
    "Tau95_56": "Recalibrated q95 only, 56 days", "Lvl7": "Level shift, last 7 days",
    "Lvl14": "Level shift, last 14 days", "Lvl28": "Level shift, last 28 days",
}  # fmt: skip


def contrasts() -> None:
    """Figure 1: CRPS difference from the faithful heuristic with Bonferroni intervals."""
    frame = (
        pl.read_csv(SCORES / "main_contrasts.csv")
        .filter(
            (pl.col("metric") == "crps") & (pl.col("slice") == "all") & (pl.col("lead") == "0-240h")
        )
        .with_columns(variant=pl.col("contrast").str.split(" - ").list.first().replace(LABELS))
    )
    base = alt.Chart(frame).encode(y=alt.Y("variant:N", sort=list(LABELS.values()), title=None))
    chart = (
        (
            base.mark_rule(color=ocf.PALETTE[1]).encode(  # ty: ignore[unresolved-attribute]
                x="low_bonferroni:Q", x2="high_bonferroni:Q"
            )
            + base.mark_point(filled=True, size=60, color=ocf.PALETTE[0]).encode(  # ty: ignore[unresolved-attribute]
                x=alt.X(
                    "difference:Q",
                    title="CRPS minus the faithful heuristic's, points of capacity (below 0 is better)",
                )
            )
            + alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="black").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
        )
        .properties(width=WIDTH - 260, height=200)
        .facet(row=alt.Row("group:N", title=None))
    )
    chart.save(DOCS / "fig1_crps_contrasts.svg")


def rank_histogram() -> None:
    """Figure 2: where the actual falls among the 13 analogues."""
    frame = pl.read_csv(STAGE0 / "rank_histogram.csv").filter(
        (pl.col("members") == 13) & (pl.col("band") == 1)
    )
    long = frame.unpivot(
        on=[f"r{i}" for i in range(14)], index="group", variable_name="rank", value_name="share"
    ).with_columns(rank=pl.col("rank").str.slice(1).cast(pl.Int32))
    chart = (
        alt.Chart(long)  # ty: ignore[unresolved-attribute]
        .mark_bar(color=ocf.PALETTE[1])
        .encode(
            x=alt.X(
                "rank:O",
                title="Number of the 13 analogues below the actual (a flat histogram is 7.1% per bar)",
            ),
            y=alt.Y("share:Q", title="Share of half-hours"),
        )
        .properties(width=WIDTH - 120, height=130)
        .facet(row=alt.Row("group:N", title=None))
    )
    chart.save(DOCS / "fig2_rank_histogram.svg")


def examples() -> None:
    """Figure 3: a day-ahead and a 7-day forecast of one eight-group Primary, faithful against EMOS."""
    series = adapter.series_table()
    sid = int(series.filter(pl.col("label") == "D3")["time_series_id"][0])
    cap = float(series.filter(pl.col("label") == "D3")["capacity_mw"][0])
    origin = datetime(2026, 1, 13, 9, tzinfo=UTC)
    actual = common.cleaned_actuals(series=series).filter(pl.col("time_series_id") == sid)
    frames = {
        "Faithful heuristic": score.faithful_frame(members=common.load_members()),
        "EMOS, 56-day window": score.variant_frame(name="E56"),
    }
    parts = []
    for name, frame in frames.items():
        part = frame.filter(
            (pl.col("time_series_id") == sid) & (pl.col("power_fcst_init_time") == origin)
        )
        parts.append(
            part.with_columns(
                arm=pl.lit(name), hours=(pl.col("valid_time") - origin).dt.total_seconds() / 3600.0
            )
        )
    fc = pl.concat(parts).join(actual, on=["time_series_id", "valid_time"], how="left")
    charts = []
    for title, hi in (("Day ahead", 24.0), ("Seven days ahead", 168.0)):
        sub = fc.filter(pl.col("hours") <= hi)
        band = (
            alt.Chart(sub)  # ty: ignore[unresolved-attribute]
            .mark_area(opacity=0.25)
            .encode(
                x=alt.X("hours:Q", title="Hours after the forecast origin"),
                y=alt.Y("q05:Q", title="Demand, MVA"),
                y2="q95:Q",
                color=alt.Color(
                    "arm:N", scale=alt.Scale(range=[ocf.PALETTE[1], ocf.PALETTE[0]]), title=None
                ),
            )
        )
        line = alt.Chart(sub).mark_line().encode(x="hours:Q", y="q50:Q", color="arm:N")  # ty: ignore[unresolved-attribute]
        obs = (
            alt.Chart(sub.filter(pl.col("actual").is_not_null()))  # ty: ignore[unresolved-attribute]
            .mark_line(color="black", strokeDash=[3, 2])
            .encode(x="hours:Q", y="actual:Q")
        )
        charts.append(
            (band + line + obs).properties(
                width=WIDTH - 90,
                height=160,
                title=f"{title}: D3 (capacity {cap:.1f} MVA), origin 2026-01-13 09:00 UTC; line is the median, band the 5% to 95% range, dashed black the actual",
            )
        )
    alt.vconcat(*charts).save(DOCS / "fig3_example_forecasts.svg")


if __name__ == "__main__":
    contrasts()
    rank_histogram()
    examples()
