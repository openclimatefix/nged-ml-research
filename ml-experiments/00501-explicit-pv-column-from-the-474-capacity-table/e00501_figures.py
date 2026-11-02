# ruff: noqa: E501
"""Draw the figures of experiment 501 from the numbers `e00501_score.py` and the capacity step store.

1. `figure_1_contrasts.svg`: the three tested contrasts (column arm minus arm A, CRPS points) with
   their 98.33% intervals, beside the minimum detectable effect of each slice;
2. `figure_2_capacity_by_fold.svg`: the out-of-fold capacity of each Primary at each fold;
3. `figure_3_example_*.svg`: a day-ahead and a 14-day forecast of one Primary, winter and summer.

Optimise each SVG with `npx svgo@4 --multipass --precision=1 --final-newline` before committing.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import altair as alt
import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00425_prep as prep425  # ty: ignore[unresolved-import]
import e00501_capacity as capacity
import e00501_fit as fit501
import e00501_score as score
import polars as pl
from plotting import ocf_theme as theme

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00501-explicit-pv-column-from-the-474-capacity-table"
)
WIDTH: Final[int] = 560
EXAMPLES: Final[dict[str, str]] = {"winter": "2026-01-12", "summer": "2026-06-15"}
EXAMPLE_PRIMARY: Final[str] = "D10"
MDE_KEY: Final[dict[str, str]] = {
    "C1 sunny-midday half-hours": "sunny_midday",
    "C2 all daylight half-hours": "daylight",
    "C3 all half-hours": "all",
}


def figure_contrasts() -> None:
    """Draw the contrast intervals and the minimum detectable effect per slice."""
    summary = json.loads((fit501.WORK_DIR / "score_summary.json").read_text())
    rows = []
    for key, value in summary.items():
        if " | " not in key:
            continue
        slice_name, group = key.split(" | ")
        rows.append(
            {
                "slice": slice_name,
                "group": group,
                "difference": value["difference"],
                "lower": value["lower"],
                "upper": value["upper"],
                "mde": -summary["mde"][MDE_KEY[slice_name]]["mde_3_seeds"],
            }
        )
    frame = pl.DataFrame(rows).to_pandas()  # ty: ignore[unresolved-attribute]
    base = alt.Chart(frame).encode(y=alt.Y("group:N", title=None, sort=None))  # ty: ignore[unresolved-attribute]
    bars = base.mark_rule(strokeWidth=3, color=theme.BRAND_ORANGE).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("lower:Q", title="Column arm minus arm A, CRPS (points; below zero is better)"),
        x2="upper:Q",
    )
    points = base.mark_point(filled=True, size=70, color="black").encode(x="difference:Q")  # ty: ignore[unresolved-attribute]
    limit = base.mark_tick(color=theme.DATA_BLUE, thickness=2, size=18).encode(x="mde:Q")  # ty: ignore[unresolved-attribute]
    chart = (
        (bars + points + limit)
        .properties(width=WIDTH, height=90)
        .facet(
            row=alt.Row(
                "slice:N",
                title=None,
                sort=list(MDE_KEY),
                header=alt.Header(labelAngle=0, labelAlign="left", labelPadding=4),
            )
        )
        .properties(
            title=alt.TitleParams(
                [
                    "Orange bar: 98.33% interval. Black dot: estimate.",
                    "Blue tick: the gain the experiment could detect (minus the minimum detectable effect).",
                ],
                anchor="start",
            )
        )
    )
    chart.save(DOCS_DIR / "figure_1_contrasts.svg")


def figure_capacity() -> None:
    """Draw each Primary's out-of-fold capacity by fold."""
    frame = pl.concat(
        [pl.read_parquet(p) for p in sorted(fit501.WORK_DIR.glob("capacity_by_fold_*.parquet"))]
    ).to_pandas()  # ty: ignore[unresolved-attribute]
    chart = (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True)
        .encode(
            x=alt.X("fold:N", title="Test month (capacity fitted on all months before it)"),
            y=alt.Y("ac_mw:Q", title="AC capacity (MW)", scale=alt.Scale(zero=False)),
            color=alt.Color("site:N", title="Primary"),
        )
        .properties(width=WIDTH, height=220)
    )
    chart.save(DOCS_DIR / "figure_2_capacity_by_fold.svg")


def figure_examples() -> None:
    """Draw a day-ahead and a 14-day forecast of the column arm, with arm A's median."""
    series = adapter.series_table()
    ctx = prep425.cleaned_scoring_context(series=series)
    frames = score.seed_frames(series=series, seeds=score.SEEDS)
    ids = dict(zip(series["label"].to_list(), series["time_series_id"].to_list(), strict=True))
    tid = ids[EXAMPLE_PRIMARY]
    for season, origin_text in EXAMPLES.items():
        origin = datetime.fromisoformat(origin_text).replace(hour=9, tzinfo=UTC)
        one = {
            name: f.filter(
                pl.col("time_series_id") == tid, pl.col("power_fcst_init_time") == origin
            )
            for name, f in frames.items()
        }
        joined = (
            one["APV_w115"]
            .join(
                one["A_w115"].select("valid_time", a_median=pl.col("q50")),
                on="valid_time",
            )
            .join(
                ctx["actuals"].filter(pl.col("time_series_id") == tid),
                on=["time_series_id", "valid_time"],
                how="left",
            )
            .with_columns(day=(pl.col("valid_time") - origin).dt.total_minutes() / 1440.0)
            .sort("valid_time")
        )
        for horizon, window in (("day-ahead", (1.0, 2.0)), ("14 days", (0.0, 14.0))):
            sub = joined.filter(pl.col("day").is_between(*window)).to_pandas()  # ty: ignore[unresolved-attribute]
            x = alt.X(
                "day:Q",
                title="Days after the forecast was made",
                scale=alt.Scale(domain=list(window), nice=False),
            )
            base = alt.Chart(sub).encode(x=x)  # ty: ignore[unresolved-attribute]
            wide = base.mark_area(opacity=0.2, color=theme.BRAND_ORANGE).encode(  # ty: ignore[unresolved-attribute]
                y=alt.Y("q05:Q", title="Power (MW)", scale=alt.Scale(zero=False)), y2="q95:Q"
            )
            narrow = base.mark_area(opacity=0.35, color=theme.BRAND_ORANGE).encode(  # ty: ignore[unresolved-attribute]
                y="q25:Q", y2="q75:Q"
            )
            median = base.mark_line(color=theme.BRAND_ORANGE, strokeWidth=2).encode(y="q50:Q")  # ty: ignore[unresolved-attribute]
            a_median = base.mark_line(color=theme.DATA_BLUE, strokeDash=[4, 3]).encode(  # ty: ignore[unresolved-attribute]
                y="a_median:Q"
            )
            actual = base.mark_line(color="black", strokeWidth=1.5).encode(y="actual:Q")  # ty: ignore[unresolved-attribute]
            chart = (wide + narrow + median + a_median + actual).properties(
                width=WIDTH,
                height=200,
                title=alt.TitleParams(
                    [
                        f"{EXAMPLE_PRIMARY}, {horizon} forecast made in {season} (column arm, mean of 3 seeds, widened 1.15).",
                        "Orange: median and 50% and 90% bands. Dashed blue: arm A's median. Black: what happened.",
                    ],
                    anchor="start",
                ),
            )
            name = f"figure_3_example_{season}_{horizon.replace(' ', '_').replace('-', '_')}.svg"
            chart.save(DOCS_DIR / name)


def main() -> None:
    """Draw every figure."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    figure_contrasts()
    figure_capacity()
    figure_examples()
    print(f"figures written to {DOCS_DIR}; capacity folds {capacity.FOLDS}")


if __name__ == "__main__":
    main()
