# ruff: noqa: E501, ISC004
"""Draw the figures of experiment 422 from the stored contrasts and forecasts (no fit).

- Figure 1: CRPS loss against each network's own fresh case, per stale arm, for the training arms.
- Figure 2: the same loss against the delay of the newest reading, to read the loss at a constant
  hour of day (P6, P30; P12, P36; P24, P48) and off the training grid (P10).
- Figure 3: what the training change cost and gained: each training arm minus T0 in the fresh case.
- Figures 4 to 7: a day-ahead and a 14-day forecast of one Primary in winter and in summer.

```bash
PYTHONPATH=<the experiment folders, as in e00422_score.py> \
uv run --no-sync python ml-experiments/00422-*/e00422_figures.py
```
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import altair as alt
import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_fit_demand as fd  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00410_stale_score as score410  # ty: ignore[unresolved-import]
import e00422_predict as predict422  # ty: ignore[unresolved-import]
import e00422_score as score422  # ty: ignore[unresolved-import]
import polars as pl
from plotting import ocf_theme as theme

OUT_DIR: Final[Path] = predict422.OUT_DIR
FIGURE_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00422-mdn-train-with-varied-origin-hour-and-age-inputs-stale-input-step-2"
)
SIGNED_MVA: Final[tuple[str, ...]] = ("signed-MW six", "MVA-only eight")
ARM_ORDER: Final[list[str]] = [
    "P6",
    "P10",
    "P12",
    "P24",
    "P30",
    "P36",
    "P48",
    "W24",
    "W48",
    "W24_P12",
    "W24_P24",
]
DELAY_HOURS: Final[dict[str, int]] = {
    "P6": 6,
    "P10": 10,
    "P12": 12,
    "P24": 24,
    "P30": 30,
    "P36": 36,
    "P48": 48,
}
SOURCES: Final[list[str]] = ["T0", "T1", "T2", "T2n", "T3", "T4"]
SOURCE_COLOURS: Final[list[str]] = [
    theme.DATA_COLOURS[0],
    theme.DATA_COLOURS[1],
    theme.DATA_COLOURS[2],
    theme.DATA_COLOURS[3],
    theme.DATA_COLOURS[4],
    "#7F7F7F",
]
EXAMPLE_SITE: Final[str] = "D5"
EXAMPLES: Final[dict[str, str]] = {"winter": "2026-01", "summer": "2026-06"}
EXAMPLE_CURVES: Final[tuple[tuple[str, str], ...]] = (("T0", "FRESH"), ("T0", "P12"), ("T3", "P12"))
CURVE_COLOURS: Final[list[str]] = [
    theme.DATA_COLOURS[0],
    theme.DATA_COLOURS[2],
    theme.DATA_COLOURS[4],
]


def contrasts() -> pl.DataFrame:
    """Return the scored contrasts of the three scoring runs, without duplicate T0 rows."""
    parts = [pl.read_parquet(f) for f in sorted((OUT_DIR / "scores").glob("contrasts_*.parquet"))]
    return pl.concat(parts).unique(subset=["contrast", "group"], keep="first")


def figure_losses(*, data: pl.DataFrame) -> None:
    """Figure 1: CRPS loss against the fresh case with 95% intervals, per arm and group."""
    sub = data.filter(
        pl.col("kind") == "C1",
        pl.col("group").is_in(SIGNED_MVA),
        pl.col("source").is_in(SOURCES),
        pl.col("test_arm").is_in(ARM_ORDER),
    ).select("source", "test_arm", "group", "diff", "lower", "upper")
    base = alt.Chart(sub.to_pandas())
    y = alt.Y("test_arm:N", sort=ARM_ORDER, title=None)
    colour = alt.Color(
        "source:N",
        scale=alt.Scale(domain=SOURCES, range=SOURCE_COLOURS),
        legend=alt.Legend(title="Training arm", orient="top"),
    )
    rules = base.mark_rule(strokeWidth=1.5).encode(  # ty: ignore[unresolved-attribute]
        y=y,
        x=alt.X("lower:Q", title="CRPS loss against the fresh case (points)"),
        x2="upper:Q",
        color=colour,
        yOffset="source:N",
    )
    dots = base.mark_point(filled=True, size=30).encode(  # ty: ignore[unresolved-attribute]
        y=y, x="diff:Q", color=colour, yOffset="source:N"
    )
    chart = (
        (rules + dots)
        .properties(width=330, height=330)
        .facet(column=alt.Column("group:N", title=None, sort=list(SIGNED_MVA)))
        .properties(
            title=alt.TitleParams(
                [
                    "CRPS loss at days 1 to 10 when the inputs are stale (interim, five months)",
                    "Dot: estimate. Line: 95% interval from resampling whole months.",
                ],
                anchor="start",
            )
        )
    )
    chart.save(FIGURE_DIR / "figure_1_loss_by_arm.svg")
    chart.save(OUT_DIR / "figure_1_loss_by_arm.png", scale_factor=2)


def figure_delay(*, data: pl.DataFrame) -> None:
    """Figure 2: the loss against the delay of the newest reading."""
    sub = (
        data.filter(
            pl.col("kind") == "C1",
            pl.col("group").is_in(SIGNED_MVA),
            pl.col("source").is_in(SOURCES),
            pl.col("test_arm").is_in(list(DELAY_HOURS)),
        )
        .with_columns(delay=pl.col("test_arm").replace_strict(DELAY_HOURS))
        .select("source", "delay", "group", "diff")
    )
    chart = (
        alt.Chart(sub.to_pandas())
        .mark_line(point=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("delay:Q", title="Delay of the newest reading (hours)"),
            y=alt.Y("diff:Q", title="CRPS loss (points)"),
            color=alt.Color(
                "source:N",
                scale=alt.Scale(domain=SOURCES, range=SOURCE_COLOURS),
                legend=alt.Legend(title="Training arm"),
            ),
        )
        .properties(width=300, height=220)
        .facet(column=alt.Column("group:N", title=None, sort=list(SIGNED_MVA)))
        .properties(
            title=alt.TitleParams(
                [
                    "Loss against the delay of the newest reading (interim)",
                    "Delays 6, 30 land at 23:30; 12, 36 at 17:30; 24, 48 at 05:30. 10 is off the grid.",
                ],
                anchor="start",
            )
        )
    )
    chart.save(FIGURE_DIR / "figure_2_loss_by_delay.svg")
    chart.save(OUT_DIR / "figure_2_loss_by_delay.png", scale_factor=2)


def figure_cost(*, data: pl.DataFrame) -> None:
    """Figure 3: each training arm minus T0 in the fresh case, with 95% intervals."""
    sub = data.filter(pl.col("kind") == "C3", pl.col("group").is_in(SIGNED_MVA)).select(
        "source", "group", "diff", "lower", "upper"
    )
    base = alt.Chart(sub.to_pandas())
    y = alt.Y("source:N", sort=SOURCES[1:], title=None)
    rule = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
        y=y,
        x=alt.X("lower:Q", title="Fresh-case CRPS minus T0's (points; negative is better)"),
        x2="upper:Q",
    )
    dot = base.mark_point(filled=True, size=60).encode(y=y, x="diff:Q")  # ty: ignore[unresolved-attribute]
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}).to_pandas())
        .mark_rule(strokeDash=[3, 3])
        .encode(x="x:Q")
    )  # ty: ignore[unresolved-attribute]
    chart = (
        (rule + dot + zero)
        .properties(width=330, height=170)
        .facet(column=alt.Column("group:N", title=None, sort=list(SIGNED_MVA)))
        .properties(
            title=alt.TitleParams(
                "What the training change cost on fresh inputs (interim)", anchor="start"
            )
        )
    )
    chart.save(FIGURE_DIR / "figure_3_fresh_cost.svg")
    chart.save(OUT_DIR / "figure_3_fresh_cost.png", scale_factor=2)


def actual_points(*, arrays: dict, site: str) -> pl.DataFrame:
    """Return a Primary's readings in points of capacity on the half-hourly grid."""
    return (
        pl.DataFrame({"valid_time": common.grid_times(), "actual": arrays[site].power * 100.0})
        .drop_nans()
        .drop_nulls()
    )


def figure_examples() -> None:
    """Figures 4 to 7: day-ahead and 14-day forecasts of one Primary."""
    _series, arrays = fd._series_and_arrays()
    sid = prep.primaries().filter(pl.col("site") == EXAMPLE_SITE)["time_series_id"][0]
    actual = actual_points(arrays=arrays, site=EXAMPLE_SITE)
    number = 4
    for season, month in EXAMPLES.items():
        origin = datetime.fromisoformat(f"{month}-15").replace(hour=9, tzinfo=UTC)
        frames = {}
        for source, test_arm in EXAMPLE_CURVES:
            stored = score422.seed_average(
                folder=OUT_DIR / "preds" / source / test_arm, months=[month], seeds=score422.SEEDS
            )
            f = score410.widen(frame=stored)
            frames[f"{source} {test_arm}"] = f.filter(
                pl.col("time_series_id") == sid, pl.col("power_fcst_init_time") == origin
            )
        names = list(frames)
        for span, hours in (("day-ahead", 48.0), ("14-day", 336.0)):
            pieces = [
                f.with_columns(
                    lead=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_hours(
                        fractional=True
                    ),
                    arm=pl.lit(name),
                ).filter(pl.col("lead") <= hours)
                for name, f in frames.items()
            ]
            data = (
                pl.concat(pieces)
                .select("arm", "valid_time", "q05", "q25", "q50", "q75", "q95")
                .with_columns(pl.col("q05", "q25", "q50", "q75", "q95") * 100.0)
            )
            lo, hi = data["valid_time"].min(), data["valid_time"].max()
            truth = actual.filter(pl.col("valid_time").is_between(lo, hi))
            colour = alt.Color(
                "arm:N",
                scale=alt.Scale(domain=names, range=CURVE_COLOURS),
                legend=alt.Legend(title=None, orient="top"),
            )
            x = alt.X("valid_time:T", title="Valid time (UTC)")
            pdata = data.to_pandas()
            band90 = (
                alt.Chart(pdata)
                .mark_area(opacity=0.12)
                .encode(  # ty: ignore[unresolved-attribute]
                    x=x,
                    y=alt.Y("q05:Q", title="Power (points of capacity)"),
                    y2="q95:Q",
                    color=colour,
                )
            )
            band50 = (
                alt.Chart(pdata)
                .mark_area(opacity=0.25)
                .encode(  # ty: ignore[unresolved-attribute]
                    x=x, y="q25:Q", y2="q75:Q", color=colour
                )
            )
            line = alt.Chart(pdata).mark_line().encode(x=x, y="q50:Q", color=colour)  # ty: ignore[unresolved-attribute]
            actual_line = (
                alt.Chart(truth.to_pandas())
                .mark_line(color="black", strokeDash=[3, 2])
                .encode(x="valid_time:T", y="actual:Q")  # ty: ignore[unresolved-attribute]
            )
            chart = (band90 + band50 + line + actual_line).properties(
                width=620,
                height=240,
                title=alt.TitleParams(
                    [
                        f"{EXAMPLE_SITE}, {season} example, {span} forecast "
                        f"(origin {origin:%Y-%m-%d %H:%M} UTC)",
                        "Line: median. Dark band: 50%. Light band: 90%. Dashed black: actual readings.",
                    ],
                    anchor="start",
                ),
            )
            name = f"figure_{number}_example_{season}_{span.replace('-', '_')}"
            chart.save(FIGURE_DIR / f"{name}.svg")
            chart.save(OUT_DIR / f"{name}.png", scale_factor=2)
            number += 1


def main() -> None:
    """Draw every figure."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    data = contrasts()
    figure_losses(data=data)
    figure_delay(data=data)
    figure_cost(data=data)
    figure_examples()


if __name__ == "__main__":
    main()
