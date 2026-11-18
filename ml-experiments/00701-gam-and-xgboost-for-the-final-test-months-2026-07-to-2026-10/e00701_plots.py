# ruff: noqa: E501
"""Experiment 701: charts of the provisional scores for the origin months 2026-07 to 2026-10.

    uv run --no-sync python ml-experiments/00701-*/e00701_plots.py [--smoke]

Reads `scores/protocol_final/{rows.parquet,scores.csv}` under this experiment's private folder (from
`e00701_protocol.py`) and writes SVG and PNG charts to `plots/protocol_final/`: the five charts of
experiment 700 for the 4 final-test months (`lanes_*`, `mae_by_day_*`), and `lanes_all_15_12m` and
`mae_by_day_all_12m` for the 12 origin months 2025-11 to 2026-10, whose rows are experiment 700's
`rows.parquet` and the 4 final-test months' together. The Primaries are the 15 that all arms share:
D15 is not in the GAM file, and D18 enters from 2026-01. No generator is plotted.

`--smoke` draws the same charts from `scores/smoke_protocol_final/` (June 2026 from experiment 700's
files, with experiment 700's rows of the origins before June as the "earlier" months), into
`plots/smoke/`, to test the plot code before the model files exist.
"""

import argparse
import textwrap
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import altair as alt
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ROOT: Path = (
    ML_EXPERIMENTS_DATA_DIR / "00701-gam-and-xgboost-for-the-final-test-months-2026-07-to-2026-10"
)
EXPERIMENT_700_ROWS: Path = (
    ML_EXPERIMENTS_DATA_DIR
    / "00700-timing-run-gam-and-xgboost-14-day-monthly-refit"
    / "scores"
    / "protocol_year"
    / "rows.parquet"
)
CAPACITY: Path = (
    ML_EXPERIMENTS_DATA_DIR / "backtest_year_2025-11_to_2026-10" / "series_capacity.parquet"
)

ARMS = {
    "GAM": "GAM",
    "XGBoost BG2 (weekly W member)": "XGBoost",
    "Step 8 monthly (9b)": "Neural network (step 8, monthly refits)",
    "Manual heuristic": "Manual heuristic",
    "Climatology": "Climatology",
}
NAMES = list(ARMS.values())
COLOURS = ["#17E58F", "#306BFF", "#FF4901", "#555555", "#B701FF"]
DASH = [[1, 0], [1, 0], [1, 0], [6, 3], [2, 2]]
LEAD_SLICES = {
    "lead_day_1": "Day 1 (0-24 h)",
    "lead_day_2": "Day 2 (24-48 h)",
    "lead_days_3_4": "Days 3-4 (48-96 h)",
    "lead_days_5_7": "Days 5-7 (96-168 h)",
    "lead_days_8_10": "Days 8-10 (168-240 h)",
}
CONDITION_SLICES = {
    "all": "All days",
    "sunny_unit_shape": "Sunny (unit shape)",
    "winter": "Winter (Dec-Feb)",
    "winter_nights": "Winter nights",
    "cold_nights": "Cold nights (3 °C or less, thin)",
    "economy_7_step": "Economy 7 step hours",
    "christmas": "Christmas (24 Dec-1 Jan)",
    "bank_holidays": "Bank holidays",
}
EMPTY_IN_FINAL_TEST: frozenset[str] = frozenset({"winter", "winter_nights", "christmas"})
"""Condition slices with no half-hour in the origin months 2026-07 to 2026-10."""
PER_SET_SLICE = "sunny_unit_shape"
"""A slice cut per Primary on each experiment's own shared rows, so it cannot pool across experiments."""


def conditions_of(source: Source) -> dict[str, str]:
    """Return the condition slices that the source's rows define alike and that hold rows."""
    drop = {PER_SET_SLICE} if source.suffix == "_12m" else EMPTY_IN_FINAL_TEST
    return {name: label for name, label in CONDITION_SLICES.items() if name not in drop}


METRICS = {"mae_median": "MAE", "crps": "CRPS", "twcrps": "twCRPS"}
GROUPS = {
    "signed-MW six": "the 6 Primaries with signed MW",
    "MVA-only seven": "the 7 MVA-only Primaries",
}
ACCOUNTING = " Points of capacity; smaller is better."


@dataclass(frozen=True)
class Source:
    """Where a chart's rows and scores come from, what it says about them, and where it is saved."""

    rows: list[Path]
    scores_csv: Path
    subtitle: str
    suffix: str
    out: Path


def sources(*, smoke: bool) -> tuple[Source, Source]:
    """Return the 4-month source and the 12-month source (the 4 months and experiment 700's rows)."""
    scores = ROOT / "scores" / ("smoke_protocol_final" if smoke else "protocol_final")
    out = ROOT / "plots" / ("smoke" if smoke else "protocol_final")
    last = "2026-06-30" if smoke else "2026-10-31"
    first = "2026-06-01" if smoke else "2026-07-01"
    months = "1 monthly refit" if smoke else "4 monthly refits"
    four = Source(
        rows=[scores / "rows.parquet"],
        scores_csv=scores / "scores.csv",
        subtitle=(
            f"Provisional: {months}, origins at 12:00 UTC from {first} to {last}, rows that all five "
            "systems share." + ACCOUNTING
        ),
        suffix="",
        out=out,
    )
    twelve = Source(
        rows=[EXPERIMENT_700_ROWS, scores / "rows.parquet"],
        scores_csv=scores / "scores.csv",
        subtitle=(
            f"Provisional: 12 monthly refits, origins at 12:00 UTC from 2025-11-01 to {last}, rows "
            "that all five systems share." + ACCOUNTING
        ),
        suffix="_12m",
        out=out,
    )
    return four, twelve


COLOUR = alt.Color(
    "arm:N",
    title=None,
    scale=alt.Scale(domain=NAMES, range=COLOURS),
    legend=alt.Legend(orient="top", columns=3, labelLimit=400),
)


LANE_STEP = 0.11
ALL_GROUP = "all 15"
GROUP_TEXT = {**GROUPS, ALL_GROUP: "all 15 Primaries with a GAM forecast"}
LANES_LINE_CHARACTERS = 130
BY_DAY_LINE_CHARACTERS = 110


def rows_of(source: Source) -> pl.LazyFrame:
    """Return the per-row losses of the source, the files of the 12-month source one after another.

    In `--smoke` the first file of the 12-month source is limited to the origins before June.
    """
    frames = [pl.scan_parquet(path) for path in source.rows]
    if len(frames) > 1 and source.rows[-1].parent.name.startswith("smoke"):
        frames[0] = frames[0].filter(
            pl.col("power_fcst_init_time") < datetime(2026, 6, 1, tzinfo=UTC)
        )
    return pl.concat(frames)


def pooled_scores(source: Source) -> pl.DataFrame:
    """Return `scores.csv`'s columns for every Primary that all five systems share, pooled.

    D15 is not in the GAM file. D18 has no climatology rows for August to October, so it is pooled
    over its other rows only. The pool holds 15 Primaries. The
    mean is over all their half-hours, so a Primary with more scored rows weighs more, as in `scores.csv`.
    """
    rows = rows_of(source).filter(pl.col("arm").is_in(list(ARMS)))
    parts = []
    for slice_name in [*LEAD_SLICES, *conditions_of(source)]:
        part = (
            rows.filter(pl.col(slice_name))
            .group_by("arm")
            .agg(*[pl.col(m).mean() for m in METRICS], half_hours=pl.len())
            .with_columns(slice=pl.lit(slice_name))
            .collect()
        )
        parts.append(part)
    return (
        pl.concat(parts)
        .unpivot(
            index=["arm", "slice", "half_hours"],
            on=list(METRICS),
            variable_name="metric",
            value_name="mean",
        )
        .with_columns(group=pl.lit(ALL_GROUP))
    )


def dot_chart(
    *, source: Source, group: str, group_text: str, bars: bool = False, lanes: bool = False
) -> alt.VConcatChart:
    """Return the two-section chart (by lead time, by condition) of one group of Primaries.

    Each system is a dot, or with `lanes` a dot at its own height within the row, or with `bars` a
    thin horizontal bar in its own lane. Every second row has a light grey background. Rows sit on
    a numeric axis (row `i` spans `i - 0.5` to `i + 0.5`), so the stripes and the lanes line up.
    """
    row_px = 50 if bars else (40 if lanes else 22)
    spread = bars or lanes
    if group == ALL_GROUP:
        scores = pooled_scores(source)
    else:
        scores = pl.read_csv(source.scores_csv).filter(
            (pl.col("group") == group) & pl.col("arm").is_in(list(ARMS))
        )
    scores = scores.with_columns(arm=pl.col("arm").replace(ARMS))
    sections = []
    for title, slices in (
        ("By forecast horizon (lead time)", LEAD_SLICES),
        ("By condition", conditions_of(source)),
    ):
        names = list(slices.values())
        position = {name: i for i, name in enumerate(names)}
        lane = {arm: (i - (len(NAMES) - 1) / 2) * LANE_STEP for i, arm in enumerate(NAMES)}
        data = scores.filter(pl.col("slice").is_in(list(slices))).with_columns(
            label=pl.col("slice").replace(slices)
        )
        data = data.with_columns(
            row=pl.col("label").replace_strict(position, return_dtype=pl.Int32),
            offset=pl.col("arm").replace_strict(lane, return_dtype=pl.Float64)
            if spread
            else pl.lit(0.0),
        ).with_columns(y=pl.col("row") + pl.col("offset"))
        y_scale = alt.Scale(domain=[len(names) - 0.5, -0.5])
        label_expr = (
            "{" + ", ".join(f"{i}: '{name}'" for i, name in enumerate(names)) + "}[datum.value]"
        )
        shaded = pl.DataFrame({"top": [i - 0.5 for i in range(1, len(names), 2)]}).with_columns(
            bottom=pl.col("top") + 1
        )
        panels = []
        for metric, metric_name in [*METRICS.items(), ("half_hours", "Half-hours")]:
            width = 60 if metric == "half_hours" else 170
            axis = alt.Axis(
                values=list(range(len(names))),
                labelExpr=label_expr,
                labels=metric == "mae_median",
                ticks=False,
                domain=False,
                grid=False,
                title=None,
            )
            stripes = (
                alt.Chart(shaded.to_pandas())
                .mark_rect(color="#EDEDED")
                .encode(  # ty: ignore[unresolved-attribute]
                    y=alt.Y("top:Q", scale=y_scale, axis=axis),
                    y2="bottom:Q",
                    x=alt.value(0),
                    x2=alt.value(width),
                )
            )
            if metric == "half_hours":
                counts = (
                    data.filter((pl.col("metric") == "mae_median") & (pl.col("arm") == "GAM"))
                    .with_columns(
                        text=pl.col("half_hours").map_elements(
                            lambda n: f"{n:,}", return_dtype=pl.String
                        )
                    )
                    .to_pandas()
                )
                front = (
                    alt.Chart(counts)
                    .mark_text(align="left", dx=4)
                    .encode(  # ty: ignore[unresolved-attribute]
                        y=alt.Y("row:Q", scale=y_scale, axis=None), x=alt.value(0), text="text:N"
                    )
                )
            else:
                frame = data.filter(pl.col("metric") == metric).to_pandas()
                if bars:
                    front = (
                        alt.Chart(frame)
                        .mark_rect(height=7)
                        .encode(  # ty: ignore[unresolved-attribute]
                            y=alt.Y("y:Q", scale=y_scale, axis=axis),
                            x=alt.X("zero:Q", title=f"{metric_name} (points of capacity)"),
                            x2="mean:Q",
                            color=COLOUR,
                        )
                        .transform_calculate(zero="0")
                    )
                else:
                    front = (
                        alt.Chart(frame)
                        .mark_circle(size=80, opacity=0.9)
                        .encode(  # ty: ignore[unresolved-attribute]
                            y=alt.Y("y:Q", scale=y_scale, axis=axis),
                            x=alt.X(
                                "mean:Q",
                                title=f"{metric_name} (points of capacity)",
                                scale=alt.Scale(zero=True),
                            ),
                            color=COLOUR,
                        )
                    )
            panels.append(
                alt.layer(stripes, front).properties(
                    width=width, height=row_px * len(names), title=metric_name
                )
            )
        sections.append(
            alt.hconcat(*panels)
            .resolve_scale(x="independent")
            .properties(title=alt.TitleParams(title, anchor="start", fontSize=14))
        )
    return alt.vconcat(*sections).properties(
        title=alt.TitleParams(
            [f"Error of five forecasts at {group_text}.", source.subtitle],
            anchor="start",
            fontSize=12,
        )
    )


def per_day_rows(source: Source) -> pl.DataFrame:
    """Return the per-row errors of the five systems, with the lead day and the capacity unit."""
    units = pl.read_parquet(CAPACITY).select("time_series_id", "power_unit")
    return (
        rows_of(source)
        .filter(pl.col("arm").is_in(list(ARMS)))
        .select(
            "time_series_id", "label", "arm", "power_fcst_init_time", "valid_time", "mae_median"
        )
        .with_columns(
            arm=pl.col("arm").replace(ARMS),
            day=(
                (pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_hours() // 24
            ).cast(pl.Int32)
            + 0.5,
        )
        .collect()
        .join(units.with_columns(pl.col("time_series_id").cast(pl.Int32)), on="time_series_id")
    )


def encode(*, domain: list[float] | None = None) -> dict:
    """Return the shared encoding of the error-by-horizon line charts."""
    return {
        "x": alt.X(
            "day:Q",
            title="Forecast horizon (days after the forecast origin)",
            axis=alt.Axis(values=[0, 2, 4, 6, 8, 10, 12, 14]),
            scale=alt.Scale(domain=[0, 14]),
        ),
        "y": alt.Y(
            "mae:Q",
            title="MAE of the median forecast (points of capacity; smaller is better)",
            scale=alt.Scale(zero=False) if domain is None else alt.Scale(domain=domain),
        ),
        "color": alt.Color(
            "arm:N",
            title=None,
            scale=alt.Scale(domain=NAMES, range=COLOURS),
            legend=alt.Legend(
                orient="top", columns=3, symbolType="stroke", symbolStrokeWidth=3, labelLimit=400
            ),
        ),
        "strokeDash": alt.StrokeDash(
            "arm:N", legend=None, scale=alt.Scale(domain=NAMES, range=DASH)
        ),
    }


def by_day(frame: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    """Return the mean absolute error of the median forecast by horizon day (and `by`)."""
    return (
        frame.group_by(*by, "arm", "day")
        .agg(mae=pl.col("mae_median").mean())
        .sort(*by, "arm", "day")
    )


def save(chart: alt.TopLevelMixin, *, source: Source, name: str) -> None:
    """Write a chart as SVG and as a PNG at twice the size."""
    source.out.mkdir(parents=True, exist_ok=True)
    chart.save(source.out / f"{name}{source.suffix}.svg")
    chart.save(source.out / f"{name}{source.suffix}.png", scale_factor=2)


def lines(text: str, *, width: int) -> list[str]:
    """Return the text wrapped to `width` characters, about the width of the chart's plot area."""
    return textwrap.wrap(text, width=width)


def lanes_chart(*, source: Source, group: str) -> alt.VConcatChart:
    """Return the lanes chart of one group of Primaries."""
    text = GROUP_TEXT[group]
    chart = dot_chart(source=source, group=group, group_text=text, lanes=True)
    return chart.properties(
        title=alt.TitleParams(
            [
                f"Error of five forecasts at {text}.",
                *lines(source.subtitle, width=LANES_LINE_CHARACTERS),
            ],
            anchor="start",
            fontSize=12,
        )
    )


def mae_by_day_charts(*, source: Source, per_primary: bool) -> dict[str, alt.TopLevelMixin]:
    """Return the error-by-horizon chart of all Primaries, and with `per_primary` the small multiples."""
    frame = per_day_rows(source)
    all_title = f"All {frame['label'].n_unique()} Primaries with a GAM forecast"
    charts: dict[str, alt.TopLevelMixin] = {
        "mae_by_day_all": (
            alt.Chart(by_day(frame, []).to_pandas())
            .mark_line(point=True, strokeWidth=2.5, clip=True)
            .encode(**encode())  # ty: ignore[unresolved-attribute]
            .properties(
                width=620,
                height=320,
                title=alt.TitleParams(
                    [
                        all_title + ".",
                        *lines(
                            source.subtitle.replace(ACCOUNTING, ""), width=BY_DAY_LINE_CHARACTERS
                        ),
                    ],
                    anchor="start",
                    fontSize=12,
                ),
            )
        )
    }
    if per_primary:
        per = by_day(frame, ["label", "power_unit"]).with_columns(
            name=pl.col("label") + pl.lit(" (") + pl.col("power_unit") + pl.lit(")")
        )
        order = sorted(
            per["name"].unique(), key=lambda n: (n.split("(")[1], int(n.split(" ")[0][1:]))
        )
        note = (
            "x: forecast horizon in days; y: MAE in points of capacity (own range in each panel). "
            + source.subtitle.replace(ACCOUNTING, "")
        )
        charts["mae_by_day_per_primary"] = (
            alt.Chart(per.to_pandas())
            .mark_line(strokeWidth=1.6, clip=True)
            .encode(  # ty: ignore[unresolved-attribute]
                **{
                    **encode(),
                    "x": alt.X(
                        "day:Q",
                        title=None,
                        axis=alt.Axis(values=[0, 7, 14]),
                        scale=alt.Scale(domain=[0, 14]),
                    ),
                    "y": alt.Y("mae:Q", title=None, scale=alt.Scale(zero=False)),
                },
                facet=alt.Facet("name:N", columns=5, sort=order, title=None),
            )
            .resolve_scale(y="independent")
            .properties(
                width=190,
                height=130,
                title=alt.TitleParams(
                    [
                        "Each Primary on its own: MAE of the median forecast by horizon.",
                        *lines(note, width=BY_DAY_LINE_CHARACTERS),
                    ],
                    anchor="start",
                    fontSize=12,
                ),
            )
        )
    return charts


def main(*, smoke: bool) -> None:
    """Draw the five charts of the 4 months and the two charts of the 12 months."""
    four, twelve = sources(smoke=smoke)
    for group in (ALL_GROUP, *GROUPS):
        stem = f"{group.split()[0].lower()}_{group.split()[1]}"
        save(lanes_chart(source=four, group=group), source=four, name=f"lanes_{stem}")
    for name, chart in mae_by_day_charts(source=four, per_primary=True).items():
        save(chart, source=four, name=name)
    save(lanes_chart(source=twelve, group=ALL_GROUP), source=twelve, name="lanes_all_15")
    for name, chart in mae_by_day_charts(source=twelve, per_primary=False).items():
        save(chart, source=twelve, name=name)
    print(sorted(p.name for p in four.out.iterdir()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke", action="store_true", help="draw June 2026 from experiment 700's files"
    )
    main(smoke=parser.parse_args().smoke)
