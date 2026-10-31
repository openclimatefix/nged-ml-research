# ruff: noqa: E501
"""Draw the figures of experiment 442 from the tables `e00442_score.py score` writes.

Reads `work/figure_tables/` (`contrasts`, `decisions`, and `absolute`, all parquet) and, for
the example forecasts, the stored forecasts of experiments 425, 437, and 442. Figures (680 pixels wide,
SVG, in the experiment's docs folder; optimise each with `npx svgo@4 --multipass --precision=1
--final-newline` before committing):

1. `figure_1_<slice>.svg`: CRPS difference of each arm from its control inside one slice, with the
   bootstrap interval, one panel for the signed-MW six and one for the MVA-only eight;
2. `figure_2_outside.svg`: the same difference on all rows outside the two primary slices (the
   no-harm-elsewhere check);
3. `figure_3_folds.svg`: the CRPS difference of T from R on cold winter nights in each fold;
4. `figure_4_absolute.svg`: the absolute CRPS of the controls and references on each slice;
5. `figure_5_tails.svg`: coverage at q01, q05, q95, and q99 of the controls and arms on cold winter
   nights;
6. `figure_6_example_*.svg`: a winter-night week of day-ahead forecasts and the 14-day forecast of
   one run, control against the best arm of the slice, one Primary of each metering group.

```bash
# PYTHONPATH and DATA_PATH_INTERNAL as in e00442_score.py
uv run --no-sync python ml-experiments/00442-*/e00442_figures.py
```
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import altair as alt
import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00407_score as s407  # ty: ignore[unresolved-import]
import e00425_prep as prep425  # ty: ignore[unresolved-import]
import e00425_score as sc  # ty: ignore[unresolved-import]
import e00442_fit as fit  # ty: ignore[unresolved-import]
import e00442_masks as masks_module  # ty: ignore[unresolved-import]
import e00442_score as score  # ty: ignore[unresolved-import]
import polars as pl
from plotting import ocf_theme as theme

DOCS_DIR: Path = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / masks_module.EXPERIMENT_DIR_NAME
)
TABLE_DIR: Path = score.FIGURE_TABLE_DIR
WIDTH: Final[int] = 600
SIX_EIGHT: Final[tuple[str, ...]] = (score.SIX, score.EIGHT)
ARMS: Final[tuple[str, ...]] = score.SCREEN_ARMS
FAMILY_PLAIN: Final[str] = "Arm trained on ensemble rows only"
FAMILY_PRE: Final[str] = "Arm pretrained, then fine-tuned"
FAMILY_CONTROL: Final[str] = "Control (R)"
FAMILY_REFERENCE: Final[str] = "Reference (not a neural network)"
FAMILIES: Final[list[str]] = [FAMILY_CONTROL, FAMILY_PLAIN, FAMILY_PRE, FAMILY_REFERENCE]
FAMILY_COLOURS: Final[list[str]] = ["#1A1A1A", theme.BRAND_ORANGE, theme.DATA_BLUE, "#8A8A8A"]
SLICE_FIGURES: Final[dict[str, str]] = {
    "cold winter nights": "cold_winter_nights",
    "cold sunny days": "cold_sunny_days",
    "cold overcast days": "cold_overcast_days",
    "Christmas": "christmas",
    "other bank holidays": "other_bank_holidays",
}
"""Slice name as drawn, then the file name stem. Heat-wave days hold no rows, so they have no chart."""
EXAMPLE_ARM_COLOURS: Final[list[str]] = [theme.BRAND_ORANGE, theme.DATA_BLUE]
ANONYMOUS: Final[dict[str, str]] = {
    score.SIX: "a signed-MW Primary",
    score.EIGHT: "an MVA-only Primary",
}


def _family(*, arm: str) -> str:
    """Return the colour family of an arm name."""
    if arm == fit.CONTROL:
        return FAMILY_CONTROL
    if arm in ("bg2", "manual_heuristic", "climatology", "S0M"):
        return FAMILY_REFERENCE
    return FAMILY_PRE if arm.endswith("_pre") else FAMILY_PLAIN


def _family_color(*, drop: tuple[str, ...] = ()) -> alt.Color:
    """Return the colour encoding of the arm families, without the families in `drop`."""
    keep = [(f, c) for f, c in zip(FAMILIES, FAMILY_COLOURS, strict=True) if f not in drop]
    return alt.Color(
        "family:N",
        sort=[f for f, _ in keep],
        scale=alt.Scale(domain=[f for f, _ in keep], range=[c for _, c in keep]),
        legend=alt.Legend(orient="top", columns=2, title=None, labelLimit=400),
    )


def _title(*lines: str) -> alt.TitleParams:
    """Return a left-anchored title made of the given lines."""
    return alt.TitleParams(list(lines), anchor="start")


def interval_panels(*, table: pl.DataFrame, groups: tuple[str, ...]) -> alt.VConcatChart:
    """Return one panel per group of arm-minus-control differences with their intervals.

    Args:
        table: Rows with `arm`, `group`, `mean`, `lower`, and `upper`.
        groups: The groups to draw, one panel each.

    Returns:
        The stacked panels.
    """
    order = [f"{a} minus {fit.CONTROL}" for a in ARMS]
    frame = table.with_columns(
        row=pl.format("{} minus {}", pl.col("arm"), pl.col("control")),
        family=pl.col("arm").map_elements(lambda a: _family(arm=a), return_dtype=pl.String),
    )
    panels = []
    for group in groups:
        sub = frame.filter(pl.col("group") == group).drop_nulls("mean")
        if sub.is_empty():
            continue
        base = alt.Chart(sub).encode(
            y=alt.Y("row:N", sort=order, title=None, axis=alt.Axis(labelLimit=200))
        )
        zero = (
            alt.Chart(pl.DataFrame({"x": [0.0]}))
            .mark_rule(color="black", strokeDash=[4, 3])
            .encode(x="x:Q")
        )
        colour = _family_color(drop=(FAMILY_CONTROL, FAMILY_REFERENCE))
        bars = base.mark_rule(strokeWidth=3).encode(
            x=alt.X(
                "lower:Q",
                title="CRPS difference from the control (points of capacity; negative is better)",
            ),
            x2="upper:Q",
            color=colour,
        )
        dots = base.mark_point(filled=True, size=70, opacity=1).encode(x="mean:Q", color=colour)
        panels.append(
            (zero + bars + dots).properties(
                width=WIDTH - 110, height=190, title=alt.TitleParams(group, anchor="start")
            )
        )
    return alt.vconcat(*panels).resolve_scale(color="shared")


def figure_slices() -> None:
    """Draw the per-slice difference charts and the no-harm-elsewhere chart."""
    contrasts = pl.read_parquet(TABLE_DIR / "contrasts.parquet")
    for name, stem in SLICE_FIGURES.items():
        inside = contrasts.filter((pl.col("slice") == name) & (pl.col("side") == "in"))
        if inside.drop_nulls("mean").is_empty():
            continue
        kind = inside["block_kind"].unique().to_list()
        chart = interval_panels(table=inside, groups=SIX_EIGHT).properties(
            title=_title(
                f"CRPS on {name}, T minus its control R.",
                "Days 1 to 10, 6 folds, interim. Dot: estimate.",
                f"Line: 97.5% interval, resampling {', '.join(kind)} blocks.",
            )
        )
        chart.save(DOCS_DIR / f"figure_1_{stem}.svg")
    for name, stem in (
        ("cold winter nights", "cold_winter_nights"),
        ("cold sunny days", "cold_sunny_days"),
    ):
        outside = contrasts.filter((pl.col("slice") == name) & (pl.col("side") == "outside"))
        chart = interval_panels(table=outside, groups=SIX_EIGHT).properties(
            title=_title(
                f"CRPS on all rows outside {name}, arm minus control.",
                "Dot: estimate. Line: 97.5% month-block interval.",
                "The decision rule allows an upper end of +0.02.",
            )
        )
        chart.save(DOCS_DIR / f"figure_2_outside_{stem}.svg")


def figure_folds() -> None:
    """Draw the CRPS difference of T from R on cold winter nights in each fold, for both groups."""
    contrasts = pl.read_parquet(TABLE_DIR / "contrasts.parquet").filter(
        (pl.col("slice") == "cold winter nights")
        & (pl.col("side") == "in")
        & pl.col("group").is_in(list(SIX_EIGHT))
    )
    rows = [
        {"group": r["group"], "fold": fold, "difference": value}
        for r in contrasts.iter_rows(named=True)
        for fold, value in json.loads(r["fold_differences"]).items()
    ]
    chart = (
        alt.Chart(pl.DataFrame(rows))
        .mark_point(filled=True, size=90, color=theme.BRAND_ORANGE)
        .encode(
            x=alt.X(
                "difference:Q",
                title="T minus R, CRPS (points of capacity; negative is better)",
            ),
            y=alt.Y("fold:N", title=None),
            row=alt.Row("group:N", sort=list(SIX_EIGHT), title=None),
        )
        .properties(
            width=WIDTH - 110,
            height=130,
            title=_title(
                "Does T help on cold winter nights in every fold?",
                "Folds with no cold winter nights (2025-10) are absent. Interim.",
            ),
        )
    )
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(strokeDash=[4, 3]).encode(x="x:Q")
    (chart + zero).save(DOCS_DIR / "figure_3_folds.svg")


def figure_absolute() -> None:
    """Draw the absolute CRPS of each control and reference on each slice."""
    table = pl.read_parquet(TABLE_DIR / "absolute.parquet").filter(
        pl.col("forecast").is_in(["R", "T", "bg2", "manual_heuristic", "climatology", "S0M"])
        & pl.col("group").is_in(list(SIX_EIGHT))
        & pl.col("slice").is_in(
            [score.ALL_ROWS, "cold winter nights", "cold sunny days", "Christmas"]
        )
    )
    frame = table.with_columns(
        family=pl.col("forecast").map_elements(lambda a: _family(arm=a), return_dtype=pl.String)
    )
    chart = (
        alt.Chart(frame)
        .mark_point(filled=True, size=80)
        .encode(
            x=alt.X(
                "crps:Q",
                title="CRPS (points of capacity; lower is better)",
                scale=alt.Scale(zero=False),
            ),
            y=alt.Y(
                "forecast:N",
                sort=["R", "T", "S0M", "bg2", "manual_heuristic", "climatology"],
                title=None,
            ),
            color=_family_color(),
            column=alt.Column(
                "slice:N",
                sort=[score.ALL_ROWS, "cold winter nights", "cold sunny days", "Christmas"],
                title=None,
            ),
            row=alt.Row("group:N", sort=list(SIX_EIGHT), title=None),
        )
        .properties(
            width=150,
            height=130,
            title=_title(
                "Absolute CRPS at days 1 to 10 of the controls and the references on each slice (interim)."
            ),
        )
    )
    chart.save(DOCS_DIR / "figure_4_absolute.svg")


def figure_tails() -> None:
    """Draw coverage at q01, q05, q95, and q99 on cold winter nights for the controls and arms."""
    table = pl.read_parquet(TABLE_DIR / "absolute.parquet").filter(
        (pl.col("slice") == "cold winter nights")
        & pl.col("group").is_in(list(SIX_EIGHT))
        & pl.col("forecast").is_in([fit.CONTROL, *ARMS])
    )
    long = table.select(
        "group",
        "forecast",
        *[pl.col(f"coverage_{q}").alias(q) for q in ("q01", "q05", "q95", "q99")],
    ).unpivot(index=["group", "forecast"], variable_name="level", value_name="coverage")
    nominal = pl.DataFrame(
        {"level": ["q01", "q05", "q95", "q99"], "nominal": [0.01, 0.05, 0.95, 0.99]}
    )
    frame = long.join(nominal, on="level").with_columns(
        family=pl.col("forecast").map_elements(lambda a: _family(arm=a), return_dtype=pl.String)
    )
    base = alt.Chart(frame).encode(y=alt.Y("forecast:N", sort=[fit.CONTROL, *ARMS], title=None))
    dots = base.mark_point(filled=True, size=70).encode(
        x=alt.X("coverage:Q", title="Share of actuals at or below the quantile"),
        color=_family_color(drop=(FAMILY_REFERENCE,)),
    )
    marks = base.mark_tick(color="black", thickness=2, size=14).encode(x="nominal:Q")
    chart = (
        (marks + dots)
        .properties(width=130, height=190)
        .facet(
            column=alt.Column("level:N", sort=["q01", "q05", "q95", "q99"], title=None),
            row=alt.Row("group:N", sort=list(SIX_EIGHT), title=None),
        )
        .properties(
            title=_title(
                "Tail coverage on cold winter nights at days 1 to 10. Black tick: nominal level."
            )
        )
    )
    chart.save(DOCS_DIR / "figure_5_tails.svg")


def stored_forecast(*, name: str, months: tuple[str, ...], series: pl.DataFrame) -> pl.DataFrame:
    """Return an arm's stored, seed-averaged forecast widened by 1.15, in MW."""
    if name == fit.CONTROL:
        stored = sc.seed_average(arm=sc.ArmFiles(name, name, fit.SEEDS, months))
    else:
        stored = score.average_seeds(months=months, seeds=fit.SEEDS)
    return adapter.forecast_frame(
        stored=s407.widen_global(frame=stored), series=series, unit="fraction"
    )


def figure_examples() -> None:
    """Draw a winter-night week of day-ahead forecasts and a 14-day forecast, control against best arm."""
    contrasts = pl.read_parquet(TABLE_DIR / "contrasts.parquet")
    masks = pl.read_parquet(masks_module.MASK_PATH)
    series = adapter.series_table()
    ctx = prep425.cleaned_scoring_context(series=series)
    ids = dict(zip(series["label"].to_list(), series["time_series_id"].to_list(), strict=True))
    for group in SIX_EIGHT:
        inside = contrasts.filter(
            (pl.col("slice") == "cold winter nights")
            & (pl.col("side") == "in")
            & (pl.col("group") == group)
        ).drop_nulls("mean")
        best = inside.sort("mean").row(0, named=True)
        best_arm, control = best["arm"], best["control"]
        label = (
            masks.filter(
                pl.col("cold_winter_night") & pl.col("label").is_in(list(score.SUBSETS[group]))
            )
            .group_by("label")
            .len()
            .sort("len", descending=True)["label"][0]
        )
        nights = masks.filter(
            (pl.col("label") == label)
            & pl.col("cold_winter_night")
            & (pl.col("lead_hours") > 24)
            & (pl.col("lead_hours") <= 48)
        )
        week_start = (
            nights.with_columns(week=pl.col("valid_time").dt.truncate("7d"))
            .group_by("week")
            .len()
            .sort("len", descending=True)["week"][0]
        )
        months = tuple(
            sorted({f"{week_start:%Y-%m}", f"{week_start.replace(day=1):%Y-%m}"} & set(fit.MONTHS))
        )
        if not months:
            print(f"example for {group} skipped: week {week_start} lies outside the screen folds")
            continue
        frames = {
            name: stored_forecast(name=name, months=months, series=series)
            for name in dict.fromkeys([control, best_arm])
        }
        actuals = ctx["actuals"].filter(pl.col("time_series_id") == ids[label])
        pieces = []
        for name, frame in frames.items():
            day_ahead = (
                frame.filter(pl.col("time_series_id") == ids[label])
                .with_columns(
                    lead=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes()
                    / 60.0
                )
                .filter((pl.col("lead") > 24) & (pl.col("lead") <= 48))
                .filter(
                    pl.col("valid_time").is_between(week_start, week_start + pl.duration(days=7))
                )
                .join(actuals, on=["time_series_id", "valid_time"], how="left")
                .with_columns(arm=pl.lit(name))
            )
            pieces.append(day_ahead)
        both = pl.concat(pieces, how="diagonal_relaxed").join(
            masks.filter(pl.col("label") == label).select(
                "power_fcst_init_time", "valid_time", "cold_winter_night"
            ),
            on=["power_fcst_init_time", "valid_time"],
            how="left",
        )
        shown = ANONYMOUS[group] if group == score.SIX else f"Primary {label}"
        domain = list(dict.fromkeys([control, best_arm]))
        colour = alt.Color(
            "arm:N",
            sort=domain,
            scale=alt.Scale(domain=domain, range=EXAMPLE_ARM_COLOURS[: len(domain)]),
            legend=alt.Legend(orient="top", columns=2, title=None),
        )
        x = alt.X("valid_time:T", title="Valid time (UTC)")
        base = alt.Chart(both).encode(x=x)
        band = base.mark_area(opacity=0.12, clip=True, aria=False).encode(
            y=alt.Y("q05:Q", title="Power (MW)", scale=alt.Scale(zero=False)),
            y2="q95:Q",
            color=colour,
        )
        median = base.mark_line(strokeWidth=1.6, clip=True, aria=False).encode(
            y="q50:Q", color=colour
        )
        actual_line = (
            alt.Chart(both.filter(pl.col("arm") == control))
            .mark_line(color="black", strokeWidth=1.4, aria=False)
            .encode(x=x, y="actual:Q")
        )
        night = (
            alt.Chart(
                both.filter(
                    (pl.col("arm") == control) & pl.col("cold_winter_night").fill_null(False)
                )
            )
            .mark_tick(color=theme.DATA_PURPLE, size=8, thickness=2, opacity=0.8, aria=False)
            .encode(x=x, y=alt.value(4))
        )
        (band + median + actual_line + night).properties(
            width=WIDTH,
            height=200,
            title=_title(
                f"{shown}, week of {week_start:%d %B %Y}: forecasts made 24 to 48 hours ahead.",
                f"{control} against {best_arm}. Lines: median. Shaded: 5% to 95%.",
                "Black: what happened. Purple ticks: cold winter night half-hours.",
            ),
        ).save(DOCS_DIR / f"figure_6_example_week_{'six' if group == score.SIX else 'eight'}.svg")

        origin = datetime.combine(week_start.date(), datetime.min.time(), tzinfo=UTC).replace(
            hour=9
        )
        long_run = pl.concat(
            [
                frame.filter(
                    (pl.col("time_series_id") == ids[label])
                    & (pl.col("power_fcst_init_time") == origin)
                )
                .join(actuals, on=["time_series_id", "valid_time"], how="left")
                .with_columns(
                    arm=pl.lit(name),
                    day=(pl.col("valid_time") - origin).dt.total_minutes() / 1440.0,
                )
                for name, frame in frames.items()
            ],
            how="diagonal_relaxed",
        )
        xd = alt.X(
            "day:Q",
            title="Days after the forecast was made",
            scale=alt.Scale(domain=[0, 14], nice=False),
        )
        b2 = alt.Chart(long_run).encode(x=xd)
        (
            b2.mark_area(opacity=0.12, clip=True, aria=False).encode(
                y=alt.Y("q05:Q", title="Power (MW)", scale=alt.Scale(zero=False)),
                y2="q95:Q",
                color=colour,
            )
            + b2.mark_line(strokeWidth=1.6, clip=True, aria=False).encode(y="q50:Q", color=colour)
            + alt.Chart(long_run.filter(pl.col("arm") == control))
            .mark_line(color="black", strokeWidth=1.4, aria=False)
            .encode(x=xd, y="actual:Q")
        ).properties(
            width=WIDTH,
            height=200,
            title=_title(
                f"{shown}, 14-day forecast made at 09:00 UTC on {origin:%d %B %Y}.",
                f"{control} against {best_arm}. Lines: median. Shaded: 5% to 95%.",
                "Black: what happened.",
            ),
        ).save(
            DOCS_DIR / f"figure_6_example_14_days_{'six' if group == score.SIX else 'eight'}.svg"
        )


def main() -> None:
    """Draw every figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only", choices=("slices", "folds", "absolute", "tails", "examples"), default=None
    )
    args = parser.parse_args()
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    steps = {
        "slices": figure_slices,
        "folds": figure_folds,
        "absolute": figure_absolute,
        "tails": figure_tails,
        "examples": figure_examples,
    }
    for name, step in steps.items():
        if args.only in (None, name):
            step()


if __name__ == "__main__":
    main()
