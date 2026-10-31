# ruff: noqa: E501
"""Draw the figures of experiment 425 from the tables `e00425_score.py` stores.

Run `e00425_score.py` first: it writes `work/figure_tables/` (CRPS by lead, CRPS per Primary, and
the tail table). The two example-forecast figures read the fitted forecasts directly. Figures (680
pixels wide, SVG, in the experiment's docs folder; optimise each with `npx svgo@4 --multipass
--precision=1 --final-newline` before committing):

1. `figure_1_crps_by_lead.svg`: CRPS by lead for A, R, A-noemb, BG2, S0M, `manual_heuristic`, and
   `climatology`, one panel per group (the signed-MW six and the MVA-only eight);
2. `figure_2_per_primary.svg`: each Primary's CRPS at days 1 to 10 beside R's, as a table of cells;
3. `figure_3_tail_calibration.svg`: coverage at q01, q05, q95, and q99 against the nominal level;
4. `figure_4_example_*.svg`: a day-ahead and a 14-day forecast of one Primary, winter and summer.

```bash
# PYTHONPATH and DATA_PATH_INTERNAL as in e00425_fit.py
uv run --no-sync python ml-experiments/00425-*/e00425_figures.py
```
"""

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import altair as alt
import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00407_score as s407  # ty: ignore[unresolved-import]
import e00425_fit as fit  # ty: ignore[unresolved-import]
import e00425_prep as prep  # ty: ignore[unresolved-import]
import e00425_score as score  # ty: ignore[unresolved-import]
import polars as pl
from plotting import ocf_theme as theme

DOCS_DIR: Path = (
    Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / prep.EXPERIMENT_DIR_NAME
)
TABLE_DIR: Path = fit.OUT_DIR / "figure_tables"
WIDTH: Final[int] = 600
"""Plot-area width; with axes and padding the chart is about 680 pixels wide."""
GROUPS: Final[tuple[str, ...]] = (score.SIX, score.EIGHT)
NAMES: Final[dict[str, str]] = {
    "A_w115": "Pretrained, tuned (A)",
    "R_w115": "ENS only (R)",
    "Anoemb_w115": "Pretrained, no Primary embedding",
    "bg2": "BG2 (weekly XGBoost)",
    "S0M_w115": "S0M (monthly XGBoost, widened)",
    "manual_heuristic": "Manual heuristic",
    "climatology": "Climatology",
}
ORDER: Final[list[str]] = list(NAMES.values())
COLOURS: Final[list[str]] = [
    theme.BRAND_ORANGE,
    "#1A1A1A",
    theme.DATA_PURPLE,
    theme.DATA_BLUE,
    theme.DATA_GREEN,
    "#8A8A8A",
    "#C0C0C0",
]
DASHES: Final[list[list[int]]] = [[1, 0], [1, 0], [1, 0], [6, 3], [6, 3], [2, 2], [2, 2]]
"""Dash pattern per series, a backup to colour: solid for the networks, dashed for references."""
LEAD_HALF_BIN_DAYS: Final[float] = score.LEAD_BIN_HOURS / 48.0
"""Half a bin in days, so each bin is drawn at its centre."""
EXAMPLES: Final[dict[str, str]] = {"D10": "signed-MW Primary", "D4": "MVA-only Primary"}
EXAMPLE_ORIGINS: Final[dict[str, str]] = {"winter": "2026-01-12", "summer": "2026-06-15"}


def _legend() -> alt.Legend:
    """Return the legend shared by the line figures: above the plot, two columns."""
    return alt.Legend(orient="top", columns=2, title=None, labelLimit=400, symbolLimit=10)


def figure_crps_by_lead() -> None:
    """Draw CRPS by lead (6-hour bins, days 0 to 14) for each series, one panel per group."""
    by_lead = pl.read_parquet(TABLE_DIR / "crps_by_lead.parquet")
    pieces = []
    for group in GROUPS:
        members = score.SUBSETS[group]
        pieces.append(
            by_lead.filter(pl.col("label").is_in(list(members)) & pl.col("arm").is_in(list(NAMES)))
            .group_by("arm", "lead_bin_hours")
            .agg(pl.col("crps").mean())
            .with_columns(
                group=pl.lit(group),
                day=pl.col("lead_bin_hours") / 24.0 + LEAD_HALF_BIN_DAYS,
                series=pl.col("arm").replace_strict(NAMES),
            )
        )
    frame = pl.concat(pieces).to_pandas()  # ty: ignore[unresolved-attribute]
    chart = (
        alt.Chart(frame)
        .mark_line(strokeWidth=2)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "day:Q", title="Days after the forecast was made", scale=alt.Scale(domain=[0, 14])
            ),
            y=alt.Y(
                "crps:Q",
                title="CRPS (points of capacity; smaller is better)",
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color(
                "series:N",
                sort=ORDER,
                scale=alt.Scale(domain=ORDER, range=COLOURS),
                legend=_legend(),
            ),
            strokeDash=alt.StrokeDash(
                "series:N", sort=ORDER, scale=alt.Scale(domain=ORDER, range=DASHES), legend=None
            ),
            row=alt.Row("group:N", title=None, sort=list(GROUPS)),
        )
        .properties(
            width=WIDTH,
            height=170,
            title=alt.TitleParams(
                [
                    "CRPS by lead at 12 monthly test folds, equal weight per Primary, mean of 6-hour bins.",
                    "Networks (solid) and references (dashed or dotted), all scored on the same half-hours.",
                ],
                anchor="start",
            ),
        )
    )
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / "figure_1_crps_by_lead.svg")


def figure_per_primary() -> None:
    """Draw each Primary's CRPS at days 1 to 10 as a cell table, coloured by the gap to R."""
    table = pl.read_parquet(TABLE_DIR / "crps_per_primary.parquet").filter(
        pl.col("arm").is_in(list(NAMES))
    )
    reference = table.filter(pl.col("arm") == "R_w115").select("label", reference=pl.col("crps"))
    cells = (
        table.join(reference, on="label")
        .with_columns(
            gap=pl.col("crps") - pl.col("reference"),
            series=pl.col("arm").replace_strict(NAMES),
            text=pl.col("crps").round(2).cast(pl.String),
        )
        .to_pandas()  # ty: ignore[unresolved-attribute]
    )
    labels = sorted(cells["label"].unique(), key=lambda s: int(s[1:]))
    base = alt.Chart(cells).encode(
        x=alt.X("series:N", sort=ORDER, title=None, axis=alt.Axis(labelAngle=-40, labelLimit=260)),
        y=alt.Y("label:N", sort=labels, title="Primary"),
    )
    rect = base.mark_rect().encode(  # ty: ignore[unresolved-attribute]
        color=alt.Color(
            "gap:Q",
            title="CRPS minus R's (points)",
            scale=alt.Scale(scheme="redblue", reverse=True, domain=[-1.5, 1.5], clamp=True),
            legend=alt.Legend(orient="top"),
        )
    )
    text = base.mark_text(fontSize=10).encode(text="text:N", color=alt.value("black"))  # ty: ignore[unresolved-attribute]
    chart = (rect + text).properties(
        width=WIDTH - 60,
        height=330,
        title=alt.TitleParams(
            [
                "CRPS at days 1 to 10 (points of capacity; smaller is better), 12 monthly folds.",
                "Blue cells beat R, red cells trail R. D15 and D18 have under 12 months of history.",
            ],
            anchor="start",
        ),
    )
    chart.save(DOCS_DIR / "figure_2_per_primary.svg")


def figure_tail_calibration() -> None:
    """Draw coverage at q01, q05, q95, and q99 against its nominal level, one panel per group."""
    tails = pl.read_parquet(TABLE_DIR / "tails.parquet")
    wanted = {
        **NAMES,
        "A_raw": "Pretrained, tuned (A), unwidened",
        "R_raw": "ENS only (R), unwidened",
    }
    long = (
        tails.filter(pl.col("arm").is_in(list(wanted)) & pl.col("group").is_in(list(GROUPS)))
        .unpivot(
            on=[f"coverage_{lv}" for lv in score.OUTER_LEVELS],
            index=["arm", "group"],
            variable_name="level",
            value_name="coverage",
        )
        .with_columns(
            level=pl.col("level").str.replace("coverage_", ""),
            series=pl.col("arm").replace_strict(wanted),
        )
        .to_pandas()  # ty: ignore[unresolved-attribute]
    )
    nominal = pl.DataFrame(
        {"level": ["q01", "q05", "q95", "q99"], "nominal": [0.01, 0.05, 0.95, 0.99]}
    ).to_pandas()  # ty: ignore[unresolved-attribute]
    domain = [*ORDER[:1], *ORDER[1:], "Pretrained, tuned (A), unwidened", "ENS only (R), unwidened"]
    colours = [*COLOURS, theme.BRAND_ORANGE_LIGHT, "#6E6E6E"]
    panels = []
    for group in GROUPS:
        points = (
            alt.Chart(long[long["group"] == group])
            .mark_point(filled=True, size=60)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("level:N", sort=["q01", "q05", "q95", "q99"], title="Quantile level"),
                y=alt.Y(
                    "coverage:Q",
                    title="Share of actuals below the quantile",
                    scale=alt.Scale(zero=True),
                ),
                color=alt.Color(
                    "series:N",
                    sort=domain,
                    scale=alt.Scale(domain=domain, range=colours),
                    legend=_legend(),
                ),
                xOffset=alt.XOffset("series:N", sort=domain),
            )
        )
        rule = (
            alt.Chart(nominal)
            .mark_tick(color="black", thickness=2, size=WIDTH / 5)
            .encode(x=alt.X("level:N"), y="nominal:Q")  # ty: ignore[unresolved-attribute]
        )
        panels.append(
            (rule + points).properties(
                width=WIDTH, height=170, title=alt.TitleParams(group, anchor="start")
            )
        )
    chart = alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "Tail calibration at days 1 to 10: dot is the share of actuals below the quantile;",
                "the black tick is the nominal level, so a dot on the tick is calibrated.",
            ],
            anchor="start",
        )
    )
    chart.save(DOCS_DIR / "figure_3_tail_calibration.svg")


def figure_examples() -> None:
    """Draw a day-ahead and a 14-day forecast of A (seed mean, widened 1.15) against the actuals."""
    series = adapter.series_table()
    ctx = prep.cleaned_scoring_context(series=series)
    months = tuple(sorted({o[:7] for o in EXAMPLE_ORIGINS.values()}))
    arm = score.ArmFiles("A", "A", score.SEEDS, months)
    if not arm.complete():
        print(f"example figures skipped: arm A is incomplete for {months}")
        return
    frame = adapter.forecast_frame(
        stored=s407.widen_global(frame=score.seed_average(arm=arm)), series=series, unit="fraction"
    )
    ids = dict(zip(series["label"].to_list(), series["time_series_id"].to_list(), strict=True))
    for season, origin_text in EXAMPLE_ORIGINS.items():
        origin = datetime.fromisoformat(origin_text).replace(hour=9, tzinfo=UTC)
        for label, kind in EXAMPLES.items():
            one = (
                frame.filter(
                    pl.col("time_series_id") == ids[label],
                    pl.col("power_fcst_init_time") == origin,
                )
                .join(
                    ctx["actuals"].filter(pl.col("time_series_id") == ids[label]),
                    on=["time_series_id", "valid_time"],
                    how="left",
                )
                .with_columns(day=(pl.col("valid_time") - origin).dt.total_minutes() / 1440.0)
                .sort("valid_time")
            )
            for horizon, window in (("day-ahead", (1.0, 2.0)), ("14 days", (0.0, 14.0))):
                sub = one.filter(pl.col("day").is_between(*window)).to_pandas()  # ty: ignore[unresolved-attribute]
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
                actual = base.mark_line(color="black", strokeWidth=1.5).encode(y="actual:Q")  # ty: ignore[unresolved-attribute]
                chart = (wide + narrow + median + actual).properties(
                    width=WIDTH,
                    height=200,
                    title=alt.TitleParams(
                        [
                            f"{label} ({kind}), {horizon} forecast made in {season} (A, mean of the seeds, widened 1.15).",
                            "Orange line: median. Bands: 50% and 90%. Black line: what happened.",
                        ],
                        anchor="start",
                    ),
                )
                chart.save(
                    DOCS_DIR
                    / f"figure_4_example_{label}_{season}_{horizon.replace(' ', '_').replace('-', '_')}.svg"
                )


def main() -> None:
    """Draw every figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="read work/smoke, write under it")
    parser.add_argument("--only", choices=("lead", "primary", "tails", "examples"), default=None)
    args = parser.parse_args()
    if args.smoke:
        global DOCS_DIR, TABLE_DIR  # noqa: PLW0603
        fit.OUT_DIR = prep.WORK_DIR / "work" / "smoke"
        TABLE_DIR = fit.OUT_DIR / "figure_tables"
        DOCS_DIR = fit.OUT_DIR / "figures"
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    steps = {
        "lead": figure_crps_by_lead,
        "primary": figure_per_primary,
        "tails": figure_tail_calibration,
        "examples": figure_examples,
    }
    for name, step in steps.items():
        if args.only in (None, name):
            step()


if __name__ == "__main__":
    main()
