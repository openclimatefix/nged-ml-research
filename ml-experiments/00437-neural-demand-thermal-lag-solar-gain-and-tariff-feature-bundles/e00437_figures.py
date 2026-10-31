# ruff: noqa: E501
"""Draw the figures of experiment 437 from the tables `e00437_score.py` stores and its score log.

Run `e00437_score.py` first and keep its output (`--log`): it writes `work/figure_tables/` (CRPS by
lead, CRPS per Primary, and the tail table), and it prints the bootstrap contrasts and the slice
scores that the headline and slice figures read from the log. The example figures read the fitted
forecasts directly. Figures (680 pixels wide, SVG, in the experiment's docs folder; optimise each
with `npx svgo@4 --multipass --precision=1 --final-newline` before committing):

1. `figure_1_headline.svg`: CRPS difference at days 1 to 10, each arm minus its control (`R` for
   `T`, `S`, `C`, `C_tariff`, `F`; `A` for the `_pre` arms), with the month-block bootstrap interval;
2. `figure_2_crps_by_lead.svg`: CRPS by lead for every arm, the controls, and the references, and
   each arm minus its control;
3. `figure_3_per_primary.svg`: each Primary's CRPS at days 1 to 10, as a table of cells;
4. `figure_4_tails_coverage.svg` and `figure_4_tails_pinball.svg`: coverage and pinball loss at
   q01, q05, q95, and q99;
5. `figure_5_slices.svg`: CRPS difference from the control on five slices;
6. `figure_6_example_*.svg`: a day-ahead and a 14-day forecast of one Primary, `R` against `T_pre`
   and `S_pre`, in winter and spring (the four screen folds hold no summer month).

```bash
# PYTHONPATH and DATA_PATH_INTERNAL as in e00437_fit.py
uv run --no-sync python ml-experiments/00437-*/e00437_figures.py --log /tmp/e437/score.log
```
"""

import argparse
import ast
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import altair as alt
import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00407_score as s407  # ty: ignore[unresolved-import]
import e00425_prep as prep425  # ty: ignore[unresolved-import]
import e00425_score as sc  # ty: ignore[unresolved-import]
import e00437_features as feats  # ty: ignore[unresolved-import]
import e00437_fit as fit  # ty: ignore[unresolved-import]
import e00437_score as score  # ty: ignore[unresolved-import]
import polars as pl
from plotting import ocf_theme as theme

DOCS_DIR: Path = (
    Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / feats.EXPERIMENT_DIR_NAME
)
TABLE_DIR: Path = fit.OUT_DIR / "figure_tables"
WIDTH: Final[int] = 600
"""Plot-area width; with axes and padding the chart is about 680 pixels wide."""
SIX_EIGHT: Final[tuple[str, ...]] = (score.SIX, score.EIGHT)
REPORT_ONLY: Final[str] = "D15 and D18"
ALL_GROUPS: Final[tuple[str, ...]] = (score.SIX, score.EIGHT, REPORT_ONLY)
ARMS: Final[tuple[str, ...]] = score.SCREEN_ARMS
"""The eight tested arms, in the order the contrasts are drawn."""
SUFFIX: Final[str] = "_w115"
"""Every stored arm name ends in the widening factor tag."""
FAMILY_CONTROL: Final[str] = "Control (R or A)"
FAMILY_PLAIN: Final[str] = "Arm trained on ensemble rows only"
FAMILY_PRE: Final[str] = "Arm pretrained, then fine-tuned"
FAMILY_REFERENCE: Final[str] = "Reference (not a neural network)"
FAMILIES: Final[list[str]] = [FAMILY_CONTROL, FAMILY_PLAIN, FAMILY_PRE, FAMILY_REFERENCE]
FAMILY_COLOURS: Final[list[str]] = ["#1A1A1A", theme.BRAND_ORANGE, theme.DATA_BLUE, "#8A8A8A"]
CONTROL_NAME: Final[dict[str, str]] = {"R": "R (ENS only)", "A": "A (pretrained)"}
LABELS: Final[dict[str, str]] = {
    "R": "R (ENS only)",
    "T": "T (thermal lag)",
    "S": "S (solar gain)",
    "C": "C (calendar, tariff, national)",
    "C_tariff": "C_tariff (Economy 7 only)",
    "F": "F (look-ahead)",
    "A": "A (pretrained)",
    "T_pre": "T_pre",
    "S_pre": "S_pre",
    "F_pre": "F_pre",
    "S0M": "S0M (monthly XGBoost)",
    "bg2": "BG2 (weekly XGBoost)",
    "manual_heuristic": "Manual heuristic",
    "climatology": "Climatology",
}
ARM_ORDER: Final[list[str]] = [
    "R",
    "T",
    "S",
    "C",
    "C_tariff",
    "F",
    "A",
    "T_pre",
    "S_pre",
    "F_pre",
    "S0M",
    "bg2",
    "manual_heuristic",
    "climatology",
]
EXAMPLE_LABEL: Final[str] = "D10"
"""A signed-MW Primary, as in experiment 425."""
EXAMPLE_ARMS: Final[tuple[str, ...]] = ("T_pre", "S_pre")
EXAMPLE_ORIGINS: Final[dict[str, str]] = {"winter": "2026-01-12", "spring": "2026-03-16"}
EXAMPLE_COLOURS: Final[dict[str, str]] = {
    "R": theme.BRAND_ORANGE,
    "T_pre": theme.DATA_BLUE,
    "S_pre": theme.DATA_PURPLE,
}
SLICES: Final[dict[str, str]] = {
    "cold nights": "cold nights",
    "sunny half-hours": "sunny half-hours",
    "Economy 7 step": "Economy 7 step",
    "cold sunny days": "cold sunny days",
    "cold overcast days": "cold overcast days",
}
"""Slice name as drawn, then the name in the score log."""
CONTRAST_LINE = re.compile(
    r"^(?P<arm>\w+) minus (?P<control>[RA])\s+(?P<group>signed-MW six|MVA-only eight|D15 and D18) ?(?:\(report-only\))?\s+"
    r"(?P<diff>[+-][\d.]+) \((?P<lo>[+-][\d.]+) to (?P<hi>[+-][\d.]+)\)"
)
SLICE_DICT_LINE = re.compile(r"^CRPS on (?P<name>[^,]+), (?P<group>[^:]+): (?P<body>\{.*\})\s*$")
SLICE_HEAD_LINE = re.compile(r"^-- (?P<name>cold (?:sunny|overcast) days), (?P<group>[^:]+): ")
SLICE_ARM_LINE = re.compile(r"^\s+(?P<arm>\w+)\s+(?P<diff>[+-][\d.]+) \(")


def _family(*, arm: str) -> str:
    """Return the colour family of an arm name without the widening tag."""
    if arm in ("R", "A"):
        return FAMILY_CONTROL
    if arm in ("bg2", "manual_heuristic", "climatology", "S0M"):
        return FAMILY_REFERENCE
    return FAMILY_PRE if arm.endswith("_pre") else FAMILY_PLAIN


def _control(*, arm: str) -> str:
    """Return the control of an arm: `A` for a pretrained arm, `R` for any other."""
    return "A" if arm.endswith("_pre") else "R"


def _strip(*, arm: str) -> str:
    """Return the arm name without the widening tag."""
    return arm.removesuffix(SUFFIX)


def _legend() -> alt.Legend:
    """Return the legend shared by the figures: above the plot, one row."""
    return alt.Legend(orient="top", columns=2, title=None, labelLimit=400, symbolLimit=10)


def _family_color(*, drop: tuple[str, ...] = ()) -> alt.Color:
    """Return the colour encoding of the arm families, without the families in `drop`."""
    keep = [(f, c) for f, c in zip(FAMILIES, FAMILY_COLOURS, strict=True) if f not in drop]
    return alt.Color(
        "family:N",
        sort=[f for f, _ in keep],
        scale=alt.Scale(domain=[f for f, _ in keep], range=[c for _, c in keep]),
        legend=_legend(),
    )


def parse_contrasts(*, log: str) -> pl.DataFrame:
    """Return the log's contrasts (arm minus control) with their bootstrap intervals."""
    rows = []
    for line in log.splitlines():
        match = CONTRAST_LINE.match(line)
        if match:
            rows.append(
                {
                    "arm": match["arm"],
                    "control": match["control"],
                    "group": match["group"],
                    "diff": float(match["diff"]),
                    "lo": float(match["lo"]),
                    "hi": float(match["hi"]),
                }
            )
    return pl.DataFrame(rows)


def parse_slices(*, log: str) -> pl.DataFrame:
    """Return each arm's CRPS difference from its control on the five slices, from the log."""
    rows = []
    crps: dict[tuple[str, str], dict[str, float]] = {}
    for line in log.splitlines():
        match = SLICE_DICT_LINE.match(line)
        if match and match["name"] in SLICES.values() and match["group"] in SIX_EIGHT:
            crps[(match["name"], match["group"])] = ast.literal_eval(match["body"])
    for (name, group), values in crps.items():
        rows.extend(
            {
                "slice": name,
                "group": group,
                "arm": arm,
                "diff": values[f"{arm}{SUFFIX}"] - values[f"{_control(arm=arm)}{SUFFIX}"],
            }
            for arm in ARMS
        )
    head: tuple[str, str] | None = None
    for line in log.splitlines():
        match = SLICE_HEAD_LINE.match(line)
        if match:
            head = (match["name"], match["group"])
            continue
        arm_match = SLICE_ARM_LINE.match(line)
        if head is not None and head[1] in SIX_EIGHT and arm_match and arm_match["arm"] in ARMS:
            rows.append(
                {
                    "slice": head[0],
                    "group": head[1],
                    "arm": arm_match["arm"],
                    "diff": float(arm_match["diff"]),
                }
            )
        elif not line.startswith(" "):
            head = None
    return pl.DataFrame(rows)


def _title(*lines: str) -> alt.TitleParams:
    """Return a left-anchored title made of the given lines."""
    return alt.TitleParams(list(lines), anchor="start")


def figure_headline(*, log: str) -> None:
    """Draw each arm's CRPS minus its control at days 1 to 10 with the bootstrap interval."""
    table = parse_contrasts(log=log).filter(pl.col("group").is_in(list(SIX_EIGHT)))
    order = [f"{a} minus {_control(arm=a)}" for a in ARMS]
    frame = table.with_columns(
        row=pl.format("{} minus {}", pl.col("arm"), pl.col("control")),
        family=pl.col("arm").map_elements(lambda a: _family(arm=a), return_dtype=pl.String),
    ).to_pandas()  # ty: ignore[unresolved-attribute]
    panels = []
    for group in SIX_EIGHT:
        sub = frame[frame["group"] == group]
        base = alt.Chart(sub).encode(
            y=alt.Y("row:N", sort=order, title=None, axis=alt.Axis(labelLimit=200))
        )
        zero = (
            alt.Chart(pl.DataFrame({"x": [0.0]}).to_pandas())  # ty: ignore[unresolved-attribute]
            .mark_rule(color="black", strokeDash=[4, 3])
            .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
        )
        bars = base.mark_rule(strokeWidth=3).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "lo:Q",
                title="CRPS difference from the control (points of capacity; negative is better)",
                scale=alt.Scale(domain=[-0.2, 0.25]),
            ),
            x2="hi:Q",
            color=_family_color(drop=(FAMILY_CONTROL, FAMILY_REFERENCE)),
        )
        dots = base.mark_point(filled=True, size=70, opacity=1).encode(  # ty: ignore[unresolved-attribute]
            x="diff:Q", color=_family_color(drop=(FAMILY_CONTROL, FAMILY_REFERENCE))
        )
        panels.append(
            (zero + bars + dots).properties(
                width=WIDTH - 110,
                height=190,
                title=alt.TitleParams(group, anchor="start"),
            )
        )
    chart = (
        alt.vconcat(*panels)
        .resolve_scale(color="shared")
        .properties(
            title=_title(
                "CRPS at days 1 to 10, each arm minus its control (R; A for a pretrained arm), 4 folds.",
                "Dot: estimate. Line: 99.69% interval from resampling whole months (a description).",
                "Dashed line: same as the control. An interval that crosses it is inside the noise.",
            )
        )
    )
    chart.save(DOCS_DIR / "figure_1_headline.svg")


def figure_crps_by_lead() -> None:
    """Draw CRPS by lead (6-hour bins, days 0 to 14), absolute and as a difference from the control."""
    by_lead = pl.read_parquet(TABLE_DIR / "crps_by_lead.parquet").with_columns(
        arm=pl.col("arm").str.replace(f"{SUFFIX}$", "")
    )
    panels = []
    for group in SIX_EIGHT:
        members = list(score.SUBSETS[group])
        mean = (
            by_lead.filter(pl.col("label").is_in(members))
            .with_columns(lead_bin_hours=(pl.col("lead_bin_hours") // 24) * 24)
            .group_by("arm", "lead_bin_hours")
            .agg(pl.col("crps").mean())
            .with_columns(day=pl.col("lead_bin_hours") / 24.0 + 0.5)
        )
        absolute = mean.filter(pl.col("arm").is_in(ARM_ORDER)).with_columns(
            family=pl.col("arm").map_elements(lambda a: _family(arm=a), return_dtype=pl.String),
            series=pl.col("arm").replace_strict(LABELS),
        )
        controls = mean.filter(pl.col("arm").is_in(["R", "A"])).select(
            "lead_bin_hours", control=pl.col("arm"), control_crps=pl.col("crps")
        )
        difference = (
            mean.filter(pl.col("arm").is_in(list(ARMS)))
            .with_columns(
                control=pl.col("arm").map_elements(
                    lambda a: _control(arm=a), return_dtype=pl.String
                )
            )
            .join(controls, on=["control", "lead_bin_hours"])
            .with_columns(
                diff=pl.col("crps") - pl.col("control_crps"),
                family=pl.col("arm").map_elements(lambda a: _family(arm=a), return_dtype=pl.String),
                series=pl.col("arm").replace_strict(LABELS),
            )
        )
        day = alt.X(
            "day:Q", title="Days after the forecast was made", scale=alt.Scale(domain=[0, 14])
        )
        thin = alt.condition(
            alt.FieldOneOfPredicate(field="family", oneOf=[FAMILY_CONTROL, FAMILY_REFERENCE]),
            alt.value(2.5),
            alt.value(1.3),
        )
        line_abs = (
            alt.Chart(absolute.to_pandas())  # ty: ignore[unresolved-attribute]
            .mark_line()
            .encode(
                x=day,
                y=alt.Y(
                    "crps:Q",
                    title="CRPS (points of capacity)",
                    scale=alt.Scale(zero=False),
                ),
                color=_family_color(),
                detail="series:N",
                strokeWidth=thin,
                strokeDash=alt.condition(
                    alt.FieldOneOfPredicate(field="family", oneOf=[FAMILY_REFERENCE]),
                    alt.value([4, 3]),
                    alt.value([1, 0]),
                ),
            )
            .properties(
                width=WIDTH,
                height=150,
                title=alt.TitleParams(f"{group}: CRPS (smaller is better)", anchor="start"),
            )
        )
        line_diff = (
            alt.Chart(difference.to_pandas())  # ty: ignore[unresolved-attribute]
            .mark_line(strokeWidth=1.5)
            .encode(
                x=day,
                y=alt.Y(
                    "diff:Q",
                    title="Arm minus control",
                    scale=alt.Scale(domain=[-0.25, 0.4], clamp=True),
                ),
                color=_family_color(),
                detail="series:N",
            )
        )
        zero = (
            alt.Chart(pl.DataFrame({"y": [0.0]}).to_pandas())  # ty: ignore[unresolved-attribute]
            .mark_rule(color="black", strokeDash=[4, 3])
            .encode(y="y:Q")  # ty: ignore[unresolved-attribute]
        )
        panels.append(line_abs)
        panels.append(
            (zero + line_diff).properties(
                width=WIDTH,
                height=110,
                title=alt.TitleParams(
                    f"{group}: arm minus its control (R, or A for a pretrained arm; negative is better)",
                    anchor="start",
                ),
            )
        )
    chart = alt.vconcat(*panels).properties(
        title=_title(
            "CRPS by lead, 4 folds, equal weight per Primary, daily mean of 6-hour bins,",
            "all on the same half-hours. Black: R and A. Grey dashed: BG2, S0M, manual heuristic,",
            "climatology.",
        )
    )
    chart.save(DOCS_DIR / "figure_2_crps_by_lead.svg")


def figure_per_primary() -> None:
    """Draw each Primary's CRPS at days 1 to 10 as a cell table, coloured by the gap to the control."""
    table = pl.read_parquet(TABLE_DIR / "crps_per_primary.parquet").with_columns(
        arm=pl.col("arm").str.replace(f"{SUFFIX}$", "")
    )
    wide = table.pivot(on="arm", index="label", values="crps")
    cells = []
    for row in wide.iter_rows(named=True):
        for arm in ARM_ORDER:
            control = "A" if arm.endswith("_pre") or arm == "A" else "R"
            if arm == "A":
                control = "R"
            cells.append(
                {
                    "label": row["label"],
                    "series": LABELS[arm],
                    "crps": row[arm],
                    "gap": row[arm] - row[control],
                    "text": f"{row[arm]:.2f}",
                }
            )
    frame = pl.DataFrame(cells).to_pandas()  # ty: ignore[unresolved-attribute]
    labels = sorted(frame["label"].unique(), key=lambda s: int(s[1:]))
    order = [LABELS[a] for a in ARM_ORDER]
    base = alt.Chart(frame).encode(
        x=alt.X("series:N", sort=order, title=None, axis=alt.Axis(labelAngle=-40, labelLimit=260)),
        y=alt.Y("label:N", sort=labels, title="Primary"),
    )
    rect = base.mark_rect().encode(  # ty: ignore[unresolved-attribute]
        color=alt.Color(
            "gap:Q",
            title="CRPS minus its control's (points)",
            scale=alt.Scale(scheme="redblue", reverse=True, domain=[-1.5, 1.5], clamp=True),
            legend=alt.Legend(orient="top"),
        )
    )
    text = base.mark_text(fontSize=9).encode(text="text:N", color=alt.value("black"))  # ty: ignore[unresolved-attribute]
    chart = (rect + text).properties(
        width=WIDTH - 60,
        height=330,
        title=_title(
            "CRPS at days 1 to 10 (points of capacity; smaller is better), 4 folds.",
            "Blue beats the control (R; A for _pre arms), red trails it.",
            "A and the references are compared with R.",
            "D15 and D18 have under 12 months of history.",
        ),
    )
    chart.save(DOCS_DIR / "figure_3_per_primary.svg")


def figure_tails(*, metric: str) -> None:
    """Draw coverage or pinball loss at q01, q05, q95, and q99: one panel per group, one column per level."""
    tails = pl.read_parquet(TABLE_DIR / "tails.parquet").with_columns(
        arm=pl.col("arm").str.replace(f"{SUFFIX}$", "")
    )
    levels = list(sc.OUTER_LEVELS)
    long = tails.unpivot(
        on=[f"{metric}_{lv}" for lv in levels],
        index=["arm", "group"],
        variable_name="level",
        value_name="value",
    ).with_columns(
        level=pl.col("level").str.replace(f"{metric}_", ""),
        family=pl.col("arm").map_elements(lambda a: _family(arm=a), return_dtype=pl.String),
        series=pl.col("arm").replace_strict(LABELS),
    )
    order = [LABELS[a] for a in ARM_ORDER]
    rows = []
    for group in ALL_GROUPS:
        columns = []
        for level in levels:
            sub = long.filter(pl.col("group") == group, pl.col("level") == level)
            first = level == levels[0]
            dots = (
                alt.Chart(sub.to_pandas())  # ty: ignore[unresolved-attribute]
                .mark_point(filled=True, size=40, opacity=1)
                .encode(
                    y=alt.Y(
                        "series:N",
                        sort=order,
                        title=None,
                        axis=alt.Axis(labelLimit=200, labelFontSize=9) if first else None,
                    ),
                    x=alt.X(
                        "value:Q",
                        title=level,
                        scale=alt.Scale(zero=False, nice=True),
                        axis=alt.Axis(
                            labelFontSize=8,
                            tickCount=3,
                            format=".3f" if metric == "coverage" else ".2f",
                        ),
                    ),
                    color=_family_color(),
                )
            )
            layers = [dots]
            if metric == "coverage":
                nominal = float(level.removeprefix("q")) / 100.0
                layers.insert(
                    0,
                    alt.Chart(pl.DataFrame({"nominal": [nominal]}).to_pandas())  # ty: ignore[unresolved-attribute]
                    .mark_rule(color="black", strokeDash=[4, 3])
                    .encode(x="nominal:Q"),  # ty: ignore[unresolved-attribute]
                )
            columns.append(alt.layer(*layers).properties(width=100, height=170))
        rows.append(alt.hconcat(*columns).properties(title=alt.TitleParams(group, anchor="start")))
    if metric == "coverage":
        lines = (
            "Tail calibration at days 1 to 10. Dot: share of actuals below the quantile.",
            "Dashed line: the nominal level, so a dot on it is calibrated. D15 and D18: report-only.",
        )
    else:
        lines = (
            "Pinball loss at q01, q05, q95, and q99, days 1 to 10 (points of capacity; smaller is",
            "better), each panel on its own scale. D15 and D18: report-only.",
        )
    chart = alt.vconcat(*rows).properties(title=_title(*lines))
    chart.save(DOCS_DIR / f"figure_4_tails_{metric}.svg")


def figure_slices(*, log: str) -> None:
    """Draw each arm's CRPS minus its control on the five slices, one panel per slice."""
    table = parse_slices(log=log)
    order = [f"{a} minus {_control(arm=a)}" for a in ARMS]
    frame = table.with_columns(
        row=pl.format(
            "{} minus {}",
            pl.col("arm"),
            pl.col("arm").map_elements(lambda a: _control(arm=a), return_dtype=pl.String),
        )
    ).to_pandas()  # ty: ignore[unresolved-attribute]
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}).to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_rule(color="black", strokeDash=[4, 3])
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    panels = []
    for name in SLICES:
        sub = frame[frame["slice"] == SLICES[name]]
        dots = (
            alt.Chart(sub)
            .mark_point(filled=True, size=60, opacity=1)
            .encode(  # ty: ignore[unresolved-attribute]
                y=alt.Y(
                    "row:N",
                    sort=order,
                    title=None,
                    axis=alt.Axis(labelFontSize=9),
                ),
                x=alt.X(
                    "diff:Q",
                    title="CRPS difference from the control (points; negative is better)",
                    scale=alt.Scale(domain=[-0.7, 0.7], clamp=True),
                    axis=alt.Axis(labelFontSize=8, tickCount=5),
                ),
                color=alt.Color(
                    "group:N",
                    sort=list(SIX_EIGHT),
                    scale=alt.Scale(
                        domain=list(SIX_EIGHT), range=[theme.BRAND_ORANGE, theme.DATA_BLUE]
                    ),
                    legend=_legend(),
                ),
            )
        )
        panels.append(
            (zero + dots).properties(
                width=WIDTH - 110, height=110, title=alt.TitleParams(name, anchor="start")
            )
        )
    chart = alt.vconcat(*panels).properties(
        title=_title(
            "CRPS at days 1 to 10 on five slices, each arm minus its control (R; A for _pre).",
            "Estimates only: the slices hold few half-hours (cold overcast: 769 and 911),",
            "so no interval. Cold: under 5 C. Sunny: clear-sky index above 0.6. Overcast: below 0.3.",
        )
    )
    chart.save(DOCS_DIR / "figure_5_slices.svg")


def figure_examples() -> None:
    """Draw a day-ahead and a 14-day forecast of one Primary: `R` against `T_pre` and `S_pre`."""
    series = adapter.series_table()
    ctx = prep425.cleaned_scoring_context(series=series)
    months = tuple(sorted({o[:7] for o in EXAMPLE_ORIGINS.values()}))
    frames: dict[str, pl.DataFrame] = {}
    r_files = sc.ArmFiles("R", "R", sc.SEEDS_FIRST, months)
    if not r_files.complete():
        print(f"example figures skipped: arm R is incomplete for {months}")
        return
    frames["R"] = adapter.forecast_frame(
        stored=s407.widen_global(frame=sc.seed_average(arm=r_files)), series=series, unit="fraction"
    )
    for arm in EXAMPLE_ARMS:
        stored = fit_average(arm=arm, months=months)
        frames[arm] = adapter.forecast_frame(
            stored=s407.widen_global(frame=stored), series=series, unit="fraction"
        )
    ids = dict(zip(series["label"].to_list(), series["time_series_id"].to_list(), strict=True))
    for season, origin_text in EXAMPLE_ORIGINS.items():
        origin = datetime.fromisoformat(origin_text).replace(hour=9, tzinfo=UTC)
        actuals = ctx["actuals"].filter(pl.col("time_series_id") == ids[EXAMPLE_LABEL])
        pieces = []
        for arm, frame in frames.items():
            pieces.append(
                frame.filter(
                    pl.col("time_series_id") == ids[EXAMPLE_LABEL],
                    pl.col("power_fcst_init_time") == origin,
                )
                .join(actuals, on=["time_series_id", "valid_time"], how="left")
                .with_columns(
                    day=(pl.col("valid_time") - origin).dt.total_minutes() / 1440.0,
                    arm=pl.lit(arm),
                )
                .sort("valid_time")
            )
        both = pl.concat(pieces, how="diagonal_relaxed")
        for horizon, window in (("day-ahead", (1.0, 2.0)), ("14 days", (0.0, 14.0))):
            sub = both.filter(pl.col("day").is_between(*window)).to_pandas()  # ty: ignore[unresolved-attribute]
            domain = list(EXAMPLE_COLOURS)
            colour = alt.Color(
                "arm:N",
                sort=domain,
                scale=alt.Scale(domain=domain, range=list(EXAMPLE_COLOURS.values())),
                legend=_legend(),
            )
            x = alt.X(
                "day:Q",
                title="Days after the forecast was made",
                scale=alt.Scale(domain=list(window), nice=False),
            )
            base = alt.Chart(sub).encode(x=x)
            wide = base.mark_area(opacity=0.12, clip=True, aria=False).encode(  # ty: ignore[unresolved-attribute]
                y=alt.Y("q05:Q", title="Power (MW)", scale=alt.Scale(zero=False)),
                y2="q95:Q",
                color=colour,
            )
            median = base.mark_line(strokeWidth=1.8, clip=True, aria=False).encode(  # ty: ignore[unresolved-attribute]
                y="q50:Q", color=colour
            )  # ty: ignore[unresolved-attribute]
            actual = (
                alt.Chart(sub[sub["arm"] == "R"])
                .mark_line(color="black", strokeWidth=1.5, clip=True, aria=False)
                .encode(x=x, y="actual:Q")  # ty: ignore[unresolved-attribute]
            )
            chart = (wide + median + actual).properties(
                width=WIDTH,
                height=200,
                title=_title(
                    f"{EXAMPLE_LABEL} (signed-MW Primary), {horizon} forecast made in {season}: R, T_pre, and S_pre.",
                    "Coloured lines: median. Shaded: 5% to 95%. Black line: what happened.",
                ),
            )
            chart.save(
                DOCS_DIR
                / f"figure_6_example_{season}_{horizon.replace(' ', '_').replace('-', '_')}.svg"
            )


def fit_average(*, arm: str, months: tuple[str, ...]) -> pl.DataFrame:
    """Return an arm's seed-averaged raw forecast on the given months."""
    return score.average_seeds(arm=arm, months=months, seeds=fit.SEEDS)


def main() -> None:
    """Draw every figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log", type=Path, required=True, help="the printed output of e00437_score.py"
    )
    parser.add_argument(
        "--only",
        choices=("headline", "lead", "primary", "tails", "slices", "examples"),
        default=None,
    )
    args = parser.parse_args()
    log = args.log.read_text()
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    steps = {
        "headline": lambda: figure_headline(log=log),
        "lead": figure_crps_by_lead,
        "primary": figure_per_primary,
        "tails": lambda: [figure_tails(metric="coverage"), figure_tails(metric="pinball")],
        "slices": lambda: figure_slices(log=log),
        "examples": figure_examples,
    }
    for name, step in steps.items():
        if args.only in (None, name):
            step()


if __name__ == "__main__":
    main()
