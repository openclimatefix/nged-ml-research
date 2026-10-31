# ruff: noqa: E501
r"""Draw experiment 392's sweep and headline figures and print every number plotted.

Writes four SVGs to the experiment's docs folder and runs `svgo` over each:

1. `figure_1_headline.svg`: pooled CRPS by arm, for three row sets (outside events and the 48 hours
   after them, inside events, the 7 days after events), raw widening, lead 0 to 240 hours, all 16
   Primaries, all four folds. The stored CRPS is already in points of effective capacity.
2. `figure_2_sweep_median.svg`: median shift as a share of the step against encroachment level k.
3. `figure_3_sweep_band.svg`: ratio of the 5%-95% band width to its width at k = 0, and the
   coverage of the event-level reading, against k.
4. `figure_4_screen_folds.svg`: CRPS outside events, arm minus R0 of the same form, per fold.

The sweep figures use `share` 0.05 and the raw (unwidened) bands. The median share of the step is
`sum(shift_sum * sign) / share / sum(n)` (both signs, oriented by the sign of the step); ratios
and coverages are ratios of sums over Primaries and refits, never averages of ratios. The lagless
forms ignore k, so their lines are flat.

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\
ml-experiments/00379-bg2-components-all-16-primaries-to-14-days:\
ml-experiments/00383-load-shift-xgboost-oracle-first-part-1:\
ml-experiments/00392-xgboost-nra-load-shift-masked-targets \
    DATA_PATH_INTERNAL=... OMP_NUM_THREADS=1 uv run --no-sync python \
    ml-experiments/00392-*/e00392_figures.py
```
"""

import subprocess
from pathlib import Path
from typing import Final

import altair as alt
import e00392_score as score  # ty: ignore[unresolved-import]
import plotting.ocf_theme as ocf
import polars as pl
from studies.charts import CONTENT_WIDTH_PX, wrapped

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00392-xgboost-nra-load-shift-masked-targets"
)
SWEEP_PATH: Final[Path] = score.WORK / "sweep" / "sweep.parquet"
ROWS_PATH: Final[Path] = score.OUT / "rows.parquet"
SWEEP_SHARE: Final[float] = 0.05
LEAD_BAND_NAMES: Final[list[str]] = [name for name, _, _ in score.LEAD_BANDS]
ARM_NAMES: Final[dict[str, str]] = {"r0": "R0", "r1": "R1", "rc1": "RC1", "r5": "R5"}
ARM_COLOURS: Final[dict[str, str]] = {
    "R0": ocf.DATA_BLUE,
    "R1": ocf.BRAND_ORANGE,
    "RC1": ocf.DATA_GREEN,
    "R5": ocf.DATA_PURPLE,
}
FORM_DASH: Final[dict[str, list[int]]] = {"lagged": [1, 0], "lagless": [5, 3]}
HEADLINE_ARMS: Final[tuple[str, ...]] = (
    "r0_lagged",
    "r1_lagged",
    "rc1_lagged",
    "r5_lagged",
    "r0_lagless",
    "r1_lagless",
    "rc1_lagless",
)
ROW_SET_TITLES: Final[dict[str, str]] = {
    "outside": "Outside events and the 48 hours after them",
    "during": "Inside events",
    "after7d": "In the 7 days after events",
}
SWEEP_GROUPS: Final[tuple[tuple[str, str], ...]] = (
    ("signed-MW six", "signed-MW six"),
    ("MVA-only eight", "MVA-only eight"),
)
FULL_WIDTH_PX: Final[int] = CONTENT_WIDTH_PX - 70
PANEL_WIDTH_PX: Final[int] = 125
BODY_FONT: Final[int] = ocf.font_size(style="Body", body_px=11)


def arm_label(*, arm_form: str) -> str:
    """Return `R0, lagged` style text for an `r0_lagged` style key."""
    arm, form = arm_form.split("_")
    return f"{ARM_NAMES[arm]}, {form}"


def caption(*, title: str, subtitle: str) -> alt.TitleParams:
    """Return a wrapped figure title with its subtitle."""
    return alt.TitleParams(
        wrapped(text=title, width=95),
        subtitle=wrapped(text=subtitle, width=105),
        anchor="start",
        offset=12,
        subtitleColor=ocf.BLACK_1,
    )


def save(*, chart: alt.TopLevelMixin, path: Path) -> None:
    """Write the chart as SVG and optimise it with svgo."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(path)
    subprocess.run(
        ["npx", "svgo@4", "--multipass", "--precision=1", "--final-newline", str(path)],
        check=True,
        capture_output=True,
    )
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")


def show(*, name: str, frame: pl.DataFrame) -> None:
    """Print a plotted table in full."""
    with pl.Config(tbl_rows=500, tbl_cols=30, tbl_width_chars=200, float_precision=4):
        print(f"\n== {name} ==")
        print(frame)


# ---------------------------------------------------------------------------------------------
# Figure 1
# ---------------------------------------------------------------------------------------------


def headline_data() -> pl.DataFrame:
    """Return pooled CRPS per arm-and-form and row set (raw widening, lead up to 240 hours)."""
    rows = pl.read_parquet(ROWS_PATH).filter(
        pl.col("widening") == 1.0, pl.col("lead_hours") <= score.HEADLINE_LEAD_HOURS
    )
    parts = []
    for set_name in ROW_SET_TITLES:
        part = (
            rows.filter(score.ROW_SETS[set_name])
            .group_by("arm")
            .agg(pl.len().alias("n"), pl.col("crps").mean())
            .with_columns(row_set=pl.lit(set_name))
        )
        parts.append(part)
    out = (
        pl.concat(parts)
        .filter(pl.col("arm").is_in(HEADLINE_ARMS))
        .with_columns(
            label=pl.col("arm").map_elements(
                lambda a: arm_label(arm_form=a), return_dtype=pl.String
            ),
            arm_name=pl.col("arm").map_elements(
                lambda a: ARM_NAMES[a.split("_")[0]], return_dtype=pl.String
            ),
            panel=pl.col("row_set").replace(ROW_SET_TITLES),
        )
    )
    show(
        name="Figure 1: pooled CRPS (points of capacity), lead 0-240 h, raw",
        frame=out.sort("row_set", "arm"),
    )
    return out


def figure_1(*, data: pl.DataFrame) -> alt.VConcatChart:
    """Return the headline figure: bars of pooled CRPS in three stacked panels."""
    order = [arm_label(arm_form=a) for a in HEADLINE_ARMS]
    panels = []
    for set_name, title in ROW_SET_TITLES.items():
        part = data.filter(pl.col("row_set") == set_name)
        base = alt.Chart(part)
        bars = base.mark_bar(aria=False).encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y(
                "label:N",
                sort=order,
                title=None,
                axis=alt.Axis(labelLimit=200),
            ),
            x=alt.X("crps:Q", title="Pooled CRPS (points of capacity; smaller is better)"),
            color=alt.Color(
                "arm_name:N",
                scale=alt.Scale(domain=list(ARM_COLOURS), range=list(ARM_COLOURS.values())),
                legend=alt.Legend(title="Arm", orient="top", direction="horizontal"),
            ),
        )
        labels = base.mark_text(align="left", dx=3, fontSize=BODY_FONT, aria=False).encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("label:N", sort=order),
            x="crps:Q",
            text=alt.Text("crps:Q", format=".3f"),
        )
        panels.append(
            alt.layer(bars, labels, width=FULL_WIDTH_PX, height=120).properties(
                title=alt.TitleParams(title, anchor="start", fontSize=BODY_FONT)
            )
        )
    subtitle = (
        "Interim, four folds (2025-10, 2026-01, 2026-03, 2026-06). Each bar is the mean CRPS of the "
        "arm over all 16 Primaries' half-hours at leads of 0 to 240 hours (days 1 to 10), with the "
        "quantile bands as stored (raw, not widened). R0: no training rows dropped; R1: rows in and "
        "just after hand-labelled load-shift events dropped from training; RC1: as many rows "
        "dropped in random blocks (control); R5: experiment 391's lagged arm."
    )
    return (
        alt.vconcat(*panels, spacing=20)
        .resolve_scale(x="independent", color="shared")
        .properties(
            title=caption(
                title="Pooled CRPS by arm, outside, inside, and after load-shift events",
                subtitle=subtitle,
            )
        )
        .configure_view(stroke=None)
    )


# ---------------------------------------------------------------------------------------------
# Figures 2 and 3
# ---------------------------------------------------------------------------------------------


def sweep_frame() -> pl.DataFrame:
    """Return the sweep rows for the plotted share, raw widening, with arm and form columns."""
    return (
        pl.read_parquet(SWEEP_PATH)
        .filter(pl.col("share") == SWEEP_SHARE, ~pl.col("widened"))
        .with_columns(
            arm_name=pl.col("arm_form").map_elements(
                lambda a: ARM_NAMES[a.split("_")[0]], return_dtype=pl.String
            ),
            form=pl.col("arm_form").str.split("_").list.get(1),
        )
        .filter(pl.col("arm_name").is_in(["R0", "R1", "RC1"]))
    )


def aggregate(*, frame: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    """Sum the sweep sums over Primaries, refits and signs; return the three plotted ratios."""
    return (
        frame.group_by(by)
        .agg(
            pl.col("n").sum(),
            (pl.col("shift_sum") * pl.col("sign")).sum().alias("oriented_shift_sum"),
            pl.col("width5_sum").sum(),
            pl.col("width5_base_sum").sum(),
            pl.col("cover_event_sum").sum(),
        )
        .with_columns(
            median_share=pl.col("oriented_shift_sum") / SWEEP_SHARE / pl.col("n"),
            band_ratio=pl.col("width5_sum") / pl.col("width5_base_sum"),
            event_coverage=pl.col("cover_event_sum") / pl.col("n"),
        )
        .sort(by)
    )


def line_grid(
    *,
    data: pl.DataFrame,
    value: str,
    y_title: str,
    row_filter: list[tuple[str, pl.Expr]],
    height: int,
) -> alt.VConcatChart:
    """Return rows of lead-band panels with one line per arm and form."""
    rows = []
    for row_title, expr in row_filter:
        part = data.filter(expr)
        base = alt.Chart(part)
        lines = base.mark_line(point=True, strokeWidth=1.8, aria=False).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "k:Q",
                title="Lags shifted (k)",
                axis=alt.Axis(values=[0, 1, 2, 3]),
                scale=alt.Scale(domain=[0, 3]),
            ),
            y=alt.Y(f"{value}:Q", title=y_title, scale=alt.Scale(zero=False)),
            color=alt.Color(
                "arm_name:N",
                scale=alt.Scale(
                    domain=["R0", "R1", "RC1"], range=[ARM_COLOURS[a] for a in ("R0", "R1", "RC1")]
                ),
                legend=alt.Legend(title="Arm", orient="top", direction="horizontal"),
            ),
            strokeDash=alt.StrokeDash(
                "form:N",
                scale=alt.Scale(
                    domain=["lagged", "lagless"], range=[FORM_DASH["lagged"], FORM_DASH["lagless"]]
                ),
                legend=alt.Legend(title="Form", orient="top", direction="horizontal"),
            ),
        )
        rows.append(
            lines.properties(width=PANEL_WIDTH_PX, height=height)
            .facet(
                column=alt.Column(
                    "lead_band:N",
                    sort=LEAD_BAND_NAMES,
                    title=row_title,
                    header=alt.Header(titleOrient="top", labelOrient="top", titleAnchor="start"),
                )
            )
            .resolve_scale(y="shared")
        )
    return alt.vconcat(*rows, spacing=14)


def figure_2(*, frame: pl.DataFrame) -> alt.VConcatChart:
    """Return the median-shift figure, one row per metering group."""
    data = aggregate(frame=frame, by=["group", "lead_band", "arm_form", "arm_name", "form", "k"])
    show(
        name="Figure 2: median shift as a share of the step (share 0.05, raw)",
        frame=data.select("group", "lead_band", "arm_form", "k", "n", "median_share"),
    )
    chart = line_grid(
        data=data,
        value="median_share",
        y_title="Median shift / step",
        row_filter=[(f"Primaries: {g}", pl.col("group") == g) for g, _ in SWEEP_GROUPS],
        height=120,
    )
    subtitle = (
        "Interim, four folds. A step of 5% of effective capacity is added to the power lags of "
        "clean forecast rows, k of them: k=1 the origin lag, k=2 also the same-time-yesterday lag, k=3 also the same-time-last-week lag (steps up and down both counted, oriented so a forecast that follows "
        "the step scores +1; 0: ignores it). One column per lead band. Solid: lagged form; dashed: "
        "lagless form, which has no lags and so stays flat at 0."
    )
    return chart.properties(
        title=caption(
            title="How far the forecast median follows a step in the lags",
            subtitle=subtitle,
        )
    ).configure_view(stroke=None)


def figure_3(*, frame: pl.DataFrame) -> alt.VConcatChart:
    """Return the band-width ratio and event-level coverage figure, pooled over groups."""
    data = aggregate(frame=frame, by=["lead_band", "arm_form", "arm_name", "form", "k"])
    show(
        name="Figure 3: band-width ratio and event-level coverage (share 0.05, raw, all groups)",
        frame=data.select("lead_band", "arm_form", "k", "n", "band_ratio", "event_coverage"),
    )
    ratio = line_grid(
        data=data,
        value="band_ratio",
        y_title="Width / width at k=0",
        row_filter=[
            ("5%-95% band width relative to its width with no lag shifted", pl.lit(value=True))
        ],
        height=120,
    )
    coverage = line_grid(
        data=data,
        value="event_coverage",
        y_title="Share of rows covered",
        row_filter=[("Share of event-level readings inside the 5%-95% band", pl.lit(value=True))],
        height=120,
    )
    subtitle = (
        "Interim, four folds. A step of 5% of effective capacity is added to k of the power lags (k=1 origin, k=2 also yesterday's, k=3 also last week's) of "
        "clean forecast rows, pooled over all Primaries (sums over Primaries and refits). The "
        "event-level reading is the actual value plus the step. Solid: lagged form; dashed: "
        "lagless form, which has no lags and so stays flat."
    )
    return (
        alt.vconcat(ratio, coverage, spacing=14)
        .properties(
            title=caption(
                title="Whether the forecast band widens, and covers the event level, as lags shift",
                subtitle=subtitle,
            )
        )
        .configure_view(stroke=None)
    )


# ---------------------------------------------------------------------------------------------
# Figure 4
# ---------------------------------------------------------------------------------------------


def screen_data() -> pl.DataFrame:
    """Return CRPS outside events per fold, as arm minus R0 of the same form."""
    rows = pl.read_parquet(ROWS_PATH).filter(
        pl.col("widening") == 1.0,
        pl.col("lead_hours") <= score.HEADLINE_LEAD_HOURS,
        score.ROW_SETS["outside"],
    )
    per = (
        rows.group_by("arm", "fold")
        .agg(pl.len().alias("n"), pl.col("crps").mean())
        .with_columns(
            form=pl.col("arm").str.split("_").list.get(1),
            arm_name=pl.col("arm").str.split("_").list.get(0).str.to_uppercase(),
        )
    )
    control = per.filter(pl.col("arm_name") == "R0").select("form", "fold", control=pl.col("crps"))
    out = (
        per.filter(pl.col("arm_name").is_in(["R1", "RC1"]))
        .join(control, on=["form", "fold"])
        .with_columns(
            diff=pl.col("crps") - pl.col("control"),
            fold_name=pl.col("fold").map_elements(
                lambda f: score.FOLD_NAMES[f], return_dtype=pl.String
            ),
        )
        .sort("form", "arm_name", "fold")
    )
    show(name="Figure 4: CRPS outside events, arm minus R0 (points of capacity)", frame=out)
    return out


def figure_4(*, data: pl.DataFrame) -> alt.VConcatChart:
    """Return the per-fold contrast figure, one panel per form."""
    panels = []
    rule_frame = pl.DataFrame({"x": [score.CRPS_TOLERANCE], "label": ["+0.02 rule"]})
    for form in ("lagged", "lagless"):
        part = data.filter(pl.col("form") == form)
        base = alt.Chart(part)
        points = base.mark_point(filled=True, size=70, opacity=1, aria=False).encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("fold_name:N", sort=list(score.FOLD_NAMES), title="Fold (refit month)"),
            x=alt.X(
                "diff:Q",
                title="CRPS minus R0's (points of capacity; more negative is better)",
                scale=alt.Scale(domain=[-0.3, 0.15]),
                axis=alt.Axis(values=[-0.3, -0.2, -0.1, 0.0, 0.1]),
            ),
            yOffset="arm_name:N",
            color=alt.Color(
                "arm_name:N",
                scale=alt.Scale(
                    domain=["R1", "RC1"], range=[ARM_COLOURS["R1"], ARM_COLOURS["RC1"]]
                ),
                legend=alt.Legend(title="Arm", orient="top", direction="horizontal"),
            ),
        )
        labels = base.mark_text(align="left", dx=6, fontSize=BODY_FONT, aria=False).encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("fold_name:N", sort=list(score.FOLD_NAMES)),
            x="diff:Q",
            yOffset="arm_name:N",
            text=alt.Text("diff:Q", format="+.3f"),
        )
        zero = (
            alt.Chart(pl.DataFrame({"x": [0.0]}))
            .mark_rule(color=ocf.BLACK_1, aria=False)
            .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
        )
        rule = (
            alt.Chart(rule_frame)
            .mark_rule(color=ocf.BLACK_1, strokeDash=[4, 3], aria=False)
            .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
        )
        rule_text = (
            alt.Chart(rule_frame)
            .mark_text(align="left", dx=3, dy=-4, baseline="bottom", fontSize=BODY_FONT, aria=False)
            .encode(  # ty: ignore[unresolved-attribute]
                x="x:Q", y=alt.value(0), text="label:N"
            )
        )
        panels.append(
            alt.layer(
                zero, rule, rule_text, points, labels, width=FULL_WIDTH_PX, height=130
            ).properties(
                title=alt.TitleParams(
                    f"{form.capitalize()} form", anchor="start", fontSize=BODY_FONT
                )
            )
        )
    subtitle = (
        "Interim, four folds. Each dot is the mean CRPS of R1 or RC1 minus that of R0 of the same "
        "form, outside events and the 48 hours after them, at leads of 0 to 240 hours, over all 16 "
        "Primaries, raw bands. Solid line: no difference. Dashed line: the screen rule's limit of "
        "+0.02 points (worse than R0)."
    )
    return (
        alt.vconcat(*panels, spacing=20)
        .properties(
            title=caption(
                title="Change in CRPS outside events against R0, fold by fold",
                subtitle=subtitle,
            )
        )
        .configure_view(stroke=None)
    )


def main() -> None:
    """Draw the four figures."""
    save(chart=figure_1(data=headline_data()), path=DOCS_DIR / "figure_1_headline.svg")
    frame = sweep_frame()
    save(chart=figure_2(frame=frame), path=DOCS_DIR / "figure_2_sweep_median.svg")
    save(chart=figure_3(frame=frame), path=DOCS_DIR / "figure_3_sweep_band.svg")
    save(chart=figure_4(data=screen_data()), path=DOCS_DIR / "figure_4_screen_folds.svg")


if __name__ == "__main__":
    main()
