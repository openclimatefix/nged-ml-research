# ruff: noqa: E501
r"""Draw the figures of experiment 446 from the stored scores and forecasts (no fit).

Scores are read from `work/scores_s012/w1_15` (Brief B, seeds 0 to 2) and `work/scores_C/w1_15`
(Brief C, base S1), all at the 1.15 widening. Figures:

- Figure 1: absolute CRPS by row set and arm (Brief B).
- Figure 2: q95 and q99 coverage on clean rows, injected genuine level changes, and real event rows,
  with the target bands.
- Figure 3: 90% band-width ratio by lead band, and by days since the level change began.
- Figure 4: contrasts against S1 with 95% intervals and the minimum detectable effect.
- Figure 5: the mean scale factor of the scale-only arms by row set.
- Figure 6: coverage on cold winter nights in each test month.
- Figures 7 and 8: Brief C coverage cells (clean rows) per metering group.
- Figure 9: example forecasts, one clean origin and one with an injected genuine level change.
- Figure 10: six real events, S1 against U2.

```bash
source <paths>; PYTHONPATH=<the folders of e00446_score.py> DATA_PATH_INTERNAL=... \
    uv run --no-sync python ml-experiments/00446-*/e00446_figures.py
```
"""

import textwrap
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import altair as alt
import e00428_data as data  # ty: ignore[unresolved-import]
import e00428_predict as pr  # ty: ignore[unresolved-import]
import e00446_run as run  # ty: ignore[unresolved-import]
import e00446_score as score  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from plotting import ocf_theme as theme  # noqa: F401

FIGURE_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00446-honest-bands-and-tail-calibration-on-the-s1-network"
)
SCORES_B: Final[Path] = run.WORK_DIR / "scores_s012" / "w1_15"
SCORES_C: Final[Path] = run.WORK_DIR / "scores_C" / "w1_15"
MONTHS: Final[tuple[str, ...]] = ("2026-06", "2026-01", "2025-10", "2026-03")
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
SIGNED: Final[str] = "signed-MW six"
MVA: Final[str] = "MVA-only eight"
WIDTH: Final[int] = 700
POINTS: Final[float] = 100.0
TITLE_CHARS: Final[int] = 78

ARMS_B: Final[list[str]] = ["R-inj", "S1", "U1n-2", "U1n-5", "U2", "U2i", "U2c", "U2-clean", "U3"]
COLOURS_B: Final[list[str]] = [
    "#888888", "#306BFF", "#FF4901", "#FF8C5A", "#B701FF", "#D57AFF", "#10C5F7", "#17E58F", "#8CF2C7",
]  # fmt: skip
ARMS_C: Final[list[str]] = ["R-inj", "S1", "T1", "T2", "T3", "T4", "T5"]
COLOURS_C: Final[list[str]] = [
    "#888888", "#306BFF", "#FF4901", "#B701FF", "#17E58F", "#10C5F7", "#FF8C5A",
]  # fmt: skip
ROW_NAMES: Final[dict[str, str]] = {
    "clean": "Clean rows",
    "inj_harm": "Injected load-shift harm rows",
    "gen_jump": "Injected genuine level change rows",
    "real_harm": "Real harm rows",
    "real_event": "Real event rows",
}
ROW_ORDER: Final[list[str]] = list(ROW_NAMES.values())
TARGETS: Final[dict[str, tuple[float, float]]] = {
    "coverage_q01": (0.005, 0.02),
    "coverage_q05": (0.03, 0.07),
    "coverage_q95": (0.93, 0.97),
    "coverage_q99": (0.98, 0.995),
}
TAIL_NAMES: Final[dict[str, str]] = {
    "coverage_q01": "q01",
    "coverage_q05": "q05",
    "coverage_q95": "q95",
    "coverage_q99": "q99",
}
LEAD_ORDER: Final[list[str]] = ["0-12h", "12-36h", "36-72h", "days4-5", "days6-10"]
GREY: Final[str] = "#888888"


def arm_colour(*, arms: list[str], colours: list[str], legend: bool = False) -> alt.Color:
    """Return the colour encoding of an arm."""
    return alt.Color(
        "arm:N",
        scale=alt.Scale(domain=arms, range=colours),
        legend=alt.Legend(title=None, orient="top", columns=len(arms)) if legend else None,
    )


def titled(*, chart: alt.LayerChart, title: str) -> alt.LayerChart:
    """Return a chart with a small left-aligned panel title."""
    return chart.properties(title=alt.TitleParams(title, anchor="start", fontSize=11))


def absolute(*, scores: Path = SCORES_B) -> pl.DataFrame:
    """Return the absolute table with a readable row-set name."""
    return pl.read_parquet(scores / "absolute.parquet").with_columns(
        row=pl.col("row_set").replace_strict(ROW_NAMES, default=None)
    )


def save(*, chart: alt.TopLevelMixin, name: str) -> None:
    """Write a chart as SVG, wrapping its title lines to the plot width."""
    title = chart.title  # ty: ignore[unresolved-attribute]
    if isinstance(title, alt.TitleParams) and isinstance(title.text, list):
        lines = [w for t in title.text for w in textwrap.wrap(t, width=TITLE_CHARS)]
        chart = chart.properties(title=alt.TitleParams(lines, anchor="start"))  # ty: ignore[unresolved-attribute]
    chart.save(FIGURE_DIR / name)  # ty: ignore[unresolved-attribute]


def figure_absolute() -> None:
    """Figure 1: absolute CRPS by row set and arm."""
    a = absolute().filter(
        pl.col("row").is_not_null(),
        pl.col("group").is_in([SIGNED, MVA]),
        pl.col("arm").is_in(ARMS_B),
    )
    base = alt.Chart(a.to_pandas()).encode(
        y=alt.Y("arm:N", sort=ARMS_B, title=None),
        x=alt.X("crps:Q", title="CRPS (points of capacity; smaller is better)", scale=alt.Scale(zero=False)),
        color=arm_colour(arms=ARMS_B, colours=COLOURS_B),
    )  # fmt: skip
    chart = (
        base.mark_point(filled=True, size=60)  # ty: ignore[unresolved-attribute]
        .properties(width=(WIDTH - 200) // 2, height=120)
        .facet(
            row=alt.Row(
                "row:N",
                sort=ROW_ORDER,
                title=None,
                header=alt.Header(labelAngle=0, labelAlign="left"),
            ),
            column=alt.Column("group:N", title=None, sort=[SIGNED, MVA]),
        )
        .resolve_scale(x="independent")
        .properties(
            title=alt.TitleParams(
                [
                    "Absolute CRPS by row set and arm (interim)",
                    "Each dot is one arm. Four test months, seeds 0 to 2, 1.15 widening, days 1 to 10.",
                ],
                anchor="start",
            )
        )
    )
    save(chart=chart, name="figure_1_absolute_crps.svg")


def band_layer(*, tail: str, orient_x: bool) -> alt.Chart:
    """Return a shaded rectangle for one tail's target range."""
    low, high = TARGETS[tail]
    frame = pl.DataFrame({"low": [low], "high": [high]}).to_pandas()
    return alt.Chart(frame).mark_rect(opacity=0.18, color="#17E58F").encode(x="low:Q", x2="high:Q")  # ty: ignore[unresolved-attribute]


def figure_coverage() -> None:
    """Figure 2: q95 and q99 coverage by arm on three row sets, with target bands."""
    a = absolute().filter(
        pl.col("row_set").is_in(["clean", "gen_jump", "real_event"]),
        pl.col("group") == SIGNED,
        pl.col("arm").is_in(ARMS_B),
    )
    rows = a.unpivot(
        index=["arm", "row"],
        on=["coverage_q95", "coverage_q99"],
        variable_name="tail",
        value_name="coverage",
    ).with_columns(tail_name=pl.col("tail").replace_strict(TAIL_NAMES))
    bands = pl.DataFrame(
        [
            {"tail_name": TAIL_NAMES[t], "low": TARGETS[t][0], "high": TARGETS[t][1]}
            for t in ("coverage_q95", "coverage_q99")
        ]
    )
    panels = []
    for tail_name in ("q95", "q99"):
        lo, hi = (0.75, 1.0) if tail_name == "q95" else (0.85, 1.0)
        column = []
        for k, row_name in enumerate(
            ROW_ORDER[:0] + [ROW_NAMES[r] for r in ("clean", "gen_jump", "real_event")]
        ):
            x = alt.X(
                "coverage:Q",
                title=f"{tail_name} coverage" if k == 2 else None,
                scale=alt.Scale(domain=[lo, hi]),
            )
            dom = alt.Scale(domain=[lo, hi])
            shade = (
                alt.Chart(bands.filter(pl.col("tail_name") == tail_name).to_pandas())  # ty: ignore[unresolved-attribute]
                .mark_rect(opacity=0.2, color="#17E58F")
                .encode(x=alt.X("low:Q", scale=dom), x2="high:Q")
            )
            nominal = (
                alt.Chart(pl.DataFrame({"v": [0.95 if tail_name == "q95" else 0.99]}).to_pandas())  # ty: ignore[unresolved-attribute]
                .mark_rule(color="#333", strokeDash=[3, 3])
                .encode(x=alt.X("v:Q", scale=dom))
            )  # ty: ignore[unresolved-attribute]
            dots = (
                alt.Chart(  # ty: ignore[unresolved-attribute]
                    rows.filter(
                        pl.col("tail_name") == tail_name, pl.col("row") == row_name
                    ).to_pandas()
                )  # ty: ignore[unresolved-attribute]
                .mark_point(filled=True, size=60)
                .encode(
                    y=alt.Y("arm:N", sort=ARMS_B, title=None),
                    x=x,
                    color=arm_colour(arms=ARMS_B, colours=COLOURS_B),
                )
            )
            column.append(
                titled(
                    chart=alt.layer(shade, nominal, dots).properties(  # ty: ignore[invalid-argument-type]
                        width=(WIDTH - 130) // 2, height=130
                    ),
                    title=f"{row_name}" if tail_name == "q95" else "",
                )
            )
        panels.append(alt.vconcat(*column))
    chart = alt.hconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "Upper-tail coverage at the signed-MW six (interim)",
                "Dot: one arm. Green band: target range. Dashed: nominal level. 1.15 widening, seeds 0 to 2.",
            ],
            anchor="start",
        )
    )
    save(chart=chart, name="figure_2_tail_coverage.svg")


def lead_widths() -> pl.DataFrame:
    """Return the mean 90% band width by arm, row set, and lead band at the signed-MW six."""
    b = pl.read_parquet(SCORES_B / "blocks.parquet").filter(
        pl.col("group") == SIGNED,
        pl.col("row_set").is_in(
            ["clean", "inj_clean", "inj_harm", "gen_clean", "gen_jump", "real_event"]
        ),
    )
    return (
        b.group_by("arm", "row_set", "lead_band")
        .agg(pl.col("n").sum(), pl.col("width90").sum())
        .with_columns(width=pl.col("width90") / pl.col("n"))
    )


def figure_width_ratio() -> None:
    """Figure 3: 90% band-width ratio by lead band and by days since the level change began."""
    w = lead_widths()
    pairs = (
        ("inj_harm", "inj_clean", "Injected load-shift harm rows"),
        ("gen_jump", "gen_clean", "Injected genuine level change rows"),
        ("real_event", "clean", "Real event rows"),
    )
    parts = []
    for numerator, denominator, label in pairs:
        top = w.filter(pl.col("row_set") == numerator).select(
            "arm", "lead_band", top=pl.col("width")
        )
        bottom = w.filter(pl.col("row_set") == denominator).select(
            "arm", "lead_band", bottom=pl.col("width")
        )
        parts.append(
            top.join(bottom, on=["arm", "lead_band"]).with_columns(
                ratio=pl.col("top") / pl.col("bottom"), comparison=pl.lit(label)
            )
        )
    t = pl.concat(parts).filter(pl.col("arm").is_in(["S1", "U1n-2", "U2", "U2-clean"]))
    arms = ["S1", "U1n-2", "U2", "U2-clean"]
    colours = ["#306BFF", "#FF4901", "#B701FF", "#17E58F"]
    one = (
        alt.Chart(pl.DataFrame({"v": [1.0]}).to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_rule(color="#333", strokeDash=[3, 3])
        .encode(y="v:Q")
    )  # ty: ignore[unresolved-attribute]
    by_lead = alt.vconcat(
        *[
            titled(
                chart=alt.layer(  # ty: ignore[invalid-argument-type]
                    one,
                    alt.Chart(t.filter(pl.col("comparison") == label).to_pandas())  # ty: ignore[unresolved-attribute]
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("lead_band:O", sort=LEAD_ORDER, title="Lead band"),
                        y=alt.Y("ratio:Q", title="Band-width ratio", scale=alt.Scale(zero=False)),
                        color=arm_colour(arms=arms, colours=colours, legend=label == pairs[0][2]),
                    ),
                ).properties(width=WIDTH - 90, height=110),
                title=label,
            )
            for _, _, label in pairs
        ]
    )
    # By days since the level change began.
    d = pl.read_parquet(SCORES_B / "width_ratios.parquet").filter(
        pl.col("group") == SIGNED,
        pl.col("row_set").is_in(["gen_jump_d0", "gen_jump_d1_3", "gen_jump_d3p"]),
        pl.col("arm").is_in(arms),
    ).with_columns(
        age=pl.col("row_set").replace_strict(
            {"gen_jump_d0": "Under 1 day", "gen_jump_d1_3": "1 to under 3 days", "gen_jump_d3p": "3 days or more"}
        )
    )  # fmt: skip
    ages = ["Under 1 day", "1 to under 3 days", "3 days or more"]
    by_age = alt.layer(
        one,
        alt.Chart(d.to_pandas())
        .mark_line(point=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("age:O", sort=ages, title="Time since the newest lag's level change began"),
            y=alt.Y(
                "width_ratio:Q",
                title="Width ratio (genuine change rows over clean)",
                scale=alt.Scale(zero=False),
            ),
            color=arm_colour(arms=arms, colours=colours),
        ),
    ).properties(width=WIDTH - 90, height=140)
    chart = alt.vconcat(by_lead, by_age).properties(
        title=alt.TitleParams(
            [
                "Does the 90% band widen where a level change is ambiguous? (signed-MW six, interim)",
                "Ratio of mean band width on the named rows to the same arm's width on matching unaffected rows. Dashed: 1.",
            ],
            anchor="start",
        )
    )
    save(chart=chart, name="figure_3_band_width_ratio.svg")


def figure_contrasts() -> None:
    """Figure 4: each arm minus S1 with 95% intervals, and the minimum detectable effect."""
    c = pl.read_parquet(SCORES_B / "contrasts.parquet").drop("fold_differences")
    mde = pl.read_parquet(SCORES_B / "mde.parquet")
    arms = ["U1n-2", "U1n-5", "U2", "U2i", "U2c", "U2-clean", "U3"]
    c = (
        c.filter(
            pl.col("metric") == "crps",
            pl.col("group") == SIGNED,
            pl.col("row_set").is_in(list(ROW_NAMES)),
            pl.col("contrast").is_in([f"{a} - S1" for a in arms]),
        )
        .with_columns(
            arm=pl.col("contrast").str.replace(" - S1", ""),
            row=pl.col("row_set").replace_strict(ROW_NAMES),
        )
        .join(
            mde.filter(pl.col("group") == SIGNED).select("row_set", "mde"), on="row_set", how="left"
        )
        .with_columns(neg_mde=-pl.col("mde"))
    )
    y = alt.Y("arm:N", sort=arms, title=None)
    colour = arm_colour(arms=ARMS_B, colours=COLOURS_B)

    def panel(row_name: str) -> alt.LayerChart:
        sub = c.filter(pl.col("row") == row_name).to_pandas()
        d = alt.Chart(sub)
        one_row = alt.Chart(sub.head(1))
        return titled(
            chart=alt.layer(  # ty: ignore[invalid-argument-type]
                one_row.mark_rect(opacity=0.15, color="#888").encode(x="neg_mde:Q", x2="mde:Q"),  # ty: ignore[unresolved-attribute]
                one_row.mark_rule(color="#333", strokeDash=[3, 3]).encode(x=alt.datum(0)),  # ty: ignore[unresolved-attribute]
                d.mark_rule(strokeWidth=1.5).encode(  # ty: ignore[unresolved-attribute]
                    y=y,
                    x=alt.X("low:Q", title="Arm minus S1, CRPS (points; negative is better)"),
                    x2="high:Q",
                    color=colour,
                ),  # ty: ignore[unresolved-attribute]
                d.mark_point(filled=True, size=50).encode(y=y, x="difference:Q", color=colour),  # ty: ignore[unresolved-attribute]
            ).properties(width=WIDTH - 220, height=110),
            title=row_name,
        )

    chart = (
        alt.vconcat(*[panel(r) for r in ROW_ORDER])
        .resolve_scale(x="independent")
        .properties(
            title=alt.TitleParams(
                [
                    "Each Brief B arm minus S1, CRPS at the signed-MW six (interim)",
                    "Dot: estimate. Line: 95% interval from resampling test-month origin days. Grey: plus or minus the minimum detectable effect.",
                ],
                anchor="start",
            )
        )
    )
    save(chart=chart, name="figure_4_contrasts.svg")


def figure_scale_factor() -> None:
    """Figure 5: the mean scale factor of the scale-only arms by row set."""
    a = absolute().filter(
        pl.col("arm").is_in(["U2", "U2i", "U2c", "U2-clean"]),
        pl.col("group") == SIGNED,
        pl.col("row").is_not_null(),
    )
    arms = ["U2", "U2i", "U2c", "U2-clean"]
    colours = ["#B701FF", "#D57AFF", "#10C5F7", "#17E58F"]
    limit = (
        alt.Chart(pl.DataFrame({"v": [1.03]}).to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_rule(color="#FF4901", strokeDash=[4, 3])
        .encode(x="v:Q")
    )  # ty: ignore[unresolved-attribute]
    one = alt.Chart(pl.DataFrame({"v": [1.0]}).to_pandas()).mark_rule(color="#333").encode(x="v:Q")  # ty: ignore[unresolved-attribute]

    def bars_for(row_name: str) -> alt.LayerChart:
        bars = (
            alt.Chart(a.filter(pl.col("row") == row_name).to_pandas())  # ty: ignore[unresolved-attribute]
            .mark_bar()
            .encode(
                y=alt.Y("arm:N", sort=arms, title=None),
                x=alt.X(
                    "scale_f_mean:Q",
                    title="Mean scale factor (1.00 means no widening)",
                    scale=alt.Scale(domain=[0.98, 1.06]),
                ),
                color=arm_colour(arms=arms, colours=colours),
            )
        )
        return titled(
            chart=alt.layer(bars, one, limit).properties(width=WIDTH - 220, height=70),  # ty: ignore[invalid-argument-type]
            title=row_name,
        )

    chart = alt.vconcat(*[bars_for(r) for r in ROW_ORDER]).properties(
        title=alt.TitleParams(
            [
                "Mean scale factor of the scale-only arms (signed-MW six, interim)",
                "Black: 1.00. Orange dashed: the 1.03 limit on clean rows. The factor multiplies each component's spread.",
            ],
            anchor="start",
        )
    )
    save(chart=chart, name="figure_5_scale_factor.svg")


def figure_cold_nights() -> None:
    """Figure 6: cold-winter-night coverage per test month, Brief B and Brief C arms."""
    b = pl.read_parquet(SCORES_B / "cold_nights.parquet")
    c = pl.read_parquet(SCORES_C / "cold_nights.parquet").filter(
        pl.col("arm").is_in(["T1", "T2", "T3", "T4", "T5"])
    )
    arms = ["R-inj", "S1", "U1n-2", "U1n-5", "U2", "T2", "T4", "T5"]
    colours = [
        "#888888",
        "#306BFF",
        "#FF4901",
        "#FF8C5A",
        "#B701FF",
        "#17E58F",
        "#10C5F7",
        "#FFB38F",
    ]
    t = (
        pl.concat([b, c])
        .filter(pl.col("group") == SIGNED, pl.col("arm").is_in(arms))
        .with_columns(
            label=pl.col("month") + pl.lit(" (") + pl.col("n").cast(pl.String) + pl.lit(" rows)")
        )
        .unpivot(
            index=["arm", "label"],
            on=["coverage_q95", "coverage_q99"],
            variable_name="tail",
            value_name="coverage",
        )
        .with_columns(tail_name=pl.col("tail").replace_strict(TAIL_NAMES))
    )
    targets = pl.DataFrame({"tail_name": ["q95", "q99"], "target": [0.92, 0.975]})
    panels = []
    for tail_name in ("q95", "q99"):
        rule = (
            alt.Chart(targets.filter(pl.col("tail_name") == tail_name).to_pandas())  # ty: ignore[unresolved-attribute]
            .mark_rule(color="#FF4901", strokeDash=[4, 3])
            .encode(x=alt.X("target:Q", scale=alt.Scale(domain=[0.75, 1.0])))
        )  # ty: ignore[unresolved-attribute]
        labels = sorted(t["label"].unique().to_list())
        panels.append(
            alt.vconcat(
                *[
                    titled(
                        chart=alt.layer(  # ty: ignore[invalid-argument-type]
                            rule,
                            alt.Chart(  # ty: ignore[unresolved-attribute]
                                t.filter(
                                    pl.col("tail_name") == tail_name, pl.col("label") == lab
                                ).to_pandas()
                            )  # ty: ignore[unresolved-attribute]
                            .mark_point(filled=True, size=60)
                            .encode(
                                y=alt.Y("arm:N", sort=arms, title=None),
                                x=alt.X(
                                    "coverage:Q",
                                    title=f"{tail_name} coverage",
                                    scale=alt.Scale(domain=[0.75, 1.0]),
                                ),
                                color=arm_colour(arms=arms, colours=colours),
                            ),
                        ).properties(width=(WIDTH - 130) // 2, height=120),
                        title=lab if tail_name == "q95" else "",
                    )
                    for lab in labels
                ]
            )
        )
    chart = alt.hconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "Coverage on cold winter nights, per test month (signed-MW six, interim)",
                "Orange dashed: the target (q95 at least 0.92, q99 at least 0.975). Only 2 of the 4 test months have cold-night rows.",
            ],
            anchor="start",
        )
    )
    save(chart=chart, name="figure_6_cold_nights.svg")


def figure_cells(*, group: str, name: str) -> None:
    """Figures 7 and 8: Brief C coverage cells on clean rows (four tails, two lead windows)."""
    a = pl.read_parquet(SCORES_C / "absolute.parquet").filter(
        pl.col("row_set") == "clean", pl.col("group") == group
    ).with_columns(window=pl.lit("Days 1 to 10"))  # fmt: skip
    lb = pl.read_parquet(SCORES_C / "clean_by_lead.parquet").filter(
        pl.col("group") == group, pl.col("lead_band") == "12-36h"
    ).with_columns(window=pl.lit("12 to 36 hours"))  # fmt: skip
    cols = ["arm", "window", *TARGETS]
    t = pl.concat([a.select(cols), lb.select(cols)]).filter(pl.col("arm").is_in(ARMS_C))
    rows = t.unpivot(
        index=["arm", "window"], variable_name="tail", value_name="coverage"
    ).with_columns(tail_name=pl.col("tail").replace_strict(TAIL_NAMES))
    bands = pl.DataFrame(
        [{"tail_name": TAIL_NAMES[k], "low": v[0], "high": v[1]} for k, v in TARGETS.items()]
    )
    panels = []
    for tail_name in ("q01", "q05", "q95", "q99"):
        sub = rows.filter(pl.col("tail_name") == tail_name)
        centre = float(sub["coverage"].mean())  # ty: ignore[invalid-argument-type]
        span = 0.03 if tail_name in ("q01", "q05") else 0.07
        lo = max(
            0.0,
            round(min(centre, bands.filter(pl.col("tail_name") == tail_name)["low"][0]) - span, 3),
        )
        hi = min(
            1.0,
            round(max(centre, bands.filter(pl.col("tail_name") == tail_name)["high"][0]) + span, 3),
        )
        shade = (
            alt.Chart(bands.filter(pl.col("tail_name") == tail_name).to_pandas())  # ty: ignore[unresolved-attribute]
            .mark_rect(opacity=0.2, color="#17E58F")
            .encode(x=alt.X("low:Q", scale=alt.Scale(domain=[lo, hi])), x2="high:Q")
        )
        stack = []
        for window in ("Days 1 to 10", "12 to 36 hours"):
            d = alt.Chart(sub.filter(pl.col("window") == window).to_pandas())  # ty: ignore[unresolved-attribute]
            stack.append(
                titled(
                    chart=alt.layer(  # ty: ignore[invalid-argument-type]
                        shade,
                        d.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
                            y=alt.Y("arm:N", sort=ARMS_C, title=None),
                            x=alt.X(
                                "coverage:Q",
                                title=f"{tail_name} coverage",
                                scale=alt.Scale(domain=[lo, hi]),
                            ),
                            color=arm_colour(arms=ARMS_C, colours=COLOURS_C),
                        ),
                    ).properties(width=(WIDTH - 150) // 2, height=100),
                    title=window,
                )
            )
        panels.append(alt.vconcat(*stack))
    chart = alt.vconcat(
        alt.hconcat(panels[0], panels[1]), alt.hconcat(panels[2], panels[3])
    ).properties(
        title=alt.TitleParams(
            [
                f"Brief C coverage cells on clean rows ({group}, interim)",
                "Dot: one arm. Green band: target range. 1.15 widening, base S1, seeds 0 to 2.",
            ],
            anchor="start",
        )
    )
    save(chart=chart, name=name)


def example_origin(*, arm: str, month: str, world: int) -> pl.DataFrame:
    """Return the forecasts of an arm for a month and world with the Primary's label."""
    f = score.read_arm(
        arm=arm, month=month, world=world, seeds=SEEDS, factor=score.score428.WIDENING
    )
    assert f is not None
    sites = data.sites()
    return f.with_columns(
        site=pl.col("primary").replace_strict(dict(enumerate(sites)), return_dtype=pl.String),
        arm=pl.lit(arm),
    )


def choose_clean_origin(*, f: pl.DataFrame) -> tuple[str, int]:
    """Return a signed-MW Primary and origin whose whole 10-day window is clean."""
    g = (
        f.filter(pl.col("site").is_in(score.SIGNED_MW_LABELS))
        .group_by("site", "origin_us")
        .agg(
            pl.len().alias("n"),
            (pl.col("win_inj") | pl.col("win_real") | pl.col("tgt_inj") | pl.col("tgt_real"))
            .any()
            .alias("dirty"),
        )
        .filter(~pl.col("dirty"), pl.col("n") == 480)
        .sort("origin_us", "site")
    )
    row = g.row(len(g) // 2, named=True)
    return row["site"], int(row["origin_us"])


def choose_jump_origin(*, month: str) -> tuple[int, str, int]:
    """Return the world, signed-MW Primary, and origin with the largest jump seen at a fresh newest lag."""
    best: tuple[float, int, str, int] | None = None
    for world in run.GEN_WORLDS:
        f = example_origin(arm="S1", month=month, world=world)
        g = (
            f.filter(pl.col("site").is_in(score.SIGNED_MW_LABELS), pl.col("shift_lag"))
            .group_by("site", "origin_us")
            .agg(
                pl.col("days_since").min().alias("days"),
                (pl.col("actual") - pl.col("actual_clean")).abs().mean().alias("deviation"),
            )
            .filter(pl.col("days") < 1.0)
            .sort("deviation", descending=True)
        )
        if g.is_empty():
            continue
        row = g.row(0, named=True)
        if best is None or row["deviation"] > best[0]:
            best = (float(row["deviation"]), world, row["site"], int(row["origin_us"]))
    assert best is not None, "no origin with a level change in force at the newest lag"
    return best[1], best[2], best[3]


EXAMPLE_ARMS: Final[list[str]] = ["S1", "U1n-2", "U2"]
EXAMPLE_COLOURS: Final[list[str]] = ["#306BFF", "#FF4901", "#B701FF"]


def example_frame(*, month: str, world: int, site: str, origin_us: int) -> pl.DataFrame:
    """Return the example arms' rows for one origin."""
    parts = [
        example_origin(arm=arm, month=month, world=world).filter(
            pl.col("site") == site, pl.col("origin_us") == origin_us
        )
        for arm in EXAMPLE_ARMS
    ]
    return pl.concat(parts, how="diagonal").with_columns(
        hours=pl.col("lead").cast(pl.Float64),
        **{q: pl.col(q) * POINTS for q in ("q01", "q05", "q50", "q95", "q99", "actual")},
        actual_clean=pl.col("actual_clean") * POINTS
        if "actual_clean" in parts[0].columns
        else pl.lit(None),
    )


def figure_examples(*, month: str = "2026-01") -> None:
    """Figure 9: S1, U1n-2, and U2 on a clean origin and on an origin with a fresh injected level change."""
    base = example_origin(arm="S1", month=month, world=0)
    clean_site, clean_origin = choose_clean_origin(f=base)
    world, jump_site, jump_origin = choose_jump_origin(month=month)
    cases = [
        ("Clean origin", 0, clean_site, clean_origin),
        ("Injected genuine level change", world, jump_site, jump_origin),
    ]
    panels = []
    for title, w, site, origin in cases:
        d = example_frame(month=month, world=w, site=site, origin_us=origin)
        for arm_i, arm in enumerate(EXAMPLE_ARMS):
            sub = d.filter(pl.col("arm") == arm).sort("hours")
            x = alt.X(
                "hours:Q",
                title="Hours after the forecast origin" if arm_i == len(EXAMPLE_ARMS) - 1 else None,
                scale=alt.Scale(domain=[0, 240]),
                axis=alt.Axis(values=[0, 48, 96, 144, 192, 240]),
            )
            pdata = sub.to_pandas()
            outer = (
                alt.Chart(pdata)  # ty: ignore[unresolved-attribute]
                .mark_area(opacity=0.12, color=EXAMPLE_COLOURS[arm_i])
                .encode(
                    x=x,
                    y=alt.Y("q01:Q", title="Power (points)", scale=alt.Scale(zero=False)),
                    y2="q99:Q",
                )
            )  # ty: ignore[unresolved-attribute]
            inner = (
                alt.Chart(pdata)  # ty: ignore[unresolved-attribute]
                .mark_area(opacity=0.28, color=EXAMPLE_COLOURS[arm_i])
                .encode(x=x, y="q05:Q", y2="q95:Q")
            )  # ty: ignore[unresolved-attribute]
            median = alt.Chart(pdata).mark_line(color=EXAMPLE_COLOURS[arm_i]).encode(x=x, y="q50:Q")  # ty: ignore[unresolved-attribute]
            truth = (
                alt.Chart(pdata)  # ty: ignore[unresolved-attribute]
                .mark_line(color="#333", strokeDash=[2, 2])
                .encode(x=x, y="actual:Q")
            )  # ty: ignore[unresolved-attribute]
            panels.append(
                (outer + inner + median + truth).properties(
                    width=WIDTH - 90,
                    height=110,
                    title=alt.TitleParams(f"{title}: {arm}", anchor="start", fontSize=11),
                )
            )
    chart = alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "Forecasts of S1, U1n-2, and U2 from one origin (10-day horizon, interim)",
                "Line: median. Dark band: 5% to 95%. Light band: 1% to 99%. Dashed: the reading, which steps at the level change.",
            ],
            anchor="start",
        )
    )
    save(chart=chart, name="figure_9_example_forecasts.svg")


def real_events(*, count: int) -> list[dict]:
    """Return the longest padded real events of each metering group."""
    padded = data.real_event_mask()
    sites = data.sites()
    chosen: list[dict] = []
    for label, members in (
        (SIGNED, score.SIGNED_MW_LABELS),
        (MVA, score.MVA_ONLY_LABELS),
    ):
        found = []
        for site in members:
            row = padded[sites.index(site)]
            edges = np.diff(np.concatenate([[0], row.astype(np.int8), [0]]))
            for start, end in zip(
                np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0], strict=True
            ):
                found.append({"group": label, "site": site, "start": int(start), "end": int(end)})
        found.sort(key=lambda e: e["end"] - e["start"], reverse=True)  # ty: ignore[unsupported-operator]
        chosen += found[: count * 4]
    return chosen


def figure_events() -> None:
    """Figure 10: six real events, three per metering group, S1 against U2."""
    import e00087_common as common  # ty: ignore[unresolved-import]

    arms = ["S1", "U2"]
    colours = ["#306BFF", "#B701FF"]
    times = common.grid_times()
    power, _ = data.load_stack(include_eval=False)
    month_of = {m: datetime.fromisoformat(f"{m}-01").replace(tzinfo=UTC) for m in MONTHS}
    picked: list[dict] = []
    per_group = {SIGNED: 0, MVA: 0}
    for e in real_events(count=14):
        if per_group[e["group"]] >= 3:
            continue
        start_time = times[e["start"]]
        month = next(
            (m for m, t in month_of.items() if t <= start_time < (t + timedelta(days=31))), None
        )
        if month is None:
            continue
        e["month"] = month
        picked.append(e)
        per_group[e["group"]] += 1
    panels = []
    for i, e in enumerate(picked, start=1):
        frames = []
        for arm in arms:
            f = score.read_arm(
                arm=arm, month=e["month"], world=0, seeds=SEEDS, factor=score.score428.WIDENING
            )
            assert f is not None
            frames.append(
                f.filter(pl.col("primary") == data.sites().index(e["site"])).with_columns(
                    arm=pl.lit(arm)
                )
            )
        f = pl.concat(frames, how="diagonal").with_columns(
            time=pl.from_epoch("valid_us", time_unit="us")
        )
        t0 = times[e["start"]] - timedelta(days=3)
        t1 = times[min(e["end"], len(times) - 1)] + timedelta(days=2)
        origin_us = int((times[e["start"]] - timedelta(days=1)).timestamp() * 1_000_000)
        origins = np.array(sorted(f["origin_us"].unique().to_list()))
        if not len(origins):
            continue
        origin = int(origins[np.argmin(np.abs(origins - origin_us))])
        d = f.filter(pl.col("origin_us") == origin, pl.col("lead") <= 120).with_columns(
            q05=pl.col("q05") * POINTS, q50=pl.col("q50") * POINTS, q95=pl.col("q95") * POINTS
        )
        grid = (
            pl.DataFrame({"time": times})
            .with_columns(actual=pl.Series(power[0][data.sites().index(e["site"])] * POINTS))
            .filter(pl.col("time").is_between(t0, t1))
            .drop_nulls()
            .drop_nans()
        )
        colour = arm_colour(arms=arms, colours=colours, legend=i == 1)
        x = alt.X("time:T", title=None, scale=alt.Scale(domain=[t0, t1]))
        pdata = d.to_pandas()
        shade = (
            alt.Chart(  # ty: ignore[unresolved-attribute]
                pl.DataFrame(
                    {"a": [times[e["start"]]], "b": [times[min(e["end"], len(times) - 1)]]}
                ).to_pandas()
            )  # ty: ignore[unresolved-attribute]
            .mark_rect(opacity=0.12, color="#888")
            .encode(x="a:T", x2="b:T")
        )
        band = (
            alt.Chart(pdata)  # ty: ignore[unresolved-attribute]
            .mark_area(opacity=0.14)
            .encode(
                x=x,
                y=alt.Y("q05:Q", title="Power (points)", scale=alt.Scale(zero=False)),
                y2="q95:Q",
                color=colour,
            )
        )  # ty: ignore[unresolved-attribute]
        median = alt.Chart(pdata).mark_line().encode(x=x, y="q50:Q", color=colour)  # ty: ignore[unresolved-attribute]
        truth = alt.Chart(grid.to_pandas()).mark_line(color="#333").encode(x=x, y="actual:Q")  # ty: ignore[unresolved-attribute]
        panels.append(
            (shade + band + median + truth).properties(
                width=WIDTH - 90,
                height=130,
                title=f"Event {i}: Primary {e['site']} ({e['group']}), forecast from the day before the event starts",
            )
        )
    chart = alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "Labelled load-shift events: S1 and U2 forecasts made the day before each event starts",
                "Shaded: the padded event. Black: the cleaned reading. Events are labelled by hand, and some real events are not logged.",
            ],
            anchor="start",
        )
    )
    save(chart=chart, name="figure_10_real_events.svg")


def main() -> None:
    """Draw every figure."""
    pr.PRED_DIR = run.PRED_DIR
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure_absolute()
    figure_coverage()
    figure_width_ratio()
    figure_contrasts()
    figure_scale_factor()
    figure_cold_nights()
    figure_cells(group=SIGNED, name="figure_7_brief_c_cells_signed_mw.svg")
    figure_cells(group=MVA, name="figure_8_brief_c_cells_mva_only.svg")
    figure_examples()
    figure_events()


if __name__ == "__main__":
    main()
