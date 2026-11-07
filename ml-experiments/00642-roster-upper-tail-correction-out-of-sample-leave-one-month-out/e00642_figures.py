# ruff: noqa: E501
"""Experiment 642: the figures of the write-up (SVG, 680 px wide)."""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00642_score as score
import polars as pl
from e00630_common import load  # ty: ignore[unresolved-import]
from plotting import ocf_theme as theme
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "00642-roster-upper-tail-correction-out-of-sample-leave-one-month-out"
DOCS: Final[Path] = Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / SLUG
TABLES: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG / "tables"
W: Final[int] = 600
LABEL: Final[dict[str, str]] = {
    "base": "Uncorrected",
    "S": "S: scale per Primary",
    "B": "B: S, then lead band and hour",
    "C": "C: rolling 8-week conformal",
    "Q": "Q: quantile mapping",
    "M": "M: median shift, then scale",
}
COLOUR: Final[dict[str, str]] = {
    "base": "#7F7F7F",
    "S": theme.DATA_BLUE,
    "B": theme.DATA_SKY,
    "C": theme.DATA_PURPLE,
    "Q": theme.DATA_GREEN,
    "M": theme.BRAND_ORANGE,
}
ARM_SCALE: Final[alt.Scale] = alt.Scale(
    domain=[LABEL[a] for a in LABEL], range=[COLOUR[a] for a in LABEL]
)
MONTHS: Final[list[str]] = ["2025-10", "2025-12", "2026-01", "2026-03", "2026-06"]


def save(chart: alt.TopLevelMixin, name: str, title: str, subtitle: str) -> None:
    """Write one chart with a wrapped title and subtitle."""
    DOCS.mkdir(parents=True, exist_ok=True)
    chart.properties(
        title=alt.TitleParams(
            textwrap.wrap(title, 62), subtitle=textwrap.wrap(subtitle, 92), anchor="start"
        )
    ).save(DOCS / name)


def headline() -> None:
    """Figure 1: pooled event-free exceedance of each arm, with day-resampling intervals."""
    rows = []
    for network, nlabel in (("K1", "K1"), ("Candidate", "Candidate")):
        r = pl.read_csv(TABLES / f"day_intervals_{network}.csv").row(0, named=True)
        for a in LABEL:
            for level, key in (
                ("Above q90 (nominal 10%)", "a90"),
                ("Above q95 (nominal 5%)", "a95"),
            ):
                rows.append(
                    {
                        "network": nlabel, "arm": LABEL[a], "level": level,
                        "v": r[f"{key}_{a}"] * 100, "lo": r[f"{key}_{a}_lo"] * 100, "hi": r[f"{key}_{a}_hi"] * 100,
                    }
                )  # fmt: skip
    d = (
        pl.DataFrame(rows)
        .with_columns(n=pl.when(pl.col("level").str.contains("q90")).then(10.0).otherwise(5.0))
        .to_pandas()
    )
    base = alt.Chart(d)
    enc = {"y": alt.Y("arm:N", sort=[LABEL[a] for a in LABEL], title=None), "yOffset": "network:N"}
    colour = alt.Color("arm:N", scale=ARM_SCALE, legend=None)
    rule = base.mark_rule().encode(  # ty: ignore[unresolved-attribute]
        **enc,
        x=alt.X(
            "lo:Q",
            title="Share of half-hours above the quantile (%)",
            scale=alt.Scale(domain=[0, 20]),
        ),
        x2="hi:Q",
        color=colour,
    )  # ty: ignore[unresolved-attribute]
    pts = base.mark_point(filled=True, size=45).encode(  # ty: ignore[unresolved-attribute]
        **enc,
        x="v:Q",
        color=colour,
        shape=alt.Shape("network:N", legend=alt.Legend(orient="top", title=None)),
    )  # ty: ignore[unresolved-attribute]
    nrule = base.mark_rule(strokeDash=[4, 3], color="#000000").encode(x="n:Q")  # ty: ignore[unresolved-attribute]
    chart = (
        alt.layer(nrule, rule, pts)
        .properties(width=W - 150, height=170)
        .facet(
            row=alt.Row("level:N", title=None, header=alt.Header(labelAngle=0, labelAlign="left"))
        )
    )  # ty: ignore[invalid-return-type]
    save(
        chart,
        "figure_1_exceedance_by_arm.svg",
        "Only the median-shift arm M moves both upper quantiles near nominal, and none reaches the bar",
        "Event-free half-hours of the five development months, 16 Primaries, each arm fitted on the other four months. Dot: pooled share. Line: 95% interval from resampling origin days. Dashed line: nominal share. Circle: Candidate, square: K1; colour is the arm.",
    )


def by_month() -> None:
    """Figure 2: q90 exceedance by held-out month, one panel per network."""
    panels = []
    for network in ("Candidate", "K1"):
        d = pl.read_csv(TABLES / f"by_month_{network}.csv").with_columns(
            arm_label=pl.col("arm").replace_strict(LABEL), v=pl.col("a90") * 100
        )
        base = alt.Chart(d.to_pandas())
        line = base.mark_line(point=True).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("month:N", sort=MONTHS, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("v:Q", title=f"{network}: above q90 (%)", scale=alt.Scale(domain=[6, 22])),
            color=alt.Color(
                "arm_label:N",
                scale=ARM_SCALE,
                legend=alt.Legend(orient="top", title=None, columns=2, labelLimit=300),
            ),
        )  # ty: ignore[unresolved-attribute]
        nominal = (
            alt.Chart(pl.DataFrame({"n": [10.0]}).to_pandas())  # ty: ignore[unresolved-attribute]
            .mark_rule(strokeDash=[4, 3], color="#000000")
            .encode(y="n:Q")
        )  # ty: ignore[unresolved-attribute]
        panels.append(alt.layer(nominal, line).properties(width=W, height=230))
    chart = alt.vconcat(*panels).resolve_scale(color="shared")  # ty: ignore[invalid-return-type]
    save(
        chart,
        "figure_2_exceedance_by_month.svg",
        "March 2026 stays above nominal for every arm, and June is overcorrected by most",
        "Share of event-free half-hours above q90 in each held-out month. Dashed line: nominal 10%. Month is the month of the forecast origin.",
    )


def by_group() -> None:
    """Figure 3: q90 and q95 exceedance by metering group for the Candidate."""
    rows = []
    for network in ("Candidate",):
        d = pl.read_csv(TABLES / f"by_group_{network}.csv").filter(
            ~pl.col("group").str.contains("report-only")
        )
        rows.append(d)
    d = (
        pl.concat(rows)
        .with_columns(arm_label=pl.col("arm").replace_strict(LABEL))
        .unpivot(
            index=["group", "arm_label"], on=["a90", "a95"], variable_name="level", value_name="v"
        )
        .with_columns(
            v=pl.col("v") * 100,
            level=pl.col("level").replace_strict(
                {"a90": "Above q90 (nominal 10%)", "a95": "Above q95 (nominal 5%)"}
            ),
        )
    )
    d = d.with_columns(n=pl.when(pl.col("level").str.contains("q90")).then(10.0).otherwise(5.0))
    base = alt.Chart(d.to_pandas())
    bars = base.mark_bar().encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("arm_label:N", sort=[LABEL[a] for a in LABEL], title=None, axis=None),
        y=alt.Y("v:Q", title="Share above the quantile (%)"),
        color=alt.Color(
            "arm_label:N",
            scale=ARM_SCALE,
            legend=alt.Legend(orient="top", title=None, columns=2, labelLimit=300),
        ),
    )  # ty: ignore[unresolved-attribute]
    nrule = base.mark_rule(strokeDash=[4, 3], color="#000000").encode(y="n:Q")  # ty: ignore[unresolved-attribute]
    chart = (
        alt.layer(bars, nrule)
        .properties(width=W // 4 - 10, height=170)
        .facet(
            column=alt.Column("group:N", title=None),
            row=alt.Row("level:N", title=None, header=alt.Header(labelAngle=0, labelAlign="left")),
        )
    )  # ty: ignore[invalid-return-type]
    save(
        chart,
        "figure_3_exceedance_by_metering_group.svg",
        "The corrections work better at the eight MVA Primaries than at the six MW Primaries",
        "Candidate, event-free half-hours, 14 Primaries (D15 and D18 left out). Dashed line: nominal share.",
    )


def example(*, primary: str, init: str, name: str, window_hours: float, title: str) -> None:
    """Figures 4 and 5: the Candidate at one Primary and one run, uncorrected and arm M."""
    full = load(network="Candidate")
    df = score.attach(df=full, tables=score.rolling_tables(df=full))
    df = df.filter(pl.col("month").is_in(MONTHS))
    fold = init[:7]
    lo, hi = score.fold_span(month=fold)
    test = df.filter(pl.col("month") == fold)
    train = df.filter(
        (pl.col("month") != fold)
        & pl.col("event_free")
        & ~((pl.col("valid_time") >= lo) & (pl.col("valid_time") < hi))
    )
    fold_tables = score.rolling_tables(df=df, exclude=(lo, hi)).select(
        "time_series_id", "band", "origin_date", pl.col("shift").alias("shift_fold")
    )
    train = (
        train.join(
            fold_tables,
            on=["time_series_id", "band", "origin_date"],
            how="left",
            maintain_order="left",
        )
        .with_columns(shift=pl.col("shift_fold"))
        .drop("shift_fold")
    )
    res = score.fit_and_apply(train=train, test=test)
    keep = (
        (test["primary_name"] == primary)
        & (test["power_fcst_init_time"].dt.strftime("%Y-%m-%d") == init)
        & (test["lead_hours"] < window_hours)
    ).to_numpy()
    sub = test.filter(pl.Series(keep))
    lead = sub["lead_hours"].to_numpy()
    panels = []
    for arm in ("base", "M"):
        q = res[arm][keep]
        d = (
            pl.DataFrame(
                {
                    "lead": lead / 24.0,
                    "q05": q[:, 1],
                    "q25": q[:, 3],
                    "q50": q[:, 5],
                    "q75": q[:, 7],
                    "q95": q[:, 9],
                    "q99": q[:, 10],
                    "y": sub["observed_power"].to_numpy(),
                }
            )
            .sort("lead")
            .to_pandas()
        )
        c = alt.Chart(d)
        x = alt.X(
            "lead:Q",
            title="Days after the forecast was issued",
            scale=alt.Scale(domain=[0, window_hours / 24.0]),
        )
        band90 = c.mark_area(
            opacity=0.25, color=COLOUR["M"] if arm == "M" else COLOUR["base"]
        ).encode(x=x, y=alt.Y("q05:Q", title="Power (MW or MVA)"), y2="q95:Q")  # ty: ignore[unresolved-attribute]
        band50 = c.mark_area(
            opacity=0.35, color=COLOUR["M"] if arm == "M" else COLOUR["base"]
        ).encode(x=x, y="q25:Q", y2="q75:Q")  # ty: ignore[unresolved-attribute]
        med = c.mark_line(color=COLOUR["M"] if arm == "M" else "#555555").encode(x=x, y="q50:Q")  # ty: ignore[unresolved-attribute]
        obs = c.mark_line(color="#000000", strokeWidth=1.2).encode(x=x, y="y:Q")  # ty: ignore[unresolved-attribute]
        panels.append(
            alt.layer(band90, band50, med, obs).properties(width=W, height=190, title=LABEL[arm])
        )
    chart = alt.vconcat(*panels)  # ty: ignore[invalid-return-type]
    save(
        chart,
        name,
        title,
        f"{primary}, run issued {init} 09:00 UTC (a winter example chosen in advance, not for a large miss). Black line: observed power. Line: median. Dark band: 50% interval. Light band: 90% interval. Candidate network.",
    )


def main() -> None:
    """Draw every figure."""
    headline()
    by_month()
    by_group()
    example(
        primary="D2",
        init="2026-01-13",
        name="figure_4_example_day_ahead.svg",
        window_hours=48.0,
        title="Day-ahead forecast: the median shift moves the Candidate's bands up",
    )
    example(
        primary="D2",
        init="2026-01-13",
        name="figure_5_example_full_horizon.svg",
        window_hours=240.0,
        title="Full 10-day horizon: the bands widen with lead and the shift stays small",
    )


if __name__ == "__main__":
    main()
