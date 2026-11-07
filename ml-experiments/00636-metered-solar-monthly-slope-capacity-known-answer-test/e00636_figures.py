"""Experiment 636: draw the four figures of the write-up from the analysis CSVs."""

from pathlib import Path

import altair as alt
import e00636_common as c
import plotting.ocf_theme as ocf
import polars as pl

DOCS = Path("docs/ml-experiments/00636-metered-solar-monthly-slope-capacity-known-answer-test")
WIDTH = 590
HEIGHT = 230
WORLDS = {
    "ac": "World 1: alternating-current loss",
    "dc": "World 2: direct-current loss under clipping",
}
FARMS = [f"Solar {f}" for f in c.TESTABLE]
FARM_COLOURS = [ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_SKY, ocf.DATA_PURPLE, ocf.DATA_GREEN]
EST_NAMES = {
    "slope": "Slope (all daylight half-hours)",
    "envelope": "Sunny-row envelope (experiment 618)",
    "slope_nozero": "Slope without zero rows (post hoc)",
}
EST_COLOURS = {"slope": ocf.BRAND_ORANGE, "envelope": ocf.DATA_BLUE, "slope_nozero": ocf.DATA_GREEN}
BAR = (
    alt.Chart(pl.DataFrame({"y": [0.8]}))
    .mark_rule(strokeDash=[5, 4], color="#000000")
    .encode(y="y:Q")
)


def stacked(build, t: pl.DataFrame, title: list[str]) -> alt.VConcatChart:
    """Return one panel per world, each built by `build(data)`, under a shared title."""
    panels = [
        build(t.filter(pl.col("world") == w)).properties(
            width=WIDTH - 90, height=HEIGHT, title=alt.TitleParams(w, fontSize=12, anchor="start")
        )
        for w in WORLDS.values()
    ]
    return alt.vconcat(*panels).properties(
        title=alt.TitleParams(title, anchor="start", fontSize=13, subtitleFontSize=11)
    )


def _recall() -> pl.DataFrame:
    return pl.read_csv(c.DATA_DIR / "recall.csv").with_columns(
        farm="Solar " + pl.col("farm"), world=pl.col("world").replace(WORLDS),
        est=pl.col("est").replace(EST_NAMES),
    )  # fmt: skip


def fig_headline() -> None:
    """Recall of a 10% step at 3 months by farm and estimator, one panel per world."""
    t = _recall().filter(
        (pl.col("loss") == 0.10) & (pl.col("k") == 3) & (pl.col("est") != EST_NAMES["slope_nozero"])
    )
    domain = [EST_NAMES["slope"], EST_NAMES["envelope"]]

    def build(d: pl.DataFrame) -> alt.LayerChart:
        base = alt.Chart(d).encode(
            x=alt.X("farm:N", title=None, sort=FARMS),
            xOffset="est:N",
            color=alt.Color("est:N", title=None, scale=alt.Scale(domain=domain, range=[EST_COLOURS["slope"], EST_COLOURS["envelope"]]), legend=alt.Legend(orient="top", labelLimit=500, columns=1)),
        )  # fmt: skip
        y = alt.Y("recall:Q", title="Share of 10% steps found", scale=alt.Scale(domain=[0, 1]))
        err = base.mark_rule(strokeWidth=2).encode(
            y=alt.Y("lo:Q", title="Share of 10% steps found"), y2="hi:Q"
        )
        return err + base.mark_point(filled=True, size=70).encode(y=y) + BAR

    stacked(build, t, ["A 10% capacity step is found at Solar C and Solar F, not at Solar A, B, or D", "Dot: share of 20 start months with the step found within 3 months. Line: 95% Wilson interval. Dashed: 80% pass bar."]).save(DOCS / "figure_1_headline.svg")  # fmt: skip  # ty: ignore[unresolved-attribute]


def _lines(
    *,
    x: str,
    x_title: str,
    values: list[float] | None,
    t: pl.DataFrame,
    title: list[str],
    name: str,
) -> None:
    def build(d: pl.DataFrame) -> alt.LayerChart:
        axis = alt.Axis(values=values) if values else alt.Axis()
        base = alt.Chart(d).encode(
            x=alt.X(f"{x}:Q", title=x_title, axis=axis),
            color=alt.Color("farm:N", title=None, scale=alt.Scale(domain=FARMS, range=FARM_COLOURS), legend=alt.Legend(orient="top", labelLimit=500)),
        )  # fmt: skip
        line = base.mark_line(point=True, strokeWidth=2).encode(
            y=alt.Y("recall:Q", title="Share of steps found", scale=alt.Scale(domain=[0, 1]))
        )
        return line + BAR

    stacked(build, t, title).save(DOCS / name)  # ty: ignore[unresolved-attribute]


def fig_size() -> None:
    """Recall of the slope against injected step size at 3 months, one panel per world."""
    t = (
        _recall()
        .filter((pl.col("est") == EST_NAMES["slope"]) & (pl.col("k") == 3))
        .with_columns(size=pl.col("loss") * 100)
    )
    _lines(x="size", x_title="Injected capacity loss (% of capacity)", values=None, t=t, name="figure_2_recall_by_size.svg", title=["Solar C needs a loss of about 6% and Solar F about 9% to be found within 3 months", "Slope estimator. Each point: share of 20 start months with the step found. Dashed: 80% pass bar."])  # fmt: skip


def fig_horizon() -> None:
    """Recall of the slope for a 10% loss against the number of months after the step."""
    t = _recall().filter((pl.col("est") == EST_NAMES["slope"]) & (pl.col("loss") == 0.10))
    _lines(x="k", x_title="Months after the step", values=[1, 3, 6, 12], t=t, name="figure_3_recall_by_horizon.svg", title=["Waiting 12 months does not bring Solar A, B, or D to 80%", "Slope estimator, 10% loss. Each point: share of 20 start months with the step found. Dashed: 80% pass bar."])  # fmt: skip


def fig_false() -> None:
    """False-alarm rates: curtailment placebo, and cloudy against ordinary months."""
    cur = (
        pl.read_csv(c.DATA_DIR / "curtail.csv")
        .filter(pl.col("farm") == "all")
        .with_columns(est=pl.col("est").replace(EST_NAMES))
    )
    base = alt.Chart(cur).encode(
        x=alt.X("k:Q", title="Months after the curtailed month", axis=alt.Axis(values=[1, 3, 6, 12])),
        color=alt.Color("est:N", title=None, scale=alt.Scale(domain=list(EST_NAMES.values()), range=list(EST_COLOURS.values())), legend=alt.Legend(orient="top", labelLimit=500, columns=1)),
    )  # fmt: skip
    lines = base.mark_line(point=True, strokeWidth=2).encode(
        y=alt.Y("rate:Q", title="False-alarm rate", scale=alt.Scale(domain=[0, 0.2]))
    )
    band = base.mark_errorbar(thickness=2).encode(
        y=alt.Y("lo:Q", title="False-alarm rate"), y2="hi:Q"
    )
    rule = (
        alt.Chart(pl.DataFrame({"y": [0.05]}))
        .mark_rule(strokeDash=[5, 4], color="#000000")
        .encode(y="y:Q")
    )
    p1 = (band + lines + rule).properties(width=WIDTH - 90, height=HEIGHT, title=alt.TitleParams("Zeroing 5% of one month's half-hours: share of 300 cases flagged as a fall", fontSize=12, anchor="start"))  # fmt: skip
    cl = pl.read_csv(c.DATA_DIR / "false_step.csv").filter(pl.col("est") == "slope").with_columns(
        months_kind=pl.col("group").replace({"standard": "Other months (20 starts a farm)", "cloudy3": "Three cloudiest months (3 a farm)", "cloudy10": "Ten cloudiest months (10 a farm)"})
    )  # fmt: skip
    b2 = alt.Chart(cl).encode(x=alt.X("k:Q", title="Months after the start month", axis=alt.Axis(values=[1, 3, 6, 12])), color=alt.Color("months_kind:N", title=None, scale=alt.Scale(range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_PURPLE]), legend=alt.Legend(orient="top", labelLimit=500, columns=1)))  # fmt: skip
    l2 = b2.mark_line(point=True, strokeWidth=2).encode(
        y=alt.Y("rate:Q", title="False-step rate", scale=alt.Scale(domain=[0, 0.35]))
    )
    e2 = b2.mark_errorbar(thickness=2).encode(y=alt.Y("lo:Q", title="False-step rate"), y2="hi:Q")
    p2 = (e2 + l2 + rule).properties(width=WIDTH - 90, height=HEIGHT, title=alt.TitleParams("Slope estimator, uninjected series: false steps in cloudy and other months", fontSize=12, anchor="start"))  # fmt: skip
    alt.vconcat(p1, p2).resolve_scale(color="independent").properties(title=alt.TitleParams(["Zeroing 5% of a month's half-hours is flagged as a fall twice as often as the 5% the threshold allows", "Line: rate. Bars: 95% Wilson interval. Dashed: 5%. Cloudy months: 15 cases, so the interval is wide."], anchor="start", fontSize=13, subtitleFontSize=11)).save(DOCS / "figure_4_false_alarms.svg")  # fmt: skip  # ty: ignore[unresolved-attribute]


if __name__ == "__main__":
    for fn in (fig_headline, fig_size, fig_horizon, fig_false):
        fn()
