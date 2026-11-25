# ruff: noqa: E501
"""Experiment 729: the figures. Synthetic series in detail, real series as aggregates only.

No figure shows the metered engine record, an engine-head output on a real day, or a date of the
engine. Real-day panels carry no calendar dates. Every function takes plain dictionaries and arrays
that the scoring stage wrote, and returns an Altair chart; `save_svg` writes it and
`e00729_report.optimise_command` prints the `svgo` command to run before committing.

The `# ty: ignore[unresolved-attribute]` comments on `.encode(` lines are the workaround of the
`ty-workarounds` skill for an upstream `ty` bug with Altair's `mark_*()`.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import altair as alt
import e00729_core as core
import numpy as np
import polars as pl
from plotting import ocf_theme as theme

WIDTH_PX: Final = 620
BLUE: Final = theme.DATA_BLUE
ORANGE: Final = theme.BRAND_ORANGE
GREEN: Final = theme.DATA_GREEN
PURPLE: Final = theme.DATA_PURPLE
SKY: Final = theme.DATA_SKY
GREY: Final = "#555555"
STRIPE: Final = "#EDEDED"
ARM_COLOURS: Final = {
    "P0f": GREY, "B*": GREY, "A": SKY, "B": BLUE, "C": GREEN, "D": ORANGE, "H": PURPLE,
}  # fmt: skip
"""Colours of the arms that carry the main story; every other arm takes grey."""


def save_svg(*, chart: alt.TopLevelMixin, path: Path) -> None:
    """Write a chart as SVG (run `svgo` on the file before committing it)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    chart.save(str(path))


def dot_chart(
    *, scores: Mapping[str, Mapping[str, float]], rows: Sequence[str], title: str,
    x_title: str,
) -> alt.Chart:  # fmt: skip
    """Return staggered dots on striped rows: one dot per arm in each subset row.

    Args:
        scores: Metric by arm and then by subset name.
        rows: The subset names, top to bottom.
        title: The chart title (what is plotted, and which direction is better).
        x_title: The x-axis title with the unit.

    Returns:
        The chart.
    """
    arms = list(scores)
    lane = {a: (i - (len(arms) - 1) / 2) * 3.0 / 40.0 for i, a in enumerate(arms)}
    data = pl.DataFrame(
        [
            {"arm": a, "row": i + lane[a], "value": scores[a].get(r, float("nan"))}
            for a in arms
            for i, r in enumerate(rows)
        ]
    ).drop_nulls()
    label = "{" + ", ".join(f"{i}: '{r}'" for i, r in enumerate(rows)) + "}[datum.value]"
    stripes = pl.DataFrame({"top": [i - 0.5 for i in range(1, len(rows), 2)]}).with_columns(
        bottom=pl.col("top") + 1
    )
    y_scale = alt.Scale(domain=[len(rows) - 0.5, -0.5])
    axis = alt.Axis(values=list(range(len(rows))), labelExpr=label, ticks=False, title=None)
    shade = (
        alt.Chart(stripes)
        .mark_rect(color=STRIPE)
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("top:Q", scale=y_scale, axis=axis), y2="bottom:Q"
        )
    )
    colours = [ARM_COLOURS.get(a, "#999999") for a in arms]
    dots = (
        alt.Chart(data)
        .mark_circle(size=60)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("value:Q", title=x_title),
            y=alt.Y("row:Q", scale=y_scale, axis=axis),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=arms, range=colours),
                legend=alt.Legend(orient="top"),
            ),
        )
    )
    return (shade + dots).properties(
        width=WIDTH_PX, height=40 * len(rows), title=alt.TitleParams(title, anchor="start")
    )


def contrast_chart(
    *, readings: Mapping[str, tuple[float, float, float]], margin: float
) -> alt.Chart:
    """Return the 12 counted contrasts and the report-only C13 as points with intervals.

    The margin m is drawn as dashed rules.

    Args:
        readings: For each contrast name, (point, lower, upper) in points of capacity.
        margin: The margin m in points.

    Returns:
        The chart.
    """
    names = list(readings)
    data = pl.DataFrame(
        {
            "contrast": names,
            "point": [readings[n][0] for n in names],
            "lower": [readings[n][1] for n in names],
            "upper": [readings[n][2] for n in names],
        }
    )
    base = alt.Chart(data).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("contrast:N", sort=names, title=None)
    )
    bars = base.mark_rule(color=BLUE, strokeWidth=3).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "lower:Q", title="Difference in signed CRPS (points of capacity; negative is a gain)"
        ),
        x2="upper:Q",
    )
    dots = base.mark_circle(color=BLUE, size=60).encode(x="point:Q")  # ty: ignore[unresolved-attribute]
    rules = (
        alt.Chart(pl.DataFrame({"x": [0.0, -margin], "kind": ["no difference", "margin m"]}))
        .mark_rule(strokeDash=[4, 3], color=ORANGE)
        .encode(x="x:Q", tooltip="kind:N")  # ty: ignore[unresolved-attribute]
    )
    return (bars + dots + rules).properties(
        width=WIDTH_PX,
        height=22 * len(names),
        title=alt.TitleParams(
            "The 12 counted contrasts (and the report-only C13) at the Bonferroni level; the orange lines mark zero and minus m",
            anchor="start",
        ),
    )


def reliability_chart(*, mean_forecast: Sequence[float], frequency: Sequence[float]) -> alt.Chart:
    """Return the reliability diagram of P(X < 0): observed frequency against forecast, by bin."""
    data = pl.DataFrame({"forecast": mean_forecast, "observed": frequency}).drop_nulls()
    diagonal = (
        alt.Chart(pl.DataFrame({"a": [0.0, 1.0], "b": [0.0, 1.0]}))  # ty: ignore[unresolved-attribute]
        .mark_line(color=GREY, strokeDash=[4, 3])
        .encode(x="a:Q", y="b:Q")
    )  # ty: ignore[unresolved-attribute]
    dots = (
        alt.Chart(data)
        .mark_line(point=True, color=ORANGE)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "forecast:Q",
                title="Forecast probability of reverse flow",
                scale=alt.Scale(domain=[0, 1]),
            ),
            y=alt.Y(
                "observed:Q", title="Observed share of reverse flow", scale=alt.Scale(domain=[0, 1])
            ),
        )
    )
    return (diagonal + dots).properties(width=WIDTH_PX / 2, height=WIDTH_PX / 2)


def architecture_chart(*, scores: Mapping[str, float]) -> alt.Chart:
    """Return the validation-month signed CRPS of arm D on each architecture (smaller is better)."""
    data = pl.DataFrame({"architecture": list(scores), "crps": list(scores.values())})
    return (
        alt.Chart(data)
        .mark_bar(color=BLUE)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "crps:Q",
                title="Mean signed CRPS on the validation months (points; smaller is better)",
            ),
            y=alt.Y("architecture:N", title=None),
        )
        .properties(width=WIDTH_PX, height=26 * len(scores))
    )


def share_pair_chart(
    *, groups: Sequence[str], synthetic: Sequence[float], real: Sequence[float], title: str
) -> alt.Chart:
    """Return paired bars of synthetic and real shares (the realism gate, and the real eras)."""
    data = pl.DataFrame(
        {
            "group": [*groups, *groups],
            "source": ["synthetic"] * len(groups) + ["real"] * len(groups),
            "share": [*synthetic, *real],
        }
    )
    return (
        alt.Chart(data)
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("share:Q", title="Share (1.0 = all)"),
            y=alt.Y("source:N", title=None),
            row=alt.Row("group:N", title=None),
            color=alt.Color(
                "source:N",
                scale=alt.Scale(domain=["synthetic", "real"], range=[ORANGE, BLUE]),
                legend=alt.Legend(orient="top"),
            ),
        )
        .properties(width=WIDTH_PX - 120, height=40, title=alt.TitleParams(title, anchor="start"))
    )


def series_chart(*, series: Mapping[str, np.ndarray], title: str) -> alt.Chart:
    """Return lines of synthetic series over 96 half-hours (days 1 and 2), one per name."""
    n = len(next(iter(series.values())))
    data = pl.DataFrame(
        {
            "hour": np.tile(np.arange(n) / 2.0, len(series)),
            "points": np.concatenate([np.asarray(v) for v in series.values()]),
            "series": np.repeat(list(series), n),
        }
    )
    return (
        alt.Chart(data)
        .mark_line()
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "hour:Q",
                title="Hours from the start of day 1 (UTC)",
                axis=alt.Axis(values=[0, 12, 24, 36, 48, 60, 72, 84, 96]),
            ),
            y=alt.Y("points:Q", title="Points of capacity"),
            color=alt.Color("series:N", legend=alt.Legend(orient="top")),
        )
        .properties(width=WIDTH_PX, height=140, title=alt.TitleParams(title, anchor="start"))
    )


def band_chart(
    *, truth: np.ndarray, lower: np.ndarray, median: np.ndarray, upper: np.ndarray, p_neg: np.ndarray,
    title: str,
) -> alt.VConcatChart:  # fmt: skip
    """Return an example window: the 5% to 95% band and median against the truth, with a P(X<0) strip."""
    hour = np.arange(len(truth)) / 2.0
    data = pl.DataFrame(
        {
            "hour": hour,
            "truth": truth,
            "lower": lower,
            "median": median,
            "upper": upper,
            "p_neg": p_neg,
        }
    )
    x = alt.X("hour:Q", title="Hours from midnight UTC", axis=alt.Axis(values=[0, 6, 12, 18, 24]))
    band = (
        alt.Chart(data).mark_area(opacity=0.3, color=ORANGE).encode(x=x, y="lower:Q", y2="upper:Q")  # ty: ignore[unresolved-attribute]
    )  # ty: ignore[unresolved-attribute]
    med = (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_line(color=ORANGE)
        .encode(x=x, y=alt.Y("median:Q", title="Signed power (points)"))
    )  # ty: ignore[unresolved-attribute]
    tru = alt.Chart(data).mark_line(color=BLUE).encode(x=x, y="truth:Q")  # ty: ignore[unresolved-attribute]
    strip = (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_rect()
        .encode(
            x=x,
            color=alt.Color(
                "p_neg:Q", scale=alt.Scale(domain=[0, 1], range=["#FFFFFF", PURPLE]), title="P(X<0)"
            ),
        )  # ty: ignore[unresolved-attribute]
        .properties(width=WIDTH_PX, height=14)
    )
    top = (band + med + tru).properties(
        width=WIDTH_PX, height=150, title=alt.TitleParams(title, anchor="start")
    )
    return alt.vconcat(top, strip)


def optimise_command(*, path: Path) -> str:
    """Return the `svgo` command to run on a saved figure before committing it."""
    return f"npx svgo@4 --multipass --precision=1 --final-newline {path}"


def margin_note() -> float:
    """Return the margin floor, so the figure code names the one constant it reads from the plan."""
    return core.MARGIN_FLOOR_POINTS
