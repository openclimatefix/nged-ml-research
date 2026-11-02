# ruff: noqa: E501
"""Experiment 509: figures and printed numbers for the write-up.

Reads `work/contrasts.csv`, `work/pit.csv`, `work/real_events.csv`, and `work/seed_mde.csv` and
writes SVG figures to the experiment's docs folder and the numbers the page quotes to stdout.

```bash
export DATA_PATH_INTERNAL=<private data folder>
PYTHONPATH=<this folder>:<repo>/src:<ml-experiments/00482 folder> uv run --no-sync python ml-experiments/00509-*/e00509_report.py
```
"""

from pathlib import Path
from typing import Final

import altair as alt
import e00482_svg_guard as guard  # ty: ignore[unresolved-import]
import e00509_paired as p  # ty: ignore[unresolved-import]
import plotting.ocf_theme as ocf
import polars as pl

DOCS: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00509-event-band-coverage-paired-injection-on-and-off"
)
WIDTH: Final[int] = 640
LABEL: Final[dict[str, str]] = {
    "LL": "Lagless network",
    "S1": "S1 (lagged)",
    "B-mix": "B-mix",
    "K2-S1": "K2-S1",
    "A-pool": "A-pool (never saw a shift)",
}
COLOURS: Final[dict[str, str]] = {
    "Lagless network": ocf.DATA_SKY,
    "S1 (lagged)": ocf.DATA_PURPLE,
    "B-mix": ocf.DATA_GREEN,
    "K2-S1": ocf.BRAND_ORANGE,
    "A-pool (never saw a shift)": ocf.DATA_MAGENTA,
}
ORDER: Final[list[str]] = list(COLOURS)


def save(*, chart: alt.TopLevelMixin, name: str) -> None:
    """Save a chart as SVG and fail if a mark is drawn in the background colour."""
    DOCS.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS / name)
    guard.assert_no_invisible_marks(svg_path=DOCS / name)


def difference_chart(*, c: pl.DataFrame, measure: str, title: str, axis: str, name: str) -> None:
    """Draw the injected-minus-injection-off difference by scenario with Bonferroni intervals."""
    d = c.filter(
        (pl.col("clusters") == "all")
        & (pl.col("cls") == "newest_in")
        & (pl.col("band") == "d1_10")
        & (pl.col("measure") == measure)
    ).with_columns(
        forecast=pl.col("arm").replace(LABEL), scenario_name=pl.col("scenario").cast(pl.String)
    )
    scale = 100.0 if measure == "width90" else 1.0
    d = d.with_columns(
        diff=pl.col("diff") * scale,
        lo=pl.col("lo_bonferroni") * scale,
        hi=pl.col("hi_bonferroni") * scale,
    )
    base = alt.Chart(d).encode(
        x=alt.X(
            "scenario_name:N",
            title="Scenario (8 is the genuine level change)",
            axis=alt.Axis(labelAngle=0),
        ),
        color=alt.Color(
            "forecast:N",
            scale=alt.Scale(domain=ORDER, range=[COLOURS[k] for k in ORDER]),
            legend=alt.Legend(orient="top", title=None),
        ),
        xOffset="forecast:N",
    )
    chart = (
        base.mark_rule().encode(y=alt.Y("lo:Q", title=axis), y2="hi:Q")
        + base.mark_point(filled=True, size=45).encode(y="diff:Q")
    ).properties(width=WIDTH, height=260, title=title)
    zero = alt.Chart(pl.DataFrame({"y": [0.0]})).mark_rule(color=ocf.GREY_3).encode(y="y:Q")
    save(chart=zero + chart, name=name)


def pit_chart(*, pit: pl.DataFrame, scenario: int, name: str) -> None:
    """Draw the PIT histograms of K2-S1 and A-pool, injected and injection-off."""
    d = pit.filter(
        (pl.col("scenario") == scenario)
        & (pl.col("cls") == "newest_in")
        & (pl.col("band") == "d1_10")
        & pl.col("arm").is_in(["K2-S1", "A-pool"])
    ).with_columns(
        forecast=pl.col("arm").replace(LABEL),
        variant=pl.col("variant").replace({"injected": "Injected", "clean": "Injection off"}),
        x=(pl.col("bin") + 0.5) / 10,
    )
    chart = (
        alt.Chart(d)
        .mark_bar()
        .encode(
            x=alt.X(
                "bin:O",
                title="Probability integral transform bin (flat = 0.1)",
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y("share:Q", title="Share of rows"),
            color=alt.Color(
                "variant:N",
                scale=alt.Scale(
                    domain=["Injection off", "Injected"], range=[ocf.DATA_SKY, ocf.BRAND_ORANGE]
                ),
                legend=alt.Legend(orient="top", title=None),
            ),
            xOffset="variant:N",
            row=alt.Row("forecast:N", title=None),
        )
        .properties(width=WIDTH - 60, height=130)
    )
    rule = (
        alt.Chart(pl.DataFrame({"y": [0.1]}))
        .mark_rule(color=ocf.BLACK_1, strokeDash=[4, 3])
        .encode(y="y:Q")
    )
    save(chart=alt.layer(chart.properties(), rule) if False else chart, name=name)


def factor_chart(*, c: pl.DataFrame, name: str) -> None:
    """Draw the widening factor for the upper tail, inside the event and on the same rows with no injection."""
    d = c.filter(
        (pl.col("clusters") == "all")
        & (pl.col("cls") == "newest_in")
        & (pl.col("band") == "d1_10")
        & (pl.col("measure") == "factor_q95")
    ).select("scenario", "arm", "injected", "clean")
    d = d.unpivot(
        index=["scenario", "arm"],
        on=["injected", "clean"],
        variable_name="variant",
        value_name="factor",
    ).with_columns(
        forecast=pl.col("arm").replace(LABEL),
        variant=pl.col("variant").replace({"injected": "Injected", "clean": "Injection off"}),
        scenario_name=pl.col("scenario").cast(pl.String),
    )
    chart = (
        alt.Chart(d)
        .mark_point(filled=True, size=55)
        .encode(
            x=alt.X("scenario_name:N", title="Scenario", axis=alt.Axis(labelAngle=0)),
            y=alt.Y(
                "factor:Q",
                title="Factor on the q95 distance from the median",
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color(
                "forecast:N",
                scale=alt.Scale(domain=ORDER, range=[COLOURS[k] for k in ORDER]),
                legend=alt.Legend(orient="top", title=None),
            ),
            shape=alt.Shape("variant:N", legend=alt.Legend(orient="top", title=None)),
            xOffset="forecast:N",
        )
        .properties(
            width=WIDTH,
            height=260,
            title=[
                "Widening factor that makes q95 coverage 0.95",
                "(circle: injected; triangle: injection off)",
            ],
        )
    )
    save(chart=chart, name=name)


def real_chart(*, r: pl.DataFrame, name: str) -> None:
    """Draw q95 coverage on rows with any lag in a labelled event and on rows with no label."""
    d = (
        r.filter(
            (pl.col("band") == "d1_10")
            & (pl.col("actual") == "nra")
            & (pl.col("measure") == "cov_q95")
            & (pl.col("group") != "report_only")
        )
        .unpivot(
            index=["group", "arm"],
            on=["event", "nolabel"],
            variable_name="rows",
            value_name="coverage",
        )
        .with_columns(
            forecast=pl.col("arm").replace(LABEL),
            rows=pl.col("rows").replace(
                {"event": "Any lag inside a labelled event", "nolabel": "No label"}
            ),
            group=pl.col("group").replace(
                {"signed_six": "Signed-MW six", "mva_eight": "MVA-only eight"}
            ),
        )
    )
    chart = (
        alt.Chart(d)
        .mark_bar()
        .encode(
            x=alt.X(
                "forecast:N", title=None, sort=ORDER, axis=alt.Axis(labelAngle=0, labelLimit=120)
            ),
            y=alt.Y(
                "coverage:Q",
                title="q95 coverage (nominal 0.95)",
                scale=alt.Scale(domain=[0.5, 1.0]),
            ),
            color=alt.Color(
                "rows:N",
                scale=alt.Scale(
                    domain=["No label", "Any lag inside a labelled event"],
                    range=[ocf.DATA_SKY, ocf.BRAND_ORANGE],
                ),
                legend=alt.Legend(orient="top", title=None),
            ),
            xOffset="rows:N",
            row=alt.Row("group:N", title=None),
        )
        .properties(width=WIDTH - 60, height=150)
    )
    save(chart=chart, name=name)


def main() -> None:
    """Draw every figure."""
    c = pl.read_csv(p.WORK / "contrasts.csv")
    pit = pl.read_csv(p.WORK / "pit.csv")
    r = pl.read_csv(p.WORK / "real_events.csv")
    difference_chart(
        c=c,
        measure="cov_q95",
        title=[
            "q95 coverage inside the event minus the same rows with the injection off",
            "(point: difference; line: Bonferroni 48 interval)",
        ],
        axis="Difference in q95 coverage",
        name="figure_1_q95_coverage_difference.svg",
    )
    difference_chart(
        c=c,
        measure="width90",
        title=["Width of the 90% interval, injected relative to injection off"],
        axis="Change in 90% band width (%)",
        name="figure_2_band_width_change.svg",
    )
    pit_chart(pit=pit, scenario=4, name="figure_3_pit_scenario_4.svg")
    factor_chart(c=c, name="figure_4_widening_factor.svg")
    real_chart(r=r, name="figure_5_real_events.svg")


if __name__ == "__main__":
    main()
