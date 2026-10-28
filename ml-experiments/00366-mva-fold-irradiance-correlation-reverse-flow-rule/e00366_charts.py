"""Draw the two figures of experiment 366 from `work/rule.parquet` and `work/census.parquet`.

uv run --no-sync python ml-experiments/00366-*/e00366_charts.py
"""

from pathlib import Path
from typing import Any, Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "mva-fold-irradiance-correlation-reverse-flow-rule"
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"00366-{SLUG}" / "work"
DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / f"00366-{SLUG}"
)
CORRELATION_THRESHOLD: Final[float] = 0.3
PERCENTILE_THRESHOLD_POINTS: Final[float] = 1.0
CASES: Final[dict[float, str]] = {
    0.0: "no injection",
    1.0: "injected, level 1.0",
    2.0: "injected, level 2.0",
}
CENSUS_LABEL: Final[str] = "MVA-only Primary, as metered"
SITES: Final[list[str]] = [
    "D5",
    "D10",
    "D11",
    "D14",
    "D16",
    "D2",
    "D3",
    "D4",
    "D8",
    "D12",
    "D19",
    "D21",
    "D22",
]


def _chart(data: Any) -> Any:
    """Return an Altair chart typed loosely, because `ty` loses the chart type after `mark_*()`."""
    return alt.Chart(data)


def _figure(*, column: str, title: str, threshold: float, path: Path) -> None:
    """Draw one statistic by substation and case, with its threshold as a dashed line."""
    folded = (
        pl.read_parquet(WORK / "rule.parquet")
        .filter(pl.col("cut") == "scored", pl.col("reactive") == "Q0")
        .select("site", case=pl.col("level").replace_strict(CASES), value=pl.col(column))
    )
    census = pl.read_parquet(WORK / "census.parquet").select(
        "site", case=pl.lit(CENSUS_LABEL), value=pl.col(column)
    )
    data = pl.concat([folded, census]).to_pandas()
    points = (
        _chart(data)
        .mark_circle(size=90)
        .encode(
            x=alt.X("site:N", sort=SITES, title="Primary substation", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("value:Q", title=title),
            color=alt.Color(
                "case:N",
                scale=alt.Scale(
                    domain=[*CASES.values(), CENSUS_LABEL],
                    range=[ocf.DATA_BLUE_LIGHT, ocf.DATA_AMBER, ocf.BRAND_ORANGE, ocf.DATA_BLUE],
                ),
                legend=alt.Legend(title=None, orient="top", direction="vertical"),
            ),
        )
    )
    rule = (
        _chart(pl.DataFrame({"y": [threshold]}).to_pandas())
        .mark_rule(strokeDash=[4, 4])
        .encode(y="y:Q")
    )
    (points + rule).properties(width=560, height=260).save(path)


def _reactive_figure(*, path: Path) -> None:
    """Draw the correlation against the 1st percentile at the 5 scored substations, per setting."""
    frame = pl.read_parquet(WORK / "rule.parquet").filter(pl.col("cut") == "scored")
    panels = []
    for reactive, setting in (("Q0", "reactive power zero"), ("Q1", "power factor 0.97")):
        data = (
            frame.filter(pl.col("reactive") == reactive)
            .select(
                case=pl.col("level").replace_strict(CASES),
                correlation=pl.col("correlation"),
                lowest=pl.col("p1_training_sunny"),
            )
            .to_pandas()
        )
        points = (
            _chart(data)
            .mark_circle(size=90)
            .encode(
                x=alt.X("correlation:Q", title="Correlation with irradiance"),
                y=alt.Y("lowest:Q", title="1st percentile of readings (% of capacity)"),
                color=alt.Color(
                    "case:N",
                    scale=alt.Scale(
                        domain=list(CASES.values()),
                        range=[ocf.DATA_BLUE_LIGHT, ocf.DATA_AMBER, ocf.BRAND_ORANGE],
                    ),
                    legend=alt.Legend(title=None, orient="top", direction="vertical"),
                ),
            )
        )
        vertical = (
            _chart(pl.DataFrame({"x": [CORRELATION_THRESHOLD]}).to_pandas())
            .mark_rule(strokeDash=[4, 4])
            .encode(x="x:Q")
        )
        horizontal = (
            _chart(pl.DataFrame({"y": [PERCENTILE_THRESHOLD_POINTS]}).to_pandas())
            .mark_rule(strokeDash=[4, 4])
            .encode(y="y:Q")
        )
        panels.append(
            (points + vertical + horizontal).properties(width=260, height=260, title=setting)
        )
    alt.hconcat(*panels).save(path)


def main() -> None:
    """Draw the correlation figure, the 1st-percentile figure, and the per-setting figure."""
    _figure(
        column="correlation",
        title="Correlation of training sunny-midday readings with irradiance",
        threshold=CORRELATION_THRESHOLD,
        path=DOCS_DIR / "figure_1_correlation_by_substation.svg",
    )
    _figure(
        column="p1_training_sunny",
        title="1st percentile of training sunny-midday readings (% of capacity)",
        threshold=PERCENTILE_THRESHOLD_POINTS,
        path=DOCS_DIR / "figure_2_lowest_readings_by_substation.svg",
    )
    _reactive_figure(path=DOCS_DIR / "figure_3_statistics_by_reactive_setting.svg")


if __name__ == "__main__":
    main()
