# ruff: noqa: E501
"""Draw experiment 00452's figures as SVG from the stored tables and forecasts.

    source /tmp/e452_env.sh
    uv run --no-sync python ml-experiments/00452-*/e00452_charts.py

Optimise afterwards with `npx svgo@4 --multipass --precision=1 --final-newline f.svg`.
"""

from pathlib import Path
from typing import Final

import altair as alt
import e00452_lagless as lagless  # ty: ignore[unresolved-import]
import plotting.ocf_theme as ocf
import polars as pl

DOCS: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00452-pretraining-cap-and-surprise-inputs-on-the-425-arm-a-recipe"
)
TABLES: Final[Path] = lagless.WORK_DIR / "figure_tables"
SHOWN: Final[tuple[str, ...]] = (
    "AS1L-avg5 - PT1-avg5", "AL-avg5 - PT1-avg5", "AS1L-avg5 - AL-avg5",
    "AS1L-avg5 - 425-A-avg5", "AL-avg5 - 425-A-avg5", "PT1-avg5 - 425-A-avg5",
)  # fmt: skip
LABELS: Final[dict[str, str]] = {
    "AS1L-avg5 - PT1-avg5": "A-S1-long minus A-S1 (longer cap)",
    "AL-avg5 - PT1-avg5": "A-long minus A-S1 (cap, no surprise inputs)",
    "AS1L-avg5 - AL-avg5": "A-S1-long minus A-long (surprise inputs)",
    "AS1L-avg5 - 425-A-avg5": "A-S1-long minus A-ref (loop confounded)",
    "AL-avg5 - 425-A-avg5": "A-long minus A-ref (loop confounded)",
    "PT1-avg5 - 425-A-avg5": "A-S1 minus A-ref (loop confounded)",
}  # fmt: skip


def contrasts() -> None:
    """Write figure 1: the paired contrasts with Bonferroni intervals at the eight and the six."""
    frame = (
        pl.read_csv(TABLES / "paired_contrasts.csv")
        .filter(
            (pl.col("folds") == "four folds") & (pl.col("slice") == "days 1 to 10")
            & pl.col("contrast").is_in(SHOWN)
            & pl.col("group").is_in(["MVA-only eight", "signed-MW six"])
        )
        .with_columns(setting=pl.col("contrast").replace(LABELS))
    )  # fmt: skip
    base = alt.Chart(frame).encode(y=alt.Y("setting:N", sort=list(LABELS.values()), title=None))
    layer = (
        base.mark_rule(color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "low_bonferroni:Q", title="CRPS difference (points of capacity; negative is better)"
            ),
            x2="high_bonferroni:Q",
        )
        + base.mark_point(color=ocf.DATA_BLUE, filled=True, size=60).encode(x="difference:Q")  # ty: ignore[unresolved-attribute]
        + alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(strokeDash=[4, 3]).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    faceted = layer.properties(width=300).facet(
        column=alt.Column("group:N", title=None), data=frame
    )
    chart = faceted.properties(
        title="The longer cap and the surprise inputs move CRPS by less than 0.06 points at the eight"
    )  # fmt: skip
    chart.save(DOCS / "figure_1_contrasts.svg")


def example(
    *, month: str, label_primary: int, origin_index: int, hours: float, name: str, title: str
) -> None:
    """Write an example forecast of `AS1L-avg5` for one Primary and one forecast origin."""
    rows = pl.read_parquet(
        lagless.WORK_DIR / "predictions/AS1L-avg5" / f"{month}_s0_w0.parquet"
    ).filter(pl.col("primary") == label_primary)
    origins = rows["origin_us"].unique().sort()
    origin = origins[origin_index]
    part = rows.filter((pl.col("origin_us") == origin) & (pl.col("lead") <= hours)).with_columns(
        time=pl.from_epoch("valid_us", time_unit="us")
    )
    band = lambda lo, hi, opacity: alt.Chart(part).mark_area(opacity=opacity, color=ocf.DATA_BLUE).encode(  # noqa: E731  # ty: ignore[unresolved-attribute]
        x=alt.X("time:T", title=None), y=alt.Y(f"{lo}:Q", title="Demand (fraction of capacity)"), y2=f"{hi}:Q"
    )  # fmt: skip
    line = alt.Chart(part).mark_line(color=ocf.DATA_BLUE).encode(x="time:T", y="q50:Q")  # ty: ignore[unresolved-attribute]
    actual = (
        alt.Chart(part).mark_line(color="black", strokeDash=[3, 2]).encode(x="time:T", y="actual:Q")  # ty: ignore[unresolved-attribute]
    )
    (band("q05", "q95", 0.2) + band("q25", "q75", 0.4) + line + actual).properties(
        width=480, title=title
    ).save(DOCS / name)


def main() -> None:
    """Write every figure."""
    contrasts()
    example(month="2026-01", label_primary=0, origin_index=15, hours=48.0, name="figure_2_day_ahead.svg",
            title="Day-ahead forecast of A-S1-long (median, 50% and 90% bands; dashed black is the actual)")  # fmt: skip
    example(month="2026-01", label_primary=0, origin_index=15, hours=240.0, name="figure_3_ten_day.svg",
            title="Ten-day forecast of A-S1-long from one origin (bands widen with lead)")  # fmt: skip


if __name__ == "__main__":
    main()
