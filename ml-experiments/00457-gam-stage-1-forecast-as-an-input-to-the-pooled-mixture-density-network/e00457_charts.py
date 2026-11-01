# ruff: noqa: E501
"""Draw experiment 457's figures as SVG from the score tables and the stored forecasts.

    source /tmp/e457_env.sh
    uv run --no-sync python ml-experiments/00457-*/e00457_charts.py

Optimise afterwards with `npx svgo@4 --multipass --precision=1 --final-newline f.svg`.
"""

from pathlib import Path
from typing import Final

import altair as alt
import e00397_prep as base  # ty: ignore[unresolved-import]
import e00425_fit as fit425  # ty: ignore[unresolved-import]
import e00452_run as run  # ty: ignore[unresolved-import]
import e00457_fit as fit457  # ty: ignore[unresolved-import]
import plotting.ocf_theme as ocf
import polars as pl

DOCS: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00457-gam-stage-1-forecast-as-an-input-to-the-pooled-mixture-density-network"
)
SCORES: Final[Path] = fit457.WORK_DIR / "scores"
LABELS: Final[dict[str, str]] = {
    "Gin-avg5 - A-avg5": "GAM as input minus arm A",
    "Gbase-avg5 - A-avg5": "GAM as base minus arm A",
    "Gin-avg5 - Gbase-avg5": "GAM as input minus GAM as base",
}


def contrasts() -> None:
    """Write figure 1: paired CRPS contrasts with Bonferroni intervals, gating row sets."""
    frame = (
        pl.read_csv(SCORES / "contrasts.csv")
        .filter(
            (pl.col("metric") == "crps") & pl.col("contrast").is_in(list(LABELS))
            & pl.col("group").is_in(["MVA-only eight", "signed-MW six"])
        )
        .with_columns(setting=pl.col("contrast").replace(LABELS))
    )  # fmt: skip
    b = alt.Chart(frame).encode(y=alt.Y("setting:N", sort=list(LABELS.values()), title=None))
    layer = (
        b.mark_rule(color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "low_bonferroni:Q", title="CRPS difference (points of capacity; negative is better)"
            ),
            x2="high_bonferroni:Q",
        )
        + b.mark_point(color=ocf.DATA_BLUE, filled=True, size=60).encode(x="difference:Q")  # ty: ignore[unresolved-attribute]
        + alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(strokeDash=[4, 3]).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    faceted = layer.properties(width=260).facet(
        column=alt.Column("group:N", title=None), row=alt.Row("row_set:N", title=None), data=frame
    )
    faceted.properties(
        title="Giving arm A the GAM forecast as an input lowers CRPS; the additive base raises it"
    ).save(DOCS / "figure_1_contrasts.svg")


def examples(
    *, month: str, label: str, origin_index: int, hours: float, name: str, title: str
) -> None:
    """Write an example forecast of `Gin` (seed 0) with arm A's median, for one Primary and origin."""
    primaries = base.primaries()
    row = primaries.filter(pl.col("site") == label).row(0, named=True)
    actual = (
        pl.read_parquet(run.WORK_DIR / "predictions/PT1" / f"{month}_s0_w0.parquet")
        .filter(pl.col("primary") == row["primary"])
        .select(
            pl.from_epoch("origin_us", time_unit="us").dt.replace_time_zone("UTC").alias("power_fcst_init_time"),
            pl.from_epoch("valid_us", time_unit="us").dt.replace_time_zone("UTC").alias("valid_time"),
            "actual", "lead",
        )
    )  # fmt: skip
    gin = pl.read_parquet(fit457.OUT_DIR / "seed_preds/Gin" / f"{month}_s0.parquet").filter(
        pl.col("time_series_id") == row["time_series_id"]
    )
    a = pl.read_parquet(fit425.OUT_DIR / "seed_preds/A" / f"{month}_s0.parquet").filter(
        pl.col("time_series_id") == row["time_series_id"]
    ).select("power_fcst_init_time", "valid_time", pl.col("q50").alias("a_q50"))  # fmt: skip
    origin = gin["power_fcst_init_time"].unique().sort()[origin_index]
    part = (
        gin.filter(pl.col("power_fcst_init_time") == origin)
        .join(actual, on=["power_fcst_init_time", "valid_time"], how="left")
        .join(a, on=["power_fcst_init_time", "valid_time"], how="left")
        .filter(pl.col("lead") <= hours)
    )

    def band(lo: str, hi: str, opacity: float) -> alt.Chart:
        return (
            alt.Chart(part)
            .mark_area(opacity=opacity, color=ocf.DATA_BLUE)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("valid_time:T", title=None),
                y=alt.Y(f"{lo}:Q", title="Demand (fraction of capacity)"),
                y2=f"{hi}:Q",
            )
        )

    median = alt.Chart(part).mark_line(color=ocf.DATA_BLUE).encode(x="valid_time:T", y="q50:Q")  # ty: ignore[unresolved-attribute]
    arm_a = (
        alt.Chart(part)
        .encode(x="valid_time:T", y="a_q50:Q")
        .mark_line(color="grey", strokeDash=[2, 2])
    )  # ty: ignore[unresolved-attribute]
    truth = (
        alt.Chart(part)
        .encode(x="valid_time:T", y="actual:Q")
        .mark_line(color="black", strokeDash=[3, 2])
    )  # ty: ignore[unresolved-attribute]
    (band("q05", "q95", 0.2) + band("q25", "q75", 0.4) + median + arm_a + truth).properties(
        width=480, title=title
    ).save(DOCS / name)


def main() -> None:
    """Write every figure."""
    contrasts()
    for month, season in (("2026-01", "winter"), ("2026-06", "summer")):
        examples(month=month, label="D5", origin_index=15, hours=48.0, name=f"figure_{season}_day_ahead.svg",
                 title=f"Day-ahead forecast, {season}, D5 (GAM as input; grey dashed: arm A median; black dashed: actual)")  # fmt: skip
        examples(month=month, label="D5", origin_index=15, hours=240.0, name=f"figure_{season}_ten_day.svg",
                 title=f"Ten-day forecast from one origin, {season}, D5 (bands widen with lead)")  # fmt: skip


if __name__ == "__main__":
    main()
