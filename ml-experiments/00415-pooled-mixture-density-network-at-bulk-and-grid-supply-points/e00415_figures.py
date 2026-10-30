# ruff: noqa: E501
"""Draw the figures of experiment 415 from its stored scores and forecasts.

Figure 1: CRPS at days 1 to 10 per node and arm. Figures 2 and 3: a day-ahead forecast and the
14-day horizon of one run of arm N1 (seed 0) at one BSP and one GSP, with the median, the 50% and
90% bands, and the actuals. Nodes are substations, so no anonymisation is needed.
"""

from pathlib import Path
from typing import Final

import altair as alt
import e00415_prep as prep  # ty: ignore[unresolved-import]
import polars as pl
from plotting import ocf_theme as theme

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00415-pooled-mixture-density-network-at-bulk-and-grid-supply-points"
)
ARM_ORDER: Final[list[str]] = [
    "N1",
    "N2",
    "N2rw",
    "bg2",
    "xgb_control_w",
    "stack",
    "manual_heuristic",
    "climatology",
]
ARM_NAMES: Final[dict[str, str]] = {
    "N1": "N1 (network, nodes added)",
    "N2": "N2 (+ Primaries beneath)",
    "N2rw": "N2rw (+ random Primaries)",
    "bg2": "BG2 (weekly XGBoost)",
    "xgb_control_w": "Monthly XGBoost, widened",
    "stack": "GAM stack",
    "manual_heuristic": "Manual heuristic",
    "climatology": "Climatology",
}
EXAMPLES: Final[tuple[tuple[str, str, str], ...]] = (
    ("D13", "2026-01", "2026-01-15"),
    ("D1", "2026-06", "2026-06-15"),
)


def figure_one() -> None:
    """Draw CRPS per node and arm."""
    scores = pl.read_parquet(prep.WORK_DIR / "scores" / "node_scores.parquet")
    data = (
        scores.filter(
            pl.col("metric") == "crps",
            pl.col("slice") == "lead_days_1_10",
            pl.col("arm").is_in(ARM_ORDER),
        )
        .with_columns(arm_name=pl.col("arm").replace_strict(ARM_NAMES))
        .select("label", "arm_name", "value")
        .to_pandas()
    )
    order = [ARM_NAMES[a] for a in ARM_ORDER]
    colours = [theme.BRAND_ORANGE, theme.DATA_BLUE, theme.DATA_PURPLE, "#1A1A1A", "#666666"]
    chart = (
        alt.Chart(data)
        .mark_point(filled=True, size=90)
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("arm_name:N", sort=order, title=None),
            x=alt.X("value:Q", title="CRPS, days 1 to 10 (points of capacity; lower is better)"),
            color=alt.Color(
                "arm_name:N",
                sort=order,
                scale=alt.Scale(domain=order, range=[*colours, *["#999999"] * 3]),
                legend=None,
            ),
            column=alt.Column("label:N", title=None, sort=["D13", "D17", "D1", "D6"]),
        )
        .properties(width=150, height=220, title="CRPS at four nodes, four test months (interim)")
    )
    chart.save(DOCS_DIR / "figure_1_crps_per_node.svg")


def figure_example(*, node: str, month: str, run_day: str, horizon_hours: float, name: str) -> None:
    """Draw one run's forecast of N1 at a node, out to `horizon_hours`."""
    series = prep.series20()
    sid = int(series.filter(pl.col("site") == node)["time_series_id"][0])
    cap = float(series.filter(pl.col("site") == node)["effective_capacity_mw"][0])
    frame = pl.read_parquet(prep.WORK_DIR / "seed_preds" / "N1" / f"{month}_s0.parquet").filter(
        pl.col("time_series_id") == sid
    )
    origins = frame["power_fcst_init_time"].dt.date().unique().sort().to_list()
    day = next(d for d in origins if str(d) >= run_day)
    one = frame.filter(pl.col("power_fcst_init_time").dt.date() == day).with_columns(
        hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_seconds() / 3600
    )
    one = one.filter(pl.col("hours") <= horizon_hours).sort("valid_time")
    if name.endswith("day_ahead"):
        one = one.filter(pl.col("hours") >= 3.5).filter(pl.col("hours") <= 36)
    pdf = one.select(
        "valid_time",
        *[(pl.col(c) * cap).alias(c) for c in ("q05", "q25", "q50", "q75", "q95", "actual")],
    ).to_pandas()
    base = alt.Chart(pdf).encode(x=alt.X("valid_time:T", title="Valid time (UTC)"))
    band90 = base.mark_area(opacity=0.2, color=theme.DATA_BLUE).encode(
        y=alt.Y("q05:Q", title="Demand (MW)"), y2="q95:Q"
    )
    band50 = base.mark_area(opacity=0.35, color=theme.DATA_BLUE).encode(y="q25:Q", y2="q75:Q")
    median = base.mark_line(color=theme.DATA_BLUE).encode(y="q50:Q")
    actual = base.mark_line(color=theme.BRAND_ORANGE).encode(y="actual:Q")
    title = f"{node}: forecast from {day} (N1, seed 0). Blue: median, 50% and 90% bands. Orange: actual."
    chart = (band90 + band50 + median + actual).properties(width=560, height=240, title=title)
    chart.save(DOCS_DIR / f"{name}_{node}.svg")


def main() -> None:
    """Draw every figure."""
    figure_one()
    numbers = {("D13", 36): 2, ("D1", 36): 3, ("D13", 336): 4, ("D1", 336): 5}
    for node, month, run_day in EXAMPLES:
        for hours, tag in ((36, "day_ahead"), (336, "14_day")):
            figure_example(
                node=node,
                month=month,
                run_day=run_day,
                horizon_hours=hours,
                name=f"figure_{numbers[(node, hours)]}_{tag}",
            )


if __name__ == "__main__":
    main()
