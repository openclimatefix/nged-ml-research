"""Draw experiment 414's gain of the counterfactual lags, one bar per scored event.

Reads the per-row scores written by `e00414_rescore.py` and writes the figure beside `result.md`.
"""

from pathlib import Path

import altair as alt
import e00414_report as report  # ty: ignore[unresolved-import]
import polars as pl
from plotting import ocf_theme as theme

OUT = Path(__file__).resolve().parents[2] / (
    "docs/ml-experiments/00414-post-hoc-gap-filled-counterfactual-lag-rescore"
)


def main() -> None:
    """Write the per-event gain figure."""
    frames = [
        pl.read_parquet(f).with_columns(pl.col("event_id").cast(pl.Int64))
        for f in sorted(report.ROWS.glob("*.parquet"))
    ]
    rows = pl.concat([f for f in frames if not f.is_empty()])
    per_event = (
        rows.group_by("site", "event_id")
        .agg(
            n=pl.len(),
            gain=(pl.col("crps_real") - pl.col("crps_cf")).mean(),
        )
        .with_columns(
            event=pl.col("site") + " event " + pl.col("event_id").cast(pl.String),
            group=pl.when(pl.col("site").is_in(report.SIX))
            .then(pl.lit("Six signed-MW Primaries"))
            .otherwise(pl.lit("Eight MVA-only Primaries")),
        )
        .sort("gain")
        .to_pandas()
    )
    chart = (
        alt.Chart(per_event)
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("event:N", title=None, sort=list(per_event["event"])),
            x=alt.X("gain:Q", title="CRPS with real lags minus CRPS with clean lags (points)"),
            color=alt.Color(
                "group:N",
                scale=alt.Scale(
                    domain=["Six signed-MW Primaries", "Eight MVA-only Primaries"],
                    range=[theme.DATA_BLUE, theme.BRAND_ORANGE],
                ),
                legend=alt.Legend(title=None, orient="top", direction="horizontal"),
            ),
            tooltip=["event", "n", "gain"],
        )
        .properties(width=420, height=260)
    )
    chart.save(OUT / "figure_1_gain_per_event.svg")


if __name__ == "__main__":
    main()
