"""Draw experiment 383's headline figure from `work/scores/rows.parquet`.

One panel per measure (CRPS, 5% to 95% band coverage) and group on the contaminated-lag rows, one
point per arm. Writes `figure_1_contaminated_rows.svg` to the experiment's docs folder.
"""

from pathlib import Path
from typing import Final

import altair as alt
import polars as pl
from e00383_score import HEADLINE_BAND, OUT

DOCS: Final[Path] = Path("docs/ml-experiments/00383-load-shift-xgboost-oracle-first-part-1")
ARMS: Final[tuple[str, ...]] = ("d", "d_w", "c1", "c1_w", "c2", "c2_w")
LABELS: Final[dict[str, str]] = {
    "d": "d: weekly XGBoost",
    "d_w": "d + widening",
    "c1": "c1: lags nulled",
    "c1_w": "c1 + widening",
    "c2": "c2: lagless fallback",
    "c2_w": "c2 + widening",
}
COLOURS: Final[tuple[str, ...]] = ("#306BFF", "#10C5F7", "#FF4901", "#FC9700", "#B701FF", "#17E58F")


def main() -> None:
    """Write the figure."""
    rows = pl.read_parquet(OUT / "rows.parquet").filter(
        pl.col("arm").is_in(list(ARMS)),
        pl.col(HEADLINE_BAND),
        ~pl.col("event_rows"),
        pl.col("lag_in_event"),
        pl.col("group").is_in(["signed-MW", "MVA-only"]),
    )
    summary = (
        rows.group_by("group", "arm")
        .agg(
            crps=pl.col("crps").mean(),
            coverage=pl.col("band_90").mean(),
        )
        .with_columns(arm_label=pl.col("arm").replace(LABELS))
        .unpivot(
            index=["group", "arm", "arm_label"], on=["crps", "coverage"], variable_name="measure"
        )
        .with_columns(
            measure=pl.col("measure").replace(
                {
                    "crps": "CRPS (points of capacity; lower is better)",
                    "coverage": "Share of readings inside the 5% to 95% band (nominal 0.90)",
                }
            )
        )
    )
    chart = (
        alt.Chart(summary.to_pandas())
        .mark_circle(size=90)
        .encode(
            x=alt.X("value:Q", title=None, scale=alt.Scale(zero=False)),
            y=alt.Y("arm_label:N", sort=[LABELS[a] for a in ARMS], title=None),
            color=alt.Color(
                "arm_label:N",
                scale=alt.Scale(domain=[LABELS[a] for a in ARMS], range=list(COLOURS)),
                legend=None,
            ),
            row=alt.Row("group:N", title=None),
            column=alt.Column("measure:N", title=None),
        )
        .resolve_scale(x="independent")
        .properties(
            width=260, height=140, title="Rows whose power lags lie inside a load-shift event"
        )
    )
    chart.save(DOCS / "figure_1_contaminated_rows.svg")


if __name__ == "__main__":
    main()
