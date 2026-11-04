# ruff: noqa: E501, E402, PLR0915, PERF401, PT018, D107
"""Draw experiment 552's figures: CRPS against the ECMWF-only arm by fold, and by lead band."""

from pathlib import Path

import altair as alt
import e00552_score as sc  # ty: ignore[unresolved-import]
import plotting.ocf_theme as ocf
import polars as pl

OUT = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00552-icon-eu-four-runs-a-day-lag-free-demand"
)
EIGHT = "MVA-only eight"
HISTORY = {
    "2026-03": "March (19 days)",
    "2026-04": "April (50 days)",
    "2026-05": "May (80 days)",
    "2026-06": "June (111 days)",
}
ORDER = list(HISTORY.values())
LABELS = {"N1": "N1: ICON-EU only", "N2": "N2: ICON-EU and ECMWF"}

df = sc.per_row().filter(pl.col("group") == EIGHT)
cell = (
    df.group_by("arm", "month", "band")
    .agg(pl.col("crps").mean())
    .pivot(on="arm", index=["month", "band"], values="crps")
)
cell = cell.with_columns(N1=pl.col("N1") - pl.col("N0"), N2=pl.col("N2") - pl.col("N0"))
data = cell.unpivot(
    on=["N1", "N2"], index=["month", "band"], variable_name="arm", value_name="diff"
).with_columns(
    fold=pl.col("month").replace_strict(HISTORY), arm=pl.col("arm").replace_strict(LABELS)
)
bands = ["0-6 h", "6-12 h", "12-24 h", "24-48 h", "48-74 h"]
fig1 = (
    alt.Chart(data.to_pandas())  # ty: ignore[unresolved-attribute]
    .mark_line(point=True)
    .encode(
        x=alt.X(
            "band:N", sort=bands, title="Lead time from the issue time", axis=alt.Axis(labelAngle=0)
        ),
        y=alt.Y("diff:Q", title="CRPS minus N0 (points of capacity; lower is better)"),
        color=alt.Color(
            "fold:N",
            sort=ORDER,
            title="Fold (ICON-EU history for training)",
            scale=alt.Scale(
                range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_GREEN, ocf.DATA_PURPLE]
                if hasattr(ocf, "DATA_PURPLE")
                else None
            ),
        ),
        strokeDash=alt.StrokeDash("arm:N", title=None),
    )
    .properties(
        width=520,
        height=260,
        title=[
            "Eight MVA-only Primaries, all four issue hours.",
            "Zero is N0 (ECMWF ensemble mean only).",
        ],
    )
)
fig1.save(OUT / "figure_1_gap_to_n0_by_fold.svg")
absolute = (
    df.group_by("arm", "band")
    .agg(pl.col("crps").mean())
    .with_columns(arm=pl.col("arm").replace({"N0": "N0: ECMWF only", **LABELS}))
)
fig2 = (
    alt.Chart(absolute.to_pandas())  # ty: ignore[unresolved-attribute]
    .mark_line(point=True)
    .encode(
        x=alt.X(
            "band:N", sort=bands, title="Lead time from the issue time", axis=alt.Axis(labelAngle=0)
        ),
        y=alt.Y("crps:Q", title="CRPS (points of capacity; lower is better)"),
        color=alt.Color(
            "arm:N",
            title=None,
            scale=alt.Scale(range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_GREEN]),
        ),
    )
    .properties(width=520, height=260, title="Eight MVA-only Primaries, four folds pooled.")
)
fig2.save(OUT / "figure_2_crps_by_lead.svg")
