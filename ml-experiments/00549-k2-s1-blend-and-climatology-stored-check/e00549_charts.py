"""Draw experiment 549's figure: CRPS minus A-pool at the eight for R-DW, K2-S1, and the blend."""

from pathlib import Path

import altair as alt
import e00521_stats as s  # ty: ignore[unresolved-import]
import plotting.ocf_theme as ocf
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ID = "00549-k2-s1-blend-and-climatology-stored-check"
OUT = Path(__file__).resolve().parents[2] / "docs/ml-experiments" / ID
BANDS = list(s.BANDS)
ARMS = ["R-DW", "K2-S1", "blend"]

absolute = pl.read_csv(ML_EXPERIMENTS_DATA_DIR / ID / "scores" / "absolute.csv").filter(
    (pl.col("set") == "eight") & pl.col("band").is_in(BANDS) & pl.col("crps").is_not_null()
)
pool = absolute.filter(pl.col("arm") == "A-pool").select("band", pool="crps")
data = (
    absolute.filter(pl.col("arm").is_in(ARMS))
    .join(pool, on="band")
    .with_columns(gap=pl.col("crps") - pl.col("pool"))
)
chart = (
    alt.Chart(data.to_pandas())  # ty: ignore[unresolved-attribute]
    .mark_line(point=True)
    .encode(
        x=alt.X("band:N", sort=BANDS, title="Lead time", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("gap:Q", title="CRPS minus A-pool (points of capacity; lower is better)"),
        color=alt.Color(
            "arm:N",
            title=None,
            scale=alt.Scale(domain=ARMS, range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_GREEN]),
        ),
    )
    .properties(
        width=620,
        height=260,
        title=["Eight MVA-only Primaries, folds 2025-12 to 2026-06.", "Zero is A-pool."],
    )
)
chart.save(OUT / "figure_1_gap_to_a_pool.svg")
