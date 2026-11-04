"""Draw experiment 546's figure from the scored first two folds (private `scores_first2/absolute.csv`)."""

from pathlib import Path

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SCORES = ML_EXPERIMENTS_DATA_DIR / "00546-teacher-student-distillation-into-lag-free-demand-network" / "scores_first2"
OUT = Path(__file__).resolve().parents[2] / "docs/ml-experiments/00546-teacher-student-distillation-into-lag-free-demand-network"
BANDS = ["0-6h", "6-12h", "12-24h", "24-48h", "48-72h", "72-120h", "120-240h"]
ARMS = ["L-1d", "S-DW", "R-DW"]
LABELS = {"L-1d": "L-1d (lag-free)", "S-DW": "Student (lag-free, distilled)", "R-DW": "Teacher (day-ago and week-ago lags)"}

absolute = pl.read_csv(SCORES / "absolute.csv").filter((pl.col("set") == "eight") & pl.col("band").is_in(BANDS))
pool = absolute.filter(pl.col("arm") == "A-pool").select("band", pool="crps")
data = (
    absolute.filter(pl.col("arm").is_in(ARMS))
    .join(pool, on="band")
    .with_columns(gap=pl.col("crps") - pl.col("pool"), arm=pl.col("arm").replace(LABELS))
)
chart = (
    alt.Chart(data.to_pandas())
    .mark_line(point=True)
    .encode(
        x=alt.X("band:N", sort=BANDS, title="Lead time"),
        y=alt.Y("gap:Q", title="CRPS minus A-pool (points of capacity; lower is better)"),
        color=alt.Color("arm:N", title=None, scale=alt.Scale(range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_GREEN])),
    )
    .properties(width=480, height=280, title="The eight MVA-only Primaries, folds 2025-10 and 2025-12")
)
chart.save(OUT / "figure_1_gap_to_a_pool.svg")
