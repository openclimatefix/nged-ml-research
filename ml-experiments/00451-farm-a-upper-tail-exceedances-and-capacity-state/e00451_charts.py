"""Draw experiment 451's figures as SVG from `work/*.parquet` and a Farm A exceedance histogram.

    source ml-experiments/00451-*/env.sh
    uv run --no-sync python ml-experiments/00451-*/e00451_charts.py

Optimise afterwards with `npx svgo@4 --multipass --precision=1 --final-newline f.svg`.
"""

from pathlib import Path
from typing import Final

import altair as alt
import e00451_exceedances as exc  # ty: ignore[unresolved-import]
import plotting.ocf_theme as ocf
import polars as pl

DOCS: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00451-farm-a-upper-tail-exceedances-and-capacity-state"
)
FARM: Final[dict[str, str]] = {"W1": "Farm A", "W2": "Farm B", "W3": "Farm C", "all": "All farms"}
LIMIT: Final[pl.DataFrame] = pl.DataFrame({"x": [1.5]})


def main() -> None:
    """Write the three figures."""
    step1 = pl.read_parquet(exc.WORK_DIR / "step1.parquet").with_columns(
        farm=pl.col("site").replace(FARM)
    )
    order = list(step1["variant"].unique(maintain_order=True))
    points = (
        alt.Chart(step1)
        .mark_circle(size=70)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("above_q99_pct:Q", title="Share of observations above q99 (%)"),
            y=alt.Y("variant:N", sort=order, title=None, axis=alt.Axis(labelLimit=330)),
            color=alt.Color("farm:N", title=None),
        )
    )
    rule = alt.Chart(LIMIT).mark_rule(strokeDash=[4, 3]).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    (points + rule).properties(
        width=380, title="Raising the upper clip brings Farm A under the 1.5% limit"
    ).save(DOCS / "figure_1_variants.svg")

    month = pl.read_parquet(exc.WORK_DIR / "step2_month.parquet")
    alt.Chart(month).mark_bar(color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("month:N", title=None, axis=alt.Axis(labelAngle=-45)),
        y=alt.Y("pct:Q", title="Share above q99 (%)"),
    ).properties(width=440, title="Farm A's exceedances cluster in a few months").save(
        DOCS / "figure_2_by_month.svg"
    )

    wind = pl.read_parquet(exc.WORK_DIR / "step2_wind_bin.parquet")
    alt.Chart(wind).mark_bar(color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "wind_bin:N",
            sort=["(-inf, 4]", "(4, 8]", "(8, 12]", "(12, 16]", "(16, inf]"],
            title="Ensemble-mean 100 m wind speed (m/s)",
        ),
        y=alt.Y("pct:Q", title="Share above q99 (%)"),
    ).properties(width=380, title="Farm A's exceedances sit in strong wind").save(
        DOCS / "figure_3_by_wind_speed.svg"
    )  # fmt: skip

    frame = exc.recalibrated(frame=exc.frame_with_extras(), extra=())
    farm_a = frame.filter(
        (pl.col("site") == "W1") & (pl.col("lead_hours") >= 24) & (pl.col("actual") > pl.col("q99"))
    )
    alt.Chart(farm_a.select("actual")).mark_bar(color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("actual:Q", bin=alt.Bin(step=0.02), title="Power over effective capacity"),
        y=alt.Y("count()", title="Farm A half-hours above q99"),
    ).properties(width=380, title="Most exceedances are at or just above full output").save(
        DOCS / "figure_4_exceedance_output.svg"
    )


if __name__ == "__main__":
    main()
