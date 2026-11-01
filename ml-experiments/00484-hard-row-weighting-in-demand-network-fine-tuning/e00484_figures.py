r"""Draw experiment 484's headline figure from the stored contrasts (no fit).

Figure 1: each weighted arm minus the stored fit (W0) on the gating CRPS slices, with Bonferroni
95% intervals (count 27) and the minimum detectable effect of a three-seed-mean contrast.

```bash
source /tmp/e484env.sh
uv run --no-sync python ml-experiments/00484-*/e00484_figures.py
```
"""

from pathlib import Path
from typing import Final

import altair as alt
import e00484_score as score484  # ty: ignore[unresolved-import]
import polars as pl
from plotting import ocf_theme as theme

FIGURE_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00484-hard-row-weighting-in-demand-network-fine-tuning"
)
SLICES: Final[dict[str, str]] = {
    "eight": "Eight, all rows",
    "six_clean": "Six, no labelled event",
    "six_nra": "Six, real events",
    "cold_eight": "Eight, cold nights",
    "cold_six": "Six, cold nights",
    "band_eight": "Eight, 12 to 36 h",
    "band_six": "Six, 12 to 36 h",
}


def main() -> None:
    """Write `figure1_contrasts.svg`."""
    _ = theme
    table = (
        pl.read_csv(score484.SCORE_DIR / "contrasts.csv")
        .filter((pl.col("metric") == "crps") & pl.col("slice").is_in(list(SLICES)))
        .with_columns(
            pl.col("slice").replace_strict(SLICES).alias("Row set"),
            pl.col("contrast").str.replace(" - W0", "").alias("Arm"),
        )
    )
    order = list(SLICES.values())
    base = alt.Chart(table).encode(
        y=alt.Y("Row set:N", sort=order, title=None),
        yOffset="Arm:N",
        color=alt.Color("Arm:N", title="Arm"),
    )
    points = base.mark_point(filled=True, size=40).encode(
        x=alt.X(
            "difference:Q", title="CRPS minus stored fit (points of capacity; below 0 is better)"
        )
    )
    bars = base.mark_rule().encode(x="low_bonferroni:Q", x2="high_bonferroni:Q")
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(strokeDash=[4, 3]).encode(x="x:Q")
    chart = (zero + bars + points).properties(width=520, height=300)
    chart.save(FIGURE_DIR / "figure1_contrasts.svg")


if __name__ == "__main__":
    main()
