# ruff: noqa: E501, C408, PERF401, ISC004
"""Experiment 468, Part A: what table P and the capacity rule change, month by month.

Reads experiment 461's `work/cells.parquet` (one row per technology, arm, series, month, and
variant, `raw` before table P and `P` after). Draws, per technology, the share of actuals above q99
by month before and after table P, and the score (mean quantile score over the 11 levels' nine-level
version, in percent of capacity) before and after, for the headline arms of experiment 468's
Part A. Private charts: no calendar dates, months only.

```bash
export DATA_PATH_INTERNAL=/home/jack/dev/nged-substation-forecast/data
uv run --no-sync python ml-experiments/00468-*/e00468_part_a_table_p.py
```
"""

from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

CELLS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00461-wind-and-solar-retrain-with-p99-9-effective-capacity"
    / "work"
    / "cells.parquet"
)
OUT: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00468-best-models-performance-analysis-with-time-series-graphs"
    / "part_a"
)
SHOWN: Final[dict[str, tuple[str, str, list[str]]]] = {
    "wind": ("member_plain", "p999", ["Farm A", "Farm B", "Farm C"]),
    "solar": ("a2_member", "old", ["Solar A", "Solar B"]),
}
WIDTH: Final[int] = 640


def main() -> None:
    """Draw the before and after charts."""
    cells = pl.read_parquet(CELLS)
    (OUT / "png").mkdir(parents=True, exist_ok=True)
    summary = []
    for tech, (arm, rule, sites) in SHOWN.items():
        d = cells.filter(
            (pl.col("tech") == tech)
            & (pl.col("arm") == arm)
            & (pl.col("rule") == rule)
            & pl.col("site").is_in(sites)
        ).with_columns(
            above_q99=(1 - pl.col("hit_q99") / pl.col("n")) * 100,
            score_pct=pl.col("score_mw_sum") / pl.col("n") / pl.col("cap_mw") * 100,
            when=pl.when(pl.col("variant") == "P")
            .then(pl.lit("After table P"))
            .otherwise(pl.lit("Before table P")),
        )
        d.select("site", "month", "variant", "n", "above_q99", "score_pct").write_csv(
            OUT / f"table_p_before_after_{tech}.csv"
        )
        pooled = d.group_by("site", "variant").agg(
            n=pl.col("n").sum(),
            above_q99=(1 - pl.col("hit_q99").sum() / pl.col("n").sum()) * 100,
            score_mw=pl.col("score_mw_sum").sum() / pl.col("n").sum(),
        )
        print(tech, arm, rule)
        print(pooled.sort("site", "variant"))
        summary.append(pooled.with_columns(tech=pl.lit(tech)))
        panels = []
        for column, title, nominal in (
            ("above_q99", "% of actuals above q99", 1.0),
            ("score_pct", "Score (% of capacity; smaller is better)", None),
        ):
            base = alt.Chart(d).encode(
                x=alt.X("month:O", title="Month scored", axis=alt.Axis(labelAngle=-45)),
                y=alt.Y(f"{column}:Q", title=title),
                color=alt.Color(
                    "when:N",
                    title=None,
                    scale=alt.Scale(
                        domain=["Before table P", "After table P"],
                        range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE],
                    ),
                    legend=alt.Legend(orient="top"),
                ),
                strokeDash=alt.StrokeDash("site:N", title=None),
            )
            layer = base.mark_line(point=True)
            if nominal is not None:
                layer += (
                    alt.Chart(pl.DataFrame({"y": [nominal]}))
                    .mark_rule(strokeDash=[4, 3])
                    .encode(y="y:Q")
                )
            panels.append(layer.properties(width=WIDTH - 110, height=140))
        chart = alt.vconcat(*panels).properties(
            title=alt.TitleParams(
                [
                    f"{tech.capitalize()} ({arm}): effect of table P by month, per series (line style).",
                    "Dashed rule: the nominal 1% above q99. Months are the scored months.",
                ],
                anchor="start",
            )
        )
        chart.save(OUT / f"table_p_before_after_{tech}.svg")
        chart.save(OUT / "png" / f"table_p_before_after_{tech}.png", scale_factor=1.5)
    pl.concat(summary).write_csv(OUT / "table_p_pooled.csv")


if __name__ == "__main__":
    main()
