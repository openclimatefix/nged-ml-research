# ruff: noqa: D103, E501
"""Experiment 677: observed Monday-to-Friday half-hour profiles of D16 and D5 before, during, and after the confirmed transfer (SVG).

Usage: python e00677_january.py OUT.svg
"""

import sys
from datetime import UTC, datetime, timedelta

import altair as alt
import e00668_score_lib as lib  # ty: ignore[unresolved-import]
import polars as pl

WEEKS = {"Mon 8 to Fri 12 Dec 2025": datetime(2025, 12, 8, tzinfo=UTC), "Mon 5 to Fri 9 Jan 2026": datetime(2026, 1, 5, tzinfo=UTC), "Mon 19 to Fri 23 Jan 2026": datetime(2026, 1, 19, tzinfo=UTC)}


def main() -> None:
    truth = pl.read_parquet(lib.FOLDER / "ground_truth_primaries.parquet").filter(pl.col("time_series_id").is_in([17, 18])).with_columns(lab=pl.when(pl.col("time_series_id") == 17).then(pl.lit("D5")).otherwise(pl.lit("D16")))
    parts = [truth.filter(pl.col("valid_time").is_between(st, st + timedelta(days=5), closed="left")).with_columns(hh=pl.col("valid_time").dt.hour() + pl.col("valid_time").dt.minute() / 60, week=pl.lit(name)) for name, st in WEEKS.items()]
    prof = pl.concat(parts).group_by("lab", "week", "hh").agg(pl.col("power").mean().alias("obs")).sort("hh").to_pandas()

    def panel(lab: str) -> alt.Chart:
        return alt.Chart(prof[prof.lab == lab]).mark_line(strokeWidth=1.2).encode(
            x=alt.X("hh:Q", title="Hour of day (UTC)", axis=alt.Axis(values=[0, 6, 12, 18, 24]), scale=alt.Scale(domain=[0, 24])),
            y=alt.Y("obs:Q", title="Mean observed power, Mon to Fri (MW or MVA)", scale=alt.Scale(zero=False)),
            color=alt.Color("week:N", title=None, scale=alt.Scale(domain=list(WEEKS), range=["#009C75", "#B701FF", "#FC9700"]), legend=alt.Legend(orient="top", columns=1)),
        ).properties(width=620, height=300, title=f"{lab}: working-day half-hour profile")

    alt.vconcat(panel("D16"), panel("D5")).properties(title=alt.TitleParams("INTERIM. Observed power only; the same weekdays before, during, and after the confirmed transfer.", anchor="start", fontSize=11)).save(sys.argv[1])


if __name__ == "__main__":
    main()
