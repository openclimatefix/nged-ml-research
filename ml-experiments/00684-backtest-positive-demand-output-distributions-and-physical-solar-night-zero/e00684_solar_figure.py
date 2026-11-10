# ruff: noqa: D103
"""Experiment 684: solar night half-hours before and after the gate, for one farm and one origin.

The x axis is hours since the forecast origin (no calendar date, so the farm cannot be identified
from the page). Run like `e00684_solar_gate.py`; the argument is the output folder.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import altair as alt
import e00666_common as common  # ty: ignore[unresolved-import]
import e00666_export as export  # ty: ignore[unresolved-import]
import e00684_solar_gate as gate
import polars as pl
from plotting import ocf_theme as theme


def main() -> None:
    sites = gate.labelled_sites()
    control = export.solar_frame().filter(
        (pl.col("time_series_id") == sites.filter(pl.col("farm") == "Solar B")["time_series_id"][0])
        & (pl.col("nwp_init_time") >= gate.REFIT_START)
        & (pl.col("nwp_init_time") < gate.REFIT_STOP)
    )
    origins = control["power_fcst_init_time"].unique().sort()
    control = control.filter(pl.col("power_fcst_init_time") == origins[len(origins) // 2])
    night = gate.elevation(frame=control, sites=sites) <= 0.0
    after = control.with_columns(
        pl.when(pl.Series(night)).then(0.0).otherwise(pl.col(q)).cast(pl.Float32).alias(q)
        for q in common.QUANTILES
    )
    frames = []
    for name, f in (("Stored (before the gate)", control), ("Gated at night (after)", after)):
        frames.append(
            f.with_columns(
                arm=pl.lit(name),
                hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes()
                / 60.0,
            ).filter(pl.col("hours") <= 60)
        )
    d = pl.concat(frames)
    base = alt.Chart(d, width=640, height=170)  # ty: ignore[unresolved-attribute]
    x = alt.X("hours:Q", title="Hours since the forecast origin")
    band = base.mark_area(opacity=0.3, color=theme.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
        x=x, y=alt.Y("q05:Q", title="MW"), y2="q95:Q"
    )
    top = base.mark_line(color=theme.BRAND_ORANGE, strokeWidth=1).encode(x=x, y="q99:Q")  # ty: ignore[unresolved-attribute]
    med = base.mark_line(color=theme.DATA_BLUE, strokeWidth=1).encode(x=x, y="q50:Q")  # ty: ignore[unresolved-attribute]
    (band + top + med).facet(row=alt.Row("arm:N", title=None)).save(
        Path(sys.argv[1]) / "solar_night_before_after.svg"
    )


if __name__ == "__main__":
    main()
