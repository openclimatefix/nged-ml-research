# ruff: noqa: E501, D103
"""Experiment 601: day-ahead forecasts on the December 2025 example event (D5 and D16).

Two panels per Primary, one forecast per panel (C1 above, C1 + G4 below, the same colours in both).
Observed power is black, the load-shift window is shaded, dashed purple lines mark the labelled
switch times, and a thin strip marks the origin days on which the G4 band is wide. Rows are the
day-ahead forecasts (12 to 36 hours ahead). The normal running arrangement is not drawn because it
is only an estimate; coverage is scored against it.
"""

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import altair as alt
import e00601_common as k
import polars as pl
from plotting import ocf_theme as theme

DOCS = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments"
    / Path(__file__).resolve().parent.name
)
START = datetime(2025, 12, 30, 1, 0, tzinfo=UTC)
END = datetime(2026, 1, 14, 11, 30, tzinfo=UTC)
W = 620
DOM0 = (START - timedelta(days=14)).isoformat()
DOM1 = (END + timedelta(days=14)).isoformat()
PNG_DIR = os.environ.get("PNG_DIR")


def load(prefix: str, sid: int) -> pl.DataFrame:
    return (
        pl.concat(
            [pl.read_parquet(k.EXPORT / f"{prefix}_{m}.parquet") for m in ("2025-12", "2026-01")]
        )
        .filter(
            pl.col("time_series_id") == sid, pl.col("lead_hours") >= 12, pl.col("lead_hours") < 36
        )
        .filter(
            pl.col("valid_time") >= START - timedelta(days=14),
            pl.col("valid_time") < END + timedelta(days=14),
        )
        .sort("valid_time")
    )


def panel(*, d: pl.DataFrame, label: str, ymax: float, show_x: bool) -> alt.LayerChart:
    pdf = d.select("valid_time", "q05", "q50", "q95", "observed_power").to_pandas()
    base = alt.Chart(pdf).encode(
        x=alt.X(
            "valid_time:T",
            title=None,
            scale=alt.Scale(domain=[DOM0, DOM1]),
            axis=alt.Axis(labels=show_x, format="%d %b"),
        )
    )
    shade = (
        alt.Chart(pl.DataFrame({"a": [START], "b": [END]}).to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_rect(color=theme.DATA_PURPLE_LIGHT, opacity=0.5)
        .encode(x="a:T", x2="b:T")
    )
    band = base.mark_area(color=theme.DATA_BLUE, opacity=0.25).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("q05:Q", title="Power (MW)", scale=alt.Scale(domain=[0, ymax])), y2="q95:Q"
    )
    med = base.mark_line(color=theme.DATA_BLUE, strokeWidth=1.5).encode(y="q50:Q")  # ty: ignore[unresolved-attribute]
    obs = base.mark_line(color="#000000", strokeWidth=1).encode(y="observed_power:Q")  # ty: ignore[unresolved-attribute]
    rules = (
        alt.Chart(pl.DataFrame({"t": [START, END]}).to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[5, 4], color=theme.DATA_PURPLE, strokeWidth=1.5)
        .encode(x="t:T")
    )
    return alt.layer(shade, band, med, obs, rules).properties(width=W, height=300, title=label)  # ty: ignore[invalid-return-type]


def one(*, sid: int, name: str) -> None:
    c1, g4 = load("C1", sid), load("C1_G4", sid)
    ymax = float(max(g4["q95"].max(), g4["observed_power"].max())) * 1.05  # ty: ignore[invalid-argument-type]
    gate = (
        g4.group_by(pl.col("power_fcst_init_time").dt.truncate("1d").alias("day"))
        .agg(pl.col("gate_on").any())
        .filter(pl.col("gate_on"))
        .sort("day")
    )
    strip = (
        alt.Chart(  # ty: ignore[unresolved-attribute]
            gate.with_columns(
                day=pl.col("day") + pl.duration(hours=9), end=pl.col("day") + pl.duration(hours=33)
            ).to_pandas()
        )
        .mark_rect(color=theme.BRAND_ORANGE)
        .encode(
            x=alt.X(
                "day:T",
                title=None,
                scale=alt.Scale(domain=[DOM0, DOM1]),
                axis=alt.Axis(format="%d %b"),
            ),
            x2="end:T",
        )
        .properties(
            width=W,
            height=14,
            title="Origin days on which the G4 band is wide (each block spans the 24 hours after that day's 09:00 origin)",
        )
    )
    keys = pl.DataFrame(
        {
            "x": [0.0, 1.6, 3.2, 0.0, 1.6, 3.2],
            "y": [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            "label": [
                "Observed power (black line)",
                "Forecast median (blue line)",
                "q05-q95 band (blue shade)",
                "Load-shift window (purple shade)",
                "Switch times (dashed purple lines)",
                "G4 band wide on that origin day (orange strip)",
            ],
            "colour": [
                "#000000",
                theme.DATA_BLUE,
                theme.DATA_BLUE_LIGHT,
                theme.DATA_PURPLE_LIGHT,
                theme.DATA_PURPLE,
                theme.BRAND_ORANGE,
            ],
        }
    )
    legend = alt.layer(
        alt.Chart(keys.to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_square(size=140, opacity=0.9)
        .encode(
            x=alt.X("x:Q", axis=None, scale=alt.Scale(domain=[-0.2, 4.8])),
            y=alt.Y("y:Q", axis=None, scale=alt.Scale(domain=[-0.5, 1.5])),
            color=alt.Color("colour:N", scale=None),
        ),
        alt.Chart(keys.to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_text(align="left", dx=8, fontSize=10)
        .encode(x="x:Q", y="y:Q", text="label:N"),
    ).properties(width=W, height=36)
    chart = alt.vconcat(
        legend,
        panel(d=c1, label=f"{name}: C1, median and q05-q95 band", ymax=ymax, show_x=False),
        panel(d=g4, label=f"{name}: C1 + G4, median and q05-q95 band", ymax=ymax, show_x=False),
        strip,
    ).resolve_scale(x="shared")
    chart.save(str(DOCS / f"fig_example_{name}.svg"))
    if PNG_DIR:
        chart.save(str(Path(PNG_DIR) / f"fig_example_{name}.png"), scale_factor=1.0)
    print(name, "gate days", gate.height, flush=True)


if __name__ == "__main__":
    for sid, name in ((17, "D5"), (18, "D16")):
        one(sid=sid, name=name)
