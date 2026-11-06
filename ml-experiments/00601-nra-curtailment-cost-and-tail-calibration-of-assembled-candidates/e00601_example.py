# ruff: noqa: E501, D103
"""Experiment 601: the maintainer's example event (December 2025 pair, D5 and D16), day-ahead forecasts.

Two panels per Primary, one forecast per panel (C1 above, C1 + G4 below; the same colours in both), observed
power in black, the load-shift window shaded, dashed lines at the labelled switch times, and a thin strip
for the days on which the G4 band is wide. Rows are the day-ahead forecasts (12 to 36 hours ahead). The
NRA is not drawn because it is only an estimate; coverage is scored against it.
"""

from datetime import UTC, datetime, timedelta
import os
from pathlib import Path

import altair as alt
import e00601_common as k
import polars as pl
from plotting import ocf_theme as theme

DOCS = Path(__file__).resolve().parents[2] / "docs/ml-experiments" / Path(__file__).resolve().parent.name
START = datetime(2025, 12, 30, 1, 0, tzinfo=UTC)
END = datetime(2026, 1, 14, 11, 30, tzinfo=UTC)
W = 620
DOM0 = (START - timedelta(days=14)).isoformat()
DOM1 = (END + timedelta(days=14)).isoformat()
PNG_DIR = os.environ.get("PNG_DIR")


def load(prefix: str, sid: int) -> pl.DataFrame:
    return (
        pl.concat([pl.read_parquet(k.EXPORT / f"{prefix}_{m}.parquet") for m in ("2025-12", "2026-01")])
        .filter(pl.col("time_series_id") == sid, pl.col("lead_hours") >= 12, pl.col("lead_hours") < 36)
        .filter(pl.col("valid_time") >= START - timedelta(days=14), pl.col("valid_time") < END + timedelta(days=14))
        .sort("valid_time")
    )


def panel(*, d: pl.DataFrame, label: str, ymax: float, show_x: bool) -> alt.LayerChart:
    pdf = d.select("valid_time", "q05", "q50", "q95", "observed_power").to_pandas()
    base = alt.Chart(pdf).encode(x=alt.X("valid_time:T", title=None, scale=alt.Scale(domain=[DOM0, DOM1]), axis=alt.Axis(labels=show_x, format="%d %b")))
    shade = alt.Chart(pl.DataFrame({"a": [START], "b": [END]}).to_pandas()).mark_rect(color=theme.DATA_PURPLE_LIGHT, opacity=0.5).encode(x="a:T", x2="b:T")
    band = base.mark_area(color=theme.DATA_BLUE, opacity=0.25).encode(y=alt.Y("q05:Q", title="Power (MW)", scale=alt.Scale(domain=[0, ymax])), y2="q95:Q")
    med = base.mark_line(color=theme.DATA_BLUE, strokeWidth=1.5).encode(y="q50:Q")
    obs = base.mark_line(color="#000000", strokeWidth=1).encode(y="observed_power:Q")
    rules = alt.Chart(pl.DataFrame({"t": [START, END]}).to_pandas()).mark_rule(strokeDash=[5, 4], color=theme.BLACK_1).encode(x="t:T")
    return alt.layer(shade, band, med, obs, rules).properties(width=W, height=300, title=label)


def one(*, sid: int, name: str) -> None:
    c1, g4 = load("C1", sid), load("C1_G4", sid)
    ymax = float(max(g4["q95"].max(), g4["observed_power"].max())) * 1.05
    gate = g4.group_by(pl.col("valid_time").dt.truncate("1d").alias("day")).agg(pl.col("gate_on").any()).filter(pl.col("gate_on")).sort("day")
    strip = alt.Chart(gate.with_columns(end=pl.col("day") + pl.duration(days=1)).to_pandas()).mark_rect(color=theme.BRAND_ORANGE).encode(x=alt.X("day:T", title=None, scale=alt.Scale(domain=[DOM0, DOM1]), axis=alt.Axis(format="%d %b")), x2="end:T").properties(width=W, height=14, title="Days on which the G4 band is wide (orange)")
    keys = pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0, 4.0], "label": ["Observed power (black line)", "Forecast median (blue line)", "q05-q95 band (blue shade)", "Load-shift window (purple shade)", "Switch times (dashed lines)"], "colour": ["#000000", theme.DATA_BLUE, theme.DATA_BLUE_LIGHT, theme.DATA_PURPLE_LIGHT, theme.BLACK_1]})
    legend = alt.layer(
        alt.Chart(keys.to_pandas()).mark_square(size=140, opacity=0.9).encode(x=alt.X("x:Q", axis=None, scale=alt.Scale(domain=[-0.2, 5.0])), color=alt.Color("colour:N", scale=None)),
        alt.Chart(keys.to_pandas()).mark_text(align="left", dx=8, fontSize=10).encode(x="x:Q", text="label:N"),
    ).properties(width=W, height=24)
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
