# ruff: noqa: E501
"""Experiment 578: the 40-day event of D5 against the lag-free expectation, and the lag effect per event."""

from pathlib import Path

import altair as alt
import e00578_common as k  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from plotting import ocf_theme as theme

DOCS = Path(__file__).resolve().parents[2] / "docs/ml-experiments" / Path(__file__).resolve().parent.name
EVENT = 146


def main() -> None:
    q = pl.read_csv(k.WORK / "step0_events.csv").filter(pl.col("event_id") == EVENT).row(0, named=True)
    sd = pl.read_csv(k.WORK / "step0_gap_sd.csv").filter(pl.col("label") == q["label"])["event_free_gap_sd"][0]
    p = int(k.sites().filter(pl.col("label") == q["label"])["primary"][0])
    cl, ex, sh = k.clean(), None, k.corrected_shift()
    ex = k.expected_lag_free(shape=cl.shape)
    d0 = q["onset_idx"] // k.PER_DAY - 8
    rows = []
    for d in range(d0, d0 + 58):
        s = slice(d * k.PER_DAY, (d + 1) * k.PER_DAY)
        day = (k.GRID_START + np.timedelta64(d, "D")).date() if False else None  # noqa: SIM108
        rows.append({
            "date": str(np.datetime64("2019-09-01") + np.timedelta64(d, "D")),
            "Observed": float(np.nanmean(cl[p, s])),
            "Corrected (event removed)": float(np.nanmean(cl[p, s] + sh[p, s])),
            "Lag-free D5 expectation": float(np.nanmean(ex[p, s])),
        })
    df = pl.DataFrame(rows).with_columns(pl.col("date").str.to_date())
    long = df.unpivot(index="date", variable_name="series", value_name="share").drop_nulls()
    start = np.datetime64("2019-09-01") + np.timedelta64(int(q["onset_idx"] // k.PER_DAY), "D")
    end = np.datetime64("2019-09-01") + np.timedelta64(int(q["end_idx"] // k.PER_DAY), "D")
    shade = alt.Chart(pl.DataFrame({"a": [str(start)], "b": [str(end)]}).with_columns(pl.col("a", "b").str.to_date())).mark_rect(opacity=0.12, color="grey").encode(x="a:T", x2="b:T")  # ty: ignore[unresolved-attribute]
    lines = alt.Chart(long).mark_line().encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("date:T", title="Date (UTC)"),
        y=alt.Y("share:Q", title="Daily mean power, share of capacity"),
        color=alt.Color("series:N", title=None, scale=alt.Scale(domain=["Observed", "Corrected (event removed)", "Lag-free D5 expectation"], range=[theme.BRAND_ORANGE, theme.DATA_BLUE, theme.DATA_PURPLE])),
    )
    chart = alt.layer(shade, lines).properties(width=560, height=240, title=f"Primary {q['label']}: the 40-day event is mostly absorbed by the lag-free expectation (event-free daily gap SD {sd:.3f})")
    chart.save(str(DOCS / "event146_expectation.svg"))
    t = pl.read_csv(k.WORK / "follow_lag_effect.csv").sort("length_days").with_columns(name=pl.format("{} (event {}, {} d)", pl.col("label"), pl.col("event_id"), pl.col("length_days").round(1)))
    bars = alt.Chart(t).mark_bar(color=theme.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("name:N", sort=None, title=None),
        x=alt.X("lag_effect:Q", title="Lag effect at 24-120 h, share of the labelled depth"),
    )
    rule = alt.Chart(pl.DataFrame({"x": [0.15]})).mark_rule(color="grey", strokeDash=[4, 3]).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    alt.layer(bars, rule).properties(width=520, height=20 * t.height + 30, title="v3+D5 minus its lags-masked twin, per labelled event (dashed: the 0.15 line)").save(str(DOCS / "lag_effect_per_event.svg"))


main()
