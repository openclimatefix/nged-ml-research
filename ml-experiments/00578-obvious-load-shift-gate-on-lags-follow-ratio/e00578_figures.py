# ruff: noqa: E501, D103
"""Experiment 578 figures: expectation for event 146, follow ratio per event, 14-day plots of events 146 and 55."""

import datetime as dt
from pathlib import Path

import altair as alt
import e00578_common as k  # ty: ignore[unresolved-import]
import e00578_follow2 as f2  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from plotting import ocf_theme as theme

DOCS = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments"
    / Path(__file__).resolve().parent.name
)
SERIES = {
    "Observed": theme.BRAND_ORANGE,
    "Corrected (event removed)": theme.DATA_BLUE,
    "Forecast, observed lags": theme.DATA_PURPLE,
    "Forecast, corrected lags": theme.DATA_GREEN,
    "Forecast, masked lags": "grey",
}


def followers() -> None:
    t = (
        pl.read_csv(k.WORK / "follow_corrected.csv")
        .sort("days")
        .with_columns(
            name=pl.format(
                "{} (event {}, {} d{})",
                pl.col("label"),
                pl.col("event_id"),
                pl.col("days").round(0).cast(pl.Int64),
                pl.when(pl.col("long_deep")).then(pl.lit(", long and deep")).otherwise(pl.lit("")),
            )
        )
    )
    bars = (
        alt.Chart(t)
        .mark_bar(color=theme.DATA_BLUE)
        .encode(
            y=alt.Y("name:N", sort=None, title=None),
            x=alt.X("follow:Q", title="Follow ratio at 24-120 h (share of the labelled depth)"),
        )
    )
    rule = (
        alt.Chart(pl.DataFrame({"x": [0.15]}))
        .mark_rule(color="grey", strokeDash=[4, 3])
        .encode(x="x:Q")
    )  # ty: ignore[unresolved-attribute]
    alt.layer(bars, rule).properties(
        width=520,
        height=20 * t.height + 30,
        title="v3+D5: observed lags minus corrected lags, per event (dashed: 0.15)",
    ).save(str(DOCS / "follow_ratio_per_event.svg"))


def fourteen_day(*, event_ids: tuple[int, ...], start: dt.datetime, name: str) -> None:
    cl, sh = k.clean(), k.corrected_shift()
    sm = k.sites()
    ev = pl.read_csv(k.WORK / "step0_events.csv")
    allm = (*k.MONTHS, *f2.NEW)
    arms = {
        "Forecast, observed lags": "V3D5",
        "Forecast, corrected lags": "V3D5-corrected",
        "Forecast, masked lags": "V3D5-removed",
    }
    fr = {
        n: f2.load(root=f2._Root(), tag=t, months=allm).rename({"q50": n}) for n, t in arms.items()
    }
    j = (
        fr["Forecast, observed lags"]
        .join(fr["Forecast, corrected lags"], on=f2.KEYS)
        .join(fr["Forecast, masked lags"], on=f2.KEYS)
    )
    j = j.with_columns(
        lead=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_hours()
    ).filter((pl.col("lead") >= 24) & (pl.col("lead") < 48))
    stop = start + dt.timedelta(days=14)
    panels = []
    for eid in event_ids:
        q = ev.filter(pl.col("event_id") == eid).row(0, named=True)
        s = sm.filter(pl.col("label") == q["label"]).row(0, named=True)
        p, tsid = s["primary"], s["time_series_id"]
        d = j.filter(
            (pl.col("time_series_id") == tsid)
            & (pl.col("valid_time") >= start)
            & (pl.col("valid_time") < stop)
        ).sort("valid_time")
        idx = ((d["valid_time"].dt.epoch("us") - k.prep.GRID_START_US) // k.prep.STEP_US).to_numpy()
        d = d.with_columns(
            Observed=pl.Series(cl[p, idx]),
            **{"Corrected (event removed)": pl.Series(cl[p, idx] + sh[p, idx])},
        )
        long = (
            d.select("valid_time", *SERIES)
            .unpivot(index="valid_time", variable_name="series", value_name="share")
            .drop_nulls()
            .filter(pl.col("share").is_finite())
        )
        t0 = k.GRID_START + dt.timedelta(minutes=30 * q["onset_idx"])
        t1 = k.GRID_START + dt.timedelta(minutes=30 * q["end_idx"])
        band = (
            alt.Chart(pl.DataFrame({"a": [max(t0, start)], "b": [min(t1, stop)]}))
            .mark_rect(opacity=0.12, color="grey")
            .encode(x="a:T", x2="b:T")
        )  # ty: ignore[unresolved-attribute]
        line = (
            alt.Chart(long)
            .mark_line(strokeWidth=1.2)
            .encode(
                x=alt.X("valid_time:T", title="Date (UTC)"),
                y=alt.Y("share:Q", title="Share of capacity", scale=alt.Scale(zero=False)),
                color=alt.Color(
                    "series:N",
                    title=None,
                    scale=alt.Scale(domain=list(SERIES), range=list(SERIES.values())),
                ),
            )
        )
        panels.append(
            alt.layer(band, line).properties(
                width=560,
                height=170,
                title=f"Primary {q['label']}, event {eid} (shaded); forecasts issued 24-48 h ahead",
            )
        )
    alt.vconcat(*panels).save(str(DOCS / name))


def expectation() -> None:
    q = (
        pl.read_csv(k.WORK / "step0_events.csv")
        .filter(pl.col("event_id") == 146)
        .row(0, named=True)
    )
    sd = pl.read_csv(k.WORK / "step0_gap_sd.csv").filter(pl.col("label") == q["label"])[
        "event_free_gap_sd"
    ][0]
    p = int(k.sites().filter(pl.col("label") == q["label"])["primary"][0])
    cl, sh = k.clean(), k.corrected_shift()
    ex = k.expected_lag_free(shape=cl.shape)
    d0 = q["onset_idx"] // k.PER_DAY - 8
    rows = []
    for d in range(d0, d0 + 58):
        s = slice(d * k.PER_DAY, (d + 1) * k.PER_DAY)
        rows.append(
            {
                "date": str(np.datetime64("2019-09-01") + np.timedelta64(d, "D")),
                "Observed": float(np.nanmean(cl[p, s])),
                "Corrected (event removed)": float(np.nanmean(cl[p, s] + sh[p, s])),
                "Lag-free D5 expectation": float(np.nanmean(ex[p, s])),
            }
        )
    long = (
        pl.DataFrame(rows)
        .with_columns(pl.col("date").str.to_date())
        .unpivot(index="date", variable_name="series", value_name="share")
        .filter(pl.col("share").is_finite())
    )
    dom = ["Observed", "Corrected (event removed)", "Lag-free D5 expectation"]
    chart = (
        alt.Chart(long)
        .mark_line()
        .encode(
            x=alt.X("date:T", title="Date (UTC)"),
            y=alt.Y("share:Q", title="Daily mean power, share of capacity"),
            color=alt.Color(
                "series:N",
                title=None,
                scale=alt.Scale(
                    domain=dom, range=[theme.BRAND_ORANGE, theme.DATA_BLUE, theme.DATA_PURPLE]
                ),
            ),
        )
        .properties(
            width=560,
            height=240,
            title=f"Primary {q['label']}, event 146: the lag-free expectation follows the event (event-free daily gap SD {sd:.3f})",
        )
    )
    chart.save(str(DOCS / "event146_expectation.svg"))


followers()
expectation()
fourteen_day(
    event_ids=(146, 55),
    start=dt.datetime(2025, 12, 28, tzinfo=dt.UTC),
    name="example_events_14_days.svg",
)
fourteen_day(
    event_ids=(173,), start=dt.datetime(2025, 7, 20, tzinfo=dt.UTC), name="event173_14_days.svg"
)
