# ruff: noqa: E501, ANN001, ANN201, D103, PERF401, RET504, E402
"""Experiment 578: NRA metrics per event, follow ratio by days since onset and end, and day-ahead plots.

Metrics (all per event, valid times inside the event unless stated; seed-mean medians; 24-120 h unless stated):
- follow ratio: mean((v3+D5 median - corrected-lag median) / depth), depth = signed labelled depth, capacity fraction.
- lag effect: mean((v3+D5 median - lags-masked median) / depth).
- error during the event: mean(|v3+D5 median - observed|) / |depth|.
- recovery error: the same error for valid times in the 7 days after the event ends.
- level share: sum(forecast - corrected power) / sum(observed - corrected power).
Writes `work/nra_metrics.csv` and `work/follow_by_day.csv`, and the SVGs in the docs folder.
"""

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
TESTABLE = (115, 143, 146, 172, 173, 175, 178)
EXTRA = (55,)
SERIES = {
    "Observed": theme.BRAND_ORANGE,
    "Corrected (event removed)": theme.DATA_BLUE,
    "v3+D5, observed lags": theme.DATA_PURPLE,
    "v3+D5, corrected lags": theme.DATA_GREEN,
    "v3+D5, lags masked": "grey",
    "Lag-free D5": "black",
}


def table() -> pl.DataFrame:
    cl, sh = k.clean(), k.corrected_shift()
    sm = k.sites()
    prim_of = dict(zip(sm["time_series_id"].to_list(), sm["primary"].to_list(), strict=True))
    allm = (*k.MONTHS, *f2.NEW)
    names = {"v": ("V3D5", allm), "c": ("V3D5-corrected", allm), "r": ("V3D5-removed", allm)}
    j = None
    for n, (tag, ms) in names.items():
        fr = f2.load(root=f2._Root(), tag=tag, months=ms).rename({"q50": n})
        j = fr if j is None else j.join(fr, on=f2.KEYS)
    e = f2.load(root=k.d6.W, tag="D5", months=k.MONTHS).rename({"q50": "e"})
    j = j.join(e, on=f2.KEYS, how="left").with_columns(
        lead=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_hours(),
        prim=pl.col("time_series_id").replace_strict(prim_of),
        idx=((pl.col("valid_time").dt.epoch("us") - k.prep.GRID_START_US) // k.prep.STEP_US).cast(
            pl.Int64
        ),
    )
    p, i = j["prim"].to_numpy(), j["idx"].to_numpy()
    obs = cl[p, i]
    return j.with_columns(obs=pl.Series(obs), cf=pl.Series(obs + sh[p, i])).filter(
        pl.col("obs").is_finite()
    )


def main() -> None:
    j = table()
    j120 = j.filter((pl.col("lead") >= 24) & (pl.col("lead") < 120))
    ev = pl.read_csv(k.WORK / "step0_events.csv").filter(pl.col("long"))
    sm = k.sites()
    group_of = dict(zip(sm["primary"].to_list(), sm["group"].to_list(), strict=True))
    out, by_day = [], []
    for q in ev.iter_rows(named=True):
        t0 = k.GRID_START + dt.timedelta(minutes=30 * q["onset_idx"])
        t1 = k.GRID_START + dt.timedelta(minutes=30 * q["end_idx"])
        depth = q["depth_share"] / 100 * np.sign(q["depth_mw"])
        mine = j120.filter(pl.col("prim") == q["primary"])
        ins = mine.filter(pl.col("valid_time").is_between(t0, t1))
        if ins.height < 200:
            continue
        aft = mine.filter(
            (pl.col("valid_time") > t1) & (pl.col("valid_time") <= t1 + dt.timedelta(days=7))
        )
        ins = ins.with_columns(
            follow=(pl.col("v") - pl.col("c")) / depth,
            lag_effect=(pl.col("v") - pl.col("r")) / depth,
            since_onset=((pl.col("power_fcst_init_time") - t0).dt.total_hours() // 24).cast(
                pl.Int64
            ),
        )
        aft = aft.with_columns(
            follow=(pl.col("v") - pl.col("c")) / depth,
            lag_effect=(pl.col("v") - pl.col("r")) / depth,
            since_end=((pl.col("valid_time") - t1).dt.total_hours() // 24).cast(pl.Int64),
        )
        den = float((ins["obs"] - ins["cf"]).sum())
        row = {
            "event_id": q["event_id"],
            "label": q["label"],
            "group": group_of[q["primary"]],
            "days": round(q["length_days"], 1),
            "depth_share": round(q["depth_share"], 1),
            "long_deep": q["long_deep"],
            "rows": ins.height,
            "follow": ins["follow"].mean(),
            "lag_effect": ins["lag_effect"].mean(),
            "error_during": float((ins["v"] - ins["obs"]).abs().mean()) / abs(depth),
            "error_corrected_lags_during": float((ins["c"] - ins["obs"]).abs().mean()) / abs(depth),
            "recovery_error": float((aft["v"] - aft["obs"]).abs().mean()) / abs(depth)
            if aft.height > 100
            else None,
            "recovery_follow": aft["follow"].mean() if aft.height > 100 else None,
            "level_share": float((ins["v"] - ins["cf"]).sum()) / den,
        }
        out.append(row)
        for r in (
            ins.group_by("since_onset")
            .agg(follow=pl.col("follow").mean(), n=pl.len())
            .iter_rows(named=True)
        ):
            by_day.append(
                {
                    "event_id": q["event_id"],
                    "axis": "onset",
                    "day": r["since_onset"],
                    "follow": r["follow"],
                    "n": r["n"],
                }
            )
        for r in (
            aft.group_by("since_end")
            .agg(follow=pl.col("follow").mean(), n=pl.len())
            .iter_rows(named=True)
        ):
            by_day.append(
                {
                    "event_id": q["event_id"],
                    "axis": "end",
                    "day": r["since_end"],
                    "follow": r["follow"],
                    "n": r["n"],
                }
            )
    t = pl.DataFrame(out)
    t.write_csv(k.WORK / "nra_metrics.csv")
    d = pl.DataFrame(by_day).filter(pl.col("n") >= 40)
    d.write_csv(k.WORK / "follow_by_day.csv")
    with pl.Config(tbl_rows=30, tbl_cols=20, tbl_width_chars=250):
        print(t.with_columns(pl.col(pl.Float64).round(3)))
    summary(d=d)
    for eid in (*TESTABLE, *EXTRA):
        day_ahead(j=j, event_id=eid)


def summary(*, d: pl.DataFrame) -> None:
    """Follow ratio by day since onset (cap 20) and since end, mean over events with a bootstrap-over-events 95% interval."""
    rng = np.random.default_rng(0)
    rows = []
    for axis, days in (("onset", range(0, 21)), ("end", range(0, 7))):
        for day in days:
            s = d.filter((pl.col("axis") == axis) & (pl.col("day") == day))
            if s.height < 3:
                continue
            x = s["follow"].to_numpy()
            boots = [rng.choice(x, len(x)).mean() for _ in range(2000)]
            rows.append(
                {
                    "axis": axis,
                    "day": day,
                    "mean": x.mean(),
                    "lo": np.percentile(boots, 2.5),
                    "hi": np.percentile(boots, 97.5),
                    "events": len(x),
                }
            )
    s = pl.DataFrame(rows)
    s.write_csv(k.WORK / "follow_by_day_summary.csv")
    print(s.with_columns(pl.col(pl.Float64).round(3)))
    panels = []
    for axis, title in (
        ("onset", "Days from event onset to forecast issue"),
        ("end", "Days from event end to the forecast's valid time"),
    ):
        sub = s.filter(pl.col("axis") == axis)
        base = alt.Chart(sub)
        band = base.mark_area(opacity=0.25, color=theme.DATA_BLUE).encode(
            x=alt.X("day:Q", title=title), y="lo:Q", y2="hi:Q"
        )  # ty: ignore[unresolved-attribute]
        line = base.mark_line(point=True, color=theme.DATA_BLUE).encode(
            x="day:Q", y=alt.Y("mean:Q", title="Follow ratio")
        )  # ty: ignore[unresolved-attribute]
        rule = (
            alt.Chart(pl.DataFrame({"y": [0.15]}))
            .mark_rule(color="grey", strokeDash=[4, 3])
            .encode(y="y:Q")
        )  # ty: ignore[unresolved-attribute]
        panels.append(alt.layer(band, line, rule).properties(width=300, height=200))
    alt.hconcat(*panels).properties(
        title="Follow ratio of v3+D5 (observed minus corrected lags), mean over events with a 95% interval from resampling events (3 or more events per day)"
    ).save(str(DOCS / "follow_by_day.svg"))


def day_ahead(*, j: pl.DataFrame, event_id: int) -> None:
    ev = (
        pl.read_csv(k.WORK / "step0_events.csv")
        .filter(pl.col("event_id") == event_id)
        .row(0, named=True)
    )
    p = int(k.sites().filter(pl.col("label") == ev["label"])["primary"][0])
    t0 = k.GRID_START + dt.timedelta(minutes=30 * ev["onset_idx"])
    t1 = k.GRID_START + dt.timedelta(minutes=30 * ev["end_idx"])
    lo, hi = t0 - dt.timedelta(days=2), t1 + dt.timedelta(days=7)
    d = j.filter(
        (pl.col("prim") == p)
        & (pl.col("lead") >= 24)
        & (pl.col("lead") < 48)
        & pl.col("valid_time").is_between(lo, hi)
    ).sort("valid_time")
    if d.height == 0:
        return
    d = d.rename(
        {
            "obs": "Observed",
            "cf": "Corrected (event removed)",
            "v": "v3+D5, observed lags",
            "c": "v3+D5, corrected lags",
            "r": "v3+D5, lags masked",
            "e": "Lag-free D5",
        }
    )
    long = (
        d.select("valid_time", *[c for c in SERIES if c in d.columns])
        .unpivot(index="valid_time", variable_name="series", value_name="share")
        .filter(pl.col("share").is_not_null() & pl.col("share").is_finite())
    )
    band = (
        alt.Chart(pl.DataFrame({"a": [t0], "b": [t1]}))
        .mark_rect(opacity=0.12, color="grey")
        .encode(x="a:T", x2="b:T")
    )  # ty: ignore[unresolved-attribute]
    snap = (
        alt.Chart(pl.DataFrame({"x": [t1]}))
        .mark_rule(color="red", strokeDash=[2, 2])
        .encode(x="x:T")
    )  # ty: ignore[unresolved-attribute]
    line = (
        alt.Chart(long)
        .mark_line(strokeWidth=1.1)
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
    alt.layer(band, snap, line).properties(
        width=620,
        height=210,
        title=f"Primary {ev['label']}, event {event_id}: day-ahead forecasts (shaded: event; red: snap-back)",
    ).save(str(DOCS / f"day_ahead_event{event_id}.svg"))


main()
