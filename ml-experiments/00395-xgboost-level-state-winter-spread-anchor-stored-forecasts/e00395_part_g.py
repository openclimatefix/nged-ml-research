"""Part G of experiment 395: experiment 390's event fans, with the events shaded and the lags shown.

Redraws the day-ahead fans of the weekly bundle through the four real load-shift events experiment
390 chose (its `fan_events.csv`, read only). Each event gets two panels. The top panel shows the
meter reading and the bundle's q05 to q95 band and median (forecasts made 24 to 48 hours earlier),
with the hand-labelled event shaded (`briefs/sw_block_detector_label_table.csv`, unpadded) and the
padding of experiment 390's mask as a lighter band. The bottom panel shows the same band against
the three lagged-power inputs the forecast read: the newest usable reading, the reading a day
earlier, and the reading a week earlier.

```bash
PYTHONPATH=ml-experiments/00377-demand-v1-first-run-feasibility:\
ml-experiments/00381-simple-weekly-xgboost-demand-against-gam-stack:\
ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\
ml-experiments/00390-xgboost-d11-d14-gap-and-load-shift-event-lags \
    uv run --no-sync python ml-experiments/00395-*/e00395_part_g.py
```
"""

import csv
import textwrap
from datetime import timedelta
from pathlib import Path
from typing import Final

import altair as alt
import e00087_common as common  # ty: ignore[unresolved-import]
import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00381_score as s381  # ty: ignore[unresolved-import]
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00395-xgboost-level-state-winter-spread-anchor-stored-forecasts"
    / "part_g"
)
FAN_EVENTS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00390-xgboost-d11-d14-gap-and-load-shift-event-lags"
    / "work"
    / "fan_events.csv"
)
LABEL_TABLE: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "briefs" / "sw_block_detector_label_table.csv"
WIDTH: Final[int] = 680
HEIGHT: Final[int] = 95
WRAP: Final[int] = 110
HOURS_AHEAD: Final[tuple[float, float]] = (24.0, 48.0)


def labelled_span(*, label: str, start: object, end: object) -> tuple[object, object]:
    """Return the unpadded hand-labelled onset and end of the padded event `label` start..end."""
    best = None
    for r in csv.DictReader(LABEL_TABLE.open()):
        if label not in r["pair"] or not r["end_utc"]:
            continue
        onset = pl.Series([r["onset_utc"]]).str.to_datetime("%Y-%m-%d %H:%M", time_zone="UTC")[0]
        finish = pl.Series([r["end_utc"]]).str.to_datetime("%Y-%m-%d %H:%M", time_zone="UTC")[0]
        if start - timedelta(days=2) <= onset <= end and (  # ty: ignore[unsupported-operator]
            best is None or abs(onset - start) < abs(best[0] - start)
        ):  # ty: ignore[unsupported-operator]
            best = (onset, finish)
    if best is None:
        msg = f"no labelled event for {label} {start}"
        raise ValueError(msg)
    return best


def power_grid(*, actuals: pl.DataFrame, sid: int) -> np.ndarray:
    """Return a Primary's readings in MW on the dense grid, NaN where absent."""
    one = actuals.filter(pl.col("time_series_id") == sid)
    grid = np.full(common.grid_times().len(), np.nan)
    grid[np.asarray(common.grid_index(times=one["valid_time"]))] = one["actual"].to_numpy()
    return grid


def main() -> None:
    """Draw the figure and print the padding of experiment 390's mask."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    series = adapter.series_table()
    ctx = s381.build_context(series=series)
    bundle = s381.arm_frame(regime="weekly", arm="bundle_solar+sunny+hl1", series=series)
    events = pl.read_csv(FAN_EVENTS, try_parse_dates=True)
    panels = []
    for i, row in enumerate(events.iter_rows(named=True)):
        sid = row["time_series_id"]
        start, end = row["start"], row["end"]
        onset, finish = labelled_span(label=row["label"], start=start, end=end)
        print(
            f"{row['label']} event {row['event_id']}: mask starts {onset - start} before... "
            f"label onset {onset}, mask start {start} (mask - label onset = {start - onset}); "
            f"label end {finish}, mask end {end} (mask end - label end = {end - finish})"
        )
        lo, hi = start - timedelta(days=1), end + timedelta(days=7)
        grid = power_grid(actuals=ctx["actuals"], sid=sid)
        fan = bundle.filter(
            pl.col("time_series_id") == sid, pl.col("valid_time").is_between(lo, hi)
        ).filter(
            (
                (pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes() / 60
            ).is_between(HOURS_AHEAD[0], HOURS_AHEAD[1], closed="left")
        )
        target = np.asarray(common.grid_index(times=fan["valid_time"]))
        origin = np.asarray(common.grid_index(times=fan["power_fcst_init_time"]))
        lags = common.origin_lags(power=grid, target=target, origin=origin)
        fan = fan.with_columns(
            lag_newest=pl.Series(lags[:, 0]),
            lag_day=pl.Series(lags[:, 1]),
            lag_week=pl.Series(lags[:, 2]),
        ).select("valid_time", "q05", "q50", "q95", "lag_newest", "lag_day", "lag_week")
        reading = ctx["actuals"].filter(
            pl.col("time_series_id") == sid, pl.col("valid_time").is_between(lo, hi)
        )
        shade = pl.DataFrame({"a": [start], "b": [end], "c": [onset], "d": [finish]}).with_columns(
            pl.all().cast(pl.Datetime("us", "UTC"))
        )
        pad = alt.Chart(shade).mark_rect(color="#808080", opacity=0.12).encode(x="a:T", x2="b:T")  # ty: ignore[unresolved-attribute]
        lab = alt.Chart(shade).mark_rect(color="#808080", opacity=0.30).encode(x="c:T", x2="d:T")  # ty: ignore[unresolved-attribute]
        band = (
            alt.Chart(fan)
            .mark_area(opacity=0.28, color=ocf.DATA_BLUE)
            .encode(x=alt.X("valid_time:T", title=None), y=alt.Y("q05:Q", title="MW"), y2="q95:Q")  # ty: ignore[unresolved-attribute]
        )
        median = (
            alt.Chart(fan)
            .mark_line(strokeWidth=1, color=ocf.DATA_BLUE)
            .encode(x="valid_time:T", y="q50:Q")  # ty: ignore[unresolved-attribute]
        )
        line = (
            alt.Chart(reading)
            .mark_line(strokeWidth=1.2, color="#202020")
            .encode(x="valid_time:T", y=alt.Y("actual:Q", title="MW"))  # ty: ignore[unresolved-attribute]
        )
        title = f"{row['label']}: event of {row['duration_hours'] / 24:.0f} days ({row['why']})"
        top = alt.layer(pad, lab, band, median, line).properties(
            width=WIDTH - 80,
            height=HEIGHT,
            title=alt.TitleParams(title, anchor="start", fontSize=11),
        )
        long = fan.unpivot(
            ["lag_newest", "lag_day", "lag_week"],
            index="valid_time",
            variable_name="lag",
            value_name="mw",
        ).with_columns(
            lag=pl.col("lag").replace_strict(
                {
                    "lag_newest": "Newest reading",
                    "lag_day": "Day-earlier reading",
                    "lag_week": "Week-earlier reading",
                }
            )
        )
        colours = {
            "Newest reading": ocf.BRAND_ORANGE,
            "Day-earlier reading": ocf.DATA_PURPLE,
            "Week-earlier reading": ocf.DATA_GREEN,
        }
        lag_lines = (
            alt.Chart(long)
            .mark_line(strokeWidth=1)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("valid_time:T", title=None),
                y=alt.Y("mw:Q", title="MW"),
                color=alt.Color(
                    "lag:N",
                    scale=alt.Scale(domain=list(colours), range=list(colours.values())),
                    legend=alt.Legend(title=None, orient="top") if i == 0 else None,
                ),
            )
        )
        bottom = alt.layer(pad, lab, band, lag_lines).properties(width=WIDTH - 80, height=HEIGHT)
        panels.extend([top, bottom])
    subtitle = (
        "Top panel: black line is the meter reading; blue band and line are the q05 to q95 range "
        "and "
        "median of the weekly bundle's forecast made 24 to 48 hours earlier. Bottom panel: the "
        "same "
        "band against the three lagged-power readings that forecast used, drawn at the valid time. "
        "Dark grey: hand-labelled event. Light grey: the 24-hour padding that experiment 390's "
        "mask adds "
        "on each side. A coloured line outside the blue band marks a lagged reading the forecast "
        "would not have expected."
    )
    chart = (
        alt.vconcat(*panels)
        .resolve_scale(x="independent", y="independent", color="shared")
        .properties(
            title=alt.TitleParams(
                text=textwrap.wrap(
                    "Day-ahead forecasts and their lagged inputs through four load-shift events", 85
                ),
                subtitle=textwrap.wrap(subtitle, WRAP),
                anchor="start",
            )
        )
    )
    chart.save(DOCS_DIR / "event_fans_shaded.svg")
    print("saved", DOCS_DIR / "event_fans_shaded.svg")


if __name__ == "__main__":
    main()
