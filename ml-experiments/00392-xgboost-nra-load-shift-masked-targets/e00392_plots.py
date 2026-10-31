# ruff: noqa: E501
r"""Draw experiment 392's time series plots of real labelled load-shift events.

**Event choice.** Per metering group (signed-MW six, MVA-only eight) the three events that the
label table labels (kind `labelled` or `labelled low-confidence`; not `estimated depth`, not
`merged event`), whose first half-hour falls inside one of the four screen folds, with the largest
labelled depth `|MW|`. A group with fewer than three events uses what exists. D15 and D18 are
never used. Events are numbered 1 to 6 (signed-MW group first) and their Primaries are lettered A,
B, ... in order of first appearance.

**Forecast shown.** For every half-hour at or after the event's onset, the forecast of the latest
run (`power_fcst_init_time`) at least 24 hours before the onset. For every earlier half-hour, the
forecast of the latest run at least 24 hours before that half-hour. Leads are those of the stored
forecasts (up to 336 hours); a half-hour with no such run stays blank.

**Axis.** Days from the event's onset, from 3 days before the onset to 3 days after the end, capped
at 10 days in total.

Writes `figure_events_<n>.svg` (n = 1 to 6) and `figure_examples.svg` (the fan of the deepest event
at its origin) to the experiment's docs folder, and runs `svgo` over each.

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\
ml-experiments/00379-bg2-components-all-16-primaries-to-14-days:\
ml-experiments/00383-load-shift-xgboost-oracle-first-part-1:\
ml-experiments/00392-xgboost-nra-load-shift-masked-targets \
    DATA_PATH_INTERNAL=... OMP_NUM_THREADS=1 uv run --no-sync python \
    ml-experiments/00392-*/e00392_plots.py
```
"""

import string
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Final, cast

import altair as alt
import e00379_adapter as adapter  # ty: ignore[unresolved-import]
import e00392_score as score  # ty: ignore[unresolved-import]
import plotting.ocf_theme as ocf
import polars as pl
from studies.charts import CONTENT_WIDTH_PX, wrapped
from studies.eval_protocol.slices import MVA_ONLY_LABELS, SIGNED_MW_LABELS

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00392-xgboost-nra-load-shift-masked-targets"
)
EVENTS_PER_GROUP: Final[int] = 3
REPORT_ONLY_LABELS: Final[tuple[str, ...]] = ("D15", "D18")
LEAD_IN: Final[timedelta] = timedelta(days=3)
LEAD_OUT: Final[timedelta] = timedelta(days=3)
MAX_SPAN: Final[timedelta] = timedelta(days=10)
ORIGIN_GAP: Final[timedelta] = timedelta(hours=24)
HALF_HOUR: Final[timedelta] = timedelta(minutes=30)
ARM_PANELS: Final[tuple[tuple[str, str], ...]] = (
    ("r0_lagged", "with lags (R0)"),
    ("r0_lagless", "without lags (R0)"),
    ("r1_lagged", "with lags, event rows removed from training (R1)"),
)
FAN_ARMS: Final[tuple[tuple[str, str], ...]] = (
    ("r0_lagged", "with lags (R0)"),
    ("r1_lagged", "with lags, event rows removed from training (R1)"),
)
NOT_IN_TABLE_NOTE: Final[str] = "Events not in the label table may also be present."
PANEL_HEIGHT_PX: Final[int] = 150
PLOT_WIDTH_PX: Final[int] = CONTENT_WIDTH_PX - 70
FORECAST_COLOUR: Final[str] = ocf.DATA_BLUE
FORECAST_BAND_COLOUR: Final[str] = ocf.DATA_BLUE_LIGHT
EVENT_COLOUR: Final[str] = ocf.GREY_3
ACTUAL_COLOUR: Final[str] = ocf.BLACK_1


GROUP_OF: Final[dict[str, str]] = {
    **dict.fromkeys(SIGNED_MW_LABELS, "signed-MW"),
    **dict.fromkeys(MVA_ONLY_LABELS, "MVA-only"),
    **dict.fromkeys(REPORT_ONLY_LABELS, "report-only"),
}
GROUP_PRIORITY: Final[dict[str, int]] = {"signed-MW": 0, "MVA-only": 1, "report-only": 2}
MIN_EVENTS: Final[int] = 6
SAME_TRANSFER_HOURS: Final[float] = 12.0


def candidate_events(*, events: pl.DataFrame) -> pl.DataFrame:
    """Return every labelled event starting in a screen fold, deepest first, with its group."""
    in_folds = pl.any_horizontal(
        (pl.col("start") >= lo) & (pl.col("start") < hi)
        for lo, hi in zip(score.FOLD_STARTS, score.FOLD_ENDS, strict=True)
    )
    return (
        events.filter(pl.col("kind").str.starts_with("labelled"), in_folds)
        .filter(pl.col("label").is_in(list(GROUP_OF)))
        .with_columns(group=pl.col("label").replace_strict(GROUP_OF, return_dtype=pl.String))
        .sort("depth_abs", descending=True)
    )


def is_same_transfer(*, first: dict, second: dict) -> bool:
    """Return whether two events at different Primaries are one transfer seen twice."""
    return (
        first["label"] != second["label"]
        and abs((first["start"] - second["start"]).total_seconds()) <= SAME_TRANSFER_HOURS * 3600
        and abs(first["depth_abs"] - second["depth_abs"]) < 0.01
    )


def choose_events(*, usable: list[dict]) -> list[dict]:
    """Pick the events to draw from `usable` (deepest first, each with a run before its start).

    The three deepest of each metering group are kept, dropping an event that repeats a deeper
    event's transfer at another Primary (the signed-MW Primary is preferred). If fewer than
    `MIN_EVENTS` remain, report-only Primaries' events (then the dropped repeats) are added,
    deepest first. The result is sorted by depth, deepest first.
    """
    ordered = sorted(usable, key=lambda e: (GROUP_PRIORITY[e["group"]], -e["depth_abs"]))
    kept: list[dict] = []
    repeats: list[dict] = []
    for event in ordered:
        if any(is_same_transfer(first=event, second=other) for other in kept):
            repeats.append(event)
        else:
            kept.append(event)
    for event in repeats:
        print(
            f"dropped {event['label']} {event['start']:%Y-%m-%d}: the same transfer as another "
            "chosen Primary's event"
        )
    chosen = [
        e
        for group in ("signed-MW", "MVA-only")
        for e in [k for k in kept if k["group"] == group][:EVENTS_PER_GROUP]
    ]
    for group in ("signed-MW", "MVA-only"):
        count = sum(e["group"] == group for e in chosen)
        if count < EVENTS_PER_GROUP:
            print(f"group {group}: only {count} usable event(s); using what exists")
    extras = sorted([k for k in kept if k["group"] == "report-only"], key=lambda e: -e["depth_abs"])
    while len(chosen) < MIN_EVENTS and extras:
        chosen.append(extras.pop(0))
    if len(chosen) < MIN_EVENTS:
        for event in sorted(repeats, key=lambda e: -e["depth_abs"]):
            if len(chosen) < MIN_EVENTS:
                print(f"nothing else left: drawing the repeat {event['label']} too")
                chosen.append(event)
    return sorted(chosen, key=lambda e: -e["depth_abs"])


def origin_for(*, forecasts: pl.DataFrame, onset: datetime) -> datetime | None:
    """Return the latest run at least 24 h before the onset, else the latest run before it."""
    inits = forecasts["power_fcst_init_time"]
    early = cast("datetime | None", inits.filter(inits <= onset - ORIGIN_GAP).max())
    if early is not None:
        return early
    return cast("datetime | None", inits.filter(inits < onset).max())


def load_forecasts(*, label: str, arm: str, series: pl.DataFrame) -> pl.DataFrame:
    """Return every stored forecast (MW) of one Primary and arm across the available screen folds."""
    pieces = []
    for fold in range(len(score.FOLD_STARTS)):
        frame = score.read_fold(directory=score.SOURCES[arm], sites=[label], fold=fold)
        if frame is not None:
            pieces.append(adapter.forecast_frame(stored=frame, series=series, unit="fraction"))
    return pl.concat(pieces).unique(subset=list(score.KEY_COLUMNS))


def forecast_for_window(
    *, forecasts: pl.DataFrame, start: datetime, origin: datetime, window: pl.DataFrame
) -> pl.DataFrame:
    """Attach to each half-hour of `window` the forecast chosen by the module docstring's rule.

    From `start` on, the run at `origin`. Before it, the latest run at least 24 hours before the
    half-hour, or where none exists the latest run at or before the half-hour.
    """
    inits = forecasts.select("power_fcst_init_time").unique().sort("power_fcst_init_time")
    wanted = window.sort("valid_time").with_columns(
        cutoff=pl.col("valid_time") - ORIGIN_GAP, cutoff_late=pl.col("valid_time")
    )
    first = wanted.join_asof(
        inits, left_on="cutoff", right_on="power_fcst_init_time", strategy="backward"
    ).rename({"power_fcst_init_time": "init_early"})
    second = first.join_asof(
        inits, left_on="cutoff_late", right_on="power_fcst_init_time", strategy="backward"
    )
    chosen = second.with_columns(
        power_fcst_init_time=pl.when(pl.col("valid_time") >= start)
        .then(pl.lit(origin))
        .otherwise(pl.coalesce("init_early", "power_fcst_init_time"))
    ).select("valid_time", "power_fcst_init_time")
    return chosen.join(
        forecasts.drop("time_series_id"),
        on=["power_fcst_init_time", "valid_time"],
        how="left",
    ).sort("valid_time")


def days_from(*, column: str, onset: datetime) -> pl.Expr:
    """Return the column as fractional days since `onset`."""
    return (pl.col(column) - pl.lit(onset)).dt.total_seconds() / 86400.0


def event_layers(*, onset_days: float, end_days: float, show_labels: bool) -> list:
    """Return the grey event band and its labelled start and end rules."""
    band = (
        alt.Chart(pl.DataFrame({"a": [onset_days], "b": [end_days]}))  # ty: ignore[unresolved-attribute]
        .mark_rect(color=EVENT_COLOUR, opacity=0.6, aria=False)
        .encode(x="a:Q", x2="b:Q")
    )
    rules = (
        alt.Chart(pl.DataFrame({"x": [onset_days, end_days]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color=ocf.BLACK_1, strokeDash=[4, 3], aria=False)
        .encode(x="x:Q")
    )
    layers = [band, rules]
    if show_labels:
        for position, text, align in ((onset_days, "start", "right"), (end_days, "end", "left")):
            layers.append(
                alt.Chart(pl.DataFrame({"x": [position], "text": [text]}))  # ty: ignore[unresolved-attribute]
                .mark_text(
                    dx=-3 if align == "right" else 3,
                    align=align,
                    baseline="top",
                    fontSize=10,
                    aria=False,
                    color=ocf.BLACK_1,
                )
                .encode(x="x:Q", y=alt.value(2), text="text:N")
            )
    return layers


def arm_panel(
    *, rows: pl.DataFrame, title: str, bounds: tuple[float, float], events: tuple[float, float]
) -> alt.LayerChart | alt.FacetChart:
    """Return one stacked panel: event band, actuals, forecast median, and 5-95% band."""
    x = alt.X(
        "day:Q",
        title="Days from the event's start",
        scale=alt.Scale(domain=list(bounds), nice=False),
        axis=alt.Axis(tickMinStep=1, format="d"),
    )
    y = alt.Y("value:Q", title="MW", scale=alt.Scale(zero=False))
    base = alt.Chart(rows)
    band = base.mark_area(color=FORECAST_BAND_COLOUR, opacity=0.8, aria=False).encode(  # ty: ignore[unresolved-attribute]
        x=x, y="q05:Q", y2="q95:Q"
    )
    median = base.mark_line(color=FORECAST_COLOUR, strokeWidth=1.5, aria=False).encode(  # ty: ignore[unresolved-attribute]
        x=x, y=alt.Y("q50:Q", title="MW", scale=alt.Scale(zero=False))
    )
    actual = base.mark_line(color=ACTUAL_COLOUR, strokeWidth=1, aria=False).encode(x=x, y=y)  # ty: ignore[unresolved-attribute]
    return alt.layer(
        *event_layers(onset_days=events[0], end_days=events[1], show_labels=True),
        band,
        median,
        actual,
        width=PLOT_WIDTH_PX,
        height=PANEL_HEIGHT_PX,
        title=alt.TitleParams(
            title, anchor="start", fontSize=ocf.font_size(style="Body", body_px=11)
        ),
    )


def caption(*, title: str, subtitle: str) -> alt.TitleParams:
    """Return a wrapped figure title with its subtitle."""
    return alt.TitleParams(
        wrapped(text=title, width=95),
        subtitle=wrapped(text=subtitle, width=105),
        anchor="start",
        offset=12,
        subtitleColor=ocf.BLACK_1,
    )


def event_figure(
    *,
    event: dict,
    number: int,
    letter: str,
    actuals: pl.DataFrame,
    forecasts_by_arm: dict[str, pl.DataFrame],
    origin: datetime,
) -> alt.VConcatChart:
    """Return one event's figure of three stacked arm panels."""
    onset, end = event["start"], event["end"]
    first = onset - LEAD_IN
    last = min(end + LEAD_OUT, first + MAX_SPAN)
    grid = pl.datetime_range(first, last, interval="30m", time_zone="UTC", eager=True)
    window = pl.DataFrame({"valid_time": grid})
    sid = event["time_series_id"]
    own = actuals.filter(pl.col("time_series_id") == sid).select("valid_time", "actual")
    panels = []
    for arm, name in ARM_PANELS:
        chosen = forecast_for_window(
            forecasts=forecasts_by_arm[arm], start=onset, origin=origin, window=window
        )
        rows = chosen.join(own, on="valid_time", how="left").with_columns(
            day=days_from(column="valid_time", onset=onset), value=pl.col("actual")
        )
        panels.append(
            arm_panel(
                rows=rows.select("day", "value", "q05", "q50", "q95"),
                title=name,
                bounds=(-3.0, (last - onset).total_seconds() / 86400.0),
                events=(0.0, (end - onset).total_seconds() / 86400.0),
            )
        )
    depth = event["depth_abs"]
    flags = "".join(
        (
            " (report-only Primary)" if event["group"] == "report-only" else "",
            " (low-confidence label)" if event["kind"] == "labelled low-confidence" else "",
        )
    )
    hours = (onset - origin).total_seconds() / 3600.0
    title = f"Substation {letter}{flags}: forecasts through a load-shift event of {depth:.1f} MW labelled depth"
    subtitle = (
        f"Event {number}, {event['group']} group. MW is power drawn from the Primary. Black: actual. "
        "Blue line: forecast median. Blue band: 5% to 95% interval. Grey band: the labelled event. "
        f"From the start onwards: the run {hours:.1f} hours before the start"
        f"{'' if hours >= 24 else ' (no run exists 24 hours before)'}; "
        "earlier half-hours: the run 24 hours before each (the latest earlier run if none). "
        f"{NOT_IN_TABLE_NOTE}"
    )
    return (
        alt.vconcat(*panels, spacing=24)
        .properties(title=caption(title=title, subtitle=subtitle))
        .configure_view(stroke=None)
    )


def fan_panel(
    *, rows: pl.DataFrame, title: str, bounds: tuple[float, float], events: tuple[float, float]
) -> alt.LayerChart | alt.FacetChart:
    """Return one fan panel: event band, actuals, and the q05-q95, q25-q75 bands and median."""
    x = alt.X(
        "day:Q",
        title="Days from the event's start",
        scale=alt.Scale(domain=list(bounds), nice=False),
        axis=alt.Axis(tickMinStep=1, format="d"),
    )
    y_scale = alt.Scale(zero=False)
    base = alt.Chart(rows)
    outer = base.mark_area(color=FORECAST_BAND_COLOUR, opacity=0.7, aria=False).encode(  # ty: ignore[unresolved-attribute]
        x=x, y=alt.Y("q05:Q", title="MW", scale=y_scale), y2="q95:Q"
    )
    inner = base.mark_area(color=FORECAST_COLOUR, opacity=0.35, aria=False).encode(  # ty: ignore[unresolved-attribute]
        x=x, y=alt.Y("q25:Q", scale=y_scale), y2="q75:Q"
    )
    median = base.mark_line(color=FORECAST_COLOUR, strokeWidth=1.5, aria=False).encode(  # ty: ignore[unresolved-attribute]
        x=x, y=alt.Y("q50:Q", scale=y_scale)
    )
    actual = base.mark_line(color=ACTUAL_COLOUR, strokeWidth=1, aria=False).encode(  # ty: ignore[unresolved-attribute]
        x=x, y=alt.Y("value:Q", scale=y_scale)
    )
    return alt.layer(
        *event_layers(onset_days=events[0], end_days=events[1], show_labels=True),
        outer,
        inner,
        median,
        actual,
        width=PLOT_WIDTH_PX,
        height=PANEL_HEIGHT_PX + 20,
        title=alt.TitleParams(
            title, anchor="start", fontSize=ocf.font_size(style="Body", body_px=11)
        ),
    )


def fan_figure(
    *,
    event: dict,
    letter: str,
    actuals: pl.DataFrame,
    forecasts_by_arm: dict[str, pl.DataFrame],
) -> alt.VConcatChart:
    """Return the fan figure: one origin, one day before the event's start, two arms."""
    onset, end = event["start"], event["end"]
    first = onset - timedelta(days=1)
    last = min(end + LEAD_OUT, onset + timedelta(days=9))
    grid = pl.datetime_range(first, last, interval="30m", time_zone="UTC", eager=True)
    own = actuals.filter(pl.col("time_series_id") == event["time_series_id"]).select(
        "valid_time", "actual"
    )
    panels = []
    for arm, name in FAN_ARMS:
        forecasts = forecasts_by_arm[arm]
        origin = origin_for(forecasts=forecasts, onset=onset)
        assert origin is not None
        print(f"fan origin for {arm}: {(onset - origin).total_seconds() / 3600:.1f} h before start")
        rows = (
            pl.DataFrame({"valid_time": grid})
            .join(
                forecasts.filter(pl.col("power_fcst_init_time") == origin).drop("time_series_id"),
                on="valid_time",
                how="left",
            )
            .join(own, on="valid_time", how="left")
            .with_columns(day=days_from(column="valid_time", onset=onset), value=pl.col("actual"))
        )
        panels.append(
            fan_panel(
                rows=rows.select("day", "value", "q05", "q25", "q50", "q75", "q95"),
                title=name,
                bounds=(-1.0, (last - onset).total_seconds() / 86400.0),
                events=(0.0, (end - onset).total_seconds() / 86400.0),
            )
        )
    title = f"Substation {letter}: forecast fan from the origin one day before the event's start"
    subtitle = (
        "MW is power drawn from the Primary. Black: actual. Blue line: median. Dark blue band: 25% to "
        "75% interval. Light blue band: 5% to 95% interval. Grey band: the labelled event, "
        f"{event['depth_abs']:.1f} MW deep. {NOT_IN_TABLE_NOTE}"
    )
    return (
        alt.vconcat(*panels, spacing=24)
        .properties(title=caption(title=title, subtitle=subtitle))
        .configure_view(stroke=None)
    )


def save(*, chart: alt.TopLevelMixin, path: Path) -> None:
    """Write the chart as SVG and optimise it with svgo."""
    chart.save(path)
    subprocess.run(
        ["npx", "svgo@4", "--multipass", "--precision=1", "--final-newline", str(path)],
        check=True,
        capture_output=True,
    )
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")


def main() -> None:
    """Choose the events, draw the figures, and print the notes."""
    series = adapter.series_table(labels=adapter.PRIMARY_LABELS)
    actuals, _, _ = score.load_actuals(source="cache", series=series)
    _, events = score.event_tables(series=series)
    cache: dict[tuple[str, str], pl.DataFrame] = {}
    arms = sorted({arm for arm, _ in ARM_PANELS})
    usable: list[dict] = []
    for event in candidate_events(events=events).to_dicts():
        label = event["label"]
        for arm in arms:
            if (label, arm) not in cache:
                cache[label, arm] = load_forecasts(label=label, arm=arm, series=series)
        by_arm = {arm: cache[label, arm] for arm in arms}
        origins = [origin_for(forecasts=by_arm[arm], onset=event["start"]) for arm in arms]
        if any(o is None for o in origins):
            print(f"skipped {label} {event['start']:%Y-%m-%d}: no run before the start")
            continue
        usable.append({**event, "by_arm": by_arm})
    chosen = choose_events(usable=usable)
    letters: dict[str, str] = {}
    for event in chosen:
        letters.setdefault(event["label"], string.ascii_uppercase[len(letters)])
    print("Substation key (kept out of the figures):", letters)
    print(
        "Notes: forecast for half-hours at or after the start = the latest run at least 24 h before "
        "the start (else the latest run before it; the hours are in each subtitle); earlier "
        "half-hours = the latest run at least 24 h before each half-hour (else the latest earlier "
        "run)."
    )
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    for number, event in enumerate(chosen, start=1):
        label = event["label"]
        by_arm = event["by_arm"]
        origin = origin_for(forecasts=by_arm["r0_lagged"], onset=event["start"])
        assert origin is not None
        print(
            f"event {number}: substation {letters[label]}, {event['group']}, start "
            f"{event['start']:%Y-%m-%d}, end {event['end']:%Y-%m-%d %H:%M}, depth "
            f"{event['depth_abs']:.2f} MW, kind {event['kind']}, origin "
            f"{(event['start'] - origin).total_seconds() / 3600:.1f} h before start"
        )
        save(
            chart=event_figure(
                event=event,
                number=number,
                letter=letters[label],
                actuals=actuals,
                forecasts_by_arm=by_arm,
                origin=origin,
            ),
            path=DOCS_DIR / f"figure_events_{number}.svg",
        )
    deepest = chosen[0]
    save(
        chart=fan_figure(
            event=deepest,
            letter=letters[deepest["label"]],
            actuals=actuals,
            forecasts_by_arm=deepest["by_arm"],
        ),
        path=DOCS_DIR / "figure_examples.svg",
    )


if __name__ == "__main__":
    main()
