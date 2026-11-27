# ruff: noqa: E501
"""Experiment 733: the `series` stage, forecast time series against the truth (no torch).

Two groups of scenarios are drawn, each at least once, and no window is drawn twice.

- Real folds at D8 (calendar dates may appear, as experiment 713's pages show them): a typical
  origin of a winter fold and of a summer fold, each as a day-ahead panel and a 14-day panel for FU,
  FS, and M*; the origin where the untangled channels move FU's day-ahead forecast most, for FU,
  FU with the channels masked (R4), and M*; and logged load-shift events, with the lead-day-2
  forecast of FU and M* from 14 days before to 7 days after each event.
- The synthetic world: the probability of reverse flow of YU, Y0, and B-hist over the 9 target days
  of lead days 2 to 10, against the true sign, for three S2 segments with reverse flow and for one
  S2 segment without. A window of source S1 is never drawn, and a panel carries no segment number,
  date, or capacity.

On validation rows a scenario whose season is absent from the rows falls back to the nearest
available month and says so, so that every code path runs on real files. On test rows a missing
scenario is an error.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Final

import e00733_baselines as baselines
import e00733_figures as figs
import e00733_scoring as scoring
import e00733_store as store
import numpy as np
import polars as pl

# isort: split
import altair as alt

WINTER_MONTHS: Final = (12, 1, 2)
SUMMER_MONTHS: Final = (5, 6, 7, 8)
DAY_AHEAD_LEAD_HOURS: Final = (15.0, 39.0)
"""Lead hours (from the 09:00 UTC origin) of the half-hours of the next UTC day."""
EVENT_DAYS_BEFORE: Final = 14
EVENT_DAYS_AFTER: Final = 7
MAX_EVENT_PANELS: Final = 8
N_SIGN_REVERSE_PANELS: Final = 3
REAL_ARMS: Final = (("FU", None), ("FS", None), ("M*", None))
"""The arms of the winter and summer panels."""
QUANTILES: Final = ("q10", "q50", "q90")
SEED: Final = 0


class SeriesError(RuntimeError):
    """A scenario has nothing to draw or a window repeats."""


@dataclass(frozen=True)
class Window:
    """One drawn window.

    Attributes:
        scenario: The scenario name.
        key: What identifies the window (a fold and an origin, an event start, or a segment and an origin).
    """

    scenario: str
    key: tuple


def assert_distinct(*, windows: Sequence[Window]) -> None:
    """Check that no key is drawn twice, within a scenario or across scenarios of one kind.

    Args:
        windows: Every drawn window.

    Raises:
        SeriesError: When two windows share a key.
    """
    seen: dict[tuple, str] = {}
    for w in windows:
        if w.key in seen:
            msg = f"the window {w.key} is drawn twice ({seen[w.key]} and {w.scenario})"
            raise SeriesError(msg)
        seen[w.key] = w.scenario


# --------------------------------------------------------------------------- real folds


def read_real(
    *, root: Path, rows: str, d8: int, fold: str, seed: int = SEED
) -> dict[str, pl.DataFrame]:
    """Return D8's forecasts of FU, FU with the channels masked, FS, and M* in one fold.

    An arm whose stored forecast does not exist is left out. On validation rows F0r stands in for M*.
    Every frame holds `power_fcst_init_time`, `valid_time`, `lead_hours`, the three quantiles drawn,
    and `actual` (FU's cleaned truth for every arm).

    Args:
        root: This experiment's data folder.
        rows: "test" or "validation".
        d8: D8's series id.
        fold: The fold month.
        seed: The seed.

    Returns:
        The frames by label ("FU", "FU masked", "FS", "M*").

    Raises:
        SeriesError: When FU's forecast of the fold is missing.
    """
    want = {
        "FU": ("FU", None),
        "FU masked": ("FU", "masked"),
        "FS": ("FS", None),
        "M*": ("M*", None),
    }
    out: dict[str, pl.DataFrame] = {}
    for label, arm in want.items():
        name, variant = scoring.swap_reference(arm=arm, rows=rows)
        if not scoring.has_job(root=root, arm=(name, variant), fold=fold, seed=seed, rows=rows):
            continue
        if name == "M*":
            frame = pl.read_parquet(baselines.m_star_file(month=fold, seed=seed))
        else:
            frame = store.read_set(
                root=root,
                name=store.set_name(arm=name, fold=fold, seed=seed, variant=variant, rows=rows),
                rows=rows,
            )
        out[label] = frame.filter(pl.col("time_series_id") == d8)
    if "FU" not in out or out["FU"].is_empty():
        msg = f"the forecast of FU at D8 in fold {fold} is missing, so no series can be drawn"
        raise SeriesError(msg)
    actual = out["FU"].select("power_fcst_init_time", "valid_time", "actual")
    for label, frame in list(out.items()):
        keep = [
            c
            for c in ("power_fcst_init_time", "valid_time", "lead_hours", *QUANTILES)
            if c in frame.columns
        ]
        out[label] = (
            frame.select(keep)
            .join(actual, on=["power_fcst_init_time", "valid_time"], how="inner")
            .sort("power_fcst_init_time", "valid_time")
        )
    return out


def mean_abs_error(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return each origin's mean absolute error of the median over its horizon."""
    return (
        frame.filter(pl.col("actual").is_not_null() & pl.col("actual").is_not_nan())
        .group_by("power_fcst_init_time")
        .agg(error=(pl.col("q50") - pl.col("actual")).abs().mean(), n=pl.len())
        .filter(pl.col("n") >= 48)
        .sort("power_fcst_init_time")
    )


def pick_typical_origin(
    *,
    frame: pl.DataFrame,
    months: Sequence[int],
    allow_fallback: bool,
    exclude: set[datetime] | None = None,
) -> tuple[datetime, str]:
    """Return the origin whose mean absolute error is the median of the origins of the months.

    A typical origin avoids drawing the best or the worst case by choice. Ties go to the earliest.

    Args:
        frame: FU's frame of one or more folds.
        months: The calendar months of the season.
        allow_fallback: Whether to take the nearest available month when the season is absent.
        exclude: The origins already drawn, which are not picked again.

    Returns:
        The origin and a note ("" or the fallback used).

    Raises:
        SeriesError: When no origin exists in the season and no fallback is allowed.
    """
    table = mean_abs_error(frame=frame).filter(
        ~pl.col("power_fcst_init_time").is_in(list(exclude or ()))
    )
    month = table["power_fcst_init_time"].dt.month()
    inside = table.filter(month.is_in(list(months)))
    note = ""
    if inside.is_empty():
        if not allow_fallback or table.is_empty():
            msg = f"no origin lies in the months {tuple(months)}"
            raise SeriesError(msg)
        target = float(np.mean(months))
        offsets = (table["power_fcst_init_time"].dt.month().cast(pl.Float64) - target).abs()
        nearest = table.filter(offsets == offsets.min())
        note = f"no origin in months {tuple(months)}; drew the nearest available month instead"
        inside = nearest
    order = inside.sort("error", "power_fcst_init_time")
    return order["power_fcst_init_time"][(order.height - 1) // 2], note


def pick_channel_origin(
    *, with_channels: pl.DataFrame, masked: pl.DataFrame, exclude: set[datetime]
) -> datetime:
    """Return the origin where the channels move FU's day-ahead median most (mean absolute difference).

    Args:
        with_channels: FU's frame.
        masked: FU's frame with every channel masked at prediction (R4).
        exclude: The origins already drawn.

    Returns:
        The origin.

    Raises:
        SeriesError: When no origin outside `exclude` has a day-ahead forecast in both frames.
    """
    lo, hi = DAY_AHEAD_LEAD_HOURS
    a = with_channels.filter((pl.col("lead_hours") >= lo) & (pl.col("lead_hours") < hi)).select(
        "power_fcst_init_time", "valid_time", a=pl.col("q50")
    )
    b = masked.filter((pl.col("lead_hours") >= lo) & (pl.col("lead_hours") < hi)).select(
        "power_fcst_init_time", "valid_time", b=pl.col("q50")
    )
    diff = (
        a.join(b, on=["power_fcst_init_time", "valid_time"])
        .group_by("power_fcst_init_time")
        .agg(d=(pl.col("a") - pl.col("b")).abs().mean(), n=pl.len())
        .filter(pl.col("n") >= 40)
        .filter(~pl.col("power_fcst_init_time").is_in(list(exclude)))
        .sort("d", "power_fcst_init_time", descending=[True, False])
    )
    if diff.is_empty():
        msg = "no origin outside the drawn ones holds both FU and masked FU day-ahead forecasts"
        raise SeriesError(msg)
    return diff["power_fcst_init_time"][0]


def lead_day_two_series(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return the day-ahead forecast of each valid half-hour (the origin of the day before), with the truth."""
    lo, hi = DAY_AHEAD_LEAD_HOURS
    return frame.filter((pl.col("lead_hours") >= lo) & (pl.col("lead_hours") < hi)).sort(
        "valid_time"
    )


def event_windows(
    *, events: pl.DataFrame, valid_times: pl.Series, cap: int = MAX_EVENT_PANELS
) -> tuple[list[dict[str, Any]], int]:
    """Return the events to draw and the number of events with forecasts.

    An event is drawn when forecasts exist at its start and at its end. When more than `cap` events
    qualify, an evenly spaced subset is drawn.

    Args:
        events: Rows `start` and `end`.
        valid_times: The valid times of the lead-day-2 series.
        cap: The most panels to draw.

    Returns:
        The drawn events (start, end, window first and last) and the number that qualified.
    """
    have = set(valid_times.to_list())
    ok = [
        r
        for r in events.sort("start").iter_rows(named=True)
        if r["start"] in have and r["end"] in have
    ]
    n = len(ok)
    if n > cap:
        ok = [ok[int(i)] for i in np.linspace(0, n - 1, cap).round()]
    return [
        {
            **r,
            "first": r["start"] - timedelta(days=EVENT_DAYS_BEFORE),
            "last": r["end"] + timedelta(days=EVENT_DAYS_AFTER),
        }
        for r in ok
    ], n


def d8_events(*, d8: int) -> pl.DataFrame:
    """Return D8's logged load-shift events (padded, hand-filtered) as `start` and `end` rows."""
    import e00377_adapter as adapter  # ty: ignore[unresolved-import]
    from studies.dp_features import default_input_paths, load_or_build_feature_cache
    from studies.sources import ML_EXPERIMENTS_DATA_DIR

    series = adapter.series_table().filter(pl.col("time_series_id") == d8)
    cache = load_or_build_feature_cache(
        paths=default_input_paths(), cache_dir=ML_EXPERIMENTS_DATA_DIR / scoring.FEATURE_CACHE
    )
    return adapter.events_from_masks(cache=cache, series=series).select("start", "end")


# --------------------------------------------------------------------------- real charts


def _long(
    *, frames: Mapping[str, pl.DataFrame], origin: datetime | None, lead: tuple[float, float] | None
) -> pl.DataFrame:
    parts = []
    for label, frame in frames.items():
        part = frame if origin is None else frame.filter(pl.col("power_fcst_init_time") == origin)
        if lead is not None:
            part = part.filter((pl.col("lead_hours") >= lead[0]) & (pl.col("lead_hours") < lead[1]))
        parts.append(part.with_columns(series=pl.lit(label)))
    return pl.concat(parts) if parts else pl.DataFrame()


def forecast_chart(
    *,
    frames: Mapping[str, pl.DataFrame],
    origin: datetime | None,
    lead: tuple[float, float] | None,
    title: str,
    band_of: str = "FU",
    window: tuple[datetime, datetime] | None = None,
    marks: tuple[datetime, datetime] | None = None,
) -> alt.LayerChart | alt.FacetChart:
    """Return one forecast panel: median lines of each arm, FU's 10 to 90% band, and the truth.

    Args:
        frames: Each arm's frame.
        origin: The origin to draw (None: every origin, for a lead-day-2 series).
        lead: The lead-hour range to draw, or None for the whole horizon.
        title: The panel title.
        band_of: The arm that gets the band.
        window: The valid-time range to draw, or None.
        marks: A logged event's start and end, drawn as rules.

    Returns:
        The chart.

    Raises:
        SeriesError: When nothing falls in the panel.
    """
    data = _long(frames=frames, origin=origin, lead=lead)
    if window is not None and not data.is_empty():
        data = data.filter(
            (pl.col("valid_time") >= window[0]) & (pl.col("valid_time") <= window[1])
        )
    if data.is_empty():
        msg = f"the panel {title!r} holds no row"
        raise SeriesError(msg)
    names = list(dict.fromkeys(data["series"].to_list()))
    colour = alt.Color(
        "series:N",
        scale=alt.Scale(
            domain=[*names, "Reading"],
            range=[
                figs.ARM_COLOURS.get(n.split(" ")[0], "#555555")
                if n != "FU masked"
                else figs.figures729.PURPLE
                for n in names
            ]
            + ["#000000"],
        ),
        legend=alt.Legend(orient="top", title=None),
    )
    x = alt.X("valid_time:T", title="Valid time (UTC)")
    lines = (
        alt.Chart(data)
        .mark_line(aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=x,
            y=alt.Y("q50:Q", title="Reading (share of capacity)", scale=alt.Scale(zero=False)),
            color=colour,
        )
    )  # ty: ignore[unresolved-attribute]
    band = (
        alt.Chart(data.filter(pl.col("series") == band_of))
        .mark_area(opacity=0.2, color=figs.figures729.BLUE)
        .encode(x=x, y="q10:Q", y2="q90:Q")  # ty: ignore[unresolved-attribute]
    )  # ty: ignore[unresolved-attribute]
    truth_rows = (
        data.unique(subset=["valid_time"])
        .select("valid_time", q50=pl.col("actual"))
        .with_columns(series=pl.lit("Reading"))
    )
    truth = (
        alt.Chart(truth_rows)
        .mark_line(aria=False, strokeWidth=1.5)
        .encode(x=x, y="q50:Q", color=colour)  # ty: ignore[unresolved-attribute]
    )  # ty: ignore[unresolved-attribute]
    layers = [band, lines, truth]
    if marks is not None:
        rules = (
            alt.Chart(pl.DataFrame({"t": list(marks)}))
            .mark_rule(color="#555555", strokeDash=[3, 3])
            .encode(x="t:T")  # ty: ignore[unresolved-attribute]
        )  # ty: ignore[unresolved-attribute]
        layers.append(rules)
    return alt.layer(*layers).properties(
        width=figs.WIDTH_PX, height=130, title=alt.TitleParams(title, anchor="start")
    )


# --------------------------------------------------------------------------- synthetic sign


@dataclass(frozen=True)
class SignWindow:
    """One origin of the synthetic world: a segment, an origin day, and the sample rows of its 9 target days."""

    kind: str
    seg: int
    origin_day: int
    rows: tuple[int, ...]


def pick_sign_windows(*, data: Any) -> list[SignWindow]:
    """Return three S2 windows with reverse flow and one S2 window without, never from source S1.

    A window is a segment's origin with the samples of every scored lead day. The windows with the
    most reverse-flow half-hours (then the most scored days, then the lowest segment) are taken from
    three different segments. The window without reverse flow has no reverse-flow half-hour and the
    highest mean probability of B-hist, which is the hardest case for a false alarm.

    Args:
        data: `e00733_signdata.SignData`.

    Returns:
        The four windows, the reverse-flow windows first.

    Raises:
        SeriesError: When fewer than three S2 segments have reverse flow, or none lacks it.
    """
    idx = data.index
    scored = np.flatnonzero(data.keep.any(axis=1))
    groups: dict[tuple[int, int], list[int]] = {}
    for r in scored:
        groups.setdefault((int(idx.seg[r]), int(idx.origin_day[r])), []).append(int(r))
    cand = []
    for (seg, day), rows in groups.items():
        if data.sources[seg] != "S2":
            continue
        n_rev = int(np.nansum(data.truth[rows] * data.keep[rows]))
        mean_p = float(np.mean(data.probabilities["B-hist"][rows][data.keep[rows]]))
        cand.append(
            (seg, day, tuple(sorted(rows, key=lambda r: int(idx.lead_index[r]))), n_rev, mean_p)
        )
    with_flow = sorted((c for c in cand if c[3] > 0), key=lambda c: (-c[3], -len(c[2]), c[0], c[1]))
    chosen: list[SignWindow] = []
    used: set[int] = set()
    for seg, day, rows, _n, _p in with_flow:
        if seg in used:
            continue
        chosen.append(SignWindow("reverse flow", seg, day, rows))
        used.add(seg)
        if len(chosen) == N_SIGN_REVERSE_PANELS:
            break
    if len(chosen) < N_SIGN_REVERSE_PANELS:
        msg = f"only {len(chosen)} S2 segments have reverse flow on scored days, and the figure needs {N_SIGN_REVERSE_PANELS}"
        raise SeriesError(msg)
    without = sorted(
        (c for c in cand if c[3] == 0 and c[0] not in used),
        key=lambda c: (-len(c[2]), -c[4], c[0], c[1]),
    )
    if not without:
        msg = "no S2 segment has a scored origin without reverse flow"
        raise SeriesError(msg)
    seg, day, rows, _n, _p = without[0]
    return [*chosen, SignWindow("no reverse flow", seg, day, rows)]


def sign_series_for(*, data: Any, window: SignWindow) -> pl.DataFrame:
    """Return the rows of one sign panel (the long frame `figs.sign_series_rows` gives, for 9 days)."""
    rows = list(window.rows)
    truth = np.where(data.keep[rows], data.truth[rows], np.nan).reshape(-1)
    forecasts = {
        name: np.where(data.keep[rows], data.probabilities[name][rows], np.nan).reshape(-1)
        for name in ("YU", "Y0", "B-hist")
    }
    return figs.sign_series_rows(truth=truth, forecasts=forecasts)


def sign_figure(*, data: Any, windows: Sequence[SignWindow]) -> alt.VConcatChart:
    """Return the sign-forecast figure: one panel per window, named by letter, with no segment number or date."""
    panels = []
    for i, w in enumerate(windows):
        title = f"Synthetic segment {'ABCDEFGH'[i]} ({w.kind}), lead days 2 to 10"
        panels.append(
            figs.sign_series_chart(rows=sign_series_for(data=data, window=w), title=title)
        )
    return alt.vconcat(*panels)


# --------------------------------------------------------------------------- the stage

Folds = dict[str, dict[str, pl.DataFrame]]


def season_figure(
    *, folds: Folds, allow_fallback: bool
) -> tuple[alt.VConcatChart, list[Window], list[str]]:
    """Return the winter and summer panels (day-ahead and 14-day) of a typical origin each."""
    fu_all = pl.concat([f["FU"] for f in folds.values()])
    panels, windows, notes = [], [], []
    for season, months in (("winter", WINTER_MONTHS), ("summer", SUMMER_MONTHS)):
        origin, note = pick_typical_origin(
            frame=fu_all,
            months=months,
            allow_fallback=allow_fallback,
            exclude={w.key[1] for w in windows},
        )
        if note:
            notes.append(f"{season}: {note}")
        fold = next(
            f for f, fr in folds.items() if (fr["FU"]["power_fcst_init_time"] == origin).any()
        )
        frames = {k: v for k, v in folds[fold].items() if k in ("FU", "FS", "M*")}
        windows.append(Window(f"{season} origin", (fold, origin)))
        d = origin.date()
        for lead, what in ((DAY_AHEAD_LEAD_HOURS, "day-ahead"), (None, "14-day")):
            label = f"{season.capitalize()}{' (fallback month, see notes)' if note else ''}"
            title = f"{label}: the {what} forecast of the origin of {d}"
            panels.append(forecast_chart(frames=frames, origin=origin, lead=lead, title=title))
    return alt.vconcat(*panels), windows, notes


def channel_figure(
    *, folds: Folds, exclude: set[datetime]
) -> tuple[alt.LayerChart | alt.FacetChart, Window]:
    """Return the panel of the origin where the untangled channels move FU's day-ahead forecast most.

    Args:
        folds: Each fold's frames by arm.
        exclude: The origins already drawn.

    Returns:
        The chart and its window.

    Raises:
        SeriesError: When no fold holds both FU and masked FU.
    """
    lo, hi = DAY_AHEAD_LEAD_HOURS
    best: tuple[float, str, datetime] | None = None
    for fold, fr in folds.items():
        if "FU masked" not in fr:
            continue
        try:
            origin = pick_channel_origin(
                with_channels=fr["FU"], masked=fr["FU masked"], exclude=exclude
            )
        except SeriesError:
            continue
        mid = (
            (pl.col("lead_hours") >= lo)
            & (pl.col("lead_hours") < hi)
            & (pl.col("power_fcst_init_time") == origin)
        )
        a, b = (fr[k].filter(mid)["q50"].to_numpy() for k in ("FU", "FU masked"))
        n = min(len(a), len(b))
        score = float(np.abs(a[:n] - b[:n]).mean()) if n else -1.0
        if best is None or score > best[0]:
            best = (score, fold, origin)
    if best is None:
        msg = "no fold holds both FU and FU with the channels masked, so the effect of the channels cannot be drawn"
        raise SeriesError(msg)
    _, fold, origin = best
    frames = {k: v for k, v in folds[fold].items() if k in ("FU", "FU masked", "M*")}
    title = f"With and without the untangled channels: the day-ahead forecast of the origin of {origin.date()}"
    return forecast_chart(
        frames=frames, origin=origin, lead=DAY_AHEAD_LEAD_HOURS, title=title
    ), Window("channels", (fold, origin))


def events_figure(*, folds: Folds, d8: int) -> tuple[alt.VConcatChart | None, list[Window], int]:
    """Return the load-shift event panels (None when no event qualifies), their windows, and the count of events."""
    day2 = {"FU": pl.concat([lead_day_two_series(frame=f["FU"]) for f in folds.values()])}
    if all("M*" in f for f in folds.values()):
        day2["M*"] = pl.concat([lead_day_two_series(frame=f["M*"]) for f in folds.values()])
    shifts, n_events = event_windows(events=d8_events(d8=d8), valid_times=day2["FU"]["valid_time"])
    panels = [
        forecast_chart(
            frames=day2, origin=None, lead=None, window=(e["first"], e["last"]), marks=(e["start"], e["end"]),
            title=f"Logged load-shift event of {e['start']:%Y-%m-%d}: lead-day-2 forecast",
        )
        for e in shifts
    ]  # fmt: skip
    windows = [Window("event", ("event", e["start"])) for e in shifts]
    return (alt.vconcat(*panels) if panels else None), windows, n_events


def run_series(*, root: Path, rows: str) -> dict[str, Any]:
    """Draw every scenario and return what was drawn.

    Args:
        root: This experiment's data folder.
        rows: "test" or "validation".

    Returns:
        `files`, `scenarios` (the panels of each), `notes` (the fallbacks used), `n_events`, and
        `n_windows`.

    Raises:
        SeriesError: When a scenario other than the load-shift events has nothing to draw, or a
            window is drawn twice.
    """
    import e00733_signdata as signdata

    d8 = baselines.d8_series_id()
    plan = scoring.choose_plan(root=root, arms=(("FU", None),), wanted_seeds=(SEED,), rows=rows)
    if plan is None:
        msg = "no fold holds a stored forecast of FU at seed 0, so no series can be drawn"
        raise SeriesError(msg)
    folds = {f: read_real(root=root, rows=rows, d8=d8, fold=f) for f in plan.folds}
    out = root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    chart, windows, notes = season_figure(folds=folds, allow_fallback=rows != "test")
    figs.save_svg(chart=chart, path=out / "series_real.svg")
    files, scenarios = ["series_real.svg"], {"winter": 2, "summer": 2}
    chart, window = channel_figure(folds=folds, exclude={w.key[1] for w in windows})
    figs.save_svg(chart=chart, path=out / "series_channels.svg")
    files.append("series_channels.svg")
    windows.append(window)
    scenarios["channels"] = 1
    chart, shifts, n_events = events_figure(folds=folds, d8=d8)
    scenarios["load-shift events"] = len(shifts)
    if chart is None:
        notes.append(
            f"D8 has {n_events} logged load-shift events with forecasts at their start and end in these rows, so no event panel is drawn"
        )
    else:
        figs.save_svg(chart=chart, path=out / "series_events.svg")
        files.append("series_events.svg")
    windows += shifts
    sign = signdata.load_sign_data(root=root, rows=rows)
    picked = pick_sign_windows(data=sign)
    windows += [Window("sign", ("sign", w.seg, w.origin_day)) for w in picked]
    figs.save_svg(chart=sign_figure(data=sign, windows=picked), path=out / "series_sign.svg")
    files.append("series_sign.svg")
    scenarios["sign forecast"] = len(picked)
    assert_distinct(windows=windows)
    private = root / "PRIVATE"
    private.mkdir(parents=True, exist_ok=True)
    listing = [{"scenario": w.scenario, "key": [str(k) for k in w.key]} for w in windows]
    (private / "series_windows.json").write_text(json.dumps(listing, indent=1))
    return {"files": files, "scenarios": scenarios, "notes": notes, "n_events": n_events, "n_windows": len(windows), "folds": list(folds)}  # fmt: skip
