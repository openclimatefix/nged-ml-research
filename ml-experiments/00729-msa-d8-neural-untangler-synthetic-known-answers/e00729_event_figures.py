# ruff: noqa: E501
"""Experiment 729: figures of the event generator, alone and against the real engine record.

**Published figures show simulated series in detail and the real record as aggregates only.** The
real engine record is an NGED generator, so its time series never appears in a published figure,
the printed report, a commit, or a pull request. A real aggregate is a share (never a count),
is computed on output normalised by the record's own 99th percentile of running output (so no
absolute size is recoverable), is binned coarsely, and is merged with its neighbour when fewer
than `MIN_REAL_EVENTS` real events fall in a bin. The same bins and normalisation apply to the
simulated draws. The metered solar farms (source S1) are NGED generators too, so S1 days appear
only in the private figures.

**Private figures go to `PRIVATE/event_generator/` and are never committed.** They show real engine
days beside simulated days with dates and megawatt-scale axes, for the maintainer only.

The script reads the stored test bank (for the solar days) and the engine record (read-only). It
fits nothing. The simulated engine series are fresh draws of `sim.draw_engine` with fixed seeds on
an invented calendar of 30 days starting on a Monday, with weekend days as the quiet days.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import altair as alt
import e00729_core as core
import e00729_figures as figures
import e00729_sim as sim
import numpy as np
import polars as pl

EVENTS_SEED: Final = 7299
N_DRAWS: Final = 400
"""Simulated engines (one 30-day segment each) behind the simulated distributions."""
N_DAYS_PER_DRAW: Final = 30
INVENTED_MONDAY: Final = np.datetime64("2030-01-07", "D")
RUN_SHARE: Final = 0.15
"""A half-hour is running when the output is at least this share of the engine's own 99th percentile of running output."""
STEP_SHARE: Final = 0.05
"""A half-hour step counts as a ramp when it changes the output by at least this share of the same reference."""
MIN_REAL_EVENTS: Final = 5
"""A bin with fewer real events than this is merged into a neighbouring bin."""
RUN_LENGTH_LOWER_H: Final = (0.0, 1.0, 2.0, 4.0, 8.0, 24.0)
RAMP_LOWER: Final = (0.05, 0.15, 0.3, 0.6)
STARTS_LOWER: Final = (0, 1, 2, 3, 5, 8)
ENERGY_LOWER: Final = (0.0, 0.005, 0.05, 0.15, 0.3, 0.5)
"""Lower edges of the bins of run length (hours), ramp size (share), starts per week, and daily energy (share of the reference output for 24 hours)."""
PANEL_WIDTH_PX: Final = 285
PANEL_HEIGHT_PX: Final = 80
SIM_COLOUR: Final = figures.BLUE
REAL_COLOUR: Final = figures.ORANGE
SOLAR_COLOUR: Final = figures.GREEN
SEASONS: Final = {
    "winter": (12, 1, 2),
    "spring": (3, 4, 5),
    "summer": (6, 7, 8),
    "autumn": (9, 10, 11),
}
MIN_SOLAR_HALF_HOURS: Final = 20


# --------------------------------------------------------------------------- statistics


@dataclass(frozen=True)
class EventStats:
    """Dimensionless statistics of one or several engines' output.

    Attributes:
        run_half_hours: Running half-hours by UTC hour (24,), for "weekday" and "weekend".
        n_days: Complete days behind each of the two hour-of-day arrays.
        run_lengths_h: Lengths in hours of the runs that start and end inside a stretch of complete days.
        steps: Absolute half-hour changes, as a share of the reference output, of at least `STEP_SHARE`.
        weekly_starts: Run starts in each block of 7 complete consecutive days.
        daily_energy: Each complete day's energy as a share of the reference output for 24 hours.
    """

    run_half_hours: dict[str, np.ndarray]
    n_days: dict[str, int]
    run_lengths_h: np.ndarray
    steps: np.ndarray
    weekly_starts: np.ndarray
    daily_energy: np.ndarray

    def run_probability(self, *, kind: str) -> np.ndarray:
        """Return the share of half-hours running in each UTC hour (NaN with no days)."""
        n = self.n_days[kind]
        return self.run_half_hours[kind] / (2.0 * n) if n else np.full(24, np.nan)


def weekday_of(*, days: np.ndarray) -> np.ndarray:
    """Return the weekday (0 Monday to 6 Sunday) of days given as `datetime64[D]`."""
    return (((days - np.datetime64("1970-01-01", "D")).astype(int)) + 3) % 7


def reference_level(*, series: np.ndarray) -> float | None:
    """Return the 99th percentile of the positive output, or None when the series has none."""
    positive = series[np.isfinite(series) & (series > 0)]
    return float(np.percentile(positive, 99)) if positive.size else None


def complete_stretches(*, days: np.ndarray) -> list[tuple[int, int]]:
    """Return the `(first, last+1)` rows of the runs of consecutive days with no missing half-hour."""
    full = np.isfinite(days).all(axis=1)
    out, start = [], None
    for i, ok in enumerate(full):
        if ok and start is None:
            start = i
        if not ok and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(full)))
    return out


def run_edges(*, running: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the start indices and the end indices (exclusive) of the runs of a boolean series."""
    padded = np.r_[False, running, False].astype(int)
    d = np.diff(padded)
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def event_stats(
    *, days: np.ndarray, dates: np.ndarray, reference: float | None = None
) -> EventStats | None:
    """Return the statistics of one engine's `(n_days, 48)` output.

    Args:
        days: The output in any unit, with NaN for a missing half-hour.
        dates: The calendar day of each row, `datetime64[D]` (only the weekday is used).
        reference: The reference output; the 99th percentile of positive output when None.

    Returns:
        The statistics, or None when the engine never runs.
    """
    ref = reference if reference is not None else reference_level(series=days.ravel())
    if ref is None or ref <= 0:
        return None
    share = days / ref
    kinds = {"weekday": weekday_of(days=dates) < 5, "weekend": weekday_of(days=dates) >= 5}
    run_hh = {k: np.zeros(24) for k in kinds}
    n_days = dict.fromkeys(kinds, 0)
    lengths: list[float] = []
    steps: list[np.ndarray] = []
    weekly: list[int] = []
    energy: list[float] = []
    for a, b in complete_stretches(days=days):
        block = share[a:b]
        flat = block.ravel()
        running = flat >= RUN_SHARE
        for k, mask in kinds.items():
            sel = mask[a:b]
            n_days[k] += int(sel.sum())
            run_hh[k] += (block[sel] >= RUN_SHARE).reshape(-1, 24, 2).sum(axis=(0, 2))
        starts, ends = run_edges(running=running)
        inside = (starts > 0) & (ends < len(flat))
        lengths += list((ends[inside] - starts[inside]) / 2.0)
        d = np.abs(np.diff(flat))
        steps.append(d[d >= STEP_SHARE])
        for w in range((b - a) // 7):
            sl = slice(w * 7 * 48, (w + 1) * 7 * 48)
            s_w, _ = run_edges(running=running[sl])
            weekly.append(int(len(s_w) - (1 if running[sl][0] else 0)))
        energy += list(block.sum(axis=1) * 0.5 / 24.0)
    return EventStats(
        run_half_hours=run_hh, n_days=n_days, run_lengths_h=np.asarray(lengths),
        steps=np.concatenate(steps) if steps else np.zeros(0), weekly_starts=np.asarray(weekly, dtype=float),
        daily_energy=np.asarray(energy),
    )  # fmt: skip


def pool_stats(*, stats: Sequence[EventStats]) -> EventStats:
    """Pool the statistics of several engines."""
    kinds = ("weekday", "weekend")
    return EventStats(
        run_half_hours={k: sum(s.run_half_hours[k] for s in stats) for k in kinds},  # ty: ignore[invalid-argument-type]
        n_days={k: sum(s.n_days[k] for s in stats) for k in kinds},
        run_lengths_h=np.concatenate([s.run_lengths_h for s in stats]),
        steps=np.concatenate([s.steps for s in stats]),
        weekly_starts=np.concatenate([s.weekly_starts for s in stats]),
        daily_energy=np.concatenate([s.daily_energy for s in stats]),
    )


def bin_counts(*, values: np.ndarray, lower: Sequence[float]) -> np.ndarray:
    """Count the values in the bins that start at `lower` (the last bin is open); lower values are dropped."""
    idx = np.searchsorted(np.asarray(lower, dtype=float), values, side="right") - 1
    idx = idx[idx >= 0]
    return np.bincount(idx, minlength=len(lower)).astype(float)


def merge_sparse_bins(
    *, real_counts: np.ndarray, min_count: int = MIN_REAL_EVENTS
) -> list[tuple[int, int]]:
    """Group neighbouring bins, left to right, until each group holds at least `min_count` real events.

    A last group that stays under the minimum merges into the group before it, and a distribution with
    fewer than `min_count` real events in all becomes one group.

    Args:
        real_counts: The real events in each bin.
        min_count: The smallest real count of a group.

    Returns:
        The groups as `(first bin, last bin + 1)`.
    """
    groups: list[tuple[int, int]] = []
    start, acc = 0, 0.0
    for i, c in enumerate(real_counts):
        acc += c
        if acc >= min_count:
            groups.append((start, i + 1))
            start, acc = i + 1, 0.0
    if start < len(real_counts):
        if groups:
            groups[-1] = (groups[-1][0], len(real_counts))
        else:
            groups.append((0, len(real_counts)))
    return groups


def grouped_shares(
    *, real: np.ndarray, sim_values: np.ndarray, lower: Sequence[float]
) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray]:
    """Return the merged groups and the real and simulated share of each group.

    Args:
        real: The real values.
        sim_values: The simulated values.
        lower: The lower edges of the bins.

    Returns:
        The groups, the real shares, and the simulated shares (each set sums to 1; NaN when empty).
    """
    rc = bin_counts(values=real, lower=lower)
    sc = bin_counts(values=sim_values, lower=lower)
    groups = merge_sparse_bins(real_counts=rc)
    rs = np.array([rc[a:b].sum() for a, b in groups])
    ss = np.array([sc[a:b].sum() for a, b in groups])
    return groups, rs / rs.sum() if rs.sum() else rs, ss / ss.sum() if ss.sum() else ss


def group_labels(
    *, lower: Sequence[float], groups: Sequence[tuple[int, int]], unit: str
) -> list[str]:
    """Return a label such as "1 to 4 h" for each group of bins (the last group is open-ended)."""

    def fmt(v: float) -> str:
        return f"{v:g}"

    out = []
    for a, b in groups:
        lo = fmt(lower[a])
        if b >= len(lower):
            out.append(f"{lo}+{unit}" if a else f"all{unit}")
        else:
            out.append(f"{lo} to {fmt(lower[b])}{unit}")
    return out


# --------------------------------------------------------------------------- simulated engines


def draw_engines(*, n: int = N_DRAWS) -> list[tuple[np.ndarray, dict[str, float]]]:
    """Draw `n` simulated engines (30 days each) with fixed seeds; return each output and its parameters."""
    dates = INVENTED_MONDAY + np.arange(N_DAYS_PER_DRAW)
    quiet = weekday_of(days=dates) >= 5
    out = []
    for i in range(n):
        rng = np.random.default_rng(np.random.SeedSequence([EVENTS_SEED, i]))
        params: dict[str, float] = {}
        engine = sim.draw_engine(
            rng=rng, utc_offset=np.zeros(N_DAYS_PER_DRAW), quiet_day=quiet, record=params
        )
        out.append((engine, params))
    return out


def simulated_stats(*, draws: Sequence[tuple[np.ndarray, dict[str, float]]]) -> EventStats:
    """Return the pooled statistics of the simulated engines, each normalised by its own reference."""
    dates = INVENTED_MONDAY + np.arange(N_DAYS_PER_DRAW)
    parts = [s for e, _ in draws if (s := event_stats(days=e, dates=dates)) is not None]
    return pool_stats(stats=parts)


def real_days(*, record: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return the record as `(n_days, 48)` points of D8's capacity over every day it spans, and the days.

    The caller keeps the points private: the aggregates divide them by their own reference.
    """
    import e00729_data as data

    start = record.select(pl.col("time") - pl.duration(minutes=30)).to_series()
    first = start.min().date()  # ty: ignore[unresolved-attribute]
    last = start.max().date()  # ty: ignore[unresolved-attribute]
    days = np.arange(np.datetime64(first, "D"), np.datetime64(last, "D") + np.timedelta64(1, "D"))
    return data.record_by_day(record=record, days=days), days


# --------------------------------------------------------------------------- gallery picking


@dataclass(frozen=True)
class GalleryCase:
    """A simulated engine's three days and what they show.

    Attributes:
        label: The kind of day.
        draw: The draw number.
        first_day: The row of the window's first day in the draw's 30 days.
        params: The draw's parameters.
    """

    label: str
    draw: int
    first_day: int
    params: Mapping[str, float]


def centroid_hh(*, day: np.ndarray) -> float:
    """Return the output-weighted mean half-hour of a day, or NaN with no output."""
    total = float(day.sum())
    return float((np.arange(48) * day).sum() / total) if total > 0 else float("nan")


def n_runs(*, day: np.ndarray, level: float) -> int:
    """Return the number of runs (half-hours at or above `level`) in a day."""
    return len(run_edges(running=day >= level)[0])


def has_dip(*, day: np.ndarray, level: float) -> bool:
    """Return whether a run of the day is interrupted by one to four half-hours below `level`."""
    starts, ends = run_edges(running=day >= level)
    gaps = starts[1:] - ends[:-1]
    return bool(((gaps >= 1) & (gaps <= 4)).any())


CASES: Final[tuple[tuple[str, Callable[[np.ndarray, Mapping[str, float], bool], bool]], ...]] = (
    ("Weekday evening run", lambda w, p, q: not q and 34 <= centroid_hh(day=w[0]) < 46),
    ("Weekend run", lambda w, p, q: q and w[0].max() > 0),
    ("Weekday morning run", lambda w, p, q: not q and 12 <= centroid_hh(day=w[0]) < 24),
    ("Weekday midday run", lambda w, p, q: not q and 20 <= centroid_hh(day=w[0]) < 28),
    ("Part output", lambda w, p, q: 0 < w[0].max() < 0.5 * p["total_points_cap"]),
    (
        "Slow ramp",
        lambda w, p, q: w[0].max() > 0 and np.abs(np.diff(w[0])).max() <= 0.5 * w[0].max(),
    ),
    ("One unit", lambda w, p, q: p["n_units"] == 1 and w[0].max() > 0),
    ("Three or more units", lambda w, p, q: p["n_units"] >= 3 and w[0].max() > 0),
    ("Multi-day spell", lambda w, p, q: bool((w.min(axis=1) >= 0.2 * p["total_points_cap"]).all())),
    (
        "Run with a trip",
        lambda w, p, q: w[0].max() > 0 and has_dip(day=w[0], level=0.15 * w[0].max()),
    ),
    ("Day with no run", lambda w, p, q: w[0].max() == 0),
    (
        "Two runs in a day",
        lambda w, p, q: n_runs(day=w[0], level=0.15 * w[0].max()) >= 2 and w[0].max() > 0,
    ),
)
"""Twelve kinds of simulated day. Each takes the window `(3, 48)`, the parameters, and whether day 1 is a weekend, and says if the window is a case."""


def pick_cases(*, draws: Sequence[tuple[np.ndarray, dict[str, float]]]) -> list[GalleryCase]:
    """Pick, for each kind of day, the first three-day window that fits, preferring a draw no earlier kind used.

    A kind with no window is left out. The rule draws no random number.
    """
    dates = INVENTED_MONDAY + np.arange(N_DAYS_PER_DRAW)
    quiet = weekday_of(days=dates) >= 5
    out: list[GalleryCase] = []
    used: set[int] = set()
    for label, predicate in CASES:
        found = None
        for allow_used in (False, True):
            for i, (engine, params) in enumerate(draws):
                if i in used and not allow_used:
                    continue
                for d in range(N_DAYS_PER_DRAW - 2):
                    if predicate(engine[d : d + 3], params, bool(quiet[d])):
                        found = GalleryCase(label, i, d, params)
                        break
                if found:
                    break
            if found:
                break
        if found:
            used.add(found.draw)
            out.append(found)
    return out


def season_of(*, month: int) -> str:
    """Return the season name of a month number."""
    return next(k for k, v in SEASONS.items() if month in v)


@dataclass(frozen=True)
class SolarDay:
    """One simulated solar day picked for the gallery.

    Attributes:
        label: The season and the sky ("summer, clearest").
        output: The 48 half-hour outputs in points.
        title_extra: A private-only note (the date), or an empty string.
    """

    label: str
    output: np.ndarray
    title_extra: str


def pick_solar_days(*, segments: Sequence[object], source: str, dated: bool) -> list[SolarDay]:
    """Pick, for each season, the clearest and the cloudiest day of a solar source.

    A day counts when at least `MIN_SOLAR_HALF_HOURS` half-hours have output and it is a base day
    without the S1 fallback. The clearest day has the highest mean clear-sky index over its
    producing half-hours, and the cloudiest the lowest; a tie goes to the lowest (segment, day).

    Args:
        segments: Segments of the stored test bank (each with `source`, `solar`, `csi`, `dates`,
            `cal_index`, and `s1_fallback`).
        source: "S1" or "S2".
        dated: Whether to keep the date in `title_extra` (private figures only).

    Returns:
        Up to 8 days, in the order winter, spring, summer, autumn, each clearest then cloudiest.
    """
    cands: dict[str, list[tuple[float, int, int, np.ndarray, str]]] = {k: [] for k in SEASONS}
    for k, seg in enumerate(segments):
        if seg.source != source:  # ty: ignore[unresolved-attribute]
            continue
        for d in np.flatnonzero(seg.cal_index >= 0):  # ty: ignore[unresolved-attribute]
            day = np.asarray(seg.solar[d], dtype=float)  # ty: ignore[unresolved-attribute]
            if seg.s1_fallback[d] or (day > 0).sum() < MIN_SOLAR_HALF_HOURS:  # ty: ignore[unresolved-attribute]
                continue
            csi = np.asarray(seg.csi[d], dtype=float)[day > 0]  # ty: ignore[unresolved-attribute]
            if not np.isfinite(csi).any():
                continue
            when = seg.dates[d].astype("datetime64[D]")  # ty: ignore[unresolved-attribute]
            month = int(str(when)[5:7])
            cands[season_of(month=month)].append(
                (float(np.nanmean(csi)), k, int(d), day, str(when) if dated else "")
            )
    out = []
    for season, items in cands.items():
        if not items:
            continue
        by_csi = sorted(items, key=lambda t: (t[0], t[1], t[2]))
        for tag, pick in (("clearest", by_csi[-1]), ("cloudiest", by_csi[0])):
            out.append(SolarDay(f"{season}, {tag}", pick[3], pick[4]))
    return out


# --------------------------------------------------------------------------- charts


def hours_axis(*, n_days: int) -> alt.X:
    """Return an x encoding in hours with ticks every 12 hours, labelled as hour of day."""
    ticks = list(range(0, 24 * n_days + 1, 12))
    return alt.X(
        "hour:Q", title="Hour of day, UTC", scale=alt.Scale(domain=[0, 24 * n_days], nice=False),
        axis=alt.Axis(values=ticks, labelExpr="datum.value % 24", grid=False),
    )  # fmt: skip


def trace_panel(
    *, values: np.ndarray, title: str, colour: str, y_max: float | None, show_y: bool
) -> alt.LayerChart:  # fmt: skip
    """Return one panel of output over `len(values) / 48` days, with rules at the midnights."""
    n_days = len(values) // 48
    frame = pl.DataFrame(
        {
            "hour": np.arange(len(values)) / 2.0,
            "value": [float(v) if np.isfinite(v) else None for v in values],
        }
    )
    y = alt.Y(
        "value:Q", title="Points of capacity" if show_y else None,
        scale=alt.Scale(domain=[0, y_max]) if y_max else alt.Scale(zero=True),
    )  # fmt: skip
    line = (
        alt.Chart(frame)
        .mark_area(
            line={"color": colour, "strokeWidth": 1.4},
            color=colour,
            opacity=0.25,
            aria=False,
            invalid=None,
        )
        .encode(x=hours_axis(n_days=n_days), y=y)  # ty: ignore[unresolved-attribute]
    )
    rules = (
        alt.Chart(pl.DataFrame({"hour": [24.0 * k for k in range(1, n_days)]}))
        .mark_rule(color="#999999", strokeDash=[3, 3], aria=False)
        .encode(x="hour:Q")  # ty: ignore[unresolved-attribute]
    )
    return (line + rules).properties(
        width=PANEL_WIDTH_PX, height=PANEL_HEIGHT_PX,
        title=alt.TitleParams(title, anchor="start", fontSize=11),
    )  # fmt: skip


def gallery_chart(
    *, panels: Sequence[tuple[str, np.ndarray]], colour: str, title: str, subtitle: str | list[str]
) -> alt.VConcatChart:  # fmt: skip
    """Return panels in two columns on a common y axis (a panel is a title and its output)."""
    top = max(float(np.nanmax(v)) for _, v in panels if np.isfinite(v).any())
    rows = []
    for i in range(0, len(panels), 2):
        pair = panels[i : i + 2]
        rows.append(
            alt.hconcat(
                *(
                    trace_panel(values=v, title=t, colour=colour, y_max=top * 1.05, show_y=j == 0)
                    for j, (t, v) in enumerate(pair)
                )
            )
        )
    return alt.vconcat(*rows).properties(
        title=alt.TitleParams(title, subtitle=subtitle, anchor="start")
    )


def engine_gallery(
    *, draws: Sequence[tuple[np.ndarray, dict[str, float]]], cases: Sequence[GalleryCase]
) -> alt.VConcatChart:
    """Return the gallery of simulated engine windows (three days each), with the parameter ranges in the subtitle."""
    panels = []
    for c in cases:
        e = draws[c.draw][0][c.first_day : c.first_day + 3].ravel()
        p = c.params
        panels.append(
            (
                f"{c.label} ({int(p['n_units'])} unit{'s' if p['n_units'] > 1 else ''}, {p['total_points_cap']:.0f} points in all)",
                e,
            )
        )
    r = core.RANGES
    note = (
        f"Each panel shows 3 simulated days; the label describes the first day, and dashed lines mark midnights. Ranges drawn from: "
        f"{r['n_units'][0]} to {r['n_units'][1]} units of {r['unit_points'][0]:g} to {r['unit_points'][1]:g} points "
        f"(total at most {r['total_points_cap'][1]:g}); evening runs start {r['evening_start_h'][0]:g} to {r['evening_start_h'][1]:g} h "
        f"for {r['evening_hours'][0]:g} to {r['evening_hours'][1]:g} h; morning runs start {r['morning_start_h'][0]:g} to {r['morning_start_h'][1]:g} h "
        f"for {r['morning_hours'][0]:g} to {r['morning_hours'][1]:g} h; midday runs start {r['midday_start_h'][0]:g} to {r['midday_start_h'][1]:g} h "
        f"for {r['midday_hours'][0]:g} to {r['midday_hours'][1]:g} h; ramps {r['ramp_half_hours'][0]:g} to {r['ramp_half_hours'][1]:g} half-hours; "
        f"weekend factor {r['weekend_factor'][0]:g} to {r['weekend_factor'][1]:g}."
    )
    return gallery_chart(
        panels=panels, colour=SIM_COLOUR, title="Twelve kinds of simulated engine day (points of D8's capacity)", subtitle=wrap_text(note, width=95)
    )  # fmt: skip


def solar_gallery(*, days: Sequence[SolarDay], title: str, colour: str) -> alt.VConcatChart:
    """Return the gallery of solar days, one day per panel."""
    panels = [(f"{d.label}{' ' + d.title_extra if d.title_extra else ''}", d.output) for d in days]
    r = core.RANGES
    note = (
        f"Capacity {r['solar_capacity_points'][0]:g} to {r['solar_capacity_points'][1]:g} points; tilt {r['tilt_deg'][0]:g} to {r['tilt_deg'][1]:g} degrees; "
        f"azimuth {r['azimuth_deg'][0]:g} to {r['azimuth_deg'][1]:g} degrees; DC to AC ratio {r['dc_ac_ratio'][0]:g} to {r['dc_ac_ratio'][1]:g}. "
        "The clearest and cloudiest day of each season, by the mean clear-sky index of the producing half-hours."
    )
    return gallery_chart(
        panels=panels, colour=colour, title=title, subtitle=wrap_text(note, width=95)
    )


def wrap_text(text: str, *, width: int) -> list[str]:
    """Wrap text to lines of at most `width` characters."""
    import textwrap

    return textwrap.wrap(text, width=width)


def hour_chart(*, real: EventStats, sim_stats: EventStats) -> alt.VConcatChart:
    """Return the run probability by UTC hour, weekday above weekend, real record against simulated draws."""
    panels = []
    for kind in ("weekday", "weekend"):
        frame = pl.concat(
            [
                pl.DataFrame(
                    {
                        "hour": np.arange(24) + 0.5,
                        "source": name,
                        "share": s.run_probability(kind=kind),
                    }
                )
                for name, s in (("Real record", real), ("Simulated draws", sim_stats))
            ]
        )
        panels.append(
            alt.Chart(frame)
            .mark_line(aria=False, strokeWidth=2, interpolate="step-after")
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X(
                    "hour:Q",
                    title="Hour of day, UTC",
                    scale=alt.Scale(domain=[0, 24], nice=False),
                    axis=alt.Axis(values=[0, 6, 12, 18, 24], grid=False),
                ),
                y=alt.Y(
                    "share:Q",
                    title=f"Share of {kind} half-hours running",
                    scale=alt.Scale(domain=[0, 1]),
                ),
                color=alt.Color(
                    "source:N",
                    scale=alt.Scale(
                        domain=["Real record", "Simulated draws"], range=[REAL_COLOUR, SIM_COLOUR]
                    ),
                    legend=alt.Legend(orient="top", title=None),
                ),
            )
            .properties(width=figures.WIDTH_PX - 100, height=110)
        )
    return alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            "How often the engine is running, by hour of day",
            subtitle=wrap_text(
                f"Running means output of at least {RUN_SHARE:.0%} of the engine's own 99th percentile of running output. Saturday and Sunday are the weekend; the simulated draws take bank holidays as weekdays.",
                width=95,
            ),
            anchor="start",
        )
    )


def share_chart(
    *,
    labels: Sequence[str],
    real: np.ndarray,
    sim_values: np.ndarray,
    title: str,
    x_title: str,
    subtitle: str,
) -> alt.Chart:
    """Return two overlaid sets of bars: the share of events in each bin, real record against simulated draws."""
    frame = pl.concat(
        [
            pl.DataFrame({"bin": list(labels), "source": name, "share": s})
            for name, s in (("Real record", real), ("Simulated draws", sim_values))
        ]
    )
    return (
        alt.Chart(frame)
        .mark_bar(aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("bin:N", sort=list(labels), title=x_title, axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("source:N"),
            y=alt.Y("share:Q", title="Share of events", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "source:N",
                scale=alt.Scale(
                    domain=["Real record", "Simulated draws"], range=[REAL_COLOUR, SIM_COLOUR]
                ),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
        .properties(
            width=figures.WIDTH_PX - 100,
            height=150,
            title=alt.TitleParams(title, subtitle=wrap_text(subtitle, width=95), anchor="start"),
        )
    )


# --------------------------------------------------------------------------- the stage


@dataclass(frozen=True)
class Distribution:
    """One categorical aggregate: its name, bins, units, and the values of each source."""

    key: str
    title: str
    x_title: str
    lower: tuple[float, ...]
    unit: str
    real: np.ndarray
    sim: np.ndarray
    subtitle: str


def distributions(*, real: EventStats, sim_stats: EventStats) -> list[Distribution]:
    """Return the four categorical aggregates for the real record and the simulated draws."""
    return [
        Distribution(
            "run_length",
            "How long a run lasts",
            "Run length (hours)",
            RUN_LENGTH_LOWER_H,
            " h",
            real.run_lengths_h,
            sim_stats.run_lengths_h,
            "Runs that start and end inside a stretch of complete days. Bins with few real runs are merged.",
        ),
        Distribution(
            "ramp",
            "How fast the output changes",
            "Change in one half-hour (share of the reference output)",
            RAMP_LOWER,
            "",
            real.steps,
            sim_stats.steps,
            f"Half-hour steps of at least {STEP_SHARE:.0%} of the reference output.",
        ),
        Distribution(
            "starts",
            "How many times the engine starts in a week",
            "Starts in 7 consecutive days",
            STARTS_LOWER,
            "",
            real.weekly_starts,
            sim_stats.weekly_starts,
            "Blocks of 7 complete consecutive days.",
        ),
        Distribution(
            "energy",
            "How much energy a day holds",
            "Daily energy (share of the reference output for 24 hours)",
            ENERGY_LOWER,
            "",
            real.daily_energy,
            sim_stats.daily_energy,
            "Complete days. A day of no run holds under half a percent.",
        ),
    ]


def aggregate_charts(*, real: EventStats, sim_stats: EventStats) -> dict[str, alt.TopLevelMixin]:
    """Return the published real-against-simulated charts, keyed by file name."""
    out: dict[str, alt.TopLevelMixin] = {
        "events_03_hour_of_day.svg": hour_chart(real=real, sim_stats=sim_stats)
    }
    for i, dist_ in enumerate(distributions(real=real, sim_stats=sim_stats), start=4):
        groups, rs, ss = grouped_shares(real=dist_.real, sim_values=dist_.sim, lower=dist_.lower)
        labels = group_labels(lower=dist_.lower, groups=groups, unit=dist_.unit)
        out[f"events_{i:02d}_{dist_.key}.svg"] = share_chart(
            labels=labels,
            real=rs,
            sim_values=ss,
            title=dist_.title,
            x_title=dist_.x_title,
            subtitle=dist_.subtitle,
        )
    return out


def real_windows(*, days: np.ndarray, dates: np.ndarray, n: int) -> list[tuple[str, np.ndarray]]:
    """Return `n` three-day windows of the real record, evenly spaced over the windows of complete days (private only)."""
    full = np.isfinite(days).all(axis=1)
    starts = [i for i in range(len(days) - 2) if full[i : i + 3].all()]
    if not starts:
        return []
    picks = [starts[(k * len(starts)) // n] for k in range(n)] if len(starts) >= n else starts
    return [(str(dates[i]), days[i : i + 3].ravel()) for i in picks]


def write_event_figures(
    *, segments: Sequence[object], record: pl.DataFrame, out_dir: Path, private_dir: Path,
    log: Callable[[str], None],
) -> list[Path]:  # fmt: skip
    """Draw the published and the private event-generator figures.

    Args:
        segments: The segments of the stored test bank (for the solar days).
        record: The metered engine record.
        out_dir: The folder of the published figures.
        private_dir: The private folder (`PRIVATE/event_generator`); real series go only here.
        log: Receives one line per figure with its name only (no real value or count).

    Returns:
        The paths of the published figures (run `svgo` on each before committing).
    """
    draws = draw_engines()
    cases = pick_cases(draws=draws)
    published: dict[str, alt.TopLevelMixin] = {
        "events_01_engine_days.svg": engine_gallery(draws=draws, cases=cases),
    }
    s2 = pick_solar_days(segments=segments, source="S2", dated=False)
    if s2:
        published["events_02_solar_s2_days.svg"] = solar_gallery(
            days=s2,
            title="Simulated solar days from the PV model on CAMS (source S2)",
            colour=SOLAR_COLOUR,
        )
    real_grid, real_dates = real_days(record=record)
    real = event_stats(days=real_grid, dates=real_dates)
    if real is not None:
        published.update(aggregate_charts(real=real, sim_stats=simulated_stats(draws=draws)))
    written = []
    for name, chart in published.items():
        figures.save_svg(chart=chart, path=out_dir / name)
        written.append(out_dir / name)
        log(f"{name}: written")
    private_dir.mkdir(parents=True, exist_ok=True)
    windows = real_windows(days=real_grid, dates=real_dates, n=12)
    if windows:
        figures.save_svg(
            chart=gallery_chart(
                panels=windows,
                colour=REAL_COLOUR,
                title="Real engine windows (private)",
                subtitle=["Points of D8's capacity; 1 point is 0.10379 MW. Dates allowed here."],
            ),
            path=private_dir / "real_engine_windows.svg",
        )
        sim_panels = [
            (c.label, draws[c.draw][0][c.first_day : c.first_day + 3].ravel())
            for c in cases[: len(windows[:6])]
        ]
        paired = [p for pair in zip(windows[:6], sim_panels, strict=False) for p in pair]
        figures.save_svg(
            chart=gallery_chart(
                panels=paired,
                colour=REAL_COLOUR,
                title="Real (left) and simulated (right) windows on one axis (private)",
                subtitle=["Points of D8's capacity."],
            ),
            path=private_dir / "real_beside_simulated.svg",
        )
    s1 = pick_solar_days(segments=segments, source="S1", dated=True)
    if s1:
        figures.save_svg(
            chart=solar_gallery(
                days=s1,
                title="Simulated solar days built from the metered farms, source S1 (private)",
                colour=SOLAR_COLOUR,
            ),
            path=private_dir / "solar_s1_days.svg",
        )
    log("private figures written to PRIVATE/event_generator")
    return written
