# ruff: noqa: E501, PLR0912
"""Experiment 730: the synthetic world gallery G1 to G9 and the result figures.

Every figure is an Altair chart that `save_svg` writes as SVG, to be optimised with `svgo`
(`figures729.optimise_command`). No axis carries a calendar date: a window is labelled by the hours
from the start of its day 1. A panel built from the metered farms (source S1) is returned in the
private set and never in the published set, and the published S1 panels are seasonal aggregates
(10%, 50%, and 90% bands). Each panel function returns the chart and whether the gallery's written
requirement holds on the arrays behind it, which `e00730_gallery` records.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import e00730_paths  # noqa: F401

# isort: split
import altair as alt
import e00729_core as core729
import e00729_figures as figures729
import e00729_sim as sim729
import e00730_banks as banks
import e00730_core as core
import e00730_gallery as gallery
import e00730_masks as masks
import e00730_sky as sky
import numpy as np
import polars as pl

WIDTH_PX: Final = figures729.WIDTH_PX
BLUE: Final = figures729.BLUE
ORANGE: Final = figures729.ORANGE
GREEN: Final = figures729.GREEN
PURPLE: Final = figures729.PURPLE
SKY: Final = figures729.SKY
GREY: Final = figures729.GREY
HIDDEN_FILL: Final = "#BBBBBB"
SEASON_MONTHS: Final = {
    "winter": (12, 1, 2),
    "spring": (3, 4, 5),
    "summer": (6, 7, 8),
    "autumn": (9, 10, 11),
}
HOUR_AXIS: Final = alt.Axis(values=[0, 12, 24, 36, 48, 60, 72, 84, 96])
N_DRAWS: Final = 3
"""Each G1 feature panel shows this many draws, from different segments."""
save_svg = figures729.save_svg
optimise_command = figures729.optimise_command


# --------------------------------------------------------------------------- generic panels


def window_chart(
    *,
    series: Mapping[str, np.ndarray],
    title: str,
    hidden: np.ndarray | None = None,
    y_title: str = "Points of capacity",
    height: int = 130,
    n_days: int = 2,
) -> alt.Chart:
    """Return lines over a 96-half-hour window (days 1 and 2) with the hidden span shaded.

    Args:
        series: Named arrays of equal length, over the window's half-hours.
        title: What the panel shows, in one or two short lines.
        hidden: Positions whose reading is hidden, shaded grey, or None.
        y_title: The y-axis title with its unit.
        height: Plot height in pixels.
        n_days: The window length in days, which sets the hour axis.

    Returns:
        The chart, with no calendar date anywhere.
    """
    n = len(next(iter(series.values())))
    data = pl.DataFrame(
        {
            "hour": np.tile(np.arange(n) / 2.0, len(series)),
            "value": np.concatenate([np.asarray(v, dtype=float) for v in series.values()]),
            "series": np.repeat(list(series), n),
        }
    )
    colours = [BLUE, ORANGE, GREEN, PURPLE, SKY, GREY]
    lines = (
        alt.Chart(data)
        .mark_line(aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "hour:Q",
                title="Hours from the start of day 1 (UTC)",
                axis=alt.Axis(values=list(range(0, 24 * n_days + 1, 12 if n_days <= 2 else 24))),
            ),
            y=alt.Y("value:Q", title=y_title),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(domain=list(series), range=colours[: len(series)]),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
    )
    layers = lines
    if hidden is not None and hidden.any():
        idx = np.flatnonzero(hidden)
        span = pl.DataFrame({"a": [idx[0] / 2.0], "b": [(idx[-1] + 1) / 2.0]})
        shade = (
            alt.Chart(span)
            .mark_rect(color=HIDDEN_FILL, opacity=0.35, aria=False)
            .encode(x="a:Q", x2="b:Q")  # ty: ignore[unresolved-attribute]
        )
        layers = shade + lines
    return layers.properties(
        width=WIDTH_PX, height=height, title=alt.TitleParams(title, anchor="start")
    )


def two_row_chart(*, top: alt.Chart, bottom: alt.Chart) -> alt.VConcatChart:
    """Stack a window's signed-power row above its probability row."""
    return alt.vconcat(top, bottom).resolve_scale(color="independent")


# --------------------------------------------------------------------------- gallery


@dataclass
class GalleryResult:
    """What the gallery stage returns.

    Attributes:
        public: The published panels, by name.
        private: The panels that show a single metered-farm (S1) day, by name.
        results: Whether each required panel (a key of `core.PANEL_REQUIREMENTS`) holds.
        notes: A counts-only note per required panel.
    """

    public: dict[str, alt.TopLevelMixin] = field(default_factory=dict)
    private: dict[str, alt.TopLevelMixin] = field(default_factory=dict)
    results: dict[str, bool] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)


def day_window(*, seg: sim729.Segment, day: int, name: str) -> np.ndarray:
    """Return days `day` and `day + 1` of a dense array of a segment as 96 values."""
    return np.asarray(getattr(seg, name)[day : day + 2]).reshape(-1)


def has_two_days(*, seg: sim729.Segment, day: int) -> bool:
    """Return whether a window of two dense days starting at `day` lies inside the segment."""
    return day >= 1 and day + 2 <= seg.engine.shape[0] - 1


def first_day(*, segments: Sequence[sim729.Segment], keep) -> tuple[sim729.Segment, int] | None:  # noqa: ANN001
    """Return the first (segment, day) of `segments` for which `keep(seg, day)` holds."""
    for seg in segments:
        for day in range(1, seg.engine.shape[0] - 2):
            if keep(seg, day):
                return seg, day
    return None


def season_of_segment(*, seg: sim729.Segment, day: int) -> str:
    """Return the season of a segment's dense day."""
    month = int(str(seg.dates[day])[5:7])
    return next(s for s, months in SEASON_MONTHS.items() if month in months)


def engine_panel(
    *,
    seg: sim729.Segment,
    day: int,
    title: str,
    n_days: int = 2,
    unit_total: float | None = None,
) -> alt.Chart:
    """Return the engine output beside the base demand over `n_days` days from `day`.

    Args:
        seg: The segment.
        day: The dense day the window starts on.
        title: The panel title.
        n_days: The window length in days.
        unit_total: When given, a line at the sum of the unit sizes.

    Returns:
        The chart.
    """
    series = {
        "engine output": window_array(seg=seg, day=day, n_days=n_days, name="engine"),
        "base demand": window_array(seg=seg, day=day, n_days=n_days, name="base"),
    }
    if unit_total is not None:
        series["sum of unit sizes"] = np.full(24 * n_days * 2, unit_total)
    return window_chart(series=series, title=title, n_days=n_days)


def window_array(*, seg: sim729.Segment, day: int, n_days: int, name: str) -> np.ndarray:
    """Return `n_days` days of a dense array of a segment from `day`, flattened to half-hours."""
    return np.asarray(getattr(seg, name)[day : day + n_days]).reshape(-1)


def pick_windows(
    *,
    segments: Sequence[sim729.Segment],
    n_days: int,
    check: Callable[[sim729.Segment, int, np.ndarray], bool],
    used: set[tuple[int, int]],
    n: int = N_DRAWS,
) -> list[tuple[sim729.Segment, int]]:
    """Return up to `n` windows from different segments whose engine array passes `check`.

    `check(seg, day, engine_window)` runs on the exact array that the panel plots. The first
    passing window of each segment is a candidate. Candidates from months not yet drawn come
    first, and windows that another panel already shows come last.

    Args:
        segments: The segments to search, in a fixed order.
        n_days: The window length in days.
        check: The feature check.
        used: The `(id(segment), day)` windows that other panels show.
        n: The number of draws wanted.

    Returns:
        The chosen `(segment, day)` pairs.
    """
    fresh: list[tuple[sim729.Segment, int]] = []
    shown: list[tuple[sim729.Segment, int]] = []
    for seg in segments:
        for day in range(1, seg.engine.shape[0] - n_days):
            if check(seg, day, window_array(seg=seg, day=day, n_days=n_days, name="engine")):
                (shown if (id(seg), day) in used else fresh).append((seg, day))
                break
    chosen: list[tuple[sim729.Segment, int]] = []
    months: set[str] = set()
    for seg, day in [*fresh, *shown]:
        if str(seg.month) not in months:
            chosen.append((seg, day))
            months.add(str(seg.month))
    for pair in [*fresh, *shown]:
        if len(chosen) >= n:
            break
        if not any(pair[0] is c[0] and pair[1] == c[1] for c in chosen):
            chosen.append(pair)
    chosen = chosen[:n]
    used.update((id(seg), day) for seg, day in chosen)
    return chosen


def is_weekend(*, seg: sim729.Segment, day: int) -> bool:
    """Return whether a segment's dense day is a Saturday or a Sunday."""
    return np.datetime64(seg.dates[day], "D").astype(int) % 7 in (2, 3)


def engine_gallery(*, segments: Sequence[sim729.Segment], out: GalleryResult) -> None:
    """Draw G1: a small multiple of draws for each engine feature, and the requirements.

    Each feature panel shows `N_DRAWS` windows from different segments. The window of a draw is
    picked by the feature itself, the check runs on the exact arrays that the panel plots, and a
    panel passes only when every one of its draws passes. A window that another panel shows is
    used again only when no other segment has the feature.
    """
    engines = [s for s in segments if s.world in ("engine", "both")]
    used: set[tuple[int, int]] = set()
    features = (
        ("G1_ramp", "a run with a ramp", 2, lambda s, d, e: gallery.check_ramp(engine=e), False),
        (
            "G1_evening", "an evening run", 2,
            lambda s, d, e: gallery.check_evening(engine=e), False,
        ),
        (
            "G1_morning", "a morning run", 2,
            lambda s, d, e: gallery.check_morning(engine=e), False,
        ),
        (
            "G1_weekend", "a weekend day", 2,
            lambda s, d, e: gallery.check_weekend(engine=e, first_day_is_weekend=is_weekend(seg=s, day=d)),
            False,
        ),
        (
            "G1_part_load", "part load", 2,
            lambda s, d, e: gallery.check_part_load(
                engine=e, unit_sizes=[s.params.get("total_points_cap", 0.0)]
            ),
            True,
        ),
        ("G1_trip", "a trip inside a run", 2, lambda s, d, e: gallery.check_trip(engine=e), False),
        (
            "G1_spell", "a multi-day spell (5 days)", 5,
            lambda s, d, e: gallery.check_spell(engine=e), False,
        ),
    )  # fmt: skip
    for key, title, n_days, check, show_total in features:
        picked = pick_windows(segments=engines, n_days=n_days, check=check, used=used)
        passes = [
            check(seg, day, window_array(seg=seg, day=day, n_days=n_days, name="engine"))
            for seg, day in picked
        ]
        ok = len(picked) == N_DRAWS and all(passes)
        panels = [
            engine_panel(
                seg=seg, day=day, n_days=n_days,
                title=f"G1: {title} (draw {i + 1} of {N_DRAWS})",
                unit_total=float(seg.params["total_points_cap"]) if show_total else None,
            )
            for i, (seg, day) in enumerate(picked)
        ]  # fmt: skip
        out.results[key] = ok
        out.notes[key] = f"{sum(passes)} of {N_DRAWS} draws pass"
        if panels:
            out.public[key] = alt.vconcat(*panels)
    # units: three one-unit and three several-unit draws, and the bank-wide share of two-level days
    one = pick_windows(
        segments=[s for s in engines if s.params.get("n_units", 0) == 1], n_days=2, used=used,
        check=lambda s, d, e: bool(e.max() > 0 and gallery.run_plateaus(engine=e[:48]) <= 1
                                   and gallery.run_plateaus(engine=e[48:]) <= 1),
    )  # fmt: skip
    many = pick_windows(
        segments=[s for s in engines if s.params.get("n_units", 0) > 1], n_days=2, used=used,
        check=lambda s, d, e: gallery.run_plateaus(engine=e[:48]) >= 2
        or gallery.run_plateaus(engine=e[48:]) >= 2,
    )  # fmt: skip
    levels = {
        label: [
            gallery.run_plateaus(engine=seg.engine[d])
            for seg in engines
            if (seg.params.get("n_units", 0) == 1) == (label == "one")
            for d in range(seg.engine.shape[0])
            if seg.engine[d].max() > 0
        ]
        for label in ("one", "several")
    }
    ok = bool(
        len(one) == N_DRAWS
        and len(many) == N_DRAWS
        and gallery.check_units(
            one_unit_levels=levels["one"], several_unit_levels=levels["several"]
        )
    )
    panels = [
        engine_panel(seg=seg, day=day, title=f"G1: one unit (draw {i + 1} of {N_DRAWS})")
        for i, (seg, day) in enumerate(one)
    ] + [
        engine_panel(seg=seg, day=day, title=f"G1: several units (draw {i + 1} of {N_DRAWS})")
        for i, (seg, day) in enumerate(many)
    ]
    out.results["G1_units"] = ok
    out.notes["G1_units"] = f"{len(one)} one-unit and {len(many)} several-unit draws"
    if panels:
        out.public["G1_units"] = alt.vconcat(*panels)


def day_mean_index(*, seg: sim729.Segment, day: int) -> float:
    """Return the mean clear-sky index over a day's sun-up half-hours, NaN when under 10 are finite."""
    c = seg.csi[day][seg.cosz[day] > 0.2]
    c = c[np.isfinite(c)]
    return float(c.mean()) if c.size >= 10 else float("nan")


def extreme_index_day(
    *, segments: Sequence[sim729.Segment], season: str, clearest: bool
) -> tuple[sim729.Segment, int] | None:
    """Return the clearest (or cloudiest) day of a season by the clear-sky index, or None.

    A day counts when the solar output is above zero on it and a two-day window starts there.
    """
    best, hit = None, None
    for seg in segments:
        for day in range(1, seg.engine.shape[0] - 2):
            if season_of_segment(seg=seg, day=day) != season or seg.solar[day].max() <= 0:
                continue
            value = day_mean_index(seg=seg, day=day)
            if not np.isfinite(value):
                continue
            if best is None or (value > best if clearest else value < best):
                best, hit = value, (seg, day)
    return hit


def solar_index_chart(*, solar: np.ndarray, index: np.ndarray, title: str) -> alt.LayerChart:
    """Return solar output (left axis) and the clear-sky index (right axis) over a 96-half-hour window."""
    n = len(solar)
    data = pl.DataFrame(
        {
            "hour": np.arange(n) / 2.0,
            "solar": np.asarray(solar, dtype=float),
            "index": np.asarray(index, dtype=float),
        }
    )
    x = alt.X("hour:Q", title="Hours from the start of day 1 (UTC)", axis=HOUR_AXIS)
    left = (
        alt.Chart(data)
        .mark_line(aria=False, color=BLUE)
        .encode(  # ty: ignore[unresolved-attribute]
            x=x,
            y=alt.Y(
                "solar:Q",
                title="Solar output (blue; share of capacity)",
                axis=alt.Axis(titleColor=BLUE),
            ),
        )
    )
    right = (
        alt.Chart(data)
        .mark_line(aria=False, color=ORANGE)
        .encode(  # ty: ignore[unresolved-attribute]
            x=x,
            y=alt.Y(
                "index:Q",
                title="Clear-sky index (orange)",
                axis=alt.Axis(titleColor=ORANGE, orient="right"),
                scale=alt.Scale(domain=[0.0, 1.2]),
            ),
        )
    )
    return (
        alt.layer(left, right)
        .resolve_scale(y="independent")
        .properties(width=WIDTH_PX, height=130, title=alt.TitleParams(title, anchor="start"))
    )


def solar_gallery(*, segments: Sequence[sim729.Segment], out: GalleryResult) -> None:
    """Draw G2 (source S2 in detail) and G3 (S1 as aggregates, single days to the private set)."""
    s2 = [s for s in segments if s.world in ("solar", "both") and s.source == "S2"]
    s1 = [s for s in segments if s.world in ("solar", "both") and s.source == "S1"]
    clear_ok, cloudy_ok, found = True, True, 0
    clear_by_season: dict[str, np.ndarray] = {}
    for kind in ("clear", "cloudy"):
        charts = []
        for season in core.COMPONENT_SEASONS:
            hit = extreme_index_day(segments=s2, season=season, clearest=kind == "clear")
            if hit is None:
                clear_ok &= kind != "clear"
                cloudy_ok &= kind != "cloudy"
                continue
            seg, day = hit
            found += 1
            capacity = float(seg.params["solar_capacity_points"])
            solar = day_window(seg=seg, day=day, name="solar") / capacity
            if kind == "clear":
                clear_by_season[season] = solar
                clear_ok &= gallery.check_clear(solar=seg.solar[day], mid=slice(20, 32))
            else:
                ref = clear_by_season.get(season)
                cloudy_ok &= ref is not None and gallery.check_cloudy(
                    cloudy=solar, clear_same_season=ref
                )
            charts.append(
                solar_index_chart(
                    solar=solar,
                    index=day_window(seg=seg, day=day, name="csi"),
                    title=f"G2: the {'clearest' if kind == 'clear' else 'cloudiest'} {season} day by clear-sky index, source S2",
                )
            )
        if charts:
            out.public[f"G2_{kind}"] = alt.vconcat(*charts)
    out.results["G2_clear"] = bool(clear_ok and found)
    out.results["G2_cloudy"] = bool(cloudy_ok and found)
    out.notes["G2_clear"] = out.notes["G2_cloudy"] = f"{found} panels found"
    bands_ok, rows = True, []
    daily, seasons = [], []
    for seg in s1:
        for d in range(1, seg.solar.shape[0] - 1):
            daily.append(seg.solar[d])
            seasons.append(season_of_segment(seg=seg, day=d))
    if daily:
        bands = gallery.seasonal_bands(values=np.stack(daily), seasons=np.array(seasons))
        for season, q in bands.items():
            bands_ok &= gallery.check_aggregate_bands(q10=q[0], q50=q[1], q90=q[2])
            for label, row in zip(("10%", "50%", "90%"), q, strict=True):
                rows += [
                    {"season": season, "band": label, "hh": i / 2.0, "value": float(v)}
                    for i, v in enumerate(row)
                ]
        out.public["G3_aggregate"] = (
            alt.Chart(pl.DataFrame(rows))
            .mark_line(aria=False)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X(
                    "hh:Q",
                    title="Hours from the start of the day (UTC)",
                    axis=alt.Axis(values=[0, 6, 12, 18, 24]),
                ),
                y=alt.Y("value:Q", title=None),
                color=alt.Color("band:N", legend=alt.Legend(orient="top", title=None)),
                row=alt.Row("season:N", title=None),
            )
            .properties(
                width=WIDTH_PX - 100,
                height=70,
                title=alt.TitleParams(
                    "G3: source S1, 10%, 50%, and 90% bands of the daily shape by season",
                    subtitle="Vertical axis of every season: solar output, points of capacity",
                    anchor="start",
                ),
            )
        )
        zero = float(np.mean([d.max() <= 0 for d in daily]))
        out.notes["G3_aggregate"] = f"{len(daily)} days; share of zero-output days {round(zero, 2)}"
    out.results["G3_aggregate"] = bool(daily and bands_ok)
    singles = []
    for seg in s1[:8]:
        d = 1 + int(np.argmax(seg.solar[1:-1].max(axis=1)))
        if has_two_days(seg=seg, day=d):
            singles.append(
                window_chart(
                    series={"solar output": day_window(seg=seg, day=d, name="solar")},
                    title="G3 (private): a single S1 day",
                )
            )
    if singles:
        out.private["G3_private"] = alt.vconcat(*singles)
    out.results["G3_private"] = True
    out.notes["G3_private"] = f"{len(singles)} single days, private folder only"


def bsp_fold_base_gallery(*, segments: Sequence[sim729.Segment], out: GalleryResult) -> None:
    """Draw G4 (BSP), G5 (fold), and G6 (base days)."""
    segments = [s for s in segments if s.source != "S1"]
    both = [s for s in segments if s.world != "none" and s.has_bsp]
    hit = first_day(
        segments=both, keep=lambda s, d: bool(np.isfinite(s.bsp[d]).all() and s.engine[d].max() > 0)
    )
    if hit:
        seg, d = hit
        step = np.diff(day_window(seg=seg, day=d, name="bsp"), prepend=np.nan)
        inj = np.diff(
            day_window(seg=seg, day=d, name="engine") + day_window(seg=seg, day=d, name="solar"),
            prepend=np.nan,
        )
        out.public["G4_bsp"] = window_chart(
            series={"synthetic BSP step": step, "injection step": inj},
            title="G4: the synthetic BSP meter steps",
        )
        ok = bool(np.nanmax(np.abs(inj)) > 0 and np.nanmax(np.abs(step)) > 0)
    else:
        ok = False
    out.results["G4_bsp"] = ok
    out.notes["G4_bsp"] = "found" if hit else "no BSP day with an engine run"
    mixed = [s for s in segments if s.world in {"both", "engine"}]
    hit = first_day(
        segments=mixed, keep=lambda s, d: bool((s.x[d] < 0).any() and np.isfinite(s.r[d]).all())
    )
    if hit:
        seg, d = hit
        r, x = seg.r[d : d + 1].reshape(-1), seg.x[d : d + 1].reshape(-1)
        out.public["G5_fold"] = window_chart(
            series={"folded reading": r, "signed truth": x},
            title="G5: the folded reading and the signed truth",
            height=110,
        )
        q_max = seg.params["q_mva"] / core729.CAPACITY_MVA * core729.POINTS
        ok = gallery.check_fold(
            reading=r, truth=x, floor=float(q_max * seg.params["q_factor"]), q_max=float(q_max)
        )
    else:
        ok = False
    out.results["G5_fold"] = ok
    out.notes["G5_fold"] = "found" if hit else "no reverse-flow day"
    none = [s for s in segments if s.world == "none"]
    hit = first_day(segments=none, keep=lambda s, d: bool(np.isfinite(s.x[d]).all()))
    ok = bool(
        hit and gallery.check_base_equals_truth(truth=hit[0].x[hit[1]], base=hit[0].base[hit[1]])
    )
    if hit:
        seg, d = hit
        out.public["G6_base"] = window_chart(
            series={
                "signed truth": day_window(seg=seg, day=d, name="x"),
                "base demand": day_window(seg=seg, day=d, name="base"),
            },
            title="G6: a day with no injection",
        )
    out.results["G6_base"] = ok
    out.notes["G6_base"] = "found" if hit else "no no-injection day"


def mask_gallery(*, segments: Sequence[sim729.Segment], out: GalleryResult) -> None:
    """Draw G7: the masked reading beside the clean-demand target for K, W, R, and the causal cut."""
    segments = [s for s in segments if s.source != "S1"]
    hit = first_day(
        segments=segments,
        keep=lambda s, d: bool(np.isfinite(s.r[d : d + 2]).all() and has_two_days(seg=s, day=d)),
    )
    if hit is None:
        out.results["G7_masks"] = False
        out.notes["G7_masks"] = "no complete window"
        return
    seg, d = hit
    r = day_window(seg=seg, day=d, name="r")
    base = day_window(seg=seg, day=d, name="base")
    charts, ok = [], True
    for style in ("K", "W", "R"):
        hidden = masks.fixed_hidden(style=style, index=0, utc_offset=0.0)
        shown = np.where(hidden, 0.0, r)
        ok &= gallery.check_mask_panel(reading=shown, hidden=hidden)
        charts.append(
            window_chart(
                series={"reading (hidden is zero)": shown, "clean demand": base},
                hidden=hidden,
                title=f"G7: masking style {style}",
                height=90,
            )
        )
    cut = int(masks.fixed_cuts(n_windows=1)[0, 0]) if hasattr(masks, "fixed_cuts") else 48
    hidden = np.arange(96) > cut
    charts.append(
        window_chart(
            series={"reading (hidden is zero)": np.where(hidden, 0.0, r), "clean demand": base},
            hidden=hidden,
            title="G7: the causal cut",
            height=90,
        )
    )
    out.public["G7_masks"] = alt.vconcat(*charts)
    out.results["G7_masks"] = bool(ok)
    out.notes["G7_masks"] = "4 panels"


def hourly_share_chart(*, synthetic: np.ndarray, real: np.ndarray) -> alt.Chart:
    """Return the share of half-hours below 1 MVA by UTC hour, real and synthetic (24 points each)."""
    hours = np.arange(24)
    data = pl.DataFrame(
        {
            "hour": [*hours, *hours],
            "source": ["synthetic"] * 24 + ["real"] * 24,
            "share": [*np.asarray(synthetic, dtype=float), *np.asarray(real, dtype=float)],
        }
    )
    return (
        alt.Chart(data)
        .mark_line(point=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("hour:Q", title="UTC hour of the day"),
            y=alt.Y("share:Q", title="Share of half-hours below 1 MVA"),
            color=alt.Color(
                "source:N",
                scale=alt.Scale(domain=["synthetic", "real"], range=[ORANGE, BLUE]),
                legend=alt.Legend(orient="top"),
            ),
        )
        .properties(width=WIDTH_PX - 120, height=140)
    )


def aggregate_gallery(
    *, shares_synthetic: Mapping[str, float], shares_real: Mapping[str, float] | None,
    component_shares: Mapping[str, Mapping[str, float]], out: GalleryResult,
    bounded_shares: Mapping[str, float] | None = None,
    hourly_synthetic: np.ndarray | None = None, hourly_real: np.ndarray | None = None,
) -> None:  # fmt: skip
    """Draw G8 (real against simulated aggregates, when the real aggregates are supplied) and G9.

    G8 holds the share of half-hours below 1 MVA by world, and by UTC hour when both hourly arrays
    are given. Its automatic result checks only that the synthetic shares are finite and that the
    panel was drawn; the design lead compares the real and synthetic shares by eye.
    """
    names = list(shares_synthetic)
    if shares_real:
        pairs = figures729.share_pair_chart(
            groups=names, synthetic=[shares_synthetic[n] for n in names],
            real=[shares_real.get(n, float("nan")) for n in names], title="G8: real and synthetic aggregates",
        )  # fmt: skip
        if hourly_synthetic is not None and hourly_real is not None:
            hourly = hourly_share_chart(synthetic=hourly_synthetic, real=hourly_real)
            out.public["G8_realism"] = alt.vconcat(pairs, hourly)
        else:
            out.public["G8_realism"] = pairs
    out.results["G8_realism"] = bool(shares_real) and gallery.check_shares(shares=shares_synthetic)
    out.notes["G8_realism"] = "drawn" if shares_real else "real aggregates not supplied"
    minimum = {group: min(values.values()) for group, values in component_shares.items() if values}
    data = pl.DataFrame({"group": list(minimum), "share": list(minimum.values())})
    bar = (
        alt.Chart(pl.DataFrame({"x": [core.MIN_OUTPUT_SHARE]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color=ORANGE, strokeDash=[4, 3], aria=False)
        .encode(x="x:Q")
    )  # ty: ignore[unresolved-attribute]
    points = (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_point(color=BLUE, filled=True, size=40, aria=False)
        .encode(
            x=alt.X(
                "share:Q",
                title="Lowest share of segments with output in the group (bar in orange)",
                scale=alt.Scale(domain=[0.95, 1.001]),
            ),
            y=alt.Y("group:N", title=None, sort=None, axis=alt.Axis(labelLimit=260)),
        )
    )  # ty: ignore[unresolved-attribute]
    panels = [(points + bar).properties(width=WIDTH_PX - 260, height=12 * max(len(data), 1))]
    bounded_ok = True
    if bounded_shares:
        rows = [
            {"check": name, "share": value, "bound": bound_of(name=name)}
            for name, value in bounded_shares.items()
        ]
        bounded_ok = all(value <= bound_of(name=name) for name, value in bounded_shares.items())
        bound_data = pl.DataFrame(rows)
        dots = (
            alt.Chart(bound_data)  # ty: ignore[unresolved-attribute]
            .mark_point(color=BLUE, filled=True, size=40, aria=False)
            .encode(
                x=alt.X(
                    "share:Q", title="Share of excluded or zeroed half-hours (bound in orange)"
                ),
                y=alt.Y("check:N", title=None, sort=None, axis=alt.Axis(labelLimit=260)),
            )
        )  # ty: ignore[unresolved-attribute]
        ticks = (
            alt.Chart(bound_data)  # ty: ignore[unresolved-attribute]
            .mark_tick(color=ORANGE, thickness=2, aria=False)
            .encode(x="bound:Q", y=alt.Y("check:N", sort=None))
        )  # ty: ignore[unresolved-attribute]
        panels.append((dots + ticks).properties(width=WIDTH_PX - 260, height=12 * len(rows)))
    out.public["G9_components"] = alt.vconcat(*panels)
    out.results["G9_components"] = bool(
        minimum and gallery.check_shares(shares=minimum, bar=core.MIN_OUTPUT_SHARE) and bounded_ok
    )
    out.notes["G9_components"] = (
        f"{len(minimum)} groups, {len(bounded_shares or {})} bounded shares"
    )


def bound_of(*, name: str) -> float:
    """Return the upper bound of a bounded share, from the end of its check name."""
    return sky.MAX_ZEROED_SHARE if "zeroed_sunup" in name else banks.METERED_ZERO_MAX_SHARE


def build_gallery(
    *, segments: Sequence[sim729.Segment], shares_synthetic: Mapping[str, float],
    shares_real: Mapping[str, float] | None,
    component_shares: Mapping[str, Mapping[str, float]],
    bounded_shares: Mapping[str, float] | None = None,
    hourly_synthetic: np.ndarray | None = None, hourly_real: np.ndarray | None = None,
) -> GalleryResult:  # fmt: skip
    """Draw the whole gallery from the segments of the test bank and the training bank of seed 0."""
    out = GalleryResult()
    engine_gallery(segments=segments, out=out)
    solar_gallery(segments=segments, out=out)
    bsp_fold_base_gallery(segments=segments, out=out)
    mask_gallery(segments=segments, out=out)
    aggregate_gallery(
        shares_synthetic=shares_synthetic,
        shares_real=shares_real,
        component_shares=component_shares,
        bounded_shares=bounded_shares,
        out=out,
        hourly_synthetic=hourly_synthetic,
        hourly_real=hourly_real,
    )
    return out


def save_gallery(*, out: GalleryResult, public_dir: Path, private_dir: Path) -> list[Path]:
    """Write the published panels to `public_dir` and the S1 panels to `private_dir`."""
    written = []
    for folder, charts in ((public_dir, out.public), (private_dir, out.private)):
        for name, chart in charts.items():
            path = folder / f"{name}.svg"
            save_svg(chart=chart, path=path)
            written.append(path)
    return written


# --------------------------------------------------------------------------- result figures


def screen_chart(*, scores: Mapping[str, Mapping[str, float]]) -> alt.Chart:
    """Return figure 1: the validation Brier score of each architecture on arms S-A and S."""
    rows = [
        {"architecture": a, "arm": arm, "brier": v}
        for arm, by in scores.items()
        for a, v in by.items()
    ]
    return (
        alt.Chart(pl.DataFrame(rows))
        .mark_bar(aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("brier:Q", title="Validation Brier score of P(X<0) (smaller is better)"),
            y=alt.Y("architecture:N", title=None),
            yOffset="arm:N",
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(range=[SKY, BLUE]),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
        .properties(width=WIDTH_PX, height=34 * len(scores))
    )


def contrast_chart(
    *, readings: Mapping[str, tuple[float, float, float]], margin: float
) -> alt.Chart:
    """Return figure 3: the 7 counted contrasts with their intervals; zero and minus m are rules."""
    names = list(readings)
    data = pl.DataFrame(
        {
            "contrast": names,
            "point": [readings[n][0] for n in names],
            "lower": [readings[n][1] for n in names],
            "upper": [readings[n][2] for n in names],
        }
    )
    base = alt.Chart(data).encode(y=alt.Y("contrast:N", sort=names, title=None))  # ty: ignore[unresolved-attribute]
    bars = base.mark_rule(color=BLUE, strokeWidth=3, aria=False).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("lower:Q", title="Difference in Brier score (negative is a gain)"), x2="upper:Q"
    )  # ty: ignore[unresolved-attribute]
    dots = base.mark_circle(color=BLUE, size=60, aria=False).encode(x="point:Q")  # ty: ignore[unresolved-attribute]
    rules = (
        alt.Chart(pl.DataFrame({"x": [0.0, -margin]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[4, 3], color=ORANGE, aria=False)
        .encode(x="x:Q")
    )  # ty: ignore[unresolved-attribute]
    return (bars + dots + rules).properties(
        width=WIDTH_PX,
        height=26 * len(names),
        title=alt.TitleParams(
            "The 7 counted contrasts at the Bonferroni level; orange lines mark zero and minus m",
            anchor="start",
        ),
    )


def reliability_panels(
    *, curves: Mapping[str, tuple[Sequence[float], Sequence[float]]]
) -> alt.VConcatChart:
    """Return figure 4: one reliability diagram of P(X<0) per arm, stacked."""
    return alt.vconcat(
        *[
            figures729.reliability_chart(mean_forecast=m, frequency=f).properties(title=name)
            for name, (m, f) in curves.items()
        ]
    )


def demand_readout_chart(
    *, errors: Mapping[str, Mapping[str, float]], climatology: float
) -> alt.Chart:
    """Return figure 6: the demand error on hidden half-hours, training style against evaluation style."""
    rows = [
        {"trained": t, "evaluated": e, "mae": v} for t, by in errors.items() for e, v in by.items()
    ]
    heat = (
        alt.Chart(pl.DataFrame(rows))
        .mark_rect(aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("evaluated:N", title="Masking style at evaluation"),
            y=alt.Y("trained:N", title="Masking style in training"),
            color=alt.Color(
                "mae:Q", title="Mean absolute error (points)", scale=alt.Scale(scheme="oranges")
            ),
        )
    )
    return heat.properties(
        width=240,
        height=26 * len(errors),
        title=f"Demand output error on hidden half-hours; the demand climatology scores {climatology:.2f} points",
    )


def causal_readout_chart(
    *, brier_by_k: Mapping[int, float], two_sided: float, p0: float
) -> alt.Chart:
    """Return figure 7: the Brier score against the half-hours visible, with SD* and P0 as lines."""
    data = pl.DataFrame({"k": list(brier_by_k), "brier": list(brier_by_k.values())})
    line = (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True, color=ORANGE, aria=False)
        .encode(
            x=alt.X("k:Q", title="Half-hours of the scored day visible (k)"),
            y=alt.Y("brier:Q", title="Brier score (smaller is better)"),
        )
    )  # ty: ignore[unresolved-attribute]
    refs = (
        alt.Chart(pl.DataFrame({"y": [two_sided, p0], "kind": ["SD* two-sided", "P0"]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[4, 3], aria=False)
        .encode(
            y="y:Q",
            color=alt.Color(
                "kind:N",
                scale=alt.Scale(range=[BLUE, GREY]),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
    )  # ty: ignore[unresolved-attribute]
    return (line + refs).properties(width=WIDTH_PX, height=160)


def attention_chart(*, attention: np.ndarray, title: str) -> alt.Chart:
    """Return figure 8: a map of the mean attention from each scored half-hour to each window position."""
    q, k = np.meshgrid(np.arange(attention.shape[0]), np.arange(attention.shape[1]), indexing="ij")
    data = pl.DataFrame(
        {
            "scored": q.reshape(-1) / 2.0,
            "window": k.reshape(-1) / 2.0,
            "weight": attention.reshape(-1),
        }
    )
    return (
        alt.Chart(data)
        .mark_rect(aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "window:O",
                title="Window position (hours from the start of day 0)",
                axis=alt.Axis(values=[0, 12, 24, 36, 48]),
            ),
            y=alt.Y(
                "scored:O",
                title="Scored half-hour (hours into the scored day)",
                sort="descending",
                axis=alt.Axis(values=[0, 6, 12, 18, 24]),
            ),
            color=alt.Color("weight:Q", scale=alt.Scale(scheme="oranges"), title="Attention"),
        )
        .properties(width=WIDTH_PX - 80, height=200, title=alt.TitleParams(title, anchor="start"))
    )


def brier_dot_chart(*, scores: Mapping[str, Mapping[str, float]]) -> alt.Chart:
    """Return figure 2: the Brier score of every arm and baseline on all, import, reverse, and bands."""
    subsets = list(next(iter(scores.values())))
    return figures729.dot_chart(
        scores=scores,
        rows=subsets,
        title="Brier score of P(X<0) by row subset and reading band (smaller is better)",
        x_title="Brier score",
    )


def signed_crps_dot_chart(*, scores: Mapping[str, Mapping[str, float]]) -> alt.Chart:
    """Return figure 9: the signed CRPS of P0f, B1, A, D, and D-sign on all, import, and reverse rows."""
    return figures729.dot_chart(
        scores=scores,
        rows=list(next(iter(scores.values()))),
        title="Signed CRPS of experiment 729's readings, redone (smaller is better)",
        x_title="Signed CRPS (points of capacity)",
    )


def harm_detection_chart(
    *,
    by_arm: Mapping[str, Mapping[str, Mapping[str, float]]],
    limit: float,
    bar: float = core.DETECTION_BAR,
) -> alt.VConcatChart:
    """Return figure 5: the harm and detection numbers by world type and reading band.

    Args:
        by_arm: For each arm, `harm` and `detection`, each a mapping of a group to its number.
        limit: The harm limit L, drawn on the top panel.
        bar: The detection bar, drawn on the bottom panel.

    Returns:
        Two stacked panels of staggered dots with a reference rule each.
    """
    panels = []
    for key, ref, title in (
        ("harm", limit, "Harm: mean P(X<0) on decidable import rows (smaller is better)"),
        (
            "detection",
            bar,
            "Detection: mean P(X<0) on decidable reverse-flow rows (larger is better)",
        ),
    ):
        scores = {arm: v[key] for arm, v in by_arm.items()}
        dots = figures729.dot_chart(
            scores=scores,
            rows=list(next(iter(scores.values()))),
            title=title,
            x_title="Mean P(X<0)",
        )
        rule = (
            alt.Chart(pl.DataFrame({"x": [ref]}))  # ty: ignore[unresolved-attribute]
            .mark_rule(strokeDash=[4, 3], color=GREY, aria=False)
            .encode(x="x:Q")
        )
        panels.append(dots + rule)
    return alt.vconcat(*panels)


def real_eras_chart(
    *, above_half: Mapping[str, float], demand_above: Mapping[str, float]
) -> alt.Chart:
    """Return figure 10: shares of real half-hours by UTC hour and era.

    Args:
        above_half: The share with P(X<0) above 0.5, keyed `era/hour`.
        demand_above: The share with the demand output above the reading, keyed `era/hour`.

    Returns:
        One line panel per era with the two shares against the UTC hour.
    """
    rows = [
        {"era": k.rsplit("/", 1)[0], "hour": int(k.rsplit("/", 1)[1]), "share": v, "kind": kind}
        for kind, series in (
            ("P(X<0) above 0.5", above_half),
            ("demand output above the reading", demand_above),
        )
        for k, v in series.items()
    ]
    return (
        alt.Chart(pl.DataFrame(rows))
        .mark_line(point=True, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "hour:Q", title="UTC hour of the day", axis=alt.Axis(values=[0, 6, 12, 18, 24])
            ),
            y=alt.Y("share:Q", title="Share of half-hours"),
            color=alt.Color(
                "kind:N",
                scale=alt.Scale(range=[ORANGE, BLUE]),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
        .properties(width=WIDTH_PX // 2 - 40, height=110)
        .facet(facet=alt.Facet("era:N", title=None), columns=2)
    )
