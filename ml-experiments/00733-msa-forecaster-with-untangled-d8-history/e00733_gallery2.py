"""Experiment 733: gallery 2, series plots of the history rows and of the synthetic arms.

Gallery 1 (`e00733_gallery`) checks the world's banks with summary panels. Gallery 2 draws time
series, so that a reader can see what the forecasters receive. Every file goes to
`PRIVATE/gallery2/`, and the stage writes `panel_check2.json` beside them.

**Real history rows (D8).** Each example is 48 hours of history, the history rows of ages 5 and 4,
so the stored untangled channels are read without the re-run. The channels come through
`e00733_inputs.gather`, the function the fits call, for the arms FU and FU-S. The origins are the
held-out validation origins of experiment 422's cycle (`prep422.purge_masks`), from the first
origin day of experiment 713's candidate to the day before the first fold month, so no test month
is read. The deterministic rule (`choose_windows`) takes, with at least 4 days between origins:

- the 3 windows with the highest P(reverse flow) (reverse-flow examples);
- the window with the highest load-transfer step probability (step example);
- among the windows whose highest P(reverse flow) is below 0.5 and whose mean clear-sky
  irradiance is in the top quarter of those windows: the highest mean clear-sky index (sunny
  example), the lowest (cloudy example), and the one closest to the median (typical example).

The clear-sky index is the ratio of the measured to the clear-sky irradiance over the daylight
half-hours (clear-sky irradiance above 100 W/m2) of the 48 hours.

**Synthetic arms.** One panel per arm Y0, YU, and YT draws the same two days of one segment of
source S2 of the validation bank: the reading before and after the injection, and the channels
that the arm receives. A segment of source S1 is never drawn.

No title carries a date, a capacity, or a series identifier.
"""

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final

import e00733_channels as ch
import e00733_core as core
import e00733_gallery as gal
import e00733_paths as paths
import e00733_synth as sy
import e00733_world as world
import numpy as np
import polars as pl

# isort: split
import altair as alt
import e00729_figures as figures729

WIDTH_PX: Final = figures729.WIDTH_PX
BLUE: Final = figures729.BLUE
ORANGE: Final = figures729.ORANGE
GREY: Final = figures729.GREY
PURPLE: Final = figures729.PURPLE
STEPS: Final = core.STEPS_PER_DAY
SHOWN_AGES: Final = (5, 4)
"""The ages of the two history rows drawn, oldest first (the stored channels, no re-run)."""
SHOWN_STEPS: Final = STEPS * len(SHOWN_AGES)
CANDIDATE_FIRST_DAY: Final = 1674
"""The grid day of experiment 713's candidate day 0 (`e00713_train.candidate_epoch_days`)."""
SEPARATION_DAYS: Final = 4
"""Two chosen origins differ by at least this many days, so no window overlaps another."""
MIN_SHARE_PRESENT: Final = 0.9
"""A candidate window needs this share of half-hours with a reading and a channel flag."""
DAYLIGHT_CLEAR_W_M2: Final = 100.0
HALF_HOURS_ABOVE_HALF: Final = 2
"""A reverse-flow example needs P(reverse flow) above 0.5 on at least this many half-hours."""
ZERO_MW: Final = 1e-6
SUN_QUANTILE: Final = 0.75
"""Sunny, cloudy, and typical windows come from the quarter with the most clear-sky irradiance."""
TICKS_48H: Final = [0, 6, 12, 18, 24, 30, 36, 42, 48]
KINDS_REAL: Final = (
    "reverse flow",
    "reverse flow",
    "reverse flow",
    "step",
    "sunny",
    "cloudy",
    "typical",
)
SYNTHETIC_ARMS: Final = ("Y0", "YU", "YT")
G14_BIN_HOURS: Final = 6.0
G14_BIG_DIFFERENCE: Final = 0.5


class Gallery2Error(RuntimeError):
    """A gallery 2 input is missing or no window satisfies a selection rule."""


@dataclass(frozen=True)
class HistoryExample:
    """One real 48-hour history window of D8 as FU and FU-S receive it.

    Attributes:
        name: The panel name (H1 to H7).
        kind: The selection rule that picked it.
        target_mw: `(96,)` the folded reading in MW, NaN where absent.
        fu: `(96, 3)` FU's input channels `[u, v, w]` (zero where the flag is off).
        fu_ok: `(96,)` FU's presence flag.
        fus: `(96, 3)` FU-S's input channels.
        fus_ok: `(96,)` FU-S's presence flag.
        stored_solar_mw: `(96,)` the stored solar output in MW that the rule drops (NaN: absent).
        daylight: `(96,)` whether the half-hour has clear-sky irradiance above the daylight floor.
        usable: The channels that the rule lets into FU.
        solar_mw_per_unit: MW of unmetered solar per unit of the input channel `v`.
    """

    name: str
    kind: str
    target_mw: np.ndarray
    fu: np.ndarray
    fu_ok: np.ndarray
    fus: np.ndarray
    fus_ok: np.ndarray
    stored_solar_mw: np.ndarray
    daylight: np.ndarray
    usable: tuple[str, ...]
    solar_mw_per_unit: float


# --------------------------------------------------------------------------- the selection rule


def separated(*, day: int, chosen: Sequence[int]) -> bool:
    """Return whether `day` is at least `SEPARATION_DAYS` from every chosen origin day."""
    return all(abs(day - c) >= SEPARATION_DAYS for c in chosen)


def choose_windows(
    *,
    origin_day: np.ndarray,
    p_max: np.ndarray,
    step_max: np.ndarray,
    csi_mean: np.ndarray,
    clear_mean: np.ndarray,
) -> list[tuple[str, int]]:
    """Return the `(kind, candidate index)` of the examples, by the rule in the module docstring.

    Ties go to the earlier origin day. Every window of the result is a different candidate, and the
    origin days are at least `SEPARATION_DAYS` apart.

    Args:
        origin_day: `(n,)` the grid day of each candidate origin.
        p_max: `(n,)` the highest P(reverse flow) of each window.
        step_max: `(n,)` the highest step probability of each window.
        csi_mean: `(n,)` the mean clear-sky index over daylight (NaN with no daylight).
        clear_mean: `(n,)` the mean clear-sky irradiance of the window.

    Returns:
        The kinds in the order of `KINDS_REAL`.

    Raises:
        Gallery2Error: When a kind finds no candidate.
    """
    chosen_days: list[int] = []
    out: list[tuple[str, int]] = []

    def take(*, kind: str, key: np.ndarray, allowed: np.ndarray) -> None:
        order = np.lexsort((origin_day, key))
        for i in order:
            if (
                allowed[i]
                and np.isfinite(key[i])
                and separated(day=int(origin_day[i]), chosen=chosen_days)
            ):
                chosen_days.append(int(origin_day[i]))
                out.append((kind, int(i)))
                return
        msg = f"no candidate window is left for the {kind} example"
        raise Gallery2Error(msg)

    everyone = np.ones(len(origin_day), dtype=bool)
    for _ in range(KINDS_REAL.count("reverse flow")):
        take(kind="reverse flow", key=-p_max, allowed=everyone)
    take(kind="step", key=-step_max, allowed=everyone)
    quiet = (p_max < 0.5) & np.isfinite(csi_mean)
    if not quiet.any():
        msg = "no candidate window has P(reverse flow) below 0.5, so no sunny day exists"
        raise Gallery2Error(msg)
    sunny_enough = quiet & (clear_mean >= np.quantile(clear_mean[quiet], SUN_QUANTILE))
    take(kind="sunny", key=-csi_mean, allowed=sunny_enough)
    take(kind="cloudy", key=csi_mean, allowed=sunny_enough)
    middle = float(np.median(csi_mean[sunny_enough]))
    take(kind="typical", key=np.abs(csi_mean - middle), allowed=sunny_enough)
    return out


# --------------------------------------------------------------------------- real windows


def dense_on_grid(*, dates: np.ndarray, values: np.ndarray, n_grid: int) -> np.ndarray:
    """Return a `(n_days, 48)` day table on the grid `(n_grid,)`, NaN where no day holds a value."""
    epoch = np.asarray(dates).astype("datetime64[D]").astype(np.int64)
    gi = ch.grid_index(
        day=np.repeat(epoch[:, None], STEPS, axis=1), hh=np.tile(np.arange(STEPS), (len(epoch), 1))
    )
    out = np.full(n_grid, np.nan)
    inside = (gi >= 0) & (gi < n_grid)
    out[gi[inside]] = np.asarray(values, dtype=np.float64)[inside]
    return out


def history_grid_index(*, origin_day: np.ndarray) -> np.ndarray:
    """Return the `(n, 14, 48)` grid index of every history token of each origin day.

    The contract origin is `c = 48 D + 24`, and history row `a` holds `c - 48 a + j`.
    """
    c = np.asarray(origin_day, dtype=np.int64) * STEPS + 24
    ages = np.arange(core.MIN_AGE, core.MAX_AGE + 1)
    return c[:, None, None] - STEPS * ages[None, :, None] + np.arange(STEPS)[None, None, :]


def shown_rows() -> list[int]:
    """Return the indices (0 for age 2) of the history rows that the figures draw."""
    return [a - core.MIN_AGE for a in SHOWN_AGES]


def candidate_origin_days(*, last_day: int) -> np.ndarray:
    """Return the grid days of the held-out validation origins from candidate day 0 to `last_day`.

    The held-out origins are those of experiment 422's cycle (`purge_masks`), counted from
    experiment 713's candidate day 0. `last_day` is exclusive: it is the first day of the first
    fold month, so no test month enters.

    Args:
        last_day: The first grid day that is not a validation day.

    Returns:
        The sorted grid days.
    """
    import e00422_prep as prep422  # ty: ignore[unresolved-import]

    days = np.arange(CANDIDATE_FIRST_DAY, last_day)
    held, _purged = prep422.purge_masks(origin_day=days - CANDIDATE_FIRST_DAY)
    return days[np.asarray(held, dtype=bool)]


def receive_channels(
    *,
    arm: str,
    choice: ch.ChannelChoice,
    table: ch.ChannelTable,
    origin_day: np.ndarray,
    reading_ok: np.ndarray,
    n_grid: int,
    target_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the channels of the shown history rows, as the fit's input code gathers them.

    Args:
        arm: FU or FU-S.
        choice: The rule's channel choice.
        table: The stored channel table.
        origin_day: `(n,)` the grid day of each origin.
        reading_ok: `(n_grid,)` whether D8's reading exists at each grid index.
        n_grid: The number of grid half-hours.
        target_std: The target standard deviation that scales the solar channel.

    Returns:
        The channels `(n, 2, 48, 3)` as `[u, v, w]` and the flag `(n, 2, 48)`.
    """
    import e00733_inputs as inputs
    import torch

    tensors = inputs.arm_tensors(
        arm=arm, choice=choice, table=table, rerun_outputs=None, n_grid=n_grid, n_primaries=1,
        d8_index=0, target_std=target_std,
    )  # fmt: skip
    hist = history_grid_index(origin_day=origin_day)
    inside = (hist >= 0) & (hist < n_grid)
    visible = np.zeros(hist.shape, dtype=bool)
    visible[inside] = reading_ok[hist[inside]]
    n = len(origin_day)
    got = inputs.gather(
        tensors=tensors,
        primary=torch.zeros(n, dtype=torch.long),
        hist_idx=torch.from_numpy(hist),
        c_idx=torch.from_numpy(np.asarray(origin_day, dtype=np.int64) * STEPS + 24),
        power_visible=torch.from_numpy(visible),
    )  # fmt: skip
    rows = shown_rows()
    values = got.channels.numpy()[:, rows].reshape(n, 2, STEPS, core.N_CHANNELS)
    return values, got.ok.numpy()[:, rows].reshape(n, 2, STEPS)


def mean_absolute_difference(*, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return the mean absolute difference over the half-hours, per channel `(3,)`."""
    return np.abs(first - second).mean(axis=0)


def build_history_examples(
    *,
    origin_day: np.ndarray,
    target_mw: np.ndarray,
    fu: np.ndarray,
    fu_ok: np.ndarray,
    fus: np.ndarray,
    fus_ok: np.ndarray,
    stored_solar_mw: np.ndarray,
    csi: np.ndarray,
    clear: np.ndarray,
    usable: tuple[str, ...],
    solar_mw_per_unit: float,
) -> list[HistoryExample]:
    """Choose the windows by the rule and return them as examples.

    Args:
        origin_day: `(n,)` the candidate origin days.
        target_mw: `(n, 96)` the folded reading in MW.
        fu: `(n, 96, 3)` FU's channels.
        fu_ok: `(n, 96)` FU's flag.
        fus: `(n, 96, 3)` FU-S's channels.
        fus_ok: `(n, 96)` FU-S's flag.
        stored_solar_mw: `(n, 96)` the stored solar output that the rule drops, in MW.
        csi: `(n, 96)` the clear-sky index (NaN with no daylight value).
        clear: `(n, 96)` the clear-sky irradiance in W/m2.
        usable: The channels that the rule lets into FU.
        solar_mw_per_unit: MW of unmetered solar per unit of the input channel `v`.

    Returns:
        One example per kind of `KINDS_REAL`, named H1 to H7.

    Raises:
        Gallery2Error: When no candidate window has enough readings and flags.
    """
    daylight = np.nan_to_num(clear) > DAYLIGHT_CLEAR_W_M2
    present = (np.isfinite(target_mw).mean(axis=1) >= MIN_SHARE_PRESENT) & (
        fu_ok.mean(axis=1) >= MIN_SHARE_PRESENT
    )
    if not present.any():
        msg = "no validation window holds a reading and a channel flag on 90% of its half-hours"
        raise Gallery2Error(msg)
    n_day = daylight.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        csi_mean = np.where(
            n_day > 0, np.where(daylight, np.nan_to_num(csi), 0.0).sum(axis=1) / n_day, np.nan
        )
    keep = np.flatnonzero(present)
    p_max = np.where(fu_ok, fu[..., core.P_NEG], 0.0).max(axis=1)
    step_max = np.where(fu_ok, fu[..., core.STEP_P], 0.0).max(axis=1)
    picked = choose_windows(
        origin_day=origin_day[keep], p_max=p_max[keep], step_max=step_max[keep],
        csi_mean=csi_mean[keep], clear_mean=np.nan_to_num(clear[keep]).mean(axis=1),
    )  # fmt: skip
    out = []
    for i, (kind, k) in enumerate(picked):
        w = int(keep[k])
        out.append(
            HistoryExample(
                name=f"H{i + 1}", kind=kind, target_mw=target_mw[w], fu=fu[w], fu_ok=fu_ok[w],
                fus=fus[w], fus_ok=fus_ok[w], stored_solar_mw=stored_solar_mw[w],
                daylight=daylight[w], usable=usable, solar_mw_per_unit=solar_mw_per_unit,
            )
        )  # fmt: skip
    return out


def check_history_example(*, example: HistoryExample) -> gal.PanelResult:
    """Return the automatic check of one real example.

    Every example needs a non-empty series and flags. A reverse-flow example needs P(reverse flow)
    above 0.5 on at least 2 half-hours. The step example needs a step probability above 0.001. The
    sunny example needs a stored solar output above zero on a daylight half-hour, and FU's solar
    channel must follow the rule (zero when the rule drops it, above zero in daylight when not).
    Every example needs FU-S to differ from FU.

    Args:
        example: The example.

    Returns:
        The result; its data hold the mean absolute differences between FU and FU-S.
    """
    u, v, w = (example.fu[:, i] for i in range(core.N_CHANNELS))
    items = {
        "target_series_non_empty": bool(np.isfinite(example.target_mw).mean() >= MIN_SHARE_PRESENT),
        "fu_channels_present": bool(example.fu_ok.mean() >= MIN_SHARE_PRESENT),
    }
    diff = mean_absolute_difference(first=example.fu, second=example.fus)
    items["placebo_differs_from_fu"] = bool(diff.sum() > ZERO_MW)
    if example.kind == "reverse flow":
        items["p_reverse_above_half_on_two_half_hours"] = bool(
            (u > 0.5).sum() >= HALF_HOURS_ABOVE_HALF
        )
    if example.kind == "step":
        items["step_probability_nonzero"] = bool(w.max() > 1e-3)
    if example.kind == "sunny":
        sun = example.daylight & np.isfinite(example.stored_solar_mw)
        items["stored_solar_nonzero_in_daylight"] = bool(
            (example.stored_solar_mw[sun] > ZERO_MW).any()
        )
        dropped = "solar" not in example.usable
        items["fu_solar_channel_follows_the_rule"] = bool(
            (not v.any()) if dropped else (v[example.daylight] > 0).any()
        )
    return gal.PanelResult(
        panel=example.name, passed=all(items.values()), items=items,
        data={"mean_abs_difference": diff, "kind": example.kind},
    )  # fmt: skip


# --------------------------------------------------------------------------- charts


def _frame(*, hours: np.ndarray, series: Mapping[str, np.ndarray]) -> pl.DataFrame:
    parts = [
        pl.DataFrame({"hour": hours, "value": np.asarray(v, dtype=np.float64), "series": name})
        for name, v in series.items()
    ]
    return pl.concat(parts).filter(pl.col("value").is_not_nan() & pl.col("value").is_not_null())


def line_panel(
    *,
    hours: np.ndarray,
    series: Mapping[str, np.ndarray],
    colours: Mapping[str, str],
    y_title: str,
    y_domain: tuple[float, float] | None,
    height: int,
    bottom: bool,
    legend: bool,
    label_expr: str,
    dashes: Mapping[str, list[int]] | None = None,
) -> alt.Chart:
    """Return one stacked line panel; only the bottom panel of a stack labels its x axis.

    Args:
        hours: `(n,)` the x values in hours since the start of the window.
        series: The y values by series name.
        colours: The colour of each series name.
        y_title: The y-axis title with the unit.
        y_domain: The y range, or None for the data range from zero.
        height: The plot height in pixels.
        bottom: Whether the panel draws the x-axis labels and title.
        legend: Whether the panel draws the colour key above itself.
        label_expr: The Vega expression of an x-axis label (`datum.value` is the tick).
        dashes: The stroke dash of a series, if any.

    Returns:
        The chart.
    """
    frame = _frame(hours=hours, series=series)
    names = list(series)
    x_axis = alt.Axis(
        values=TICKS_48H, labelExpr=label_expr, labels=bottom, ticks=True,
        title="Hour of day (UTC, period end)" if bottom else None,
    )  # fmt: skip
    scale = alt.Scale(domain=list(y_domain)) if y_domain is not None else alt.Scale(zero=True)
    dash_range = [(dashes or {}).get(n, []) for n in names]
    chart = (
        alt.Chart(frame)
        .mark_line(aria=False, clip=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("hour:Q", axis=x_axis, scale=alt.Scale(domain=[0, 48], nice=False)),
            y=alt.Y("value:Q", title=y_title, scale=scale),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(domain=names, range=[colours[n] for n in names]),
                legend=alt.Legend(orient="top", title=None) if legend else None,
            ),
            strokeDash=alt.StrokeDash(
                "series:N", scale=alt.Scale(domain=names, range=dash_range), legend=None
            ),
        )
    )  # fmt: skip
    return chart.properties(width=WIDTH_PX, height=height)


def stack(*, panels: Sequence[alt.Chart], title: str, subtitle: Sequence[str]) -> alt.VConcatChart:
    """Return the panels stacked on one shared time axis under a title and a subtitle."""
    return (
        alt.vconcat(*panels, spacing=6)
        .properties(title=alt.TitleParams(title, subtitle=list(subtitle), anchor="start"))
        .resolve_scale(x="shared", color="independent", strokeDash="independent")
    )


def history_chart(*, example: HistoryExample, with_placebo: bool) -> alt.VConcatChart:
    """Return the stacked chart of one real example.

    Without the placebo, the chart draws the folded reading and FU's three channels. With it, the
    chart draws FU and FU-S on each channel and prints the mean absolute differences.

    Args:
        example: The example.
        with_placebo: Whether to draw FU-S beside FU.

    Returns:
        The chart.
    """
    hours = (np.arange(SHOWN_STEPS) + 1) / 2.0
    label_expr = "(datum.value + 12) % 24"
    panels: list[alt.Chart] = []
    colours = {"FU": BLUE, "FU-S": ORANGE, "stored output, not given to FU": GREY, "reading": GREY}
    channels = [
        ("P(reverse flow)", core.P_NEG, (0.0, 1.0)),
        ("Unmetered solar (MW)", core.SOLAR, None),
        ("Load-transfer step probability", core.STEP_P, (0.0, 1.0)),
    ]
    if not with_placebo:
        panels.append(
            line_panel(
                hours=hours, series={"reading": example.target_mw},
                colours={**colours, "reading": BLUE},
                y_title="Folded target (MW)", y_domain=None, height=100, bottom=False, legend=False,
                label_expr=label_expr,
            )
        )  # fmt: skip
    for i, (label, idx, domain) in enumerate(channels):
        unit = example.solar_mw_per_unit if idx == core.SOLAR else 1.0
        series = {"FU": example.fu[:, idx] * unit}
        if with_placebo:
            series["FU-S"] = example.fus[:, idx] * unit
        if idx == core.SOLAR and not with_placebo and "solar" not in example.usable:
            series["stored output, not given to FU"] = example.stored_solar_mw
        panels.append(
            line_panel(
                hours=hours, series=series, colours=colours, y_title=label, y_domain=domain,
                height=62, bottom=i == len(channels) - 1, legend=with_placebo and i == 0,
                label_expr=label_expr, dashes={"FU-S": [5, 3]},
            )
        )  # fmt: skip
    diff = mean_absolute_difference(first=example.fu, second=example.fus)
    if with_placebo:
        title = f"{example.kind.capitalize()} example: FU and the placebo FU-S"
        subtitle = [
            "FU-S reads the same channels taken 28 days earlier.",
            "Mean absolute difference over the 96 half-hours:",
            (
                f"P(reverse flow) {diff[core.P_NEG]:.3f}, unmetered solar "
                f"{diff[core.SOLAR] * example.solar_mw_per_unit:.3f} MW, "
                f"step probability {diff[core.STEP_P]:.3f}."
            ),
        ]
    else:
        title = f"{example.kind.capitalize()} example: 48 hours of history that FU receives"
        subtitle = [
            "Folded target and the untangler's channels, history rows of ages 5 and 4, as the fit",
            "reads them. The solar channel is zero when the channel rule drops it; the grey line",
            "is the stored solar output that FU is not given. Validation origins only.",
        ]
    return stack(panels=panels, title=title, subtitle=subtitle)


# --------------------------------------------------------------------------- synthetic arms


@dataclass(frozen=True)
class SyntheticWindow:
    """Two days of one S2 segment of the synthetic validation bank.

    Attributes:
        before_mw: `(96,)` the folded base demand without the injection, in MW.
        after_mw: `(96,)` the folded reading with the injection, in MW.
        engine_mw: `(96,)` the injected engine output in MW.
        solar_mw: `(96,)` the injected solar output in MW.
        x_neg: `(96,)` the indicator of X < 0.
        untangled_p: `(96,)` the untangled P(X<0), NaN where the flag is off.
        untangled_solar_mw: `(96,)` the untangled solar output in MW, NaN where the flag is off.
    """

    before_mw: np.ndarray
    after_mw: np.ndarray
    engine_mw: np.ndarray
    solar_mw: np.ndarray
    x_neg: np.ndarray
    untangled_p: np.ndarray
    untangled_solar_mw: np.ndarray


def match_bank_segment(
    *, arrays: sy.SegmentArrays, bank_segments: Sequence[Any], constants: Mapping[str, float]
) -> int | None:
    """Return the index of the bank segment that `arrays` was built from, or None.

    The world's folded reading is the bank's reading in points, standardised, so the match compares
    the two on the half-hours where both exist.

    Args:
        arrays: One world segment.
        bank_segments: The segments of experiment 729's bank, in bank order.
        constants: The reading's `mean` and `std` in fractions of capacity.

    Returns:
        The index into `bank_segments`.
    """
    want = np.asarray(arrays.reading_z, dtype=np.float64)
    for i, seg in enumerate(bank_segments):
        r = np.asarray(seg.r, dtype=np.float64).reshape(-1)
        if len(r) != len(want):
            continue
        z = (r / 100.0 - constants["mean"]) / constants["std"]
        both = np.isfinite(z) & np.isfinite(want)
        if both.any() and np.allclose(z[both], want[both], atol=1e-3):
            return i
    return None


def choose_synthetic_window(
    *, arrays: Sequence[sy.SegmentArrays], sources: Sequence[str]
) -> tuple[int, int]:
    """Return the `(segment index, first day)` of two days of an S2 segment to draw.

    The two days must be consecutive base days with untangled channels on every half-hour. The
    choice maximises the number of reverse-flow half-hours (ties to the lowest segment and day) and
    requires some injected solar output.

    Args:
        arrays: The validation bank's world segments.
        sources: The source ("S1" or "S2") of each segment. A segment of source S1 is never chosen.

    Returns:
        The segment index and the dense index of the first of the two days.

    Raises:
        Gallery2Error: When no S2 segment has two such days with reverse flow.
    """
    best: tuple[int, int, int] | None = None
    for i, (seg, source) in enumerate(zip(arrays, sources, strict=True)):
        if source != "S2":
            continue
        for d in range(seg.n_days - 1):
            w = slice(d * STEPS, (d + 2) * STEPS)
            if not (seg.u_ok[w].all() and np.isfinite(seg.x_neg[w]).all()):
                continue
            n_rev = int(seg.x_neg[w].sum())
            has_solar = float(seg.solar_points[w].max()) > 0.0
            if n_rev > 0 and has_solar and (best is None or n_rev > best[0]):
                best = (n_rev, i, d)
    if best is None:
        msg = "no S2 segment of the validation bank has two consecutive days with reverse flow"
        raise Gallery2Error(msg)
    return best[1], best[2]


def synthetic_window(
    *, arrays: sy.SegmentArrays, segment: Any, first_day: int, mw_per_point: float
) -> SyntheticWindow:
    """Return the window of two days from the world segment and the bank segment behind it.

    Args:
        arrays: The world segment.
        segment: The bank's segment (it holds the base demand and the injected series).
        first_day: The dense index of the first day.
        mw_per_point: MW in one point (1% of capacity).

    Returns:
        The window.
    """
    w = slice(first_day * STEPS, (first_day + 2) * STEPS)
    ok = arrays.u_ok[w]
    u = np.asarray(arrays.u_raw[w], dtype=np.float64)
    return SyntheticWindow(
        before_mw=np.abs(np.asarray(segment.base, dtype=np.float64).reshape(-1)[w]) * mw_per_point,
        after_mw=np.asarray(segment.r, dtype=np.float64).reshape(-1)[w] * mw_per_point,
        engine_mw=np.asarray(segment.engine, dtype=np.float64).reshape(-1)[w] * mw_per_point,
        solar_mw=np.asarray(segment.solar, dtype=np.float64).reshape(-1)[w] * mw_per_point,
        x_neg=np.asarray(arrays.x_neg[w], dtype=np.float64),
        untangled_p=np.where(ok, u[:, core.P_NEG], np.nan),
        untangled_solar_mw=np.where(ok, u[:, core.SOLAR] * mw_per_point, np.nan),
    )


def arm_signals(*, arm: str, window: SyntheticWindow) -> dict[str, np.ndarray | None]:
    """Return the history channels that an arm receives: P(X<0) and the solar output in MW.

    Y0 receives none (None), YU the untangler's outputs, and YT the true sign and solar output.
    """
    if arm == "Y0":
        return {"p": None, "solar": None}
    if arm == "YU":
        return {"p": window.untangled_p, "solar": window.untangled_solar_mw}
    if arm == "YT":
        return {"p": window.x_neg, "solar": window.solar_mw}
    msg = f"{arm} is not a synthetic arm"
    raise Gallery2Error(msg)


def synthetic_chart(*, arm: str, window: SyntheticWindow) -> alt.VConcatChart:
    """Return the stacked chart of one synthetic arm: the injection and what the arm receives.

    Args:
        arm: Y0, YU, or YT.
        window: The shared window.

    Returns:
        The chart.
    """
    hours = np.arange(2 * STEPS) / 2.0
    label_expr = "datum.value % 24"
    colour = {"Y0": GREY, "YU": BLUE, "YT": PURPLE}[arm]
    got = arm_signals(arm=arm, window=window)
    panels = [
        line_panel(
            hours=hours,
            series={"before injection": window.before_mw, "after injection": window.after_mw},
            colours={"before injection": GREY, "after injection": colour},
            y_title="Folded reading (MW)", y_domain=None, height=90, bottom=False, legend=True,
            label_expr=label_expr, dashes={"before injection": [5, 3]},
        ),
        line_panel(
            hours=hours,
            series={"injected engine": window.engine_mw, "injected solar": window.solar_mw},
            colours={"injected engine": ORANGE, "injected solar": figures729.SKY},
            y_title="Injection (MW)", y_domain=None, height=62, bottom=False, legend=False,
            label_expr=label_expr,
        ),
    ]  # fmt: skip
    rows = (("p", "P(reverse flow)", (0.0, 1.0), False), ("solar", "Solar (MW)", None, True))
    for key, label, domain, last in rows:
        values = got[key]
        series = {"truth": window.x_neg if key == "p" else window.solar_mw}
        colours = {"truth": GREY, f"{arm} receives": colour}
        if values is not None:
            series[f"{arm} receives"] = values
        else:
            colours = {"truth": GREY}
        panels.append(
            line_panel(
                hours=hours, series=series, colours=colours, y_title=label, y_domain=domain,
                height=62, bottom=last, legend=False, label_expr=label_expr,
                dashes={"truth": [5, 3]},
            )
        )  # fmt: skip
    if arm == "Y0":
        title = "Y0 (no channel is injected into its history): the folded reading only"
        note = "The grey truth lines are drawn for reference only: Y0 receives no channel."
    elif arm == "YU":
        title = "YU: the untangler's outputs on the injected series"
        note = "The coloured lines are the untangler's P(reverse flow) and solar output."
    else:
        title = "YT: the true sign and the true injected solar output"
        note = "The coloured lines are the oracle channels, equal to the injected truth."
    return stack(
        panels=panels, title=title,
        subtitle=[
            "Two days of one synthetic segment (source S2) without and with the injected engine",
            f"and solar output. {note}",
        ],
    )  # fmt: skip


def check_synthetic_arm(*, arm: str, window: SyntheticWindow) -> gal.PanelResult:
    """Return the automatic check of one synthetic arm.

    All arms need a non-empty reading and an injection that changes the reading. Y0 needs no
    channel. YU needs P(reverse flow) above 0.5 on most reverse-flow half-hours and a nonzero solar
    output where solar was injected. YT needs reverse flow in its true sign.

    Args:
        arm: Y0, YU, or YT.
        window: The shared window.

    Returns:
        The result.
    """
    got = arm_signals(arm=arm, window=window)
    items = {
        "reading_non_empty": bool(np.isfinite(window.after_mw).all() and window.after_mw.max() > 0),
        "injection_changes_the_reading": bool(
            np.nanmax(np.abs(window.after_mw - window.before_mw)) > ZERO_MW
        ),
    }
    rev = window.x_neg > 0.5
    if arm == "Y0":
        items["no_channel_received"] = got["p"] is None and got["solar"] is None
    if arm == "YU" and got["p"] is not None and got["solar"] is not None:
        items["p_reverse_above_half_on_most_reverse_half_hours"] = bool(
            rev.any() and np.nanmean(got["p"][rev] > 0.5) > gal.MOST
        )
        sun = window.solar_mw > ZERO_MW
        items["untangled_solar_nonzero_where_solar_injected"] = bool(
            sun.any() and np.nanmax(np.abs(got["solar"][sun])) > ZERO_MW
        )
    if arm == "YT":
        items["true_sign_has_reverse_flow"] = bool(rev.any())
    return gal.PanelResult(
        panel=f"S_{arm}", passed=all(items.values()), items=items, data={}
    )  # fmt: skip


# --------------------------------------------------------------------------- G14 explained


def g14_counts(*, segments: Sequence[sy.SegmentArrays]) -> dict[str, float]:
    """Count the half-hours of G14's nearest bin and describe those that differ most.

    The nearest bin holds the half-hours from 0 up to 6 hours before the visibility limit, as
    `e00733_gallery.binned_difference` bins them. The counts are over the validation bank, the
    bank that G14 reads.

    Args:
        segments: The validation bank's world segments.

    Returns:
        `n_rows`, `n_big` (absolute stored-minus-re-run P(X<0) above `G14_BIG_DIFFERENCE`),
        `n_big_true_reverse`, `n_big_at_limit`, `n_big_segments`, `n_segments`, `big_stored_mean`,
        `big_rerun_mean`, `hours_to_first_target`, and `hours_to_first_scored`.

    Raises:
        Gallery2Error: When the bank holds no re-run in the nearest bin.
    """
    stored, rerun, truth, hours, seg_id = [], [], [], [], []
    for i, seg in enumerate(segments):
        if seg.u_rerun is None or seg.u_rerun_ok is None:
            continue
        for d in np.flatnonzero(seg.present & seg.u_rerun_ok.any(axis=(1, 2))):
            for r in (0, 1):
                day = int(d) - 2 + r
                if day < 0:
                    continue
                stamp = STEPS * (r - 2) + np.arange(STEPS) + 1
                after = (stamp + core.NEWEST_VISIBLE_BEFORE_CONTRACT - 24) / 2.0
                pick = seg.u_rerun_ok[d, r] & (after <= 0.0) & (after > -G14_BIN_HOURS)
                sl = slice(day * STEPS, (day + 1) * STEPS)
                stored.append(seg.u_raw[sl, core.P_NEG][pick])
                rerun.append(seg.u_rerun[d, r, :, core.P_NEG][pick])
                truth.append(seg.x_neg[sl][pick])
                hours.append(after[pick])
                seg_id.append(np.full(int(pick.sum()), i))
    if not stored or sum(len(a) for a in stored) == 0:
        msg = "the validation bank holds no re-run half-hour in the nearest bin"
        raise Gallery2Error(msg)
    st, rr, tr, hr, sg = (np.concatenate(a) for a in (stored, rerun, truth, hours, seg_id))
    big = np.abs(st - rr) > G14_BIG_DIFFERENCE
    first = sy.target_slots(origin_day=np.array([0]), lead_index=np.array([0]))[0, 0]
    scored = sy.target_slots(origin_day=np.array([0]), lead_index=np.array([sy.SCORED_K0[0]]))[0, 0]
    limit_slot = 24 - core.NEWEST_VISIBLE_BEFORE_CONTRACT - 1
    return {
        "n_rows": float(len(st)), "n_big": float(big.sum()),
        "n_big_true_reverse": float((tr[big] > 0.5).sum()),
        "n_big_at_limit": float((hr[big] == 0.0).sum()),
        "n_big_segments": float(len(set(sg[big].tolist()))),
        "n_segments": float(len(set(sg.tolist()))),
        "big_stored_mean": float(st[big].mean()) if big.any() else float("nan"),
        "big_rerun_mean": float(rr[big].mean()) if big.any() else float("nan"),
        "hours_to_first_target": float(first - limit_slot) / 2.0,
        "hours_to_first_scored": float(scored - limit_slot) / 2.0,
    }  # fmt: skip


def g14_explanation(*, counts: Mapping[str, float]) -> str:
    """Return the three sentences that explain G14's nearest bin, with the counts.

    Args:
        counts: The output of `g14_counts`.

    Returns:
        The text.
    """
    c = {k: int(v) for k, v in counts.items() if k.startswith("n_")}
    share = 100.0 * counts["n_big"] / counts["n_rows"]
    return (
        f"In the 6 hours before the visibility limit the synthetic validation bank holds "
        f"{c['n_rows']} half-hour rows, and {c['n_big']} of them ({share:.1f}%) have an absolute "
        f"stored-minus-re-run P(X<0) above 0.5.\n\n"
        f"These {c['n_big']} rows come from {c['n_big_segments']} of {c['n_segments']} segments, "
        f"{c['n_big_at_limit']} are the last visible half-hour itself, and "
        f"{c['n_big_true_reverse']} are true reverse-flow half-hours on which the stored output "
        f"averages "
        f"{counts['big_stored_mean']:.2f} and the re-run output {counts['big_rerun_mean']:.2f}, "
        f"which fits a re-run that cannot see the half-hours after the limit.\n\n"
        f"All {c['n_rows']} rows are history tokens that YU, FU, and FS read through the re-run "
        f"(FU-S reads the stored table, and Y0, F0r, and YT read no untangler output), and none is "
        f"a scored target row, because the nearest target row lies "
        f"{counts['hours_to_first_target']:g} hours after the limit and the nearest scored one "
        f"{counts['hours_to_first_scored']:g} hours after it.\n"
    )


# --------------------------------------------------------------------------- the stage


def save_chart(*, chart: alt.TopLevelMixin, path: Path, png_dir: Path | None) -> None:
    """Write the chart as SVG and, when `png_dir` is given, a PNG copy of the same SVG."""
    figures729.save_svg(chart=chart, path=path)
    if png_dir is not None:
        import vl_convert

        png_dir.mkdir(parents=True, exist_ok=True)
        png = vl_convert.svg_to_png(path.read_text(), scale=1.5)
        (png_dir / f"{path.stem}.png").write_bytes(png)


def load_real_examples(*, root: Path) -> list[HistoryExample]:
    """Read the real files and return the examples (reads every file; writes nothing).

    Args:
        root: This experiment's data folder.

    Returns:
        The examples H1 to H7.

    Raises:
        Gallery2Error: When a window or an origin reaches a test month.
    """
    import e00729_core as core729  # ty: ignore[unresolved-import]
    import e00729_stages as stages729  # ty: ignore[unresolved-import]

    cal = stages729.build_context().table.calendar
    choice = ch.read_choice(path=paths.channels_json(root=root))
    table = ch.read_table(path=paths.table_path(root=root))
    n_grid = world.read_weather_cache(root=root).n_grid
    mw_per_point = core729.CAPACITY_MVA / core729.POINTS
    reading = dense_on_grid(dates=cal.dates, values=cal.reading, n_grid=n_grid)
    csi = dense_on_grid(dates=cal.dates, values=cal.csi, n_grid=n_grid)
    clear = dense_on_grid(dates=cal.dates, values=cal.clear, n_grid=n_grid)
    first_fold_day = int(np.datetime64(f"{core.ALL_FOLDS[0]}-01", "D").astype(np.int64))
    last_day = first_fold_day - ch.GRID_START_EPOCH_DAY
    origins = candidate_origin_days(last_day=last_day)
    if len(origins) == 0 or origins.max() >= last_day:
        msg = "the validation origins are empty or reach the first fold month"
        raise Gallery2Error(msg)
    grid = history_grid_index(origin_day=origins)[:, shown_rows(), :].reshape(len(origins), -1)
    if grid.max() >= last_day * STEPS:
        msg = "a history window reaches the first fold month"
        raise Gallery2Error(msg)
    std = float(world.read_meta(root=root)["constants"]["std"])
    reading_ok = np.isfinite(reading)
    fu, fu_ok = receive_channels(
        arm="FU", choice=choice, table=table, origin_day=origins, reading_ok=reading_ok,
        n_grid=n_grid, target_std=std,
    )  # fmt: skip
    fus, fus_ok = receive_channels(
        arm="FU-S", choice=choice, table=table, origin_day=origins, reading_ok=reading_ok,
        n_grid=n_grid, target_std=std,
    )  # fmt: skip
    every = replace(choice, usable=core.CHANNEL_NAMES)
    outputs = ch.load_outputs(choice=every, root_732=paths.root_732(), root_730=paths.root_730())
    dense, flag = ch.build_table(outputs=outputs).dense(n_grid=n_grid)
    pv = np.where(flag, dense[:, core.SOLAR].astype(np.float64) * mw_per_point, np.nan)
    n = len(origins)
    return build_history_examples(
        origin_day=origins, target_mw=reading[grid] * mw_per_point,
        fu=fu.reshape(n, SHOWN_STEPS, -1), fu_ok=fu_ok.reshape(n, SHOWN_STEPS),
        fus=fus.reshape(n, SHOWN_STEPS, -1), fus_ok=fus_ok.reshape(n, SHOWN_STEPS),
        stored_solar_mw=pv[grid], csi=csi[grid], clear=clear[grid], usable=choice.usable,
        solar_mw_per_unit=std * core729.CAPACITY_MVA,
    )  # fmt: skip


def load_synthetic(*, root: Path) -> tuple[list[sy.SegmentArrays], list[Any], dict[str, float]]:
    """Read the validation world, the bank behind it, and the reading's constants."""
    import e00729_banks as b729  # ty: ignore[unresolved-import]
    import e00730_banks as b730  # ty: ignore[unresolved-import]

    arrays, _extra = world.read_world(root=root, kind="validation")
    bank = b729.load_bank(path=b730.bank_path(root=paths.root_730(), kind="validation", seed=None))
    constants = world.read_meta(root=root)["constants"]
    return arrays, list(bank.segments), constants


def write_check(*, path: Path, payload: Mapping[str, Any]) -> Path:
    """Write `panel_check2.json` atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=1, default=float))
    tmp.rename(path)
    return path


def stage_gallery2(args: Any, root: Path) -> dict[str, Any]:
    """Draw gallery 2 into `PRIVATE/gallery2/` and write `panel_check2.json`.

    The stage reads the real files and writes nothing outside `PRIVATE/gallery2/` and the PNG
    folder that `--png-dir` names.

    Args:
        args: The parsed arguments (`--png-dir`).
        root: This experiment's data folder.

    Returns:
        The content of `panel_check2.json`.
    """
    out = root / "PRIVATE" / "gallery2"
    png = getattr(args, "png_dir", None)
    png_dir = Path(png) if png else None
    results: list[gal.PanelResult] = []
    for ex in load_real_examples(root=root):
        for with_placebo, stem in ((False, f"history_{ex.name}"), (True, f"placebo_{ex.name}")):
            save_chart(
                chart=history_chart(example=ex, with_placebo=with_placebo),
                path=out / f"{stem}.svg", png_dir=png_dir,
            )  # fmt: skip
        results.append(check_history_example(example=ex))
    arrays, bank_segments, constants = load_synthetic(root=root)
    matched = [
        match_bank_segment(arrays=a, bank_segments=bank_segments, constants=constants)
        for a in arrays
    ]  # fmt: skip
    sources = [str(bank_segments[m].source) if m is not None else "unmatched" for m in matched]
    i, day = choose_synthetic_window(arrays=arrays, sources=sources)
    match = matched[i]
    if match is None:
        msg = "the chosen world segment has no bank segment"
        raise Gallery2Error(msg)
    window = synthetic_window(
        arrays=arrays[i], segment=bank_segments[match], first_day=day,
        mw_per_point=_mw_per_point(),
    )  # fmt: skip
    for arm in SYNTHETIC_ARMS:
        save_chart(
            chart=synthetic_chart(arm=arm, window=window), path=out / f"synthetic_{arm}.svg",
            png_dir=png_dir,
        )  # fmt: skip
        results.append(check_synthetic_arm(arm=arm, window=window))
    counts = g14_counts(segments=arrays)
    (out / "g14_explained.md").write_text(g14_explanation(counts=counts))
    results.append(
        gal.PanelResult(
            panel="G14_explained", passed=counts["n_rows"] > 0 and counts["n_big"] >= 0,
            items={"nearest_bin_has_rows": counts["n_rows"] > 0}, data=counts,
        )
    )  # fmt: skip
    payload = {
        "banks_digest": world.read_meta(root=root)["banks_digest"],
        "panels": {r.panel: bool(r.passed) for r in results},
        "items": {r.panel: dict(r.items) for r in results},
        "g14_counts": counts,
    }
    write_check(path=out / "panel_check2.json", payload=payload)
    for r in results:
        print(f"gallery2 {r.panel}: {'pass' if r.passed else 'FAIL'} {dict(r.items)}")
    return payload


def _mw_per_point() -> float:
    import e00729_core as core729  # ty: ignore[unresolved-import]

    return core729.CAPACITY_MVA / core729.POINTS
