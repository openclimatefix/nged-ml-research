# ruff: noqa: E501
"""Experiment 732: the rules that pick the 25 time series windows W1 to W25, and the window figure.

Each rule picks one window (one scored day) of a scored bank from a table of per-window numbers, with
no randomness, so that a rerun draws the same windows. A window whose solar output comes from source S1
is drawn to the private folder only; the published figure draws the same rule over source S2 and
engine-only windows. No window carries a calendar date, and a window is labelled by hours from the
start of its day.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import e00732_paths  # noqa: F401

# isort: split
import altair as alt
import e00730_series_figures as series730
import e00732_core as core
import e00732_gallery as gallery
import numpy as np

NEAR_ZERO_POINTS: Final = 0.3 / 3.0 * 100.0
"""0.3 MVA in points (the capacity of D8's reading scale is 3 MVA per 100 points)."""
NEAR_ZERO_HALF_HOURS: Final = 4
BIG_NOISE_UNITS: Final = 4.0
NOT_IMPLEMENTED: Final = ()
"""Rules with no picker (none: every window W1 to W25 has a rule; a real window is None without real predictions)."""


@dataclass
class WindowTable:
    """Per-window numbers of the test bank (arrays of length `n`; a transfer attribute is False on a window without a transfer).

    Attributes:
        world: The world type of each window's segment.
        engine_run: The window has engine output above zero on its scored day.
        solar_out: The window has solar output above zero on its scored day.
        source: The window's solar source ("S1", "S2", or "none").
        depth: The deepest reverse flow of the scored day in points (positive for reverse flow).
        near_zero: Half-hours of the day with the reading within 0.3 MVA of zero.
        evening: The engine run starts after 14:00 UTC.
        mean_p: U*'s mean P(X<0) on the day's decidable half-hours.
        max_p: U*'s largest P(X<0) of the day.
        gain_vs_s: S's Brier score minus U*'s on the day (positive: U* gains).
        solar_error: U*'s mean absolute solar error on the day's sun-up half-hours.
        pv_max: U*'s largest solar output on the day.
        adds: The window holds an edge of a transfer that adds load to D8.
        removes: The window holds an edge of a transfer that removes load.
        in_solar: The transfer edge sits inside a solar spell.
        other_bsp: The transfer's partner sits under another BSP.
        solar_shape: The transfer has a solar shape.
        mirrored: The transfer's partner is a read neighbour.
        step_gain_vs_un: U-N's step Brier score minus U*'s on the day.
        missed_noise_units: Size in noise units of the largest edge that U* misses (0 where none).
        real: The window is a real day.
    """

    world: np.ndarray
    engine_run: np.ndarray
    solar_out: np.ndarray
    source: np.ndarray
    depth: np.ndarray
    near_zero: np.ndarray
    evening: np.ndarray
    mean_p: np.ndarray
    max_p: np.ndarray
    gain_vs_s: np.ndarray
    solar_error: np.ndarray
    pv_max: np.ndarray
    adds: np.ndarray
    removes: np.ndarray
    in_solar: np.ndarray
    other_bsp: np.ndarray
    solar_shape: np.ndarray
    mirrored: np.ndarray
    step_gain_vs_un: np.ndarray
    missed_noise_units: np.ndarray
    real: np.ndarray


def plan_windows(*, t: WindowTable, published: bool = True) -> dict[str, int | None]:
    """Return the window each rule W1 to W23 picks (None where no window fits the rule).

    W9 to W11 (injection pairs) and W24 and W25 (real windows) come from other tables and are None here.

    Args:
        t: The per-window table.
        published: Restrict the rules to source S2 and engine-only windows, as in the published figures.

    Returns:
        The index into the table of each rule.
    """
    ok = (t.source != "S1") if published else np.ones(len(t.world), dtype=bool)
    syn = ~t.real & ok
    rev = syn & (t.depth > 0)
    med, ext = series730.median_depth_index, series730.extreme_index
    plan: dict[str, int | None] = {
        "W1": med(mask=rev & (t.world == "engine") & t.engine_run & t.evening, depth=t.depth),
        "W2": med(mask=rev & (t.world == "engine") & t.engine_run & ~t.evening, depth=t.depth),
        "W3": med(mask=rev & (t.world == "solar") & t.solar_out, depth=t.depth),
        "W4": med(mask=rev & (t.world == "both"), depth=t.depth),
        "W5": ext(mask=syn & (t.near_zero >= NEAR_ZERO_HALF_HOURS), key=t.near_zero, largest=True),
        "W6": ext(mask=syn & (t.depth <= 0), key=t.mean_p, largest=True),
        "W7": ext(mask=rev | syn, key=t.gain_vs_s, largest=True),
        "W8": ext(mask=rev | syn, key=t.gain_vs_s, largest=False),
        "W12": ext(mask=syn & (t.world == "engine"), key=t.pv_max, largest=True),
        "W13": ext(mask=syn & (t.source == "S2") & t.solar_out, key=t.solar_error, largest=True),
    }
    for name, mask in (
        ("W14", t.adds), ("W16", t.removes), ("W18", t.in_solar), ("W19", t.other_bsp),
        ("W20", t.solar_shape & t.mirrored), ("W21", t.solar_shape & ~t.mirrored),
    ):  # fmt: skip
        plan[name] = ext(mask=syn & mask, key=np.arange(len(t.world), dtype=float), largest=False)
    plan["W15"] = _second(mask=syn & t.adds)
    plan["W17"] = _second(mask=syn & t.removes)
    plan["W22"] = ext(mask=syn & t.mirrored, key=t.step_gain_vs_un, largest=True)
    plan["W23"] = ext(
        mask=syn & (t.missed_noise_units >= BIG_NOISE_UNITS), key=t.missed_noise_units, largest=True
    )
    for name in ("W9", "W10", "W11", "W24", "W25"):
        plan.setdefault(name, None)
    return {k: plan[k] for k in core.SERIES_WINDOWS}


def _second(*, mask: np.ndarray) -> int | None:
    """Return the second window of a mask (the first is the rule above it)."""
    idx = np.flatnonzero(mask)
    return int(idx[1]) if idx.size > 1 else None


def window_figure(*, window: Mapping[str, np.ndarray], title: str) -> alt.VConcatChart:
    """Return one window as three rows: the truth and the reading, the solar target and output, and the step event and probability.

    Args:
        window: Arrays over the window's half-hours: `truth`, `reading`, `solar`, `pv`, `step_event`,
            `step_p`, optionally `hidden` (bool) and `p` (P(X<0)).
        title: The rule's name and what it picked (no date).

    Returns:
        The stacked chart, built from `gallery.lines_chart` so that no calendar date reaches the axis.
    """
    n_days = max(len(window["truth"]) // 48, 1)
    rows = [
        gallery.lines_chart(series={"signed truth": window["truth"], "reading": window["reading"]}, title=title, hidden=window.get("hidden"), n_days=n_days),
        gallery.lines_chart(series={"solar target": window["solar"], "solar output": window["pv"]}, title="Solar output", height=90, n_days=n_days),
        gallery.lines_chart(series={"step event": window["step_event"], "step probability": window["step_p"]}, title="Step output", height=90, n_days=n_days, y_title="Probability or event"),
    ]  # fmt: skip
    if "p" in window:
        rows.insert(
            1,
            gallery.lines_chart(
                series={"P(X<0)": window["p"]},
                title="Probability of reverse flow",
                height=90,
                n_days=n_days,
                y_title="P(X<0)",
            ),
        )
    return gallery.stack(charts=rows, title=title)
