"""Experiment 730: the rules that pick the time series windows, and the two-row window figure.

Each of the 16 rules S1 to S16 picks one (segment, day) of a scored bank from a table of per-window
numbers, with no randomness, so that a rerun draws the same windows. A window whose solar output
comes from source S1 is drawn to the private folder only; the published figure draws the same rule
over source S2 and engine-only windows. No window carries a calendar date.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final

import e00730_paths  # noqa: F401

# isort: split
import altair as alt
import e00730_figures as figures
import numpy as np

NEAR_ZERO_MVA: Final = 0.3
NEAR_ZERO_HALF_HOURS: Final = 4
RULES: Final = (
    "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12", "S13", "S14", "S15",
    "S16",
)  # fmt: skip
"""The 16 rules, each drawing one figure; the plan asks for at least 15."""


@dataclass
class WindowTable:
    """Per-window numbers of one bank, all arrays of length `n`.

    Attributes:
        world: The world type of each window's segment ("none", "engine", "solar", "both").
        solar_run: Whether the window has engine output above zero on its scored day.
        solar_out: Whether the window has solar output above zero on its scored day.
        source: The window's solar source ("S1", "S2", or "none").
        depth: The deepest reverse flow of the scored day in points (positive for reverse flow).
        near_zero: Half-hours of the day with the reading within 0.3 MVA of zero.
        evening: The engine run of the day starts after 14:00 UTC.
        sd_max_p: SD*'s largest P(X<0) of the day.
        sd_mean_p: SD*'s mean P(X<0) on the day's decidable half-hours.
        gain: S's Brier score minus SD*'s on the day (positive is a gain of SD*).
        real: The window is a real pre-engine day.
    """

    world: np.ndarray
    solar_run: np.ndarray
    solar_out: np.ndarray
    source: np.ndarray
    depth: np.ndarray
    near_zero: np.ndarray
    evening: np.ndarray
    sd_max_p: np.ndarray
    sd_mean_p: np.ndarray
    gain: np.ndarray
    real: np.ndarray


def median_depth_index(*, mask: np.ndarray, depth: np.ndarray) -> int | None:
    """Return the window of `mask` whose reverse-flow depth is the median of the masked windows."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return None
    order = idx[np.argsort(depth[idx], kind="stable")]
    return int(order[(len(order) - 1) // 2])


def extreme_index(*, mask: np.ndarray, key: np.ndarray, largest: bool) -> int | None:
    """Return the masked window with the largest or the smallest `key` (ties go to the first)."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return None
    k = key[idx]
    return int(idx[int(np.argmax(k) if largest else np.argmin(k))])


def median_index(*, mask: np.ndarray, key: np.ndarray) -> int | None:
    """Return the masked window whose `key` is the median of the masked windows."""
    return median_depth_index(mask=mask, depth=key)


def plan_windows(*, t: WindowTable, published: bool = True) -> dict[str, int | None]:
    """Return the window each rule picks, or None where no window fits the rule.

    Args:
        t: The per-window table of the bank.
        published: Restrict the synthetic rules to source S2 and engine-only windows, as in the
            published figures. The private run passes False and may pick an S1 window.

    Returns:
        The index into the table of each rule S1 to S16.
    """
    ok = (t.source != "S1") if published else np.ones(len(t.world), dtype=bool)
    synthetic = ~t.real
    rev = synthetic & ok & (t.depth > 0)
    plan: dict[str, int | None] = {
        "S1": median_depth_index(
            mask=rev & (t.world == "engine") & t.solar_run & t.evening, depth=t.depth
        ),
        "S2": median_depth_index(
            mask=rev & (t.world == "engine") & t.solar_run & ~t.evening, depth=t.depth
        ),
        "S3": median_depth_index(mask=rev & (t.world == "solar") & t.solar_out, depth=t.depth),
        "S4": median_depth_index(mask=rev & (t.world == "both"), depth=t.depth),
        "S5": extreme_index(
            mask=synthetic & ok & (t.near_zero >= NEAR_ZERO_HALF_HOURS),
            key=t.near_zero,
            largest=True,
        ),
        "S6": median_index(mask=synthetic & ok & (t.world == "none"), key=t.sd_max_p),
        "S7": extreme_index(mask=synthetic & ok & (t.depth <= 0), key=t.sd_mean_p, largest=True),
        "S8": extreme_index(mask=rev, key=t.gain, largest=True),
        "S9": extreme_index(mask=rev, key=t.gain, largest=False),
        "S10": extreme_index(mask=synthetic & ok, key=t.depth, largest=True),
        "S11": extreme_index(
            mask=synthetic & ok & (t.world != "none") & t.solar_run, key=t.depth, largest=True
        ),
        "S12": median_depth_index(mask=rev & t.solar_run, depth=t.depth),
        "S13": median_depth_index(mask=rev & t.solar_out, depth=t.depth),
        "S14": median_index(mask=t.real, key=t.sd_max_p),
        "S15": extreme_index(mask=t.real, key=t.sd_max_p, largest=True),
        "S16": extreme_index(mask=rev & t.solar_out, key=t.depth, largest=True),
    }
    return plan


def window_figure(*, window: Mapping[str, np.ndarray], title: str) -> alt.VConcatChart:
    """Return one window: signed truth, reading, and demand in the top row, probabilities below.

    Args:
        window: 96-half-hour arrays: `truth`, `reading`, `demand`, `demand_mean`, `demand_low`,
            `demand_high`, `hidden` (bool), and `p_<arm>` for each arm drawn in the second row.
        title: The rule's name and what it picked (no date).

    Returns:
        The two-row chart.
    """
    top = figures.window_chart(
        series={
            "signed truth": window["truth"],
            "reading": window["reading"],
            "clean demand": window["demand"],
            "demand output (mean)": window["demand_mean"],
        },
        title=title,
        hidden=window.get("hidden"),
    )
    probs = {k[2:]: v for k, v in window.items() if k.startswith("p_")}
    bottom = figures.window_chart(
        series=probs, title="Probability of reverse flow, P(X<0)", y_title="P(X<0)", height=90
    )
    return figures.two_row_chart(top=top, bottom=bottom)


Picker = Callable[[WindowTable], dict[str, int | None]]
