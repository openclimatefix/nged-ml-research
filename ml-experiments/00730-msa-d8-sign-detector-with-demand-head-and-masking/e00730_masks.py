"""Experiment 730: the masking styles, the future cut of the causal variant, and the fixed masks.

A window has 96 half-hours, from 12:00 UTC on the day before the scored day to 12:00 UTC on the day
after, so the scored day is positions 24 to 71 and every window starts on an even half-hour. A
hidden span sets the reading and its presence mask to zero (the rows `READING_ROWS`), as experiment
729's `mask_span` does. The causal cut also builds the half-hour weather channels from hourly source
values with every hour after the cut's hour hidden before the interpolation, because a half-hour
channel is built from hourly values and the second half-hour of an hour would otherwise read the
next hour. No function here needs torch.
"""

from collections.abc import Sequence
from typing import Final

import e00730_paths  # noqa: F401

# isort: split
import e00729_core as core729
import e00729_data as data729
import e00730_core as core
import numpy as np

WINDOW: Final = core729.WINDOW_STEPS
SCORED_START: Final = core729.MIDDLE_START
SCORED_STEPS: Final = core729.WINDOW_STEPS_24
READING_ROWS: Final = (0, 1)
"""Rows 0 and 1 of every channel stack hold the reading and its presence mask."""
OBSERVED_ROWS: Final = (
    "r", "r_mask", "csi", "cams_mask", "d_bsp", "bsp_mask", "t2m", "wind10", "precip",
)  # fmt: skip
"""The channels that a cut hides after the cut: the reading and its mask, the CAMS clear-sky index
and its mask, the BSP steps and their mask, and the ERA5 temperature, wind speed, and
precipitation."""
KNOWN_IN_ADVANCE_ROWS: Final = (
    "clear", "cosz", "local_sin", "local_cos", "dow_sin", "dow_cos", "holiday",
)  # fmt: skip
"""The channels that stay visible after a cut: they are known in advance."""
HOURLY_ROWS: Final = ("csi", "t2m", "wind10", "precip")
"""The channels built from hourly source values, which the cut rebuilds from visible hours only."""
STYLE_INDEX: Final = {"0": 0, "K": 1, "W": 2, "R": 3}


def hours_to_steps(*, hours: float) -> int:
    """Return a span of hours as a whole number of half-hours."""
    return round(hours * 2.0)


def hidden_k(*, rng: np.random.Generator, utc_offset: float) -> np.ndarray:
    """Return the hidden positions of style K: the morning block, the evening block, or both.

    The block starts are in local time (`MORNING_BLOCK_START_H`, `EVENING_BLOCK_START_H`) on the
    scored day. Each of the three choices (morning, evening, both) has probability 1/3.

    Args:
        rng: The generator.
        utc_offset: Local time minus UTC in hours on the scored day.

    Returns:
        A boolean array of the window's 96 positions.
    """
    hidden = np.zeros(WINDOW, dtype=bool)
    choice = int(rng.integers(0, 3))
    blocks = (
        (core.MORNING_BLOCK_START_H, core.MORNING_BLOCK_HOURS),
        (core.EVENING_BLOCK_START_H, core.EVENING_BLOCK_HOURS),
    )
    for index, (start_range, hours_range) in enumerate(blocks):
        if choice not in (index, 2):
            continue
        start_local = float(rng.uniform(*start_range))
        hours = float(rng.uniform(*hours_range))
        first = SCORED_START + hours_to_steps(hours=start_local - utc_offset)
        last = first + hours_to_steps(hours=hours)
        hidden[max(first, 0) : min(last, WINDOW)] = True
    return hidden


def hidden_w() -> np.ndarray:
    """Return the hidden positions of style W: all 48 half-hours of the scored day."""
    hidden = np.zeros(WINDOW, dtype=bool)
    hidden[SCORED_START : SCORED_START + SCORED_STEPS] = True
    return hidden


def hidden_r(*, rng: np.random.Generator) -> np.ndarray:
    """Return the hidden positions of style R: one span of 1 to 24 hours anywhere in the window."""
    hidden = np.zeros(WINDOW, dtype=bool)
    hours = int(rng.integers(core729.MASK_HOURS[0], core729.MASK_HOURS[1] + 1))
    length = min(hours * 2, WINDOW)
    first = int(rng.integers(0, WINDOW - length + 1))
    hidden[first : first + length] = True
    return hidden


def hidden_positions(*, style: str, rng: np.random.Generator, utc_offset: float) -> np.ndarray:
    """Return the hidden positions of a masking style ("0" hides nothing)."""
    if style == "0":
        return np.zeros(WINDOW, dtype=bool)
    if style == "K":
        return hidden_k(rng=rng, utc_offset=utc_offset)
    if style == "W":
        return hidden_w()
    if style == "R":
        return hidden_r(rng=rng)
    msg = f"unknown masking style {style!r}"
    raise ValueError(msg)


def apply_hidden(*, window: np.ndarray, hidden: np.ndarray) -> np.ndarray:
    """Set the reading and its presence mask to zero at the hidden positions, in place.

    Args:
        window: One window `(C, 96)`; rows 0 and 1 hold the reading and its mask.
        hidden: The hidden positions, `(96,)`.

    Returns:
        The same array, modified.
    """
    window[0, hidden] = 0.0
    window[1, hidden] = 0.0
    return window


def fixed_rng(*, style: str, index: int) -> np.random.Generator:
    """Return the generator of window `index` under `style` for the fixed masks (seed 730).

    Every arm that is evaluated on the fixed masks sees the same masks, because the generator
    depends on the style and the window's index alone.
    """
    return np.random.default_rng(
        np.random.SeedSequence([core.MASK_SEED, STYLE_INDEX[style], index])
    )


def fixed_hidden(*, style: str, index: int, utc_offset: float) -> np.ndarray:
    """Return the fixed mask of window `index` under `style`."""
    return hidden_positions(
        style=style, rng=fixed_rng(style=style, index=index), utc_offset=utc_offset
    )


WINDOW_ID_STRIDE: Final = 100_000
"""Base days per segment that a window id leaves room for."""


def window_id(*, segment: int, day: int) -> int:
    """Return the id of the window of base day `day` in segment `segment`.

    The id depends on the window alone, so a window gets the same fixed mask whatever the order of
    the picks that a bank lists.
    """
    if not 0 <= day < WINDOW_ID_STRIDE:
        msg = f"base day {day} lies outside 0..{WINDOW_ID_STRIDE - 1}"
        raise ValueError(msg)
    return segment * WINDOW_ID_STRIDE + day


def fixed_hidden_for_picks(
    *,
    style: str,
    picks: Sequence[tuple[int, int]],
    offsets: Sequence[Sequence[float] | np.ndarray],
    segment_ids: Sequence[int] | None = None,
) -> np.ndarray:
    """Return the fixed masks `(len(picks), 96)` of the picks (segment, day), by window id.

    Args:
        style: The masking style.
        picks: The (segment, base day) of each window.
        offsets: For each segment, the UTC offset in hours of each of its base days.
        segment_ids: The id of each segment in the window id (its position in the bank's list),
            or None to use the segment's position in `offsets`. A prepared list that drops a
            segment keeps the ids of the others.

    Returns:
        One row per pick, in the order of `picks`.
    """
    return np.stack(
        [
            fixed_hidden(
                style=style,
                index=window_id(segment=s if segment_ids is None else segment_ids[s], day=d),
                utc_offset=float(offsets[s][d]),
            )
            for s, d in picks
        ]
    )


def fixed_cuts(*, n_windows: int, n_cuts: int = core.CAUSAL_VALIDATION_CUTS) -> np.ndarray:
    """Return the fixed cuts of the causal validation: `n_cuts` scored half-hours per window.

    Args:
        n_windows: The number of validation windows.
        n_cuts: The number of cuts per window (4).

    Returns:
        An integer array `(n_windows, n_cuts)` of scored half-hours `0..47`, from seed 730.
    """
    rng = np.random.default_rng(np.random.SeedSequence([core.MASK_SEED, 99]))
    return rng.integers(0, SCORED_STEPS, size=(n_windows, n_cuts))


# --------------------------------------------------------------------------- the hourly grid


def hourly_from_half_hour(*, values: np.ndarray) -> np.ndarray:
    """Return the hourly values behind a half-hour channel that `hourly_to_half_hour` built.

    The first half-hour of hour `k` is `0.75 v_k + 0.25 v_{k-1}` and the second is `0.75 v_k + 0.25
    v_{k+1}`, so `v_k = (3 a_k - b_{k-1}) / 2` where both neighbours exist. The inversion is exact
    on an array that starts at a whole hour. Where the previous hour lies outside the array, the
    first half-hour is read as `v_k`.

    Args:
        values: Half-hours (NaN where missing), an even length, starting at a whole hour.

    Returns:
        One value per hour, `len(values) // 2`.
    """
    pairs = values.reshape(-1, 2)
    a, b = pairs[:, 0], pairs[:, 1]
    b_prev = np.insert(b[:-1], 0, np.nan)
    return np.where(np.isfinite(b_prev), (3.0 * a - b_prev) / 2.0, a)


def half_hour_from_hourly(*, hourly: np.ndarray) -> np.ndarray:
    """Return the half-hour channel of an hourly array, as experiment 729 builds it."""
    return data729.hourly_to_half_hour(values=np.repeat(hourly, core.HOURLY_PAIR))


def copy_half_hour_from_hourly(*, hourly: np.ndarray) -> np.ndarray:
    """Return the half-hour channel that copies each hourly value to both half-hours of its hour."""
    return np.repeat(hourly, core.HOURLY_PAIR)


def causal_half_hours(
    *, hourly: np.ndarray, last_visible_hour: int, copy: bool = False
) -> np.ndarray:
    """Return the half-hour channel built from the hourly values up to `last_visible_hour` only.

    Every hourly source value stamped after the start of the cut's hour (an index above
    `last_visible_hour`) is hidden before the interpolation or the copy, and the channel is built
    from the remaining values. A half-hour in a later hour is returned as NaN.

    Args:
        hourly: The hourly values of the window and one earlier hour, `(H,)`.
        last_visible_hour: The index of the cut's hour in `hourly`.
        copy: Copy each hourly value to both half-hours instead of interpolating (ERA5-style).

    Returns:
        The half-hour channel, `(2 H,)`.
    """
    visible = hourly.copy()
    visible[last_visible_hour + 1 :] = np.nan
    build = copy_half_hour_from_hourly if copy else half_hour_from_hourly
    return build(hourly=visible)
