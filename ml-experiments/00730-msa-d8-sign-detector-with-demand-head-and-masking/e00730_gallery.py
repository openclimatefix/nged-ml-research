"""Experiment 730: the synthetic-generator gallery's automatic checks and its `panel_check.json`.

The gallery shows plenty of generator-output plots before any fit. Each panel of
`core.PANEL_REQUIREMENTS` has a numeric check on the arrays behind it, written here as a pure
function; the design lead then looks at every panel against the written list. `panel_check.json`
records the outcome of each panel, and `banks.gallery_passed` reads it before any fit or
pre-training run starts. Single days of a metered farm (S1) go to the private folder only. The
published gallery draws S1 as seasonal aggregates (10%, 50%, and 90% bands).
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import e00730_paths  # noqa: F401

# isort: split
import e00730_banks as banks
import e00730_core as core
import numpy as np

PRIVATE_SUBFOLDER: Final = "gallery"
PANEL_FILE: Final = "panel_check.json"
PUBLIC_FOLDER_NAME: Final = "published"
RAMP_HALF_HOURS: Final = 2


def gallery_private_dir(*, root: Path) -> Path:
    """Return the private folder that holds single S1 days and `panel_check.json`."""
    return root / "PRIVATE" / PRIVATE_SUBFOLDER


RAMP_SHARE: Final = 0.97
"""A half-hour at or above this share of the running level has finished its ramp."""
TRIP_FLANK_TOLERANCE: Final = 0.02
"""A zero run is a trip when the levels before and after it differ by at most this share."""
TRIP_MAX_HALF_HOURS: Final = 4
"""The longest trip, in half-hours (`core729.RANGES["trip_half_hours"]`)."""
PART_LOAD_SHARE: Final = 0.9
"""Part load means the output stays below this share of the sum of the unit sizes."""


def check_ramp(*, engine: np.ndarray) -> bool:
    """Check that a run rises to its level over at least 2 half-hours, as the plotted arrays show.

    A run starts where the output turns positive. Its first half-hour must be under 0.75 of the
    run's level (the highest of the next 8 half-hours) and its second half-hour must still be under
    `RAMP_SHARE` of the level, so that the plotted line shows a staircase and not a step.
    """
    up = np.flatnonzero((engine[:-1] <= 0.0) & (engine[1:] > 0.0)) + 1
    for first in up:
        level = float(engine[first : first + 8].max())
        if (
            first + 2 < len(engine)
            and engine[first] < 0.75 * level
            and engine[first + 1] < RAMP_SHARE * level
        ):
            return True
    return False


def check_blocks(*, engine: np.ndarray, before: slice, block: slice) -> bool:
    """Check that the output is above zero on `block` and zero on `before`."""
    return bool((engine[block] > 0.0).any() and (engine[before] <= 0.0).all())


def check_evening(*, engine: np.ndarray) -> bool:
    """Check that some day of the plotted window has an evening block and no output before it."""
    return any(
        check_blocks(engine=day, before=slice(0, 20), block=slice(34, 46))
        for day in engine.reshape(-1, 48)
    )


def check_morning(*, engine: np.ndarray) -> bool:
    """Check that some day of the plotted window has a morning block and no output before it."""
    return any(
        check_blocks(engine=day, before=slice(0, 8), block=slice(10, 18))
        for day in engine.reshape(-1, 48)
    )


def check_weekend(*, engine: np.ndarray, first_day_is_weekend: bool) -> bool:
    """Check that the window's first day is a weekend day and has engine output."""
    return bool(first_day_is_weekend and engine[:48].max() > 0.0)


def check_part_load(*, engine: np.ndarray, unit_sizes: Sequence[float]) -> bool:
    """Check that the output is above zero and stays under 0.9 of the sum of the unit sizes."""
    return bool((engine > 0.0).any() and engine.max() < PART_LOAD_SHARE * float(sum(unit_sizes)))


def check_trip(*, engine: np.ndarray) -> bool:
    """Check that the output drops to zero for 1 to 4 half-hours inside a block at one level.

    Two blocks of one day with a gap between them do not count, because their levels differ. A
    trip leaves the level before it and the level after it equal, within
    `TRIP_FLANK_TOLERANCE`.
    """
    pos = engine > 0.0
    zero = np.flatnonzero(~pos)
    if zero.size == 0:
        return False
    starts = zero[np.concatenate([[True], np.diff(zero) > 1])]
    stops = zero[np.concatenate([np.diff(zero) > 1, [True]])] + 1
    for a, b in zip(starts, stops, strict=True):
        if b - a > TRIP_MAX_HALF_HOURS or a < 3 or b + 3 > len(engine):
            continue
        before, after = float(engine[a - 3 : a].max()), float(engine[b : b + 3].max())
        if (
            before > 0.0
            and after > 0.0
            and abs(before - after) <= TRIP_FLANK_TOLERANCE * max(before, after)
        ):
            return True
    return False


def check_spell(*, engine: np.ndarray) -> bool:
    """Check that the window shows a spell: whole days above zero, with a day before and after.

    The output must stay above zero through at least 2 consecutive whole days, and the window must
    hold a day that is not whole before the first and after the last, so that the plotted spell
    starts and ends inside the window.
    """
    full = (engine.reshape(-1, 48) > 0.0).all(axis=1)
    n = len(full)
    for i in range(1, n - 2):
        if full[i] and full[i + 1] and not full[i - 1]:
            j = i + 1
            while j + 1 < n and full[j + 1]:
                j += 1
            if j + 1 < n and not full[j + 1]:
                return True
    return False


def run_plateaus(*, engine: np.ndarray, min_step: float = 2.0) -> int:
    """Return the most distinct steady levels inside one contiguous run of output.

    A level is a value that two half-hours in a row share, and two levels are distinct when they
    differ by at least `min_step` points, so that the panel shows the step. A day of one unit has
    one level in each run, because a run holds one unit at one share of its size. A day of
    several units has a run with two or more levels whenever two units run together at different
    shares.
    """
    best = 0
    pos = engine > 0.0
    start = None
    for i in range(len(engine) + 1):
        if i < len(engine) and pos[i]:
            start = i if start is None else start
            continue
        if start is not None:
            run = engine[start:i]
            steady = run[1:][np.abs(np.diff(run)) < 1e-6]
            levels = np.unique(np.round(steady, 2))
            count = 0 if levels.size == 0 else 1 + int((np.diff(levels) >= min_step).sum())
            best = max(best, count)
            start = None
    return best


ONE_UNIT_MAX_MULTI_LEVEL_SHARE: Final = 0.10
"""At most this share of one-unit days may show two levels in a run (overlapping runs, spells)."""


def check_units(*, one_unit_levels: Sequence[int], several_unit_levels: Sequence[int]) -> bool:
    """Check that several-unit days show several levels in a run far more often than one-unit days.

    A one-unit day can show two levels when two runs overlap or a spell meets a run, so the check
    compares shares: the share of one-unit days with two or more levels is at most
    `ONE_UNIT_MAX_MULTI_LEVEL_SHARE` and below the share of several-unit days.

    Args:
        one_unit_levels: `run_plateaus` of every one-unit day with output.
        several_unit_levels: `run_plateaus` of every several-unit day with output.

    Returns:
        Whether the shares of days with two levels satisfy the check.
    """
    if not one_unit_levels or not several_unit_levels:
        return False
    one = float(np.mean(np.asarray(one_unit_levels) >= 2))
    several = float(np.mean(np.asarray(several_unit_levels) >= 2))
    return one <= ONE_UNIT_MAX_MULTI_LEVEL_SHARE and several > one


def check_clear(*, solar: np.ndarray, mid: slice) -> bool:
    """Check that a clear-day panel has solar output above zero through mid-day."""
    return bool((solar[mid] > 0.0).all())


def check_cloudy(*, cloudy: np.ndarray, clear_same_season: np.ndarray) -> bool:
    """Check that a cloudy day's peak output is below the clear day's of its season.

    Both arrays are in the same unit; the gallery divides each by the segment's solar capacity
    first, because the capacity differs from one draw to the next.
    """
    return bool(cloudy.max() < clear_same_season.max())


def check_aggregate_bands(*, q10: np.ndarray, q50: np.ndarray, q90: np.ndarray) -> bool:
    """Check that the S1 aggregate bands are ordered in every season."""
    return bool(np.all(q10 <= q50) and np.all(q50 <= q90))


def check_fold(*, reading: np.ndarray, truth: np.ndarray, floor: float, q_max: float) -> bool:
    """Check that the reading is the signed truth folded with a reactive floor.

    The meter reads `sqrt(x**2 + q**2)`, so the reading is at least the floor and never negative,
    and `reading**2 - truth**2` is `q**2`, which lies between the square of the floor and the square
    of the largest reactive term `q_max`. Far from zero the reading then equals the truth up to
    `q**2 / (2 |x|)`, and near zero it sits at the floor.
    """
    r = reading.astype(float)
    q2 = r**2 - truth.astype(float) ** 2
    slack = 1e-5 * (1.0 + r**2)
    return bool(
        np.all(r >= floor - 1e-6)
        and np.all(q2 >= floor**2 - slack)
        and np.all(q2 <= q_max**2 + slack)
    )


def check_base_equals_truth(*, truth: np.ndarray, base: np.ndarray) -> bool:
    """Check that a no-injection day's truth equals the base demand."""
    return bool(np.allclose(truth, base, atol=1e-6))


def check_mask_panel(*, reading: np.ndarray, hidden: np.ndarray) -> bool:
    """Check that the reading channel is zero on every hidden position."""
    return bool(hidden.any() and (reading[hidden] == 0.0).all())


def check_shares(*, shares: Mapping[str, float], bar: float | None = None) -> bool:
    """Check that every share is finite and, when a bar is given, at least the bar."""
    return bool(shares) and all(
        np.isfinite(v) and (bar is None or v >= bar) for v in shares.values()
    )


def build_panel_check(*, results: Mapping[str, bool], notes: Mapping[str, str]) -> dict:
    """Return the content of `panel_check.json`: one flag per required panel, and the notes.

    A panel the list requires but `results` lacks is recorded as failed, so a missing panel keeps
    every fit closed.
    """
    panels = {name: bool(results.get(name, False)) for name in core.PANEL_REQUIREMENTS}
    return {
        "panels": panels,
        "requirements": dict(core.PANEL_REQUIREMENTS),
        "notes": dict(notes),
        "all_passed": all(panels.values()),
    }


def write_panel_check(*, root: Path, record: Mapping[str, object]) -> Path:
    """Write `PRIVATE/gallery/panel_check.json` and return its path."""
    path = gallery_private_dir(root=root) / PANEL_FILE
    banks.write_json(path=path, payload=record)
    return path


def seasonal_bands(*, values: np.ndarray, seasons: np.ndarray) -> dict[str, np.ndarray]:
    """Return, per season, the 10%, 50%, and 90% bands across days of a `(days, 48)` array.

    The published gallery draws metered-farm (S1) days only through these aggregates, never as a
    single day.
    """
    out = {}
    for season in core.COMPONENT_SEASONS:
        part = values[seasons == season]
        if part.shape[0] == 0:
            continue
        out[season] = np.quantile(part, [0.1, 0.5, 0.9], axis=0)
    return out
