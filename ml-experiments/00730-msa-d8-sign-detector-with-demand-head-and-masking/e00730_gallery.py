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


def check_ramp(*, engine: np.ndarray) -> bool:
    """Check that the output rises to its running level over 2 half-hours (no one-step jump)."""
    up = np.flatnonzero((engine[:-1] <= 0.0) & (engine[1:] > 0.0))
    if up.size == 0:
        return False
    first = int(up[0]) + 1
    run = engine[first : first + 6]
    return bool(run.size >= 3 and engine[first] < 0.99 * run.max())


def check_blocks(*, engine: np.ndarray, before: slice, block: slice) -> bool:
    """Check that the output is above zero on `block` and zero on `before`."""
    return bool((engine[block] > 0.0).any() and (engine[before] <= 0.0).all())


def check_part_load(*, engine: np.ndarray, unit_sizes: Sequence[float]) -> bool:
    """Check that the output stays below the sum of the unit sizes and above zero somewhere."""
    return bool((engine > 0.0).any() and engine.max() < float(sum(unit_sizes)))


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


def check_trip(*, engine: np.ndarray) -> bool:
    """Check that the output drops to zero inside a running block (between two positive runs)."""
    pos = engine > 0.0
    idx = np.flatnonzero(pos)
    if idx.size < 2:
        return False
    return bool((~pos[idx[0] : idx[-1] + 1]).any())


def check_spell(*, daily_max: np.ndarray, days: int = 2) -> bool:
    """Check that the output stays above zero on at least `days` consecutive days."""
    run = 0
    for v in daily_max > 0.0:
        run = run + 1 if v else 0
        if run >= days:
            return True
    return False


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
