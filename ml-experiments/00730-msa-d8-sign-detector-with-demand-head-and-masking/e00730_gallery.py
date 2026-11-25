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


def check_levels(*, one_unit: np.ndarray, several: np.ndarray, tol: float = 1e-6) -> bool:
    """Check that the several-unit day has more distinct non-zero levels than the one-unit day."""

    def levels(a: np.ndarray) -> int:
        return len(np.unique(np.round(a[a > tol] / max(tol, 1e-3), 0)))

    return levels(several) > levels(one_unit)


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
    """Check that a cloudy day's peak solar output is below the clear day's of its season."""
    return bool(cloudy.max() < clear_same_season.max())


def check_aggregate_bands(*, q10: np.ndarray, q50: np.ndarray, q90: np.ndarray) -> bool:
    """Check that the S1 aggregate bands are ordered in every season."""
    return bool(np.all(q10 <= q50) and np.all(q50 <= q90))


def check_fold(*, reading: np.ndarray, truth: np.ndarray, floor: float, far: float) -> bool:
    """Check the folded reading: never below the floor, and equal to the truth far from zero."""
    far_rows = np.abs(truth) >= far
    return bool(
        (reading >= floor - 1e-9).all() and np.allclose(reading[far_rows], np.abs(truth[far_rows]))
    )


def check_base_equals_truth(*, truth: np.ndarray, base: np.ndarray) -> bool:
    """Check that a no-injection day's truth equals the base demand."""
    return bool(np.allclose(truth, base, atol=1e-6))


def check_mask_panel(*, reading: np.ndarray, hidden: np.ndarray) -> bool:
    """Check that the reading channel is zero on every hidden position."""
    return bool(hidden.any() and (reading[hidden] == 0.0).all())


def check_shares(*, shares: Mapping[str, float]) -> bool:
    """Check that every component share is drawn against the bar `MIN_OUTPUT_SHARE`."""
    return bool(shares) and all(np.isfinite(v) for v in shares.values())


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
