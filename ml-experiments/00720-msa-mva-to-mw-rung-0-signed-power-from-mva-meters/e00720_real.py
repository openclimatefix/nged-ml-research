# ruff: noqa: E501
"""Experiment 720: read-only access to the stored panel, and the fold windows. Nothing is written here.

The loader `load_panel` imports experiment 87's and experiment 506's modules, which do not import
`torch`. The pure functions (`to_day_matrix`, `fold_windows`, and the census and gate functions in the
scripts) need no earlier experiment and are tested without one.
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl

US_PER_HALF_HOUR: Final[int] = 1_800_000_000
US_PER_DAY: Final[int] = 86_400 * 1_000_000
STEPS_PER_DAY: Final[int] = 48
FOLDS: Final[tuple[str, ...]] = (
    "2025-11", "2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06",
)  # fmt: skip
"""The eight scored months of experiment 713."""
RUNG0_FOLDS: Final[tuple[str, ...]] = ("2025-12", "2026-04", "2026-05", "2026-06")
"""The folds of the design: three sunny months and one dark month."""
SIGNED_SIX: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16")
SIGNED_REPORT_ONLY: Final[tuple[str, ...]] = ("D15", "D18")
MVA_ONLY: Final[tuple[str, ...]] = ("D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22")
SHORT_HISTORY: Final[tuple[str, ...]] = ("D15", "D18")
SHORT_HISTORY_CUT_US: Final[int] = int(datetime(2025, 3, 1, tzinfo=UTC).timestamp() * 1_000_000)
CUTOFF_AGE: Final[timedelta] = timedelta(hours=3.5)
WINDOW_DAYS: Final[int] = 730
HOLDOUT_US: Final[int] = int(datetime(2026, 7, 1, tzinfo=UTC).timestamp() * 1_000_000)
PYTHONPATH_NOTE: Final[str] = (
    "Experiment 716's folder (branch msa-f7-pv-part-1) is put on PYTHONPATH by run.sh and nothing is "
    "merged from that branch."
)


@dataclass(frozen=True)
class FoldWindow:
    """The reference (training) span and the scored span of one fold, in microseconds.

    Attributes:
        fold: The month, `2026-04`.
        cutoff_us: The first instant of the scored month.
        train_start_us: `cutoff − 730 days`.
        train_end_us: `cutoff − 3.5 h`: the last valid time that a refit at the cutoff may read.
        score_end_us: The first instant after the scored month, capped at the holdout start.
    """

    fold: str
    cutoff_us: int
    train_start_us: int
    train_end_us: int
    score_end_us: int


def fold_windows(*, fold: str) -> FoldWindow:
    """Return the windows of a fold; no span reaches the holdout."""
    year, month = (int(x) for x in fold.split("-"))
    start = datetime(year, month, 1, tzinfo=UTC)
    end = datetime(year + (month == 12), month % 12 + 1, 1, tzinfo=UTC)
    us = lambda t: int(t.timestamp() * 1_000_000)  # noqa: E731
    return FoldWindow(
        fold=fold,
        cutoff_us=us(start),
        train_start_us=us(start - timedelta(days=WINDOW_DAYS)),
        train_end_us=us(start - CUTOFF_AGE),
        score_end_us=min(us(end), HOLDOUT_US),
    )


def to_day_matrix(*, values: np.ndarray, stamp_us: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return `(n_days, 48)` values and the first instant (microseconds) of each UTC day.

    `stamp_us` stamps the END of each half-hour (the convention of the dense grid), so a half-hour
    belongs to the day that holds its start. The grid must be contiguous and 30 minutes apart.
    """
    start = stamp_us - US_PER_HALF_HOUR
    if not np.all(np.diff(stamp_us) == US_PER_HALF_HOUR):
        msg = "the grid is not contiguous at 30 minutes"
        raise ValueError(msg)
    day0 = int(start[0] // US_PER_DAY)
    first_step = int((start[0] - day0 * US_PER_DAY) // US_PER_HALF_HOUR)
    n_days = (first_step + len(values) + STEPS_PER_DAY - 1) // STEPS_PER_DAY
    out = np.full(n_days * STEPS_PER_DAY, np.nan)
    out[first_step : first_step + len(values)] = values
    days_us = (day0 + np.arange(n_days, dtype=np.int64)) * US_PER_DAY
    return out.reshape(n_days, STEPS_PER_DAY), days_us


def days_in(*, days_us: np.ndarray, lo_us: int, hi_us: int) -> np.ndarray:
    """Return the indices of days that lie wholly in `[lo_us, hi_us)`."""
    return np.flatnonzero((days_us >= lo_us) & (days_us + US_PER_DAY <= hi_us))


@dataclass
class Panel:
    """The stored panel as arrays on the dense grid.

    Attributes:
        sites: The 16 labels in Primary order.
        stamp_us: Valid times (the end of each half-hour), `(G,)`.
        power_pct: Cleaned power in points of capacity, `(16, G)`; NaN where flagged. Signed for the
            MW Primaries, a magnitude for the MVA ones. D15 and D18 are NaN before 2025-03-01.
        u: The scaled CAMS unit shape, `(16, G)`.
        k: The CAMS clear-sky index, `(16, G)`.
        temperature_c: ERA5 2 m temperature, `(16, G)`.
    """

    sites: list[str]
    stamp_us: np.ndarray
    power_pct: np.ndarray
    u: np.ndarray
    k: np.ndarray
    temperature_c: np.ndarray


def apply_short_history_cut(
    *, power_pct: np.ndarray, sites: list[str], stamp_us: np.ndarray
) -> np.ndarray:
    """Return a copy of the power with D15 and D18 set to NaN before 2025-03-01."""
    out = power_pct.copy()
    early = stamp_us < SHORT_HISTORY_CUT_US
    for i, s in enumerate(sites):
        if s in SHORT_HISTORY:
            out[i, early] = np.nan
    return out


def check_grid_lengths(*, n_grid: int, dense_raw_len: int, cams_len: int) -> int:
    """Return the length to use, or raise when the arrays do not share the grid's start.

    Experiment 87's grid (2019-09-01 to 2026-07-31) has 121,248 half-hours, and the CAMS arrays of
    experiment 412 have that length. Experiment 713's `dense_raw.npy` is longer (126,624) because
    experiment 667 moved the grid's end to 2026-11-20, and both grids start on 2019-09-01, so the
    dense array is cut to the first `n_grid` steps. Any other mismatch raises.
    """
    if cams_len != n_grid:
        msg = f"CAMS arrays have {cams_len} steps, the grid {n_grid}"
        raise ValueError(msg)
    if dense_raw_len < n_grid:
        msg = f"dense_raw has {dense_raw_len} steps, fewer than the grid's {n_grid}"
        raise ValueError(msg)
    return n_grid


def check_dense_raw_alignment(*, day_of_week_columns: np.ndarray, stamp_us: np.ndarray) -> int:
    """Return the number of steps checked after asserting that dense_raw's day-of-week one-hot columns match `stamp_us`.

    `day_of_week_columns` is `dense_raw[0, :n, 14:21]` (Monday first), and the weekday is that of the
    stamp in Europe/London local time. A shifted start of `dense_raw`, which a length check cannot see,
    breaks the match on many steps.
    """
    weekday = (
        pl.Series(stamp_us)
        .cast(pl.Datetime("us", "UTC"))
        .dt.convert_time_zone("Europe/London")
        .dt.weekday()
        .to_numpy()
        - 1
    )
    n_bad = int((np.asarray(day_of_week_columns).argmax(axis=1) != weekday).sum())
    if n_bad:
        msg = f"dense_raw's day-of-week columns disagree with the grid stamps on {n_bad} of {len(stamp_us)} steps"
        raise ValueError(msg)
    return len(stamp_us)


def load_panel(*, dense_raw_path: Path) -> Panel:
    """Read the cleaned power, the CAMS arrays, and the temperature on experiment 87's grid.

    The power, the unit shape `u`, and the clear-sky index `k` come from experiment 506's loaders
    (`e00506_data`), which do not import `torch`, and the temperature is column 2 of the first
    `n_grid` steps of experiment 713's dense array. Nothing on or after the holdout is read as power.
    Needs the folders of experiments 87 and 506 (`run.sh` builds `PYTHONPATH`).
    """
    import e00087_common as common  # ty: ignore[unresolved-import]
    import e00506_data as data  # ty: ignore[unresolved-import]

    stamp = common.grid_times().dt.epoch("us").to_numpy()
    primaries = common.series_table().filter(pl.col("substation_type") == "Primary")
    sites = sorted(
        primaries["site"].to_list(), key=lambda x: int(x[1:])
    )  # the Primary-index order of `e00397_prep.primaries()`
    raw = np.load(dense_raw_path, mmap_mode="r")
    n = len(stamp)
    n_use = check_grid_lengths(n_grid=n, dense_raw_len=raw.shape[1], cams_len=n)
    power = np.stack([data.power_points(site=s) for s in sites])
    u = np.zeros_like(power)
    k = np.zeros_like(power)
    for i, s in enumerate(sites):
        u[i], k[i] = data.cams(site=s)
    if power.shape[1] != n_use or u.shape[1] != n_use:
        msg = f"power has {power.shape[1]} steps, grid {n_use}"
        raise ValueError(msg)
    if raw.shape[0] != len(sites):
        msg = f"dense_raw holds {raw.shape[0]} Primaries and the site table {len(sites)}"
        raise ValueError(msg)
    check_dense_raw_alignment(day_of_week_columns=raw[0, :n_use, 14:21], stamp_us=stamp[:n_use])
    temp = np.asarray(raw[:, :n_use, 2], dtype=np.float64)
    return Panel(
        sites,
        stamp,
        apply_short_history_cut(power_pct=power, sites=sites, stamp_us=stamp),
        u,
        k,
        temp,
    )


F7_COMMIT: Final[str] = "126577d1d6d34ed880a79b542d86ecde10e3689e"
"""The commit of experiment 716 (branch `msa-f7-pv-part-1`) whose `e00716_pv` this experiment imports."""


def check_f7_commit(*, module_file: Path) -> str:
    """Return the commit of the tree that holds `module_file`, after asserting it is `F7_COMMIT` and clean.

    Args:
        module_file: The path of a file in the folder of experiment 716's code; its git tree is
            checked.

    Returns:
        The full hash of the checked-out commit.

    Raises:
        RuntimeError: When the tree is at another commit or has uncommitted changes.
    """
    import subprocess

    folder = module_file.resolve().parent
    head = subprocess.run(
        ["git", "-C", str(folder), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(folder), "status", "--porcelain", "--", str(folder)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if head != F7_COMMIT or dirty:
        msg = f"experiment 716 is at {head[:9]} (dirty: {bool(dirty)}); the plan pins {F7_COMMIT[:9]} and a clean tree"
        raise RuntimeError(msg)
    return head
