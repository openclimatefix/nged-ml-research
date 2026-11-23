# ruff: noqa: E501
"""Experiment 720: read-only access to the stored panel, and the fold windows. Nothing is written here.

The loaders import earlier experiments' modules (`e00510_common`, `e00087_common`, `e00412_inputs`),
which import `torch`, so run any script that calls `load_panel` with a Python that has the `gpu`
group installed. They read on the CPU only. The pure functions (`to_day_matrix`, `fold_windows`, and
the census and gate functions in the scripts) need no earlier experiment and are tested without one.
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np

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


def load_panel(*, dense_raw_path: Path) -> Panel:
    """Read the cleaned power, the CAMS arrays, and the temperature on the dense grid.

    Needs `PYTHONPATH` to hold the folders of experiments 87, 397, 412, 418, 422, 428, 510, and the
    like (`run.sh` builds it). It raises when the arrays do not share one grid length.
    """
    import e00087_common as common  # ty: ignore[unresolved-import]
    import e00412_inputs as inp412  # ty: ignore[unresolved-import]
    import e00510_common as c510  # ty: ignore[unresolved-import]

    sites = c510.sites()
    stamp = common.grid_times().dt.epoch("us").to_numpy()
    clean = c510.clean_stack().astype(np.float64) * 100.0
    u = np.zeros_like(clean)
    k = np.zeros_like(clean)
    scale = inp412.u99()
    for i, s in enumerate(sites):
        z = np.load(inp412.WORK_DIR / "cams" / f"{s}.npz")
        if len(z["u"]) != clean.shape[1]:
            msg = f"CAMS array of {s} has {len(z['u'])} steps, the cleaned power {clean.shape[1]}"
            raise ValueError(msg)
        u[i] = z["u"] / scale
        k[i] = z["k"]
    raw = np.load(dense_raw_path, mmap_mode="r")
    if raw.shape[1] != clean.shape[1] or len(stamp) != clean.shape[1]:
        msg = (
            f"grid lengths differ: raw {raw.shape[1]}, power {clean.shape[1]}, stamps {len(stamp)}"
        )
        raise ValueError(msg)
    temp = np.asarray(raw[:, :, 2], dtype=np.float64)
    return Panel(
        sites,
        stamp,
        apply_short_history_cut(power_pct=clean, sites=sites, stamp_us=stamp),
        u,
        k,
        temp,
    )
