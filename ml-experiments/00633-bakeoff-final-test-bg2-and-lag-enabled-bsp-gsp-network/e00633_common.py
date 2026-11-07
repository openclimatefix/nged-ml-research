# ruff: noqa: E501
"""Experiment 633: shared constants and the redirect onto the extended inputs (grid to 2026-11-10).

The redirect is a copy of `redirect()` in experiment 620's `e00620_extend.py` (an unmerged
experiment, so copied rather than imported): experiment 87's loaders read ERA5 features and observed
power from experiment 607's extended files, on a grid that runs to 2026-11-10.
"""

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import numpy as np
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final[str] = "00633-bakeoff-final-test-bg2-and-lag-enabled-bsp-gsp-network"
ROOT: Final[Path] = ML_EXPERIMENTS_DATA_DIR / NAME
E607: Final[Path] = next(ML_EXPERIMENTS_DATA_DIR.glob("00607-*")) / "work"
GRID_END: Final[datetime] = datetime(2026, 11, 10, 23, 30, tzinfo=UTC)
CUTOFF: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
"""The first instant not trained on; the single refit of BG2-single-fit."""
FIRST_RUN: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
LAST_RUN: Final[datetime] = datetime(2026, 10, 31, tzinfo=UTC)
N_RUNS: Final[int] = 123
SHORT_HISTORY_SITES: Final[tuple[str, ...]] = ("D15", "D18")
SHORT_HISTORY_CUT: Final[datetime] = datetime(2025, 3, 1, tzinfo=UTC)
"""The maintainer's cut of the noisy start of D15 and D18 (2026-10-30)."""
SPREAD_FACTOR: Final[float] = common.SPREAD_FACTOR
SEED: Final[int] = 0
FRAMES: Final[Path] = ROOT / "frames"
BOOSTERS: Final[Path] = ROOT / "boosters"
PHASE: Final[str] = os.environ.get("E00633_PHASE", "")
"""`train`: the held-out power is hidden. `predict`: the window is open (after the go)."""


def runs() -> list[datetime]:
    """Return the 00:00 UTC initialisation times of the 123 daily runs of the window."""
    out = [FIRST_RUN + timedelta(days=i) for i in range(N_RUNS)]
    assert out[-1] == LAST_RUN, out[-1]
    return out


def redirect() -> None:
    """Point experiment 87's loaders at the extended inputs of experiment 607 and a longer grid."""
    common.GRID_END = GRID_END
    common.ERA5_PATH = E607 / "era5_features.parquet"
    common.OBSERVED_PATH = E607 / "observed.parquet"


def hide_window(*, arrays: dict) -> None:
    """Set every series' power from the cutoff on to NaN, so no code path can read a held-out reading."""
    cut = int(common.grid_index(times=CUTOFF))
    for a in arrays.values():
        a.power[cut:] = np.nan
        assert not np.isfinite(a.power[cut:]).any()


def cut_short_history(*, arrays: dict) -> None:
    """Set D15's and D18's power before 2025-03-01 to NaN (training rows, lags, and scoring actuals)."""
    cut = int(common.grid_index(times=SHORT_HISTORY_CUT))
    for site in SHORT_HISTORY_SITES:
        arrays[site].power[:cut] = np.nan
