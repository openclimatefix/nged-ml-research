# ruff: noqa: E501
"""Experiment 701: point experiment 87's XGBoost scripts at the origin months 2026-07 to 2026-10.

Importing this module patches `e00087_common`, `e00087_regimes`, and `e00087_offset` in place, so
that experiment 87's fit, offset, and assembly code runs unchanged on the 123 daily runs of
2026-07-01 to 2026-10-31 and writes only under this experiment's private sandbox. Nothing in
experiment 87's folder or data is edited. Import this module before any other `e00087_*` module that
reads a constant, and put the folders of experiments 87, 33, and 48 on `PYTHONPATH`:

    PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\
ml-experiments/00033-best-guess-architecture-vs-production:\
ml-experiments/00048-weekly-retraining-confirmation-and-extension-demand-generators:\
ml-experiments/00701-gam-and-xgboost-for-the-final-test-months-2026-07-to-2026-10 \
        uv run --no-sync python ml-experiments/00701-*/e00701_fit_demand.py ...

The patch must also run in the worker processes of the fit, which import experiment 87's module by
name. A worker re-runs the main script's imports, and every script of this folder imports this
module first, so the workers are patched as well.

What the patch changes, and why:

- **The inputs.** `OBSERVED_PATH` and `ERA5_PATH` point at experiment 607's extended copies (which
  equal the stored tables on every shared row), and `GRID_END` moves to 2026-11-30 23:30.
- **The runs.** `FIRST_RUN` and `LAST_RUN` become 2026-07-01 and 2026-10-31, so that
  `all_origins()` is the 123 new runs and the fit covers only them. `HOLDOUT` moves to 2027-01-01
  so that the blend weights of an origin month may use any earlier valid time.
- **The frames.** `demand_frame_path` finds a new run's frame in this experiment's sandbox and a
  stored run's frame where experiment 87 reads it (the stored runs serve the reproduction checks).
- **The training rows.** `AFTER_DIR` is a folder of links to experiment 34's control-member runs to
  2026-06-29 and this experiment's runs from 2026-06-30 (`e00701_build_inputs.py links`).
- **The refits.** The weekly regime gains the refits every 7 days from 2026-06-30 to 2026-10-27, and
  the monthly regime the refits of 2026-07-01 to 2026-10-01. A run on or before 2026-06-30 09:00
  keeps the refit that experiment 87 assigned to it, so the stored forecasts stay as they were.
- **The outputs.** Refit predictions, the offset tables, and the GPU-hour log go to the sandbox.
"""

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_offset as offsets  # ty: ignore[unresolved-import]
import e00087_regimes as regimes  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 701
SLUG: Final[str] = "gam-and-xgboost-for-the-final-test-months-2026-07-to-2026-10"
ROOT: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
"""This experiment's private data folder: the only folder the scripts write to."""
WORK: Final[Path] = ROOT / "work"
E607_WORK: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00607-bakeoff-extend-cached-inputs-to-2026-10-31" / "work"
)
"""Experiment 607's extended inputs, read only."""
E87_WORK: Final[Path] = common.WORK_DIR
"""Experiment 87's work folder, read only (the path before the patch below)."""

NEW_FIRST_RUN: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
NEW_LAST_RUN: Final[datetime] = datetime(2026, 10, 31, tzinfo=UTC)
STORED_LAST_RUN: Final[datetime] = datetime(2026, 6, 30, tzinfo=UTC)
STORED_FIRST_RUN: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)
AFTER_FIRST_NEW_RUN: Final[datetime] = datetime(2026, 6, 30, tzinfo=UTC)
"""Experiment 34's control-member runs end at 2026-06-29, so this experiment's start at 2026-06-30."""
AFTER_LAST_NEW_RUN: Final[datetime] = datetime(2026, 10, 26, tzinfo=UTC)
"""The latest refit is 2026-10-27 00:00, which sees only runs whose origin is at or before it."""
FIRST_NEW_REFIT: Final[datetime] = datetime(2026, 6, 30, tzinfo=UTC)

FRAMES_DIR: Final[Path] = WORK / "frames_demand"
"""One `init=<date>.parquet` demand frame per new run, built by `e00701_build_inputs.py frames`."""
FRAMES_CHECK_DIR: Final[Path] = WORK / "frames_check"
"""Rebuilt frames of stored runs, for the reproduction check and the smoke test."""
AFTER_NEW_DIR: Final[Path] = WORK / "ens_after_new"
AFTER_ALL_DIR: Final[Path] = WORK / "ens_after_all"
AFTER_CHECK_DIR: Final[Path] = WORK / "ens_after_check"
OFFSET_DIR: Final[Path] = WORK / "offset"
ANCHOR_DIR: Final[Path] = WORK / "anchor_preds"
"""Fit output of the new runs; same layout as experiment 87's `anchor_preds`."""
ASSEMBLE_DIR: Final[Path] = WORK / "assemble"
"""Holds `anchor_preds` (stored and new files together), `arms`, and `results`."""
BACKTESTS_DIR: Final[Path] = ROOT / "backtests"
"""The exported forecast files, in experiment 700's layout."""

NEW_WEEKLY: Final[tuple[datetime, ...]] = tuple(
    FIRST_NEW_REFIT + timedelta(days=7 * i) for i in range(18)
)
"""Every 7 days from 2026-06-30 to 2026-10-27 (experiment 87's schedule runs every 7 days from
2025-07-01, so 2026-06-30 is the next date)."""
NEW_MONTHLY: Final[tuple[datetime, ...]] = tuple(
    datetime(2026, month, 1, tzinfo=UTC) for month in (7, 8, 9, 10)
)

_STORED_ORIGIN_LIMIT: Final[datetime] = common.origin_of(run=STORED_LAST_RUN)


def stored_frame_path(*, run: datetime) -> Path:
    """Return where experiment 87 reads the demand frame of one of its 365 stored runs."""
    name = f"init={run:%Y-%m-%d}.parquet"
    if (run - STORED_FIRST_RUN).days % common.EVERY5D_STRIDE_DAYS == 0:
        return common.VALIDATION_DIR / name
    return common.EXPERIMENT_00048_WORK_DIR / "frames_demand" / name


def demand_frame_path(*, run: datetime) -> Path:
    """Return the demand frame of a run: this sandbox's if it holds one, else the stored one."""
    name = f"init={run:%Y-%m-%d}.parquet"
    for folder in (FRAMES_DIR, FRAMES_CHECK_DIR):
        if (folder / name).exists():
            return folder / name
    if run > STORED_LAST_RUN:
        msg = f"no demand frame of the new run {run:%Y-%m-%d}; run e00701_build_inputs.py frames"
        raise FileNotFoundError(msg)
    return stored_frame_path(run=run)


def chosen_refit(*, refits: tuple[datetime, ...], origin: datetime) -> datetime:
    """Return the latest refit at or before an origin, as `e00087_regimes.chosen_refit` does.

    A stored run (origin at or before 2026-06-30 09:00) ignores the refits from 2026-06-30 on, so
    that it keeps the refit that experiment 87 assigned to it.

    Args:
        refits: The refit times to choose from.
        origin: The forecast origin.

    Returns:
        The latest refit time at or before the origin.

    Raises:
        ValueError: If every refit is later than the origin.
    """
    limit = FIRST_NEW_REFIT if origin <= _STORED_ORIGIN_LIMIT else None
    earlier = [r for r in refits if r <= origin and (limit is None or r < limit)]
    if not earlier:
        msg = f"no refit at or before {origin}"
        raise ValueError(msg)
    return max(earlier)


def apply() -> None:
    """Patch experiment 87's modules in place; safe to call twice."""
    if getattr(common, "_E00701_PATCHED", False):
        return
    common.OBSERVED_PATH = E607_WORK / "observed.parquet"
    common.ERA5_PATH = E607_WORK / "era5_features.parquet"
    common.GRID_END = datetime(2026, 11, 30, 23, 30, tzinfo=UTC)
    common.FIRST_RUN = NEW_FIRST_RUN
    common.LAST_RUN = NEW_LAST_RUN
    common.HOLDOUT = datetime(2027, 1, 1, tzinfo=UTC)
    common.AFTER_DIR = AFTER_ALL_DIR
    common.FRAMES_DEMAND_DIR = FRAMES_DIR
    common.ANCHOR_PREDS_DIR = ANCHOR_DIR
    common.GPU_HOURS_LOG = WORK / "gpu_hours.jsonl"
    common.WORK_DIR = WORK
    common.demand_frame_path = demand_frame_path  # ty: ignore[invalid-assignment]
    regimes.MONTHLY = (*regimes.MONTHLY, *NEW_MONTHLY)
    regimes.WEEKLY = (*regimes.WEEKLY, *NEW_WEEKLY)
    regimes.REGIMES["monthly"] = regimes.MONTHLY
    regimes.REGIMES["weekly"] = regimes.WEEKLY
    regimes.chosen_refit = chosen_refit  # ty: ignore[invalid-assignment]
    offsets.OFFSET_DIR = OFFSET_DIR
    offsets.U_MEAN_DIR = OFFSET_DIR / "u_mean"
    offsets.ANOMALY_FILE = OFFSET_DIR / "anomaly.parquet"
    common._E00701_PATCHED = True  # ty: ignore[unresolved-attribute]
    os.environ.setdefault("E00701_PATCHED", "1")


apply()
