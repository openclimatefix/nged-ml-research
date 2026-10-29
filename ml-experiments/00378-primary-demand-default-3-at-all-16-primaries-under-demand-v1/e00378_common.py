"""Constants shared by the scripts of experiment 378 (default 3 at all 16 Primaries, demand-v1).

Every input is a stored file of an earlier experiment, read and never written. This experiment
writes only under its own `WORK_DIR`.
"""

import json
from pathlib import Path
from typing import Final

import numpy as np
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 378
SLUG: Final[str] = "primary-demand-default-3-at-all-16-primaries-under-demand-v1"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
WORK_DIR: Final[Path] = DATA_DIR / "work"
MEMBERS_DIR: Final[Path] = WORK_DIR / "members"
ARMS_DIR: Final[Path] = WORK_DIR / "arms"
REPORT_DIR: Final[Path] = WORK_DIR / "report"

E377_WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00377-demand-v1-first-run-feasibility" / "work"
"""Experiment 377's folder, read only: stage 1 (`dp_preds`) and stage 2 (`dp_preds_s2`) per site."""
SOURCE_OF_SITE: Final[dict[str, str]] = {
    **dict.fromkeys(("D5", "D9", "D10", "D11", "D14", "D16", "D18"), "control"),
    **dict.fromkeys(("D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22"), "mva"),
    "D15": "d15",
}
"""Which of experiment 377's `work/<source>/` folders holds each site, and its `E00377_SOURCE`."""
QUANTILES: Final[tuple[str, ...]] = (
    "q01",
    "q05",
    "q10",
    "q25",
    "q40",
    "q50",
    "q60",
    "q75",
    "q90",
    "q95",
    "q99",
)
LEVELS: Final[np.ndarray] = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99])
HORIZON_HOURS: Final[float] = 240.0
STORED_HORIZON_HOURS: Final[float] = 168.0
LONG_BAND_EDGES_HOURS: Final[tuple[float, ...]] = (168.0, 192.0, 216.0, 240.0)
WINDOW_DAYS: Final[int] = 365


def stage_path(*, site: str, stage: str) -> Path:
    """Return experiment 377's stored file for a site: `stage` is `dp_preds_s2` (stage 2)."""
    source = SOURCE_OF_SITE[site]
    return E377_WORK / source / "arms" / "a0" / stage / f"stack_d_{site}.parquet"


def long_band(*, lead_hours: np.ndarray) -> np.ndarray:
    """Return the long lead band (0, 1, 2) of leads above 168 hours, and -1 at 168 hours or less."""
    index = np.searchsorted(np.array(LONG_BAND_EDGES_HOURS), lead_hours, side="left") - 1
    return np.where(lead_hours > STORED_HORIZON_HOURS, np.clip(index, 0, 2), -1)


def write_json_atomic(*, path: Path, content: dict) -> None:
    """Write a JSON file through a temporary name so an interrupted write leaves no file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(content, indent=2, sort_keys=True, default=float))
    temporary.rename(path)
