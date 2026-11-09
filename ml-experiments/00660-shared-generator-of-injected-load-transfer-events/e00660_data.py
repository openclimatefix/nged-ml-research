"""Inputs of experiment 660: the real power panel, the edge list, noise units, and logged onsets.

The panel is the 15 Primaries on experiment 258's half-hourly grid (2022-01-01 to 2026-06-30), as
stored by experiment 327 (`power/power_<id>.npy`, cleaned power, NaN where no reading). Noise units
are experiment 653's robust spread of 6-hour block means of the residual (`noise_units.json`).
Nothing on or after 2026-07-01 is read.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from e00660_generator import PowerPanel, StartTimes, start_times_from
from studies.eval_protocol.slices import MVA_ONLY_LABELS
from studies.sources import ML_EXPERIMENTS_DATA_DIR

GRID_START: Final[datetime] = datetime(2022, 1, 1, tzinfo=UTC)
GRID_LENGTH: Final[int] = 78816
"""Half-hours from 2022-01-01 to 2026-07-01 (exclusive)."""
LABEL_ORIGIN: Final[datetime] = datetime(2019, 9, 1, tzinfo=UTC)
"""Origin of the half-hour indices in experiment 580's label tables."""
DAY: Final[int] = 48
SERIES_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR.parent / "studies/per_study/stage1_pilot/demand/inputs/series.parquet"
)
E327: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00327-lsc-label-free-detect-mask-train-pipeline"
E580: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00580-verified-load-shift-event-set-and-label-audit/work"
)
E653: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00653-edge-state-event-table-penalised-segmentation"
NEIGHBOURS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00383-load-shift-xgboost-oracle-first-part-1/work/neighbours_nged.json"
)
WORK: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00660-shared-generator-of-injected-load-transfer-events"
)
SERIES_IDS: Final[tuple[int, ...]] = (3, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 24, 25, 26)
"""The 15 series with a stored forecast (the label of series id 12, D15, is not on the grid)."""
SHORT_HISTORY_CUT: Final[dict[str, int]] = {
    "D18": (datetime(2025, 3, 1, tzinfo=UTC) - GRID_START).days * DAY
}
"""Rows before this index are dropped (the noisy start of a short-history Primary)."""
VERIFIED_PAIRS: Final[dict[str, tuple[str, str, str, str]]] = {
    "D5/D16": ("D5", "D16", "2025-12-29", "2026-01-13"),
    "D3/D22": ("D22", "D3", "2024-10-26", "2024-11-22"),
}
"""The two verified transfers: donor, recipient, first day, and last day of the event."""


def idx_of(*, when: datetime) -> int:
    """Half-hour index of a UTC datetime on the 2022-01-01 grid."""
    return int((when - GRID_START).total_seconds() // 1800)


def series_info() -> dict[str, dict]:
    """Site label, group, and capacity of the 15 Primaries, keyed by site label."""
    table = pl.read_parquet(SERIES_PATH)
    by_id = {int(r["time_series_id"]): r for r in table.iter_rows(named=True)}
    return {by_id[s]["site"]: {**by_id[s], "id": s} for s in SERIES_IDS}


def load_panel() -> PowerPanel:
    """The real, un-injected power panel of the 15 Primaries (MW and MVA kept apart)."""
    info = series_info()
    noise = json.loads((E653 / "noise_units.json").read_text())["sigma_block_mw"]
    power: dict[str, np.ndarray] = {}
    for site, row in info.items():
        series = np.load(E327 / f"power/power_{row['id']}.npy").astype(float)
        series[: SHORT_HISTORY_CUT.get(site, 0)] = np.nan
        power[site] = series
    raw = json.loads(NEIGHBOURS.read_text())
    sites = set(power)
    neighbours = {a: sorted(b for b in raw.get(a, []) if b in sites) for a in sites}
    for a in sites:
        for b in neighbours[a]:
            if a not in neighbours[b]:
                neighbours[b] = sorted([*neighbours[b], a])
    kept = sorted(s for s in sites if s in noise)
    return PowerPanel(
        power={s: power[s] for s in kept},
        is_mva={s: s in MVA_ONLY_LABELS for s in kept},
        noise={s: float(noise[s]) for s in kept},
        neighbours={s: [n for n in neighbours[s] if n in kept] for s in kept},
        origin=GRID_START,
    )


def load_forecast(*, site: str) -> np.ndarray:
    """Experiment 327's arm C forecast of one Primary (lag-free convex demand model), same grid."""
    return np.load(E327 / f"forecast/C/forecast_{series_info()[site]['id']}.npy").astype(float)


def logged_start_times(*, before: datetime | None = None) -> StartTimes:
    """Hour of day and weekday of NGED's logged onsets, those strictly before `before`."""
    table = pl.read_parquet(E580 / "events_labelled_status.parquet")
    idx = table["label_onset_idx"].drop_nulls().to_numpy()
    cut = (before or datetime(2026, 7, 1, tzinfo=UTC)) - LABEL_ORIGIN
    idx = idx[idx < int(cut.total_seconds() // 1800)]
    return start_times_from(onset_index=idx, origin=LABEL_ORIGIN)
