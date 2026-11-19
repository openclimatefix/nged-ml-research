"""Experiment 702: the first cleaned reading of each Primary in the fit pipeline's power array.

Usage (after `uv sync --group gpu`, because the imported modules import `torch`):

    source ml-experiments/00524*/env.sh
    uv run --no-sync python ml-experiments/00702-*/e00702_history.py

The script reads `cleaned_arrays` of experiment 422, the array every K1 fit trains on, and prints
for each Primary the date of its first cleaned half-hour and the share of half-hours present from
that date. The plan (`result.md`) takes its history groups from this table, which is printed and
written to `history_first_clean_reading.csv` in the experiment's private data folder.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00422_prep as prep422  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final[str] = "00702-per-substation-versus-global-training-of-the-modular-demand-network"
OUT: Final[Path] = ML_EXPERIMENTS_DATA_DIR / NAME
GRID_START: Final[datetime] = datetime(2019, 9, 1, tzinfo=UTC)
"""Time of half-hour index 0 of the cleaned power arrays."""


def main() -> None:
    """Print and store each Primary's first cleaned half-hour and share of half-hours present."""
    assert int(GRID_START.timestamp() * 1e6) == prep422.GRID_START_US
    rows = []
    for label, power in prep422.cleaned_arrays().items():
        present = np.isfinite(power)
        first = int(np.flatnonzero(present)[0])
        start = GRID_START + timedelta(minutes=30 * first)
        rows.append(
            {
                "label": label,
                "first_cleaned_reading": start.date(),
                "share_present_from_start": round(float(present[first:].mean()), 4),
            }
        )
    table = pl.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    table.write_csv(OUT / "history_first_clean_reading.csv")
    with pl.Config(tbl_rows=20):
        print(table)


if __name__ == "__main__":
    main()
