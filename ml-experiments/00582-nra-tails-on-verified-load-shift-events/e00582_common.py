# ruff: noqa: E501
"""Experiment 582: shared loaders (verified load-shift events of experiment 580, the verified NRA shift).

Time is a half-hour grid index from 2019-09-01 UTC; power is a fraction of effective capacity, as in
experiments 556, 565, 572, and 578. Arrays are (16 Primaries, grid).
"""

from datetime import timedelta
from pathlib import Path
from typing import Final

import e00578_common as k  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final[str] = "00582-nra-tails-on-verified-load-shift-events"
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / NAME / "work"
V580: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00580-verified-load-shift-event-set-and-label-audit/work"
)
"""Experiment 580's work folder: `events_verified.parquet` and `events_labelled_status.parquet`."""
FOLD_MONTHS: Final[tuple[str, ...]] = (
    "2025-07",
    "2025-08",
    "2025-10",
    "2025-12",
    "2026-01",
    "2026-03",
    "2026-05",
    "2026-06",
)
"""Chronological fold months with stored v3+D5 forecasts (2026-05 is fitted by this experiment)."""
EVENT_MONTHS: Final[tuple[str, ...]] = ("2025-12", "2026-01", "2026-05")
"""Months whose corrected-lag twin is rebuilt from the verified transfers."""


def verified_events() -> pl.DataFrame:
    """Return one row per (transfer, Primary): label, primary index, onset and end grid index, signed depth (MW and capacity share)."""
    v = pl.read_parquet(V580 / "events_verified.parquet")
    sm = k.sites()
    prim = dict(zip(sm["label"].to_list(), sm["primary"].to_list(), strict=True))
    cap = dict(zip(sm["label"].to_list(), sm["capacity_mw"].to_list(), strict=True))
    rows = []
    for r in v.iter_rows(named=True):
        for side in ("a", "b"):
            lab = r[f"primary_{side}"]
            if lab not in prim:
                continue
            rows.append(
                {
                    "transfer_id": r["transfer_id"],
                    "label": lab,
                    "primary": prim[lab],
                    "partner": r["primary_b" if side == "a" else "primary_a"],
                    "onset_idx": r[f"onset_{side}_idx"],
                    "end_idx": r[f"end_{side}_idx"],
                    "length_days": (r[f"end_{side}_idx"] - r[f"onset_{side}_idx"]) / 48.0,
                    "depth_mw": r[f"depth_{side}_mw"],
                    "depth_share": r[f"depth_{side}_mw"] / cap[lab],
                    "labelled": bool(r["labelled_event_ids"]),
                }
            )
    return pl.DataFrame(rows)


def verified_shift(*, shape: tuple[int, int]) -> np.ndarray:
    """Return the additive NRA correction (capacity fraction): counterfactual = observed + shift = observed - depth inside [onset, end), summed over the transfers active at a Primary."""
    sh = np.zeros(shape, np.float32)
    for r in verified_events().iter_rows(named=True):
        sh[r["primary"], r["onset_idx"] : r["end_idx"]] -= np.float32(r["depth_share"])
    return sh


def date_of(idx: int) -> str:
    """Return the UTC calendar date (YYYY-MM-DD) of a half-hour grid index."""
    return (k.GRID_START + timedelta(minutes=30 * int(idx))).strftime("%Y-%m-%d")
