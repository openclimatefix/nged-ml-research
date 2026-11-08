# ruff: noqa: E501
"""Experiment 650: shared paths and loaders for the extended-shift departure head."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00578_common as k578  # ty: ignore[unresolved-import]
import e00650_bank as gen
import numpy as np
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final[str] = "00650-extended-shift-departure-head-against-decayed-recent-error"
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / NAME
INJECT: Final[Path] = WORK / "inject"
ROWS: Final[Path] = WORK / "k1_bank"
MONTHS: Final[tuple[str, ...]] = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
TRAIN_PASSES: Final[tuple[int, ...]] = tuple(range(gen.N_TRAIN_PASSES))
HELD_OUT_PASSES: Final[tuple[int, ...]] = tuple(
    range(gen.HELD_OUT_FIRST, gen.HELD_OUT_FIRST + gen.N_HELD_OUT_PASSES)
)
ALL_PASSES: Final[tuple[int, ...]] = (*TRAIN_PASSES, *HELD_OUT_PASSES)
WINDOW_START: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
WINDOW_START_IDX: Final[int] = int((WINDOW_START - k578.GRID_START).total_seconds() // 1800)


def month_start_idx(*, month: str) -> int:
    """Return the half-hour grid index of 00:00 UTC on the first day of `month`."""
    t = datetime.fromisoformat(month + "-01").replace(tzinfo=UTC)
    return int((t - k578.GRID_START).total_seconds() // 1800)


def world() -> tuple[dict[int, list[int]], np.ndarray, np.ndarray]:
    """Return (electrical neighbours per Primary, capacity in MW, is-signed-MW flag)."""
    import e00400_blend as b400  # ty: ignore[unresolved-import]

    sm = k578.sites()
    labels = sm["label"].to_list()
    lab2p = {lab: i for i, lab in enumerate(labels)}
    of: dict[int, list[int]] = {i: [] for i in range(len(labels))}
    for a, v in b400.nged_neighbours().items():
        if a in lab2p:
            of[lab2p[a]] = [lab2p[x] for x in v if x in lab2p]
    is_mw = np.array([str(g).startswith("signed-MW") for g in sm["group"].to_list()])
    return of, sm["capacity_mw"].to_numpy().astype(np.float64), is_mw


def shifts_for(*, clean: np.ndarray, month: str, r: int):  # noqa: ANN201
    """Return (own, mirror, events) of pass `r` in `month`; the draw is keyed on the month's start index."""
    nb_of, cap, is_mw = world()
    ms = month_start_idx(month=month)
    return gen.build(
        clean=clean, month_idx=ms, month_start_idx=ms, r=r, nb_of=nb_of, capacity=cap, is_mw=is_mw,
        window_end_idx=WINDOW_START_IDX,
    )  # fmt: skip
