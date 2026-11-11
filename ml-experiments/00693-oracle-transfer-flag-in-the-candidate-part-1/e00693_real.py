# ruff: noqa: E501
"""Experiment 693: the real load transfers as spans on the Candidate's half-hourly grid.

Two sets of spans mark real rows. The verified set is the transfers found in the telemetry alone
(experiment 580's `events_verified.parquet`: a mirrored pair of sustained steps at two Primaries, 7
to 60 days long). The label-plus-verified set adds every event in NGED's logs (experiment 580's
`events_labelled_status.parquet`). The verified set decides; the other is a sensitivity.

Indices in experiment 580's tables count half-hours from 2019-09-01 00:00 UTC. This module shifts
them onto the grid that starts at 2022-01-01 00:00 UTC.
"""

from datetime import UTC, datetime
from typing import Final

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

E580: Final = (
    ML_EXPERIMENTS_DATA_DIR / "00580-verified-load-shift-event-set-and-label-audit" / "work"
)
LABEL_ORIGIN: Final = datetime(2019, 9, 1, tzinfo=UTC)
GRID_START: Final = datetime(2022, 1, 1, tzinfo=UTC)
OFFSET: Final[int] = int((GRID_START - LABEL_ORIGIN).total_seconds() // 1800)
"""Half-hours from the label origin to the grid start; subtract it from a label index."""
CUTOFF: Final = datetime(2026, 7, 1, tzinfo=UTC)
"""No scored valid time reaches this instant."""


def verified_spans() -> dict[str, list[tuple[int, int]]]:
    """Return per Primary label the (onset, end) grid indices of verified transfers (end exclusive)."""
    d = pl.read_parquet(E580 / "events_verified.parquet")
    out: dict[str, list[tuple[int, int]]] = {}
    for r in d.iter_rows(named=True):
        for side in ("a", "b"):
            out.setdefault(r[f"primary_{side}"], []).append(
                (int(r[f"onset_{side}_idx"]) - OFFSET, int(r[f"end_{side}_idx"]) - OFFSET)
            )
    return out


def labelled_spans() -> dict[str, list[tuple[int, int]]]:
    """Return per Primary label the (onset, end) grid indices of every event in NGED's logs."""
    d = pl.read_parquet(E580 / "events_labelled_status.parquet")
    out: dict[str, list[tuple[int, int]]] = {}
    for r in d.iter_rows(named=True):
        out.setdefault(r["label"], []).append(
            (int(r["label_onset_idx"]) - OFFSET, int(r["label_end_idx"]) - OFFSET)
        )
    return out


def mask_of(
    *,
    spans: dict[str, list[tuple[int, int]]],
    labels: list[str],
    length: int,
    before: int | None = None,
) -> np.ndarray:
    """Return the (Primaries, length) mask of half-hours inside any span.

    `before` truncates every span at that grid index, so a mask built for a fold cannot reach the
    fold's cut-off; spans starting at or after it are dropped.
    """
    out = np.zeros((len(labels), length), bool)
    for p, lab in enumerate(labels):
        for onset, end in spans.get(lab, []):
            stop = end if before is None else min(end, before)
            lo = max(onset, 0)
            if stop > lo and lo < length:
                out[p, lo : min(stop, length)] = True
    return out


def onsets_of(
    *, spans: dict[str, list[tuple[int, int]]], labels: list[str], before: int | None = None
) -> dict[int, list[tuple[int, int | None]]]:
    """Return per Primary index the (onset, end) spans for the prevailing head's oracle age.

    A span that reaches `before` (a fold's cut-off) is cut there, and its end becomes unknown (`None`)
    where the real end lies at or after `before`, because the end is not known at the cut-off.
    """
    out: dict[int, list[tuple[int, int | None]]] = {}
    for p, lab in enumerate(labels):
        for onset, end in spans.get(lab, []):
            if before is not None and onset >= before:
                continue
            out.setdefault(p, []).append(
                (onset, None if before is not None and end >= before else end)
            )
    return out
