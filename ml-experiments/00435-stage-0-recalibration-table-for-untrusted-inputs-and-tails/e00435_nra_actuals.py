"""Experiment 435, step 1: NRA-corrected actuals for the scored months (no fit).

The normal running arrangement (NRA) reading is the metered reading minus the labelled step over
the event span, at the donor and at every recipient. The steps come from
`events_corrected.parquet` of experiment 400 (the corrected label table, one row per event and
Primary, with the signed depth in MW and the grid index of the first and last half-hour). The
correction is subtracted in MW and divided by the Primary's capacity, because experiment 428's
stored `actual` is a share of capacity. Depth variants scale every depth by 0.75 and 1.25.

An event is dropped (and counted) when its depth is missing or zero, or when its span lies
outside the dense half-hourly grid. Output: `nra_correction_share.npz` (one array per depth
factor, shape primaries by grid) and `nra_events_report.json`, both in this experiment's work
folder.

```bash
PYTHONPATH=ml-experiments/00087-... DATA_PATH_INTERNAL=... uv run --no-sync python \
    ml-experiments/00435-*/e00435_nra_actuals.py
```
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00435-stage-0-recalibration-table-for-untrusted-inputs-and-tails"
    / "work"
)
EVENTS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00400-lagged-lagless-forecast-blend-around-load-shifts-stored"
    / "work"
    / "events_corrected.parquet"
)
DEPTH_FACTORS: Final[tuple[float, ...]] = (0.75, 1.0, 1.25)
SCORED_MONTHS: Final[tuple[str, ...]] = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
"""Experiment 428's four fold months, plus 2025-12."""
N_PRIMARIES: Final[int] = 16


def primary_labels(*, events: pl.DataFrame) -> list[str]:
    """Return the 16 Primary labels in experiment 428's Primary-index order (label number)."""
    labels = sorted(set(events["label"].to_list()), key=lambda x: int(x[1:]))
    assert len(labels) == N_PRIMARIES, labels
    return labels


def month_slice(*, month: str) -> slice:
    """Return the grid-index slice of one calendar month (UTC)."""
    start = datetime(int(month[:4]), int(month[5:]), 1, tzinfo=UTC)
    end = datetime(start.year + (start.month == 12), start.month % 12 + 1, 1, tzinfo=UTC)
    return slice(int(common.grid_index(times=start)), int(common.grid_index(times=end)))


def main() -> None:
    """Build the correction arrays and the event report."""
    events = pl.read_parquet(EVENTS)
    labels = primary_labels(events=events)
    size = common.grid_times().len()
    valid = (
        events["depth_mw"].is_not_null()
        & (events["depth_mw"] != 0)
        & (events["onset_idx"] >= 0)
        & (events["end_idx"] < size)
        & (events["end_idx"] >= events["onset_idx"])
    )
    dropped = events.filter(~valid)
    kept = events.filter(valid)
    arrays = {f: np.zeros((N_PRIMARIES, size), dtype=np.float32) for f in DEPTH_FACTORS}
    for row in kept.iter_rows(named=True):
        p = labels.index(row["label"])
        for f in DEPTH_FACTORS:
            arrays[f][p, row["onset_idx"] : row["end_idx"] + 1] += (
                f * row["depth_mw"] / row["capacity_mw"]
            )
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(WORK_DIR / "nra_correction_share.npz", **{f"f{f}": a for f, a in arrays.items()})
    per_month: dict[str, dict[str, int]] = {}
    for month in SCORED_MONTHS:
        s = month_slice(month=month)
        a = arrays[1.0][:, s]
        overlap = kept.filter((pl.col("onset_idx") < s.stop) & (pl.col("end_idx") >= s.start))
        per_month[month] = {
            "events_overlapping": overlap.height,
            "primary_half_hours_corrected": int((a != 0).sum()),
        }
    report = {
        "events_total": events.height,
        "events_dropped": dropped.height,
        "dropped_event_ids": dropped["event_id"].to_list(),
        "depth_factors": list(DEPTH_FACTORS),
        "per_month": per_month,
        "labels": labels,
    }
    (WORK_DIR / "nra_events_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
