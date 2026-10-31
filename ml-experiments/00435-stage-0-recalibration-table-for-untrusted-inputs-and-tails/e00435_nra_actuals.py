"""Experiment 435, step 1: NRA-corrected actuals for the scored months (no fit).

The normal running arrangement (NRA) reading is the metered reading minus the labelled step over
the event span, at the donor and at every recipient. The steps come from
`events_corrected.parquet` of experiment 400 (the corrected label table, one row per event and
Primary, with the signed depth in MW and the grid index of the first and last half-hour). The
correction is a share of the Primary's capacity, because experiment 428's stored `actual` is a
share of capacity.

**Measured deviation first.** Experiment 402's step 2 redo found that the labelled depth is not the
measured deviation. For each event the script measures the deviation as experiment 402 does: the
mean over the event span of the reading minus the stage-1 median, less the stage-1 bias (the mean
reading minus median over the 7 days before the event). The measurement is used when it has the
labelled sign, is at least `MIN_SHARE_OF_LABEL` of the labelled depth, and exceeds twice the bias
(experiment 402's own filters). Otherwise the labelled depth is used and the event is counted as
label-only. The depth variants scale the used step by 0.75 and 1.25, so they are centred on the
measured deviation where one exists.

The report also checks the event durations against the label table (median hours, share above
7 days). An event is dropped (and counted) when its depth is missing or zero, or when its span lies
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
import e00087_fit_demand as fd  # ty: ignore[unresolved-import]
import e00402_counterfactual as cf  # ty: ignore[unresolved-import]
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
LABEL_TABLE: Final[Path] = (
    Path(__file__).resolve().parents[1]
    / "00395-xgboost-level-state-winter-spread-anchor-stored-forecasts"
    / "sw_block_detector_label_table_corrected.csv"
)
SCORED_MONTHS: Final[tuple[str, ...]] = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
"""Experiment 428's four fold months, plus 2025-12."""
N_PRIMARIES: Final[int] = 16


def primary_labels(*, events: pl.DataFrame) -> list[str]:
    """Return the 16 Primary labels in experiment 428's Primary-index order (label number)."""
    labels = sorted(set(events["label"].to_list()), key=lambda x: int(x[1:]))
    assert len(labels) == N_PRIMARIES, labels
    return labels


def measured_steps(*, events: pl.DataFrame) -> dict[int, float]:
    """Return the measured deviation (share of capacity) of each event that passes the filters."""
    _, arrays_of = fd._series_and_arrays()
    steps: dict[int, float] = {}
    for label in sorted(set(events["label"].to_list()), key=lambda x: int(x[1:])):
        site_events = events.filter(pl.col("label") == label)
        capacity = float(site_events["capacity_mw"][0])
        median = cf.stage1_median(site=label, capacity=capacity)
        power = arrays_of[label].power
        for ev in site_events.iter_rows(named=True):
            lo, hi = ev["onset_idx"], ev["end_idx"]
            label_shift = ev["depth_mw"] / capacity
            if np.isnan(median[lo : hi + 1]).any() or np.isnan(power[lo : hi + 1]).any():
                continue
            gap = (
                power[max(0, lo - cf.PRE_EVENT_STEPS) : lo]
                - median[max(0, lo - cf.PRE_EVENT_STEPS) : lo]
            )
            bias = float(np.nanmean(gap)) if np.isfinite(gap).any() else 0.0
            shift = float(np.mean(power[lo : hi + 1] - (median[lo : hi + 1] + bias)))
            if shift * label_shift <= 0 or abs(shift) < cf.MIN_SHARE_OF_LABEL * abs(label_shift):
                continue
            if abs(bias) > 0.5 * abs(shift):
                continue
            steps[int(ev["event_id"])] = shift
    return steps


def duration_check(*, events: pl.DataFrame) -> dict[str, float]:
    """Return the median duration and the share above 7 days, per event row and per label row."""
    labels = pl.read_csv(LABEL_TABLE)
    return {
        "event_rows_median_hours": float(events["length_days"].median() * 24.0),
        "event_rows_share_above_7_days": float((events["length_days"] > 7).mean()),
        "label_table_rows": labels.height,
        "label_table_median_hours": float(labels["duration_hours"].median()),
        "label_table_share_above_7_days": float((labels["duration_hours"] > 168).mean()),
    }


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
    measured = measured_steps(events=kept)
    arrays = {f: np.zeros((N_PRIMARIES, size), dtype=np.float32) for f in DEPTH_FACTORS}
    for row in kept.iter_rows(named=True):
        p = labels.index(row["label"])
        for f in DEPTH_FACTORS:
            step = measured.get(int(row["event_id"]), row["depth_mw"] / row["capacity_mw"])
            arrays[f][p, row["onset_idx"] : row["end_idx"] + 1] += f * step
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(WORK_DIR / "nra_correction_share.npz", **{f"f{f}": a for f, a in arrays.items()})
    per_month: dict[str, dict[str, int]] = {}
    for month in SCORED_MONTHS:
        s = month_slice(month=month)
        covered = np.zeros((N_PRIMARIES, s.stop - s.start), dtype=bool)
        for row in kept.iter_rows(named=True):
            lo, hi = max(row["onset_idx"], s.start), min(row["end_idx"] + 1, s.stop)
            if lo < hi:
                covered[labels.index(row["label"]), lo - s.start : hi - s.start] = True
        overlap = kept.filter((pl.col("onset_idx") < s.stop) & (pl.col("end_idx") >= s.start))
        per_month[month] = {
            "events_overlapping": overlap.height,
            "primary_half_hours_corrected": int(covered.sum()),
        }
    report = {
        "events_total": events.height,
        "events_dropped": dropped.height,
        "dropped_event_ids": dropped["event_id"].to_list(),
        "depth_factors": list(DEPTH_FACTORS),
        "events_with_measured_deviation": len(measured),
        "events_label_only": kept.height - len(measured),
        "measured_over_label_depth": {
            "median": float(
                np.median(
                    [
                        measured[i] / (d / c)
                        for i, d, c in kept.select(
                            "event_id", "depth_mw", "capacity_mw"
                        ).iter_rows()
                        if i in measured
                    ]
                )
            )
        },
        "durations": duration_check(events=events),
        "per_month": per_month,
        "labels": labels,
    }
    (WORK_DIR / "nra_events_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
