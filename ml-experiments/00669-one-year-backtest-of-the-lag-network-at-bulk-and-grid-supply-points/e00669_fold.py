# ruff: noqa: ANN001, ANN003, ANN202, D103, E501, PLR0915
"""Experiment 669: monthly folds of the year, patched onto experiment 574's pipeline.

A generalisation of experiment 633's `e00633_fold.py` (one final-test fold) to the 12 test months
2025-11 to 2026-10. Each fold has one cutoff, 00:00 UTC on the first day of its month, no refit
afterwards, and a scored window that is the whole month. Differences from experiment 633:

- inputs are always redirected to experiment 633's extended inputs (grid to 2026-11-10);
- test rows come from this experiment's own matrices (`e00669_testrows.py`), which keep
  every half-hour, and their lead limit is 351 hours;
- with `E00669_HIDE=1` the cleaned power is set to NaN from the cutoff on, so a training run
  cannot read a later reading (used by the pre-flight gate).
"""

import dataclasses
import os
from datetime import UTC, datetime

import numpy as np

HIDE = os.environ.get("E00669_HIDE") == "1"


def month_start(*, month: str) -> datetime:
    return datetime.fromisoformat(f"{month}-01").replace(tzinfo=UTC)


def install(e, *, months: list[str], group: str) -> None:
    import e00428_data as data  # ty: ignore[unresolved-import]
    import e00437_features as f437  # ty: ignore[unresolved-import]
    import e00633_common as k633  # ty: ignore[unresolved-import]
    import e00669_common as k

    k633.redirect()
    e.prep.SUMMARY_DIR = k633.ROOT / "inputs" / "ens_summary_nodes"
    e.prep.RAW_DIR = k633.ROOT / "inputs" / "raw"  # training matrices (observed half-hours only)
    f437.TRAILING_DIR = k633.ROOT / "inputs" / "trailing437_nodes"
    e.W560 = k633.ROOT / "inputs" / "w560_nodes" / group
    e.data.HOLDOUT = datetime(2027, 1, 1, tzinfo=UTC)
    data.HEADLINE_MAX_HOURS = k.MAX_LEAD_HOURS
    c, kk = e.c, e.k

    _frame_path = e.common.demand_frame_path

    def demand_frame_path(*, run):
        path = _frame_path(run=run)
        return path if path.exists() else k633.FRAMES / f"init={run:%Y-%m-%d}.parquet"

    e.common.demand_frame_path = demand_frame_path

    _inner = e._build_test

    def build_test_from_full_rows(*, month, standardiser):
        """Read the test month's rows from the matrices that keep unobserved half-hours."""
        old = e.prep.RAW_DIR
        e.prep.RAW_DIR = k.ROOT / "inputs" / "raw_test"
        try:
            return _inner(month=month, standardiser=standardiser)
        finally:
            e.prep.RAW_DIR = old

    e._build_test = build_test_from_full_rows

    def cutoffs(*, month, cadence):
        return [month_start(month=month)]

    _month_stop = c.month_stop
    kk.cutoffs, e.cutoffs = cutoffs, cutoffs
    _extension = c._extension

    def extension(*, refit, last, clean, rows):
        """Return experiment 510's later-cutoff rows; none exist when the only cutoff is the refit."""
        if last != refit:
            return _extension(refit=refit, last=last, clean=clean, rows=rows)
        empty = lambda a, dtype=None: np.empty((0, *a.shape[1:]), dtype=dtype or a.dtype)  # noqa: E731
        return {
            "x": empty(rows.x, np.float32),
            "y": empty(rows.y),
            "origin": np.empty(0, np.int64),
            "valid": np.empty(0, np.int64),
            "target_idx": np.empty(0, np.int64),
            "lead": empty(rows.lead),
            "primary": np.empty(0, np.int64),
            "is_val": empty(rows.is_val),
        }

    c._extension = extension
    assert _month_stop(month="2025-12") == datetime(2026, 1, 1, tzinfo=UTC)
    assert _month_stop(month="2026-02") == datetime(2026, 3, 1, tzinfo=UTC)
    if HIDE:
        assert len(months) == 1, "the hidden run takes one fold"
        _clean_stack = c.clean_stack
        cut = int(
            (c.us(t=month_start(month=months[0])) - c.prep422.GRID_START_US) // c.prep422.STEP_US
        )

        def clean_stack():
            clean = _clean_stack().copy()
            clean[:, cut:] = np.nan
            return clean

        c.clean_stack = clean_stack
        _bt = e.build_test

        def build_test(*, month, standardiser):
            return None

        def window_frame(**_):
            return None

        e.data.build_test = build_test
        e.build_test = build_test
        e.f.window_frame = window_frame
        e.window_frame = window_frame
    _ = dataclasses
