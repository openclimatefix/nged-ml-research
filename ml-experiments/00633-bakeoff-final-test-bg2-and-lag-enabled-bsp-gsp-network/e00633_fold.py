# ruff: noqa: ANN001, ANN003, ANN202, D103, E501
"""Experiment 633, Part 2: the final-test fold (cutoff 2026-07-01), patched onto the development pipeline.

A copy of experiment 620's `e00620_fold.py` (an unmerged experiment) with the extended inputs of the 4 nodes.

The development pipeline (experiments 510, 511, 574, 591, 613) works on folds named by month, and
refits on days 0 and 14. The final-test fold `2026-07` differs in three ways:

- it has one cutoff, 2026-07-01 00:00 UTC, and no refit afterwards, so every network is fitted on
  data before 2026-07-01 only;
- its scored window runs from 2026-07-01 to 2026-11-01 (forecast origins on 123 days);
- in the training phase (`E00633_PHASE=train`) the cleaned power is set to NaN from 2026-07-01
  onwards, so no code path can read a held-out reading.

`install(e)` patches the module of experiment 574 (`e00574_fit`, or a wrapper that exposes the same
names) in place. Nothing in an earlier experiment's folder is edited.
"""

import os
from datetime import UTC, datetime
from typing import Final

import numpy as np

FINAL_MONTH: Final = "2026-07"
FINAL_START: Final = datetime(2026, 7, 1, tzinfo=UTC)
FINAL_STOP: Final = datetime(2026, 11, 1, tzinfo=UTC)
PHASE: Final = os.environ.get("E00633_PHASE", "")
"""`train`: the held-out power is hidden. `predict`: the scored window is open (after the go)."""
TRAIN_ONLY: Final = PHASE == "train"


def is_final(*, months) -> bool:
    return FINAL_MONTH in months


def redirect_to_extended_inputs(e, *, group: str) -> None:
    """Point the pipeline at the nodes' extended inputs (grid to 2026-11-10, raw matrices, ensemble summaries, tables).

    The held-out power is readable after this call, so it only runs in the prediction phase.
    `e` is experiment 574's module. `group` is `bsp` or `gsp`; the expected-input cache is per group.
    """
    import e00437_features as f437  # noqa: PLC0415
    import e00633_common as k633  # noqa: PLC0415

    k633.redirect()
    e.prep.SUMMARY_DIR = k633.ROOT / "inputs" / "ens_summary_nodes"
    e.prep.RAW_DIR = k633.ROOT / "inputs" / "raw"
    f437.TRAILING_DIR = k633.ROOT / "inputs" / "trailing437_nodes"
    e.W560 = k633.ROOT / "inputs" / "w560_nodes" / group
    e.data.HOLDOUT = datetime(2027, 1, 1, tzinfo=UTC)
    # the timing check of experiment 574 reads a run's demand frame (run time and origin); the weather
    # frames of this experiment hold both for the runs after 2026-06-30
    _frame_path = e.common.demand_frame_path

    def demand_frame_path(*, run):
        path = _frame_path(run=run)
        return path if path.exists() else k633.FRAMES / f"init={run:%Y-%m-%d}.parquet"

    e.common.demand_frame_path = demand_frame_path
    _inner = e._build_test

    def build_test_all_months(*, month, standardiser):
        """Return the months' test rows; the final fold's window is July to October together."""
        if month != FINAL_MONTH:
            return _inner(month=month, standardiser=standardiser)
        parts = [
            _inner(month=m, standardiser=standardiser)
            for m in ("2026-07", "2026-08", "2026-09", "2026-10")
        ]
        first = parts[0]
        import dataclasses

        if os.environ.get(
            "E00633_DRY"
        ):  # coverage check: build the window's input rows, forecast nothing
            raise SystemExit(
                f"DRY RUN OK: the window's test rows were built ({len(first.target_idx) if hasattr(first, 'target_idx') else '?'} rows in July); nothing was forecast"
            )

        return dataclasses.replace(
            first,
            **{
                f.name: np.concatenate([getattr(p, f.name) for p in parts])
                for f in dataclasses.fields(first)
            },
        )

    e._build_test = build_test_all_months


def install(e, *, months: list[str], group: str, reference_month: str | None = None) -> None:
    """Patch experiment 574's module `e` for the final fold; do nothing when no month is final.

    In the training phase `build_test` returns None, or, when `reference_month` is given (the
    class 2 candidate's training-noise plan compares training rows with a month of real inputs),
    that development month's test rows, built with the final fold's standardiser.
    """
    if PHASE == "predict":
        redirect_to_extended_inputs(e, group=group)
    if not is_final(months=months):
        return
    assert PHASE in {"train", "predict"}, "set E00633_PHASE to train or predict for the final fold"
    c, k = e.c, e.k
    _cutoffs, _window_of = k.cutoffs, k.window_of
    _month_stop = c.month_stop

    def cutoffs(*, month, cadence):
        if month == FINAL_MONTH:
            return [FINAL_START]
        return _cutoffs(month=month, cadence=cadence)

    def window_of(*, month, cadence, index):
        if month == FINAL_MONTH:
            assert index == 0
            return c.us(t=FINAL_START), c.us(t=FINAL_STOP)
        return _window_of(month=month, cadence=cadence, index=index)

    def month_stop(*, month):
        return FINAL_STOP if month == FINAL_MONTH else _month_stop(month=month)

    _extension = c._extension

    def extension(*, refit, last, clean, rows):
        """Return experiment 510's later-cutoff rows; none exist when the only cutoff is the refit."""
        if refit != FINAL_START:
            return _extension(refit=refit, last=last, clean=clean, rows=rows)
        assert last == refit, "the final fold has one cutoff"
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
    k.cutoffs, k.window_of, c.month_stop = cutoffs, window_of, month_stop
    e.cutoffs = cutoffs
    if TRAIN_ONLY:
        assert list(months) == [FINAL_MONTH], "the training phase runs the final fold alone"
        _clean_stack = c.clean_stack
        cut = int((c.us(t=FINAL_START) - c.prep422.GRID_START_US) // c.prep422.STEP_US)

        def clean_stack():
            """Return the cleaned power with every half-hour from 2026-07-01 hidden."""
            clean = _clean_stack().copy()
            clean[:, cut:] = np.nan
            assert not np.isfinite(clean[:, cut:]).any()
            return clean

        c.clean_stack = clean_stack

        _build_test = e.build_test

        def build_test(*, month, standardiser):
            """Keep the scored window closed while the networks are trained."""
            if reference_month is None or month != FINAL_MONTH:
                return (
                    None
                    if month == FINAL_MONTH
                    else _build_test(month=month, standardiser=standardiser)
                )
            assert reference_month < FINAL_MONTH
            e.CHECK["months"].setdefault(reference_month, {})
            return _build_test(month=reference_month, standardiser=standardiser)

        def window_frame(**_):
            return None

        e.data.build_test = build_test
        e.build_test = build_test
        e.f.window_frame = window_frame
        e.window_frame = window_frame
