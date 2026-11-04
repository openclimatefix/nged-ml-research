# ruff: noqa: E501
r"""Check row by row that every ensemble window comes from runs issued at or before the origin.

For a random sample of test rows, rebuild each window position directly from the stored ensemble
summaries and compare it with `WindowStore.window`:

- a position after the row's origin must come from the row's own run (the run with that origin);
- a position at or before the origin must come from an earlier run's first day (origin strictly
  before the row's origin).

The check asserts equality of every value and prints the largest source origin used by any row
against that row's origin, and one example row. Run with `PYTHONPATH` as `e00516_fit.py` shows.
"""

import argparse

import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import e00516_fit as fit
import e00516_window as win
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

SHIFT = 22


def main() -> None:
    """Run the check on one fold's test rows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--month", default="2025-12")
    parser.add_argument("--rows", type=int, default=3000)
    args = parser.parse_args()
    clean = c.clean_stack()
    fold = fit.Fold(month=args.month, clean=clean, need_pre=False, device="cpu")
    stage = fold.stage(arm="C-CNN", stage="test")
    test = fold.base.test
    rng = np.random.default_rng(0)
    pick = rng.choice(len(test.primary), size=args.rows, replace=False)
    windows = stage(torch.from_numpy(pick)).numpy()
    unstandardised = windows * fold.store.std.numpy() + fold.store.mean.numpy()
    sources = {}
    for p in np.unique(test.primary[pick]):
        row = prep.primaries().filter(pl.col("primary") == int(p)).row(0, named=True)
        names = [f"mean_{v}" for v in fit.prep418.WEATHER]
        s = prep.load_summary(series_id=int(row["time_series_id"])).sort(
            "power_fcst_init_time", "valid_time"
        )
        o = (
            s["power_fcst_init_time"].dt.epoch("us").to_numpy() - c.prep422.GRID_START_US
        ) // c.prep422.STEP_US
        v = (
            s["valid_time"].dt.epoch("us").to_numpy() - c.prep422.GRID_START_US
        ) // c.prep422.STEP_US
        sources[int(p)] = (o * (1 << SHIFT) + v, s.select(names).to_numpy())
    worst, max_gap, checked, example = 0.0, -(10**9), 0, None
    for k, i in enumerate(pick):
        p, origin, valid = int(test.primary[i]), int(test.origin_idx[i]), int(test.target_idx[i])
        keys, values = sources[p]
        for col, off in enumerate(win.CENTRED_OFFSETS):
            pos = valid + int(off)
            if pos > origin:
                source_origin = origin
            else:
                source_origin = origin - win.HALF_HOURS_PER_DAY * (
                    1 + (origin - pos) // win.HALF_HOURS_PER_DAY
                )
            assert source_origin <= origin, "a window position came from a later run"
            if pos <= origin:
                assert source_origin < origin
            max_gap = max(max_gap, source_origin - origin)
            at = np.searchsorted(keys, source_origin * (1 << SHIFT) + pos)
            found = at < len(keys) and keys[at] == source_origin * (1 << SHIFT) + pos
            got = unstandardised[k, :, col]
            if found:
                worst = max(
                    worst, float(np.nanmax(np.abs(values[at] - got) / (1 + np.abs(values[at]))))
                )
                checked += 1
            else:
                assert np.all(windows[k, :, col] == 0.0), (
                    "a missing source position was not zero-filled"
                )
        if example is None and valid - origin > 4:
            example = (p, origin, valid)
    print(
        f"{args.month}: {args.rows} test rows x 48 positions, {checked} positions matched to the source run"
    )
    print(f"largest relative difference to the independently rebuilt value: {worst:.2e}")
    print(
        f"largest (source run origin minus row origin) over every position: {max_gap} half-hours (must be <= 0)"
    )
    assert worst < 1e-4
    p, origin, valid = example  # ty: ignore[not-iterable]
    print(f"example row: Primary index {p}, origin grid index {origin}, valid grid index {valid}")
    print(
        f"  positions after the origin read run origin {origin}; positions at or before read first days of runs with origins {origin - 48}, {origin - 96}, ..."
    )


if __name__ == "__main__":
    main()
