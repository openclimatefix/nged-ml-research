"""Experiment 611, Design 2: experiment 415's N1 lag network, pooled per group of 2 series.

Experiment 415's `plain_pooled` network (setting g09, `summary` weather scheme, lags of 1 day and
1 week at least 24 h old, a series-type input) is fitted once per fold on the 24 months before the
test month, with the training rows restricted to the group's 2 series. Only `series20` is patched;
the rows, the standardiser, the held-out weeks, the weights, and the fit are experiment 415's.
The shared lag encoder A1 of experiment 591 is not used: it lives in the three-step stack of
experiment 574 for the Primaries and has no counterpart in experiment 415's network.

Usage: python e00611_lag.py --group {bsp|gsp} [--months ...] [--seeds ...]
"""

import argparse

import e00415_fit as f415  # ty: ignore[unresolved-import]
import e00415_prep as prep  # ty: ignore[unresolved-import]
import e00611_pool as pool
import polars as pl

parser = argparse.ArgumentParser()
parser.add_argument("--group", choices=tuple(pool.GROUPS), required=True)
parser.add_argument(
    "--months", nargs="+", default=["2025-10", "2025-12", "2026-01", "2026-03", "2026-06"]
)
parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
ARGS = parser.parse_args()

pool.GROUP = ARGS.group
pool._table = None
OUT = pool.WORK / "work" / ARGS.group


def series20() -> pl.DataFrame:
    """Return the group's 2 series with a row index `primary` of 0 and 1."""
    return pool.group_table()


prep.series20 = series20
f415.OUT_DIR = OUT

if __name__ == "__main__":
    f415.run(arms=["N1"], months=ARGS.months, seeds=ARGS.seeds, device="cuda", time_only=False)
