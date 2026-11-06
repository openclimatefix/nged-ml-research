# ruff: noqa: D103
"""Experiment 586: tail metrics and lag-following on experiment 581's injected bank, for S0 to S4.

Experiment 581's `e00581_tails` patched, not copied. Arm `v3d5` is S0 (experiment 572's stored
weights; its injected forecasts are experiment 581's). Arms `s1` to `s4` read this experiment's
forecasts. Outputs go under this experiment's data folder (`tails/`). The extra fold month of
experiment 581 is dropped so every arm covers the same five folds.
"""

import sys
from pathlib import Path

import e00581_summary as summary  # ty: ignore[unresolved-import]
import e00581_tails as t  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00586-lag-encoder-sharing-and-lag-id-embedding"
WORK = ML_EXPERIMENTS_DATA_DIR / NAME
S0_INJECT = t.WORK / "arms" / "v3d5" / "work" / "inject"


def arm_root(arm: str, month: str = "") -> Path:
    _ = month
    return t.W572 if arm == "v3d5" else WORK / "arms" / arm / "work"


def inject_dir(arm: str) -> Path:
    return S0_INJECT if arm == "v3d5" else WORK / "arms" / arm / "work" / "inject"


t.WORK, t.arm_root, t.inject_dir, t.EXTRA_MONTHS = WORK, arm_root, inject_dir, ()
summary.NAME = NAME

if __name__ == "__main__":
    if sys.argv[1:2] == ["summary"]:
        import polars as pl

        with pl.Config(tbl_rows=200, tbl_cols=20, tbl_width_chars=240):
            print(summary.injected(arms=sys.argv[2:]).with_columns(pl.col(pl.Float64).round(3)))
    else:
        t.main()
