# ruff: noqa: E501, D103, PT018, PERF401
"""Experiment 590: per-event follow ratios and phase coverage on the lag-only injected bank, all four arms.

Reuses experiment 581's `event_table` (follow ratio = mean over rows of (q_tau with the injected lags
minus q_tau of the same weights with clean lags) x sign / depth). The twin is the arm's regular
forecast of the fold (`fits/P5`), reached through the symlink views that `e00584_score.py` builds
(`view_<arm>/fits/V3D5` -> the arm's `fits/P5`; `view_<arm>/inject` -> its injected-bank forecasts).
T0 is experiment 584's C1.

Output: `follow_events.parquet` in this experiment's data folder (one row per month, pass, Primary,
phase, band, arm).
"""

from pathlib import Path

import e00578_common as k578  # ty: ignore[unresolved-import]
import e00581_bank as bank  # ty: ignore[unresolved-import]
import e00581_tails as t  # ty: ignore[unresolved-import]
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ROOT = ML_EXPERIMENTS_DATA_DIR / "00590-synthetic-load-shifts-in-the-training-targets"
ROOT584 = ML_EXPERIMENTS_DATA_DIR / "00584-nra-candidate-pack-x6"
VIEWS = {
    "t0": ROOT584 / "view_c1",
    "t1": ROOT / "view_t1",
    "t2": ROOT / "view_t2",
    "t3": ROOT / "view_t3",
    "p5": ROOT / "view_p5",
}


def arm_root(arm: str, month: str = "") -> Path:
    return VIEWS[arm]


def inject_dir(arm: str) -> Path:
    return VIEWS[arm] / "inject"


t.arm_root, t.inject_dir = arm_root, inject_dir

if __name__ == "__main__":
    sites, clean = k578.sites(), k578.clean()
    parts = []
    for arm, view in VIEWS.items():
        assert (view / "fits/V3D5").exists() and any(inject_dir(arm).iterdir()), arm
        for month in k578.MONTHS:
            for r in range(bank.N_PASSES):
                parts.append(
                    t.event_table(arm=arm, month=month, r=r, sites=sites, clean=clean, free=None)
                )
        print(arm, "done", flush=True)
    pl.concat(parts).write_parquet(ROOT / "follow_events.parquet")
