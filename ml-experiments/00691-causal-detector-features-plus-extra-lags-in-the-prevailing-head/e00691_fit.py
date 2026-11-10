# ruff: noqa: D103
"""Experiment 691, step 1: fit the FL1 head (detector features plus extra lags) (CPU).

FL1 is experiment 687's hazard head with the extra columns of F1 (features D1 to D6 at the day
lag, week lag, and 6-hour cutoff) followed by the extra columns of L1 (experiment 681's 16 lag
columns). The age that gates the survival probability is the larger of F1's and L1's newest-reading
ages. Everything else (training months, banks, optimiser, loss, dropout on the extra columns, early
stopping, seeds) is experiment 687's code, called unchanged. B0, L1, and F1 are not refitted: the
fits of experiment 687 are the arms that this code would give.

Output per fit, `fits/FL1_<month>_s<seed>.npz` and `.pt` under experiment 691's data folder.

Usage: python e00691_fit.py [--months ...] [--seeds 0 1] [--workers 4]
"""

import argparse
import multiprocessing as mp

import e00650_common as c650  # ty: ignore[unresolved-import]
import e00652_common as c652  # ty: ignore[unresolved-import]
import e00687_fit as f687  # ty: ignore[unresolved-import]
import numpy as np
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00691-causal-detector-features-plus-extra-lags-in-the-prevailing-head"
WORK = ML_EXPERIMENTS_DATA_DIR / NAME
OUT = WORK / "fits"
_ORIGINAL = f687.extra_block


def extra_block(*, arm: str, z: dict, month: str, kind: str) -> tuple[np.ndarray, np.ndarray]:
    if arm != "FL1":
        return _ORIGINAL(arm=arm, z=z, month=month, kind=kind)
    feat, feat_age = _ORIGINAL(arm="F1", z=z, month=month, kind=kind)
    lag, lag_age = _ORIGINAL(arm="L1", z=z, month=month, kind=kind)
    return np.concatenate([feat, lag], axis=1), np.maximum(feat_age, lag_age)


def fit_one(job: tuple[str, str, int]) -> str:
    f687.extra_block = extra_block
    f687.OUT = OUT
    return f687.fit_one(job)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="+", default=list(c650.MONTHS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(c652.SEEDS))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    jobs = [("FL1", m, s) for m in a.months for s in a.seeds]
    print(f"{len(jobs)} fits", flush=True)
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for msg in pool.imap_unordered(fit_one, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
