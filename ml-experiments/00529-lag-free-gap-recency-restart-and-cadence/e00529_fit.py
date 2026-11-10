# ruff: noqa: ANN001, ANN003, ANN201, E501
r"""Experiment 529, `e00529_fit.py`: kept for the record.

Fit the arms of experiment 529 with experiment 511's harness (outputs redirected to this
experiment's folder).

New arms (all weekly cutoffs unless named): `W-none` (no recency weight), `W-365` (half-life 365
days), `W-scratch` (random start, trained on the month's ENS rows alone at the pretraining rate,
half-life 90 days), `M-90` (one fit at the month's first day, scored on the whole month,
half-life 90 days). Stored references: L-1d (511, daily), L-Ww (510, weekly, half-life 90 days,
pretrained start).

```bash
PYTHONPATH=<every folder under ml-experiments/> DATA_PATH_INTERNAL=<private data folder> OMP_NUM_THREADS=1 \
  systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
  uv run --no-sync python ml-experiments/00529-*/e00529_fit.py fit --arm W-none
```
"""

import argparse
from datetime import timedelta

import e00510_common as c  # ty: ignore[unresolved-import]
import e00510_fit as f510  # ty: ignore[unresolved-import]
import e00511_common as k  # ty: ignore[unresolved-import]
import e00511_fit as f511  # ty: ignore[unresolved-import]
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

W = ML_EXPERIMENTS_DATA_DIR / "00529-lag-free-gap-recency-restart-and-cadence" / "work"
SEEDS = (0, 1)
f510.W = W
f511.W = W

ARMS = {
    "W-none": f511.Arm("W-none", "w", float("inf")),
    "W-365": f511.Arm("W-365", "w", 365.0),
    "W-scratch": f511.Arm("W-scratch", "w", 90.0, pretrained="SCRATCH"),
    "M-90": f511.Arm("M-90", "m", 90.0),
}
SCRATCH_ARMS = {"W-scratch"}
_current = {"arm": ""}

_orig_cutoffs = k.cutoffs
_orig_load = c.load_pretrained
_orig_train = c.train
_orig_new = c.new_model
_orig_pretrained_file = f511.pretrained_file


def cutoffs(*, month: str, cadence: str):
    """Add the monthly cadence: one cutoff, the month's first day."""
    if cadence == "m":
        return [c.month_start(month=month)]
    return _orig_cutoffs(month=month, cadence=cadence)


def pretrained_file(*, arm, month, seed):
    """Scratch arms have no weights file."""
    if arm.pretrained == "SCRATCH":
        return W / "none.pt"
    return _orig_pretrained_file(arm=arm, month=month, seed=seed)


def load_pretrained(*, model, path):
    """Skip loading for scratch arms."""
    if _current["arm"] in SCRATCH_ARMS:
        return
    _orig_load(model=model, path=path)


def train(**kw):
    """Scratch arms train at the pretraining rate."""
    if _current["arm"] in SCRATCH_ARMS:
        kw["lr"] = c.PRETRAIN_LR
    return _orig_train(**kw)


k.cutoffs = cutoffs
f511.pretrained_file = pretrained_file
c.load_pretrained = load_pretrained
c.train = train


def assemble(*, arm: f511.Arm) -> None:
    """Concatenate window forecasts into one table per month and seed under `seed_preds/<arm>/`."""
    for month in c.MONTHS:
        cuts = k.cutoffs(month=month, cadence=arm.cadence)
        for seed in SEEDS:
            parts = [
                W
                / "fits"
                / arm.tag
                / f"{month}_d{f511.day_of(month=month, cut=cut)}_s{seed}.parquet"
                for cut in cuts
            ]
            if all(p.exists() or p.with_suffix(".empty").exists() for p in parts):
                frame = pl.concat([pl.read_parquet(p) for p in parts if p.exists()])
                f511.atomic_frame(
                    frame=frame, target=W / "seed_preds" / arm.tag / f"{month}_s{seed}.parquet"
                )
                print("wrote", arm.tag, month, seed, frame.height, flush=True)
            else:
                print("incomplete", arm.tag, month, seed, flush=True)


def main() -> None:
    """Run a command."""
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("fit", "assemble"))
    parser.add_argument("--arm", required=True, choices=tuple(ARMS))
    parser.add_argument("--months", nargs="+", default=list(c.MONTHS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    a = parser.parse_args()
    arm = ARMS[a.arm]
    _current["arm"] = arm.tag
    if a.command == "fit":
        f511.fit_arm(
            arm=arm,
            months=a.months,
            seeds=list(SEEDS),
            stress=False,
            limit=a.limit,
            device=a.device,
        )
    else:
        assemble(arm=arm)
    _ = timedelta, torch


if __name__ == "__main__":
    main()
