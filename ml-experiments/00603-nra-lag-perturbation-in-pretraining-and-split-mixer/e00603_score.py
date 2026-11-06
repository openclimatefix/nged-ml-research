# ruff: noqa: E501, ANN001, ANN201, ANN003, E402, PT018, B023, F821, ANN202, D103, SIM115, PTH123, PERF401, UP037, F841, RUF100
"""Experiment 603: score arms J1 to J5 (and J0 = the stored C1) with experiment 584's scorer.

No fit and no GPU. Experiment 584's `e00584_score` points experiment 583's scoring at one arm's
stored forecasts (fits and injected bank). This script adds this experiment's arms to its table and
writes under this experiment's folder, optionally on a subset of fold months.

Usage: python e00603_score.py ARM [--months 2025-12 ...] [--tag NAME]   (ARM: c1 for J0, or J1..J5)
"""

import argparse
from pathlib import Path

import e00583_common as c  # ty: ignore[unresolved-import]
import e00584_score as base  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00603-nra-lag-perturbation-in-pretraining-and-split-mixer"
ROOT = ML_EXPERIMENTS_DATA_DIR / NAME

p = argparse.ArgumentParser()
p.add_argument("arm")
p.add_argument("--months", nargs="+", default=None)
p.add_argument("--tag", default="")
args = p.parse_args()

base.ROOT = ROOT
for j in ("J1", "J2", "J3", "J3c", "J4", "J4b", "J5"):
    base.ARMS[j.lower()] = {
        "fits": ROOT / j / "work/fits",
        "fit_tag": "P5",
        "inject": ROOT / j / "work/inject",
    }
base.ARMS["c1"] = {  # J0: the stored C1 of experiment 584
    "fits": ML_EXPERIMENTS_DATA_DIR / "00584-nra-candidate-pack-x6/c1/work/fits",
    "fit_tag": "P5",
    "inject": ML_EXPERIMENTS_DATA_DIR / "00584-nra-candidate-pack-x6/c1/work/inject",
}
if args.months:
    c.MONTHS = tuple(args.months)
arm = args.arm.lower()
if args.tag:  # separate output folder per scoring run
    _setup = base.setup

    def setup(arm: str, override: float | None = None) -> Path:
        out = _setup(arm, override)
        c.WORK = ROOT / f"score_{arm}{args.tag}"
        return c.WORK

    base.setup = setup
base.main(arm)
