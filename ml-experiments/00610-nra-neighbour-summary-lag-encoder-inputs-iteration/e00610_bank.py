# ruff: noqa: E402, D103, E501
"""Experiment 610: the target-side bank with a mirrored neighbour (experiment 595's method) for one arm.

Usage: python e00610_bank.py ARM [--cases ...] [--seeds ...]

Experiment 595's `e00595_run.py` fits arm A1 on event-carrying power: for each case it builds the event
world E and the no-event world N, fits on E and forecasts with E's inputs, forecasts the same weights
with N's inputs (`E_on_N`, never refitted), and fits on N. This script does the same for an arm of
this experiment. The arm's signal columns are built from the world's power by the arm's own code
(`e00610_arm.run_fit`), so the injected event is in the targets, the lags, and the signals; the
neighbour columns see the synthetic shifts mirrored onto one neighbour exactly as in training, and the
event itself is not mirrored (the real December 2025 pair is the only verified mirrored event; the
injected designs of experiment 587 move one Primary).

Outputs go under `data/ml-experiments/00610-*/bank/<arm>/<world>/work/`, read by `e00610_bank_score.py`.
"""

import argparse
import json
import sys
import time
from pathlib import Path

_p = argparse.ArgumentParser()
_p.add_argument("arm")
_p.add_argument("--cases", nargs="+", default=None)
_p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
MINE = _p.parse_args()
sys.argv = ["e00610_arm.py", MINE.arm, "fit", "--seeds", *map(str, MINE.seeds)]

import e00587_common as c587  # ty: ignore[unresolved-import]
import e00610_arm as arm  # ty: ignore[unresolved-import]
import numpy as np
import torch  # ty: ignore[unresolved-import]

e = arm.e
OUT = arm.ROOT / "bank" / arm.DIR


def world_dir(*, world: str, on: str | None = None) -> Path:
    return OUT / (world if on is None else f"{world}_on_{on}")


def done(*, world: str, month: str, on: str | None = None) -> bool:
    d = world_dir(world=world, on=on) / "work" / "fits" / e.SPEC.tag
    return all(
        (d / f"{month}_d{day}_s{s}.parquet").exists() or (d / f"{month}_d{day}_s{s}.empty").exists()
        for day in (0, 14)
        for s in MINE.seeds
    )


def run_world(*, world: str, obs: np.ndarray, month: str, on: str | None = None) -> None:
    """Fit (or reload) the arm on `world`, forecast with the inputs of `on` (default: `world`)."""
    inputs = world if on is None else on
    if done(world=world, month=month, on=on):
        return
    base = world_dir(world=world, on=on)
    e.W = base / "work"
    e.WORK = base
    if on is not None:  # reload the fits of `world`; never refit
        src = world_dir(world=world) / "work" / "models"
        missing = [
            f"{month}_d{d}_s{s}.pt"
            for d in (0, 14)
            for s in MINE.seeds
            if not (src / e.SPEC.tag / f"{month}_d{d}_s{s}.pt").exists()
        ]
        assert not missing, f"E_on_N would refit: missing weights {missing}"
        (base / "work").mkdir(parents=True, exist_ok=True)
        link = base / "work" / "models"
        if not link.exists():
            link.symlink_to(src)
    stack = c587.world_clean(world=inputs, obs=obs)
    e.c.clean_stack = lambda: stack
    e.CHECK.update(train_rows=0, test_rows=0, origins_weather_checked=0, months={})
    e.ARGS.months = [month]
    t = time.monotonic()
    arm.run_fit(device="cuda")
    (base / "checks.json").write_text(json.dumps(e.CHECK, indent=1, default=str))
    print(
        f"[{arm.ARM}] world {world} on {inputs} {month}: {time.monotonic() - t:.0f} s", flush=True
    )
    torch.cuda.empty_cache()


def main() -> None:
    obs = e.c.clean_stack()
    cases = MINE.cases or ["verified", *[f"inj{i}" for i in c587.designs()["spec_id"].to_list()]]
    for case in cases:
        ev, no = c587.case_worlds(case=case)
        for month in c587.case_months(case=case):
            run_world(world=ev, obs=obs, month=month)
            run_world(world=ev, obs=obs, month=month, on=no)
            run_world(world=no, obs=obs, month=month)


if __name__ == "__main__":
    main()
