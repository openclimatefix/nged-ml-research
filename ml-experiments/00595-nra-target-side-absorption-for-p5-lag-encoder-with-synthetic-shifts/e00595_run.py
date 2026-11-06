# ruff: noqa: E402, D103
"""Experiment 595: fit C1 or A1 on event-carrying power and forecast, one process per arm.

Usage: python e00595_run.py ARM [--cases ...] [--seeds ...]

ARM is `C1` (experiment 584: P5 plus lag-only synthetic load-shifts in fine-tuning) or `A1`
(experiment 591: shared lag encoder with a lag-id embedding, same fine-tuning). The wrapper is
experiment 587's `e00587_run.py` with the arm module of 584 or 591 in place of 574's: for each
case it builds the event world E and the no-event world N, then per fold month

1. fits the arm on E (refits on days 0 and 14, seeds 0 and 1) and forecasts the scored windows with
   E's inputs (`E`);
2. forecasts the same weights with N's inputs (`E_on_N`, through a symlinked `models` folder, so the
   fit is reloaded and never refitted);
3. fits the arm on N and forecasts with N's inputs (`N`; a world shared between cases is fitted
   once).

What the arm modules read unchanged: P5's step-2 weights and B1 statistics (experiment 574 for C1,
experiment 591's shared chain for A1), read only. The synthetic lag-only shifts are drawn per month
from the month alone, so E and N carry the same synthetic schedule. The injected event is in the
world's power: in the targets, in the lag inputs, and (through the same power) in the synthetic
builder's input.
"""

import argparse
import sys

_p = argparse.ArgumentParser()
_p.add_argument("arm", choices=("C1", "A1"))
_p.add_argument("--cases", nargs="+", default=None)
# the scorer expects seeds 0 and 1, so other seeds are for smoke tests only
_p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
MINE = _p.parse_args()
ARM = MINE.arm
sys.argv = (
    ["e00584_arm.py", "fit", "--seeds", *map(str, MINE.seeds)]
    if ARM == "C1"
    else ["e00591_arm.py", "A1", "fit", "--seeds", *map(str, MINE.seeds)]
)
import json
import time

import e00587_common as c587  # ty: ignore[unresolved-import]
import e00595_common as common  # ty: ignore[unresolved-import]
import numpy as np
import torch  # ty: ignore[unresolved-import]

if ARM == "C1":
    import e00584_arm as arm  # ty: ignore[unresolved-import]

    e = arm.e  # build_month is already patched by the module
else:
    import e00591_arm as arm  # ty: ignore[unresolved-import]

    e = arm.e
    # the wiring that e00591_arm.py applies under `__main__` for the `fit` command
    e.row_nll = arm.row_nll
    e.k.build_month = arm.build_month_syn
    e.k.GpuMonth = arm.AuxGpuMonth
    assert arm.LAMBDA == 0.0, "A1 has no auxiliary loss"
TAG = common.tag(arm=ARM)
assert e.SPEC.tag == TAG


def done(*, world: str, month: str, on: str | None = None) -> bool:
    """Return whether every refit window of the arm, world, and month has its forecast file."""
    d = common.world_dir(arm=ARM, world=world, on=on) / "work" / "fits" / TAG
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
    base = common.world_dir(arm=ARM, world=world, on=on)
    e.W = base / "work"
    if on is not None:  # reload the fits of `world`; never refit
        src = common.world_dir(arm=ARM, world=world) / "work" / "models"
        missing = [
            f"{month}_d{d}_s{s}.pt" for d in (0, 14) for s in MINE.seeds
            if not (src / TAG / f"{month}_d{d}_s{s}.pt").exists()
        ]  # fmt: skip
        assert not missing, f"E_on_N would refit: missing weights {missing}"
        (base / "work").mkdir(parents=True, exist_ok=True)
        link = base / "work" / "models"
        if not link.exists():
            link.symlink_to(src)
    stack = c587.world_clean(world=inputs, obs=obs)
    e.c.clean_stack = lambda: stack
    e.CHECK.update(train_rows=0, test_rows=0, origins_weather_checked=0, months={})
    t = time.monotonic()
    e.fit_arm(months=[month], seeds=list(MINE.seeds), stress=False, limit=None, device="cuda")
    (base / "work").mkdir(parents=True, exist_ok=True)
    (base / "checks.json").write_text(json.dumps(e.CHECK, indent=1, default=str))
    print(f"[{ARM}] world {world} on {inputs} {month}: {time.monotonic() - t:.0f} s", flush=True)
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
