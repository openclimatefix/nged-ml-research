# ruff: noqa: E402, D103, E501
"""Experiment 599: fit A1 with a changed fine-tune embargo or row weights on event-carrying power.

Usage: python e00599_run.py ARM [--cases ...] [--seeds ...]

ARM is `E1` (embargo of the daily fine-tune lengthened to 7 days), `E2w` (3-day embargo; fine-tune
rows in the newest 7 days left after the embargo get half weight), `E3` (both), or `E0chk` (no
change, a smoke run that must reproduce experiment 595's A1). The wrapper is experiment 595's
`e00595_run.py` for A1, with two patches applied to experiment 574's fit module before any fit:

- `EMBARGO_DAYS` (read by `embargoed` at call time) is the arm's embargo;
- `select_rows` multiplies the weight of the rows whose age at the cutoff lies in
  `[embargo, embargo + 7)` days by 0.5. The multiplier is applied to the training and the
  early-stopping rows alike, on top of the existing 90-day recency weight.

Step-2 weights and pre-training are experiment 591's stored A1 chain, read only. They never see the
injected shift, so the step-2 embargo stays at 3 days in every arm.
"""

import argparse
import sys

_p = argparse.ArgumentParser()
_p.add_argument("arm", choices=("E1", "E2w", "E3", "E0chk"))
_p.add_argument("--cases", nargs="+", default=None)
_p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
MINE = _p.parse_args()
ARM = MINE.arm
sys.argv = ["e00591_arm.py", "A1", "fit", "--seeds", *map(str, MINE.seeds)]
import json
import time

import e00587_common as c587  # ty: ignore[unresolved-import]
import e00591_arm as arm  # ty: ignore[unresolved-import]
import e00599_common as common  # ty: ignore[unresolved-import]
import numpy as np
import torch  # ty: ignore[unresolved-import]

e = arm.e
# the wiring that e00591_arm.py applies under `__main__` for the `fit` command (as experiment 595)
e.row_nll = arm.row_nll
e.k.build_month = arm.build_month_syn
e.k.GpuMonth = arm.AuxGpuMonth
assert arm.LAMBDA == 0.0, "A1 has no auxiliary loss"
assert e.SPEC.tag == common.TAG
EMBARGO_DAYS, DOWN_WEIGHT_ON = common.VARIANTS[ARM]
assert e.EMBARGO_DAYS == 3, "the base embargo is experiment 574's 3 days"
e.EMBARGO_DAYS = EMBARGO_DAYS
_select_rows = e.c.select_rows


def select_rows(**kwargs):  # noqa: ANN003, ANN201
    """Return experiment 510's rows, with the weights of the newest 7 days left after the embargo halved."""
    refit = _select_rows(**kwargs)
    if not DOWN_WEIGHT_ON:
        return refit
    rows, cutoff = kwargs["rows"], kwargs["cutoff"]
    age_days = (e.c.us(t=cutoff) - rows.valid_us[refit.index]) / e.c.DAY_US
    window = (age_days >= EMBARGO_DAYS) & (age_days < EMBARGO_DAYS + common.DOWN_WEIGHT_DAYS)
    weight = np.where(window, refit.weight * common.DOWN_WEIGHT, refit.weight).astype(np.float32)
    report = {**refit.report, "downweighted_rows": int(window.sum()), "embargo_days": EMBARGO_DAYS}
    return e.c.Refit(refit.index, refit.y, weight, refit.is_val, report)


e.c.select_rows = select_rows
TAG = common.TAG


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
