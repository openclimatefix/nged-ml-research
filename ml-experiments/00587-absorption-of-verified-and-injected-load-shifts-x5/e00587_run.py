# ruff: noqa: E501, E402, D103, ANN202
"""Experiment 587: fit the arms on event-carrying power and forecast, one process per harness.

Usage: python e00587_run.py HARNESS --arm ARM [--cases ...] [--seeds ...]

HARNESS is `565` (arms M0 = `C-M0`, D5) or `574` (arm P5); the two harnesses patch shared
modules, so they run in separate processes. For each case this wrapper builds the event world E
and the no-event world N (`e00587_common`), then, per fold month of the case and per arm:

1. fits the arm on E (refits on days 0 and 14, seeds 0 and 1) and forecasts the scored windows
   with E's inputs (`E`);
2. forecasts the same fitted weights with N's inputs (`E_on_N`, by pointing the fit folder's
   `models` at a symlink, so `fit_arm` reloads and does not refit);
3. fits the arm on N and forecasts with N's inputs (`N`; a world that is shared between cases is
   fitted once).

What is reused unchanged from the harness: rows, recency weights, early stopping, seeds, scoring
windows, origin-time weather assertions, pre-trained and step-2 weights (read only from the
harness' own folder). Nothing in an injected world reaches step 2 or the pre-training.
"""

import argparse
import sys

_p = argparse.ArgumentParser()
_p.add_argument("harness", choices=("565", "574"))
_p.add_argument("--arm", required=True, choices=("M0", "D5", "P5"))
_p.add_argument("--cases", nargs="+", default=None)
_p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
MINE = _p.parse_args()
import e00587_common as common  # ty: ignore[unresolved-import]

assert common.ARMS[MINE.arm][0] == MINE.harness, "arm and harness disagree"
ARMS = [MINE.arm]  # one arm per process: the harness keeps arm-specific module state
_tag = common.ARMS[ARMS[0]][1]
sys.argv = [f"e00{MINE.harness}_fit.py", "fit", "--arm", _tag, "--seeds", *map(str, MINE.seeds)]
if MINE.harness == "565":
    import e00565_fit as e  # ty: ignore[unresolved-import]
else:
    import e00574_fit as e  # ty: ignore[unresolved-import]
import json
import time

import numpy as np
import torch  # ty: ignore[unresolved-import]

W_ORIG = e.W
"""The harness' own work folder: pre-trained weights, step-2 weights, and statistics are read from here only."""
_pretrained_dir = e.pretrained_dir


def _pretrained_dir_orig():
    keep = e.W
    e.W = W_ORIG
    try:
        return _pretrained_dir()
    finally:
        e.W = keep


e.pretrained_dir = _pretrained_dir_orig
if MINE.harness == "574":
    _s2, _st = e.step2_file, e.stats_file

    def _step2_orig(*, month: str, seed: int):
        keep, e.W = e.W, W_ORIG
        try:
            return _s2(month=month, seed=seed)
        finally:
            e.W = keep

    def _stats_orig(*, month: str):
        keep, e.W = e.W, W_ORIG
        try:
            return _st(month=month)
        finally:
            e.W = keep

    e.step2_file, e.stats_file = _step2_orig, _stats_orig
else:
    _s2 = e.step2_file

    def _step2_orig(*, month: str, seed: int):
        keep, e.W = e.W, W_ORIG
        try:
            return _s2(month=month, seed=seed)
        finally:
            e.W = keep

    e.step2_file = _step2_orig


def done(*, arm: str, world: str, month: str, on: str | None = None) -> bool:
    """Return whether every refit window of an arm, world, and month has its forecast file."""
    d = common.world_dir(arm=arm, world=world, on=on) / "work" / "fits" / common.ARMS[arm][1]
    return all(
        (d / f"{month}_d{day}_s{s}.parquet").exists() or (d / f"{month}_d{day}_s{s}.empty").exists()
        for day in (0, 14)
        for s in MINE.seeds
    )


def run_world(*, arm: str, world: str, obs: np.ndarray, month: str, on: str | None = None) -> None:
    """Fit (or reload) `arm` on `world`, forecast with the inputs of `on` (default: `world`)."""
    inputs = world if on is None else on
    if done(arm=arm, world=world, month=month, on=on):
        return
    base = common.world_dir(arm=arm, world=world, on=on)
    e.W = base / "work"
    if on is not None:  # reload the fits of `world`; never refit
        tag = common.ARMS[arm][1]
        missing = [
            f"{month}_d{d}_s{s}.pt" for d in (0, 14) for s in MINE.seeds
            if not (common.world_dir(arm=arm, world=world) / "work" / "models" / tag / f"{month}_d{d}_s{s}.pt").exists()
        ]  # fmt: skip
        assert not missing, f"E_on_N would refit: missing weights {missing}"
        src = common.world_dir(arm=arm, world=world) / "work" / "models"
        (base / "work").mkdir(parents=True, exist_ok=True)
        link = base / "work" / "models"
        if not link.exists():
            link.symlink_to(src)
    stack = common.world_clean(world=inputs, obs=obs)
    e.c.clean_stack = lambda: stack
    e.CHECK.update(train_rows=0, test_rows=0, origins_weather_checked=0, months={})
    t = time.monotonic()
    e.fit_arm(months=[month], seeds=list(MINE.seeds), stress=False, limit=None, device="cuda")
    (base / "work").mkdir(parents=True, exist_ok=True)
    (base / "checks.json").write_text(json.dumps(e.CHECK, indent=1, default=str))
    print(f"[{arm}] world {world} on {inputs} {month}: {time.monotonic() - t:.0f} s", flush=True)
    torch.cuda.empty_cache()


def main() -> None:
    obs = e.c.clean_stack()
    cases = MINE.cases or ["verified", *[f"inj{i}" for i in common.designs()["spec_id"].to_list()]]
    for arm in ARMS:
        for case in cases:
            ev, no = common.case_worlds(case=case)
            for month in common.case_months(case=case):
                run_world(arm=arm, world=ev, obs=obs, month=month)
                if arm not in common.LAG_FREE:  # lag-free arms read no power lags: E_on_N equals E
                    run_world(arm=arm, world=ev, obs=obs, month=month, on=no)
                run_world(arm=arm, world=no, obs=obs, month=month)


if __name__ == "__main__":
    main()
