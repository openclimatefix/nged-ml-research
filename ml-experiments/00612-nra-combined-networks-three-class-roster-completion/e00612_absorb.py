# ruff: noqa: E402, D103, E501, ANN001, ANN201, ANN003
"""Experiment 612: fit one network on event-carrying power and forecast, for the absorption measure (GPU).

Usage: python e00612_absorb.py KIND [--cases ...] [--seeds ...] [--months-only M ...]

KIND is `c2` (the class 2 candidate: experiment 610's P7 with 50% signal dropout, with E3) or `k3d5`
(the lag-free K3 with the D5 offset and E3). The structure is experiment 595's `e00595_run.py` and
experiment 587's `e00587_run.py`: for each of the 12 injected designs and the verified December pair,
build the event world E and the no-event world N, then per fold month

1. fit on E (refits on days 0 and 14, seeds 0 and 1) and forecast with E's inputs (`E`);
2. forecast the same weights with N's inputs (`E_on_N`; not needed for the lag-free K3, which reads no
   lags and so has `E_on_N` equal to `E`);
3. fit on N and forecast with N's inputs (`N`; a world shared between cases is fitted once).

For `c2` the signals of the class 2 candidate (gap, weighted averages, neighbour summaries) are rebuilt
from each world's power, so the neighbour of an injected transfer carries the mirrored opposite shift:
this is experiment 595's target-side bank with a mirrored neighbour. Pre-training, step 2, B1
statistics, and the lag-free table `T` are read only from the stored chains.
"""

import argparse
import sys

_p = argparse.ArgumentParser()
_p.add_argument("kind", choices=("c2", "k3d5"))
_p.add_argument("--cases", nargs="+", default=None)
_p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
MINE = _p.parse_args()
KIND = MINE.kind
SEEDS = list(MINE.seeds)

if KIND == "c2":
    sys.argv = [
        "e00612_nbarm.py", "P7+D50", "fit", "--seeds", *map(str, SEEDS), "--work-suffix", "_abs", "--e3", "1",
    ]  # fmt: skip
else:
    sys.argv = ["e00565_fit.py", "fit", "--arm", "D5", "--seeds", *map(str, SEEDS)]

import json
import time

import e00587_common as c587  # ty: ignore[unresolved-import]
import numpy as np
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00612-nra-combined-networks-three-class-roster-completion"
OUT = ML_EXPERIMENTS_DATA_DIR / NAME / "absorb"
TAGS = {"c2": "P5", "k3d5": "D5"}
TAG = TAGS[KIND]
LAG_FREE = KIND == "k3d5"


def world_dir(*, world: str, on: str | None = None):
    return OUT / KIND / (world if on is None else f"{world}_on_{on}")


if KIND == "c2":
    import e00612_nbarm as arm  # ty: ignore[unresolved-import]

    e = arm.e
    assert e.SPEC.tag == TAG

    def fit_world_month(*, month: str) -> None:
        e.ARGS.months = [month]
        arm.FOLD.clear()
        arm.run_fit()
else:
    import e00565_fit as e  # ty: ignore[unresolved-import]
    import numpy as np  # noqa: F811

    W565 = e.W
    e.pretrained_dir = lambda: W565 / "pretrained_D5" / e.SPEC.columns
    assert e.SPEC.tag == "D5" and e.SPEC.start == "fresh" and e.SPEC.frozen
    E3_EMBARGO_DAYS, DOWN_WEIGHT, DOWN_WEIGHT_DAYS = 7, 0.5, 7
    _select_rows = e.c.select_rows

    def select_rows(**kwargs):
        """Return experiment 510's rows without the newest 7 days, the 7 days before them at half weight."""
        refit = _select_rows(**kwargs)
        rows, cutoff = kwargs["rows"], kwargs["cutoff"]
        age_days = (e.c.us(t=cutoff) - rows.valid_us[refit.index]) / e.c.DAY_US
        keep = age_days >= E3_EMBARGO_DAYS
        window = (age_days >= E3_EMBARGO_DAYS) & (age_days < E3_EMBARGO_DAYS + DOWN_WEIGHT_DAYS)
        weight = np.where(window, refit.weight * DOWN_WEIGHT, refit.weight).astype(np.float32)
        return e.c.Refit(refit.index[keep], refit.y[keep], weight[keep], refit.is_val[keep], refit.report)

    e.c.select_rows = select_rows

    def fit_world_month(*, month: str) -> None:
        e.fit_arm(months=[month], seeds=SEEDS, stress=False, limit=None, device="cuda")


def done(*, world: str, month: str, on: str | None = None) -> bool:
    d = world_dir(world=world, on=on) / "work" / "fits" / TAG
    return all(
        (d / f"{month}_d{day}_s{s}.parquet").exists() or (d / f"{month}_d{day}_s{s}.empty").exists()
        for day in (0, 14)
        for s in SEEDS
    )


def run_world(*, world: str, obs: np.ndarray, month: str, on: str | None = None) -> None:
    """Fit (or reload) on `world`, forecast with the inputs of `on` (default: `world`)."""
    inputs = world if on is None else on
    if done(world=world, month=month, on=on):
        return
    base = world_dir(world=world, on=on)
    e.W = base / "work"
    if on is not None:  # reload the fits of `world`; never refit
        src = world_dir(world=world) / "work" / "models"
        missing = [
            f"{month}_d{d}_s{s}.pt" for d in (0, 14) for s in SEEDS if not (src / TAG / f"{month}_d{d}_s{s}.pt").exists()
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
    fit_world_month(month=month)
    (base / "work").mkdir(parents=True, exist_ok=True)
    (base / "checks.json").write_text(json.dumps(e.CHECK, indent=1, default=str))
    print(f"[{KIND}] world {world} on {inputs} {month}: {time.monotonic() - t:.0f} s", flush=True)
    torch.cuda.empty_cache()


def main() -> None:
    obs = e.c.clean_stack()
    cases = MINE.cases or ["verified", *[f"inj{i}" for i in c587.designs()["spec_id"].to_list()]]
    for case in cases:
        ev, no = c587.case_worlds(case=case)
        for month in c587.case_months(case=case):
            run_world(world=ev, obs=obs, month=month)
            if not LAG_FREE:
                run_world(world=ev, obs=obs, month=month, on=no)
            run_world(world=no, obs=obs, month=month)


if __name__ == "__main__":
    main()
