# ruff: noqa: E501, ANN001, ANN201, E402, PT018, B023, F821, ANN202
"""Experiment 584: fit and inject arm C1 (GPU) = experiment 574's P5 plus lag-only synthetic load-shifts.

A wrapper around experiment 574's `e00574_fit.py` (imported with its module-level state patched, as
experiment 581 did for 572). What is reused unchanged: the rows, recency weights, early stopping, seeds,
scoring windows, origin-time weather assertions, refit days 0 and 14, the embargoed daily fine-tune,
and P5's stored step-2 weights and B1 statistics (experiment 574's work folder, read only). What
differs: during fine-tuning the lag columns (37:45) are rebuilt from power carrying lag-only synthetic
events on 5% of each Primary's time (experiment 581's `synthetic_clean`, same seed rule); targets stay
clean. The test rows are untouched.

Usage: python e00584_arm.py COMMAND [--months ...] [--seeds ...] [--passes ...]
COMMAND: `fit`, `assemble`, or `inject` (forecast the injected bank with the saved weights).
"""

import argparse
import dataclasses
import json
import sys

_p = argparse.ArgumentParser()
_p.add_argument("command", choices=("fit", "assemble", "inject"))
_p.add_argument("--months", nargs="+", default=None)
_p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
_p.add_argument("--passes", nargs="+", type=int, default=None)
MINE = _p.parse_args()
CMD = MINE.command
_argv = ["e00574_fit.py", {"inject": "fit"}.get(CMD, CMD), "--arm", "P5"]
if MINE.months:
    _argv += ["--months", *MINE.months]
_argv += ["--seeds", *map(str, MINE.seeds)]
sys.argv = _argv

import e00574_fit as e  # ty: ignore[unresolved-import]
import e00581_bank as bank  # ty: ignore[unresolved-import]
import numpy as np
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00584-nra-candidate-pack-x6"
WORK = ML_EXPERIMENTS_DATA_DIR / NAME / "c1"
W574 = e.W
"""Experiment 574's work folder: P5's step-2 weights and B1 statistics are read from here, never written."""
e.WORK = WORK
e.W = WORK / "work"
assert e.SPEC.tag == "P5" and e.SPEC.sched == "3sE" and e.SPEC.lags == "24h"


def step2_file(*, month: str, seed: int):
    """Return P5's stored step-2 weights (experiment 574)."""
    p = W574 / "step2" / e.SPEC.tag / f"{month}_s{seed}.pt"
    assert p.exists(), p
    return p


def stats_file(*, month: str):
    """Return the fold's stored B1 standardisation statistics (experiment 574)."""
    p = W574 / "stats" / e.SPEC.b1 / f"{month}.npz"
    assert p.exists(), p
    return p


e.step2_file, e.stats_file = step2_file, stats_file
LAG_FIRST, LAG_WIDTH = 37, 8
_build_month_p5 = e.k.build_month


def build_month_syn(*, month, clean):
    """Build the month's P5 rows, then replace the 8 lag columns by ones read from power with synthetic events.

    Only the lag inputs change; the targets stay clean. The schedule depends only on the month.
    """
    mr = _build_month_p5(month=month, clean=clean)
    origin_idx = (mr.origin_us - e.c.prep422.GRID_START_US) // e.c.prep422.STEP_US
    first, last = int(mr.target_idx.min()), int(mr.target_idx.max())
    fixed, n_events = bank.synthetic_clean(
        clean=clean, first=first, last=last, seed=int(month.replace("-", ""))
    )
    blk = e.block_of(
        clean=fixed, primary=mr.primary, target_idx=mr.target_idx, origin_idx=origin_idx
    )
    assert blk.shape[1] == LAG_WIDTH, blk.shape
    sl = slice(LAG_FIRST, LAG_FIRST + LAG_WIDTH)
    changed = float((np.abs(mr.x[:, LAG_FIRST] - blk[:, 0]) > 1e-6).mean())
    # only values change: missing flags, ages, and grid indices (columns 38, 39, 41, 42, 43, 44) stay as 574 built them
    for old, new in ((38, 1), (39, 2), (41, 4), (42, 5), (43, 6), (44, 7)):
        assert np.array_equal(mr.x[:, old], blk[:, new]), (month, old)
    mr.x[:, sl] = blk
    print(
        f"{month}: {n_events} synthetic events; share of rows with a changed day lag {changed:.4f}",
        flush=True,
    )
    return mr


e.k.build_month = build_month_syn


def _out(out_dir, month, r, day, seed):
    return out_dir / f"{month}_r{r:02d}_d{day}_s{seed}.parquet"


def _done(*, out) -> bool:
    return out.exists() or out.with_suffix(".empty").exists()


def run_inject(*, device: str = "cuda") -> None:
    """Forecast every pass of the injected bank with the saved C1 weights; one parquet per cell."""
    clean = e.c.clean_stack()
    passes = MINE.passes if MINE.passes is not None else list(range(bank.N_PASSES))
    for month in e.ARGS.months:
        out_dir = e.W / "inject"
        cuts = e.k.cutoffs(month=month, cadence=e.CADENCE)
        todo = [
            (r, i, seed)
            for i in range(len(cuts))
            for seed in e.ARGS.seeds
            for r in passes
            if not _done(out=_out(out_dir, month, r, e.f.day_of(month=month, cut=cuts[i]), seed))
        ]
        if not todo:
            continue
        # the training-month rows are only needed for the standardiser and target moments; the synthetic schedule does not matter here
        mr = _build_month_p5(month=month, clean=clean)
        test = e.build_test(month=month, standardiser=mr.base.standardiser)
        e.CTX["variants"] = {}
        e.CTX["on"] = False
        start = e.c.month_start(month=month)
        start_idx = (e.c.us(t=start) - e.c.prep422.GRID_START_US) // e.c.prep422.STEP_US
        sl = slice(LAG_FIRST, LAG_FIRST + LAG_WIDTH)

        def x_for(r: int) -> np.ndarray:
            fixed, _ = bank.contaminate(clean=clean, month_start_idx=start_idx, r=r)
            blk = e.block_of(
                clean=fixed, primary=test.primary, target_idx=test.target_idx,
                origin_idx=test.origin_idx,
            )  # fmt: skip
            x = test.x.copy()
            x[:, sl] = blk
            assert np.array_equal(x[:, 43:45], test.x[:, 43:45]), "contamination moved a lag time"
            return x

        models: dict[tuple[int, int], torch.nn.Module] = {}
        for r, i, seed in todo:
            cut = cuts[i]
            day = e.f.day_of(month=month, cut=cut)
            if (i, seed) not in models:
                models.clear()
                m = e.new_model(n_primaries=len(mr.base.target_mean)).to(device)
                m.load_state_dict(
                    torch.load(
                        e.model_file(tag=e.SPEC.tag, month=month, day=day, seed=seed),
                        map_location=device,
                    )
                )
                m.eval()
                models[(i, seed)] = m
            lo, hi = e.k.window_of(month=month, cadence=e.CADENCE, index=i)
            frame = e.window_frame(
                model=models[(i, seed)], test=dataclasses.replace(test, x=x_for(r)), mr=mr,
                lo=lo, hi=hi, flag=False, device=device,
            )  # fmt: skip
            target = _out(out_dir, month, r, day, seed)
            if frame is None:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.with_suffix(".empty").touch()
            else:
                e.atomic_frame(frame=frame, target=target)
            print(f"C1 inject {month} r{r:02d} d{day} s{seed}", flush=True)
        del mr, test, models
        torch.cuda.empty_cache()


if __name__ == "__main__":
    if CMD == "assemble":
        e.assemble()
    elif CMD == "inject":
        run_inject()
    else:
        e.fit_arm(months=e.ARGS.months, seeds=e.ARGS.seeds, stress=False, limit=None, device="cuda")
        out = WORK / f"checks_fit_C1_s{'-'.join(map(str, e.ARGS.seeds))}.json"
        out.write_text(json.dumps(e.CHECK, indent=1, default=str))
        print("CHECKS PASSED", {a: v for a, v in e.CHECK.items() if a != "months"})
