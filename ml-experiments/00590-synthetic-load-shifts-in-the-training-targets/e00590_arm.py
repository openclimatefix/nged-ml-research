# ruff: noqa: E501, ANN001, ANN201, E402, PT018, B023, F821, ANN202
"""Experiment 590: fit and inject arms T1, T2, and T3 (GPU), variants of experiment 584's arm C1.

Experiment 584's arm C1 (= T0 here) fine-tunes experiment 574's P5 recipe on lag inputs that carry
synthetic load-shifts on 5% of each Primary's time, with clean targets. This wrapper changes only what
the fine-tune sees, by patching `build_month` the way `e00584_arm.py` does. What is reused unchanged:
the rows, recency weights, early stopping, seeds, scoring windows, origin-time weather assertions,
refit days 0 and 14, the embargoed daily fine-tune, and P5's stored step-2 weights and B1 statistics
(experiment 574's work folder, read only). The test rows are untouched.

Arms (`--arm`):

- `t1`: the same schedule as C1 (experiment 581's `synthetic_clean`, same seed rule); the lag columns
  read the shifted power and the target of every row whose valid time lies inside an event is shifted
  by the event's depth, as a real load-shift moves both.
- `t2`: lag-only events on 2.5% of each Primary's time (schedule seed `s`) and lag-plus-target events
  on a further 2.5% (schedule seed `s + 7919`).
- `p5`: reference only, `inject` only: experiment 574's P5 weights (fine-tuned on clean lags)
  forecast the injected bank.
- `t3`: placebo. The lag columns read the power plus independent Gaussian noise, one draw per Primary
  and half-hour, whose standard deviation equals the standard deviation, over the month's grid
  span, of the shift C1 would add at that Primary. No persistent shift; targets stay clean.

The standardiser and target moments stay those of the clean power (as for C1). A shifted target is
shifted by the same amount as the lag the schedule shifts, and where the clean power is missing the
target stays missing.

Usage: python e00590_arm.py COMMAND --arm {t1,t2,t3} [--months ...] [--seeds ...] [--passes ...]
COMMAND: `fit`, `assemble`, or `inject` (forecast the injected bank with the saved weights).
"""

import argparse
import dataclasses
import json
import sys

_p = argparse.ArgumentParser()
_p.add_argument("command", choices=("fit", "assemble", "inject"))
_p.add_argument("--arm", required=True, choices=("t1", "t2", "t3", "p5"))
_p.add_argument("--months", nargs="+", default=None)
_p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
_p.add_argument("--passes", nargs="+", type=int, default=None)
MINE = _p.parse_args()
CMD, ARM = MINE.command, MINE.arm
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

NAME = "00590-synthetic-load-shifts-in-the-training-targets"
WORK = ML_EXPERIMENTS_DATA_DIR / NAME / ARM
W574 = e.W
"""Experiment 574's work folder: P5's step-2 weights and B1 statistics are read from here, never written."""
e.WORK = WORK
e.W = WORK / "work"
assert e.SPEC.tag == "P5" and e.SPEC.sched == "3sE" and e.SPEC.lags == "24h"

LAG_FIRST, LAG_WIDTH = 37, 8
HALF_SHARE = 0.025
"""Share of each Primary's time covered by each of arm T2's two event sets."""
SECOND_SEED_OFFSET = 7919
"""Added to the month's schedule seed for arm T2's lag-plus-target event set."""
NOISE_SEED_OFFSET = 104_729
"""Added to the month's schedule seed for arm T3's noise draws."""


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
_build_month_p5 = e.k.build_month


def shifted_series(
    *, clean: np.ndarray, first: int, last: int, seed: int, share: float
) -> np.ndarray:
    """Return `clean` plus random constant shifts covering `share` of `[first, last)` per Primary.

    The same procedure as `e00581_bank.synthetic_clean` with a variable share; at `share=0.05` the
    result equals that function's (asserted in `check_schedule`).
    """
    rng = np.random.default_rng(seed)
    out = clean.copy()
    for p in range(clean.shape[0]):
        covered = 0
        while covered < share * (last - first):
            length = int(rng.choice(bank.LENGTHS_DAYS)) * bank.PER_DAY
            on = int(rng.integers(first, max(first + 1, last - length)))
            shift = float(rng.choice(bank.DEPTHS)) * float(rng.choice([-1.0, 1.0]))
            seg = out[p, on : on + length]
            out[p, on : on + length] = np.where(np.isfinite(seg), seg + shift, seg)
            covered += length
    return out


def check_schedule(*, clean: np.ndarray, first: int, last: int, seed: int) -> None:
    """Assert that `shifted_series` at 5% reproduces experiment 581's `synthetic_clean` (arm C1's lags)."""
    ref, _ = bank.synthetic_clean(clean=clean, first=first, last=last, seed=seed)
    mine = shifted_series(clean=clean, first=first, last=last, seed=seed, share=bank.SYN_SHARE)
    assert np.array_equal(ref, mine, equal_nan=True), "schedule differs from experiment 581's"


def lag_and_target_series(
    *, clean: np.ndarray, first: int, last: int, seed: int
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return the series the lags read, the target shift per Primary and half-hour, and a description."""
    if ARM == "t1":
        lag = shifted_series(clean=clean, first=first, last=last, seed=seed, share=bank.SYN_SHARE)
        return lag, lag - clean, {"lag_share": bank.SYN_SHARE, "target_share": bank.SYN_SHARE}
    if ARM == "t2":
        only = shifted_series(clean=clean, first=first, last=last, seed=seed, share=HALF_SHARE)
        both = shifted_series(
            clean=only, first=first, last=last, seed=seed + SECOND_SEED_OFFSET, share=HALF_SHARE
        )
        return both, both - only, {"lag_share": 2 * HALF_SHARE, "target_share": HALF_SHARE}
    ref = shifted_series(clean=clean, first=first, last=last, seed=seed, share=bank.SYN_SHARE)
    sd = np.nanstd((ref - clean)[:, first:last], axis=1)
    noise = (
        np.random.default_rng(seed + NOISE_SEED_OFFSET).standard_normal(clean.shape) * sd[:, None]
    )
    lag = np.where(np.isfinite(clean), clean + noise, clean)
    return lag, np.zeros_like(clean), {"lag_noise_sd_by_primary": np.round(sd, 4).tolist()}


def build_month_variant(*, month, clean):
    """Build the month's P5 rows, then change the 8 lag columns and the targets as the arm says."""
    mr = _build_month_p5(month=month, clean=clean)
    origin_idx = (mr.origin_us - e.c.prep422.GRID_START_US) // e.c.prep422.STEP_US
    first, last = int(mr.target_idx.min()), int(mr.target_idx.max())
    seed = int(month.replace("-", ""))
    if month == e.ARGS.months[0]:
        check_schedule(clean=clean, first=first, last=last, seed=seed)
    lag_series, delta, info = lag_and_target_series(clean=clean, first=first, last=last, seed=seed)
    blk = e.block_of(
        clean=lag_series, primary=mr.primary, target_idx=mr.target_idx, origin_idx=origin_idx
    )
    assert blk.shape[1] == LAG_WIDTH, blk.shape
    sl = slice(LAG_FIRST, LAG_FIRST + LAG_WIDTH)
    changed = float((np.abs(mr.x[:, LAG_FIRST] - blk[:, 0]) > 1e-6).mean())
    # only values change: missing flags, ages, and grid indices (columns 38, 39, 41, 42, 43, 44) stay as 574 built them
    for old, new in ((38, 1), (39, 2), (41, 4), (42, 5), (43, 6), (44, 7)):
        assert np.array_equal(mr.x[:, old], blk[:, new]), (month, old)
    mr.x[:, sl] = blk
    # the targets are the clean power at (Primary, target time); shift them where the schedule says
    target_clean = clean[mr.primary, mr.target_idx]
    assert np.allclose(mr.y, target_clean, atol=1e-5, equal_nan=True), (
        "targets are not the clean power"
    )
    d = delta[mr.primary, mr.target_idx]
    d = np.where(np.isfinite(d), d, 0.0)
    mr.y = (mr.y + d).astype(mr.y.dtype)
    share_t = float((d != 0).mean())
    print(
        f"{ARM} {month}: share of rows with a changed day lag {changed:.4f}, "
        f"share with a shifted target {share_t:.4f}, mean |target shift| on shifted rows "
        f"{float(np.abs(d[d != 0]).mean()) if share_t else 0.0:.3f} | {json.dumps(info)[:200]}",
        flush=True,
    )
    e.CHECK["months"].setdefault(month, {})["arm_variant"] = {
        "arm": ARM,
        "share_changed_day_lag": changed,
        "share_shifted_target": share_t,
        "info": info,
    }
    if ARM == "t1":
        assert share_t > 0.02, share_t
    return mr


e.k.build_month = build_month_variant


def _out(out_dir, month, r, day, seed):
    return out_dir / f"{month}_r{r:02d}_d{day}_s{seed}.parquet"


def _done(*, out) -> bool:
    return out.exists() or out.with_suffix(".empty").exists()


def run_inject(*, device: str = "cuda") -> None:
    """Forecast every pass of the injected bank with the saved weights; one parquet per cell."""
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
        # the training-month rows are only needed for the standardiser and target moments
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
            print(f"{ARM} inject {month} r{r:02d} d{day} s{seed}", flush=True)
        del mr, test, models
        torch.cuda.empty_cache()


if __name__ == "__main__":
    if CMD == "assemble":
        e.assemble()
    elif CMD == "inject":
        run_inject()
    else:
        e.fit_arm(months=e.ARGS.months, seeds=e.ARGS.seeds, stress=False, limit=None, device="cuda")
        out = WORK / f"checks_fit_{ARM}_s{'-'.join(map(str, e.ARGS.seeds))}.json"
        out.write_text(json.dumps(e.CHECK, indent=1, default=str))
        print("CHECKS PASSED", {a: v for a, v in e.CHECK.items() if a != "months"})
