# ruff: noqa: E501, ANN001, ANN201, E402, PT018, B023, F821, ANN202
"""Experiment 594: fit and inject the two placebo arms PS and PP (GPU), variants of experiment 584's arm C1.

Experiment 584's arm C1 (= T0 here) fine-tunes experiment 574's P5 recipe on lag inputs that carry
synthetic load-shifts on 5% of each Primary's time, with clean targets. This wrapper (copied from
experiment 590's `e00590_arm.py`) keeps C1's event schedule exactly (the same start, length, and depth
of every event, drawn from the same random generator in the same order) and changes only what the
lags read inside each event, by patching `build_month` as experiments 584 and 590 do. Rows, recency
weights, early stopping, seeds, scoring windows, origin-time weather assertions, refit days 0 and
14, and P5's stored step-2 weights and B1 statistics (experiment 574's work folder, read only) are
unchanged. The test rows are untouched. Targets stay clean in both arms.

Arms (`--arm`):

- `ps` (P-share): inside each event the lags read the power plus independent Gaussian noise with
  mean 0 and standard deviation equal to the event's depth (0.15, 0.30, or 0.50 of capacity), one
  draw per half-hour. Same rows, same magnitude, sign-symmetric, not persistent.
- `pp` (P-persist): inside each event the lags read the power plus a shift of the event's depth whose
  sign is drawn independently for every block of one day (48 half-hours). A shift persists for a
  day but the event has no sustained sign, so its net shift averages about zero.
- `p5`: not used here; experiment 590's stored `p5` and `t3` arms are read in place.

The standardiser and target moments stay those of the clean power (as for C1). Where the clean power
is missing the lag stays missing.

Usage: python e00594_arm.py COMMAND --arm {ps,pp} [--months ...] [--seeds ...] [--passes ...]
COMMAND: `fit`, `assemble`, or `inject` (forecast the injected bank with the saved weights).
"""

import argparse
import dataclasses
import json
import sys

_p = argparse.ArgumentParser()
_p.add_argument("command", choices=("fit", "assemble", "inject"))
_p.add_argument("--arm", required=True, choices=("ps", "pp"))
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

NAME = "00594-persistent-and-share-matched-placebos-for-synthetic-load-shifts"
WORK = ML_EXPERIMENTS_DATA_DIR / NAME / ARM
W574 = e.W
"""Experiment 574's work folder: P5's step-2 weights and B1 statistics are read from here, never written."""
e.WORK = WORK
e.W = WORK / "work"
assert e.SPEC.tag == "P5" and e.SPEC.sched == "3sE" and e.SPEC.lags == "24h"

LAG_FIRST, LAG_WIDTH = 37, 8
NOISE_SEED_OFFSET = 104_729
"""Added to the month's schedule seed for the placebo noise and sign draws (separate generator)."""
BLOCK = bank.PER_DAY
"""Half-hours per sign block in arm PP."""


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


def placebo_series(
    *, clean: np.ndarray, first: int, last: int, seed: int, kind: str, share: float = bank.SYN_SHARE
) -> tuple[np.ndarray, dict]:
    """Return `clean` with a placebo in place of each synthetic event of experiment 581's schedule.

    The schedule generator `rng` is consumed exactly as in `e00581_bank.synthetic_clean` (length,
    start, depth, sign, in that order), so event positions, lengths, and depths equal arm C1's. The
    shift sign drawn from `rng` is discarded and the placebo is drawn from `noise_rng`, which is
    separate. `kind="ps"` adds N(0, depth^2) noise per half-hour; `kind="pp"` adds
    `depth * s` where `s` is +1 or -1 drawn independently per block of `BLOCK` half-hours.
    """
    assert kind in ("ps", "pp"), kind
    rng = np.random.default_rng(seed)
    noise_rng = np.random.default_rng(seed + NOISE_SEED_OFFSET)
    out = clean.copy()
    events = 0
    net = []
    for p in range(clean.shape[0]):
        covered = 0
        while covered < share * (last - first):
            length = int(rng.choice(bank.LENGTHS_DAYS)) * bank.PER_DAY
            on = int(rng.integers(first, max(first + 1, last - length)))
            depth = float(rng.choice(bank.DEPTHS))
            rng.choice([-1.0, 1.0])  # C1's sign draw, kept so the schedule matches
            if kind == "ps":
                delta = noise_rng.standard_normal(length) * depth
            else:
                signs = noise_rng.choice([-1.0, 1.0], size=-(-length // BLOCK))
                delta = np.repeat(signs, BLOCK)[:length] * depth
            seg = out[p, on : on + length]
            out[p, on : on + length] = np.where(np.isfinite(seg), seg + delta, seg)
            net.append(abs(float(delta.mean())) / depth)
            covered += length
            events += 1
    return out, {"events": events, "mean_abs_net_shift_over_depth": round(float(np.mean(net)), 4)}


def check_schedule(*, clean: np.ndarray, first: int, last: int, seed: int) -> None:
    """Assert that the placebo schedule has C1's event positions: the changed lags coincide in both."""
    ref, n_events = bank.synthetic_clean(clean=clean, first=first, last=last, seed=seed)
    mine, info = placebo_series(clean=clean, first=first, last=last, seed=seed, kind=ARM)
    assert info["events"] == n_events, (info, n_events)
    c1 = np.isfinite(clean) & (ref != clean)
    ours = np.isfinite(clean) & (mine != clean)
    assert (c1 & ~ours).mean() < 1e-3, "placebo leaves C1 events unperturbed"
    assert (ours & ~c1).mean() < 1e-3, "placebo perturbs outside C1 events"


def build_month_variant(*, month, clean):
    """Build the month's P5 rows, then change the 8 lag columns and the targets as the arm says."""
    mr = _build_month_p5(month=month, clean=clean)
    origin_idx = (mr.origin_us - e.c.prep422.GRID_START_US) // e.c.prep422.STEP_US
    first, last = int(mr.target_idx.min()), int(mr.target_idx.max())
    seed = int(month.replace("-", ""))
    if month == e.ARGS.months[0]:
        check_schedule(clean=clean, first=first, last=last, seed=seed)
    lag_series, info = placebo_series(
        clean=clean, first=first, last=last, seed=seed, kind=ARM
    )
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
    # the targets are the clean power at (Primary, target time) and stay clean in both placebo arms
    target_clean = clean[mr.primary, mr.target_idx]
    assert np.allclose(mr.y, target_clean, atol=1e-5, equal_nan=True), (
        "targets are not the clean power"
    )
    lag_delta = np.where(np.isfinite(clean), lag_series - clean, 0.0)
    print(
        f"{ARM} {month}: share of rows with a changed day lag {changed:.4f}, "
        f"mean |lag change| over the month {float(np.abs(lag_delta[:, first:last]).mean()):.4f} "
        f"| {json.dumps(info)}",
        flush=True,
    )
    e.CHECK["months"].setdefault(month, {})["arm_variant"] = {
        "arm": ARM,
        "share_changed_day_lag": changed,
        "info": info,
    }
    assert 0.02 < changed < 0.20, changed
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
