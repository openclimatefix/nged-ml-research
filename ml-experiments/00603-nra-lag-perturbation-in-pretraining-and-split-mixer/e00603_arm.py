# ruff: noqa: E501, ANN001, ANN201, ANN003, E402, PT018, B023, F821, ANN202, D103, SIM115, PTH123, PERF401, UP037, F841, RUF100
"""Experiment 603: pre-train, step 2, fine-tune, and inject arms J1 to J5 (GPU).

A wrapper around experiment 574's `e00574_fit.py` (arm P5), in the style of experiment 584's
`e00584_arm.py`. Reused unchanged: the rows, recency weights, early stopping, seeds, scoring
windows, origin-time weather assertions, refit days 0 and 14, the embargoed daily fine-tune, and
experiment 584's synthetic lag-only shifts (`e00581_bank.synthetic_clean`, same seed rule).
What differs, per arm: whether the pre-training lag block and the step-2 lag columns carry synthetic
shifts, and whether the mixer is split (`e00603_model`).

Usage: python e00603_arm.py COMMAND ARM [--months ...] [--seeds ...] [--passes ...]
COMMAND: `pretrain`, `step2`, `fit`, `assemble`, or `inject`. ARM: J1, J2, J3, J3c, J4, J4b, J5.
The fine-tune (step 3) always carries synthetic shifts. Pre-training weights are shared by arms with
the same `PRE` key (J1 and J2).
"""

import argparse
import dataclasses
import json
import sys
from typing import Final

ARMS: Final[dict[str, dict]] = {
    # kind None = unsplit mixer; pre/s2 = whether pre-training / step 2 carry synthetic shifts
    "J1": {"kind": None, "pre": True, "s2": False},
    "J2": {"kind": None, "pre": True, "s2": True},
    "J3": {"kind": "j3", "pre": True, "s2": True},
    "J3c": {"kind": "j3", "pre": False, "s2": False},
    "J4": {"kind": "j4", "pre": True, "s2": True},
    "J4b": {"kind": "j4b", "pre": True, "s2": True},
    "J5": {"kind": "j5", "pre": True, "s2": True},
}
_p = argparse.ArgumentParser()
_p.add_argument("command", choices=("pretrain", "step2", "fit", "assemble", "inject"))
_p.add_argument("arm", choices=tuple(ARMS))
_p.add_argument("--months", nargs="+", default=None)
_p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
_p.add_argument("--passes", nargs="+", type=int, default=None)
MINE = _p.parse_args()
CMD, ARM = MINE.command, MINE.arm
A = ARMS[ARM]
_argv = ["e00574_fit.py", {"inject": "fit", "pretrain": "pretrain-wt"}.get(CMD, CMD), "--arm", "P5"]
if MINE.months:
    _argv += ["--months", *MINE.months]
_argv += ["--seeds", *map(str, MINE.seeds)]
sys.argv = _argv

import e00574_fit as e  # ty: ignore[unresolved-import]
import e00581_bank as bank  # ty: ignore[unresolved-import]
import e00603_model as model603
import numpy as np
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00603-nra-lag-perturbation-in-pretraining-and-split-mixer"
WORK = ML_EXPERIMENTS_DATA_DIR / NAME
W574 = e.W
PRE_KEY = f"{A['kind'] or 'unsplit'}_{'pert' if A['pre'] else 'clean'}"
"""Pre-training weights are shared by arms with the same key (J1 and J2)."""
STATS_DIR = WORK / "stats"
e.WORK = WORK / ARM
e.W = e.WORK / "work"
assert e.SPEC.tag == "P5" and e.SPEC.sched == "3sE" and e.SPEC.lags == "24h"
LAG_FIRST, LAG_WIDTH = 37, 8


def pretrained_dir():
    return WORK / "pretrained" / PRE_KEY


def pretrained_file(*, month: str, seed: int):
    p = pretrained_dir() / f"{month}_s{seed}.pt"
    assert p.exists(), p
    return p


def step2_file(*, month: str, seed: int):
    return e.W / "step2" / f"{month}_s{seed}.pt"


def stats_file(*, month: str):
    """B1 statistics are fitted on pre-training weather rows, which the perturbation does not touch.

    Once the shared file exists, a later pre-training writes its own copy elsewhere, so a
    concurrent process never reads a half-written shared file.
    """
    p = STATS_DIR / f"{month}.npz"
    if CMD == "pretrain" and p.exists():
        return e.W / "stats_scratch" / f"{month}.npz"
    return p


e.pretrained_dir, e.pretrained_file, e.step2_file, e.stats_file = (
    pretrained_dir, pretrained_file, step2_file, stats_file,
)  # fmt: skip

# ---- the split mixer ----
if A["kind"] is not None:

    def new_model(*, n_primaries: int, stage: str = "ft"):
        mdl = model603.SplitMdn(
            n_primaries=n_primaries, stage=stage, n_knots=e.KNOTS, kind=A["kind"], n_b1=e.N_B1
        )
        mdl.table = e.CTX.get(f"table_{stage}")
        return mdl

    def fresh_mixer(*, model):
        """Give stage 2 (`mixer_ft`) a seed-12345 initialisation (one hidden layer)."""
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(e.FRESH_SEED)
            fresh = e.base.mlp(
                n_in=model.mixer_ft[0].in_features, hidden=e.m.MIXER_HIDDEN, layers=1,
                n_out=3 * e.m.COMPONENTS, dropout=e.m.DROPOUT,
            )  # fmt: skip
            e.base.init_head(head=fresh, components=e.m.COMPONENTS)
        model.mixer_ft.load_state_dict(fresh.state_dict())

    _train = e.train

    def train_freezing_stage1(**kw):
        """Freeze stage 1 in steps 2 and 3 (the `ft` network); it is learnt in pre-training."""
        if kw["model"].stage == "ft":
            kw["model"].mixer1.freeze()
        return _train(**kw)

    e.new_model, e.fresh_mixer, e.train = new_model, fresh_mixer, train_freezing_stage1

# ---- lag perturbation ----
_build_month_p5 = e.k.build_month


def build_month_syn(*, month, clean):
    """Build the month's P5 rows with the 8 lag columns read from power carrying synthetic events (584's rule)."""
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
    for old, new in ((38, 1), (39, 2), (41, 4), (42, 5), (43, 6), (44, 7)):
        assert np.array_equal(mr.x[:, old], blk[:, new]), (month, old)
    mr.x[:, LAG_FIRST : LAG_FIRST + LAG_WIDTH] = blk
    print(f"{month}: {n_events} synthetic events in the training rows", flush=True)
    return mr


# the fine-tune (step 3) always perturbs; step 2 only for arms with s2
if CMD == "step2":
    e.k.build_month = build_month_syn if A["s2"] else _build_month_p5
elif CMD in {"fit", "inject"}:
    e.k.build_month = build_month_syn

# pre-training: the lag block comes from power with synthetic events over [0, refit)
_block_of, _pretrain_rows = e.block_of, e.r448.pretrain_rows
PRE_STATE = {"flag": False, "month": ""}


def pretrain_rows_marking(**kw):
    PRE_STATE["flag"] = True
    return _pretrain_rows(**kw)


def block_of_pre(*, clean, primary, target_idx, origin_idx, extra_days=0):
    if PRE_STATE["flag"] and A["pre"]:
        PRE_STATE["flag"] = False
        month = PRE_STATE["month"]
        fixed, n_events = bank.synthetic_clean(
            clean=clean, first=int(target_idx.min()), last=e.fit_until(month=month),
            seed=int(str(month).replace("-", "")) + 603,
        )  # fmt: skip
        blk_check = _block_of(
            clean=fixed, primary=primary[:500_000], target_idx=target_idx[:500_000],
            origin_idx=origin_idx[:500_000], extra_days=extra_days,
        )  # fmt: skip
        blk_clean = _block_of(
            clean=clean, primary=primary[:500_000], target_idx=target_idx[:500_000],
            origin_idx=origin_idx[:500_000], extra_days=extra_days,
        )  # fmt: skip
        both = np.isfinite(blk_check[:, 0]) & np.isfinite(blk_clean[:, 0])
        changed = float((np.abs(blk_check[both, 0] - blk_clean[both, 0]) > 1e-6).mean())
        print(
            f"{month}: {n_events} synthetic events in the pre-training power; share of rows with a changed day lag {changed:.4f}",
            flush=True,
        )
        clean = fixed
    PRE_STATE["flag"] = False
    return _block_of(
        clean=clean, primary=primary, target_idx=target_idx, origin_idx=origin_idx,
        extra_days=extra_days,
    )  # fmt: skip


if CMD == "pretrain":
    e.r448.pretrain_rows = pretrain_rows_marking
    e.block_of = block_of_pre
    _pre = e.pretrain_wt

    def pretrain_per_month(*, device: str):
        """Run `pretrain_wt` one month at a time so the perturbation seed knows the month."""
        months = list(e.ARGS.months)
        for month in months:
            PRE_STATE["month"] = month
            e.ARGS.months = [month]
            _pre(device=device)
        e.ARGS.months = months

    e.pretrain_per_month = pretrain_per_month


# ---- injection (584's, with this arm's weights) ----
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
    elif CMD == "pretrain":
        e.pretrain_per_month(device="cuda")
    elif CMD == "step2":
        e.step2(months=e.ARGS.months, seeds=e.ARGS.seeds, device="cuda")
    else:
        e.fit_arm(months=e.ARGS.months, seeds=e.ARGS.seeds, stress=False, limit=None, device="cuda")
        out = e.WORK / f"checks_fit_{ARM}_s{'-'.join(map(str, e.ARGS.seeds))}.json"
        out.write_text(json.dumps(e.CHECK, indent=1, default=str))
        print("CHECKS PASSED", {a: v for a, v in e.CHECK.items() if a != "months"})
