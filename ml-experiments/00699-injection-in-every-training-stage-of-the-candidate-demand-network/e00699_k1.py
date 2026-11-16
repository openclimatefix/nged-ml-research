# ruff: noqa: ANN001, ANN003, ANN201, ANN202, D103, E402, E501, I001, PT018
"""Experiment 699: pre-train (stage 1) and train step 2 (stage 2) of the step-5 network with shifted lags (GPU).

A copy of experiment 677's `e00677_k1.py`, restricted to the commands `pretrain` and `step2` of
ladder step 5, with two options:

- `--inject-stage1 1`: the lag columns of the ERA5 and CAMS pre-training rows come from the
  cleaned power plus experiment 581's synthetic load shifts (the world seed is 5,000,000 plus the
  fold month as a number), as experiment 603's arm J1 did (`pretrain_rows_marking` and
  `block_of_pre` of `e00603_arm.py` are copied with a new seed and an added check);
- `--inject-stage2 1`: the lag columns of the stage-2 rows carry the synthetic shifts, through the
  same builder that stage 3 uses (`e00591_arm.build_month_syn`), with 2,000,000 added to the world
  seed so that stage 2 draws its own world. Stage 3 keeps the seed of experiment 677.

The shifts reach the lag value columns (37 and 40) only. The training targets, the lag ages, and
the lag-time columns come from the unshifted power. The script writes `injection.json` beside the
chain's weights with the world seeds, the number of events, the share of rows with a shifted lag,
the last half-hour any world changed, and a hash of the initial weights of each seed.

Usage: python e00699_k1.py k1 {pretrain|step2} --step 5 [--inject-stage1 0|1] [--inject-stage2 0|1]
[--months M] [--seeds ...]. Run it through `gpu699.sh` with `E00677_PHASE=train`, `E00677_ROOT`,
`E00677_MONTH`, `CHAIN677`, and `STATS677` set, as `chain699.sh` does.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

_p = argparse.ArgumentParser()
_p.add_argument("name")
_p.add_argument("command", choices=("pretrain", "step2"))
_p.add_argument("--step", choices=("5",), default="5")
_p.add_argument("--inject-stage1", type=int, default=0)
_p.add_argument("--inject-stage2", type=int, default=0)
_p.add_argument("--months", nargs="+", required=True)
_p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
MINE = _p.parse_args()
assert len(MINE.months) == 1, "one process per fold month"
assert not (MINE.command == "pretrain" and MINE.inject_stage2)
assert not (MINE.command == "step2" and MINE.inject_stage1)
sys.argv = [
    "e00591_arm.py", "A1", MINE.command, "--months", *MINE.months, "--seeds", *map(str, MINE.seeds),
]  # fmt: skip

import e00591_arm as arm  # ty: ignore[unresolved-import]
import e00581_bank as bank  # ty: ignore[unresolved-import]
import numpy as np
import torch  # ty: ignore[unresolved-import]

import e00677_fold as fold  # ty: ignore[unresolved-import]

MONTH = MINE.months[0]
assert list(MINE.months) == [fold.FINAL_MONTH], "one process per fold month: set E00677_MONTH"
assert fold.TRAIN_ONLY, "stage 1 and stage 2 are fitted with the held-out power hidden"
MONTH_NUMBER = int(MONTH.replace("-", ""))
STAGE_SEED_BASE = {"stage1": 5_000_000, "stage2": 2_000_000}
SHARE_RANGE = (0.04, 0.10)

e = arm.e
assert arm.LAMBDA == 0.0, "A1 has no auxiliary loss"
assert e.EMBARGO_DAYS == 3, "the base embargo is experiment 574's 3 days"
CHAIN_ENV = os.environ.get("CHAIN677")
assert CHAIN_ENV, "set CHAIN677 to the chain's work folder"
CHAIN = Path(CHAIN_ENV)
EXPERIMENT_ROOT = Path(os.environ["E00677_ROOT"]).resolve()
assert "00699" in str(EXPERIMENT_ROOT) and "roots" in EXPERIMENT_ROOT.parts, EXPERIMENT_ROOT
assert str(CHAIN.resolve().parent.parent.parent.parent) == str(EXPERIMENT_ROOT), (
    "the chain folder must sit inside this experiment's own root"
)
arm.CHAIN = CHAIN
fold.install(e, months=MINE.months)
INJECTION_FILE = CHAIN / "injection.json"


def record(*, key: str, info: dict) -> None:
    """Merge `info` under `key` in `injection.json`, atomically."""
    current = json.loads(INJECTION_FILE.read_text()) if INJECTION_FILE.exists() else {}
    current[key] = info
    tmp = INJECTION_FILE.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(current, indent=1, default=str))
    tmp.rename(INJECTION_FILE)


def world_extent(*, clean: np.ndarray, fixed: np.ndarray) -> dict:
    """Return the first and last half-hour at which the world changed any Primary's power."""
    changed = np.isfinite(fixed) & np.isfinite(clean) & (fixed != clean)
    columns = np.nonzero(changed.any(axis=0))[0]
    assert len(columns), "the world changed nothing"
    return {"first_changed_idx": int(columns[0]), "last_changed_idx": int(columns[-1])}


# the final-fold statistics file: pre-training derives the B1 statistics into STATS677
def _final_stats_file(*, month: str):
    return Path(os.environ.get("STATS677", CHAIN)) / "stats" / e.SPEC.b1 / f"{month}.npz"


e.stats_file = _final_stats_file
arm.check_stats = lambda: None
e.WORK, e.W = CHAIN.parent, CHAIN
GRID_CUT = int(
    (e.c.us(t=e.c.month_start(month=MONTH)) - e.c.prep422.GRID_START_US) // e.c.prep422.STEP_US
)
"""Grid index of the fold's cutoff: no world may change a half-hour at or after it."""

INIT: dict[int, dict] = {}
_new_model = e.new_model


def new_model_recording(*, n_primaries: int, stage: str = "ft"):
    """Build the network and record a hash of its initial weights (the seed was set just before)."""
    model = _new_model(n_primaries=n_primaries, stage=stage)
    if stage == "pre":
        digest = hashlib.sha256()
        for key, value in sorted(model.state_dict().items()):
            digest.update(key.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
        INIT[int(torch.initial_seed())] = {
            "sha256": digest.hexdigest(), "n_primaries": n_primaries, "n_knots": e.KNOTS, "n_b1": e.N_B1,
        }  # fmt: skip
    return model


if MINE.command == "pretrain":
    e.new_model = new_model_recording
    _block_of, _pretrain_rows = e.block_of, e.r448.pretrain_rows
    PRE_STATE = {"flag": False}

    def pretrain_rows_marking(**kw):
        PRE_STATE["flag"] = True
        return _pretrain_rows(**kw)

    def block_of_pre(*, clean, primary, target_idx, origin_idx, extra_days=0):
        """Return the pre-training rows' lag block; with `--inject-stage1`, from power with synthetic shifts."""
        if not (PRE_STATE["flag"] and MINE.inject_stage1):
            PRE_STATE["flag"] = False
            return _block_of(
                clean=clean, primary=primary, target_idx=target_idx, origin_idx=origin_idx,
                extra_days=extra_days,
            )  # fmt: skip
        PRE_STATE["flag"] = False
        seed = STAGE_SEED_BASE["stage1"] + MONTH_NUMBER
        first, last = int(target_idx.min()), e.fit_until(month=MONTH)
        fixed, n_events = bank.synthetic_clean(clean=clean, first=first, last=last, seed=seed)
        extent = world_extent(clean=clean, fixed=fixed)
        assert extent["last_changed_idx"] < GRID_CUT, (extent, GRID_CUT)
        kw = dict(  # noqa: C408
            primary=primary, target_idx=target_idx, origin_idx=origin_idx, extra_days=extra_days
        )
        blk = _block_of(clean=fixed, **kw)
        base = _block_of(clean=clean, **kw)
        assert blk.shape == base.shape and blk.shape[1] == 8, blk.shape
        # the shifts move the lag values only: ages, missing flags, and lag times are unchanged
        assert np.array_equal(
            blk[:, [1, 2, 4, 5, 6, 7]], base[:, [1, 2, 4, 5, 6, 7]], equal_nan=True
        )
        differs = (np.abs(blk[:, 0] - base[:, 0]) > 1e-6) | (np.abs(blk[:, 3] - base[:, 3]) > 1e-6)
        share = float(differs.mean())
        print(
            f"{MONTH}: stage 1 world seed {seed}: {n_events} synthetic events; share of rows with a shifted lag {share:.4f}; last changed half-hour {extent['last_changed_idx']} (cutoff {GRID_CUT})",
            flush=True,
        )
        record(
            key="stage1",
            info={
                "seed": seed, "n_events": n_events, "share": share, "rows": len(blk), "first": first,
                "last": last, "cutoff_idx": GRID_CUT, **extent, "target_unchanged": True,
                "lag_times_unchanged": True,
            },
        )  # fmt: skip
        del base
        return blk

    e.r448.pretrain_rows = pretrain_rows_marking
    e.block_of = block_of_pre
    e.pretrain_wt(device="cuda")
    current = json.loads(INJECTION_FILE.read_text()) if INJECTION_FILE.exists() else {}
    record(
        key="init_weights",
        info={**current.get("init_weights", {}), **{str(s): v for s, v in INIT.items()}},
    )
else:
    STAGE2: dict = {}
    if MINE.inject_stage2:
        _synthetic_clean = bank.synthetic_clean

        def synthetic_clean_stage2(*, clean, first, last, seed):
            """Return experiment 581's world with its own seed: stage 3's month number plus 2,000,000."""
            assert seed == MONTH_NUMBER, f"stage 3's builder passes the month number, got {seed}"
            own = seed + STAGE_SEED_BASE["stage2"]
            fixed, n_events = _synthetic_clean(clean=clean, first=first, last=last, seed=own)
            extent = world_extent(clean=clean, fixed=fixed)
            assert extent["last_changed_idx"] < GRID_CUT, (extent, GRID_CUT)
            STAGE2.update(
                seed=own, n_events=n_events, first=first, last=last, cutoff_idx=GRID_CUT, **extent
            )
            return fixed, n_events

        bank.synthetic_clean = synthetic_clean_stage2
        _build_month_syn = arm.build_month_syn

        def build_month_recording(*, month, clean):
            """Return the stage-2 rows with shifted lags and record the share of rows with a shifted lag."""
            mr = _build_month_syn(month=month, clean=clean)
            labels = arm.AUX[id(mr)]
            STAGE2["share"] = float((np.nan_to_num(labels) != 0).any(axis=1).mean())
            STAGE2["rows_built"] = len(mr.y)
            return mr

        e.row_nll = arm.row_nll
        e.k.build_month = build_month_recording
        e.k.GpuMonth = arm.AuxGpuMonth
    e.step2(months=e.ARGS.months, seeds=e.ARGS.seeds, device="cuda")
    if MINE.inject_stage2 and STAGE2:  # empty when every seed's stage-2 file already existed
        assert SHARE_RANGE[0] <= STAGE2["share"] <= SHARE_RANGE[1], STAGE2
        record(key="stage2", info={**STAGE2, "target_unchanged": True, "lag_times_unchanged": True})
        print(f"{MONTH}: stage 2 world {STAGE2}", flush=True)
torch.cuda.empty_cache()
