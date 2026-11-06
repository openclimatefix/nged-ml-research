# ruff: noqa: E501, ANN001, E402, PT018, D103, B023, F821, ANN202
"""Experiment 586: fit, assemble, stress, and inject lag-only events for the lag-wiring arms S1 to S4 (GPU).

A wrapper around experiment 572's `e00572_fit.py` (imported, with its module-level state patched, as
experiments 578 and 581 did): the same rows, recency weights, early stopping, seeds, scoring windows,
frozen-encoder D5 fine-tune, origin-time weather assertions, and refit days 0 and 14. Only the
network differs (`e00586_model`) and where the files go. Arm S0 is experiment 572's stored v3+D5.

Usage: python e00586_arm.py KIND COMMAND [--months ...] [--seeds ...] [--passes ...]

- KIND: `s1` (shared encoder plus learnt lag ID), `s2` (shared, no identity), `s3` (separate
  encoders), `s4` (shared plus a fixed period encoding).
- COMMAND: `pretrain`, `fit`, `stress` (the injected 7-day shift of experiment 572, seed 0),
  `assemble`, or `inject` (forecast with lag-only contaminated lags using the saved weights).
"""

import argparse
import dataclasses
import json
import sys

_p = argparse.ArgumentParser()
_p.add_argument("kind", choices=("s1", "s2", "s3", "s4"))
_p.add_argument("command", choices=("pretrain", "fit", "stress", "assemble", "inject"))
_p.add_argument("--months", nargs="+", default=None)
_p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
_p.add_argument("--passes", nargs="+", type=int, default=None)
MINE = _p.parse_args()
KIND, CMD = MINE.kind, MINE.command
_argv = [
    "e00572_fit.py",
    {"pretrain": "pretrain-wt", "inject": "fit", "stress": "fit"}.get(CMD, CMD),
    "--arm",
    "V3D5",
]
if CMD == "stress":
    _argv += ["--stress"]
if MINE.months:
    _argv += ["--months", *MINE.months]
_argv += ["--seeds", *map(str, MINE.seeds)]
sys.argv = _argv

import e00572_fit as e  # ty: ignore[unresolved-import]
import e00581_bank as bank  # ty: ignore[unresolved-import]
import e00586_model as mymodel
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00586-lag-encoder-sharing-and-lag-id-embedding"
WORK = ML_EXPERIMENTS_DATA_DIR / NAME
e.WORK = WORK
e.W = WORK / "arms" / KIND / "work"
assert e.SPEC.variant == "lagoff" and e.SPEC.frozen


def new_model(*, n_primaries: int, stage: str = "ft") -> torch.nn.Module:
    model = mymodel.SharedLagOffsetMdn(
        n_primaries=n_primaries, stage=stage, n_knots=e.KNOTS, kind=KIND
    )
    model.table = e.CTX.get(f"table_{stage}")  # ty: ignore[invalid-assignment]
    return model


e.new_model = new_model


def run_inject(*, device: str = "cuda") -> None:
    """Forecast every pass of the injected bank with the arm's saved weights; one parquet per cell."""
    clean = e.c.clean_stack()
    passes = MINE.passes if MINE.passes is not None else [*range(bank.N_PASSES), bank.REAL_PASS]
    rows: list[dict] = []
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
        mr = e.k.build_month(month=month, clean=clean)
        test = e.data.build_test(month=month, standardiser=mr.base.standardiser)
        e.CTX["variants"] = {}
        start = e.c.month_start(month=month)
        start_idx = (e.c.us(t=start) - e.c.prep422.GRID_START_US) // e.c.prep422.STEP_US
        tau_from = e.N_LAG_INPUTS
        assert test.x.shape[1] == tau_from + 1, test.x.shape

        def x_for(r: int) -> np.ndarray:
            """Return the test inputs with pass `r`'s contaminated lag columns (built on demand)."""
            if r == bank.REAL_PASS:
                fixed = bank.corrected_clean(clean=clean, events=real_events())
            else:
                fixed, _ = bank.contaminate(clean=clean, month_start_idx=start_idx, r=r)
            blk = e.block_of(
                clean=fixed,
                primary=test.primary,
                target_idx=test.target_idx,
                origin_idx=test.origin_idx,
            )
            x = test.x.copy()
            x[:, 37:tau_from] = blk
            assert np.array_equal(x[:, 43:45], test.x[:, 43:45]), "contamination moved a lag time"
            return x

        for r in [p for p in passes if p != bank.REAL_PASS]:
            rows += [
                {"month": month, **q}
                for q in bank.contaminate(clean=clean, month_start_idx=start_idx, r=r)[1]
            ]
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
            frame = e.f.window_frame(
                model=models[(i, seed)], test=dataclasses.replace(test, x=x_for(r)), mr=mr,
                lo=lo, hi=hi, flag=False, device=device,
            )  # fmt: skip
            target = _out(out_dir, month, r, day, seed)
            if frame is None:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.with_suffix(".empty").touch()
            else:
                e._atomic_frame(frame=frame, target=target)
            print(f"{KIND} inject {month} r{r} d{day} s{seed}", flush=True)
        del mr, test, models
        torch.cuda.empty_cache()
    if rows:
        WORK.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(rows).write_parquet(
            WORK / f"bank_events_{KIND}_{'-'.join(e.ARGS.months)}.parquet"
        )


def real_events() -> list[dict]:
    """Return the verified real events (experiment 580) as per-Primary rows."""
    sm = pl.read_csv(
        ML_EXPERIMENTS_DATA_DIR
        / "00521-lag-advantage-by-lead-time-k2-s1-vs-daily-lag-free/site_map.csv"
    )
    table = pl.read_parquet(
        ML_EXPERIMENTS_DATA_DIR
        / "00580-verified-load-shift-event-set-and-label-audit/work/events_verified.parquet"
    )
    return bank.verified_events(
        capacity_mw=dict(zip(sm["label"].to_list(), sm["capacity_mw"].to_list(), strict=True)),
        label_to_primary=dict(zip(sm["label"].to_list(), sm["primary"].to_list(), strict=True)),
        table=table,
    )


def _out(out_dir, month, r, day, seed):
    return out_dir / f"{month}_r{r:02d}_d{day}_s{seed}.parquet"


def _done(*, out) -> bool:
    return out.exists() or out.with_suffix(".empty").exists()


if __name__ == "__main__":
    if CMD == "assemble":
        e.assemble()
    elif CMD == "pretrain":
        e.pretrain_wt(device="cuda")
    elif CMD == "inject":
        run_inject()
    else:
        stress = CMD == "stress"
        if stress and list((e.W / "models" / e.SPEC.tag).glob("*_stress*.pt")):
            raise SystemExit(
                "stress models exist; delete them and their fits first (see experiment 572)"
            )
        e.fit_arm(
            months=e.ARGS.months, seeds=e.ARGS.seeds, stress=stress, limit=None, device="cuda"
        )
        out = (
            WORK
            / f"checks_fit_{KIND}{'_stress' if stress else ''}_s{'-'.join(map(str, e.ARGS.seeds))}.json"
        )
        out.write_text(json.dumps(e.CHECK, indent=1, default=str))
        print("CHECKS PASSED", {a: v for a, v in e.CHECK.items() if a != "months"})
