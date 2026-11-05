# ruff: noqa: E501, ANN001, ANN201, E402, PT018, D103, B023, F821, ANN202
"""Experiment 585: fit, assemble, and inject lag-only events for the lag-count and year-ago arms (GPU).

A wrapper around experiment 572's `e00572_fit.py` (imported, with its module-level state patched, as
experiments 578 and 581 did): the same rows, recency weights, early stopping, seeds, scoring windows,
frozen-encoder D5 fine-tune, origin-time weather assertions, and refit days 0 and 14. What differs is
the lag block (`e00585_lags`), the lag encoder (`e00585_model`), the block dropout of the extra
lags, and where the files go.

Usage: python e00585_arm.py KIND COMMAND [--months ...] [--seeds ...] [--passes ...]

- KIND: `n1` (week lag only), `n3` (day, week, and a second weekly lag, 14 days before the target's newest same weekday), `y1` (day, week, and 364-day
  lags), `y1c` (as y1 at 365 days, the same calendar date), `y3` (day, week, and 357-, 364-, and 371-day lags), or `y1mask` (Y1's saved weights, with
  the year-ago lag masked inside verified load-shift events; only `fit` is meaningful, and it loads
  Y1's weights instead of training).
- COMMAND: `pretrain`, `fit`, `assemble`, or `inject` (forecast with lag-only contaminated lags).
"""

import argparse
import dataclasses
import json
import sys

_p = argparse.ArgumentParser()
_p.add_argument("kind", choices=("n1", "n3", "y1", "y1c", "y3", "y1mask"))
_p.add_argument("command", choices=("pretrain", "fit", "assemble", "inject"))
_p.add_argument("--months", nargs="+", default=None)
_p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
_p.add_argument("--passes", nargs="+", type=int, default=None)
MINE = _p.parse_args()
KIND, CMD = MINE.kind, MINE.command
_argv = [
    "e00572_fit.py",
    {"pretrain": "pretrain-wt", "inject": "fit"}.get(CMD, CMD),
    "--arm",
    "V3D5",
]
if MINE.months:
    _argv += ["--months", *MINE.months]
_argv += ["--seeds", *map(str, MINE.seeds)]
sys.argv = _argv

import e00572_fit as e  # ty: ignore[unresolved-import]
import e00581_bank as bank  # ty: ignore[unresolved-import]
import e00585_lags as mylags
import e00585_model as mymodel
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00585-lag-count-and-year-ago-lags-through-the-lag-encoder"
WORK = ML_EXPERIMENTS_DATA_DIR / NAME
WEIGHTS_KIND = "y1" if KIND == "y1mask" else KIND
e.WORK = WORK
e.W = WORK / "arms" / WEIGHTS_KIND / "work"
if KIND == "y1mask":
    OUT_W = WORK / "arms" / "y1mask" / "work"

PERIODS = {"n1": (), "n3": (-7,), "y1": (364,), "y1c": (365,), "y3": (357, 364, 371), "y1mask": (364,)}
mylags.set_periods(periods_days=PERIODS[KIND], drop_day=KIND == "n1")
N_EXTRA = len(PERIODS[KIND])
e.lg = mylags
e.N_LAG_INPUTS = 37 + mylags.n_block()
assert e.SPEC.variant == "lagoff" and e.SPEC.frozen
_orig_dropout = e.apply_dropout


def apply_dropout(*, x, origin_idx, seed) -> dict:
    """Experiment 572's block dropout, plus an independent whole-origin-day mask for each extra lag."""
    share = _orig_dropout(x=x, origin_idx=origin_idx, seed=seed)
    amax = mylags.EXTRA_AMAX
    for k in range(N_EXTRA):
        mask_all, _ = e.lags.block_dropout_masks(origin_idx=origin_idx, seed=seed + 1000 * (k + 1))
        m = torch.from_numpy(mask_all).to(x.device) | torch.from_numpy(
            e.lags.block_dropout_masks(origin_idx=origin_idx, seed=seed)[0]
        ).to(x.device)
        c = mylags.extra_value_col(k)
        x[m, c], x[m, c + 1], x[m, c + 2] = 0.0, 1.0, float(amax[k])
        share[f"share_extra{k}_missing"] = float((x[:, c + 1] > 0.5).float().mean())
    return share


def new_model(*, n_primaries: int, stage: str = "ft") -> torch.nn.Module:
    model = mymodel.MultiLagOffsetMdn(
        n_primaries=n_primaries,
        stage=stage,
        n_knots=e.KNOTS,
        n_extra=N_EXTRA,
        drop_day=KIND == "n1",
    )
    model.table = e.CTX.get(f"table_{stage}")  # ty: ignore[invalid-assignment]
    return model


e.apply_dropout, e.new_model = apply_dropout, new_model


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


def masked_extra_clean(*, clean: np.ndarray) -> np.ndarray:
    """Return a copy of `clean` with every verified-event half-hour set to NaN (for the extra lags only)."""
    out = clean.copy()
    for q in real_events():
        out[q["primary"], q["onset_idx"] : q["end_idx"]] = np.nan
    return out


def _out(out_dir, month, r, day, seed):
    return out_dir / f"{month}_r{r:02d}_d{day}_s{seed}.parquet"


def _done(*, out) -> bool:
    return out.exists() or out.with_suffix(".empty").exists()


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


if __name__ == "__main__":
    if CMD == "assemble":
        e.assemble()
    elif CMD == "pretrain":
        e.pretrain_wt(device="cuda")
    elif CMD == "inject":
        run_inject()
    else:
        e.fit_arm(months=e.ARGS.months, seeds=e.ARGS.seeds, stress=False, limit=None, device="cuda")
        out = WORK / f"checks_fit_{KIND}_s{'-'.join(map(str, e.ARGS.seeds))}.json"
        out.write_text(json.dumps(e.CHECK, indent=1, default=str))
        print("CHECKS PASSED", {a: v for a, v in e.CHECK.items() if a != "months"})
