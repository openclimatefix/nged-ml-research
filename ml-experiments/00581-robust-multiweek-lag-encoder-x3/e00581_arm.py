# ruff: noqa: E501, ANN001, ANN201, E402, PT018, D103, B023, F821, ANN202
"""Experiment 581: fit, assemble, and inject lag-only events for the X3 arms (GPU).

A wrapper around experiment 572's `e00572_fit.py` (imported, with its module-level state patched, as
experiment 578 did): the same rows, recency weights, early stopping, seeds, scoring windows,
frozen-encoder D5 fine-tune, origin-time weather assertions, and refit days 0 and 14. What differs
is the lag block (`e00581_lags`), the lag encoder (`e00581_model`), and where the files go.

Usage: python e00581_arm.py KIND COMMAND [--months ...] [--seeds ...] [--passes ...]

- KIND: `x3syn` / `v3syn` (X3-med / v3+D5 fine-tuned with lag-only synthetic load-shift events on 5% of the
  time; they reuse the pre-trained weights of `x3med` / `v3d5`), `x3med` (5-week median and spread), `placebo` (the same encoder on weeks 6 to 10),
  `x335` (X3-med with the newest reading 3.5 h before the origin), or `v3d5` (experiment 572's stored
  weights; only `inject` is meaningful).
- COMMAND: `pretrain`, `fit`, `assemble`, or `inject` (forecast with lag-only contaminated lags,
  using the saved fine-tuned weights; one parquet per month, pass, refit day, and seed).
"""

import argparse
import dataclasses
import json
import sys

_p = argparse.ArgumentParser()
_p.add_argument("kind", choices=("x3med", "placebo", "x335", "v3d5", "x3syn", "v3syn"))
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
import e00581_bank as bank
import e00581_lags as mylags
import e00581_model as mymodel
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00581-robust-multiweek-lag-encoder-x3"
WORK = ML_EXPERIMENTS_DATA_DIR / NAME
W572 = e.W
W582 = ML_EXPERIMENTS_DATA_DIR / "00582-nra-tails-on-verified-load-shift-events/work"
"""Experiment 582's work folder: its v3+D5 weights for the extra fold months are reused, never refitted."""
e.WORK = WORK
e.W = WORK / "arms" / KIND / "work"

ROBUST_KINDS = ("x3med", "placebo", "x335", "x3syn")
SYN_KINDS = ("x3syn", "v3syn")
if KIND in ROBUST_KINDS:
    mylags.WEEK_OFFSET = 5 if KIND == "placebo" else 0
    e.lg = mylags
    e.N_LAG_INPUTS = 37 + mylags.N_BLOCK
    if KIND == "x335":
        e.SPEC = dataclasses.replace(e.SPEC, lags="3.5h")
        e.CAP_HOURS = e.SPEC.cap_steps * 0.5
    assert e.SPEC.variant == "lagoff" and e.SPEC.frozen
    _orig_dropout = e.apply_dropout

    def apply_dropout(*, x, origin_idx, seed) -> dict:
        """Experiment 572's block dropout, plus the robust columns zeroed where their lags are masked."""
        share = _orig_dropout(x=x, origin_idx=origin_idx, seed=seed)
        mask_all, mask_day = e.lags.block_dropout_masks(origin_idx=origin_idx, seed=seed)
        ma = torch.from_numpy(mask_all).to(x.device)
        md = torch.from_numpy(mask_day).to(x.device) | ma
        f0 = mylags.EXTRA_FIRST
        for c in (f0 + 1, f0 + 2, f0 + 3, f0 + 5):
            x[md, c] = 0.0
        for c in (f0, f0 + 4, f0 + 6):
            x[ma, c] = 0.0
        return share

    def new_model(*, n_primaries: int, stage: str = "ft") -> torch.nn.Module:
        model = mymodel.RobustLagOffsetMdn(n_primaries=n_primaries, stage=stage, n_knots=e.KNOTS)
        model.table = e.CTX.get(f"table_{stage}")  # ty: ignore[invalid-assignment]
        return model

    e.apply_dropout, e.new_model = apply_dropout, new_model
if KIND == "v3d5":
    _model_file = e.model_file

    def model_file(*, tag: str, month: str, day: int, seed: int, sfx: str = ""):
        """Return experiment 572's saved weights when they exist, else this experiment's own path (extra folds)."""
        f = W572 / "models" / tag / f"{month}_d{day}_s{seed}{sfx}.pt"
        g = W582 / "models" / tag / f"{month}_d{day}_s{seed}{sfx}.pt"
        return (
            f
            if f.exists()
            else g
            if g.exists()
            else _model_file(tag=tag, month=month, day=day, seed=seed, sfx=sfx)
        )

    e.model_file = model_file


if KIND in SYN_KINDS:
    BASE_PRETRAINED = (
        W572 if KIND == "v3syn" else WORK / "arms" / "x3med" / "work"
    ) / "pretrained_V3D5"
    e.pretrained_dir = lambda: BASE_PRETRAINED
    _build_month_syn = e.k.build_month

    def build_month_syn(*, month, clean):
        """Build the month's rows, then replace the lag columns by ones read from power with synthetic events.

        Only the lag inputs change; the targets stay clean. The schedule is seeded by the month only.
        """
        mr = _build_month_syn(month=month, clean=clean)
        origin_idx = (mr.origin_us - e.c.prep422.GRID_START_US) // e.c.prep422.STEP_US
        first, last = int(mr.target_idx.min()), int(mr.target_idx.max())
        fixed, n_events = bank.synthetic_clean(
            clean=clean, first=first, last=last, seed=int(month.replace("-", ""))
        )
        blk = e.block_of(
            clean=fixed, primary=mr.primary, target_idx=mr.target_idx, origin_idx=origin_idx
        )
        before = mr.x[:, 37 : 37 + blk.shape[1]]
        changed = float((np.abs(before[:, 0] - blk[:, 0]) > 1e-6).mean())
        mr.x[:, 37 : 37 + blk.shape[1]] = blk
        print(
            f"{month}: {n_events} synthetic events; share of rows with a changed day lag {changed:.4f}",
            flush=True,
        )
        return mr

    e.k.build_month = build_month_syn


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
        e.fit_arm(months=e.ARGS.months, seeds=e.ARGS.seeds, stress=False, limit=None, device="cuda")
        out = WORK / f"checks_fit_{KIND}_s{'-'.join(map(str, e.ARGS.seeds))}.json"
        out.write_text(json.dumps(e.CHECK, indent=1, default=str))
        print("CHECKS PASSED", {a: v for a, v in e.CHECK.items() if a != "months"})
