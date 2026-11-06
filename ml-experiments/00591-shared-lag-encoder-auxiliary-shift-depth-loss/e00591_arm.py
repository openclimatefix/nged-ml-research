# ruff: noqa: E501, ANN001, ANN003, ANN201, E402, PT018, B023, F821, ANN202, D102, D103
"""Experiment 591: fit the shared-lag-encoder arms with an auxiliary depth loss (GPU).

A wrapper around experiment 574's `e00574_fit.py` (imported with its module-level state patched, as
experiment 584 did): the same rows, recency weights, early stopping, seeds, scoring windows,
origin-time weather assertions, refit days 0 and 14, the embargoed three-step schedule with the
thermal-lag and solar-gain columns (arm P5). Only the network differs (`e00591_model`), plus the
auxiliary loss in the daily fine-tune on rows carrying experiment 581's synthetic lag-only
load-shifts (as experiment 584's C1).

Usage: python e00591_arm.py ARM COMMAND [--months ...] [--seeds ...] [--passes ...]

- ARM: `A1` (shared, no auxiliary loss), `A2` (lambda from `calibrate`), `A3` (3 x A2's lambda), `A4`
  (separate encoders, A2's lambda), `A5` (A2 plus the detector input; starts from A2's step 2).
- COMMAND: `pretrain` and `step2` (the arm's chain: shared for A1, A2, A3, A5; separate for A4),
  `calibrate` (A2's lambda, one-batch gradient probe), `fit`, `assemble`, `inject`.
"""

import argparse
import dataclasses
import json
import sys

_p = argparse.ArgumentParser()
_p.add_argument("arm", choices=("A1", "A2", "A3", "A4", "A5"))
_p.add_argument("command", choices=("pretrain", "step2", "calibrate", "fit", "assemble", "inject"))
_p.add_argument("--months", nargs="+", default=None)
_p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
_p.add_argument("--passes", nargs="+", type=int, default=None)
MINE = _p.parse_args()
ARM, CMD = MINE.arm, MINE.command
_argv = [
    "e00574_fit.py",
    {"pretrain": "pretrain-wt", "step2": "step2", "assemble": "assemble"}.get(CMD, "fit"),
    "--arm",
    "P5",
]
if MINE.months:
    _argv += ["--months", *MINE.months]
_argv += ["--seeds", *map(str, MINE.seeds)]
sys.argv = _argv

import e00574_fit as e  # ty: ignore[unresolved-import]
import e00581_bank as bank  # ty: ignore[unresolved-import]
import e00591_model as mymodel
import numpy as np
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00591-shared-lag-encoder-auxiliary-shift-depth-loss"
ROOT = ML_EXPERIMENTS_DATA_DIR / NAME
KIND = "separate" if ARM == "A4" else "shared"
CHAIN = ROOT / f"chain_{KIND}" / "work"
"""Pre-training and step-2 weights of the arm's chain (A1, A2, A3, and A5 share one; A4 has its own)."""
LAMBDA_FILE = ROOT / "lambda.json"
W574 = e.W
"""Experiment 574's work folder: the fold's stored B1 statistics are read from here, never written."""
LAM_MULT = {"A1": 0.0, "A2": 1.0, "A3": 3.0, "A4": 1.0, "A5": 1.0}
LAMBDA = 0.0
if CMD in {"fit", "inject"} and ARM != "A1":
    LAMBDA = LAM_MULT[ARM] * json.loads(LAMBDA_FILE.read_text())["lambda"]
if CMD in {"pretrain", "step2"}:
    e.WORK, e.W = CHAIN.parent, CHAIN
else:
    e.WORK = ROOT / ARM
    e.W = e.WORK / "work"
assert e.SPEC.tag == "P5" and e.SPEC.sched == "3sE" and e.SPEC.lags == "24h" and e.SPEC.frozen


def _step2_file(*, month: str, seed: int):
    """Return the chain's step-2 weights (A5 reads the shared chain, which A2 also uses)."""
    base = CHAIN if CMD not in {"pretrain", "step2"} else e.W
    return base / "step2" / e.SPEC.tag / f"{month}_s{seed}.pt"


def _stats_file(*, month: str):
    """Return the fold's B1 statistics: experiment 574's stored file, read only.

    Pre-training re-derives the statistics and writes them to the chain folder instead (never over
    experiment 574's file); `check_stats` then asserts the two files are identical.
    """
    stored = W574 / "stats" / e.SPEC.b1 / f"{month}.npz"
    assert stored.exists(), stored
    return CHAIN / "stats" / e.SPEC.b1 / f"{month}.npz" if CMD == "pretrain" else stored


def check_stats() -> None:
    """Assert that every statistics file written by pre-training equals experiment 574's."""
    for mine in sorted((CHAIN / "stats" / e.SPEC.b1).glob("*.npz")):
        a, b = np.load(mine), np.load(W574 / "stats" / e.SPEC.b1 / mine.name)
        assert np.array_equal(a["mean"], b["mean"]) and np.array_equal(a["std"], b["std"]), mine


e.step2_file, e.stats_file = _step2_file, _stats_file


def new_model(*, n_primaries: int, stage: str = "ft") -> torch.nn.Module:
    model = mymodel.AuxLagMdn(
        n_primaries=n_primaries, stage=stage, n_knots=e.KNOTS, n_b1=e.N_B1, kind=KIND,
        detector=ARM == "A5",
    )  # fmt: skip
    model.table = e.CTX.get(f"table_{stage}")  # ty: ignore[invalid-assignment]
    return model


e.new_model = new_model
LAG_FIRST, LAG_WIDTH = 37, 8
_build_month_p5 = e.k.build_month
AUX: dict[int, np.ndarray] = {}
"""Per month-rows object: the (rows, 2) depth labels of the day and week lag (NaN: no label)."""


def build_month_syn(*, month, clean):
    """Build the month's P5 rows with synthetic lag-only load-shifts in the lag columns, and the depth labels.

    As experiment 584's builder; in addition the label of a row's lag is the shift that
    `synthetic_clean` added at that lag's grid time (zero where none), NaN where the lag is missing.
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
    for old, new in ((38, 1), (39, 2), (41, 4), (42, 5), (43, 6), (44, 7)):
        assert np.array_equal(mr.x[:, old], blk[:, new]), (month, old)
    mr.x[:, LAG_FIRST : LAG_FIRST + LAG_WIDTH] = blk
    shift = np.where(np.isfinite(fixed) & np.isfinite(clean), fixed - clean, 0.0).astype(np.float32)
    lab = np.full((len(mr.primary), 2), np.nan, np.float32)
    for j, miss_col in enumerate((38, 41)):
        idx = mr.x[:, 43 + j].astype(np.int64)
        ok = (idx >= 0) & (mr.x[:, miss_col] < 0.5)
        lab[ok, j] = shift[mr.primary[ok], idx[ok]]
    AUX[id(mr)] = lab
    share = float((np.nan_to_num(lab) != 0).any(axis=1).mean())
    print(
        f"{month}: {n_events} synthetic events; share of rows with a shifted lag {share:.4f}",
        flush=True,
    )
    return mr


class AuxGpuMonth(e.LagGpuMonth):
    """The month's rows on the GPU; the fine-tuning tensors get the two depth-label columns appended."""

    def refit_tensors(self, *, refit, shift_z=None):
        t = super().refit_tensors(refit=refit, shift_z=shift_z)  # applies block dropout
        lab = torch.from_numpy(AUX[id(self.mr)][refit.index]).to(self.device)
        x = t["x"]
        masked = torch.stack([x[:, 38] > 0.5, x[:, 41] > 0.5], dim=1)  # lags that dropout hid
        lab = torch.where(masked, torch.full_like(lab, float("nan")), lab)
        t["x"] = torch.cat([x, lab], dim=1)
        return t


def row_nll(*, out, z, p):
    """The mixture negative log-likelihood, plus lambda times the auxiliary squared error (training only)."""
    nll = e.base.mixture_nll(out=out, z=z, components=e.m.COMPONENTS)
    aux = mymodel.STASH.pop("aux", None)
    if aux is not None and LAMBDA > 0:
        nll = nll + LAMBDA * aux
    return nll


def calibrate_train(*, model, x, z, w, p, is_val, seed, **_):
    """Stand-in for `train` that measures the gradient norms of the two losses on the lag encoder and stops."""
    torch.manual_seed(seed)
    model.train()
    for part in (model.weather, model.datetime):
        part.eval()
    params = [q for n, q in model.named_parameters() if n.startswith(("lag.", "lag_id."))]
    train_idx = torch.nonzero(~is_val).squeeze(1)
    rows = []
    for _i in range(20):
        b = train_idx[torch.randint(len(train_idx), (e.c.BATCH,), device=x.device)]
        nll = e.base.mixture_nll(out=model(x[b], p[b]), z=z[b], components=e.m.COMPONENTS)
        aux = mymodel.STASH.pop("aux")
        l_nll = (w[b] * nll).sum() / w[b].sum()
        l_aux = (w[b] * aux).sum() / w[b].sum()
        g_nll = torch.autograd.grad(l_nll, params, retain_graph=True, allow_unused=True)
        g_aux = torch.autograd.grad(l_aux, params, allow_unused=True)
        n1 = float(torch.sqrt(sum((g**2).sum() for g in g_nll if g is not None)))
        n2 = float(torch.sqrt(sum((g**2).sum() for g in g_aux if g is not None)))
        rows.append(
            {"grad_norm_nll": n1, "grad_norm_aux": n2, "nll": float(l_nll), "aux_mse": float(l_aux)}
        )
    lam = float(
        np.mean([r["grad_norm_nll"] for r in rows]) / np.mean([r["grad_norm_aux"] for r in rows])
    )
    LAMBDA_FILE.parent.mkdir(parents=True, exist_ok=True)
    LAMBDA_FILE.write_text(json.dumps({"lambda": lam, "batches": rows}, indent=1))
    print("lambda", lam, rows[:3], flush=True)
    return 0, 0.0


def _out(out_dir, month, r, day, seed):
    return out_dir / f"{month}_r{r:02d}_d{day}_s{seed}.parquet"


def _done(*, out) -> bool:
    return out.exists() or out.with_suffix(".empty").exists()


def run_inject(*, device: str = "cuda") -> None:
    """Forecast every pass of the injected bank with the saved weights; one parquet per cell (as experiment 584)."""
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
    elif CMD == "pretrain":
        e.pretrain_wt(device="cuda")
        check_stats()
    elif CMD == "step2":
        e.step2(months=e.ARGS.months, seeds=e.ARGS.seeds, device="cuda")
    elif CMD == "inject":
        run_inject()
    else:
        e.row_nll = row_nll
        e.k.build_month = build_month_syn
        e.k.GpuMonth = AuxGpuMonth
        if CMD == "calibrate":
            assert ARM == "A2"
            e.train = calibrate_train
            months = e.ARGS.months or ["2025-10"]
            clean = e.c.clean_stack()
            mr = e.k.build_month(month=months[0], clean=clean)
            g = e.k.GpuMonth(mr=mr, device="cuda")
            cut = e.c.month_start(month=months[0])
            e.fine_tune(g=g, month=months[0], cut=cut, seed=e.ARGS.seeds[0], device="cuda")
        else:
            e.fit_arm(
                months=e.ARGS.months, seeds=e.ARGS.seeds, stress=False, limit=None, device="cuda"
            )
            out = e.WORK / f"checks_fit_{ARM}_s{'-'.join(map(str, e.ARGS.seeds))}.json"
            out.write_text(json.dumps(e.CHECK, indent=1, default=str))
            print("CHECKS PASSED", {a: v for a, v in e.CHECK.items() if a != "months"})
