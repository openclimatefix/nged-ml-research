"""G1 cross-fitted trust fit with 12 folds instead of 6, one scored month at a time.

Reuses experiment 463's fit loop (`e00463_crossfit`) with the fold count raised from 6 to 12.
The 12-fold lagless table is rebuilt into this experiment's own folder, so experiment 404's frozen
6-fold table (`work/lagless`) is never written. There is no 12-fold G1-insample control.

```bash
PYTHONPATH=ml-experiments/00463-lag-trust-cross-fitted-trust-fit-seed-zero:\
ml-experiments/00397-plain-and-attention-mixture-density-network-for-primary-demand:\
ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\
ml-experiments/00048-weekly-retraining-confirmation-and-extension-demand-generators:\
ml-experiments/00404-mdn-lagless-and-calibration-module-trust-weights \
DATA_PATH_INTERNAL=<the internal data folder> OMP_NUM_THREADS=1 \
    systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00469-*/e00469_crossfit12.py lagless --month 2026-06
```

Commands: `lagless` builds the 12-fold table for a month; `time-fold` fits one fold network and
prints its seconds; `fit` fits the 12 fold networks and the trust head, and writes the forecast.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Final

import e00404_lagless as lagless  # ty: ignore[unresolved-import]
import e00404_models as models  # ty: ignore[unresolved-import]
import e00463_crossfit as base  # ty: ignore[unresolved-import]
import numpy as np
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

FOLDS: Final[int] = 12
WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00469-lag-trust-cross-fitted-trust-fit-twelve-folds"
)
LAGLESS_DIR: Final[Path] = WORK_DIR / "lagless_12fold"
FROZEN_LAGLESS_DIR: Final[Path] = lagless.WORK_DIR


def _redirect() -> None:
    """Point experiment 463's and 404's folds, tables and outputs at this experiment's folder."""
    if LAGLESS_DIR == FROZEN_LAGLESS_DIR:
        msg = "refusing to write over the frozen 6-fold lagless table"
        raise RuntimeError(msg)
    lagless.WORK_DIR = LAGLESS_DIR  # ty: ignore[invalid-assignment]
    base.FOLDS = FOLDS  # ty: ignore[invalid-assignment]
    base.WORK_DIR = WORK_DIR  # ty: ignore[invalid-assignment]
    base.PRED_DIR = WORK_DIR / "predictions"  # ty: ignore[invalid-assignment]
    base.OOF_DIR = WORK_DIR / "oof"  # ty: ignore[invalid-assignment]
    base.LOG_DIR = WORK_DIR / "logs"  # ty: ignore[invalid-assignment]


def build_lagless(*, month: str, seed: int, device: str) -> None:
    """Build and store the 12-fold lagless table for a scored month."""
    started = time.monotonic()
    table = lagless.crossfit_table(month=month, seed=seed, device=device, folds=FOLDS)
    target = lagless.table_path(month=month, seed=seed)
    if FROZEN_LAGLESS_DIR in target.parents:
        msg = f"{target} is inside the frozen 6-fold folder"
        raise RuntimeError(msg)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    table.write_parquet(temporary)
    temporary.rename(target)
    seconds = time.monotonic() - started
    print(f"{month} seed {seed}: {table.height:,} rows, {seconds:.0f} s -> {target}")


def time_fold(*, month: str, seed: int, device: str) -> None:
    """Fit one held-out fold network and print the seconds it takes."""
    ctx = base.build_context(month=month, seed=seed, device=device)
    fit_mask = ctx.block != 0
    mean, std = base.target_scalers(train=ctx.train, mask=fit_mask)
    started = time.monotonic()
    base.fit_plain(ctx=ctx, mask=fit_mask, mean=mean, std=std, seed=seed, device=device)
    seconds = time.monotonic() - started
    print(f"{month}: one fold fit on {int(fit_mask.sum()):,} rows in {seconds:.1f} s")


def run_month(*, month: str, seed: int, device: str) -> None:
    """Fit the 12 fold networks and the trust head for a month, and write the G1-12 forecast."""
    started = time.monotonic()
    ctx = base.build_context(month=month, seed=seed, device=device)
    print(f"{month}: {len(ctx.train):,} training rows, {FOLDS} folds")
    base.check_test_rows(month=month, seed=seed, ctx=ctx, device=device)
    base.OOF_DIR.mkdir(parents=True, exist_ok=True)
    base.LOG_DIR.mkdir(parents=True, exist_ok=True)
    cache = base.OOF_DIR / f"oof_{month}_s{seed}.npy"
    if cache.exists():
        oof = torch.tensor(np.load(cache), device=device)
        fold_log = json.loads((base.OOF_DIR / f"oof_{month}_s{seed}.json").read_text())
    else:
        oof, fold_log = base.out_of_fold_heads(ctx=ctx, seed=seed, device=device)
        np.save(cache, oof.cpu().numpy().astype(np.float32))
        (base.OOF_DIR / f"oof_{month}_s{seed}.json").write_text(json.dumps(fold_log))
    base.set_trust_statistics(ctx=ctx, seed=seed, device=device)
    trust_started = time.monotonic()
    trust_net, trust_losses = base.fit_trust(ctx=ctx, oof=oof, seed=seed, device=device)
    trust_seconds = time.monotonic() - trust_started
    lw, lm, ls = models.mixture_parts(out=ctx.train.ll, components=models.COMPONENTS)
    diagnostics = {
        "month": month, "seed": seed, "rows": len(ctx.train), "folds": fold_log,
        "trust_losses": trust_losses, "trust_seconds": trust_seconds,
        "nll_lagged_out_of_fold": base.mean_nll(ctx=ctx, mixture=oof, device=device),
        "nll_lagless_cross_fitted": base.mean_nll(
            ctx=ctx, mixture=torch.cat([lw, lm, ls], dim=1), device=device
        ),
    }  # fmt: skip
    frame = base.forecast_month(month=month, seed=seed, trust_net=trust_net, ctx=ctx, device=device)
    base.check_keys(month=month, seed=seed, frame=frame)
    arm_dir = base.PRED_DIR / "G1-trust-12"
    arm_dir.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(arm_dir / f"{month}_s{seed}_w-1.parquet")
    torch.save(trust_net.state_dict(), base.OOF_DIR / f"G1-trust-12_{month}_s{seed}.pt")
    diagnostics["seconds"] = time.monotonic() - started
    (base.LOG_DIR / f"{month}_s{seed}.json").write_text(json.dumps(diagnostics, indent=2))
    print(json.dumps({k: v for k, v in diagnostics.items() if k != "folds"}, indent=2))


def main() -> None:
    """Run one command for one scored month."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("lagless", "time-fold", "fit"))
    parser.add_argument("--month", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    _redirect()
    if args.command == "lagless":
        build_lagless(month=args.month, seed=args.seed, device=args.device)
    elif args.command == "time-fold":
        time_fold(month=args.month, seed=args.seed, device=args.device)
    else:
        run_month(month=args.month, seed=args.seed, device=args.device)


if __name__ == "__main__":
    main()
