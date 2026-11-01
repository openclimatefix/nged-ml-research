r"""Fine-tune experiment 425's arm A network with hard-row weights (experiment 484).

Each fit loads experiment 425's stored pretrained weights for the fold and seed
(`00425-.../work/pretrained/A/<month>_s<seed>.pt`, never rewritten), then runs experiment 425's
fine-tune (`e00425_fit.fine_tune`: learning rate 2e-4, summary scheme, setting `g09`) on the same
rows, with each row's training weight multiplied by the arm's weight (`e00484_weights`). The
held-out-week validation loss that stops the fit is unweighted. The arm `W0` (weight 1 everywhere)
is only the reproduction check: one (month, seed) refitted to prove the code path matches
experiment 425's stored fit; the reference scores read experiment 425's stored fits.

Outputs go under this experiment's `work/` folder: `seed_preds/<arm>/<month>_s<seed>.parquet` (raw
quantiles as fractions of capacity, the layout of experiment 425), `models/<arm>/<month>/s<seed>`,
and `logs/<month>.json` (epochs, validation NLL, seconds, and the share of rows up-weighted).

```bash
source /tmp/e484env.sh   # DATA_PATH_INTERNAL, PYTHONPATH with every ml-experiments folder
systemd-run --user --scope -p MemoryMax=12G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00484-*/e00484_fit.py --months 2026-01 --seeds 0 \
    --arms W0
```
"""

import argparse
import json
import time
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00397_fit as fit397  # ty: ignore[unresolved-import]
import e00397_prep as base  # ty: ignore[unresolved-import]
import e00425_fit as fit425  # ty: ignore[unresolved-import]
import e00425_prep as prep425  # ty: ignore[unresolved-import]
import e00484_weights as weights  # ty: ignore[unresolved-import]
import numpy as np
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_DIR_NAME: Final[str] = "00484-hard-row-weighting-in-demand-network-fine-tuning"
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / EXPERIMENT_DIR_NAME / "work"
FOLDS: Final[tuple[str, ...]] = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
"""The five folds of experiment 449."""
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
ARMS: Final[tuple[str, ...]] = ("W0", *weights.WEIGHTED_ARMS)
A_ARM: Final[fit425.Arm] = fit425.ARMS["A"]


def pretrained_path(*, month: str, seed: int) -> Path:
    """Return experiment 425's stored pretrained weights (read only)."""
    return fit425.OUT_DIR / "pretrained" / "A" / f"{month}_s{seed}.pt"


def final_path(*, arm: str, month: str, seed: int) -> Path:
    """Return where an arm's seed-level raw quantile forecast is stored."""
    return WORK / "seed_preds" / arm / f"{month}_s{seed}.parquet"


def run(*, months: list[str], arms: list[str], seeds: list[int], device: str) -> None:
    """Fine-tune each arm on each fold and seed that is not yet stored."""
    for month in months:
        todo = [
            (arm, seed)
            for arm in arms
            for seed in seeds
            if not final_path(arm=arm, month=month, seed=seed).exists()
        ]
        if not todo:
            continue
        refit = base.refit_of(month=month)
        fold = prep425.build_ens_fold(refit=refit)
        log_path = WORK / "logs" / f"{month}.json"
        try:
            log: dict = json.loads(log_path.read_text()) if log_path.exists() else {}
        except json.JSONDecodeError:
            log = {}
        log |= {"month": month, "fine_tune_rows": len(fold.x)}
        print(f"fold {month}: {len(fold.x):,} fine-tuning rows", flush=True)
        train_mask = ~fold.is_val
        for arm, seed in todo:
            multiplier = (
                np.ones(len(fold.x), dtype=np.float32)
                if arm == "W0"
                else weights.row_multiplier(
                    arm=arm, valid_us=fold.valid, primary=fold.primary, y=fold.y, work_dir=WORK
                )
            )
            share = float((multiplier[train_mask] > 1.0).mean())
            effective = float(
                (fold.weight[train_mask] * multiplier[train_mask]).sum()
                / fold.weight[train_mask].sum()
            )
            t = fit425.tensors(
                x=fold.x,
                y=fold.y,
                primary=fold.primary,
                weight=fold.weight * multiplier,
                is_val=fold.is_val,
                fold=fold,
                device=device,
            )
            started = time.monotonic()
            torch.manual_seed(seed)
            model = fit425.new_model(arm=A_ARM, fold=fold).to(device)
            model.load_state_dict(torch.load(pretrained_path(month=month, seed=seed)))
            setting = fit425.replace(fit425.SETTING, lr=fit425.FINE_TUNE_LR)
            epochs, best = fit425.train(model=model, setting=setting, t=t, seed=seed)
            fitted = fit397.Fitted(
                model=model,
                setting=setting,
                arm="plain_pooled",
                seed=seed,
                epochs=epochs,
                val_nll=best,
            )
            fit397.save_fits(
                fits={None: fitted},
                fold=fold,
                target=WORK / "models" / arm / month / f"s{seed}",
            )
            q, _ = fit397.forecast_month(
                fits={None: fitted}, fold=fold, scheme="summary", month=month, device=device
            )
            final = final_path(arm=arm, month=month, seed=seed)
            final.parent.mkdir(parents=True, exist_ok=True)
            common.atomic_write(frame=q, target=final)
            seconds = round(time.monotonic() - started)
            log[f"{arm}_s{seed}"] = {
                "epochs": epochs,
                "val_nll": best,
                "seconds": seconds,
                "up_weighted_share": share,
                "mean_weight_ratio": effective,
            }
            print(
                f"{arm} {month} seed {seed}: {epochs} epochs, val NLL {best:.4f}, {seconds} s, "
                f"up-weighted share {share:.3f}, mean weight ratio {effective:.3f}",
                flush=True,
            )
            log_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = log_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(log, indent=1))
            tmp.replace(log_path)
            del t, model, fitted
            torch.cuda.empty_cache()


def main() -> None:
    """Fine-tune the named arms on the named folds and seeds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", nargs="+", default=list(FOLDS))
    parser.add_argument(
        "--arms", nargs="+", choices=list(ARMS), default=list(weights.WEIGHTED_ARMS)
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(months=args.months, arms=args.arms, seeds=args.seeds, device=args.device)


if __name__ == "__main__":
    main()
