# ruff: noqa: E501
r"""Fit the assembled pooled mixture density network of experiment 425 and its two comparison arms.

Every arm is experiment 397's pooled plain network (summary scheme, setting `g09`) on the rows of
`e00425_prep` (the D15 and D18 start cut and the flagged exact-zero drop), standardised per Primary:

- `R`: trained on the ENS rows only (the 8-wide Primary embedding is in);
- `A`: pretrained on ERA5-and-CAMS rows, then every weight fine-tuned on the ENS rows at a learning
  rate of 2e-4 (experiment 418's arm P1);
- `Anoemb`: `A` built on experiment 406's `pooled_noemb` network (no Primary embedding of any kind)
  in both the pretraining and the fine-tuning stage.

Each fit writes the raw quantiles only (fraction of capacity); the scorer widens them by 1.15. The
fitted weights go under `work/models/` and `work/pretrained/` of this experiment's data folder.
`--no-cut` skips both cuts, writes under `<arm>_nocut`, and exists only for the 3-fit cross-check
of the plan section "Arms" (seed 0, three months).

```bash
source <(printf 'export PYTHONPATH=%s\n' \
  ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\
ml-experiments/00048-weekly-retraining-confirmation-and-extension-demand-generators:\
ml-experiments/00377-demand-v1-first-run-feasibility:\
ml-experiments/00381-simple-weekly-xgboost-demand-against-gam-stack:\
ml-experiments/00397-plain-and-attention-mixture-density-network-for-primary-demand:\
ml-experiments/00407-pooled-mixture-density-network-tail-arms-and-recency-weights:\
ml-experiments/00418-mdn-pretrain-on-era5-and-cams-then-fine-tune-on-ens:\
ml-experiments/00425-best-mdn-architecture-assembly-with-ablations)
export DATA_PATH_INTERNAL=<path to the private data folder>
OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00425-*/e00425_fit.py --device cuda
```
"""

import argparse
import json
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00397_fit as fit397  # ty: ignore[unresolved-import]
import e00397_prep as base  # ty: ignore[unresolved-import]
import e00425_prep as prep  # ty: ignore[unresolved-import]
import numpy as np
import torch  # ty: ignore[unresolved-import]

SETTING: Final[fit397.Setting] = fit397.setting_by_name(name="g09")
FINE_TUNE_LR: Final[float] = 2e-4
DEFAULT_SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
OUT_DIR: Path = prep.WORK_DIR / "work"
"""Replaced by `work/smoke` under `--smoke`."""


@dataclass(frozen=True)
class Arm:
    """One arm: whether the network has a Primary embedding, and whether it is pretrained."""

    name: str
    embedding: bool
    pretrained: bool


ARMS: Final[dict[str, Arm]] = {
    a.name: a
    for a in (
        Arm("R", embedding=True, pretrained=False),
        Arm("A", embedding=True, pretrained=True),
        Arm("Anoemb", embedding=False, pretrained=True),
    )
}


def folder_name(*, arm: str, cut: bool) -> str:
    """Return the folder of an arm's outputs (`R`, or `R_nocut` for the cross-check)."""
    return arm if cut else f"{arm}_nocut"


def new_model(*, arm: Arm, fold: prep.EnsFold) -> torch.nn.Module:
    """Return an untrained pooled plain network for a fold, with or without the Primary embedding."""
    return fit397.make_model(
        arm="plain_pooled" if arm.embedding else "plain_own",
        setting=SETTING,
        layout=fold.layout,
        n_primaries=len(fold.target_mean),
        n_inputs=fold.x.shape[1],
    )


def tensors(
    *,
    x: np.ndarray,
    y: np.ndarray,
    primary: np.ndarray,
    weight: np.ndarray,
    is_val: np.ndarray,
    fold: prep.EnsFold,
    device: str,
) -> dict[str, torch.Tensor]:
    """Return the training tensors on the device, with the target standardised per Primary."""
    z = (y - fold.target_mean[primary]) / fold.target_std[primary]
    return {
        "x": torch.from_numpy(x).to(device),
        "z": torch.from_numpy(z.astype(np.float32)).to(device),
        "w": torch.from_numpy(weight.astype(np.float32)).to(device),
        "p": torch.from_numpy(primary).to(device),
        "v": torch.from_numpy(is_val).to(device),
    }


def train(
    *, model: torch.nn.Module, setting: fit397.Setting, t: dict[str, torch.Tensor], seed: int
) -> tuple[int, float]:
    """Train a network in place with experiment 397's loop; return epochs and best NLL."""
    _, epochs, best = fit397.train_network(
        model=model,
        setting=setting,
        x=t["x"],
        z=t["z"],
        w=t["w"],
        primary=t["p"],
        is_val=t["v"],
        seed=seed,
    )
    return epochs, best


def pretrained_path(*, arm: Arm, cut: bool, month: str, seed: int) -> Path:
    """Return where a pretrained network's weights are stored."""
    return OUT_DIR / "pretrained" / folder_name(arm=arm.name, cut=cut) / f"{month}_s{seed}.pt"


def atomic_save(*, state: dict, target: Path) -> None:
    """Write a state dict to a temporary file, then rename it into place."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    torch.save(state, temporary)
    temporary.rename(target)


def pretrain(
    *,
    arms: list[Arm],
    fold: prep.EnsFold,
    refit: datetime,
    month: str,
    seeds: list[int],
    cut: bool,
    device: str,
    log: dict,
) -> None:
    """Pretrain and store one network per pretrained arm and seed (equal row weights)."""
    todo = [
        (arm, seed)
        for arm in arms
        if arm.pretrained
        for seed in seeds
        if not pretrained_path(arm=arm, cut=cut, month=month, seed=seed).exists()
        and not final_path(arm=arm, cut=cut, month=month, seed=seed).exists()
    ]
    if not todo:
        return
    pre, dropped = prep.fold_pretrain(fold=fold, refit=refit, cut=cut)
    log["pretrain_rows"] = len(pre.x)
    log["pretrain_val_rows"] = int(pre.is_val.sum())
    log["pretrain_rows_dropped"] = dropped.as_dict()
    print(
        f"  pretraining rows {len(pre.x):,} ({int(pre.is_val.sum()):,} held out); "
        f"dropped {dropped.as_dict()}",
        flush=True,
    )
    t = tensors(
        x=pre.x,
        y=pre.y,
        primary=pre.primary,
        weight=np.ones(len(pre.y), dtype=np.float32),
        is_val=pre.is_val,
        fold=fold,
        device=device,
    )
    del pre
    for arm, seed in todo:
        started = time.monotonic()
        torch.manual_seed(seed)
        model = new_model(arm=arm, fold=fold).to(device)
        epochs, best = train(model=model, setting=SETTING, t=t, seed=seed)
        atomic_save(
            state=model.state_dict(),
            target=pretrained_path(arm=arm, cut=cut, month=month, seed=seed),
        )
        seconds = round(time.monotonic() - started)
        log[f"pretrain_{arm.name}_s{seed}"] = {
            "epochs": epochs,
            "val_nll": best,
            "seconds": seconds,
        }
        print(
            f"  pretrain {arm.name} {month} seed {seed}: {epochs} epochs, val NLL {best:.4f}, {seconds} s",
            flush=True,
        )
    del t
    torch.cuda.empty_cache()


def fine_tune(
    *,
    arm: Arm,
    fold: prep.EnsFold,
    month: str,
    seed: int,
    cut: bool,
    device: str,
    t: dict[str, torch.Tensor],
) -> fit397.Fitted:
    """Fine-tune a pretrained network (or train from scratch for `R`) and return it."""
    torch.manual_seed(seed)
    model = new_model(arm=arm, fold=fold).to(device)
    setting = SETTING
    if arm.pretrained:
        model.load_state_dict(torch.load(pretrained_path(arm=arm, cut=cut, month=month, seed=seed)))
        setting = replace(SETTING, lr=FINE_TUNE_LR)
    epochs, best = train(model=model, setting=setting, t=t, seed=seed)
    return fit397.Fitted(
        model=model,
        setting=setting,
        arm="plain_pooled" if arm.embedding else "pooled_noemb",
        seed=seed,
        epochs=epochs,
        val_nll=best,
    )


def final_path(*, arm: Arm, cut: bool, month: str, seed: int) -> Path:
    """Return where an arm's seed-level raw quantile forecast is stored."""
    return OUT_DIR / "seed_preds" / folder_name(arm=arm.name, cut=cut) / f"{month}_s{seed}.parquet"


def run_arms(
    *, months: list[str], arms: list[Arm], seeds: list[int], cut: bool, device: str
) -> None:
    """Pretrain as needed, fit each arm on each fold and seed, and store the forecasts."""
    for month in months:
        todo = [
            (arm, seed)
            for arm in arms
            for seed in seeds
            if not final_path(arm=arm, cut=cut, month=month, seed=seed).exists()
        ]
        if not todo:
            continue
        refit = base.refit_of(month=month)
        fold = prep.build_ens_fold(refit=refit, cut=cut)
        log_path = OUT_DIR / "logs" / f"{month}{'' if cut else '_nocut'}.json"
        try:
            log: dict = json.loads(log_path.read_text()) if log_path.exists() else {}
        except json.JSONDecodeError:
            log = {}
        log |= {
            "month": month,
            "cut": cut,
            "fine_tune_rows": len(fold.x),
            "fine_tune_val_rows": int(fold.is_val.sum()),
            "fine_tune_rows_dropped": fold.dropped.as_dict(),
        }
        print(
            f"fold {month}: {len(fold.x):,} fine-tuning rows; dropped {fold.dropped.as_dict()}",
            flush=True,
        )
        pretrain(
            arms=arms,
            fold=fold,
            refit=refit,
            month=month,
            seeds=seeds,
            cut=cut,
            device=device,
            log=log,
        )
        t = tensors(
            x=fold.x,
            y=fold.y,
            primary=fold.primary,
            weight=fold.weight,
            is_val=fold.is_val,
            fold=fold,
            device=device,
        )
        for arm, seed in todo:
            started = time.monotonic()
            fitted = fine_tune(
                arm=arm, fold=fold, month=month, seed=seed, cut=cut, device=device, t=t
            )
            fit397.save_fits(
                fits={None: fitted},
                fold=fold,
                target=OUT_DIR / "models" / folder_name(arm=arm.name, cut=cut) / month / f"s{seed}",
            )
            q, _ = fit397.forecast_month(
                fits={None: fitted}, fold=fold, scheme="summary", month=month, device=device
            )
            common.atomic_write(
                frame=q, target=final_path(arm=arm, cut=cut, month=month, seed=seed)
            )
            seconds = round(time.monotonic() - started)
            log[f"{arm.name}_s{seed}"] = {
                "epochs": fitted.epochs,
                "val_nll": fitted.val_nll,
                "seconds": seconds,
            }
            print(
                f"{folder_name(arm=arm.name, cut=cut)} {month} seed {seed}: {fitted.epochs} epochs, "
                f"val NLL {fitted.val_nll:.4f}, {seconds} s (fine-tune and forecast)",
                flush=True,
            )
            log_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = log_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(log, indent=1))
            tmp.replace(log_path)
        del t
        torch.cuda.empty_cache()


def main() -> None:
    """Fit the named arms on the named folds and seeds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", nargs="+", default=list(prep.MONTHS))
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true", help="one epoch, outputs under work/smoke")
    parser.add_argument(
        "--no-cut",
        action="store_true",
        help="skip both cuts; only for the 3-fit cross-check (seed 0, three months)",
    )
    args = parser.parse_args()
    if args.no_cut and (args.seeds == list(DEFAULT_SEEDS) or args.months == list(prep.MONTHS)):
        parser.error("--no-cut needs explicit --months and --seeds (seed 0, three months)")
    if args.smoke:
        global OUT_DIR  # noqa: PLW0603
        OUT_DIR = prep.WORK_DIR / "work" / "smoke"
        fit397.MAX_EPOCHS = 1
    run_arms(
        months=args.months,
        arms=[ARMS[a] for a in args.arms],
        seeds=args.seeds,
        cut=not args.no_cut,
        device=args.device,
    )


if __name__ == "__main__":
    main()
