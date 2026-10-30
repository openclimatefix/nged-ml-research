# ruff: noqa: E501
r"""Pretrain the pooled mixture density network on ERA5 and CAMS weather, fine-tune it on ENS.

Arms (all `pooled-emb-summary-g09`, experiment 397's pooled plain network with the 8-wide Primary
embedding, summary inputs, and setting g09):

- `P1`: pretrain on ERA5 and CAMS rows, then fine-tune every weight on the ENS rows.
- `P2`: the same pretrained network; the second linear layer (`net[3]`) is frozen.
- `P3`: pretrain with forecast-like noise on the weather means, then fine-tune every weight.

`lowdata` repeats the ENS-only network and one pretrained arm on only the first six months of ENS.

```bash
source <(printf 'export PYTHONPATH=%s\n' \
  ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\
ml-experiments/00048-weekly-retraining-confirmation-and-extension-demand-generators:\
ml-experiments/00397-plain-and-attention-mixture-density-network-for-primary-demand:\
ml-experiments/00418-mdn-pretrain-on-era5-and-cams-then-fine-tune-on-ens)
OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00418-*/e00418_fit.py screen --device cuda
```
"""

import argparse
import json
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00397_fit as fit397  # ty: ignore[unresolved-import]
import e00397_prep as base  # ty: ignore[unresolved-import]
import e00418_prep as prep  # ty: ignore[unresolved-import]
import numpy as np
import torch  # ty: ignore[unresolved-import]

SETTING: Final[fit397.Setting] = fit397.setting_by_name(name="g09")
FINE_TUNE_LR: Final[float] = 2e-4
TAG: Final[str] = "pooled-emb-summary-g09"
OUT_DIR: Path = prep.WORK_DIR / "work"
SCREEN_MONTHS: Final[tuple[str, ...]] = ("2025-10", "2026-01", "2026-03", "2026-06")
ARMS: Final[dict[str, dict[str, object]]] = {
    "P1": {"name": f"{TAG}_era5cams-pretrain_ft-all", "noisy": False, "freeze": False},
    "P2": {"name": f"{TAG}_era5cams-pretrain_ft-input-head", "noisy": False, "freeze": True},
    "P3": {"name": f"{TAG}_era5cams-noisy-pretrain_ft-all", "noisy": True, "freeze": False},
}
ONLY_NAME: Final[str] = f"{TAG}_ens-only"


def new_model(*, fold: prep.EnsFold) -> torch.nn.Module:
    """Return an untrained pooled plain network for a fold."""
    return fit397.make_model(
        arm="plain_pooled",
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
    """Train a network in place on tensors with experiment 397's loop; return epochs and best NLL."""
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


def freeze_middle(*, model: torch.nn.Module) -> int:
    """Freeze the second linear layer of the network; return the number of frozen parameters."""
    layer = model.net[3]  # ty: ignore[not-subscriptable]
    assert isinstance(layer, torch.nn.Linear)
    for parameter in layer.parameters():
        parameter.requires_grad_(False)
    return sum(p.numel() for p in layer.parameters())


def pretrained_path(*, month: str, noisy: bool, seed: int) -> Path:
    """Return where a pretrained network's weights are stored."""
    return OUT_DIR / "pretrained" / f"{'noisy' if noisy else 'clean'}" / f"{month}_s{seed}.pt"


def pretrain(
    *,
    fold: prep.EnsFold,
    refit: datetime,
    month: str,
    noisy: bool,
    seeds: list[int],
    device: str,
    log: dict,
) -> None:
    """Pretrain and store one network per seed on the ERA5 and CAMS rows (equal row weights)."""
    todo = [s for s in seeds if not pretrained_path(month=month, noisy=noisy, seed=s).exists()]
    if not todo:
        return
    pre = prep.fold_pretrain(
        fold=fold, refit=refit, noisy=noisy, rng_seed=int(month.replace("-", ""))
    )
    log["pretrain_rows"] = len(pre.x)
    log["pretrain_val_rows"] = int(pre.is_val.sum())
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
    for seed in todo:
        started = time.monotonic()
        model = new_model(fold=fold).to(device)
        epochs, best = train(model=model, setting=SETTING, t=t, seed=seed)
        target = pretrained_path(month=month, noisy=noisy, seed=seed)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), target)
        log[f"pretrain_{'noisy' if noisy else 'clean'}_s{seed}"] = {
            "epochs": epochs,
            "val_nll": best,
            "seconds": round(time.monotonic() - started),
        }
        print(
            f"pretrain {month} noisy={noisy} seed {seed}: {epochs} epochs, val NLL {best:.4f}, {time.monotonic() - started:.0f} s",
            flush=True,
        )
    del t
    torch.cuda.empty_cache()


def fine_tune(
    *, fold: prep.EnsFold, month: str, arm: str, seed: int, device: str, t: dict[str, torch.Tensor]
) -> fit397.Fitted:
    """Fine-tune a pretrained network (or train from scratch for `ens-only`) and return it."""
    model = new_model(fold=fold).to(device)
    setting = SETTING
    if arm != "ens-only":
        spec = ARMS[arm]
        model.load_state_dict(
            torch.load(pretrained_path(month=month, noisy=bool(spec["noisy"]), seed=seed))
        )
        setting = replace(SETTING, lr=FINE_TUNE_LR)
        if spec["freeze"]:
            freeze_middle(model=model)
    epochs, best = train(model=model, setting=setting, t=t, seed=seed)
    return fit397.Fitted(
        model=model, setting=setting, arm="plain_pooled", seed=seed, epochs=epochs, val_nll=best
    )


def arm_name(*, arm: str) -> str:
    """Return the full arm name."""
    return ONLY_NAME if arm == "ens-only" else str(ARMS[arm]["name"])


def run_arms(
    *, months: list[str], arms: list[str], seeds: list[int], device: str, low_data: bool
) -> None:
    """Pretrain as needed, fine-tune each arm on each fold and seed, and store the forecasts."""
    for month in months:
        refit = base.refit_of(month=month)
        fold = prep.build_ens_fold(refit=refit)
        log: dict = {
            "month": month,
            "fine_tune_rows": len(fold.x),
            "fine_tune_val_rows": int(fold.is_val.sum()),
        }
        print(f"fold {month}: {len(fold.x):,} fine-tuning rows", flush=True)
        for noisy in sorted({bool(ARMS[a]["noisy"]) for a in arms if a != "ens-only"}):
            pretrain(
                fold=fold,
                refit=refit,
                month=month,
                noisy=noisy,
                seeds=seeds,
                device=device,
                log=log,
            )
        if low_data:
            keep = fold.valid < int(prep._us(t=prep.LOW_DATA_END))
            fold = subset(fold=fold, keep=keep)
            log["low_data_rows"] = len(fold.x)
        folder = "lowdata" if low_data else "seed_preds"
        t = tensors(
            x=fold.x,
            y=fold.y,
            primary=fold.primary,
            weight=fold.weight,
            is_val=fold.is_val,
            fold=fold,
            device=device,
        )
        for arm in arms:
            for seed in seeds:
                target = OUT_DIR / folder / arm_name(arm=arm) / f"{month}_s{seed}.parquet"
                if target.exists():
                    continue
                started = time.monotonic()
                fitted = fine_tune(fold=fold, month=month, arm=arm, seed=seed, device=device, t=t)
                q, _ = fit397.forecast_month(
                    fits={None: fitted}, fold=fold, scheme="summary", month=month, device=device
                )
                common.atomic_write(frame=q, target=target)
                log[f"{arm}_s{seed}"] = {
                    "epochs": fitted.epochs,
                    "val_nll": fitted.val_nll,
                    "seconds": round(time.monotonic() - started),
                }
                print(
                    f"{arm_name(arm=arm)} {month} seed {seed}: {fitted.epochs} epochs, val NLL {fitted.val_nll:.4f}, {time.monotonic() - started:.0f} s",
                    flush=True,
                )
        del t
        torch.cuda.empty_cache()
        (OUT_DIR / "logs").mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "logs" / f"{'lowdata_' if low_data else ''}{month}.json").write_text(
            json.dumps(log, indent=1)
        )


def subset(*, fold: prep.EnsFold, keep: np.ndarray) -> prep.EnsFold:
    """Return a fold limited to the rows in `keep`, with the same standardiser and targets."""
    fields = {
        k: getattr(fold, k)[keep]
        for k in ("x", "y", "primary", "weight", "is_val", "lead", "origin_day", "valid")
    }
    return replace(fold, **fields)


def main() -> None:
    """Run the screen or the low-data check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("screen", "lowdata"))
    parser.add_argument("--months", nargs="+", default=list(SCREEN_MONTHS))
    parser.add_argument("--arms", nargs="+", default=["P1", "P2", "P3"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true", help="one epoch, outputs under work/smoke")
    args = parser.parse_args()
    if args.smoke:
        global OUT_DIR  # noqa: PLW0603
        OUT_DIR = prep.WORK_DIR / "smoke"
        fit397.MAX_EPOCHS = 1
    run_arms(
        months=args.months,
        arms=args.arms if args.stage == "screen" else ["ens-only", *args.arms],
        seeds=args.seeds,
        device=args.device,
        low_data=args.stage == "lowdata",
    )


if __name__ == "__main__":
    main()
