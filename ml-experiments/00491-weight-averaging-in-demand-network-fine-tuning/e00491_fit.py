# ruff: noqa: E501
r"""Fine-tune experiment 425's arm A with weight averaging in one run (experiment 491).

Each fit loads experiment 425's stored pretrained weights for the fold and seed
(`00425-.../work/pretrained/A/<month>_s<seed>.pt`, never rewritten) and runs experiment 397's
training loop for a fixed budget of `FINE_TUNE_EPOCHS` epochs (the median of experiment 425's
stored fine-tune lengths) at experiment 425's fine-tune learning rate, constant. The loop is a
copy of `e00397_fit.train_network` with the averaging added, so the batch order, the recency
weights, and the optimiser are those of experiment 425. One run yields three networks from the
same trajectory:

- `S0f`: the weights after the last epoch (no early stopping, no best-epoch restore);
- `EMA`: the exponential moving average of the weights over optimiser steps, decay `EMA_DECAY`,
  zero-debiased (the average is divided by `1 - decay ** step`);
- `SWA`: the uniform average of the weights over the optimiser steps of the last
  `SWA_FRACTION` of the budget.

No network is chosen by the held-out-week validation loss. That loss (the unweighted negative
log-likelihood on the held-out week) is logged after every epoch for the raw weights and the EMA
weights, as a diagnostic. The row weights are experiment 425's recency weights, unchanged.

`--repro` runs the same loop with experiment 397's early stopping (patience, best-epoch restore)
for one (month, seed) and writes under `repro/`; it must reproduce experiment 425's stored fit.

`--swa-pre` first pretrains from scratch for `PRETRAIN_EPOCHS` epochs (the median of experiment
425's stored pretraining lengths), keeping both the final weights and the SWA average over the last
`SWA_FRACTION` of the budget, then fine-tunes each as above. Arms: `P0f` (final pretrained weights,
final fine-tuned weights), `Pswa` (SWA pretrained weights, final fine-tuned weights), `Pswa+EMA`,
and `Pswa+SWA`.

Outputs go under this experiment's `work/` folder: `seed_preds/<arm>/<month>_s<seed>.parquet`
(raw quantiles as fractions of capacity, the layout of experiment 425), `models/<arm>/<month>/s<seed>`,
and `logs/<month>.json`.

```bash
source /tmp/e491env.sh   # DATA_PATH_INTERNAL, PYTHONPATH with every ml-experiments folder
systemd-run --user --scope -p MemoryMax=12G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00491-*/e00491_fit.py --months 2026-01 --seeds 0 --repro
```
"""

import argparse
import json
import math
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00397_fit as fit397  # ty: ignore[unresolved-import]
import e00397_prep as base  # ty: ignore[unresolved-import]
import e00425_fit as fit425  # ty: ignore[unresolved-import]
import e00425_prep as prep425  # ty: ignore[unresolved-import]
import numpy as np
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_DIR_NAME: Final[str] = "00491-weight-averaging-in-demand-network-fine-tuning"
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / EXPERIMENT_DIR_NAME / "work"
FOLDS: Final[tuple[str, ...]] = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
"""The five folds of experiment 449."""
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
FINE_TUNE_EPOCHS: Final[int] = 4
"""The median best epoch of experiment 425's 25 stored arm A fine-tunes: stored epochs run minus the early-stopping patience of 6 (the median length is 10)."""
PRETRAIN_EPOCHS: Final[int] = 32
"""The median best epoch of experiment 425's 25 stored arm A pretrainings (median length 38 minus the patience of 6)."""
EMA_DECAY: Final[float] = 0.999
FAST_EMA_DECAY: Final[float] = 0.99
"""A descriptive second average with a memory of about 100 steps."""
SWA_FRACTION: Final[float] = 0.25
SWA_PRE_FOLD: Final[str] = "2026-01"
A_ARM: Final[fit425.Arm] = fit425.ARMS["A"]
FINE_TUNE_SETTING: Final[fit397.Setting] = replace(fit425.SETTING, lr=fit425.FINE_TUNE_LR)


class Averager:
    """Running EMA (zero-debiased) and SWA (uniform, from a start step) of a network's weights."""

    def __init__(self, *, model: torch.nn.Module, swa_start_step: int) -> None:
        """Start both averages at zero; SWA begins after `swa_start_step` optimiser steps."""
        self.live = list(model.state_dict().values())
        self.ema = [torch.zeros_like(v) for v in self.live]
        self.fast = [torch.zeros_like(v) for v in self.live]
        self.swa = [torch.zeros_like(v) for v in self.live]
        self.swa_start_step = swa_start_step
        self.step = 0
        self.swa_count = 0

    def update(self) -> None:
        """Fold the current weights in after an optimiser step."""
        self.step += 1
        torch._foreach_mul_(self.ema, EMA_DECAY)
        torch._foreach_add_(self.ema, self.live, alpha=1.0 - EMA_DECAY)
        torch._foreach_mul_(self.fast, FAST_EMA_DECAY)
        torch._foreach_add_(self.fast, self.live, alpha=1.0 - FAST_EMA_DECAY)
        if self.step > self.swa_start_step:
            self.swa_count += 1
            torch._foreach_lerp_(self.swa, self.live, 1.0 / self.swa_count)

    def ema_state(self) -> list[torch.Tensor]:
        """Return the debiased EMA weights."""
        scale = 1.0 - EMA_DECAY**self.step
        return [v / scale for v in self.ema]

    def fast_state(self) -> list[torch.Tensor]:
        """Return the debiased fast EMA weights."""
        scale = 1.0 - FAST_EMA_DECAY**self.step
        return [v / scale for v in self.fast]


def load_weights(*, model: torch.nn.Module, weights: list[torch.Tensor]) -> None:
    """Copy a list of tensors (in `state_dict` order) into a network."""
    for live, new in zip(model.state_dict().values(), weights, strict=True):
        live.copy_(new)


def snapshot(*, model: torch.nn.Module) -> list[torch.Tensor]:
    """Return a copy of a network's weights in `state_dict` order."""
    return [v.detach().clone() for v in model.state_dict().values()]


def validation_nll(
    *, model: torch.nn.Module, setting: fit397.Setting, t: dict[str, torch.Tensor]
) -> float:
    """Return the unweighted negative log-likelihood on the held-out week."""
    val_idx = torch.nonzero(t["v"]).squeeze(1)
    model.eval()
    with torch.no_grad():
        return float(
            torch.cat(
                [
                    fit397.models.mixture_nll(
                        out=fit397.forward(model=model, x=t["x"][b], primary=t["p"][b]),
                        z=t["z"][b],
                        components=setting.components,
                    )
                    for b in val_idx.split(fit397.BATCH * 4)
                ]
            ).mean()
        )


def train_averaged(
    *,
    model: torch.nn.Module,
    setting: fit397.Setting,
    t: dict[str, torch.Tensor],
    seed: int,
    epochs: int,
) -> tuple[dict[str, list[torch.Tensor]], dict]:
    """Train for a fixed budget (a copy of `fit397.train_network`); return the three weight sets."""
    torch.manual_seed(seed)
    train_idx = torch.nonzero(~t["v"]).squeeze(1)
    steps_per_epoch = math.ceil(len(train_idx) / fit397.BATCH)
    total_steps = steps_per_epoch * epochs
    averager = Averager(model=model, swa_start_step=round((1.0 - SWA_FRACTION) * total_steps))
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=setting.lr, weight_decay=setting.weight_decay
    )
    val_raw, val_ema = [], []
    for _ in range(epochs):
        model.train()
        order = train_idx[torch.randperm(len(train_idx), device=t["x"].device)]
        for start in range(0, len(order), fit397.BATCH):
            b = order[start : start + fit397.BATCH]
            nll = fit397.models.mixture_nll(
                out=fit397.forward(model=model, x=t["x"][b], primary=t["p"][b]),
                z=t["z"][b],
                components=setting.components,
            )
            loss = (t["w"][b] * nll).sum() / t["w"][b].sum()
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), fit397.GRAD_CLIP)
            optimiser.step()
            averager.update()
        val_raw.append(validation_nll(model=model, setting=setting, t=t))
        final = snapshot(model=model)
        load_weights(model=model, weights=averager.ema_state())
        val_ema.append(validation_nll(model=model, setting=setting, t=t))
        load_weights(model=model, weights=final)
    states = {
        "final": final,
        "EMA": averager.ema_state(),
        "EMA99": averager.fast_state(),
        "SWA": list(averager.swa),
    }
    info = {
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "swa_snapshots": averager.swa_count,
        "val_nll_raw": val_raw,
        "val_nll_ema": val_ema,
    }
    return states, info


def forecast_and_store(
    *,
    arm: str,
    model: torch.nn.Module,
    weights: list[torch.Tensor],
    fold: prep425.EnsFold,
    month: str,
    seed: int,
    device: str,
) -> None:
    """Load weights into the network, store the network, and write its raw quantile forecast."""
    load_weights(model=model, weights=weights)
    model.eval()
    fitted = fit397.Fitted(
        model=model, setting=FINE_TUNE_SETTING, arm="plain_pooled", seed=seed, epochs=0, val_nll=0.0
    )
    fit397.save_fits(
        fits={None: fitted}, fold=fold, target=WORK / "models" / arm / month / f"s{seed}"
    )
    q, _ = fit397.forecast_month(
        fits={None: fitted}, fold=fold, scheme="summary", month=month, device=device
    )
    target = final_path(arm=arm, month=month, seed=seed)
    target.parent.mkdir(parents=True, exist_ok=True)
    common.atomic_write(frame=q, target=target)


def stored_weights(
    *, month: str, seed: int, fold: prep425.EnsFold, device: str
) -> list[torch.Tensor]:
    """Load experiment 425's pretrained weights by name into a network; return them in order."""
    model = fit425.new_model(arm=A_ARM, fold=fold).to(device)
    model.load_state_dict(torch.load(pretrained_path(month=month, seed=seed)))
    return snapshot(model=model)


def pretrained_path(*, month: str, seed: int) -> Path:
    """Return experiment 425's stored pretrained weights (read only)."""
    return fit425.OUT_DIR / "pretrained" / "A" / f"{month}_s{seed}.pt"


def final_path(*, arm: str, month: str, seed: int) -> Path:
    """Return where an arm's seed-level raw quantile forecast is stored."""
    return WORK / "seed_preds" / arm / f"{month}_s{seed}.parquet"


def write_log(*, path: Path, log: dict) -> None:
    """Write the log atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(log, indent=1))
    tmp.replace(path)


def read_log(*, path: Path) -> dict:
    """Return the stored log, or an empty one."""
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except json.JSONDecodeError:
        return {}


def fine_tune_arms(
    *,
    names: dict[str, dict[str, str]],
    start: dict[str, list[torch.Tensor]],
    fold: prep425.EnsFold,
    t: dict[str, torch.Tensor],
    month: str,
    seed: int,
    device: str,
    log: dict,
) -> None:
    """Fine-tune from each named starting weights; `names` maps a start key to arm names."""
    for key, arm_names in names.items():
        todo = {
            kind: arm
            for kind, arm in arm_names.items()
            if not final_path(arm=arm, month=month, seed=seed).exists()
        }
        if not todo:
            continue
        started = time.monotonic()
        torch.manual_seed(seed)
        model = fit425.new_model(arm=A_ARM, fold=fold).to(device)
        load_weights(model=model, weights=start[key])
        states, info = train_averaged(
            model=model, setting=FINE_TUNE_SETTING, t=t, seed=seed, epochs=FINE_TUNE_EPOCHS
        )
        for kind, arm in todo.items():
            forecast_and_store(
                arm=arm, model=model, weights=states[kind], fold=fold, month=month, seed=seed,
                device=device,
            )  # fmt: skip
        seconds = round(time.monotonic() - started)
        log[f"{key}_s{seed}"] = info | {"seconds": seconds, "arms": list(todo.values())}
        print(
            f"{key} {month} seed {seed}: {FINE_TUNE_EPOCHS} epochs, {info['steps_per_epoch']} "
            f"steps/epoch, val NLL raw {info['val_nll_raw'][-1]:.4f} EMA "
            f"{info['val_nll_ema'][-1]:.4f}, {seconds} s (fine-tune, three forecasts)",
            flush=True,
        )


def run(*, months: list[str], seeds: list[int], swa_pre: bool, device: str) -> None:
    """Fine-tune (and, with `swa_pre`, pretrain) each fold and seed that is not yet stored."""
    for month in months:
        refit = base.refit_of(month=month)
        fold = prep425.build_ens_fold(refit=refit)
        log_path = WORK / "logs" / f"{month}.json"
        log = read_log(path=log_path) | {"month": month, "fine_tune_rows": len(fold.x)}
        print(f"fold {month}: {len(fold.x):,} fine-tuning rows", flush=True)
        t = fit425.tensors(
            x=fold.x, y=fold.y, primary=fold.primary, weight=fold.weight, is_val=fold.is_val,
            fold=fold, device=device,
        )  # fmt: skip
        for seed in seeds:
            if not swa_pre:
                fine_tune_arms(
                    names={"stored": {"final": "S0f", "EMA": "EMA", "EMA99": "EMA99", "SWA": "SWA"}},
                    start={"stored": stored_weights(month=month, seed=seed, fold=fold, device=device)},
                    fold=fold, t=t, month=month, seed=seed, device=device, log=log,
                )  # fmt: skip
                write_log(path=log_path, log=log)
                continue
            pre_arms = {"final": "P0f", "SWA": "Pswa"}
            if all(final_path(arm=a, month=month, seed=seed).exists() for a in
                   (*pre_arms.values(), "Pswa+EMA", "Pswa+SWA")):  # fmt: skip
                continue
            pretrained = pretrain_averaged(fold=fold, refit=refit, month=month, seed=seed,
                                           device=device, log=log)  # fmt: skip
            fine_tune_arms(
                names={
                    "P0f": {"final": "P0f"},
                    "Pswa": {"final": "Pswa", "EMA": "Pswa+EMA", "SWA": "Pswa+SWA"},
                },
                start={"P0f": pretrained["final"], "Pswa": pretrained["SWA"]},
                fold=fold, t=t, month=month, seed=seed, device=device, log=log,
            )  # fmt: skip
            write_log(path=log_path, log=log)
        del t
        torch.cuda.empty_cache()


def pretrain_averaged(
    *, fold: prep425.EnsFold, refit: datetime, month: str, seed: int, device: str, log: dict
) -> dict[str, list[torch.Tensor]]:
    """Pretrain from scratch for the fixed budget; return the final and SWA weights."""
    started = time.monotonic()
    saved = {k: WORK / "pretrained" / k / f"{month}_s{seed}.pt" for k in ("final", "SWA")}
    if all(p.exists() for p in saved.values()):
        return {k: list(torch.load(p, map_location=device).values()) for k, p in saved.items()}
    pre, _ = prep425.fold_pretrain(fold=fold, refit=refit, cut=True)
    t = fit425.tensors(
        x=pre.x, y=pre.y, primary=pre.primary, weight=np.ones(len(pre.y), dtype=np.float32),
        is_val=pre.is_val, fold=fold, device=device,
    )  # fmt: skip
    del pre
    torch.manual_seed(seed)
    model = fit425.new_model(arm=A_ARM, fold=fold).to(device)
    states, info = train_averaged(
        model=model, setting=fit425.SETTING, t=t, seed=seed, epochs=PRETRAIN_EPOCHS
    )
    for k, path in saved.items():
        model = fit425.new_model(arm=A_ARM, fold=fold).to(device)
        load_weights(model=model, weights=states[k])
        fit425.atomic_save(state=model.state_dict(), target=path)
    seconds = round(time.monotonic() - started)
    log[f"pretrain_s{seed}"] = info | {"seconds": seconds}
    print(
        f"pretrain {month} seed {seed}: {PRETRAIN_EPOCHS} epochs, {info['steps_per_epoch']} "
        f"steps/epoch, val NLL raw {info['val_nll_raw'][-1]:.4f}, {seconds} s",
        flush=True,
    )
    del t
    torch.cuda.empty_cache()
    return states


def repro(*, month: str, seed: int, device: str) -> None:
    """Run experiment 425's early-stopped fine-tune through this script's data path."""
    refit = base.refit_of(month=month)
    fold = prep425.build_ens_fold(refit=refit)
    t = fit425.tensors(
        x=fold.x, y=fold.y, primary=fold.primary, weight=fold.weight, is_val=fold.is_val,
        fold=fold, device=device,
    )  # fmt: skip
    torch.manual_seed(seed)
    model = fit425.new_model(arm=A_ARM, fold=fold).to(device)
    model.load_state_dict(torch.load(pretrained_path(month=month, seed=seed)))
    epochs, best = fit425.train(model=model, setting=FINE_TUNE_SETTING, t=t, seed=seed)
    fitted = fit397.Fitted(
        model=model, setting=FINE_TUNE_SETTING, arm="plain_pooled", seed=seed, epochs=epochs,
        val_nll=best,
    )  # fmt: skip
    q, _ = fit397.forecast_month(
        fits={None: fitted}, fold=fold, scheme="summary", month=month, device=device
    )
    best_epoch = epochs - 6
    probe = fit425.new_model(arm=A_ARM, fold=fold).to(device)
    probe.load_state_dict(torch.load(pretrained_path(month=month, seed=seed)))
    _, info = train_averaged(model=probe, setting=FINE_TUNE_SETTING, t=t, seed=seed, epochs=epochs)
    gap = abs(info["val_nll_raw"][best_epoch - 1] - best)
    print(f"averaged loop: epoch {best_epoch} val NLL differs from the stored best by {gap:.2e}")
    assert gap < 1e-6, "train_averaged does not follow experiment 397's trajectory"
    target = WORK / "repro" / f"{month}_s{seed}.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    common.atomic_write(frame=q, target=target)
    stored = pretrained_path(month=month, seed=seed).parents[2] / "logs" / f"{month}.json"
    print(
        f"repro {month} seed {seed}: {epochs} epochs, val NLL {best:.8f}; stored "
        f"{json.loads(stored.read_text())[f'A_s{seed}']}",
        flush=True,
    )


def main() -> None:
    """Fit the named folds and seeds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", nargs="+", default=list(FOLDS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--swa-pre", action="store_true")
    parser.add_argument("--repro", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.repro:
        repro(month=args.months[0], seed=args.seeds[0], device=args.device)
        return
    run(months=args.months, seeds=args.seeds, swa_pre=args.swa_pre, device=args.device)


if __name__ == "__main__":
    main()
