# ruff: noqa: E501, PLR0915
"""Fit and forecast the settings-sweep networks of experiment 447.

This module reuses experiment 428's fit loop, rows, and prediction unchanged, apart from the
training loop, which it replaces with a copy that also reads the optimiser, the learning-rate
schedule, and the batch size of the arm's `common.Config`. The other settings (hidden width, depth,
mixture components, dropout, weight decay, learning rate) are experiment 428's `fit.SETTING`, which
`install_config` replaces for the arm in hand. With every setting at its S1 level the loop is
experiment 428's, so `X-ref` (the reference refitted) reproduces S1.

```bash
export DATA_PATH_INTERNAL=<path to the private data folder>
PYTHONPATH=<the experiment 397, 404, and 428 folders>:ml-experiments/00447-* OMP_NUM_THREADS=1 \
  systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
  uv run --no-sync python ml-experiments/00447-*/e00447_run.py fit --arms w128 --months 2026-01 --seeds 0
  uv run --no-sync python ml-experiments/00447-*/e00447_run.py predict --arms w128 --months 2026-01 --seeds 0
```
"""

import argparse
import dataclasses
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Final

import e00397_models as base  # ty: ignore[unresolved-import]
import e00428_data as data  # ty: ignore[unresolved-import]
import e00428_fit as fit  # ty: ignore[unresolved-import]
import e00428_models as models  # ty: ignore[unresolved-import]
import e00428_predict as pr  # ty: ignore[unresolved-import]
import e00447_common as common
import torch  # ty: ignore[unresolved-import]

WORK_DIR: Final[Path] = common.WORK_DIR
SETTING0: Final = fit.SETTING
"""Experiment 428's setting g09, the base of every arm."""
_ORIGINAL_MAKE_MODEL: Final = fit.make_model
_CURRENT: dict[str, common.Config] = {"config": common.Config()}


class Lion(torch.optim.Optimizer):
    """The Lion optimiser (sign of an interpolated momentum, decoupled weight decay)."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        *,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ) -> None:
        """Store the learning rate, the two momentum coefficients, and the weight decay."""
        super().__init__(params, {"lr": lr, "betas": betas, "weight_decay": weight_decay})

    @torch.no_grad()
    def step(self, closure=None):  # noqa: ANN001, ANN201
        """Apply one update to every parameter that has a gradient."""
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(p)
                m = state["exp_avg"]
                update = (m * beta1 + p.grad * (1 - beta1)).sign_()
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update, alpha=-group["lr"])
                m.mul_(beta2).add_(p.grad, alpha=1 - beta2)


def install_config(*, config: common.Config) -> None:
    """Make experiment 428's modules use `config` for the next fit or forecast."""
    _CURRENT["config"] = config
    fit.SETTING = dataclasses.replace(
        SETTING0, hidden=config.hidden, layers=config.layers, components=config.components,
        dropout=config.dropout, weight_decay=config.weight_decay, lr=config.lr,
    )  # fmt: skip


def make_model(*, arm: fit.Arm, n_inputs: int, n_primaries: int) -> torch.nn.Module:
    """Return the S1 network with the Primary embedding width of the current config."""
    embedding = _CURRENT["config"].embedding
    if embedding == 8:
        return _ORIGINAL_MAKE_MODEL(arm=arm, n_inputs=n_inputs, n_primaries=n_primaries)
    s = fit.SETTING
    extra = 12  # S1 adds 9 lag columns and 3 surprise columns
    model = models.LaggedMdn(n_inputs=n_inputs + extra, n_primaries=n_primaries, hidden=s.hidden,
                             layers=s.layers, components=s.components, dropout=s.dropout)  # fmt: skip
    model.net = base.PlainMdn(
        n_inputs=n_inputs + extra, n_primaries=n_primaries, hidden=s.hidden, layers=s.layers,
        components=s.components, dropout=s.dropout, embedding=embedding,
    )  # fmt: skip
    return model


def make_optimiser(*, model: torch.nn.Module, config: common.Config) -> torch.optim.Optimizer:
    """Return the optimiser of `config`."""
    params = model.parameters()
    if config.optimiser == "adamw":
        return torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
    if config.optimiser == "adam":
        return torch.optim.Adam(params, lr=config.lr)
    if config.optimiser == "lion":
        return Lion(params, lr=config.lr, weight_decay=config.weight_decay)
    if config.optimiser == "sgd":
        return torch.optim.SGD(
            params, lr=config.lr, momentum=0.9, nesterov=True, weight_decay=config.weight_decay
        )
    msg = f"unknown optimiser {config.optimiser!r}"
    raise ValueError(msg)


def train(
    *, arm: fit.Arm, model: torch.nn.Module, inputs: fit.Inputs, z: torch.Tensor, w: torch.Tensor,
    is_val: torch.Tensor, seed: int, device: str,
) -> tuple[int, float]:  # fmt: skip
    """Experiment 428's `fit.train`, with the optimiser, schedule, and batch size of the config."""
    config = _CURRENT["config"]
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    train_idx = torch.nonzero(~is_val).squeeze(1)
    val_idx = torch.nonzero(is_val).squeeze(1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if len(val_idx) > fit.VALIDATION_ROWS:
        pick = torch.randperm(len(val_idx), generator=generator)[: fit.VALIDATION_ROWS]
        val_idx = val_idx[pick.to(device)]
    val_world = (val_idx % (data.N_TRAIN_WORLDS + 1)) if arm.mixed else torch.zeros_like(val_idx)
    optimiser = make_optimiser(model=model, config=config)
    batch_size = config.batch
    steps_per_epoch = math.ceil(int(arm.row_fraction * len(train_idx)) / batch_size)
    scheduler = None
    if config.schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=fit.MAX_EPOCHS)
    elif config.schedule == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimiser, max_lr=2 * config.lr, total_steps=fit.MAX_EPOCHS * steps_per_epoch
        )
    best, best_state, stale, epochs = float("inf"), None, 0, 0
    k = config.components
    for epoch in range(fit.MAX_EPOCHS):
        model.train()
        order = train_idx[torch.randperm(len(train_idx), device=device)]
        order = order[: int(arm.row_fraction * len(order))]
        for start in range(0, len(order), batch_size):
            b = order[start : start + batch_size]
            world = (
                torch.randint(0, data.N_TRAIN_WORLDS + 1, (len(b),), device=device)
                if arm.mixed
                else torch.zeros(len(b), dtype=torch.long, device=device)
            )
            x, primary, windows = inputs.batch(arm=arm, rows=b, world=world)
            nll = base.mixture_nll(
                out=fit.forward(model=model, x=x, primary=primary, windows=windows),
                z=z[b],
                components=k,
            )
            loss = (w[b] * nll).sum() / w[b].sum()
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), fit.GRAD_CLIP)
            optimiser.step()
            if config.schedule == "onecycle":
                assert scheduler is not None
                scheduler.step()
        if config.schedule == "cosine":
            assert scheduler is not None
            scheduler.step()
        model.eval()
        with torch.no_grad():
            losses = []
            for chunk, chunk_world in zip(
                val_idx.split(batch_size * 2), val_world.split(batch_size * 2), strict=True
            ):
                x, primary, windows = inputs.batch(arm=arm, rows=chunk, world=chunk_world)
                out = fit.forward(model=model, x=x, primary=primary, windows=windows)
                losses.append(base.mixture_nll(out=out, z=z[chunk], components=k))
            val = float(torch.cat(losses).mean())
        epochs = epoch + 1
        print(f"  epoch {epochs}: val nll {val:.4f}", flush=True)
        if val < best - 1e-5:
            best, stale = val, 0
            best_state = {a: v.detach().clone() for a, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= arm.patience:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return epochs, best


def register(*, names: list[str]) -> dict[str, common.Config]:
    """Register the arms `X-<name>` (clones of S1) in experiment 428's tables; return the configs."""
    configs = {"ref": common.Config(), **common.all_configs()}
    for name in names:
        fit.ARMS[common.ARM_PREFIX + name] = dataclasses.replace(
            fit.ARMS[common.REFERENCE], name=common.ARM_PREFIX + name
        )
    return {n: configs[n] for n in names}


def install() -> None:
    """Redirect experiment 428's modules to this experiment's folders and loops."""
    fit.MODELS_DIR = WORK_DIR / "models"
    pr.PRED_DIR = WORK_DIR / "predictions"
    fit.train = train
    fit.make_model = make_model


def main() -> None:
    """Fit or forecast the arms."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fit", "predict"))
    parser.add_argument("--arms", nargs="+", required=True, help="short names (`ref`, `w128`, ...)")
    parser.add_argument("--months", nargs="+", default=list(common.SCREEN_FOLDS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(common.SCREEN_SEEDS))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    configs = register(names=args.arms)
    install()
    if args.command == "fit":
        power = fit.power_tensor(device=args.device)
        stack, _ = data.load_stack(include_eval=False)
        cache: dict = {}
        for month in args.months:
            for seed in args.seeds:
                for name in args.arms:
                    install_config(config=configs[name])
                    print(
                        fit.fit_one(
                            arm_name=common.ARM_PREFIX + name, month=month, seed=seed, power=power,
                            stack=stack, device=args.device, rows_cache=cache,
                        ),
                        flush=True,
                    )  # fmt: skip
        return
    power_np, code = data.load_stack(include_eval=True)
    power = torch.from_numpy(power_np).to(args.device)
    flags = pr.Flags(code=code, real=data.real_event_mask(), power=power_np, device=args.device)
    for month in args.months:
        for seed in args.seeds:
            for name in args.arms:
                install_config(config=configs[name])
                for line in pr.predict_month(
                    arm_name=common.ARM_PREFIX + name, month=month, seed=seed, device=args.device,
                    power=power, flags=flags, y_clean=power_np[0],
                ):  # fmt: skip
                    print(line, flush=True)


if __name__ == "__main__":
    main()
