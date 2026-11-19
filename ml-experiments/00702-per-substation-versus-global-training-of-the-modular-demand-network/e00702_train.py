# ruff: noqa: PLR0912, PLR0915
"""Experiment 702: experiment 574's training loop with a batch argument and a fixed-epoch option.

A copy of `train` in `e00574_fit.py`. The loop, the optimiser, the early stopping, and the
validation subsample are unchanged. Four things are new:

- `batch` replaces `c.BATCH` (the plan's batch of 512 for per-substation phases);
- `fixed_epochs` trains for exactly that many epochs with no validation and keeps the last epoch's
  weights (the plan's fallback when a Primary has too few held weeks);
- a model may name extra modules in `fresh_modules` (Hybrid 3's two new encoders). They join the
  mixer in the fresh-learning-rate group when `fresh_lr` is given, and the ordinary group otherwise;
- the settings that experiment 510 holds as module constants are arguments with the same values,
  and the driver asserts that they still equal `e00510_common`'s.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import torch  # ty: ignore[unresolved-import]


@dataclass(frozen=True)
class Settings:
    """The loop's settings, with the values of `e00510_common`."""

    patience: int = 8
    row_fraction: float = 0.2
    weight_decay: float = 1e-3
    grad_clip: float = 1.0
    validation_rows: int = 200_000


SETTINGS: Final[Settings] = Settings()


def train(
    *,
    model: torch.nn.Module,
    x: torch.Tensor,
    z: torch.Tensor,
    w: torch.Tensor,
    p: torch.Tensor,
    is_val: torch.Tensor,
    seed: int,
    lr: float,
    max_epochs: int,
    fresh_lr: float | None,
    frozen: bool,
    batch: int,
    device: str,
    row_nll: Callable[..., torch.Tensor],
    fixed_epochs: int | None = None,
    settings: Settings = SETTINGS,
) -> tuple[int, float]:
    """Train `model` in place and return the epochs run and the best (or last) validation loss.

    Args:
        model: The network. It must have `weather`, `datetime`, and `mixer_ft` modules.
        x: Inputs of the selected rows, on `device`.
        z: Standardised targets.
        w: Row weights.
        p: Primary index of each row.
        is_val: Early-stopping rows.
        seed: Seed of the batch order and of the validation subsample.
        lr: Learning rate of the ordinary parameters.
        max_epochs: Epoch limit when early stopping applies.
        fresh_lr: Learning rate of the fresh mixer, or None to train it at `lr`.
        frozen: Whether the weather and date-time encoders are frozen.
        batch: Rows per gradient step.
        device: The torch device.
        row_nll: Maps `out, z, p` to each row's negative log-likelihood.
        fixed_epochs: When given, train that many epochs on the non-held rows, validate nothing,
            and keep the last epoch.
        settings: Patience, row fraction, weight decay, gradient clip, and validation cap.

    Returns:
        The number of epochs run and the validation loss of the kept weights (NaN when fixed).
    """
    torch.manual_seed(seed)
    train_idx = torch.nonzero(~is_val).squeeze(1)
    val_idx = torch.nonzero(is_val).squeeze(1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if len(val_idx) > settings.validation_rows:
        pick = torch.randperm(len(val_idx), generator=generator)[: settings.validation_rows]
        val_idx = val_idx[pick.to(device)]
    encoder_ids = {id(q) for part in (model.weather, model.datetime) for q in part.parameters()}  # ty: ignore[unresolved-attribute]
    if frozen:
        for part in (model.weather, model.datetime):
            for q in part.parameters():  # ty: ignore[unresolved-attribute]
                q.requires_grad_(False)
    fresh_names = ("mixer_ft", *getattr(model, "fresh_modules", ()))
    fresh = (
        [q for name in fresh_names for q in getattr(model, name).parameters()]
        if fresh_lr is not None
        else []
    )
    fresh_ids = {id(q) for q in fresh}
    normal = [
        q
        for q in model.parameters()
        if id(q) not in fresh_ids and not (frozen and id(q) in encoder_ids)
    ]
    groups = [{"params": normal, "lr": lr}]
    if fresh:
        groups.append({"params": fresh, "lr": fresh_lr})
    optimiser = torch.optim.AdamW(groups, weight_decay=settings.weight_decay)
    best, best_state, stale, epochs = float("inf"), None, 0, 0
    last_val = float("nan")
    for epoch in range(max_epochs if fixed_epochs is None else fixed_epochs):
        model.train()
        if frozen:
            model.weather.eval()  # ty: ignore[unresolved-attribute]
            model.datetime.eval()  # ty: ignore[unresolved-attribute]
        order = train_idx[torch.randperm(len(train_idx), device=device)]
        order = order[: int(settings.row_fraction * len(order))]
        for start in range(0, len(order), batch):
            b = order[start : start + batch]
            nll = row_nll(out=model(x[b], p[b]), z=z[b], p=p[b])
            loss = (w[b] * nll).sum() / w[b].sum()
            if hasattr(model, "penalty"):
                loss = loss + model.penalty()  # ty: ignore[call-non-callable]
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings.grad_clip)
            optimiser.step()
        epochs = epoch + 1
        if fixed_epochs is not None:
            continue
        model.eval()
        with torch.no_grad():
            last_val = float(
                torch.cat(
                    [
                        row_nll(out=model(x[i], p[i]), z=z[i], p=p[i])
                        for i in val_idx.split(batch * 2)
                    ]
                ).mean()
            )
        if last_val < best - 1e-5:
            best, stale = last_val, 0
            best_state = {a: v.detach().clone() for a, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= settings.patience:
                break
    if fixed_epochs is None:
        assert best_state is not None
        model.load_state_dict(best_state)
    model.eval()
    for q in model.parameters():
        q.requires_grad_(True)
    return epochs, (best if fixed_epochs is None else float("nan"))
