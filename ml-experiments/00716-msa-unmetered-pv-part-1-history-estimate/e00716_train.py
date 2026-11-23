# ruff: noqa: E501, PLR0915, E731
"""Experiment 716: the copied training loop with two optimiser groups and two clips (design section 6, WP4).

Copied from experiment 713's `e00713_train.py` (functions `batch_for`, `validation_loss`, `param_groups`,
and `train_stage`, at the commit that `e00716_arms.code_record()` logs), because the batch, the
parameter groups, and the clip differ. Everything else (the samples, the stage cuts, the recency
weights, the synthetic shifts, `World`) is imported from 713 unchanged.

Differences from 713's loop, and nothing else:

1. Stages 2 and 3 only, and the target row always reads the ensemble.
2. The batch is a `PvSource` batch, with the two PV masks drawn from `pv_dropout_masks`.
3. The parameters of `M*` (every parameter whose name does not start with `pv_`) form 713's groups
   with their rates, the 300-step warm-up, and a gradient clip of 1.0 over those parameters alone.
4. The new parameters form their own groups with no warm-up, their own clip of 1.0, and no weight
   decay on the gains: 1e-2 for `a` and `c`, 3e-3 for `γ_h` and `γ_t`, and 2e-3 for `W1`, `W2`, and
   `pv_head_in`. Weight decay is written into every group.
"""

import time
from collections.abc import Callable
from typing import Final

import e00713_layout as lay  # ty: ignore[unresolved-import]
import e00713_train as tr  # ty: ignore[unresolved-import]
import e00716_batch as pvb
import e00716_model as pvm
import numpy as np
import torch  # ty: ignore[unresolved-import]
from e00558_lags import block_dropout_masks  # ty: ignore[unresolved-import]

NEW_RATES: Final[dict[str, tuple[float, float]]] = {
    "pv_a": (1e-2, 0.0), "pv_c": (1e-2, 0.0),
    "pv_hist_gamma": (3e-3, 0.0), "pv_top_gamma": (3e-3, 0.0),
}  # fmt: skip
"""Learning rate and weight decay of the gains. Every other new parameter (W1, W2, `pv_head_in`) trains at 2e-3 with decay 1e-3."""
NEW_DEFAULT: Final = (2e-3, tr.WEIGHT_DECAY)
GAIN_NAMES: Final = tuple(NEW_RATES)


def batch_for(
    *, source: pvb.PvSource, samples: tr.SampleSet, index: np.ndarray, lead: np.ndarray, cut: int,
    dropout: tuple[np.ndarray, np.ndarray] | None = None,
    pv_masks: tuple[np.ndarray, np.ndarray] | None = None,
) -> pvb.PvAssembled:  # fmt: skip
    """Assemble the batch of `samples[index]` at lead days `lead`.

    `dropout` is 713's pair (mask all power, mask the newest row) and `pv_masks` the pair (CAMS-rule
    mask, mask of ages 2 to 5). 713's `batch_for` with a placebo's `ages` is not used here.
    """
    assert samples.ages is None, "F7's placebos change the PV input, not the history rows"
    dev = source.device
    to = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)
    return source.assemble(
        primary=to(samples.primary[index]), c_idx=to(samples.c_idx[index]), lead_day=to(lead),
        mask_all_power=None if dropout is None else to(dropout[0]),
        mask_newest_row=None if dropout is None else to(dropout[1]),
        cams_rule=None if pv_masks is None else to(pv_masks[0]),
        convention=None if pv_masks is None else to(pv_masks[1]),
        target_end=cut,
    )  # fmt: skip


def validation_loss(
    *, model: pvm.PvNet, source: pvb.PvSource, samples: tr.SampleSet, cut: int
) -> float:
    """Return the unweighted mixture negative log-likelihood on the held-out (sample, lead day) pairs, with no mask."""
    model.eval()
    idx = np.nonzero(samples.is_val)[0]
    total, count = 0.0, 0
    with torch.no_grad():
        for lead_day in tr.VALIDATION_LEAD_DAYS:
            for start in range(0, len(idx), tr.BATCH * 2):
                sel = idx[start : start + tr.BATCH * 2]
                b = batch_for(
                    source=source, samples=samples, index=sel,
                    lead=np.full(len(sel), lead_day, dtype=np.int64), cut=cut,
                )  # fmt: skip
                nll = pvm.mdl.per_token_nll(
                    out=model(b.inputs).out, target=torch.nan_to_num(b.target)
                )
                ok = torch.isfinite(b.target)
                total += float(nll[ok].sum())
                count += int(ok.sum())
    return total / max(count, 1)


def param_groups(
    *, model: pvm.PvNet, stage: int, encoders: str, new_lr_scale: float = 1.0
) -> list[dict]:
    """Return the optimiser groups: 713's three for the parameters of `M*`, then one per rate for the new parameters.

    Every group carries `weight_decay` explicitly, a `base_lr`, and `warmup` (true for the trunk's groups, false for the new ones).
    `new_lr_scale` multiplies the new rates (0 is the check KF13b).
    """
    base_lr = tr.PRETRAIN_LR if stage == 1 else tr.FINE_TUNE_LR
    trunk = {n: p for n, p in model.named_parameters() if not n.startswith(pvm.NEW_PREFIX)}
    enc = {id(p) for p in model.encoder_parameters()}
    spread = {id(p) for p in model.proj_s.parameters()}
    rest = [
        p for p in trunk.values() if p.requires_grad and id(p) not in enc and id(p) not in spread
    ]
    groups: list[dict] = [{"params": rest, "lr": base_lr}]
    if encoders == "trained":
        groups.append(
            {
                "params": model.encoder_parameters(),
                "lr": tr.LOADED_ENCODER_LR if stage == 1 else tr.FINE_TUNE_LR,
            }
        )
    elif encoders == "random":
        groups[0]["params"] = rest + model.encoder_parameters()
    groups.append(
        {"params": list(model.proj_s.parameters()), "lr": tr.SPREAD_LR if stage == 2 else base_lr}
    )
    for g in groups:
        g["base_lr"] = g["lr"]
        g["weight_decay"] = tr.WEIGHT_DECAY
        g["warmup"] = True
        g["trunk"] = True
    new: dict[tuple[float, float], list] = {}
    for name, p in model.named_parameters():
        if name.startswith(pvm.NEW_PREFIX) and p.requires_grad:
            new.setdefault(NEW_RATES.get(name, NEW_DEFAULT), []).append(p)
    for (lr, wd), params in new.items():
        groups.append({
            "params": params, "lr": lr * new_lr_scale, "base_lr": lr * new_lr_scale,
            "weight_decay": wd, "warmup": False, "trunk": False,
        })  # fmt: skip
    return groups


def calibration_batches(*, world: tr.World, samples: tr.SampleSet, source: pvb.PvSource, cut: int):  # noqa: ANN201
    """Yield the training samples of the arm in batches, with no mask and lead day `index mod 14`, to set `r_h` and `r_t`."""
    train_idx = np.nonzero(~samples.is_val)[0]
    for start in range(0, len(train_idx), tr.BATCH * 2):
        sel = train_idx[start : start + tr.BATCH * 2]
        lead = (sel % lay.N_LEAD_DAYS).astype(np.int64)
        yield batch_for(source=source, samples=samples, index=sel, lead=lead, cut=cut).inputs


def train_stage(
    *, model: pvm.PvNet, world: tr.World, stage: int, samples: tr.SampleSet, seed: int,
    encoders: str, max_passes: int = 60, log: tr.StageLog | None = None, patience: int = tr.PATIENCE,
    on_pass: Callable[[int, pvm.PvNet], None] | None = None, new_lr_scale: float = 1.0,
    max_steps: int | None = None, train_source: pvb.PvSource | None = None,
) -> tr.StageLog:  # fmt: skip
    """Train `model` on stage 2 or 3 in place and leave it at the state of the best validation loss.

    `max_steps` stops the loop after that many optimiser steps (the checks KF13 and KF13b run 30).
    """
    assert stage in {2, 3}, stage
    started = time.monotonic()
    log = log or tr.StageLog(stage=stage)
    source = tr.synthetic_source(world=world, samples=samples) if stage == 3 else world.source
    fit_source = (
        source if train_source is None else train_source
    )  # the gating fits train on shifted histories and validate on the unshifted source
    cut = tr.target_end(world=world, stage=stage)
    torch.manual_seed(seed * 7 + stage)
    optimiser = torch.optim.AdamW(
        param_groups(model=model, stage=stage, encoders=encoders, new_lr_scale=new_lr_scale)
    )
    # the trunk clip takes the trunk's parameters in `model.parameters()` order, as experiment 713's clip does, so that
    # the norm is summed in the same order and a binding clip scales the gradient bit-for-bit alike
    is_trunk = {id(p) for g in optimiser.param_groups if g["trunk"] for p in g["params"]}
    trunk_params = [p for p in model.parameters() if id(p) in is_trunk]
    new_params = [p for g in optimiser.param_groups if not g["trunk"] for p in g["params"]]
    train_idx = np.nonzero(~samples.is_val)[0]
    best, best_state, stale, step = float("inf"), None, 0, 0
    stopped = False
    for epoch in range(max_passes):
        model.train()
        rng = np.random.default_rng([seed, stage, epoch])
        order = rng.permutation(train_idx)
        lead_all = rng.integers(0, lay.N_LEAD_DAYS, size=len(order))
        for batch_index, start in enumerate(range(0, len(order), tr.BATCH)):
            sel = order[start : start + tr.BATCH]
            native = samples.c_idx[sel] - lay.NATIVE_OFFSET_STEPS
            drop = block_dropout_masks(origin_idx=native, seed=(seed * 1009 + stage) * 1009 + epoch)
            masks = pvb.pv_dropout_masks(
                seed=seed, stage=stage, epoch=epoch, batch=batch_index, n=len(sel)
            )
            b = batch_for(
                source=fit_source, samples=samples, index=sel, lead=lead_all[start : start + tr.BATCH],
                cut=cut, dropout=drop, pv_masks=masks,
            )  # fmt: skip
            w = tr.token_weights(target_idx=b.target_idx, world=world, stage=stage)
            ok = torch.isfinite(b.target)
            w = w * ok
            nll = pvm.mdl.per_token_nll(out=model(b.inputs).out, target=torch.nan_to_num(b.target))
            loss = (w * nll).sum() / w.sum().clamp_min(1e-9)
            step += 1
            for g in optimiser.param_groups:
                g["lr"] = g["base_lr"] * (min(1.0, step / tr.WARMUP_STEPS) if g["warmup"] else 1.0)
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trunk_params, tr.GRAD_CLIP)
            if new_params:
                torch.nn.utils.clip_grad_norm_(new_params, tr.GRAD_CLIP)
            optimiser.step()
            if max_steps is not None and step >= max_steps:
                stopped = True
                break
        if stopped:
            break
        val = validation_loss(model=model, source=source, samples=samples, cut=cut)
        log.val_history.append(val)
        log.passes = epoch + 1
        if on_pass is not None:
            on_pass(epoch + 1, model)
        if val < best - 1e-5:
            best, stale = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    else:
        log.cap_hit = True
    if stopped:
        model.eval()
        log.seconds = time.monotonic() - started
        return log
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    log.best_val = best
    log.seconds = time.monotonic() - started
    return log
