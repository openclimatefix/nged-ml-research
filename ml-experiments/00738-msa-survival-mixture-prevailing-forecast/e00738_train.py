"""Training loops for the head arms and for W0, on injected worlds.

The loop is a copy of `train_stage` of experiment 713 (`e00713_train.py`, not edited) with three
changes: every training pass draws a new world and uses it as both the history power and the target,
a fixed validation world serves early stopping, and the forward pass is a function so that the head
arms (which train only the level head on a frozen base) and W0 (which trains the whole network)
share
the loop. Batches, sample sets, loss weights, the learning-rate warm-up, gradient clipping, and the
block dropout are experiment 713's own.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import e00558_lags as lags  # ty: ignore[unresolved-import]
import e00713_batch as bt  # ty: ignore[unresolved-import]
import e00713_layout as lay  # ty: ignore[unresolved-import]
import e00713_model as mdl  # ty: ignore[unresolved-import]
import e00713_train as tr  # ty: ignore[unresolved-import]
import e00738_common as common
import e00738_model as lmdl
import e00738_world as world
import numpy as np
import torch  # ty: ignore[unresolved-import]

ForwardNll = Callable[[torch.nn.Module, bt.Assembled], torch.Tensor]


@dataclass
class FitLog:
    """What one stage reports, with the loss histories for the loss plots."""

    stage: int
    passes: int = 0
    best_val: float = float("inf")
    best_pass: int = 0
    cap_hit: bool = False
    seconds: float = 0.0
    train_history: list[float] = field(default_factory=list)
    val_history: list[float] = field(default_factory=list)


def world_source(
    *,
    base: bt.Source,
    clean: np.ndarray,
    own: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    device: str,
) -> bt.Source:
    """Return a source whose history power and targets both carry the injected shifts.

    Args:
        base: The fold's source (weather, calendar, ensemble table are kept).
        clean: Cleaned power over capacity (16, grid), NaN where dropped.
        own: The shift over capacity (16, grid).
        target_mean: Each Primary's mean power over capacity.
        target_std: Each Primary's standard deviation of power over capacity.
        device: The device.

    Returns:
        The source.
    """
    z = ((world.inject(clean=clean, own=own) - target_mean[:, None]) / target_std[:, None]).astype(
        np.float32
    )
    t = torch.from_numpy(z).to(device)
    return bt.Source(
        power_z=t,
        dense_weather=base.dense_weather,
        dense_datetime=base.dense_datetime,
        day_type=base.day_type,
        ens=base.ens,
        power_in=t,
    )


def draw_world(
    *,
    space: str,
    fit_seed: int,
    month: str,
    pass_no: int,
    clean: np.ndarray,
    cutoff_idx: int,
    is_mw: np.ndarray,
) -> np.ndarray:
    """Return the shift (16, grid) of one training or validation world.

    Args:
        space: `train` or `valid`.
        fit_seed: The seed of the fit.
        month: The fold month.
        pass_no: The pass (0 for the fixed validation world).
        clean: Cleaned power over capacity.
        cutoff_idx: The fold's cutoff; no shift runs past it.
        is_mw: Per Primary, whether it reports signed megawatts.

    Returns:
        The float32 shift.
    """
    shifts = []
    for p in range(clean.shape[0]):
        shifts.extend(
            world.renewal_shifts(
                space=space,
                fit_seed=fit_seed,
                month_idx=common.month_index(month=month),
                pass_no=pass_no,
                primary=p,
                window_start_idx=0,
                window_end_idx=cutoff_idx,
                is_mw=is_mw,
            )
        )
    return world.build_own(clean=clean, shifts=shifts, is_mw=is_mw)


def head_forward(model: torch.nn.Module, b: bt.Assembled) -> torch.Tensor:
    """Return the negative log-likelihood of a head arm's mixture (B, 48) for an assembled batch."""
    assert isinstance(model, lmdl.LevelNet)
    lo = model.level_forward(b.inputs)
    return lmdl.nll_of(out=lo.out, target=torch.nan_to_num(b.target), components=lo.components)


def w0_forward(model: torch.nn.Module, b: bt.Assembled) -> torch.Tensor:
    """Return the negative log-likelihood of W0's three-component mixture (B, 48)."""
    assert isinstance(model, mdl.MsaNet)
    return mdl.per_token_nll(out=model(b.inputs).out, target=torch.nan_to_num(b.target))


def validation_nll(
    *,
    model: torch.nn.Module,
    fwd: ForwardNll,
    source: bt.Source,
    samples: tr.SampleSet,
    ensemble: bool,
    cut: int,
) -> float:
    """Return the unweighted mixture negative log-likelihood on the held-out pairs."""
    model.eval()
    idx = np.nonzero(samples.is_val)[0]
    total, count = 0.0, 0
    with torch.no_grad():
        for lead_day in tr.VALIDATION_LEAD_DAYS:
            for start in range(0, len(idx), tr.BATCH * 2):
                sel = idx[start : start + tr.BATCH * 2]
                b = tr.batch_for(
                    source=source,
                    samples=samples,
                    index=sel,
                    lead=np.full(len(sel), lead_day, dtype=np.int64),
                    ensemble=ensemble,
                    cut=cut,
                    dropout=None,
                )
                nll = fwd(model, b)
                ok = torch.isfinite(b.target)
                total += float(nll[ok].sum())
                count += int(ok.sum())
    return total / max(count, 1)


def run_stage(  # noqa: PLR0915 - one linear training loop
    *,
    model: torch.nn.Module,
    fwd: ForwardNll,
    groups: list[dict],
    tw: tr.World,
    stage: int,
    samples: tr.SampleSet,
    seed: int,
    fit_seed: int,
    clean: np.ndarray,
    is_mw: np.ndarray,
    cap: int,
    patience: int,
    state_of: Callable[[torch.nn.Module], dict],
    on_pass: Callable[[int, FitLog], None] | None = None,
) -> FitLog:
    """Train one stage in place on injected worlds and leave the model at the best validation state.

    Args:
        model: The network.
        fwd: The forward pass returning the negative log-likelihood (B, 48).
        groups: The optimiser groups (each with `params`, `lr`, and `base_lr`).
        tw: The fold's `World` of experiment 713.
        stage: 2 or 3 (3 for the head arms).
        samples: The stage's samples.
        seed: The seed that drives the order of samples and the dropout.
        fit_seed: The seed of the world draws.
        clean: Cleaned power over capacity (16, grid).
        is_mw: Per Primary, whether it reports signed megawatts.
        cap: The pass cap.
        patience: The patience in passes.
        state_of: Returns the part of the state to keep as the best (the head, or the whole
        network).
        on_pass: Called after each pass's validation with the pass number and the log.

    Returns:
        The stage log, with both loss histories.
    """
    started = time.monotonic()
    log = FitLog(stage=stage)
    ensemble = True
    cut = tr.target_end(world=tw, stage=stage)
    torch.manual_seed(seed * 7 + stage)
    optimiser = torch.optim.AdamW(groups, weight_decay=tr.WEIGHT_DECAY)
    device = str(tw.source.device)
    val_own = draw_world(
        space="valid",
        fit_seed=fit_seed,
        month=tw.month,
        pass_no=0,
        clean=clean,
        cutoff_idx=tw.cutoff_idx,
        is_mw=is_mw,
    )
    val_source = world_source(
        base=tw.source,
        clean=clean,
        own=val_own,
        target_mean=tw.target_mean,
        target_std=tw.target_std,
        device=device,
    )
    train_idx = np.nonzero(~samples.is_val)[0]
    best, best_state, stale, step = float("inf"), None, 0, 0
    for epoch in range(cap):
        model.train()
        own = draw_world(
            space="train",
            fit_seed=fit_seed,
            month=tw.month,
            pass_no=epoch,
            clean=clean,
            cutoff_idx=tw.cutoff_idx,
            is_mw=is_mw,
        )
        source = world_source(
            base=tw.source,
            clean=clean,
            own=own,
            target_mean=tw.target_mean,
            target_std=tw.target_std,
            device=device,
        )
        rng = np.random.default_rng([seed, stage, epoch])
        order = rng.permutation(train_idx)
        lead_all = rng.integers(0, lay.N_LEAD_DAYS, size=len(order))
        loss_sum, w_sum = 0.0, 0.0
        for start in range(0, len(order), tr.BATCH):
            sel = order[start : start + tr.BATCH]
            native = samples.c_idx[sel] - lay.NATIVE_OFFSET_STEPS
            drop = lags.block_dropout_masks(
                origin_idx=native, seed=(seed * 1009 + stage) * 1009 + epoch
            )
            b = tr.batch_for(
                source=source,
                samples=samples,
                index=sel,
                lead=lead_all[start : start + tr.BATCH],
                ensemble=ensemble,
                cut=cut,
                dropout=drop,
            )
            w = tr.token_weights(target_idx=b.target_idx, world=tw, stage=stage)
            w = w * torch.isfinite(b.target)
            loss = (w * fwd(model, b)).sum() / w.sum().clamp_min(1e-9)
            step += 1
            for g in optimiser.param_groups:
                g["lr"] = g["base_lr"] * min(1.0, step / tr.WARMUP_STEPS)
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], tr.GRAD_CLIP)
            optimiser.step()
            loss_sum += float(loss.detach()) * float(w.sum())
            w_sum += float(w.sum())
        val = validation_nll(
            model=model, fwd=fwd, source=val_source, samples=samples, ensemble=ensemble, cut=cut
        )
        log.train_history.append(loss_sum / max(w_sum, 1e-9))
        log.val_history.append(val)
        log.passes = epoch + 1
        if on_pass is not None:
            on_pass(epoch + 1, log)
        if val < best - 1e-5:
            best, stale, log.best_pass = val, 0, epoch + 1
            best_state = {k: v.detach().clone() for k, v in state_of(model).items()}
        else:
            stale += 1
            if stale >= patience:
                break
    else:
        log.cap_hit = True
    assert best_state is not None
    model.load_state_dict(best_state, strict=False)
    model.eval()
    log.best_val = best
    log.seconds = time.monotonic() - started
    return log


def head_groups(*, model: lmdl.LevelNet) -> list[dict]:
    """Return the optimiser group of a head arm: the level head at the head learning rate."""
    return [{"params": model.level_parameters(), "lr": common.HEAD_LR, "base_lr": common.HEAD_LR}]


def head_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return the level head's state (the only part of a head arm that trains)."""
    assert isinstance(model, lmdl.LevelNet)
    return {f"level_head.{k}": v for k, v in model.level_head.state_dict().items()}


def full_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return the whole state of W0."""
    return model.state_dict()


def fit_head_arm(
    *,
    model: lmdl.LevelNet,
    tw: tr.World,
    clean: np.ndarray,
    is_mw: np.ndarray,
    seed: int,
    remedy: bool,
    on_pass: Callable[[int, FitLog], None] | None = None,
) -> FitLog:
    """Train the level head of one arm on injected worlds, in one stage on the frozen base.

    Args:
        model: The `LevelNet`, with its base loaded and frozen.
        tw: The fold's `World`.
        clean: Cleaned power over capacity (16, grid).
        is_mw: Per Primary, whether it reports signed megawatts.
        seed: The seed of the fit.
        remedy: Use the gate's pre-registered remedy (cap 120, patience 15).
        on_pass: Called after each pass.

    Returns:
        The log.
    """
    samples = tr.make_samples(world=tw, stage=3, placebo=False)
    return run_stage(
        model=model,
        fwd=head_forward,
        groups=head_groups(model=model),
        tw=tw,
        stage=3,
        samples=samples,
        seed=seed,
        fit_seed=seed,
        clean=clean,
        is_mw=is_mw,
        cap=common.pass_cap(remedy=remedy),
        patience=common.REMEDY_PATIENCE if remedy else common.PATIENCE,
        state_of=head_state,
        on_pass=on_pass,
    )


def fit_w0(
    *,
    model: mdl.MsaNet,
    tw: tr.World,
    clean: np.ndarray,
    is_mw: np.ndarray,
    seed: int,
    on_pass: Callable[[int, FitLog], None] | None = None,
) -> list[FitLog]:
    """Run stages 2 and 3 of experiment 713's recipe on injected worlds, from a stage 1 state.

    Args:
        model: The `MsaNet` with `M1`'s stage 1 state loaded and its encoders frozen.
        tw: The fold's `World`.
        clean: Cleaned power over capacity (16, grid).
        is_mw: Per Primary, whether it reports signed megawatts.
        seed: The seed of the fit.
        on_pass: Called after each pass.

    Returns:
        The logs of stage 2 and stage 3.
    """
    logs = []
    for stage in (2, 3):
        samples = tr.make_samples(world=tw, stage=stage, placebo=False)
        logs.append(
            run_stage(
                model=model,
                fwd=w0_forward,
                groups=tr.param_groups(model=model, stage=stage, encoders="frozen"),
                tw=tw,
                stage=stage,
                samples=samples,
                seed=seed,
                fit_seed=seed,
                clean=clean,
                is_mw=is_mw,
                cap=common.pass_cap(remedy=False),
                patience=common.PATIENCE,
                state_of=full_state,
                on_pass=on_pass,
            )
        )
    return logs
