# ruff: noqa: E501, PT018, D205, D209
"""Experiment 715: the training loop of F4, stages 2 and 3, copied from experiment 713 (design
sections 5 and 11.1).

Experiment 713's `train_stage`, `batch_for`, `validation_loss`, `param_groups`, and `synthetic_source`
(`e00713_train.py` at commit 3ac110652, lines 543 to 698) are copied, because the batch, the parameter
groups, and the stage 3 source differ. Every constant, `World`, `SampleSet`, `StageLog`,
`make_samples`, `token_weights`, `target_end`, `block_pool`, `placebo_ages_for`, and `save_atomic` is
imported from experiment 713 unchanged, so the recipe has one definition.

What differs from experiment 713's loop, and nothing else:

- The model is an `F4Net` that also reads the neighbour tensors, or an `MsaNet` with no neighbour path
  (arm `F4-0`), which draws no new random number and has no new parameter.
- The neighbour path's dropout masks come from `F4Net.nbr_generator`, re-keyed each pass on
  (seed, stage, pass), and the neighbour dropout of whole slots from a NumPy generator keyed the same
  way, so experiment 713's sample order, lead-day draws, block-dropout draws, and trunk dropout masks
  are the draws `M*` made.
- The 36,888 new parameters train at 5e-4 in both stages. `proj_s` trains at 5e-4 in stage 2 and at
  2e-4 in stage 3, as in experiment 713.
- Stage 3 reads experiment 581's shifted array as host history and the cleaned power as neighbour power
  (arms `F4`, `F4-R`, `F4-T`, `F4E`), or the mirror world's arrays (arms `F4-W` and `F4-WR`).
- Arm `F4-T` reads each neighbour at the shuffled ages of experiment 713's placebo draw, from a pool
  that also needs a reading at one neighbour or more.
"""

import time
from dataclasses import dataclass, field
from typing import Final

import e00713_batch as bt  # ty: ignore[unresolved-import]
import e00713_layout as lay  # ty: ignore[unresolved-import]
import e00713_model as mdl  # ty: ignore[unresolved-import]
import e00713_train as tr  # ty: ignore[unresolved-import]
import e00715_arms as arms
import e00715_batch as fb
import e00715_model as f4
import e00715_neighbours as nb
import e00715_worlds as wd
import numpy as np
import torch  # ty: ignore[unresolved-import]
from e00558_lags import block_dropout_masks  # ty: ignore[unresolved-import]

NEW_LR: Final = 5e-4
"""The learning rate of the 36,888 new parameters in stages 2 and 3."""
KEY_SCALE: Final = 1009


@dataclass
class F4World:
    """What an F4 training job reads, on top of experiment 713's `World`.

    Attributes:
        base: Experiment 713's `World` (the source holds the cleaned power).
        table: The neighbour table (the real one, or a pair table in the synthetic worlds).
        mw_scale: (16,) `target_std * capacity_mw`.
        spec: The arm.
        rewire_seed: The seed of the rewired draw (the seed of the neural network).
        capacity_mw: (16,) effective capacity, for the mirror world.
        mva_only: (16,) bool, for the mirror world's floor.
        shuffle_pool: The pool of arm `F4-T`'s shuffled ages, or None for the other arms.
    """

    base: tr.World
    table: nb.NeighbourTable
    mw_scale: np.ndarray
    spec: arms.ArmSpec
    rewire_seed: int
    capacity_mw: np.ndarray | None = None
    mva_only: np.ndarray | None = None
    shuffle_pool: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Build the pool of blocks of arm `F4-T` (the host and at least one neighbour have a reading)."""
        if self.spec.slots == "shuffled" and self.shuffle_pool is None:
            assert self.base.placebo_pool is not None
            self.shuffle_pool = nb.shuffle_pool(table=self.table, pool=self.base.placebo_pool)


@dataclass
class F4Samples(tr.SampleSet):
    """The (Primary, origin day) pairs of one stage, with the shuffled ages of arm `F4-T`."""

    nbr_ages: np.ndarray | None = None  # (N, 14), -1 where the slot has no block


def make_samples(*, world: F4World, stage: int) -> F4Samples:
    """Return experiment 713's pairs of `stage`, with the shuffled neighbour ages for arm `F4-T`."""
    s = tr.make_samples(world=world.base, stage=stage, placebo=False)
    ages = None
    if world.spec.slots == "shuffled":
        assert world.shuffle_pool is not None
        ages = tr.placebo_ages_for(primary=s.primary, c_idx=s.c_idx, pool=world.shuffle_pool)
    return F4Samples(primary=s.primary, c_idx=s.c_idx, is_val=s.is_val, ages=None, nbr_ages=ages)  # ty: ignore[unknown-argument]


def shifted_history(*, world: tr.World, samples: tr.SampleSet) -> torch.Tensor:
    """Return the stage 3 host history array: experiment 581's lag-only shifts, as `e00713_train.synthetic_source`.

    A copy of `e00713_train.synthetic_source` (lines 543 to 558 at commit 3ac110652) that returns the
    array and not a source, so that the caller names the neighbour array separately.
    """
    import e00581_bank as bank  # ty: ignore[unresolved-import]

    first = int(samples.c_idx.min()) - tr.SYNTHETIC_FIRST_MARGIN
    last = int(samples.c_idx.max()) + 672
    fixed, _ = bank.synthetic_clean(
        clean=world.clean, first=first, last=last, seed=int(world.month.replace("-", ""))
    )
    z = (fixed - world.target_mean[:, None]) / world.target_std[:, None]
    return torch.from_numpy(z.astype(np.float32)).to(world.source.device)


def mirror_arrays(*, world: F4World, samples: tr.SampleSet) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the mirror world's (history, target) arrays, standardised, on the device (design section 5.3)."""
    assert world.capacity_mw is not None and world.mva_only is not None
    base = world.base
    first = int(samples.c_idx.min()) - tr.SYNTHETIC_FIRST_MARGIN
    last = int(samples.c_idx.max()) + 672
    mirror = wd.mirror_world(
        clean=base.clean, first=first, last=last, seed=int(base.month.replace("-", "")),
        month_key=int(base.month.replace("-", "")), table=world.table,
        capacity_mw=world.capacity_mw, mva_only=world.mva_only,
    )  # fmt: skip
    dev = base.source.device
    z = lambda a: torch.from_numpy(  # noqa: E731
        ((a - base.target_mean[:, None]) / base.target_std[:, None]).astype(np.float32)
    ).to(dev)
    return z(mirror.history), z(mirror.target)


def stage_source(*, world: F4World, samples: tr.SampleSet, stage: int) -> fb.F4Source:
    """Return the source of a stage, by the mode table of design section 5.2.

    Stage 2 reads the cleaned power for the host and the neighbours (`stage2`). Stage 3 reads experiment
    581's shifted array for the host and the cleaned power for the neighbours (`stage3_main`), or the
    mirror world's history array for both and its target array for the targets (`stage3_mirror`).
    """
    base = world.base.source
    mode = (
        "stage2"
        if stage == 2
        else ("stage3_mirror" if world.spec.world == "mirror" else "stage3_main")
    )
    named: dict[fb.ArrayName, torch.Tensor] = {"cleaned": base.power_z}
    host = base
    if stage == 3 and world.spec.world == "mirror":
        history, target = mirror_arrays(world=world, samples=samples)
        named["mirror"] = history
        host = bt.Source(
            power_z=target, dense_weather=base.dense_weather, dense_datetime=base.dense_datetime,
            day_type=base.day_type, ens=base.ens, power_in=history,
        )  # fmt: skip
    elif stage == 3:
        named["shifted"] = shifted_history(world=world.base, samples=samples)
    elif stage == 1:
        msg = "F4 starts from the stage 1 state of M*: stage 1 is never run here"
        raise AssertionError(msg)
    host_array, nbr_array, label = fb.arrays_for_mode(mode=mode, named=named)  # ty: ignore[invalid-argument-type]
    return fb.F4Source(
        host=host.with_power_in(power_in=host_array), neighbour_power=nbr_array, mw_scale=world.mw_scale,
        table=world.table, mode=world.spec.slots, rewire_seed=world.rewire_seed, label=label,
    )  # fmt: skip


def batch_for(
    *, source: fb.F4Source, samples: F4Samples, index: np.ndarray, lead: np.ndarray, ensemble: bool,
    cut: int, dropout: tuple[np.ndarray, np.ndarray] | None, drop_slots: np.ndarray | None = None,
    mask_all_slots: bool = False,
) -> fb.F4Assembled:  # fmt: skip
    """Assemble the batch of `samples[index]` at lead days `lead`: experiment 713's host batch and
    the neighbour tensors."""
    dev = source.device
    to = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)  # noqa: E731
    nbr_ages = nbr_exists = None
    if samples.nbr_ages is not None:
        a = samples.nbr_ages[index]
        nbr_exists = to(a >= 0)
        nbr_ages = to(np.where(a >= 0, a, lay.PLACEBO_MIN_AGE))
    return source.assemble(
        primary=to(samples.primary[index]), c_idx=to(samples.c_idx[index]), lead_day=to(lead),
        ensemble=ensemble,
        mask_all_power=None if dropout is None else to(dropout[0]),
        mask_newest_row=None if dropout is None else to(dropout[1]),
        target_end=cut, nbr_ages=nbr_ages, nbr_age_exists=nbr_exists,
        drop_slots=None if drop_slots is None else to(drop_slots), mask_all_slots=mask_all_slots,
    )  # fmt: skip


def run_model(*, model: mdl.MsaNet, batch: fb.F4Assembled, neighbours: bool) -> torch.Tensor:
    """Return the raw mixture outputs: `F4Net` with the neighbour tensors, or `MsaNet` (arm `F4-0`) without."""
    if neighbours:
        assert isinstance(model, f4.F4Net)
        return model(batch.host.inputs, batch.nbr).out
    return model(batch.host.inputs).out


def validation_loss(
    *, model: mdl.MsaNet, source: fb.F4Source, samples: F4Samples, ensemble: bool, cut: int,
    neighbours: bool,
) -> float:  # fmt: skip
    """Return the unweighted mixture negative log-likelihood on the held-out (sample, lead day) pairs.

    A copy of `e00713_train.validation_loss` (lines 582 to 602) with the neighbour tensors; no
    neighbour dropout and no block dropout.
    """
    model.eval()
    idx = np.nonzero(samples.is_val)[0]
    total, count = 0.0, 0
    with torch.no_grad():
        for lead_day in tr.VALIDATION_LEAD_DAYS:
            for start in range(0, len(idx), tr.BATCH * 2):
                sel = idx[start : start + tr.BATCH * 2]
                b = batch_for(
                    source=source, samples=samples, index=sel,
                    lead=np.full(len(sel), lead_day, dtype=np.int64), ensemble=ensemble, cut=cut,
                    dropout=None,
                )  # fmt: skip
                nll = mdl.per_token_nll(
                    out=run_model(model=model, batch=b, neighbours=neighbours),
                    target=torch.nan_to_num(b.host.target),
                )
                ok = torch.isfinite(b.host.target)
                total += float(nll[ok].sum())
                count += int(ok.sum())
    return total / max(count, 1)


def param_groups(
    *, model: mdl.MsaNet, stage: int, encoders: str, new_lr: float | None = None
) -> list[dict]:
    """Return the optimiser groups: base rate, loaded encoders, `proj_s`, and the new parameters at 5e-4.

    A copy of `e00713_train.param_groups` (lines 605 to 633) with a group for the 36,888 new
    parameters, which are left out of the base group.
    """
    base_lr = tr.PRETRAIN_LR if stage == 1 else tr.FINE_TUNE_LR
    enc = {id(p) for p in model.encoder_parameters()}
    spread = {id(p) for p in model.proj_s.parameters()}
    new = {id(p) for p in model.neighbour_parameters()} if isinstance(model, f4.F4Net) else set()
    groups: list[dict] = []
    rest = [
        p
        for p in model.parameters()
        if p.requires_grad and id(p) not in enc and id(p) not in spread and id(p) not in new
    ]
    groups.append({"params": rest, "lr": base_lr})
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
    if new:
        assert isinstance(model, f4.F4Net)
        groups.append(
            {"params": model.neighbour_parameters(), "lr": NEW_LR if new_lr is None else new_lr}
        )
    for g in groups:
        g["base_lr"] = g["lr"]
    return groups


def train_stage(
    *, model: mdl.MsaNet, world: F4World, stage: int, samples: F4Samples, seed: int, encoders: str,
    max_passes: int = 60, log: tr.StageLog | None = None, source: fb.F4Source | None = None,
    ensemble: bool = True, new_lr: float | None = None,
) -> tr.StageLog:  # fmt: skip
    """Train `model` on stage 2 or 3 in place and leave it at the state of the best validation loss.

    A copy of `e00713_train.train_stage` (lines 636 to 698) with the neighbour tensors. The global
    random stream is seeded as experiment 713 seeds it, `torch.manual_seed(seed * 7 + stage)`, and the
    neighbour path draws from its own generator, so the draws of the trunk are the draws of `M*`.
    """
    assert stage in {2, 3} or source is not None, (
        "stage 1 needs the explicit source of a synthetic world"
    )
    neighbours = world.spec.neighbour_path
    started = time.monotonic()
    log = log or tr.StageLog(stage=stage)
    source = source or stage_source(world=world, samples=samples, stage=stage)
    cut = tr.target_end(world=world.base, stage=stage)
    torch.manual_seed(seed * 7 + stage)
    optimiser = torch.optim.AdamW(
        param_groups(model=model, stage=stage, encoders=encoders, new_lr=new_lr),
        weight_decay=tr.WEIGHT_DECAY,
    )
    train_idx = np.nonzero(~samples.is_val)[0]
    best, best_state, stale, step = float("inf"), None, 0, 0
    for epoch in range(max_passes):
        model.train()
        if neighbours:
            assert isinstance(model, f4.F4Net)
            model.seed_neighbour_path(key=(seed * KEY_SCALE + stage) * KEY_SCALE + epoch)
        rng = np.random.default_rng([seed, stage, epoch])
        order = rng.permutation(train_idx)
        lead_all = rng.integers(0, lay.N_LEAD_DAYS, size=len(order))
        for start in range(0, len(order), tr.BATCH):
            sel = order[start : start + tr.BATCH]
            native = samples.c_idx[sel] - lay.NATIVE_OFFSET_STEPS
            drop = block_dropout_masks(origin_idx=native, seed=(seed * 1009 + stage) * 1009 + epoch)
            slots = (
                fb.drop_slots_for(n=len(sel), key=(seed, stage, epoch, start))
                if neighbours
                else None
            )
            b = batch_for(
                source=source, samples=samples, index=sel, lead=lead_all[start : start + tr.BATCH],
                ensemble=ensemble, cut=cut, dropout=drop, drop_slots=slots,
            )  # fmt: skip
            w = tr.token_weights(target_idx=b.host.target_idx, world=world.base, stage=stage)
            ok = torch.isfinite(b.host.target)
            w = w * ok
            nll = mdl.per_token_nll(
                out=run_model(model=model, batch=b, neighbours=neighbours),
                target=torch.nan_to_num(b.host.target),
            )
            loss = (w * nll).sum() / w.sum().clamp_min(1e-9)
            step += 1
            for g in optimiser.param_groups:
                g["lr"] = g["base_lr"] * min(1.0, step / tr.WARMUP_STEPS)
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tr.GRAD_CLIP)
            optimiser.step()
        val = validation_loss(
            model=model, source=source, samples=samples, ensemble=ensemble, cut=cut, neighbours=neighbours
        )  # fmt: skip
        log.val_history.append(val)
        log.passes = epoch + 1
        if val < best - 1e-5:
            best, stale = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= tr.PATIENCE:
                break
    else:
        log.cap_hit = True
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    log.best_val = best
    log.seconds = time.monotonic() - started
    return log
