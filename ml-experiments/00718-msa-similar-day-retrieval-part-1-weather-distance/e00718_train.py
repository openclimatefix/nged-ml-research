# ruff: noqa: E501, PLR0915
"""Experiment 718: the three-stage training loop with row tables (design sections 5 and 6).

The loop is a copy of `e00713_train.train_stage` (the function of experiment 713's
`e00713_train.py` at commit 5deae9ec8) with five changes: the source is an `F9Source` whose row
tables choose the history rows; the network is `F9Net`; the batch carries the age buckets; stage
3 applies a second schedule of synthetic shifts over the older rows (section 5); and `accumulate`
splits an optimiser step into half-batches whose losses are normalised by the weight sum of the
full batch (the memory rule of section 11). Everything else is experiment 713's: the optimiser
groups, the learning rates, the warm-up, the gradient clip, the early stopping, the lead-day
draws, the dropout stream, the target cuts, and the recency weights. The functions that need no
change are imported.
"""

import time
from collections.abc import Callable

import e00558_lags as lags  # ty: ignore[unresolved-import]
import e00713_layout as lay713  # ty: ignore[unresolved-import]
import e00713_model as mdl  # ty: ignore[unresolved-import]
import e00713_train as tr  # ty: ignore[unresolved-import]
import e00718_batch as fb
import e00718_model as f9
import numpy as np
import torch  # ty: ignore[unresolved-import]

SECOND_SCHEDULE_DAYS = 380
"""The second schedule starts this many days before the first target (the 379-day pool plus one)."""
SECOND_SCHEDULE_SALT = 1
"""The second generator is seeded by (month, 1)."""


def synthetic_source(
    *, world: tr.World, samples: tr.SampleSet, tables: dict, arm: str
) -> fb.F9Source:
    """Return the stage 3 source: history power with the lag-only shifts of two schedules, targets
    clean.

    Experiment 713's call is kept unchanged (`first = c.min - 768`), so every row that `M*` read
    gets `M*`'s shifts. The second schedule adds events on `[c.min - 48 * 380, c.min - 768)`, and
    on `[0, c.min - 768)` for arm `UX`, from a generator seeded by (month, 1) at the same
    coverage share. The ranges are disjoint, so the schedules never overlap.
    """  # noqa: D205
    import e00581_bank as bank  # ty: ignore[unresolved-import]

    first_target = int(samples.c_idx.min())
    first = first_target - tr.SYNTHETIC_FIRST_MARGIN
    last = int(samples.c_idx.max()) + 672
    month_seed = int(world.month.replace("-", ""))
    fixed, _ = bank.synthetic_clean(clean=world.clean, first=first, last=last, seed=month_seed)
    lo = 0 if arm == "UX" else max(0, first_target - lay713.STEPS_PER_DAY * SECOND_SCHEDULE_DAYS)
    if first > lo:
        fixed, _ = bank.synthetic_clean(
            clean=fixed, first=lo, last=first, seed=[month_seed, SECOND_SCHEDULE_SALT]
        )  # ty: ignore[invalid-argument-type]
    z = (fixed - world.target_mean[:, None]) / world.target_std[:, None]
    power_in = torch.from_numpy(z.astype(np.float32)).to(world.source.device)
    return fb.F9Source(base=world.source.with_power_in(power_in=power_in), tables=tables)


def batch_for(
    *, source: fb.F9Source, samples: tr.SampleSet, index: np.ndarray, lead: np.ndarray, ensemble: bool,
    cut: int, dropout: tuple[np.ndarray, np.ndarray] | None,
) -> fb.F9Assembled:  # fmt: skip
    """Assemble the batch of `samples[index]` at lead days `lead`."""
    dev = source.device
    to = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)  # noqa: E731
    return source.assemble(
        primary=to(samples.primary[index]), c_idx=to(samples.c_idx[index]), lead_day=to(lead),
        ensemble=ensemble,
        mask_all_power=None if dropout is None else to(dropout[0]),
        mask_newest_row=None if dropout is None else to(dropout[1]),
        target_end=cut,
    )  # fmt: skip


def validation_loss(
    *, model: f9.F9Net, source: fb.F9Source, samples: tr.SampleSet, ensemble: bool, cut: int
) -> float:
    """Return the unweighted mixture negative log-likelihood on the held-out (sample, lead day) pairs."""
    model.eval()
    idx = np.nonzero(samples.is_val)[0]
    total, count = 0.0, 0
    with torch.no_grad():
        for lead_day in tr.VALIDATION_LEAD_DAYS:
            for start in range(0, len(idx), tr.BATCH * 2):
                sel = idx[start : start + tr.BATCH * 2]
                b = batch_for(
                    source=source, samples=samples, index=sel,
                    lead=np.full(len(sel), lead_day, dtype=np.int64), ensemble=ensemble, cut=cut, dropout=None,
                )  # fmt: skip
                nll = mdl.per_token_nll(out=model(b.inputs).out, target=torch.nan_to_num(b.target))
                ok = torch.isfinite(b.target)
                total += float(nll[ok].sum())
                count += int(ok.sum())
    return total / max(count, 1)


def train_stage(
    *, model: f9.F9Net, world: tr.World, tables: dict, arm: str, stage: int, samples: tr.SampleSet,
    seed: int, encoders: str, max_passes: int = 60, log: tr.StageLog | None = None,
    patience: int = tr.PATIENCE, on_pass: Callable[[int, f9.F9Net], None] | None = None,
    accumulate: int = 1, max_steps: int | None = None,
) -> tr.StageLog:  # fmt: skip
    """Train `model` on one stage in place and leave it at the state of the best validation loss.

    `accumulate` splits each optimiser step of 256 samples into that many half-batches, each loss
    divided by the weight sum of the full 256 (the block-dropout draw is per origin day and the
    same). `max_steps` stops a pass early and exists only for the equality check against
    experiment 713's loop.
    """
    started = time.monotonic()
    log = log or tr.StageLog(stage=stage)
    ensemble = stage >= 2
    base = (
        synthetic_source(world=world, samples=samples, tables=tables, arm=arm)
        if stage == 3
        else fb.F9Source(base=world.source, tables=tables)
    )
    cut = tr.target_end(world=world, stage=stage)
    torch.manual_seed(seed * 7 + stage)
    optimiser = torch.optim.AdamW(
        tr.param_groups(model=model, stage=stage, encoders=encoders), weight_decay=tr.WEIGHT_DECAY
    )  # ty: ignore[invalid-argument-type]
    train_idx = np.nonzero(~samples.is_val)[0]
    best, best_state, stale, step = float("inf"), None, 0, 0
    for epoch in range(max_passes):
        model.train()
        rng = np.random.default_rng([seed, stage, epoch])
        order = rng.permutation(train_idx)
        lead_all = rng.integers(0, lay713.N_LEAD_DAYS, size=len(order))
        for start in range(0, len(order), tr.BATCH):
            sel = order[start : start + tr.BATCH]
            native = samples.c_idx[sel] - lay713.NATIVE_OFFSET_STEPS
            drop = lags.block_dropout_masks(
                origin_idx=native, seed=(seed * 1009 + stage) * 1009 + epoch
            )
            parts = np.array_split(np.arange(len(sel)), accumulate)
            batches = [
                batch_for(source=base, samples=samples, index=sel[p], lead=lead_all[start : start + tr.BATCH][p], ensemble=ensemble, cut=cut, dropout=(drop[0][p], drop[1][p]))
                for p in parts
            ]  # fmt: skip
            weights = [
                tr.token_weights(target_idx=b.target_idx, world=world, stage=stage)
                * torch.isfinite(b.target)
                for b in batches
            ]
            total_w = sum(float(w.sum()) for w in weights)
            optimiser.zero_grad()
            for b, w in zip(batches, weights, strict=True):
                nll = mdl.per_token_nll(out=model(b.inputs).out, target=torch.nan_to_num(b.target))
                loss = (
                    (w * nll).sum() / max(total_w, 1e-9)
                    if accumulate > 1
                    else (w * nll).sum() / w.sum().clamp_min(1e-9)
                )
                loss.backward()
            step += 1
            for g in optimiser.param_groups:
                g["lr"] = g["base_lr"] * min(1.0, step / tr.WARMUP_STEPS)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tr.GRAD_CLIP)
            optimiser.step()
            if max_steps is not None and step >= max_steps:
                break
        val = validation_loss(model=model, source=base, samples=samples, ensemble=ensemble, cut=cut)
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
        if max_steps is not None and step >= max_steps:
            break
    else:
        log.cap_hit = True
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    log.best_val = best
    log.seconds = time.monotonic() - started
    return log
