# ruff: noqa: E501
"""Experiment 713: the three-stage training loop of the MSA network (design 6.3).

The stages follow experiment 667's recipe, with one stated departure: the trunk is kept from stage 1
into stage 2. A pass visits every (Primary, origin day) pair of the stage once, with one lead day
drawn uniformly per pair and seeded by (seed, pass). Early stopping uses the held-out origin days of
`e00422_prep.purge_masks`, grouped by origin day, at lead days 1, 2, 4, 7, 10, and 14.
The loop takes a
`World` so that the synthetic world of `e00713_known_answer.py` and the real folds share it.
"""

import os
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00422_prep as prep422  # ty: ignore[unresolved-import]
import e00558_lags as lags  # ty: ignore[unresolved-import]
import e00713_batch as bt
import e00713_layout as lay
import e00713_model as mdl
import numpy as np
import torch  # ty: ignore[unresolved-import]

BATCH: Final = 256
WARMUP_STEPS: Final = 300
WEIGHT_DECAY: Final = 1e-3
GRAD_CLIP: Final = 1.0
PATIENCE: Final = 8
VALIDATION_LEAD_DAYS: Final = (0, 1, 3, 6, 9, 13)
"""Lead days 1, 2, 4, 7, 10, and 14, as zero-based lead-day indices."""
PRETRAIN_LR: Final = 5e-4
FINE_TUNE_LR: Final = 2e-4
LOADED_ENCODER_LR: Final = 2e-4
SPREAD_LR: Final = 5e-4
WINDOW_DAYS: Final = 730
EMBARGO_STAGE: Final = {1: 0, 2: 3, 3: 7}
"""Days before the cutoff (beyond 3.5 h) whose targets each stage masks."""
RECENCY_HALF_LIFE_DAYS: Final = 90.0
DOWN_WEIGHT_DAYS: Final = (7, 14)
CUTOFF_AGE_STEPS: Final = 7
"""3.5 hours in half-hours: targets stamped after the cutoff minus this are masked in every stage."""
SYNTHETIC_FIRST_MARGIN: Final = 768
"""16 days of half-hours: the synthetic shifts start this far before the first target (design 4.4)."""


@dataclass
class World:
    """What a training job reads.

    Attributes:
        source: The dense arrays on the device.
        cutoff_idx: Grid index of the fold's cutoff.
        clean: (16, G) cleaned power over capacity, for the synthetic shifts of stage 3.
        target_mean: (16,) mean of the power over capacity.
        target_std: (16,) standard deviation of the power over capacity.
        month: The fold month (the seed of the synthetic shifts).
        placebo_pool: (16, G // 48) bool: whether the 24-hour block starting at 12:00 holds a reading.
    """

    source: bt.Source
    cutoff_idx: int
    clean: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    month: str
    placebo_pool: np.ndarray | None = None


@dataclass
class SampleSet:
    """The (Primary, origin day) pairs of one stage."""

    primary: np.ndarray
    c_idx: np.ndarray
    is_val: np.ndarray
    ages: np.ndarray | None = None  # (N, 14) placebo ages, -1 where absent

    def __len__(self) -> int:
        """Return the number of pairs."""
        return len(self.primary)


@dataclass
class StageLog:
    """What a stage reports."""

    stage: int
    passes: int = 0
    best_val: float = float("inf")
    cap_hit: bool = False
    seconds: float = 0.0
    val_history: list[float] = field(default_factory=list)


def block_pool(*, clean: np.ndarray) -> np.ndarray:
    """Return, per Primary and 24-hour block starting at 12:00 UTC, whether the block has a reading.

    Block `b` covers grid indices `[48 b + 24, 48 b + 72)` (12:00 UTC of day `b` to 11:30 the next day),
    which is the block whose first column is the half-hour stamped 12:00 on day `b`.
    """
    n_days = clean.shape[1] // lay.STEPS_PER_DAY - 1
    return np.stack(
        [
            np.isfinite(clean[:, 48 * b + 24 : 48 * b + 72]).any(axis=1)
            for b in range(n_days)
        ],
        axis=1,
    )  # fmt: skip


def placebo_ages_for(*, primary: np.ndarray, c_idx: np.ndarray, pool: np.ndarray) -> np.ndarray:
    """Return (N, 14) placebo ages per pair (-1 where the slot has no block).

    The block of age `a` before the contract origin `c` (12:00 UTC of day `d = c // 48`) starts at the
    12:00 of day `d - a`, so it is block `d - a` of `pool`.
    """
    out = np.full((len(primary), lay.N_HISTORY_ROWS), -1, dtype=np.int64)
    ages_all = np.arange(lay.PLACEBO_MIN_AGE, lay.PLACEBO_MAX_AGE + 1)
    for i, (p, c) in enumerate(zip(primary, c_idx, strict=True)):
        day = int(c) // lay.STEPS_PER_DAY
        block = day - ages_all
        ok = (block >= 0) & (block < pool.shape[1])
        avail = np.zeros(len(ages_all), dtype=bool)
        avail[ok] = pool[p, block[ok]]
        got = lay.placebo_ages(primary=int(p), origin_day=day, available=avail)
        out[i, : len(got)] = got
    return out


def candidate_epoch_days() -> int:
    """Return the days from the grid start to the Candidate's day 0 (`e00397_prep.FIRST_RUN`, 2024-04-01).

    The Candidate counts the origin day from `FIRST_RUN` before it cuts the held-out and purged weeks
    (`e00510_common.py`), so the MSA arms count the same way and hold out the same days (code review 1, S2).
    """
    import e00397_prep as prep397  # ty: ignore[unresolved-import]
    import e00422_prep as prep422  # ty: ignore[unresolved-import]

    grid_start = datetime.fromtimestamp(prep422.GRID_START_US / 1e6, tz=UTC)
    days = (prep397.FIRST_RUN - grid_start).days
    assert days == 1674, days
    return days


def make_samples(
    *, world: World, stage: int, placebo: bool, primaries: tuple[int, ...] | None = None
) -> SampleSet:
    """Return the pairs of `stage`.

    Stage 1 takes every origin day from the first with a full history to the last whose contract
    origin is at or before the cutoff. Stages 2 and 3 take the pairs with ensemble rows in the 730 days
    before the cutoff. A pair needs at least one finite target in its 14 target rows.
    """
    src = world.source
    first_c = lay.STEPS_PER_DAY * (lay.MAX_AGE + 1) + 24
    days = np.arange(first_c // 48, world.cutoff_idx // 48)
    c_all = days * 48 + 24
    c_all = c_all[c_all <= world.cutoff_idx]
    prim, cc = np.meshgrid(
        np.arange(bt.N_PRIMARIES) if primaries is None else np.array(primaries),
        c_all,
        indexing="ij",
    )
    prim, cc = prim.ravel(), cc.ravel()
    if stage >= 2:
        assert src.ens is not None
        lo = world.cutoff_idx - WINDOW_DAYS * lay.STEPS_PER_DAY
        day_idx = cc // 48 - src.ens.first_day
        ok = (cc >= lo) & (day_idx >= 0) & (day_idx < src.ens.filled.shape[1])
        have = src.ens.filled.any(dim=2).cpu().numpy()
        ok &= have[prim, np.clip(day_idx, 0, have.shape[1] - 1)]
        prim, cc = prim[ok], cc[ok]
    finite = np.isfinite(world.clean).astype(np.int64)
    csum = np.concatenate([np.zeros((finite.shape[0], 1), np.int64), finite.cumsum(axis=1)], axis=1)
    end = np.minimum(cc + 672, world.clean.shape[1])
    has_target = (csum[prim, end] - csum[prim, np.minimum(cc, end)]) > 0
    prim, cc = prim[has_target], cc[has_target]
    held, purged = prep422.purge_masks(origin_day=cc // lay.STEPS_PER_DAY - candidate_epoch_days())
    keep = ~purged
    prim, cc, held = prim[keep], cc[keep], held[keep]
    ages = None
    if placebo:
        assert world.placebo_pool is not None
        ages = placebo_ages_for(primary=prim, c_idx=cc, pool=world.placebo_pool)
    return SampleSet(primary=prim, c_idx=cc, is_val=held, ages=ages)


def target_end(*, world: World, stage: int) -> int:
    """Return the grid index at and after which a stage masks its targets."""
    return world.cutoff_idx - CUTOFF_AGE_STEPS - EMBARGO_STAGE[stage] * lay.STEPS_PER_DAY


def token_weights(*, target_idx: torch.Tensor, world: World, stage: int) -> torch.Tensor:
    """Return the loss weight of every target token (stage 3: recency, half weight for days 7 to 14)."""
    if stage < 3:
        return torch.ones(target_idx.shape, device=target_idx.device)
    age_days = (world.cutoff_idx - target_idx).float() / lay.STEPS_PER_DAY
    weight = 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS)
    lo, hi = DOWN_WEIGHT_DAYS
    return torch.where((age_days >= lo) & (age_days < hi), weight * 0.5, weight)


def synthetic_source(*, world: World, samples: SampleSet) -> bt.Source:
    """Return the stage 3 source: history power with experiment 581's lag-only shifts, targets clean.

    `e00667_cand.py` draws the shifts over the targets' index range, which would leave the oldest
    history rows of the first targets without shifts. This function starts 16 days earlier.
    """
    import e00581_bank as bank  # ty: ignore[unresolved-import]

    first = int(samples.c_idx.min()) - SYNTHETIC_FIRST_MARGIN
    last = int(samples.c_idx.max()) + 672
    fixed, _ = bank.synthetic_clean(
        clean=world.clean, first=first, last=last, seed=int(world.month.replace("-", ""))
    )
    z = (fixed - world.target_mean[:, None]) / world.target_std[:, None]
    power_in = torch.from_numpy(z.astype(np.float32)).to(world.source.device)
    return world.source.with_power_in(power_in=power_in)


def batch_for(
    *, source: bt.Source, samples: SampleSet, index: np.ndarray, lead: np.ndarray, ensemble: bool,
    cut: int, dropout: tuple[np.ndarray, np.ndarray] | None,
) -> bt.Assembled:  # fmt: skip
    """Assemble the batch of `samples[index]` at lead days `lead`."""
    dev = source.device
    to = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)  # noqa: E731
    ages = age_exists = None
    if samples.ages is not None:
        a = samples.ages[index]
        age_exists = to(a >= 0)
        ages = to(np.where(a >= 0, a, lay.PLACEBO_MIN_AGE))
    return source.assemble(
        primary=to(samples.primary[index]), c_idx=to(samples.c_idx[index]), lead_day=to(lead),
        ensemble=ensemble, ages=ages, age_exists=age_exists,
        mask_all_power=None if dropout is None else to(dropout[0]),
        mask_newest_row=None if dropout is None else to(dropout[1]),
        target_end=cut,
    )  # fmt: skip


def validation_loss(
    *, model: mdl.MsaNet, source: bt.Source, samples: SampleSet, ensemble: bool, cut: int
) -> float:
    """Return the unweighted mixture negative log-likelihood on the held-out (sample, lead day) pairs."""
    model.eval()
    idx = np.nonzero(samples.is_val)[0]
    total, count = 0.0, 0
    with torch.no_grad():
        for lead_day in VALIDATION_LEAD_DAYS:
            for start in range(0, len(idx), BATCH * 2):
                sel = idx[start : start + BATCH * 2]
                b = batch_for(
                    source=source, samples=samples, index=sel,
                    lead=np.full(len(sel), lead_day, dtype=np.int64), ensemble=ensemble, cut=cut,
                    dropout=None,
                )  # fmt: skip
                nll = mdl.per_token_nll(out=model(b.inputs).out, target=torch.nan_to_num(b.target))
                ok = torch.isfinite(b.target)
                total += float(nll[ok].sum())
                count += int(ok.sum())
    return total / max(count, 1)


def param_groups(
    *, model: mdl.MsaNet, stage: int, encoders: str
) -> list[dict]:  # fmt: skip
    """Return the optimiser groups: base rate, loaded encoders at 2e-4, and `P_s` at 5e-4 (stage 2)."""
    base_lr = PRETRAIN_LR if stage == 1 else FINE_TUNE_LR
    enc = {id(p) for p in model.encoder_parameters()}
    spread = {id(p) for p in model.proj_s.parameters()}
    groups: list[dict] = []
    rest = [
        p
        for p in model.parameters()
        if p.requires_grad and id(p) not in enc and id(p) not in spread
    ]
    groups.append({"params": rest, "lr": base_lr})
    if encoders == "trained":
        groups.append(
            {
                "params": model.encoder_parameters(),
                "lr": LOADED_ENCODER_LR if stage == 1 else FINE_TUNE_LR,
            }
        )
    elif encoders == "random":
        groups[0]["params"] = rest + model.encoder_parameters()
    groups.append(
        {"params": list(model.proj_s.parameters()), "lr": SPREAD_LR if stage == 2 else base_lr}
    )
    for g in groups:
        g["base_lr"] = g["lr"]
    return groups


def train_stage(
    *, model: mdl.MsaNet, world: World, stage: int, samples: SampleSet, seed: int,
    encoders: str, max_passes: int = 60, log: StageLog | None = None,
) -> StageLog:  # fmt: skip
    """Train `model` on one stage in place and leave it at the state of the best validation loss."""
    started = time.monotonic()
    log = log or StageLog(stage=stage)
    ensemble = stage >= 2
    source = synthetic_source(world=world, samples=samples) if stage == 3 else world.source
    cut = target_end(world=world, stage=stage)
    torch.manual_seed(seed * 7 + stage)
    optimiser = torch.optim.AdamW(
        param_groups(model=model, stage=stage, encoders=encoders), weight_decay=WEIGHT_DECAY
    )
    train_idx = np.nonzero(~samples.is_val)[0]
    best, best_state, stale, step = float("inf"), None, 0, 0
    for epoch in range(max_passes):
        model.train()
        rng = np.random.default_rng([seed, stage, epoch])
        order = rng.permutation(train_idx)
        lead_all = rng.integers(0, lay.N_LEAD_DAYS, size=len(order))
        for start in range(0, len(order), BATCH):
            sel = order[start : start + BATCH]
            native = samples.c_idx[sel] - lay.NATIVE_OFFSET_STEPS
            drop = lags.block_dropout_masks(
                origin_idx=native, seed=(seed * 1009 + stage) * 1009 + epoch
            )
            b = batch_for(
                source=source, samples=samples, index=sel, lead=lead_all[start : start + BATCH],
                ensemble=ensemble, cut=cut, dropout=drop,
            )  # fmt: skip
            w = token_weights(target_idx=b.target_idx, world=world, stage=stage)
            ok = torch.isfinite(b.target)
            w = w * ok
            nll = mdl.per_token_nll(out=model(b.inputs).out, target=torch.nan_to_num(b.target))
            loss = (w * nll).sum() / w.sum().clamp_min(1e-9)
            step += 1
            for g in optimiser.param_groups:
                g["lr"] = g["base_lr"] * min(1.0, step / WARMUP_STEPS)
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimiser.step()
        val = validation_loss(
            model=model, source=source, samples=samples, ensemble=ensemble, cut=cut
        )
        log.val_history.append(val)
        log.passes = epoch + 1
        if val < best - 1e-5:
            best, stale = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    else:
        log.cap_hit = True
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    log.best_val = best
    log.seconds = time.monotonic() - started
    return log


def save_atomic(*, state: dict, path: Path) -> None:
    """Write a state file atomically (temporary name, then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(
        path.name + f".{os.getpid()}.tmp"
    )  # two jobs on one state cannot share a temporary file
    torch.save(state, tmp)
    tmp.rename(path)
