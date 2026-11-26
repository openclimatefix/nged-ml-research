# ruff: noqa: E501, PLR0915
"""Experiment 733: the training loops (torch).

Two loops share the batch and loss helpers:

- `train_stage`: stages 2 and 3 of experiment 713's recipe on the real folds, copied from experiment
  716's `e00716_train.py` (the batch, the parameter groups, and the clip differ). The parameters of M*
  keep 713's groups, the warm-up, and a clip of 1.0 over them; the new parameters (`u_*`) form
  their own groups with no warm-up, their own clip of 1.0, a gain at 3e-3 without weight decay,
  and every other new parameter at 2e-3. The two mask draws hide the channels of the age-2 row and
  part of the age-3 row on 10% of training samples, and of ages 2 to 5 on another 10%.
- `train_synthetic`: one stage on the synthetic forecasting world: AdamW at 5e-4 with weight decay
  1e-3, batch 256, gradient clip 1.0, at most 80 passes with a patience of 20 on the validation loss,
  which is the mixture negative log-likelihood plus the sign head's cross-entropy, each with weight 1.0.

Both loops record, for every pass, the training loss and the validation loss (`History`), and the
pass whose state was stored, so that the loss curves are drawn from stored data and not from a log.
Every device-dependent line is a function of the tensors' own device, so the CPU path and the CUDA
path run the same code, and `to_numpy` is the only place a tensor becomes an array (`detach`, `cpu`,
then `numpy`).
"""

import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Final

import e00733_core as core
import e00733_inputs as inputs
import e00733_leakage as leak
import e00733_model as model
import e00733_paths as paths  # noqa: F401
import e00733_synth as sy
import numpy as np

# isort: split
import e00713_layout as lay  # ty: ignore[unresolved-import]
import e00713_train as tr  # ty: ignore[unresolved-import]
import torch  # ty: ignore[unresolved-import]
from e00558_lags import block_dropout_masks  # ty: ignore[unresolved-import]

NEW_RATES: Final[dict[str, tuple[float, float]]] = {"u_gamma": (3e-3, 0.0)}
"""Learning rate and weight decay of the gain; every other new parameter trains at 2e-3 with decay 1e-3."""
NEW_DEFAULT: Final = (2e-3, tr.WEIGHT_DECAY)
VALIDATION_LEADS_SYNTHETIC: Final = (1, 3, 6, 9)
"""Zero-based lead indices (days 2, 4, 7, and 10) of the synthetic validation loss."""


def to_numpy(*, tensor: torch.Tensor) -> np.ndarray:
    """Return a tensor as a NumPy array: detach, move to the host, then convert.

    A CUDA tensor cannot be converted directly.
    """
    return tensor.detach().cpu().numpy()


@dataclass
class History:
    """The per-pass losses of one fit, for the loss curves.

    Attributes:
        stage: The stage (2 or 3), or 0 for the one-stage synthetic loop.
        train: The mean training loss of each pass.
        val: The validation loss of each pass.
        val_nll: The mixture part of the validation loss (synthetic loop; empty otherwise).
        val_bce: The sign head's part of the validation loss (synthetic loop; empty otherwise).
        stored_pass: The pass (1-based) whose state was stored (0 when no pass ran).
        cap_hit: Whether the loop ended at the cap of passes.
        still_falling: Whether the validation loss fell in the last pass before the cap.
        seconds: The wall time of the loop.
    """

    stage: int
    train: list[float] = field(default_factory=list)
    val: list[float] = field(default_factory=list)
    val_nll: list[float] = field(default_factory=list)
    val_bce: list[float] = field(default_factory=list)
    stored_pass: int = 0
    cap_hit: bool = False
    still_falling: bool = False
    seconds: float = 0.0

    def as_dict(self) -> dict[str, object]:
        """Return the JSON content."""
        return asdict(self)


# --------------------------------------------------------------------------- the real-fold batch


@dataclass
class UntAssembled:
    """One assembled real-fold batch with the untangled channels."""

    inputs: model.UntInputs
    target: torch.Tensor
    target_idx: torch.Tensor
    hist_idx: torch.Tensor
    c_idx: torch.Tensor
    primary: torch.Tensor
    run_init_us: torch.Tensor | None = None


def day_start_us(*, c_idx: np.ndarray) -> np.ndarray:
    """Return 00:00 UTC of the origin's day in microseconds.

    This is the latest valid time that the plan's example row prints for the other target-row
    columns.
    """
    stamp = lay.GRID_START_US + np.asarray(c_idx, dtype=np.int64) * lay.STEP_US
    return stamp // lay.DAY_US * lay.DAY_US


class UntSource:
    """A 713 `Source` plus the untangled channels of one arm (the counterpart of 716's `PvSource`)."""

    def __init__(self, *, source: object, tensors: inputs.ChannelTensors) -> None:
        """Hold the wrapped source and the channel tensors; both sit on one device."""
        self.source, self.tensors = source, tensors
        self.device = source.device  # ty: ignore[unresolved-attribute]

    def with_power_in(self, *, power_in: torch.Tensor) -> UntSource:
        """Return a source whose history power is `power_in` (stage 3's synthetic shifts); the channels are unchanged."""
        return UntSource(
            source=self.source.with_power_in(power_in=power_in),  # ty: ignore[unresolved-attribute]
            tensors=self.tensors,
        )

    @property
    def ens(self) -> object:
        """Return the wrapped source's ensemble table."""
        return self.source.ens  # ty: ignore[unresolved-attribute]

    @property
    def steps(self) -> torch.Tensor:
        """Return the 48 column indices of the wrapped source."""
        return self.source.steps  # ty: ignore[unresolved-attribute]

    @property
    def power_z(self) -> torch.Tensor:
        """Return the wrapped source's standardised power."""
        return self.source.power_z  # ty: ignore[unresolved-attribute]

    def assemble(
        self, *, primary: torch.Tensor, c_idx: torch.Tensor, lead_day: torch.Tensor,
        mask_all_power: torch.Tensor | None = None, mask_newest_row: torch.Tensor | None = None,
        target_end: int | None = None, cams_rule: torch.Tensor | None = None,
        convention: torch.Tensor | None = None, check: bool = True,
    ) -> UntAssembled:  # fmt: skip
        """Return the batch: 713's inputs and the channels of the 14 history rows (none for a target row).

        The target row's weather is 713's ensemble row of the origin's own 00Z run, and the batch is
        checked row by row: the run was initialised at 00:00 UTC on day D and was available at the
        native origin, and the channel tensors are finite, flagged, and present at D8 only.
        """
        out = self.source.assemble(  # ty: ignore[unresolved-attribute]
            primary=primary, c_idx=c_idx, lead_day=lead_day, ensemble=True,
            mask_all_power=mask_all_power, mask_newest_row=mask_newest_row, target_end=target_end,
        )  # fmt: skip
        hist = inputs.gather(
            tensors=self.tensors, primary=primary, hist_idx=out.hist_idx, c_idx=c_idx,
            power_visible=out.inputs.hist_visible, cams_rule=cams_rule, convention=convention,
        )  # fmt: skip
        batch_inputs = model.UntInputs(
            **out.inputs.__dict__, hist_u=hist.channels, hist_u_ok=hist.ok
        )
        run = getattr(self.ens, "run_us", None)
        run_init = None
        if (
            run is not None
        ):  # the issue time of the run behind each (Primary, origin day); 0 where absent
            day = (c_idx // lay.STEPS_PER_DAY - self.ens.first_day)[:, None]  # ty: ignore[unresolved-attribute]
            run_init = run[primary[:, None], day][:, 0]
        batch = UntAssembled(
            inputs=batch_inputs, target=out.target, target_idx=out.target_idx,
            hist_idx=out.hist_idx, c_idx=c_idx, primary=primary, run_init_us=run_init,
        )  # fmt: skip
        if check:
            check_batch(batch=batch)
        return batch


def check_batch(*, batch: UntAssembled) -> None:
    """Run the per-batch assertions.

    The assertions cover the origin-time weather of every target token, the history order, and the
    channel ranges.

    Args:
        batch: The assembled batch.

    Raises:
        CorrectnessFailureError: When a target token reads a run that is not the origin's own 00Z run
            or was available only after the native origin, or the history lies after the target.
    """
    if bool((batch.hist_idx >= batch.target_idx[:, :1, None]).any()):
        msg = "a history token lies at or after the first target token"
        raise leak.CorrectnessFailureError(msg)
    if batch.run_init_us is not None:
        c = to_numpy(tensor=batch.c_idx)
        filled = to_numpy(tensor=torch.isfinite(batch.target))
        leak.assert_origin_time_weather(
            c_idx=c, run_init_us=to_numpy(tensor=batch.run_init_us),
            target_idx=to_numpy(tensor=batch.target_idx), filled=filled,
            other_latest_valid_us=day_start_us(c_idx=c),
        )  # fmt: skip
    ch = batch.inputs.hist_u
    ok = batch.inputs.hist_u_ok
    if not bool(torch.isfinite(ch).all()) or bool((ch[~ok] != 0).any()):
        msg = "a history channel is non-finite, or a masked token carries a value"
        raise leak.CorrectnessFailureError(msg)


def batch_for(
    *, source: UntSource, samples: object, index: np.ndarray, lead: np.ndarray, cut: int,
    dropout: tuple[np.ndarray, np.ndarray] | None = None,
    masks: tuple[np.ndarray, np.ndarray] | None = None,
) -> UntAssembled:  # fmt: skip
    """Assemble the batch of `samples[index]` at lead days `lead`.

    `dropout` is experiment 713's pair, and `masks` are the channel masks.
    """
    dev = source.device

    def to(a: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(a)).to(dev)

    return source.assemble(
        primary=to(samples.primary[index]), c_idx=to(samples.c_idx[index]), lead_day=to(lead),  # ty: ignore[unresolved-attribute]
        mask_all_power=None if dropout is None else to(dropout[0]),
        mask_newest_row=None if dropout is None else to(dropout[1]),
        cams_rule=None if masks is None else to(masks[0]),
        convention=None if masks is None else to(masks[1]),
        target_end=cut,
    )  # fmt: skip


# --------------------------------------------------------------------------- shared helpers


def param_groups(
    *, net: model.UntangledNet, stage: int, encoders: str, new_lr_scale: float = 1.0
) -> list[dict]:
    """Return the optimiser groups: 713's three for the parameters of M*, then one per rate for the new parameters.

    Every group carries `weight_decay`, a `base_lr`, and `warmup` (true for the trunk, false for the
    new parameters). `new_lr_scale` multiplies the new rates (0 is the check that the new path never
    moves the trunk).
    """
    base_lr = tr.PRETRAIN_LR if stage == 1 else tr.FINE_TUNE_LR
    trunk = {n: p for n, p in net.named_parameters() if not n.startswith(model.NEW_PREFIXES)}
    enc = {id(p) for p in net.encoder_parameters()}
    spread = {id(p) for p in net.proj_s.parameters()}
    rest = [
        p for p in trunk.values() if p.requires_grad and id(p) not in enc and id(p) not in spread
    ]
    groups: list[dict] = [{"params": rest, "lr": base_lr}]
    if encoders == "trained":
        groups.append(
            {
                "params": net.encoder_parameters(),
                "lr": tr.LOADED_ENCODER_LR if stage == 1 else tr.FINE_TUNE_LR,
            }
        )
    elif encoders == "random":
        groups[0]["params"] = rest + net.encoder_parameters()
    groups.append(
        {"params": list(net.proj_s.parameters()), "lr": tr.SPREAD_LR if stage == 2 else base_lr}
    )
    for g in groups:
        g["base_lr"] = g["lr"]
        g["weight_decay"] = tr.WEIGHT_DECAY
        g["warmup"] = True
        g["trunk"] = True
    new: dict[tuple[float, float], list] = {}
    for name, p in net.named_parameters():
        if name.startswith(model.NEW_PREFIXES) and p.requires_grad:
            new.setdefault(NEW_RATES.get(name, NEW_DEFAULT), []).append(p)
    for (lr, wd), params in new.items():
        groups.append({
            "params": params, "lr": lr * new_lr_scale, "base_lr": lr * new_lr_scale,
            "weight_decay": wd, "warmup": False, "trunk": False,
        })  # fmt: skip
    return groups


def split_params(*, optimiser: torch.optim.Optimizer, net: torch.nn.Module) -> tuple[list, list]:
    """Return the trunk's parameters in `net.parameters()` order and the new parameters.

    Each group is clipped on its own.
    """
    is_trunk = {id(p) for g in optimiser.param_groups if g["trunk"] for p in g["params"]}
    trunk = [p for p in net.parameters() if id(p) in is_trunk]
    new = [p for g in optimiser.param_groups if not g["trunk"] for p in g["params"]]
    return trunk, new


def dropout_masks(
    *, seed: int, stage: int, epoch: int, batch: int, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return the two channel mask draws of one batch.

    The draws use experiment 733's own generator, never the global torch stream.
    """
    return inputs.dropout_masks(seed=seed, stage=stage, epoch=epoch, batch=batch, n=n)


def calibration_batches(*, samples: object, source: UntSource, cut: int):  # noqa: ANN201
    """Yield the training samples' `UntInputs` in batches, to set the path's scale `r`.

    The batches carry no mask and use lead day `index mod 14`.
    """
    train_idx = np.nonzero(~samples.is_val)[0]  # ty: ignore[unresolved-attribute]
    for start in range(0, len(train_idx), tr.BATCH * 2):
        sel = train_idx[start : start + tr.BATCH * 2]
        lead = (sel % lay.N_LEAD_DAYS).astype(np.int64)
        yield batch_for(source=source, samples=samples, index=sel, lead=lead, cut=cut).inputs


def validation_loss(
    *, net: model.UntangledNet, source: UntSource, samples: object, cut: int
) -> float:
    """Return the unweighted mixture negative log-likelihood on the held-out pairs, with no mask.

    The pairs are (sample, lead day).
    """
    net.eval()
    idx = np.nonzero(samples.is_val)[0]  # ty: ignore[unresolved-attribute]
    total, count = 0.0, 0
    with torch.no_grad():
        for lead_day in tr.VALIDATION_LEAD_DAYS:
            for start in range(0, len(idx), tr.BATCH * 2):
                sel = idx[start : start + tr.BATCH * 2]
                b = batch_for(
                    source=source, samples=samples, index=sel,
                    lead=np.full(len(sel), lead_day, dtype=np.int64), cut=cut,
                )  # fmt: skip
                nll = model.mdl.per_token_nll(
                    out=net(b.inputs).out, target=torch.nan_to_num(b.target)
                )
                ok = torch.isfinite(b.target)
                total += float(nll[ok].sum())
                count += int(ok.sum())
    return total / max(count, 1)


# --------------------------------------------------------------------------- the real-fold loop


def train_stage(
    *, net: model.UntangledNet, world: object, stage: int, samples: object, seed: int, encoders: str,
    max_passes: int = 60, patience: int = tr.PATIENCE, new_lr_scale: float = 1.0,
    max_steps: int | None = None, train_source: UntSource | None = None,
    on_pass: Callable[[int, model.UntangledNet], None] | None = None,
) -> tuple[tr.StageLog, History]:  # fmt: skip
    """Train `net` on stage 2 or 3 in place and leave it at the state of the best validation loss.

    Args:
        net: The network, loaded from M*'s stage 1 state.
        world: Experiment 713's `World` of the fold; its `source` is an `UntSource`.
        stage: 2 or 3.
        samples: The stage's `SampleSet`.
        seed: The seed.
        encoders: "frozen", "trained", or "random", as M* resolves.
        max_passes: The cap on passes.
        patience: Passes without improvement before stopping.
        new_lr_scale: Multiplies the new parameters' rates (0 is the check that the trunk is unchanged).
        max_steps: Stops after this many optimiser steps (the checks run a few).
        train_source: A source to train on while validating on `world.source` (the placebo's check).
        on_pass: Called after each pass with the pass number and the network.

    Returns:
        Experiment 713's stage log and the per-pass `History`.
    """
    assert stage in {2, 3}, stage
    started = time.monotonic()
    log, hist = tr.StageLog(stage=stage), History(stage=stage)
    source = tr.synthetic_source(world=world, samples=samples) if stage == 3 else world.source  # ty: ignore[unresolved-attribute]
    fit_source = source if train_source is None else train_source
    cut = tr.target_end(world=world, stage=stage)
    torch.manual_seed(seed * 7 + stage)
    optimiser = torch.optim.AdamW(
        param_groups(net=net, stage=stage, encoders=encoders, new_lr_scale=new_lr_scale)
    )
    trunk_params, new_params = split_params(optimiser=optimiser, net=net)
    train_idx = np.nonzero(~samples.is_val)[0]  # ty: ignore[unresolved-attribute]
    best, best_state, stale, step, stopped = float("inf"), None, 0, 0, False
    for epoch in range(max_passes):
        net.train()
        rng = np.random.default_rng([seed, stage, epoch])
        order = rng.permutation(train_idx)
        lead_all = rng.integers(0, lay.N_LEAD_DAYS, size=len(order))
        total, weight_sum = 0.0, 0.0
        for batch_index, start in enumerate(range(0, len(order), tr.BATCH)):
            sel = order[start : start + tr.BATCH]
            native = samples.c_idx[sel] - lay.NATIVE_OFFSET_STEPS  # ty: ignore[unresolved-attribute]
            drop = block_dropout_masks(origin_idx=native, seed=(seed * 1009 + stage) * 1009 + epoch)
            masks = dropout_masks(
                seed=seed, stage=stage, epoch=epoch, batch=batch_index, n=len(sel)
            )
            b = batch_for(
                source=fit_source, samples=samples, index=sel, lead=lead_all[start : start + tr.BATCH],
                cut=cut, dropout=drop, masks=masks,
            )  # fmt: skip
            w = tr.token_weights(target_idx=b.target_idx, world=world, stage=stage)
            ok = torch.isfinite(b.target)
            w = w * ok
            nll = model.mdl.per_token_nll(out=net(b.inputs).out, target=torch.nan_to_num(b.target))
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
            total += float((w * nll).sum().detach().cpu())
            weight_sum += float(w.sum().detach().cpu())
            if max_steps is not None and step >= max_steps:
                stopped = True
                break
        if stopped:
            break
        hist.train.append(total / max(weight_sum, 1e-9))
        val = validation_loss(net=net, source=source, samples=samples, cut=cut)
        hist.val.append(val)
        log.val_history.append(val)
        log.passes = epoch + 1
        if on_pass is not None:
            on_pass(epoch + 1, net)
        if val < best - 1e-5:
            best, stale, hist.stored_pass = val, 0, epoch + 1
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    else:
        log.cap_hit = hist.cap_hit = True
    hist.still_falling = hist.cap_hit and len(hist.val) > 1 and hist.val[-1] < hist.val[-2]
    log.seconds = hist.seconds = time.monotonic() - started
    if stopped:
        net.eval()
        return log, hist
    assert best_state is not None
    net.load_state_dict(best_state)
    net.eval()
    log.best_val = best
    return log, hist


# --------------------------------------------------------------------------- the synthetic loop


def synthetic_arrays_to_inputs(*, batch: sy.SynthBatch, device: str) -> model.UntInputs:
    """Return the `UntInputs` of a synthetic batch on `device`."""
    a = batch.arrays
    return model.UntInputs(
        **{k: torch.from_numpy(np.ascontiguousarray(v)).to(device) for k, v in a.items()}
    )


def synthetic_loss(
    *, net: model.UntangledNet, batch: sy.SynthBatch, device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the total loss, the mixture negative log-likelihood, and the sign head's cross-entropy of a batch.

    Each part is a mean over the tokens whose target exists, and the total adds them with the weights
    of `core.SYNTHETIC_LOSS_WEIGHTS`.
    """
    inp = synthetic_arrays_to_inputs(batch=batch, device=device)
    target = torch.from_numpy(batch.target_z).to(device)
    truth = torch.from_numpy(batch.sign_truth).to(device)
    out = net(inp)
    ok = torch.isfinite(target)
    nll = model.mdl.per_token_nll(out=out.out, target=torch.nan_to_num(target))
    nll_mean = (nll * ok).sum() / ok.sum().clamp_min(1)
    assert out.sign_logit is not None, "the synthetic loop needs the sign head"
    bce = model.sign_loss(logit_=out.sign_logit, truth=truth, weight=ok.to(out.sign_logit.dtype))
    total = (
        core.SYNTHETIC_LOSS_WEIGHTS["mixture_nll"] * nll_mean
        + core.SYNTHETIC_LOSS_WEIGHTS["sign_bce"] * bce
    )
    return total, nll_mean, bce


def synthetic_validation(
    *, net: model.UntangledNet, segments: Sequence[sy.SegmentArrays], index: sy.SampleIndex,
    d8_index: int, target_std: float, mode: str, device: str, batch_size: int = 256,
) -> tuple[float, float, float]:  # fmt: skip
    """Return the validation loss, its mixture part, and its sign part.

    The samples are those at the validation lead days.
    """
    net.eval()
    rows = np.flatnonzero(np.isin(index.lead_index, np.asarray(VALIDATION_LEADS_SYNTHETIC)))
    tot = nll = bce = 0.0
    n = 0
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            sel = rows[start : start + batch_size]
            b = sy.gather_batch(
                segments=segments, index=index, rows=sel, d8_index=d8_index, target_std=target_std, mode=mode
            )  # fmt: skip
            t, a, c = synthetic_loss(net=net, batch=b, device=device)
            tot += float(t.cpu()) * len(sel)
            nll += float(a.cpu()) * len(sel)
            bce += float(c.cpu()) * len(sel)
            n += len(sel)
    n = max(n, 1)
    return tot / n, nll / n, bce / n


def train_synthetic(
    *,
    net: model.UntangledNet,
    train_segments: Sequence[sy.SegmentArrays],
    train_index: sy.SampleIndex,
    val_segments: Sequence[sy.SegmentArrays],
    val_index: sy.SampleIndex,
    seed: int,
    mode: str,
    d8_index: int,
    target_std: float,
    device: str,
    max_epochs: int = core.SYNTHETIC_MAX_EPOCHS,
    patience: int = core.SYNTHETIC_PATIENCE,
    batch_size: int = core.SYNTHETIC_BATCH,
    max_steps: int | None = None,
) -> History:  # fmt: skip
    """Train the synthetic forecaster in place (one stage).

    The network is left at the state of the best validation loss.

    The loop trains the whole network from the random start of `net`, with the channel masks of the
    real folds applied to the channels only, and stops on the validation bank. No test row is read.

    Args:
        net: The network (random start).
        train_segments: The training banks' segments of seeds 3 and 4.
        train_index: Their samples.
        val_segments: The validation bank's segments.
        val_index: Its samples.
        seed: The seed.
        mode: The channel mode of the arm (`untangled`, `absent`, or `true`).
        d8_index: The Primary index of D8.
        target_std: D8's target standard deviation.
        device: "cuda" or "cpu".
        max_epochs: The cap on passes.
        patience: Passes without improvement before stopping.
        batch_size: The batch size.
        max_steps: Stops after this many steps (the checks run a few).

    Returns:
        The per-pass history, with the stored pass marked.
    """
    started = time.monotonic()
    hist = History(stage=0)
    torch.manual_seed(
        seed * 7 + core.GENERATOR_SEED
    )  # the trunk's dropout draws from the global stream
    opt = torch.optim.AdamW(
        net.parameters(), lr=core.SYNTHETIC_LR, weight_decay=core.SYNTHETIC_WEIGHT_DECAY
    )
    best, best_state, stale, step = float("inf"), None, 0, 0
    for epoch in range(max_epochs):
        net.train()
        rng = np.random.default_rng([733, seed, epoch])
        order = rng.permutation(len(train_index))
        total, count = 0.0, 0
        for batch_no, start in enumerate(range(0, len(order), batch_size)):
            sel = order[start : start + batch_size]
            masks = dropout_masks(seed=seed, stage=0, epoch=epoch, batch=batch_no, n=len(sel))
            b = sy.gather_batch(
                segments=train_segments, index=train_index, rows=sel, d8_index=d8_index,
                target_std=target_std, mode=mode, mask_history=masks,
            )  # fmt: skip
            loss, _, _ = synthetic_loss(net=net, batch=b, device=device)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), core.SYNTHETIC_CLIP)
            opt.step()
            total += float(loss.detach().cpu()) * len(sel)
            count += len(sel)
            step += 1
            if max_steps is not None and step >= max_steps:
                break
        hist.train.append(total / max(count, 1))
        val, v_nll, v_bce = synthetic_validation(
            net=net, segments=val_segments, index=val_index, d8_index=d8_index,
            target_std=target_std, mode=mode, device=device,
        )  # fmt: skip
        hist.val.append(val)
        hist.val_nll.append(v_nll)
        hist.val_bce.append(v_bce)
        if val < best - 1e-5:
            best, stale, hist.stored_pass = val, 0, epoch + 1
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
        if max_steps is not None and step >= max_steps:
            break
    else:
        hist.cap_hit = True
    hist.still_falling = hist.cap_hit and len(hist.val) > 1 and hist.val[-1] < hist.val[-2]
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    hist.seconds = time.monotonic() - started
    return hist
