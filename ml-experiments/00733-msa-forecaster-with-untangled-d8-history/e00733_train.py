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
  1e-3, batch 256, gradient clip 1.0, at most 40 passes with a patience of 20 on the validation loss,
  which is the mixture negative log-likelihood plus the sign head's cross-entropy, each with
  weight 1.0
  (the post hoc arms Y0b, YUb, and YOb store the pass of the lowest validation sign cross-entropy
  alone and count the patience on it, and train on the same sum).

Both loops record, for every pass, the training loss and the validation loss (`History`), and the
pass whose state was stored, so that the loss curves are drawn from stored data and not from a log.
Every device-dependent line is a function of the tensors' own device, so the CPU path and the CUDA
path run the same code, and `to_numpy` is the only place a tensor becomes an array (`detach`, `cpu`,
then `numpy`).
"""

import math
import time
from collections import deque
from collections.abc import Callable, Generator, Iterable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Final

import e00733_core as core
import e00733_inputs as inputs
import e00733_leakage as leak
import e00733_model as model
import e00733_paths as paths
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
PREFETCH_WORKERS: Final = 4
"""Threads that cut synthetic batches ahead of the training step (the slot has 6 cores, and the
main thread drives the GPU)."""
PREFETCH_DEPTH: Final = 8
"""Batches cut ahead of the training step, at about 35 MB each."""


class NonFiniteLossError(RuntimeError):
    """A validation loss was NaN or infinite, so the fit has no state worth storing (a failed fit)."""


def assert_finite_loss(*, value: float, stage: str, epoch: int) -> None:
    """Raise `NonFiniteLossError` when a validation loss is not finite.

    Args:
        value: The validation loss.
        stage: The stage's name, for the message.
        epoch: The pass number (from 0).

    Raises:
        NonFiniteLossError: When `value` is NaN or infinite.
    """
    if not math.isfinite(value):
        msg = f"the validation loss of {stage} is {value} at pass {epoch + 1}: the fit has diverged"
        raise NonFiniteLossError(msg)


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
    run_avail_us: torch.Tensor | None = None
    other_valid_us: torch.Tensor | None = None


class UntSource:
    """A 713 `Source` plus the untangled channels of one arm (the counterpart of 716's `PvSource`)."""

    def __init__(
        self, *, source: object, tensors: inputs.ChannelTensors, root: Path | None = None
    ) -> None:
        """Hold the wrapped source and the channel tensors; both sit on one device.

        Args:
            source: The wrapped 713 source.
            tensors: The channel tensors.
            root: This experiment's data folder, which receives the example row of the first
                checked batch (`None`: the row is printed and not written).
        """
        self.source, self.tensors, self.root = source, tensors, root
        self.device = source.device  # ty: ignore[unresolved-attribute]

    def with_power_in(self, *, power_in: torch.Tensor) -> UntSource:
        """Return a source whose history power is `power_in` (stage 3's synthetic shifts); the channels are unchanged."""
        return UntSource(
            source=self.source.with_power_in(power_in=power_in),  # ty: ignore[unresolved-attribute]
            tensors=self.tensors,
            root=self.root,
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
            power_shifted=shifted_tokens(source=self, primary=primary, hist_idx=out.hist_idx),
        )  # fmt: skip
        batch_inputs = model.UntInputs(
            **out.inputs.__dict__, hist_u=hist.channels, hist_u_ok=hist.ok
        )
        run_init, run_avail, other_valid = run_times_of(source=self, primary=primary, c_idx=c_idx)
        batch = UntAssembled(
            inputs=batch_inputs, target=out.target, target_idx=out.target_idx,
            hist_idx=out.hist_idx, c_idx=c_idx, primary=primary, run_init_us=run_init,
            run_avail_us=run_avail, other_valid_us=other_valid,
        )  # fmt: skip
        if check:
            check_batch(batch=batch, root=self.root)
        return batch


def shifted_tokens(
    *, source: UntSource, primary: torch.Tensor, hist_idx: torch.Tensor
) -> torch.Tensor | None:
    """Return which history tokens carry stage 3's shifted power (None when no power is shifted).

    The history power of a stage 3 source differs from the clean power where the synthetic shift
    moved it. The stored channel output at such a token belongs to the unshifted power, so the
    channel is absent there.

    Args:
        source: The batch's source.
        primary: `(B,)` long.
        hist_idx: `(B, 14, 48)` long, the grid index of each history token.

    Returns:
        A `(B, 14, 48)` bool tensor, or None for a source that reads the clean power.
    """
    wrapped = source.source
    power_in, power_z = wrapped.power_in, wrapped.power_z  # ty: ignore[unresolved-attribute]
    if power_in is power_z:
        return None
    p3 = primary[:, None, None]
    a, b = power_in[p3, hist_idx], power_z[p3, hist_idx]
    return ~((a == b) | (torch.isnan(a) & torch.isnan(b)))


def run_times_of(
    *, source: UntSource, primary: torch.Tensor, c_idx: torch.Tensor
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Return the run times of each sample's origin day as the run files record them.

    The times are the initialisation time and the availability time of the run, and the valid time of
    the other target-row columns. Experiment 713's own `run_us` is the origin minus 9 hours by
    construction, so it is not read. A source with an ensemble table that holds no file times
    raises.

    Args:
        source: The source whose ensemble table holds the run times.
        primary: `(B,)` the Primary of each sample.
        c_idx: `(B,)` the grid index of each contract origin.

    Returns:
        The three times as `(B,)` tensors, or three Nones when the source has no run times.

    Raises:
        CorrectnessFailureError: When the ensemble table has run times but no file times.
    """
    ens = source.ens
    if getattr(ens, "run_us", None) is None:
        return None, None, None
    names = ("file_run_init_us", "file_run_avail_us", "file_other_valid_us")
    tables = [getattr(ens, n, None) for n in names]
    if any(t is None for t in tables):
        msg = "the ensemble table has no run times read from the run files, so nothing was checked"
        raise leak.CorrectnessFailureError(msg)
    day = to_numpy(tensor=c_idx // lay.STEPS_PER_DAY - ens.first_day)  # ty: ignore[unresolved-attribute]
    prim = to_numpy(tensor=primary)
    return tuple(torch.from_numpy(np.asarray(t)[prim, day]) for t in tables)  # ty: ignore[invalid-return-type]


_EXAMPLE_PRINTED: list[str] = []
"""The example row of this process's first checked batch (empty until one is checked)."""


def _print_example(
    *, batch: UntAssembled, c: np.ndarray, filled: np.ndarray, root: Path | None
) -> None:
    """Print the plan's example row for the first target token of the batch's first sample.

    The row is read from the batch's own run times and written to
    `PRIVATE/origin_time_example_real.txt` under `root` (not written when `root` is None).
    """
    i = int(np.flatnonzero(filled.any(axis=1))[0])
    target = to_numpy(tensor=batch.target_idx)[i][filled[i]][0]
    row = leak.example_row(
        c_idx=int(c[i]), target_idx=int(target), run_init_us=int(to_numpy(tensor=batch.run_init_us)[i]),
        other_latest_valid_us=int(to_numpy(tensor=batch.other_valid_us)[i]),
        run_available_us=int(to_numpy(tensor=batch.run_avail_us)[i]),
    )  # fmt: skip
    _EXAMPLE_PRINTED.append(row)
    print(row, flush=True)
    if root is not None:
        (paths.private_dir(root=root) / "origin_time_example_real.txt").write_text(row + "\n")


def check_batch(*, batch: UntAssembled, root: Path | None = None) -> None:
    """Run the per-batch assertions.

    The assertions cover the origin-time weather of every target token, the history order, and the
    channel ranges.

    Args:
        batch: The assembled batch.
        root: This experiment's data folder, which receives the example row (None: none written).

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
        checked = leak.assert_origin_time_weather(
            c_idx=c, run_init_us=to_numpy(tensor=batch.run_init_us),
            target_idx=to_numpy(tensor=batch.target_idx), filled=filled,
            other_latest_valid_us=to_numpy(tensor=batch.other_valid_us),
            run_available_us=to_numpy(tensor=batch.run_avail_us),
        )  # fmt: skip
        if checked and not _EXAMPLE_PRINTED:
            _print_example(batch=batch, c=c, filled=filled, root=root)
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


def train_stage(  # noqa: PLR0912
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
        assert_finite_loss(value=val, stage=f"stage {stage}", epoch=epoch)
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
    if best_state is None:
        msg = f"stage {stage} stored no state: no pass improved on an infinite loss"
        raise NonFiniteLossError(msg)
    net.load_state_dict(best_state)
    net.eval()
    log.best_val = best
    return log, hist


# --------------------------------------------------------------------------- the synthetic loop


def prefetched[T, R](
    *, items: Iterable[T], make: Callable[[T], R], workers: int | None = None,
    depth: int = PREFETCH_DEPTH,
) -> Generator[R]:  # fmt: skip
    """Yield `make(item)` for each item in order, computing up to `depth` of them ahead on threads.

    The results come in the order of `items`, and `make` must not share mutable state between items,
    so the sequence equals `(make(item) for item in items)`. An exception raised by `make` is raised
    where the result would have been yielded. Closing the generator drops the results not yet taken.

    Args:
        items: The inputs, consumed lazily.
        make: Builds one result from one item (a read-only gather from shared arrays).
        workers: The number of threads (1 or less computes in the calling thread); `None` is
            `PREFETCH_WORKERS`.
        depth: The most results cut ahead.

    Yields:
        The results, in order.
    """
    workers = PREFETCH_WORKERS if workers is None else workers
    if workers <= 1:
        for item in items:
            yield make(item)
        return
    pending: deque[Future[R]] = deque()
    source = iter(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        try:
            for item in source:
                pending.append(pool.submit(make, item))
                if len(pending) >= depth:
                    yield pending.popleft().result()
            while pending:
                yield pending.popleft().result()
        finally:
            for future in pending:
                future.cancel()


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
    *, net: model.UntangledNet, segments: Sequence[sy.SegmentArrays] | sy.PackedBank,
    index: sy.SampleIndex, d8_index: int, target_std: float, mode: str, device: str,
    batch_size: int = 256,
) -> tuple[float, float, float]:  # fmt: skip
    """Return the validation loss, its mixture part, and its sign part.

    The samples are those at the validation lead days.
    """
    net.eval()
    rows = np.flatnonzero(np.isin(index.lead_index, np.asarray(VALIDATION_LEADS_SYNTHETIC)))
    tot = nll = bce = 0.0
    n = 0
    with torch.no_grad():
        sels = [rows[start : start + batch_size] for start in range(0, len(rows), batch_size)]
        batches = prefetched(
            items=sels,
            make=lambda sel: sy.gather_batch(
                segments=segments, index=index, rows=sel, d8_index=d8_index, target_std=target_std,
                mode=mode,
            ),
        )  # fmt: skip
        for sel, b in zip(sels, batches, strict=True):
            t, a, c = synthetic_loss(net=net, batch=b, device=device)
            tot += float(t.cpu()) * len(sel)
            nll += float(a.cpu()) * len(sel)
            bce += float(c.cpu()) * len(sel)
            n += len(sel)
    n = max(n, 1)
    return tot / n, nll / n, bce / n


def train_synthetic(  # noqa: PLR0912
    *,
    net: model.UntangledNet,
    train_segments: Sequence[sy.SegmentArrays] | sy.PackedBank,
    train_index: sy.SampleIndex,
    val_segments: Sequence[sy.SegmentArrays] | sy.PackedBank,
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
    stop_on: str = core.STOP_ON_TOTAL,
) -> History:  # fmt: skip
    """Train the synthetic forecaster in place (one stage).

    The network is left at the state of the best validation loss, or of the lowest validation sign
    BCE when `stop_on` is `core.STOP_ON_SIGN_BCE`.

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
        stop_on: `core.STOP_ON_TOTAL` stores the pass of the lowest validation loss (the sum of the
            mixture NLL and the sign BCE) and counts the patience on that loss.
            `core.STOP_ON_SIGN_BCE` does both on the sign head's own validation BCE; the network
            still trains on the sum of the two losses.

    Returns:
        The per-pass history, with the stored pass marked.

    Raises:
        ValueError: When `stop_on` is neither of the two criteria.
    """
    if stop_on not in (core.STOP_ON_TOTAL, core.STOP_ON_SIGN_BCE):
        msg = (
            f"stop_on must be {core.STOP_ON_TOTAL!r} or {core.STOP_ON_SIGN_BCE!r}, not {stop_on!r}"
        )
        raise ValueError(msg)
    started = time.monotonic()
    if not isinstance(train_segments, sy.PackedBank):
        train_segments = sy.pack_segments(segments=train_segments)
    if not isinstance(val_segments, sy.PackedBank):
        val_segments = sy.pack_segments(segments=val_segments)
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
        sels = [order[start : start + batch_size] for start in range(0, len(order), batch_size)]

        def cut(item: tuple[int, np.ndarray], epoch: int = epoch) -> sy.SynthBatch:
            batch_no, sel = item
            masks = dropout_masks(seed=seed, stage=0, epoch=epoch, batch=batch_no, n=len(sel))
            return sy.gather_batch(
                segments=train_segments, index=train_index, rows=sel, d8_index=d8_index,
                target_std=target_std, mode=mode, mask_history=masks,
            )  # fmt: skip

        stream = prefetched(items=list(enumerate(sels)), make=cut)
        try:
            for sel, b in zip(sels, stream, strict=False):
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
        finally:
            stream.close()
        hist.train.append(total / max(count, 1))
        val, v_nll, v_bce = synthetic_validation(
            net=net, segments=val_segments, index=val_index, d8_index=d8_index,
            target_std=target_std, mode=mode, device=device,
        )  # fmt: skip
        assert_finite_loss(value=val, stage="the synthetic fit", epoch=epoch)
        hist.val.append(val)
        hist.val_nll.append(v_nll)
        hist.val_bce.append(v_bce)
        criterion = v_bce if stop_on == core.STOP_ON_SIGN_BCE else val
        if criterion < best - 1e-5:
            best, stale, hist.stored_pass = criterion, 0, epoch + 1
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
        if max_steps is not None and step >= max_steps:
            break
    else:
        hist.cap_hit = True
    curve = hist.val_bce if stop_on == core.STOP_ON_SIGN_BCE else hist.val
    hist.still_falling = hist.cap_hit and len(curve) > 1 and curve[-1] < curve[-2]
    if best_state is None and max_steps is None:
        msg = "the synthetic fit stored no state: no pass improved on an infinite loss"
        raise NonFiniteLossError(msg)
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    hist.seconds = time.monotonic() - started
    return hist
