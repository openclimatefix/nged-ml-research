"""The extended training loop and the forecast functions of experiment 710.

**`train_stage` is a copy of `e00707_net.train_network` that asks a weight provider for the weights
at the start of each epoch.** With flat weights, no sampler, a flat learning rate, no averaging, no
step cap, and selection from the first epoch, it makes the same calls to the random number
generator in the same order as the loop of experiment 707, so the two give the same per-step
losses (known-answer test 1).

The extensions, each off by default:

- **Epoch-dependent weights.** `weight_provider(epoch)` returns the epoch's weight tensor, or
  `None` for all ones.
- **A sampler.** Each batch is drawn with replacement with probability proportional to the weight,
  by inverse-CDF sampling (`torch.searchsorted` on the cumulative weights in float64), because
  `torch.multinomial` accepts at most 2^24 categories. The loss is then unweighted.
- **A learning-rate schedule.** A rate per epoch, or a cosine with warm restarts per step
  ([Loshchilov and Hutter (2017)](https://arxiv.org/abs/1608.03983)).
- **Weight averaging.** An exponential moving average of the weights after every step. The averaged
  network is validated, selected, and returned.
- **A step cap, a fixed number of epochs, a first epoch that counts for selection, and `last`
  selection** (the recent-only final epoch).
- **A per-epoch log** of the loss, the steps, the effective rows, and the weighted share of rows
  with a shifted lag.
"""

import copy
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Final

import e00707_common as common  # ty: ignore[unresolved-import]
import e00707_models as models  # ty: ignore[unresolved-import]
import e00707_net as net707  # ty: ignore[unresolved-import]
import e00707_prep as prep  # ty: ignore[unresolved-import]
import e00710_arms as arms  # ty: ignore[unresolved-import]
import e00710_folds as folds  # ty: ignore[unresolved-import]
import e00710_weights as weights  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

BATCH: Final[int] = net707.BATCH
GRAD_CLIP: Final[float] = net707.GRAD_CLIP
MIN_IMPROVEMENT: Final[float] = 1e-5
COSINE_FLOOR_RATIO: Final[float] = 0.1
WeightProvider = Callable[[int], torch.Tensor | None]
"""A function from the epoch index (from 0) to the epoch's weight tensor, or `None` for ones."""


@dataclass
class StageData:
    """The tensors of one stage, all on one device.

    Attributes:
        x: The input matrix.
        z: The target standardised per Primary.
        primary: Each row's Primary index.
        train_idx: The indices of the rows that the stage trains on.
        val_idx: The indices of the selection rows.
        n_primaries: The number of Primaries.
        shifted: Whether each row carries a shifted lag, for the log; `None` in the `n` state.
    """

    x: torch.Tensor
    z: torch.Tensor
    primary: torch.Tensor
    train_idx: torch.Tensor
    val_idx: torch.Tensor
    n_primaries: int
    shifted: torch.Tensor | None = None


@dataclass(frozen=True)
class StageConfig:
    """How a stage trains: the rules that are not data.

    Attributes:
        max_epochs: The most epochs.
        patience: Epochs without improvement that stop the stage.
        select_from: The first epoch (0-based) that counts for selection and patience.
        select: `best` or `last`.
        fixed_epochs: Whether to always run `max_epochs` epochs.
        sampler: Whether batches are drawn in proportion to the weight.
        ema_decay: The decay of the moving average of the weights, or `None`.
        step_cap: The most optimiser steps, or `None`.
        lr_rule: The learning-rate schedule.
    """

    max_epochs: int = arms.MAX_EPOCHS
    patience: int = arms.PATIENCE
    select_from: int = 0
    select: str = "best"
    fixed_epochs: bool = False
    sampler: bool = False
    ema_decay: float | None = None
    step_cap: int | None = None
    lr_rule: arms.LrRule = field(default_factory=arms.LrRule)


@dataclass
class StageResult:
    """What a stage reports.

    Attributes:
        epochs: The epochs run.
        best_val: The selection loss of the selected state.
        best_epoch: The 0-based epoch of the selected state.
        steps: The optimiser steps taken.
        initial_val: The selection loss before the first step.
        stopped_by: `patience`, `epoch_limit`, `step_cap`, or `fixed`.
        seconds: Wall-clock seconds.
        epoch_log: One record per epoch.
        step_losses: The loss of every step, only if the loop was asked to record them.
    """

    epochs: int
    best_val: float
    best_epoch: int
    steps: int
    initial_val: float
    stopped_by: str
    seconds: float
    epoch_log: list[dict] = field(default_factory=list)
    step_losses: list[float] = field(default_factory=list)

    @property
    def hit_limit(self) -> bool:
        """Return whether the stage stopped at its epoch limit or step cap."""
        return self.stopped_by in ("epoch_limit", "step_cap")


def learning_rate(
    *, rule: arms.LrRule, base: float, epoch: int, step_in_epoch: int, steps_per_epoch: int
) -> float | None:
    """Return the learning rate of one step, or `None` to leave the optimiser's rate alone.

    Args:
        rule: The schedule.
        base: The stage's base rate.
        epoch: The 0-based epoch.
        step_in_epoch: The 0-based step inside the epoch.
        steps_per_epoch: The steps in one epoch.

    Returns:
        The rate; `None` for a flat schedule.
    """
    if rule.kind == "flat":
        return None
    if rule.kind == "by_epoch":
        rate = base
        for first_epoch, value in rule.rates:
            if epoch >= first_epoch:
                rate = value
        return rate
    cycle, inside = divmod(epoch, rule.cycle_epochs)
    start = rule.start_rates[min(cycle, len(rule.start_rates) - 1)]
    progress = (inside * steps_per_epoch + step_in_epoch) / (rule.cycle_epochs * steps_per_epoch)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return start * (COSINE_FLOOR_RATIO + (1.0 - COSINE_FLOOR_RATIO) * cosine)


def _validation_loss(*, model: torch.nn.Module, data: StageData, components: int) -> float:
    """Return the unweighted mean negative log-likelihood on the selection rows."""
    model.eval()
    with torch.no_grad():
        return float(
            torch.cat(
                [
                    models.mixture_nll(
                        out=model(data.x[b], data.primary[b]), z=data.z[b], components=components
                    )
                    for b in data.val_idx.split(BATCH * 4)
                ]
            ).mean()
        )


def _weight_log(*, w: torch.Tensor, data: StageData) -> dict:
    """Return the effective rows and the weighted shares of one epoch's weights."""
    wt = w[data.train_idx]
    entry: dict = {
        "stage_effective_share": float(weights.effective_rows(w=wt) / len(wt)),
    }
    primary = data.primary[data.train_idx]
    total = torch.bincount(primary, weights=wt.double(), minlength=data.n_primaries)
    entry["primary_weight_share"] = (total / total.sum()).tolist()
    if data.shifted is not None:
        flag = data.shifted[data.train_idx].double()
        entry["weighted_shifted_share"] = float((wt.double() * flag).sum() / wt.double().sum())
        shifted_total = torch.bincount(
            primary, weights=wt.double() * flag, minlength=data.n_primaries
        )
        entry["weighted_shifted_share_per_primary"] = (shifted_total / total).tolist()
    return entry


def train_stage(  # noqa: PLR0912, PLR0915
    *,
    model: torch.nn.Module,
    setting: net707.Setting,
    data: StageData,
    config: StageConfig,
    weight_provider: WeightProvider,
    seed: int,
    record_losses: bool = False,
) -> StageResult:
    """Train in place with the normalised weighted negative log-likelihood; return the result.

    The selected network is loaded back into `model`. Every batch has `BATCH` rows except the last
    of an epoch, and the loss is `(w * nll).sum() / w.sum()` (`weights.weighted_mean_nll`).

    Args:
        model: The network to train, changed in place.
        setting: The hyperparameters; `setting.lr` is the base learning rate.
        data: The stage's tensors and row indices.
        config: The stopping, selection, sampler, averaging, step cap, and rate rules.
        weight_provider: The weights of each epoch.
        seed: The seed for the torch random number generator.
        record_losses: Whether to keep every step's loss (a synchronisation per step, for tests).

    Returns:
        The epochs, the selected loss, the steps, and the log.

    Raises:
        ValueError: If no epoch counts for selection.
    """
    started = time.monotonic()
    torch.manual_seed(seed)
    device = data.x.device
    n_train = len(data.train_idx)
    steps_per_epoch = math.ceil(n_train / BATCH)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=setting.lr, weight_decay=setting.weight_decay
    )
    average = copy.deepcopy(model) if config.ema_decay is not None else None
    judged = model if average is None else average
    initial = _validation_loss(model=judged, data=data, components=setting.components)
    ones = torch.ones(len(data.x), device=device, dtype=torch.float32)
    best, best_state, best_epoch, stale, epochs, steps = float("inf"), None, -1, 0, 0, 0
    stopped_by = "epoch_limit"
    log: list[dict] = []
    step_losses: list[float] = []
    capped = False
    for epoch in range(config.max_epochs):
        w_full = weight_provider(epoch)
        w_full = ones if w_full is None else w_full
        cumulative = None
        order = None
        if config.sampler:
            cumulative = torch.cumsum(w_full[data.train_idx].double(), dim=0)
            cumulative = cumulative / cumulative[-1]
        else:
            order = data.train_idx[torch.randperm(n_train, device=device)]
        model.train()
        effective = torch.zeros((), device=device, dtype=torch.float64)
        batches = 0
        epoch_steps = 0
        for step in range(steps_per_epoch):
            if cumulative is not None:
                b = sample_batch(cumulative=cumulative, train_idx=data.train_idx, size=BATCH)
                wb = ones[b]
            else:
                assert order is not None
                b = order[step * BATCH : (step + 1) * BATCH]
                wb = w_full[b]
            rate = learning_rate(
                rule=config.lr_rule,
                base=setting.lr,
                epoch=epoch,
                step_in_epoch=step,
                steps_per_epoch=steps_per_epoch,
            )
            if rate is not None:
                for group in optimiser.param_groups:
                    group["lr"] = rate
            nll = models.mixture_nll(
                out=model(data.x[b], data.primary[b]), z=data.z[b], components=setting.components
            )
            loss = weights.weighted_mean_nll(nll=nll, w=wb)
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimiser.step()
            if average is not None:
                _update_average(average=average, model=model, decay=config.ema_decay or 0.0)
            if record_losses:
                step_losses.append(float(loss))
            effective += weights.effective_rows(w=wb)
            batches += 1
            steps += 1
            epoch_steps += 1
            if config.step_cap is not None and steps >= config.step_cap:
                capped = True
                break
        val = _validation_loss(model=judged, data=data, components=setting.components)
        epochs = epoch + 1
        entry = {
            "epoch": epoch,
            "val_nll": val,
            "steps": epoch_steps,
            "effective_rows_per_batch": float(effective / max(batches, 1)),
        } | _weight_log(w=w_full, data=data)
        log.append(entry)
        counts = epoch >= config.select_from
        if counts and (config.select == "last" or val < best - MIN_IMPROVEMENT):
            best, stale, best_epoch = val, 0, epoch
            best_state = {k: v.detach().clone() for k, v in judged.state_dict().items()}
        elif counts:
            stale += 1
            if not config.fixed_epochs and stale >= config.patience:
                stopped_by = "patience"
                break
        if capped:
            stopped_by = "step_cap"
            break
    else:
        stopped_by = "fixed" if config.fixed_epochs else "epoch_limit"
    if best_state is None:
        msg = "training produced no selection score: no epoch counted for selection"
        raise ValueError(msg)
    model.load_state_dict(best_state)
    model.eval()
    return StageResult(
        epochs=epochs,
        best_val=best,
        best_epoch=best_epoch,
        steps=steps,
        initial_val=initial,
        stopped_by=stopped_by,
        seconds=time.monotonic() - started,
        epoch_log=log,
        step_losses=step_losses,
    )


def sample_batch(*, cumulative: torch.Tensor, train_idx: torch.Tensor, size: int) -> torch.Tensor:
    """Draw `size` rows with replacement, in proportion to the weights, by inverse-CDF sampling.

    Args:
        cumulative: The normalised cumulative weights of the training rows, float64, ending at 1.
        train_idx: The row index of each training row.
        size: The batch size.

    Returns:
        The row indices of the batch.
    """
    u = torch.rand(size, device=cumulative.device, dtype=torch.float64)
    pick = torch.searchsorted(cumulative, u).clamp(max=len(train_idx) - 1)
    return train_idx[pick]


def _update_average(*, average: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    """Move the moving average one step towards the model: `a = decay * a + (1 - decay) * m`."""
    with torch.no_grad():
        for a, m in zip(average.parameters(), model.parameters(), strict=True):
            a.mul_(decay).add_(m.detach(), alpha=1.0 - decay)


def new_model(*, n_primaries: int) -> torch.nn.Module:
    """Return an untrained pooled plain network of setting `g09`.

    Args:
        n_primaries: The number of Primaries.

    Returns:
        The network, on the CPU.
    """
    n_inputs = prep.feature_names()[2].n_inputs
    return net707.new_model(n_inputs=n_inputs, n_primaries=n_primaries, setting=net707.G09)


def forecast_month(
    *,
    fitted: net707.Fitted,
    scale: net707.Scale,
    month: str,
    device: str,
    power: dict[str, np.ndarray] | None = None,
) -> pl.DataFrame:
    """Return the test month's raw quantiles (fraction of capacity), unwidened.

    This is `e00707_net.forecast_month` with the lags built from `power`, a dictionary of each
    Primary's power on the grid by site label (`None`: the observed power).

    Args:
        fitted: The pooled network.
        scale: Its standardiser and target statistics.
        month: The test month, such as `2026-01`.
        device: The torch device.
        power: The power that feeds the lags: a bank pass, the corrected power, or `None`.

    Returns:
        One row per series, origin, and valid time with the 11 `common.LEVEL_COLUMNS`.
    """
    pieces = []
    for origin in prep.test_origins(month=month):
        for row, keys, x in folds.run_inputs(
            origin=origin, standardiser=scale.standardiser, power=power
        ):
            p = int(row["primary"])
            q = net707.predict_z_quantiles(fitted=fitted, x=x, primary=p, device=device)
            q = np.sort(q * scale.target_std[p] + scale.target_mean[p], axis=1).astype(np.float32)
            pieces.append(
                keys.drop("lead_hours").hstack(
                    pl.DataFrame(q, schema=list(common.LEVEL_COLUMNS), orient="row")
                )
            )
    return pl.concat(pieces)


def forecast_tuning(
    *, fitted: net707.Fitted, scale: net707.Scale, month: str, device: str
) -> pl.DataFrame:
    """Return a tuning fold's raw quantiles and the target, one row per cached ENS row.

    Args:
        fitted: The pooled network.
        scale: Its standardiser and target statistics.
        month: The tuning month, such as `2025-01`.
        device: The torch device.

    Returns:
        The keys, the 11 raw quantiles, `lead_hours`, and `y` (the target as a fraction of
        capacity).
    """
    pieces = []
    for row, keys, x, y in folds.tuning_rows(standardiser=scale.standardiser, month=month):
        p = int(row["primary"])
        q = net707.predict_z_quantiles(fitted=fitted, x=x, primary=p, device=device)
        q = np.sort(q * scale.target_std[p] + scale.target_mean[p], axis=1).astype(np.float32)
        pieces.append(
            keys.hstack(
                pl.DataFrame(q, schema=list(common.LEVEL_COLUMNS), orient="row")
            ).with_columns(y=pl.Series(y.astype(np.float32)))
        )
    return pl.concat(pieces)


def stage_log_json(*, result: StageResult, extra: dict) -> str:
    """Return a stage's result and extra fields as JSON text for a log file.

    Args:
        result: The stage's result.
        extra: Fields to add, such as the rows and the stage name.

    Returns:
        The JSON text.
    """
    payload = {
        "epochs": result.epochs,
        "best_val": result.best_val,
        "best_epoch": result.best_epoch,
        "steps": result.steps,
        "initial_val": result.initial_val,
        "stopped_by": result.stopped_by,
        "hit_limit": result.hit_limit,
        "seconds": round(result.seconds, 1),
        "epoch_log": result.epoch_log,
    }
    return json.dumps(payload | extra, indent=1)
