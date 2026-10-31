# ruff: noqa: E501, PLR0912, PLR0915, PT018
"""Fit and forecast the Brief B arms of experiment 446 on experiment 428's S1 harness.

This module reuses experiment 428's rows, folds, fit settings, prediction code, and row-class flags
unchanged and redirects the modules' output folders to this experiment's private work folder
(`e00439_run.py` and `e00440_run.py` did the same). Nothing in experiment 428's, 404's, 439's, or
441's folders is written.

**Arms** (`ARMS446`). All use S1's inputs (the non-lag columns, the three origin lags with missing
flags and ages, and the single-step surprise at each lag from experiment 404's cross-fitted lagless
tables; seeds 0 to 2 read 404's tables for the same seed).

- `U1n-2`, `U1n-5`: S1 trained on worlds that add genuine level changes to lags and target together
  (`e00446_worlds`), at 2% and 5% of half-hours, with token dropout 0.2 on each lag and on each
  lag's surprise, and the origin of the lags drawn from 8 hours (0 to 7 hours before the usual
  origin). `U1n-2t` and `U1n-5t` use the trimmed duration distribution.
- `U2`: `U1n-2` plus the scale-only path of `e00446_models.ScaleMdn` and a penalty on the mean log
  factor on clean rows. `U2i` adds the 90% and 98% interval score on corrupted rows. `U2c` adds the
  coverage-matching penalty (improper). `U2-5`, `U2i-5`, and `U2c-5` are the 5% versions (run only
  if the 5% `U1n` matches the held-out world's coverage better).
- `U2-clean`: the `U2` network on worlds without genuine level changes (experiment 428 holds no such
  arm, so it is fitted here).
- `U3`: prediction only. 20 passes of `U1n-2` with each lag masked at random with a probability that
  rises with the lag's absolute surprise; the 20 predictive mixtures are pooled with equal weight.

**Corrupted and clean rows in training.** A row is corrupted when, after token dropout, a shown lag
differs from the unshifted clean series at the same grid index; every other row is clean. The
scale path adds its own inputs; the location path is S1's network, unchanged, and receives no
gradient from the scale-only loss terms (see `e00446_models`).

**Prediction.** For each arm, month, and seed one parquet per world, as in experiment 428
(`predictions/<arm>/<month>_s<seed>_w<world>.parquet`), for worlds 0, 4, 5, 6 (experiment 428's)
and 7 to 9 (the held-out worlds with genuine level changes). Worlds 7 to 9 add `actual_clean`,
`shift_lag`, `shift_tgt`, and `days_since` (days from the onset of the level change in force at the
newest lag, NaN if none). Scale arms add `scale_f`, the geometric mean over components of the
factor. Reference arms (`R-inj`, `S1`, `L0`, `R`, `Bmix`) are linked from experiment 428's folder
for worlds 0 and 4 to 6 (`link`); `predict` forecasts worlds 7 to 9 for `R-inj`, `S1`, and `L0`
from 428's models.

A mkdir lock per (arm, month, seed) cell lets two processes share cells; a crashed process leaves
its lock folder behind, and `rmdir` on it releases the cell.

```bash
source <paths>; PYTHONPATH=$PP OMP_NUM_THREADS=1 \
  systemd-run --user --scope -q -p MemoryMax=8G -p OOMPolicy=continue \
  uv run --no-sync python ml-experiments/00446-*/e00446_run.py fit --arms U1n-2 U2 --months 2026-01 --seeds 0
uv run --no-sync python ml-experiments/00446-*/e00446_run.py predict --arms U1n-2 U2 U3 --months 2026-01 --seeds 0
uv run --no-sync python ml-experiments/00446-*/e00446_run.py selftest
```
"""

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import e00397_models as base  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00428_data as data  # ty: ignore[unresolved-import]
import e00428_fit as fit  # ty: ignore[unresolved-import]
import e00428_predict as pr  # ty: ignore[unresolved-import]
import e00428_windows as win  # ty: ignore[unresolved-import]
import e00446_models as models  # ty: ignore[unresolved-import]
import e00446_worlds as worlds  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.eval_protocol.slices import MVA_ONLY_LABELS

WORK_DIR: Final[Path] = worlds.WORK_DIR
MODELS_DIR: Final[Path] = WORK_DIR / "models"
PRED_DIR: Final[Path] = WORK_DIR / "predictions"
LOCK_DIR: Final[Path] = WORK_DIR / "locks"
REFERENCE_ARMS: Final[tuple[str, ...]] = ("R-inj", "S1", "L0", "R", "Bmix")
FORECAST_REFERENCES: Final[tuple[str, ...]] = ("R-inj", "S1", "L0")
"""Reference arms whose networks are forecast here on the held-out worlds 7 to 9."""
GEN_WORLDS: Final[tuple[int, ...]] = (7, 8, 9)
P_LAG_DROP: Final[float] = 0.2
P_SPREAD_DROP: Final[float] = 0.2
ORIGIN_HOURS: Final[int] = 8
HALF_HOURS_PER_HOUR: Final[int] = 2
LOG_FACTOR_WEIGHT: Final[float] = 1.0
INTERVAL_WEIGHT: Final[float] = 0.25
COVERAGE_WEIGHT: Final[float] = 1.0
U3_PASSES: Final[int] = 20
U3_BASE_MASK: Final[float] = 0.2
U3_SURPRISE_MASK: Final[float] = 0.7
"""The mask probability of a lag is 0.2 + 0.7 * min(|surprise| / 5, 1)."""
CORRUPTION_TOLERANCE: Final[float] = 1e-6
EPOCH_SEEDS_FOR_DEFAULT: Final[tuple[str, ...]] = ("U1n-2", "U1n-5", "U2", "U2i", "U2c", "U2-clean")


@dataclass(frozen=True)
class Arm446:
    """How an arm of experiment 446 is built and trained."""

    name: str
    rate: float
    trimmed: bool = False
    scale: bool = False
    interval: bool = False
    coverage: bool = False


ARMS446: Final[dict[str, Arm446]] = {
    "U1n-2": Arm446("U1n-2", 0.02),
    "U1n-5": Arm446("U1n-5", 0.05),
    "U1n-2t": Arm446("U1n-2t", 0.02, trimmed=True),
    "U1n-5t": Arm446("U1n-5t", 0.05, trimmed=True),
    "U2": Arm446("U2", 0.02, scale=True),
    "U2i": Arm446("U2i", 0.02, scale=True, interval=True),
    "U2c": Arm446("U2c", 0.02, scale=True, coverage=True),
    "U2-5": Arm446("U2-5", 0.05, scale=True),
    "U2i-5": Arm446("U2i-5", 0.05, scale=True, interval=True),
    "U2c-5": Arm446("U2c-5", 0.05, scale=True, coverage=True),
    "U2-clean": Arm446("U2-clean", 0.0, scale=True),
}
PREDICT_ONLY: Final[dict[str, str]] = {"U3": "U1n-2"}
"""Arms with no fit of their own, and the fitted arm whose networks they forecast with."""


def install() -> None:
    """Register the arms and redirect experiment 428's modules to this experiment's folders."""
    for name in ARMS446:
        fit.ARMS[name] = fit.Arm(name, "lags", 0, True, surprise=True, history="s1")
    fit.MODELS_DIR = MODELS_DIR
    pr.PRED_DIR = PRED_DIR
    if getattr(fit.make_model, "_installed_446", False):
        return
    original = fit.make_model
    lead_cols = models.lead_columns()

    def make_model(*, arm: fit.Arm, n_inputs: int, n_primaries: int) -> torch.nn.Module:
        spec = ARMS446.get(arm.name)
        if spec is None or not spec.scale:
            return original(arm=arm, n_inputs=n_inputs, n_primaries=n_primaries)
        return models.ScaleMdn(
            n_inputs=n_inputs, n_primaries=n_primaries, lead_cols=lead_cols,
            hidden=fit.SETTING.hidden, layers=fit.SETTING.layers,
            components=fit.SETTING.components, dropout=fit.SETTING.dropout,
        )  # fmt: skip

    make_model._installed_446 = True  # ty: ignore[unresolved-attribute]
    fit.make_model = make_model


def use_models(*, arm_name: str) -> None:
    """Point experiment 428's loader at the folder that holds the arm's saved networks."""
    fit.MODELS_DIR = data.WORK_DIR / "models" if arm_name in REFERENCE_ARMS else MODELS_DIR


def nonnegative_primaries() -> np.ndarray:
    """Return (primaries,) True for the MVA-only Primaries, whose power cannot go below zero."""
    return np.array([s in MVA_ONLY_LABELS for s in data.sites()])


class Inputs446(fit.Inputs):
    """Experiment 428's inputs plus training-time augmentation and the shifted target.

    With `train` set, `raw` draws the origin of the lags 0 to `ORIGIN_HOURS - 1` hours earlier and
    drops each lag with probability `P_LAG_DROP` (its value becomes unobserved, so its missing flag
    is 1 and its surprise 0); `assemble` also zeros each lag's surprise with probability
    `P_SPREAD_DROP`. With `train` unset the class is experiment 428's.
    """

    def __init__(
        self,
        *,
        clean_power: torch.Tensor | None = None,
        target_power: torch.Tensor | None = None,
        train: bool = False,
        **kwargs,  # noqa: ANN003
    ) -> None:
        """Hold the unshifted clean series (primaries, grid) and the per-world target series."""
        super().__init__(**kwargs)
        self.clean_power = clean_power
        self.target_power = target_power
        self.train = train

    def raw(self, *, arm: fit.Arm, rows: torch.Tensor, world: torch.Tensor) -> dict:
        """Return experiment 428's raw inputs, augmented when `train` is set."""
        out = super().raw(arm=arm, rows=rows, world=world)
        if not self.train:
            return out
        n = len(rows)
        hours = torch.randint(0, ORIGIN_HOURS, (n,), device=rows.device)
        origin = out["origin"] - hours * HALF_HOURS_PER_HOUR
        lags = win.origin_lags(
            power=self.power, world=world, primary=out["primary"], origin=origin,
            target=out["target"],
        )  # fmt: skip
        lags[torch.rand(n, 3, device=rows.device) < P_LAG_DROP] = float("nan")
        out["origin"], out["lags"] = origin, lags
        out["spread_drop"] = torch.rand(n, 3, device=rows.device) < P_SPREAD_DROP
        return out

    def assemble(self, *, arm: fit.Arm, raw: dict, masked: bool = False) -> tuple:
        """Return experiment 428's `(x, primary, windows)`, with dropped surprises zeroed."""
        x, primary, windows = super().assemble(arm=arm, raw=raw, masked=masked)
        if "spread_drop" in raw and not masked:
            x[:, -3:] = torch.where(raw["spread_drop"], torch.zeros_like(x[:, -3:]), x[:, -3:])
        return x, primary, windows

    def target_z(self, *, rows: torch.Tensor, world: torch.Tensor) -> torch.Tensor:
        """Return the standardised target of `rows` in `world` (clean plus the world's level change)."""
        assert self.target_power is not None
        assert self.target_stats is not None
        mean, std = self.target_stats
        primary = self.primary[rows]
        y = win.take(
            power=self.target_power, world=world, primary=primary, index=self.target[rows]
        )  # fmt: skip
        return (y - mean[primary]) / std[primary]

    def corrupted(self, *, raw: dict) -> torch.Tensor:
        """Return (rows,) True where a shown lag differs from the unshifted clean series."""
        assert self.clean_power is not None
        index = win.lag_indices(origin=raw["origin"], target=raw["target"])
        zero = torch.zeros_like(raw["primary"])
        clean = win.take(
            power=self.clean_power[None], world=zero, primary=raw["primary"], index=index
        )
        lags = raw["lags"]
        bad = (
            torch.isfinite(lags)
            & torch.isfinite(clean)
            & ((lags - clean).abs() > CORRUPTION_TOLERANCE)
        )
        return bad.any(dim=1)


def train_stacks(*, spec: Arm446, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return `(power, target_power, clean)` for the four training worlds of an arm.

    `power` is experiment 428's worlds 0 to 3 plus each world's level changes; `target_power` is the
    clean series plus the same level changes (the target the injected load-shift events leave alone);
    `clean` is the unshifted clean series.
    """
    stack, _ = data.load_stack(include_eval=False)
    nonneg = nonnegative_primaries()
    clean = stack[0]
    powers, targets = [], []
    for w in range(data.N_TRAIN_WORLDS + 1):
        if spec.rate > 0:
            shift, _ = worlds.load_shift(
                rate=spec.rate, trimmed=spec.trimmed, seed=worlds.TRAIN_SEEDS[w]
            )
        else:
            shift = np.zeros(clean.shape, dtype=np.float32)
        powers.append(worlds.apply_shift(power=stack[w], shift=shift, nonnegative=nonneg))
        targets.append(worlds.apply_shift(power=clean, shift=shift, nonnegative=nonneg))
    return (
        torch.from_numpy(np.stack(powers)).to(device),
        torch.from_numpy(np.stack(targets)).to(device),
        torch.from_numpy(clean).to(device),
    )


def train(
    *, spec: Arm446, arm: fit.Arm, model: torch.nn.Module, inputs: Inputs446, w: torch.Tensor,
    is_val: torch.Tensor, seed: int, device: str,
) -> tuple[int, float]:  # fmt: skip
    """Train with weighted negative log-likelihood plus the arm's extra terms; return epochs, best."""
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    train_idx = torch.nonzero(~is_val).squeeze(1)
    val_idx = torch.nonzero(is_val).squeeze(1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if len(val_idx) > fit.VALIDATION_ROWS:
        pick = torch.randperm(len(val_idx), generator=generator)[: fit.VALIDATION_ROWS]
        val_idx = val_idx[pick.to(device)]
    val_world = val_idx % (data.N_TRAIN_WORLDS + 1)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=fit.SETTING.lr, weight_decay=fit.SETTING.weight_decay
    )
    k = fit.SETTING.components
    best, best_state, stale, epochs = float("inf"), None, 0, 0
    for epoch in range(fit.MAX_EPOCHS):
        model.train()
        inputs.train = True
        order = train_idx[torch.randperm(len(train_idx), device=device)]
        order = order[: int(arm.row_fraction * len(order))]
        for start in range(0, len(order), fit.BATCH):
            b = order[start : start + fit.BATCH]
            world = torch.randint(0, data.N_TRAIN_WORLDS + 1, (len(b),), device=device)
            raw = inputs.raw(arm=arm, rows=b, world=world)
            x, primary, _ = inputs.assemble(arm=arm, raw=raw)
            z = inputs.target_z(rows=b, world=world)
            if spec.scale:
                out, factor = model.forward_parts(x, primary)  # ty: ignore[call-non-callable]
            else:
                out, factor = model(x, primary), None
            nll = base.mixture_nll(out=out, z=z, components=k)
            loss = (w[b] * nll).sum() / w[b].sum()
            if spec.scale:
                assert factor is not None
                bad = inputs.corrupted(raw=raw)
                loss = loss + LOG_FACTOR_WEIGHT * models.log_factor_penalty(
                    factor=factor, rows=~bad
                )
                if spec.interval or spec.coverage:
                    log_w, mean, std = models.scale_only_parts(out=out, factor=factor, components=k)
                    quantiles = models.tail_quantiles_st(log_w=log_w, mean=mean, std=std)
                    if spec.interval:
                        loss = loss + INTERVAL_WEIGHT * models.interval_loss(
                            quantiles=quantiles, y=z, rows=bad
                        )
                    if spec.coverage:
                        loss = loss + COVERAGE_WEIGHT * models.coverage_penalty(
                            quantiles=quantiles, y=z, rows=bad
                        )
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), fit.GRAD_CLIP)
            optimiser.step()
        model.eval()
        inputs.train = False
        with torch.no_grad():
            losses = []
            for chunk, chunk_world in zip(
                val_idx.split(fit.BATCH * 2), val_world.split(fit.BATCH * 2), strict=True
            ):
                x, primary, _ = inputs.batch(arm=arm, rows=chunk, world=chunk_world)
                out = fit.forward(model=model, x=x, primary=primary, windows=None)
                losses.append(
                    base.mixture_nll(
                        out=out, z=inputs.target_z(rows=chunk, world=chunk_world), components=k
                    )
                )
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


class Cell:
    """A mkdir lock on one (arm, month, seed) cell; `held` is False if another process has it."""

    def __init__(self, *, kind: str, arm: str, month: str, seed: int) -> None:
        """Try to take the lock."""
        self.path = LOCK_DIR / f"{kind}__{arm}__{month}__s{seed}"
        LOCK_DIR.mkdir(parents=True, exist_ok=True)
        try:
            self.path.mkdir()
            self.held = True
        except FileExistsError:
            self.held = False

    def release(self) -> None:
        """Release the lock if this process holds it."""
        if self.held:
            self.path.rmdir()
            self.held = False


def fit_one(
    *, arm_name: str, month: str, seed: int, device: str, rows_cache: dict, stack_cache: dict,
) -> str:  # fmt: skip
    """Fit and save one arm, fold, and seed; return a progress line."""
    target = MODELS_DIR / arm_name / month / f"s{seed}"
    if (target / "meta.json").exists():
        return f"{arm_name} {month} s{seed}: exists"
    cell = Cell(kind="fit", arm=arm_name, month=month, seed=seed)
    if not cell.held:
        return f"{arm_name} {month} s{seed}: locked by another process"
    if (target / "meta.json").exists():
        cell.release()
        return f"{arm_name} {month} s{seed}: exists"
    try:
        spec, arm = ARMS446[arm_name], fit.ARMS[arm_name]
        started = time.monotonic()
        key = (spec.rate, spec.trimmed)
        if key not in stack_cache:
            stack_cache.clear()
            stack_cache[key] = train_stacks(spec=spec, device=device)
        power, target_power, clean_power = stack_cache[key]
        if month not in rows_cache:
            rows_cache.clear()
            stack, _ = data.load_stack(include_eval=False)
            rows_cache[month] = data.build_fold(refit=prep.refit_of(month=month), clean=stack[0])
        rows: data.Rows = rows_cache[month]
        inputs = Inputs446(
            power=power, x=rows.x, primary=rows.primary, origin_idx=rows.origin_idx,
            target_idx=rows.target_idx, power_mean=rows.power_mean, power_std=rows.power_std,
            device=device, surprise=fit.surprise_for(arm=arm, month=month, seed=seed, device=device),
            clean_power=clean_power, target_power=target_power,
        )  # fmt: skip
        inputs.target_stats = fit.target_stats(
            mean=rows.target_mean, std=rows.target_std, device=device
        )
        w = torch.from_numpy(rows.weight).to(device)
        is_val = torch.from_numpy(rows.is_val).to(device)
        torch.manual_seed(seed)
        model = fit.make_model(
            arm=arm, n_inputs=rows.x.shape[1], n_primaries=len(rows.target_mean)
        ).to(device)
        epochs, best = train(
            spec=spec, arm=arm, model=model, inputs=inputs, w=w, is_val=is_val, seed=seed,
            device=device,
        )  # fmt: skip
        target.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), target / "model.pt")
        np.savez(
            target / "standardiser.npz",
            mean=rows.standardiser.mean, std=rows.standardiser.std, mask=rows.standardiser.mask,
            target_mean=rows.target_mean, target_std=rows.target_std,
        )  # fmt: skip
        (target / "meta.json").write_text(
            json.dumps(
                {
                    "arm": arm_name, "month": month, "seed": seed, "epochs": epochs,
                    "val_nll": best, "power_mean": rows.power_mean, "power_std": rows.power_std,
                    "n_rows": len(rows.x), "unusable_targets": rows.dropped,
                    "n_inputs": int(rows.x.shape[1]),
                    "parameters": models.count_parameters(model=model),
                    "rate": spec.rate, "trimmed": spec.trimmed,
                }
            )
        )  # fmt: skip
        return (
            f"{arm_name} {month} s{seed}: {epochs} epochs, val nll {best:.4f}, "
            f"{time.monotonic() - started:.0f} s"
        )
    finally:
        cell.release()


def eval_stack() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return `(power, code, shift, onset)` over worlds 0 to 9 (and the held-out shifts, 3 worlds).

    Worlds 0 to 6 are experiment 428's; worlds 7 to 9 are the real clean series plus the held-out
    level changes (seeds 400 to 402). `code` is zero for worlds 7 to 9.
    """
    power, code = data.load_stack(include_eval=True)
    nonneg = nonnegative_primaries()
    shifts, onsets = [], []
    for seed in worlds.HELD_OUT_SEEDS:
        shift, onset = worlds.load_shift(rate=worlds.HELD_OUT_RATE, trimmed=False, seed=seed)
        shifts.append(shift)
        onsets.append(onset)
    gen = np.stack(
        [worlds.apply_shift(power=power[0], shift=s, nonnegative=nonneg) for s in shifts]
    )
    return (
        np.concatenate([power, gen]),
        np.concatenate([code, np.zeros(gen.shape, dtype=np.int8)]),
        np.stack(shifts),
        np.stack(onsets),
    )


def generated_columns(
    *, test: data.TestRows, shift: np.ndarray, onset: np.ndarray
) -> dict[str, np.ndarray]:
    """Return `shift_lag`, `shift_tgt`, and `days_since` for the rows of a held-out world."""
    origin = torch.from_numpy(test.origin_idx)
    target = torch.from_numpy(test.target_idx)
    lags = win.lag_indices(origin=origin, target=target).numpy()
    size = shift.shape[1]
    p = test.primary
    inside = (lags >= 0) & (lags < size)
    safe = np.where(inside, lags, 0)
    moved = (shift[p[:, None], safe] != 0) & inside
    newest = lags[:, 0]
    started = np.where(inside[:, 0], onset[p, safe[:, 0]], -1)
    days = np.where(started >= 0, (newest - started) / 48.0, np.nan)
    return {
        "shift_lag": moved.any(axis=1),
        "shift_tgt": shift[p, np.clip(test.target_idx, 0, size - 1)] != 0,
        "days_since": days,
    }


def forecast(
    *, loaded: pr.Loaded, inputs: fit.Inputs, rows: torch.Tensor, world: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Return experiment 428's forecast of `rows`, plus `scale_f` for a scale-path network."""
    result = pr.forecast_rows(loaded=loaded, inputs=inputs, rows=rows, world=world)
    if isinstance(loaded.model, models.ScaleMdn):
        with torch.no_grad():
            raw = inputs.raw(arm=loaded.arm, rows=rows, world=world)
            x, _, _ = inputs.assemble(arm=loaded.arm, raw=raw)
            result["scale_f"] = torch.exp(torch.log(loaded.model.factors(x)).mean(dim=1))
    return result


def forecast_u3(
    *, loaded: pr.Loaded, inputs: fit.Inputs, rows: torch.Tensor, world: torch.Tensor,
    generator: torch.Generator, edit: Callable[[dict], None] | None = None, masked: bool = False,
) -> torch.Tensor:  # fmt: skip
    """Return the 11 quantiles of U3: `U3_PASSES` masked passes pooled with equal weight.

    `edit` (the encroachment sweep's step) is applied to the raw lags before the masks are drawn,
    so each lag's mask probability follows the surprise after the step.
    """
    pieces = []
    primary = inputs.primary[rows]
    mu = torch.from_numpy(loaded.target_mean).to(rows.device)[primary][:, None]
    sd = torch.from_numpy(loaded.target_std).to(rows.device)[primary][:, None]

    def modify(raw: dict) -> None:
        if edit is not None:
            edit(raw)
        lags = raw["lags"]
        surprise = inputs.lag_surprise(raw=raw, lags=lags).abs()
        p = U3_BASE_MASK + U3_SURPRISE_MASK * (surprise / win.CLIP_SD).clamp(0.0, 1.0)
        draw = torch.rand(lags.shape, generator=generator, device=lags.device)
        lags[draw < p] = float("nan")

    for _ in range(U3_PASSES):
        res = pr.forecast_rows(
            loaded=loaded, inputs=inputs, rows=rows, world=world, modify=modify, masked=masked
        )
        pieces.append(
            (res["log_w"] - float(np.log(U3_PASSES)), res["mean"] * sd + mu, res["std"] * sd)
        )
    log_w = torch.cat([p[0] for p in pieces], dim=1)
    mean = torch.cat([p[1] for p in pieces], dim=1)
    std = torch.cat([p[2] for p in pieces], dim=1)
    return torch.sort(base.mixture_quantiles(log_w=log_w, mean=mean, std=std), dim=1).values


def predict_month(
    *, arm_name: str, month: str, seed: int, device: str, power: torch.Tensor,
    stack: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], flags: pr.Flags,
) -> list[str]:  # fmt: skip
    """Forecast a test month in every world not yet on disk; return progress lines."""
    power_np, _, shift, onset = stack
    source = PREDICT_ONLY.get(arm_name, arm_name)
    reference = arm_name in REFERENCE_ARMS
    worlds_todo = list(GEN_WORLDS) if reference else [*pr.EVAL_WORLDS, *GEN_WORLDS]
    missing = [
        w
        for w in worlds_todo
        if not (PRED_DIR / arm_name / f"{month}_s{seed}_w{w}.parquet").exists()
    ]
    if not missing:
        return [f"{arm_name} {month} s{seed}: exists"]
    cell = Cell(kind="predict", arm=arm_name, month=month, seed=seed)
    if not cell.held:
        return [f"{arm_name} {month} s{seed}: locked by another process"]
    missing = [
        w
        for w in worlds_todo
        if not (PRED_DIR / arm_name / f"{month}_s{seed}_w{w}.parquet").exists()
    ]
    if not missing:
        cell.release()
        return [f"{arm_name} {month} s{seed}: exists"]
    try:
        use_models(arm_name=source)
        loaded = pr.load(arm_name=source, month=month, seed=seed, device=device)
        test = data.build_test(month=month, standardiser=loaded.standardiser)
        inputs = fit.Inputs(
            power=power, x=test.x, primary=test.primary, origin_idx=test.origin_idx,
            target_idx=test.target_idx, power_mean=loaded.power_mean, power_std=loaded.power_std,
            device=device,
            surprise=fit.surprise_for(arm=loaded.arm, month=month, seed=seed, device=device),
        )  # fmt: skip
        inputs.target_stats = fit.target_stats(
            mean=loaded.target_mean, std=loaded.target_std, device=device
        )
        clean = power_np[0]
        lines = []
        for w in missing:
            gen = w in GEN_WORLDS
            pieces: dict[str, list[np.ndarray]] = {}
            generator = torch.Generator(device=device).manual_seed(1_000 * seed + w)
            for start in range(0, len(test.x), pr.CHUNK):
                rows = torch.arange(start, min(start + pr.CHUNK, len(test.x)), device=device)
                world = torch.full((len(rows),), w, dtype=torch.long, device=device)
                if arm_name in PREDICT_ONLY:
                    q = forecast_u3(
                        loaded=loaded, inputs=inputs, rows=rows, world=world, generator=generator
                    )
                    res: dict[str, torch.Tensor] = {"q": q}
                else:
                    res = forecast(loaded=loaded, inputs=inputs, rows=rows, world=world)
                for i, name in enumerate(pr.LEVEL_NAMES):
                    pieces.setdefault(name, []).append(res["q"][:, i].float().cpu().numpy())
                if "scale_f" in res:
                    pieces.setdefault("scale_f", []).append(res["scale_f"].float().cpu().numpy())
                for name, value in flags.of(inputs=inputs, rows=rows, world=world).items():
                    pieces.setdefault(name, []).append(value.cpu().numpy())
            extra = (
                generated_columns(
                    test=test, shift=shift[w - GEN_WORLDS[0]], onset=onset[w - GEN_WORLDS[0]]
                )
                if gen
                else {}
            )
            actual = (
                power_np[w][test.primary, test.target_idx]
                if gen
                else clean[test.primary, test.target_idx]
            )
            frame = pl.DataFrame(
                {
                    "primary": test.primary, "origin_us": test.origin, "valid_us": test.valid,
                    "lead": test.lead, "actual": actual,
                    "actual_clean": clean[test.primary, test.target_idx],
                    **{k: np.concatenate(v) for k, v in pieces.items()},
                    **extra,
                }
            )  # fmt: skip
            target = PRED_DIR / arm_name / f"{month}_s{seed}_w{w}.parquet"
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".tmp")
            frame.write_parquet(temporary)
            temporary.rename(target)
            lines.append(f"{arm_name} {month} s{seed} w{w}: {frame.height:,} rows")
        return lines
    finally:
        cell.release()


def link_references(*, seeds: tuple[int, ...] = fit.SEEDS) -> int:
    """Link experiment 428's forecasts of the reference arms (worlds 0, 4, 5, 6) into this folder."""
    n = 0
    for arm in REFERENCE_ARMS:
        target = PRED_DIR / arm
        target.mkdir(parents=True, exist_ok=True)
        for source in (data.WORK_DIR / "predictions" / arm).glob("*.parquet"):
            seed = int(source.stem.split("_s")[1].split("_")[0])
            if seed in seeds:
                link = target / source.name
                if not link.is_symlink() and not link.exists():
                    link.symlink_to(source)
                    n += 1
    return n


def selftest() -> None:
    """Check the shifted target, the token dropout, the origin draw, and the corrupted-row flag."""
    torch.manual_seed(0)
    n_p, grid = 2, 2_000
    rng = np.random.default_rng(0)
    clean = rng.uniform(0.2, 0.8, (n_p, grid)).astype(np.float32)
    shift = np.zeros((n_p, grid), dtype=np.float32)
    shift[0, 900:1_300] = 0.25
    nonneg = np.array([False, False])
    injected = clean.copy()
    injected[1, 100:200] += 0.5  # a load-conserving injected event: target unaffected
    power = np.stack(
        [
            worlds.apply_shift(power=clean, shift=shift, nonnegative=nonneg),
            worlds.apply_shift(power=injected, shift=shift, nonnegative=nonneg),
        ]
    )
    target_power = worlds.apply_shift(power=clean, shift=shift, nonnegative=nonneg)
    target_power = np.stack([target_power, target_power])
    rows = 400
    origin = rng.integers(1_000, 1_300, rows)
    target = origin + rng.integers(1, 200, rows)
    primary = np.zeros(rows, dtype=np.int64)
    median = torch.from_numpy(clean + 0.05)
    half = torch.full((n_p, grid), 0.1)
    inputs = Inputs446(
        power=torch.from_numpy(power), x=np.zeros((rows, 4), dtype=np.float32), primary=primary,
        origin_idx=origin, target_idx=target, power_mean=0.5, power_std=0.2, device="cpu",
        surprise=(median, half), clean_power=torch.from_numpy(clean),
        target_power=torch.from_numpy(target_power),
    )  # fmt: skip
    inputs.target_stats = (torch.zeros(n_p), torch.ones(n_p))
    arm = fit.ARMS["S1"]
    r = torch.arange(rows)
    world0 = torch.zeros(rows, dtype=torch.long)
    # Lags and target carry the same level change: inside 900..1300 both read +0.25.
    lag_idx = win.lag_indices(
        origin=torch.from_numpy(origin), target=torch.from_numpy(target)
    ).numpy()
    raw = inputs.raw(arm=arm, rows=r, world=world0)
    z = inputs.target_z(rows=r, world=world0).numpy()
    inside_target = (target >= 900) & (target < 1_300)
    assert np.allclose(z[inside_target], clean[0, target[inside_target]] + 0.25, atol=1e-5)
    assert np.allclose(z[~inside_target], clean[0, target[~inside_target]], atol=1e-5)
    lags = raw["lags"].numpy()
    inside_lag = (lag_idx >= 900) & (lag_idx < 1_300)
    moved = lags - clean[0, lag_idx]
    assert np.allclose(moved[inside_lag], 0.25, atol=1e-5), "lags must carry the level change"
    assert np.allclose(moved[~inside_lag], 0.0, atol=1e-5)
    assert (
        inside_target.any() and inside_lag.any() and (inside_lag.any(axis=1) & inside_target).any()
    ), "toy rows must put a lag and a target inside the event together"
    # An injected event changes the lags of world 1 but not its target.
    world1 = torch.ones(rows, dtype=torch.long)
    z1 = inputs.target_z(rows=r, world=world1).numpy()
    assert np.allclose(z1, z, atol=1e-6), "injected events must leave the target alone"
    # Corrupted rows: unshifted clean series is the comparison, so level changes count too.
    flagged = inputs.corrupted(raw=raw).numpy()
    assert np.array_equal(flagged, inside_lag.any(axis=1)), "corrupted flag must follow the shift"
    # Training augmentation.
    inputs.train = True
    big = torch.arange(rows).repeat(40)
    bigworld = torch.zeros(len(big), dtype=torch.long)
    raw_t = inputs.raw(arm=arm, rows=big, world=bigworld)
    dropped = float(torch.isnan(raw_t["lags"]).float().mean())
    assert 0.15 < dropped < 0.28, dropped  # 0.2 plus lags that fall off no edge in this toy
    hours = (torch.from_numpy(origin).repeat(40) - raw_t["origin"]) / HALF_HOURS_PER_HOUR
    assert int(hours.min()) == 0 and int(hours.max()) == ORIGIN_HOURS - 1, "8 origin hours"
    x, _, _ = inputs.assemble(arm=arm, raw=raw_t)
    missing = x[:, -9:-6]
    assert torch.equal(missing.bool(), torch.isnan(raw_t["lags"])), "dropped lag is flagged missing"
    assert float((x[:, -3:][torch.isnan(raw_t["lags"])]).abs().max()) == 0.0, (
        "dropped lag has no surprise"
    )
    spread_zero = float(((x[:, -3:] == 0) & ~torch.isnan(raw_t["lags"])).float().mean())
    assert 0.1 < spread_zero < 0.3, spread_zero  # the surprise is exactly zero only when dropped
    inputs.train = False
    raw_e = inputs.raw(arm=arm, rows=r, world=world0)
    assert torch.equal(raw_e["origin"], torch.from_numpy(origin)), "no origin draw at test time"
    assert not torch.isnan(raw_e["lags"]).any()
    print("selftest passed")


def main() -> None:
    """Fit, forecast, link, or self-test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fit", "predict", "link", "selftest"))
    parser.add_argument("--arms", nargs="+", default=list(EPOCH_SEEDS_FOR_DEFAULT))
    parser.add_argument("--months", nargs="+", default=list(fit.SCREEN_MONTHS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(fit.SEEDS))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "selftest":
        install()
        selftest()
        return
    install()
    if args.command == "link":
        print(f"linked {link_references()} forecast files")
        return
    if args.command == "fit":
        cache: dict = {}
        stack_cache: dict = {}
        for month in args.months:
            for seed in args.seeds:
                for name in args.arms:
                    print(
                        fit_one(
                            arm_name=name, month=month, seed=seed, device=args.device,
                            rows_cache=cache, stack_cache=stack_cache,
                        ),
                        flush=True,
                    )  # fmt: skip
        return
    stack = eval_stack()
    power = torch.from_numpy(stack[0]).to(args.device)
    flags = pr.Flags(code=stack[1], real=data.real_event_mask(), power=stack[0], device=args.device)
    for month in args.months:
        for seed in args.seeds:
            for name in args.arms:
                for line in predict_month(
                    arm_name=name, month=month, seed=seed, device=args.device, power=power,
                    stack=stack, flags=flags,
                ):  # fmt: skip
                    print(line, flush=True)


if __name__ == "__main__":
    main()
