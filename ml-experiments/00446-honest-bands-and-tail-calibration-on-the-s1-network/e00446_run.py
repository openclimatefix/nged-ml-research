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

**Brief C arms** (`tail_arms`). Each is a change to the tails of a base network, `S1` by default.
`--base NAME` names another base (a Brief B arm, for example `U2`); the arm is then called
`T2-on-U2`, and every command takes the plain kind (`T2`) with `--base`. All use the base's
training worlds and fit settings.

| Arm | What changes from the base |
|---|---|
| `T1` | Student-t mixture components (`e00446_models.StudentMdn`), degrees of freedom learned, 2.5 or more |
| `T2` | Loss is the negative log-likelihood plus 0.2 times the pinball losses at q01, q05, q95, q99 |
| `T3` | Trunk frozen; the output layer is fitted for `FINETUNE_EPOCHS` epochs on the four tail pinball losses, from the base model of the same fold and seed |
| `T4` | Seeds 3 and 4 of the base recipe are fitted here; the forecast pools seeds 0 to 2 of the base with them (5 mixtures) |
| `T5` | `T2` with the tail weight `0.2 * (1 + w * cold)`, `w = COLD_SLOPE = 1.0` |

The tail loss of `T2`, `T3`, and `T5` sums the pinball losses at the four tail levels of the
mixture. Those four levels sit on the 11-level grid of the forecast output (`TAIL_INDEX`), and
`models.tail_quantiles_st` finds each by bisection and attaches the implicit-function gradient, so
the loss is differentiable in the mixture weights, means, and spreads. The `T3` loss has no
likelihood term, and its validation criterion is the same tail loss. `cold` is a row flag built
from the fold's own training rows (`cold_flags`): local month November to March, cosine of the solar
zenith below zero, and the ensemble-mean 2 m temperature of the row's own run at or below the
Primary's 10th percentile of winter-night values in the fold's training window, as experiment 441
defined cold winter nights. `T4` writes one pooled forecast, copied to the files of seeds 0 to 2,
so its seed spread is zero.

A mkdir lock per (arm, month, seed) cell lets two processes share cells; a crashed process leaves
its lock folder behind, and `rmdir` on it releases the cell.

```bash
source <paths>; PYTHONPATH=$PP OMP_NUM_THREADS=1 \
  systemd-run --user --scope -q -p MemoryMax=8G -p OOMPolicy=continue \
  uv run --no-sync python ml-experiments/00446-*/e00446_run.py fit --arms U1n-2 U2 --months 2026-01 --seeds 0
uv run --no-sync python ml-experiments/00446-*/e00446_run.py predict --arms U1n-2 U2 U3 --months 2026-01 --seeds 0
uv run --no-sync python ml-experiments/00446-*/e00446_run.py selftest
uv run --no-sync python ml-experiments/00446-*/e00446_run.py fit --arms T1 T2 T3 T5 --base S1 --months 2026-01 --seeds 0
uv run --no-sync python ml-experiments/00446-*/e00446_run.py fit --arms T4 --seeds 3 4
uv run --no-sync python ml-experiments/00446-*/e00446_run.py fit --arms T2 --smoke --months 2026-01 --seeds 0
```

`--smoke` fits one epoch and writes under `work/smoke/` instead of the real folders.
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
import e00422_prep as prep422  # ty: ignore[unresolved-import]
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
MODELS_DIR: Path = WORK_DIR / "models"
PRED_DIR: Path = WORK_DIR / "predictions"
LOCK_DIR: Path = WORK_DIR / "locks"
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
TAIL_KINDS: Final[tuple[str, ...]] = ("T1", "T2", "T3", "T4", "T5")
DEFAULT_BASE: Final[str] = "S1"
TAIL_WEIGHT: Final[float] = 0.2
"""Weight of the four tail pinball losses beside the negative log-likelihood (arms T2 and T5)."""
COLD_SLOPE: Final[float] = 1.0
"""The `w` of arm T5: the tail weight is `TAIL_WEIGHT * (1 + COLD_SLOPE * cold)`."""
FINETUNE_EPOCHS: Final[int] = 5
EXTRA_SEEDS: Final[tuple[int, ...]] = (3, 4)
WINTER_MONTHS: Final[tuple[int, ...]] = (11, 12, 1, 2, 3)
WINTER_NIGHT_QUANTILE: Final[float] = 0.10
TIMEZONE: Final[str] = "Europe/London"
REAL_MODELS_DIR: Final[Path] = MODELS_DIR
"""Where the Brief B models live; `--smoke` redirects `MODELS_DIR` and leaves this one alone."""
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
    augment: bool = True
    """Train with the token dropout and the origin draw of `Inputs446` (S1's own recipe has neither)."""
    tail: str = ""
    """`student`, `pinball`, `pinball_cold`, or `finetune` for the tail arms; empty otherwise."""
    base: str = ""
    """The base arm of a tail arm."""
    pool: bool = False
    """Forecast by pooling the base's seeds 0 to 2 with this arm's seeds 3 and 4 (arm T4)."""


ARMS446: dict[str, Arm446] = {
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


def base_spec(*, base_name: str) -> Arm446:
    """Return the spec of a tail arm's base: a Brief B arm, or S1 as experiment 428 trains it."""
    if base_name in ARMS446 and not ARMS446[base_name].base:
        return ARMS446[base_name]
    if base_name == DEFAULT_BASE:
        return Arm446(DEFAULT_BASE, 0.0, augment=False)
    msg = f"unknown base {base_name!r}: use S1 or a Brief B arm ({', '.join(ARMS446)})"
    raise ValueError(msg)


def tail_arm_name(*, kind: str, base_name: str) -> str:
    """Return `T2` for the S1 base and `T2-on-U2` for another base."""
    return kind if base_name == DEFAULT_BASE else f"{kind}-on-{base_name}"


def tail_arms(*, base_name: str) -> dict[str, Arm446]:
    """Return the Brief C arms built on `base_name`, keyed by arm name (no `T1` on a scale base)."""
    b = base_spec(base_name=base_name)
    shared = {
        "rate": b.rate, "trimmed": b.trimmed, "scale": b.scale, "interval": b.interval,
        "coverage": b.coverage, "augment": b.augment, "base": base_name,
    }  # fmt: skip
    specs = {"T1": "student", "T2": "pinball", "T3": "finetune", "T5": "pinball_cold"}
    if b.scale:
        del specs["T1"]  # Student-t components have no scale path
    out = {
        tail_arm_name(kind=kind, base_name=base_name): Arm446(
            tail_arm_name(kind=kind, base_name=base_name),
            tail=tail,
            **shared,  # ty: ignore[invalid-argument-type]
        )
        for kind, tail in specs.items()
    }
    name = tail_arm_name(kind="T4", base_name=base_name)
    out[name] = Arm446(name, pool=True, **shared)  # ty: ignore[invalid-argument-type]
    return out


ARMS446.update(tail_arms(base_name=DEFAULT_BASE))
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
        if spec is not None and spec.tail == "student":
            return models.StudentMdn(
                n_inputs=n_inputs, n_primaries=n_primaries, hidden=fit.SETTING.hidden,
                layers=fit.SETTING.layers, components=fit.SETTING.components,
                dropout=fit.SETTING.dropout,
            )  # fmt: skip
        if spec is None or not spec.scale:
            return original(arm=arm, n_inputs=n_inputs, n_primaries=n_primaries)
        return models.ScaleMdn(
            n_inputs=n_inputs, n_primaries=n_primaries, lead_cols=lead_cols,
            hidden=fit.SETTING.hidden, layers=fit.SETTING.layers,
            components=fit.SETTING.components, dropout=fit.SETTING.dropout,
        )  # fmt: skip

    make_model._installed_446 = True  # ty: ignore[unresolved-attribute]
    fit.make_model = make_model  # ty: ignore[invalid-assignment]


def use_models(*, arm_name: str) -> None:
    """Point experiment 428's loader at the folder that holds the arm's saved networks."""
    if arm_name in REFERENCE_ARMS:
        fit.MODELS_DIR = data.WORK_DIR / "models"
    elif arm_name in ARMS446 and ARMS446[arm_name].base:
        fit.MODELS_DIR = MODELS_DIR
    else:
        fit.MODELS_DIR = REAL_MODELS_DIR


def base_model_dir(*, base_name: str, month: str, seed: int) -> Path:
    """Return the folder of a base arm's saved network for a fold and seed.

    The Brief B arms are in this experiment's folder; S1 is in experiment 428's (seeds 0 to 2) or
    experiment 439's (seeds 3 to 5).
    """
    import e00439_run as run439  # ty: ignore[unresolved-import]

    for root in (REAL_MODELS_DIR, data.WORK_DIR / "models", run439.WORK_DIR / "models"):
        folder = root / base_name / month / f"s{seed}"
        if (folder / "meta.json").exists():
            return folder
    msg = f"no saved network for base {base_name} in fold {month}, seed {seed}"
    raise FileNotFoundError(msg)


def cold_flags(
    *, temperature: np.ndarray, cos_zenith: np.ndarray, local_month: np.ndarray,
    primary: np.ndarray, n_primaries: int,
) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """Return `(cold, threshold)`: the cold-winter-night flag per row and the Primary thresholds.

    A row is a winter night when its local month is November to March and the cosine of its solar
    zenith is below zero. The threshold of a Primary is the 10th percentile of the ensemble-mean
    2 m temperature over its winter-night rows, and a winter-night row is cold when its temperature
    is at or below the threshold (experiment 441's definition, on the rows of one fold's training
    window).
    """
    night = np.isin(local_month, WINTER_MONTHS) & (cos_zenith < 0.0)
    threshold = np.full(n_primaries, -np.inf)
    for p in range(n_primaries):
        sel = night & (primary == p)
        if sel.any():
            threshold[p] = np.quantile(temperature[sel], WINTER_NIGHT_QUANTILE)
    return night & (temperature <= threshold[primary]), threshold


def fold_cold(*, rows: data.Rows) -> np.ndarray:
    """Return the cold-winter-night flag of the rows of a fold (see `cold_flags`)."""
    _, names, _ = data.non_lag_layout()
    sd = rows.standardiser
    t_col, z_col = names.index("mean_temperature_2m"), names.index("cos_zenith")
    temperature = rows.x[:, t_col] * sd.std[t_col] + sd.mean[t_col]
    cos_zenith = rows.x[:, z_col] * sd.std[z_col] + sd.mean[z_col]
    valid_us = prep422.GRID_START_US + rows.target_idx * prep422.STEP_US
    month = (
        pl.Series(valid_us, dtype=pl.Int64)
        .cast(pl.Datetime("us"))
        .dt.replace_time_zone("UTC")
        .dt.offset_by("-30m")
        .dt.convert_time_zone(TIMEZONE)
        .dt.month()
        .to_numpy()
    )
    cold, _ = cold_flags(
        temperature=temperature, cos_zenith=cos_zenith, local_month=month,
        primary=rows.primary, n_primaries=len(rows.target_mean),
    )  # fmt: skip
    return cold


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
        self.cold: torch.Tensor | None = None
        """Cold-winter-night flag per fitted row (set for the cold-weighted arm T5)."""

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


def tail_rows(
    *, spec: Arm446, out: torch.Tensor, z: torch.Tensor, k: int, cold: torch.Tensor | None
) -> torch.Tensor | None:
    """Return the per-row tail loss of an arm (None for an arm with no tail term).

    The four tail quantiles come from `models.tail_quantiles_st` (bisection on the 11-level grid's
    four tail levels, implicit-function gradient). `T2` weights the pinball sum by `TAIL_WEIGHT`;
    `T5` by `TAIL_WEIGHT * (1 + COLD_SLOPE * cold)`; `T3` by 1.
    """
    if spec.tail not in ("pinball", "pinball_cold", "finetune"):
        return None
    log_w, mean, std = base.mixture_parts(out=out, components=k)
    quantiles = models.tail_quantiles_st(log_w=log_w, mean=mean, std=std)
    pin = models.tail_pinball(quantiles=quantiles, y=z)
    if spec.tail == "finetune":
        return pin
    weight = models.tail_weights(
        cold=cold if spec.tail == "pinball_cold" else None,
        weight=TAIL_WEIGHT, slope=COLD_SLOPE, like=pin,
    )  # fmt: skip
    return weight * pin


def row_loss(
    *, spec: Arm446, out: torch.Tensor, z: torch.Tensor, k: int, cold: torch.Tensor | None
) -> torch.Tensor:
    """Return the per-row loss: the negative log-likelihood plus the tail term (`T3`: tail term only)."""
    tail = tail_rows(spec=spec, out=out, z=z, k=k, cold=cold)
    if spec.tail == "finetune":
        assert tail is not None
        return tail
    nll = (
        models.student_mixture_nll(out=out, z=z, components=k)
        if spec.tail == "student"
        else base.mixture_nll(out=out, z=z, components=k)
    )
    return nll if tail is None else nll + tail


def train(
    *, spec: Arm446, arm: fit.Arm, model: torch.nn.Module, inputs: Inputs446, w: torch.Tensor,
    is_val: torch.Tensor, seed: int, device: str,
) -> tuple[int, float]:  # fmt: skip
    """Train with weighted negative log-likelihood plus the arm's extra terms; return epochs, best.

    The `T3` arm freezes the trunk, trains the output layer on the tail loss alone for
    `FINETUNE_EPOCHS` epochs, and keeps the epoch with the lowest validation tail loss.
    """
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    train_idx = torch.nonzero(~is_val).squeeze(1)
    val_idx = torch.nonzero(is_val).squeeze(1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if len(val_idx) > fit.VALIDATION_ROWS:
        pick = torch.randperm(len(val_idx), generator=generator)[: fit.VALIDATION_ROWS]
        val_idx = val_idx[pick.to(device)]
    val_world = val_idx % (data.N_TRAIN_WORLDS + 1)
    finetune = spec.tail == "finetune"
    parameters = models.freeze_trunk(model=model) if finetune else list(model.parameters())
    optimiser = torch.optim.AdamW(
        parameters, lr=fit.SETTING.lr, weight_decay=fit.SETTING.weight_decay
    )
    k = fit.SETTING.components
    max_epochs = FINETUNE_EPOCHS if finetune else fit.MAX_EPOCHS
    patience = max_epochs if finetune else arm.patience
    best, best_state, stale, epochs = float("inf"), None, 0, 0
    for epoch in range(min(max_epochs, fit.MAX_EPOCHS)):
        model.train()
        inputs.train = spec.augment
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
            cold = None if inputs.cold is None else inputs.cold[b]
            loss = (w[b] * row_loss(spec=spec, out=out, z=z, k=k, cold=cold)).sum() / w[b].sum()
            if spec.scale and not finetune:
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
            torch.nn.utils.clip_grad_norm_(parameters, fit.GRAD_CLIP)
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
                cold = None if inputs.cold is None else inputs.cold[chunk]
                losses.append(
                    row_loss(
                        spec=spec, out=out, z=inputs.target_z(rows=chunk, world=chunk_world),
                        k=k, cold=cold,
                    )
                )  # fmt: skip
            val = float(torch.cat(losses).mean())
        epochs = epoch + 1
        print(f"  epoch {epochs}: val loss {val:.4f}", flush=True)
        if val < best - 1e-5:
            best, stale = val, 0
            best_state = {a: v.detach().clone() for a, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
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
        if spec.tail == "pinball_cold":
            inputs.cold = torch.from_numpy(fold_cold(rows=rows)).to(device)
        if spec.tail == "finetune":
            folder = base_model_dir(base_name=spec.base, month=month, seed=seed)
            model.load_state_dict(torch.load(folder / "model.pt", map_location=device))
        n_parameters = models.count_parameters(model=model)
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
                    "parameters": n_parameters,
                    "rate": spec.rate, "trimmed": spec.trimmed, "tail": spec.tail,
                    "base": spec.base,
                }
            )
        )  # fmt: skip
        return (
            f"{arm_name} {month} s{seed}: {epochs} epochs, val loss {best:.4f}, "
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
    if isinstance(loaded.model, models.StudentMdn):
        return forecast_student(loaded=loaded, inputs=inputs, rows=rows, world=world)
    result = pr.forecast_rows(loaded=loaded, inputs=inputs, rows=rows, world=world)
    if isinstance(loaded.model, models.ScaleMdn):
        with torch.no_grad():
            raw = inputs.raw(arm=loaded.arm, rows=rows, world=world)
            x, _, _ = inputs.assemble(arm=loaded.arm, raw=raw)
            result["scale_f"] = torch.exp(torch.log(loaded.model.factors(x)).mean(dim=1))
    return result


def forecast_student(
    *, loaded: pr.Loaded, inputs: fit.Inputs, rows: torch.Tensor, world: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Return the 11 sorted quantiles (capacity fractions) of a Student-t mixture network (arm T1)."""
    with torch.no_grad():
        raw = inputs.raw(arm=loaded.arm, rows=rows, world=world)
        x, primary, _ = inputs.assemble(arm=loaded.arm, raw=raw)
        out = loaded.model(x, primary).double()
        log_w, mean, scale, df = models.student_parts(out=out, components=fit.SETTING.components)
        q = models.student_mixture_quantiles(log_w=log_w, mean=mean, scale=scale, df=df)
        mu = torch.from_numpy(loaded.target_mean).to(q.device)[primary][:, None]
        sd = torch.from_numpy(loaded.target_std).to(q.device)[primary][:, None]
        return {"q": torch.sort(q * sd + mu, dim=1).values}


def forecast_pooled(
    *, members: list[tuple[pr.Loaded, fit.Inputs]], rows: torch.Tensor, world: torch.Tensor
) -> torch.Tensor:
    """Return the 11 sorted quantiles of the equal-weight pool of Gaussian-mixture networks (arm T4).

    Each network's components are converted to the capacity scale with its own target statistics,
    and the pooled CDF is the average of the networks' CDFs (`models.pool_mixtures`).
    """
    pieces = []
    for loaded, inputs in members:
        res = pr.forecast_rows(loaded=loaded, inputs=inputs, rows=rows, world=world)
        primary = inputs.primary[rows]
        mu = torch.from_numpy(loaded.target_mean).to(rows.device)[primary][:, None]
        sd = torch.from_numpy(loaded.target_std).to(rows.device)[primary][:, None]
        pieces.append((res["log_w"], res["mean"] * sd + mu, res["std"] * sd))
    log_w, mean, std = models.pool_mixtures(pieces=pieces)
    return torch.sort(base.mixture_quantiles(log_w=log_w, mean=mean, std=std), dim=1).values


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


def pool_members(*, arm_name: str) -> list[tuple[str, int]]:
    """Return the `(arm, seed)` networks pooled by arm T4: the base's seeds 0 to 2, then its own 3 and 4."""
    return [(ARMS446[arm_name].base, s) for s in fit.SEEDS] + [(arm_name, s) for s in EXTRA_SEEDS]


def predict_month(
    *, arm_name: str, month: str, seed: int, device: str, power: torch.Tensor,
    stack: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], flags: pr.Flags,
) -> list[str]:  # fmt: skip
    """Forecast a test month in every world not yet on disk; return progress lines.

    A pooled arm (T4) forecasts once for all of `fit.SEEDS` and writes the same frame to each seed's
    file.
    """
    power_np, _, shift, onset = stack
    source = PREDICT_ONLY.get(arm_name, arm_name)
    reference = arm_name in REFERENCE_ARMS
    pooled = arm_name in ARMS446 and ARMS446[arm_name].pool
    out_seeds = list(fit.SEEDS) if pooled else [seed]
    lock_seed = out_seeds[0]
    worlds_todo = list(GEN_WORLDS) if reference else [*pr.EVAL_WORLDS, *GEN_WORLDS]

    def path_of(w: int, s: int) -> Path:
        return PRED_DIR / arm_name / f"{month}_s{s}_w{w}.parquet"

    def todo() -> list[int]:
        return [w for w in worlds_todo if not all(path_of(w, s).exists() for s in out_seeds)]

    if not todo():
        return [f"{arm_name} {month} s{seed}: exists"]
    cell = Cell(kind="predict", arm=arm_name, month=month, seed=lock_seed)
    if not cell.held:
        return [f"{arm_name} {month} s{seed}: locked by another process"]
    missing = todo()
    if not missing:
        cell.release()
        return [f"{arm_name} {month} s{seed}: exists"]
    try:
        member_arms = pool_members(arm_name=arm_name) if pooled else [(source, seed)]
        loaded_members = []
        for member_arm, member_seed in member_arms:
            use_models(arm_name=member_arm)
            loaded_members.append(
                pr.load(arm_name=member_arm, month=month, seed=member_seed, device=device)
            )
        loaded = loaded_members[0]
        for other in loaded_members[1:]:
            assert np.allclose(other.standardiser.mean, loaded.standardiser.mean), (
                "pooled networks must share the fold's standardiser"
            )
        test = data.build_test(month=month, standardiser=loaded.standardiser)
        member_inputs = []
        for member in loaded_members:
            inputs = fit.Inputs(
                power=power, x=test.x, primary=test.primary, origin_idx=test.origin_idx,
                target_idx=test.target_idx, power_mean=member.power_mean,
                power_std=member.power_std, device=device,
                surprise=fit.surprise_for(arm=member.arm, month=month, seed=member.seed, device=device),
            )  # fmt: skip
            inputs.target_stats = fit.target_stats(
                mean=member.target_mean, std=member.target_std, device=device
            )
            member_inputs.append(inputs)
        inputs = member_inputs[0]
        clean = power_np[0]
        lines = []
        for w in missing:
            gen = w in GEN_WORLDS
            pieces: dict[str, list[np.ndarray]] = {}
            generator = torch.Generator(device=device).manual_seed(1_000 * seed + w)
            for start in range(0, len(test.x), pr.CHUNK):
                rows = torch.arange(start, min(start + pr.CHUNK, len(test.x)), device=device)
                world = torch.full((len(rows),), w, dtype=torch.long, device=device)
                if pooled:
                    q = forecast_pooled(
                        members=list(zip(loaded_members, member_inputs, strict=True)),
                        rows=rows, world=world,
                    )  # fmt: skip
                    res: dict[str, torch.Tensor] = {"q": q}
                elif arm_name in PREDICT_ONLY:
                    q = forecast_u3(
                        loaded=loaded, inputs=inputs, rows=rows, world=world, generator=generator
                    )
                    res = {"q": q}
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
            for out_seed in out_seeds:
                target = path_of(w, out_seed)
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
    selftest_brief_c()
    print("selftest passed")


def selftest_brief_c() -> None:
    """Check the arm table, the cold flag, the tail weights in the loss, and the T3 loss."""
    # Arm table: S1-based arms follow experiment 428's recipe; a scale base has no T1.
    t = tail_arms(base_name="S1")
    assert set(t) == {"T1", "T2", "T3", "T4", "T5"} and not t["T2"].augment and t["T2"].rate == 0.0
    assert t["T1"].tail == "student" and t["T3"].tail == "finetune" and t["T4"].pool
    on_u2 = tail_arms(base_name="U2")
    assert "T1-on-U2" not in on_u2 and on_u2["T2-on-U2"].scale and on_u2["T2-on-U2"].rate == 0.02
    assert resolve_arms(names=["T2", "U2", "T5"], base_name="U2") == ["T2-on-U2", "U2", "T5-on-U2"]
    assert pool_members(arm_name="T4") == [("S1", 0), ("S1", 1), ("S1", 2), ("T4", 3), ("T4", 4)]
    # Cold flag: the 10th percentile of each Primary's winter-night temperatures, nights only.
    rng = np.random.default_rng(0)
    n = 4_000
    primary = rng.integers(0, 2, n)
    temperature = rng.normal(5.0, 4.0, n) + 10.0 * primary
    month = rng.choice([1, 2, 7], n)
    cos_zenith = rng.choice([-0.5, 0.5], n)
    cold, threshold = cold_flags(
        temperature=temperature, cos_zenith=cos_zenith, local_month=month, primary=primary,
        n_primaries=2,
    )  # fmt: skip
    night = np.isin(month, WINTER_MONTHS) & (cos_zenith < 0)
    assert not cold[~night].any(), "only winter nights can be cold"
    for p in range(2):
        sel = night & (primary == p)
        assert abs(float(cold[sel].mean()) - 0.10) < 0.02, (p, cold[sel].mean())
        assert threshold[1] > threshold[0] + 5.0
    # T5's tail weight inside the loss: 0.2 off cold rows and 0.2 * (1 + w) on them; T2's is 0.2.
    torch.manual_seed(0)
    k = 3
    out = torch.randn(8, 3 * k) * 0.3
    z = torch.randn(8)
    cold_rows = torch.tensor([True, False] * 4)
    plain = tail_rows(spec=ARMS446["T2"], out=out, z=z, k=k, cold=None)
    weighted = tail_rows(spec=ARMS446["T5"], out=out, z=z, k=k, cold=cold_rows)
    tune = tail_rows(spec=ARMS446["T3"], out=out, z=z, k=k, cold=None)
    assert plain is not None and weighted is not None and tune is not None
    ratio = weighted / plain
    assert torch.allclose(ratio[~cold_rows], torch.ones(4)), ratio
    assert torch.allclose(ratio[cold_rows], torch.full((4,), 1.0 + COLD_SLOPE)), ratio
    assert torch.allclose(plain, TAIL_WEIGHT * tune), "T2's term is 0.2 times T3's"
    lw, mean, std = base.mixture_parts(out=out, components=k)
    direct = models.tail_pinball(
        quantiles=models.tail_quantiles_st(log_w=lw, mean=mean, std=std), y=z
    )
    assert torch.allclose(tune, direct)
    nll = base.mixture_nll(out=out, z=z, components=k)
    total = row_loss(spec=ARMS446["T2"], out=out, z=z, k=k, cold=None)
    assert torch.allclose(total, nll + plain), "T2 loss is the likelihood plus the tail term"
    assert torch.allclose(row_loss(spec=ARMS446["T3"], out=out, z=z, k=k, cold=None), tune), (
        "T3 trains on the tail loss alone"
    )
    print("selftest brief C passed")


def resolve_arms(*, names: list[str], base_name: str) -> list[str]:
    """Map the plain kinds `T1` to `T5` to the arm names built on `base_name`."""
    return [tail_arm_name(kind=n, base_name=base_name) if n in TAIL_KINDS else n for n in names]


def redirect_to_smoke() -> None:
    """Send fits, forecasts, and locks to `work/smoke/`, and cap every fit at one epoch."""
    global MODELS_DIR, PRED_DIR, LOCK_DIR
    root = WORK_DIR / "smoke"
    MODELS_DIR, PRED_DIR, LOCK_DIR = root / "models", root / "predictions", root / "locks"
    fit.MAX_EPOCHS = 1


def main() -> None:
    """Fit, forecast, link, or self-test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fit", "predict", "link", "selftest"))
    parser.add_argument("--arms", nargs="+", default=list(EPOCH_SEEDS_FOR_DEFAULT))
    parser.add_argument("--months", nargs="+", default=list(fit.SCREEN_MONTHS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(fit.SEEDS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--base", default=DEFAULT_BASE, help="base of the Brief C arms: S1 or a Brief B arm"
    )
    parser.add_argument(
        "--smoke", action="store_true", help="fit one epoch and write under work/smoke/"
    )
    args = parser.parse_args()
    if args.base != DEFAULT_BASE:
        ARMS446.update(tail_arms(base_name=args.base))
    if args.smoke:
        redirect_to_smoke()
    arm_names = resolve_arms(names=args.arms, base_name=args.base)
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
                for name in arm_names:
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
            for name in arm_names:
                for line in predict_month(
                    arm_name=name, month=month, seed=seed, device=args.device, power=power,
                    stack=stack, flags=flags,
                ):  # fmt: skip
                    print(line, flush=True)


if __name__ == "__main__":
    main()
