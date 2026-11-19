r"""Fit one (arm, fold, seed, injection state) job of experiment 710 and write its forecasts.

**The outputs have experiment 707's layout** under `<data>/00710-*/work/`:

- `seed_preds/<arm>-<state>/<month>_s<seed>.parquet`: raw quantiles as a fraction of capacity,
  unwidened, keyed on `time_series_id`, `power_fcst_init_time`, `valid_time`. The file is the job's
  completion marker and is written last, through a temporary name. A tuning fold's file also holds
  `lead_hours` and the target `y`.
- `bank/<arm>-<state>/<month>_s<seed>_p<pass>.parquet`: the same quantiles forecast with the lags of
  one pass of experiment 581's injected-event bank (passes `k` and `k + 12` for fold number `k`),
  or, for pass 100, with the lags of the verified real events removed (`corrected_clean`).
- `models/<arm>-<state>/<month>/s<seed>/`: `pooled.pt`, `meta.json`, `standardiser.npz`.
- `pretrained/<state key>/<state>/<month>_s<seed>.pt` and `.json`: the weights after a stage that
  other arms share (`era`, `ensF`, `era+ensF`).
- `logs/<arm>-<state>/<month>_s<seed>.json`: rows, stage logs with the per-epoch weight shares and
  the shifted-lag shares, steps, and seconds.

**A job is resumable.** A job whose forecast file exists is skipped. A stage whose state file exists
is loaded and not trained. In the `n` state of a scored fold with seeds 0 to 4, the stored states of
experiment 707 (`pretrained/ensF_ftw730` and the others) are the stage's state.

**The jobs of one fold, seed, and state share one world and one set of rows.** The world (synthetic
events) is built once for the (fold, seed) and read by every stage of every arm.

**`--stored M1-n|M2-n|M3-n` forecasts, without a fit,** from the stored networks of experiment 707:
the two bank passes (and, for `M3-n`, the ordinary forecast) from `models/A_F`, `models/R_F`, and
`pretrained/ensF_ftw730`.

**`--pilot` fits one fold and prints the timing and the memory.** It writes under `work/pilot`, fits
`M1` and `W-h0.5` in the `i` state on fold 2026-01 with seed 0 unless told otherwise, and prints the
origin-time weather example of the design's section c.

```bash
export PYTHONPATH=$PWD/ml-experiments/00710-ens-only-one-shot-training-recipe
for d in 00707-ens-backfill-pretraining-gain-against-ens-history-length \
         00581-robust-multiweek-lag-encoder-x3; do
    export PYTHONPATH=$PYTHONPATH:$PWD/ml-experiments/$d
done
export DATA_PATH_INTERNAL=<path to the private data folder>
export NGED_S3_BUCKET_URL=https://example.com NGED_S3_BUCKET_ACCESS_KEY=dummy
export NGED_S3_BUCKET_SECRET=dummy
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 POLARS_MAX_THREADS=1 \
    RAYON_NUM_THREADS=1 systemd-run --user --scope \
    -p MemoryMax=16G -p MemorySwapMax=0 -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00710-*/e00710_fit.py --pilot
```
"""

import argparse
import contextlib
import hashlib
import json
import math
import os
import resource
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from functools import cache
from pathlib import Path
from typing import Final

import e00581_bank as bank581  # ty: ignore[unresolved-import]
import e00707_arms as arms707  # ty: ignore[unresolved-import]
import e00707_common as common  # ty: ignore[unresolved-import]
import e00707_folds as folds707  # ty: ignore[unresolved-import]
import e00707_net as net707  # ty: ignore[unresolved-import]
import e00707_prep as prep  # ty: ignore[unresolved-import]
import e00710_arms as arms  # ty: ignore[unresolved-import]
import e00710_folds as folds  # ty: ignore[unresolved-import]
import e00710_inject as inject  # ty: ignore[unresolved-import]
import e00710_net as net  # ty: ignore[unresolved-import]
import e00710_weights as weights  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

OUT_DIR: Final[Path] = arms.WORK_DIR / "work"
PILOT_DIR: Final[Path] = OUT_DIR / "pilot"
STORED_707_WORK: Final[Path] = arms.EXPERIMENT_707_DIR / "work"
MARKER: Final[Path] = OUT_DIR / "known_answers_passed.json"
"""Written by `e00710_known_answer.py`; the fit refuses to start without it (except the pilot)."""
REAL_PASS: Final[int] = bank581.REAL_PASS
STORED_ARMS: Final[dict[str, tuple[str, str | None]]] = {
    "M1-n": ("A_F", None),
    "M2-n": ("R_F", None),
    "M3-n": ("R_F", "ensF_ftw730"),
}
"""A stored network as (the 707 models folder that holds its scale, the 707 state folder if any)."""
LOCK_POLL_SECONDS: Final[float] = 20.0
"""How long a stream waits between looks at a stage state that another stream is training."""
EMPTY_LOCK_SECONDS: Final[float] = 60.0
"""A lock file that holds no pid and is older than this is stale (its stream died while writing)."""
EVENT_MARGIN_DAYS: Final[int] = 7
"""A real event matters for a month if it overlaps the month or the 7 days before it."""


@dataclass
class RowTensors:
    """One set of stage rows on the device, with the arrays the weight rules need.

    Attributes:
        x: Inputs.
        z: The target standardised per Primary.
        primary: Each row's Primary index.
        is_val: Whether each row is in a held-out week.
        shifted: Whether each row carries a shifted lag.
        age: Row age at the refit in days.
        valid_us: Valid times in microseconds since the epoch.
        primary_np: The Primary indices as a numpy array.
        is_val_np: The held-out mask as a numpy array.
        shifted_np: The shifted-lag mask as a numpy array.
        lead: Each row's lead in hours.
        origin_us: Each row's origin in microseconds since the epoch.
    """

    x: torch.Tensor
    z: torch.Tensor
    primary: torch.Tensor
    is_val: torch.Tensor
    shifted: torch.Tensor
    age: np.ndarray
    valid_us: np.ndarray
    primary_np: np.ndarray
    is_val_np: np.ndarray
    shifted_np: np.ndarray
    lead: np.ndarray
    origin_us: np.ndarray


@dataclass
class MonthScale:
    """What a fold keeps after its unshifted rows are built: the scale and the ERA spread table."""

    scale: net707.Scale
    spread: np.ndarray
    n_recent_rows: int


@dataclass
class Context:
    """Everything that one (fold, seed, state) group shares between its arms."""

    month: str
    seed: int
    state: arms.StateType
    refit: datetime
    month_scale: MonthScale
    world: inject.World | None
    device: str
    h_t: float
    pp_half_lives: np.ndarray | None
    out_dir: Path
    use_stored: bool = True
    shift_share_range: tuple[float, float] | None = None
    tensors: dict[str, RowTensors] | None = None

    def power(self) -> dict[str, np.ndarray] | None:
        """Return the world's power by site label, or `None` in the `n` state."""
        return None if self.world is None else self.world.power


def write_json(*, payload: dict, target: Path) -> None:
    """Write a JSON file through a temporary name.

    Args:
        payload: The content.
        target: The final path.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=1))
    temporary.replace(target)


def forecast_path(*, out_dir: Path, arm: str, state: str, month: str, seed: int) -> Path:
    """Return where a job's raw quantile forecast is stored."""
    return out_dir / "seed_preds" / f"{arm}-{state}" / f"{month}_s{seed}.parquet"


def bank_path(*, out_dir: Path, arm: str, state: str, month: str, seed: int, pass_id: int) -> Path:
    """Return where one bank pass's forecast of a job is stored."""
    return out_dir / "bank" / f"{arm}-{state}" / f"{month}_s{seed}_p{pass_id}.parquet"


def state_path(*, out_dir: Path, key: str, state: str, month: str, seed: int) -> Path:
    """Return where the weights after a shared stage are stored."""
    return out_dir / "pretrained" / key / state / f"{month}_s{seed}.pt"


def save_state(*, state: dict, target: Path) -> None:
    """Write a stage's weights through a temporary name that carries this process's id.

    Args:
        state: The network's state dictionary.
        target: The final path.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    torch.save(state, temporary)
    temporary.rename(target)


def lock_is_stale(*, lock: Path) -> bool:
    """Return whether a lock file names a process that no longer runs, or is unreadable and old.

    Args:
        lock: The lock file.

    Returns:
        `True` if the pid in the file is dead, or if the file holds no pid and is older than
        `EMPTY_LOCK_SECONDS`. A reused pid cannot be told apart from a live stream.
    """
    try:
        text = lock.read_text().strip()
        age = time.time() - lock.stat().st_mtime
    except FileNotFoundError:
        return False
    try:
        pid = int(text)
    except ValueError:
        return age > EMPTY_LOCK_SECONDS
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def claim_state(*, own: Path) -> Path | None:
    """Take the lock of a shared stage state, or wait for the stream that holds it.

    Two streams that fit different arms of one (fold, seed, state) group share the stage states
    (`ensF`, `era`, `era+ensF`). The lock file `<state>.lock` lets exactly one stream train a state.
    Any other stream waits and then loads the state that the first stream saved, so that every arm
    of the group starts from one network. The pid is written to a temporary file that is linked
    to the lock name, so the lock is never empty. A waiter removes a stale lock by renaming it to a
    name of its own first, so that only one waiter wins.

    Args:
        own: This experiment's path of the state file.

    Returns:
        `None` if this process now holds the lock and must train and save the state (and then call
        `release_state`), or the state's path if the file exists.
    """
    lock = own.with_name(own.name + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    mine = own.with_name(f"{own.name}.{os.getpid()}.pid")
    mine.write_text(str(os.getpid()))
    announced = False
    try:
        while True:
            if own.exists():
                return own
            try:
                os.link(mine, lock)
            except FileExistsError:
                if lock_is_stale(lock=lock):
                    grave = lock.with_name(f"{lock.name}.{os.getpid()}.stale")
                    try:
                        lock.rename(grave)
                    except FileNotFoundError:
                        continue
                    if not lock_is_stale(lock=grave):
                        # Another waiter took a fresh lock between the check and the rename.
                        with contextlib.suppress(FileExistsError):
                            os.link(grave, lock)
                    grave.unlink(missing_ok=True)
                    continue
                if not announced:
                    print(f"waiting for {lock}", flush=True)
                    announced = True
                time.sleep(LOCK_POLL_SECONDS)
                continue
            if own.exists():
                lock.unlink(missing_ok=True)
                return own
            return None
    finally:
        mine.unlink(missing_ok=True)


def release_state(*, own: Path) -> None:
    """Remove the lock that `claim_state` took."""
    own.with_name(own.name + ".lock").unlink(missing_ok=True)


def stored_707_state(*, key: str, state: str, month: str, seed: int) -> Path | None:
    """Return experiment 707's stored state that equals a stage, or `None` if there is none.

    Only the `n` state of a scored fold with a stored seed can use experiment 707's states.

    Args:
        key: The stage's `state_key`.
        state: `i` or `n`.
        month: The fold.
        seed: The network seed.

    Returns:
        The path of the stored `.pt` file, or `None`.
    """
    if state != "n" or month not in arms.SCORED_MONTHS or seed not in arms.STORED_SEEDS:
        return None
    folder = arms.STORED_707_KEYS.get(key)
    if folder is None:
        return None
    path = STORED_707_WORK / "pretrained" / folder / f"{month}_s{seed}.pt"
    return path if path.exists() else None


def month_scale(*, month: str) -> MonthScale:
    """Build a fold's unshifted last-730-days rows and keep their scale and spread table.

    Args:
        month: The fold.

    Returns:
        The standardiser and target statistics (fitted on the unshifted rows), the ERA5 spread
        table, and the number of recent rows.
    """
    fold = folds.build_fold(refit=prep.refit_of(month=month), window=arms707.W24, power=None)
    spread = folds707.spread_table(fold=fold)
    scale = net707.Scale(
        standardiser=fold.standardiser, target_mean=fold.target_mean, target_std=fold.target_std
    )
    return MonthScale(scale=scale, spread=spread, n_recent_rows=len(fold.x))


def to_tensors(
    *, rows: folds.Rows, scale: net707.Scale, refit: datetime, device: str
) -> RowTensors:
    """Return the rows as device tensors with the age and the masks the weight rules need.

    Args:
        rows: The stage rows.
        scale: The fold's scale, for the target standardisation.
        refit: The fold's refit time.
        device: The torch device.

    Returns:
        The tensors and arrays.
    """
    z = (rows.y - scale.target_mean[rows.primary]) / scale.target_std[rows.primary]
    shifted = rows.shifted if rows.shifted is not None else np.zeros(len(rows.y), dtype=bool)
    return RowTensors(
        x=torch.from_numpy(rows.x).to(device),
        z=torch.from_numpy(z.astype(np.float32)).to(device),
        primary=torch.from_numpy(rows.primary).to(device),
        is_val=torch.from_numpy(rows.is_val).to(device),
        shifted=torch.from_numpy(shifted).to(device),
        age=weights.age_days(valid_us=rows.valid, refit_us=int(refit.timestamp() * 1e6)),
        valid_us=rows.valid,
        primary_np=rows.primary,
        is_val_np=rows.is_val,
        shifted_np=shifted,
        lead=rows.lead,
        origin_us=rows.origin,
    )


def tensors_for(*, ctx: Context, which: str) -> RowTensors:
    """Return the tensors of the `era` or `F` rows of a group, building them on first use.

    Args:
        ctx: The group.
        which: `era` or `F`.

    Returns:
        The tensors of the rows.
    """
    if ctx.tensors is None:
        ctx.tensors = {}
    if which not in ctx.tensors:
        started = time.monotonic()
        ms = ctx.month_scale
        if which == "F":
            rows, _ = folds.ens_rows(
                standardiser=ms.scale.standardiser,
                refit=ctx.refit,
                window=arms707.W_F,
                power=ctx.power(),
            )
        else:
            rows, _ = folds.fold_pretrain(
                standardiser=ms.scale.standardiser,
                spread=ms.spread,
                refit=ctx.refit,
                power=ctx.power(),
            )
        ctx.tensors[which] = to_tensors(
            rows=rows, scale=ms.scale, refit=ctx.refit, device=ctx.device
        )
        print(
            f"  rows {which} {ctx.month} s{ctx.seed} {ctx.state}: {len(rows.y):,} rows, "
            f"{int(rows.is_val.sum()):,} held out, built in {time.monotonic() - started:.0f} s",
            flush=True,
        )
    return ctx.tensors[which]


def stage_masks(*, plan: arms.StagePlan, rt: RowTensors) -> tuple[np.ndarray, np.ndarray]:
    """Return a stage's training mask and selection mask over its rows.

    Training rows are the non-held-out rows inside the stage's window. The selection rows are all
    held-out rows (`all_held`) or the held-out rows of the last 730 days (`recent_held`, `V_sel`).

    Args:
        plan: The stage.
        rt: The stage's rows.

    Returns:
        The two boolean masks.
    """
    window = {"era": None, "F": None, "W730": arms.RECENT_DAYS, "W365": 365}[plan.rows]
    inside = (
        np.ones(len(rt.age), dtype=bool)
        if window is None
        else weights.within_window(age=rt.age, window_days=window)
    )
    train = ~rt.is_val_np & inside
    recent = weights.within_window(age=rt.age, window_days=arms.RECENT_DAYS)
    held = rt.is_val_np if plan.val == "all_held" else rt.is_val_np & recent
    return train, held


def make_provider(
    *, plan: arms.StagePlan, rt: RowTensors, ctx: Context, train: np.ndarray
) -> net.WeightProvider:
    """Return the weight provider of a stage.

    Args:
        plan: The stage.
        rt: The stage's rows.
        ctx: The group (for the half-life `h_T`, the per-Primary half-lives, and the placebo seed).
        train: The training mask, which defines the Primary shares of `per_primary`.

    Returns:
        A function from the 0-based epoch to a weight tensor, or `None` for flat weights.

    Raises:
        ValueError: If the rule needs a setting that was not given.
    """
    rule, device = plan.weight, ctx.device

    def tensor(w: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(w).to(device)

    if rule.kind == "flat":
        return lambda _epoch: None
    if rule.kind == "exp":
        assert rule.half_life_years is not None
        fixed = tensor(weights.exponential(age=rt.age, half_life_years=rule.half_life_years))
        return lambda _epoch: fixed
    if rule.kind == "floor":
        assert rule.half_life_years is not None
        floor = tensor(
            weights.floor_exponential(
                age=rt.age, half_life_years=rule.half_life_years, floor=rule.floor
            )
        )
        return lambda _epoch: floor
    if rule.kind == "per_primary":
        if ctx.pp_half_lives is None:
            msg = "arm PP needs --pp-half-lives"
            raise ValueError(msg)
        pp = tensor(
            weights.per_primary(
                age=rt.age,
                primary=rt.primary_np,
                half_lives_years=ctx.pp_half_lives,
                counted=train,
            )
        )
        return lambda _epoch: pp
    if rule.kind == "placebo":
        permuted = weights.permuted_age_days(
            valid_us=rt.valid_us,
            refit_us=int(ctx.refit.timestamp() * 1e6),
            floor_us=int(arms707.LONG_WINDOW_FLOOR.timestamp() * 1e6),
            fold_yyyymm=int(ctx.month.replace("-", "")),
        )
        placebo = tensor(weights.exponential(age=permuted, half_life_years=ctx.h_t))
        return lambda _epoch: placebo
    assert rule.schedule is not None
    cache: dict[float, torch.Tensor] = {}
    schedule = rule.schedule

    def scheduled(epoch: int) -> torch.Tensor:
        h = weights.schedule_half_life(schedule=schedule, epoch=epoch)
        if h not in cache:
            cache.clear()
            cache[h] = tensor(weights.exponential(age=rt.age, half_life_years=h))
        return cache[h]

    return scheduled


def m2_step_cap(*, ctx: Context) -> int:
    """Return the step cap of `M2`: 40 epochs on window F plus 40 on the last 730 days.

    Args:
        ctx: The group; its window-F rows give the two training-row counts.

    Returns:
        The number of optimiser steps.
    """
    rt = tensors_for(ctx=ctx, which="F")
    train_f = int((~rt.is_val_np).sum())
    train_recent = int((~rt.is_val_np & weights.within_window(age=rt.age, window_days=730)).sum())
    per_epoch = math.ceil(train_f / net.BATCH) + math.ceil(train_recent / net.BATCH)
    return arms.MAX_EPOCHS * per_epoch


def run_stage(
    *, plan: arms.StagePlan, model: torch.nn.Module, ctx: Context
) -> tuple[net.StageResult, dict]:
    """Train one stage in place and return its result and the extra fields for its log.

    Args:
        plan: The stage.
        model: The network, on the device.
        ctx: The group.

    Returns:
        The result and a description of the stage's rows.
    """
    rt = tensors_for(ctx=ctx, which="era" if plan.rows == "era" else "F")
    train, held = stage_masks(plan=plan, rt=rt)
    device = ctx.device
    data = net.StageData(
        x=rt.x,
        z=rt.z,
        primary=rt.primary,
        train_idx=torch.from_numpy(np.flatnonzero(train)).to(device),
        val_idx=torch.from_numpy(np.flatnonzero(held)).to(device),
        n_primaries=len(ctx.month_scale.scale.target_mean),
        shifted=rt.shifted if ctx.world is not None else None,
    )
    config = net.StageConfig(
        max_epochs=plan.max_epochs,
        patience=plan.patience,
        select_from=plan.select_from,
        select=plan.select,
        fixed_epochs=plan.fixed_epochs,
        sampler=plan.mechanism == "sampler",
        ema_decay=plan.ema_decay,
        step_cap=m2_step_cap(ctx=ctx) if plan.step_cap == "m2" else None,
        lr_rule=plan.lr_rule,
    )
    provider = make_provider(plan=plan, rt=rt, ctx=ctx, train=train)
    result = net.train_stage(
        model=model,
        setting=replace(net707.G09, lr=plan.lr),
        data=data,
        config=config,
        weight_provider=provider,
        seed=ctx.seed,
    )
    n_primaries = len(ctx.month_scale.scale.target_mean)
    extra = {
        "stage": plan.name,
        "rows": plan.rows,
        "train_rows": int(train.sum()),
        "selection_rows": int(held.sum()),
        "step_cap": config.step_cap,
        "max_epochs": plan.max_epochs,
        "weight_rule": plan.weight.kind,
        "mechanism": plan.mechanism,
        "shifted_share": inject.injected_share(
            shifted=rt.shifted_np[train],
            primary=rt.primary_np[train],
            weight=None,
            n_primaries=n_primaries,
        )
        if ctx.world is not None
        else None,
        "world_events": None if ctx.world is None else ctx.world.n_events,
    }
    return result, extra


def check_shift_share(*, extra: dict, ctx: Context, label: str) -> None:
    """Stop the run if the share of rows with a shifted lag lies outside the pilot's range.

    A share near 1 means that a lag-source mismatch marks every row as shifted, and a share near 0
    means that the world's events never reach a lag. Only the `i` state is checked.

    Args:
        extra: The stage's extra log, whose `shifted_share` holds the unweighted share.
        ctx: The group, whose `shift_share_range` is the allowed range or `None`.
        label: The arm, state, and stage for the message.

    Raises:
        SystemExit: If the unweighted share lies outside the range.
    """
    if ctx.shift_share_range is None or extra["shifted_share"] is None:
        return
    share = float(extra["shifted_share"]["unweighted"])
    low, high = ctx.shift_share_range
    if not low <= share <= high:
        print(
            f"STOP: {label} {ctx.month} s{ctx.seed}: the share of rows with a shifted lag is "
            f"{share:.4f}, outside the pre-registered range [{low}, {high}]",
            flush=True,
        )
        raise SystemExit(2)


def new_seeded_model(*, ctx: Context) -> torch.nn.Module:
    """Return an untrained network seeded as experiment 707 seeds a stage, on the device."""
    torch.manual_seed(ctx.seed)
    return net.new_model(n_primaries=len(ctx.month_scale.scale.target_mean)).to(ctx.device)


def run_chain(
    *, arm: arms.Arm, ctx: Context
) -> tuple[torch.nn.Module, net.StageResult | None, dict]:
    """Run an arm's stages and return the final network, the last stage's result, and the logs.

    A stage with a `state_key` loads its stored state (experiment 707's for the `n` state of a
    scored fold, or this experiment's) if there is one, and otherwise trains and stores it.

    Args:
        arm: The arm.
        ctx: The group.

    Returns:
        The final network, the result of the final stage (`None` if it was loaded from a state),
        and a dictionary of stage logs.
    """
    model: torch.nn.Module | None = None
    result: net.StageResult | None = None
    stages: dict[str, dict] = {}
    for plan in arm.stages:
        stored = None
        own: Path | None = None
        locked = False
        if plan.state_key is not None:
            stored = (
                stored_707_state(
                    key=plan.state_key, state=ctx.state, month=ctx.month, seed=ctx.seed
                )
                if ctx.use_stored
                else None
            )
            own = state_path(
                out_dir=ctx.out_dir,
                key=plan.state_key,
                state=ctx.state,
                month=ctx.month,
                seed=ctx.seed,
            )
            if stored is None:
                stored = claim_state(own=own)
                locked = stored is None
        if plan.init == "previous" and model is not None:
            fresh = model
        else:
            fresh = new_seeded_model(ctx=ctx)
        if stored is not None:
            fresh.load_state_dict(torch.load(stored, map_location=ctx.device))
            fresh.eval()
            model, result = fresh, None
            stages[plan.name] = {"loaded_from": str(stored)}
            continue
        try:
            result, extra = run_stage(plan=plan, model=fresh, ctx=ctx)
            check_shift_share(extra=extra, ctx=ctx, label=f"{arm.name}-{ctx.state} {plan.name}")
            model = fresh
            stages[plan.name] = json.loads(net.stage_log_json(result=result, extra=extra))
            print(
                f"  {arm.name}-{ctx.state} {plan.name} {ctx.month} s{ctx.seed}: "
                f"{result.epochs} epochs, {result.steps} steps, val {result.best_val:.4f} "
                f"(start {result.initial_val:.4f}), "
                f"{result.seconds:.0f} s, stopped by {result.stopped_by}",
                flush=True,
            )
            if own is not None:
                save_state(state=model.state_dict(), target=own)
        finally:
            if locked and own is not None:
                release_state(own=own)
    assert model is not None
    return model, result, stages


@cache
def clean_stack() -> tuple[list[str], np.ndarray]:
    """Return the Primaries' site labels and their observed power stacked, Primary by grid."""
    power = inject.observed_power()
    sites = list(power)
    return sites, np.stack([power[s] for s in sites])


@cache
def bank_powers(*, month: str) -> dict[int, dict[str, np.ndarray]]:
    """Return the power of each bank pass that a scored fold forecasts, by pass number.

    Fold number `k` of the 12 scored folds uses passes `k` and `k + 12` of experiment 581's bank,
    and pass 100 (the verified real events removed from the observed power) if a real event
    overlaps the month or the 7 days before it.

    Args:
        month: A scored fold.

    Returns:
        The power by site label of each pass.
    """
    k = arms.SCORED_MONTHS.index(month)
    sites, stack = clean_stack()
    start = int(common.grid_index(times=prep.refit_of(month=month)))  # ty: ignore[invalid-argument-type]
    out = {}
    for r in (k, k + len(bank581.COMBOS)):
        shifted, _ = bank581.contaminate(clean=stack, month_start_idx=start, r=r)
        out[r] = {s: shifted[i] for i, s in enumerate(sites)}
    month_end = int(
        common.grid_index(  # ty: ignore[invalid-argument-type]
            times=(prep.refit_of(month=month) + timedelta(days=32)).replace(day=1)
        )
    )
    lo = start - EVENT_MARGIN_DAYS * bank581.PER_DAY
    events = [
        e for e in inject.real_events_table() if e["end_idx"] > lo and e["onset_idx"] < month_end
    ]
    if events:
        corrected = bank581.corrected_clean(clean=stack, events=events)
        out[REAL_PASS] = {s: corrected[i] for i, s in enumerate(sites)}
    return out


def write_bank(
    *,
    fitted: net707.Fitted,
    scale: net707.Scale,
    job: arms.Job,
    out_dir: Path,
    device: str,
    overwrite: bool,
) -> dict:
    """Forecast every bank pass of a scored fold with a network and store each; return the timings.

    Args:
        fitted: The network.
        scale: Its scale.
        job: The job (arm, state, fold, seed).
        out_dir: The output folder.
        device: The torch device.
        overwrite: Whether to replace a pass that exists. A freshly fitted network sets it, so that
            no pass of an earlier, killed fit of the same job survives; a stored network (fixed
            weights) leaves it off, so that a resume skips the finished passes.

    Returns:
        Seconds per pass.
    """
    seconds = {}
    for pass_id, power in bank_powers(month=job.month).items():
        target = bank_path(
            out_dir=out_dir,
            arm=job.arm,
            state=job.state,
            month=job.month,
            seed=job.seed,
            pass_id=pass_id,
        )
        if target.exists() and not overwrite:
            continue
        started = time.monotonic()
        frame = net.forecast_month(
            fitted=fitted, scale=scale, month=job.month, device=device, power=power
        )
        common.atomic_write(frame=frame, target=target)
        seconds[pass_id] = round(time.monotonic() - started, 1)
    return seconds


def forecast_job(
    *,
    fitted: net707.Fitted,
    scale: net707.Scale,
    job: arms.Job,
    out_dir: Path,
    device: str,
    bank: bool,
) -> tuple[dict, pl.DataFrame]:
    """Return the ordinary forecast of a job and write its bank passes (scored folds only).

    Args:
        fitted: The network.
        scale: Its scale.
        job: The job.
        out_dir: The output folder.
        device: The torch device.
        bank: Whether to write the bank passes.

    Returns:
        The bank timings and the forecast frame (not yet written).
    """
    timings: dict = {}
    if job.month in arms.TUNING_MONTHS:
        return timings, net.forecast_tuning(
            fitted=fitted, scale=scale, month=job.month, device=device
        )
    frame = net.forecast_month(fitted=fitted, scale=scale, month=job.month, device=device)
    if bank:
        timings = write_bank(
            fitted=fitted, scale=scale, job=job, out_dir=out_dir, device=device, overwrite=True
        )
    return timings, frame


def fit_job(*, arm: arms.Arm, ctx: Context, bank: bool) -> None:
    """Fit one arm in a group, forecast, and write the model, log, and forecast file (last).

    Args:
        arm: The arm.
        ctx: The group.
        bank: Whether to forecast the bank passes.
    """
    job = arms.Job(arm=arm.name, state=ctx.state, month=ctx.month, seed=ctx.seed)
    started = time.monotonic()
    model, result, stages = run_chain(arm=arm, ctx=ctx)
    setting = replace(net707.G09, lr=arm.stages[-1].lr)
    fitted = net707.Fitted(
        model=model,
        setting=setting,
        seed=ctx.seed,
        epochs=0 if result is None else result.epochs,
        val_nll=float("nan") if result is None else result.best_val,
    )
    scale = ctx.month_scale.scale
    net707.save_fit(
        fitted=fitted,
        scale=scale,
        target=ctx.out_dir / "models" / f"{arm.name}-{ctx.state}" / ctx.month / f"s{ctx.seed}",
    )
    timings, frame = forecast_job(
        fitted=fitted, scale=scale, job=job, out_dir=ctx.out_dir, device=ctx.device, bank=bank
    )
    seconds = round(time.monotonic() - started)
    write_json(
        payload={
            "arm": arm.name,
            "state": ctx.state,
            "month": ctx.month,
            "seed": ctx.seed,
            "stages": stages,
            "world_seed": None if ctx.world is None else ctx.world.world_seed,
            "bank_seconds": timings,
            "seconds_total": seconds,
            "peak_rss_mb": peak_rss_mb(),
            "peak_gpu_mb": peak_gpu_mb(),
            "h_t": ctx.h_t,
            "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        },
        target=ctx.out_dir / "logs" / f"{arm.name}-{ctx.state}" / f"{ctx.month}_s{ctx.seed}.json",
    )
    common.atomic_write(
        frame=frame,
        target=forecast_path(
            out_dir=ctx.out_dir, arm=arm.name, state=ctx.state, month=ctx.month, seed=ctx.seed
        ),
    )
    print(f"{arm.name}-{ctx.state} {ctx.month} seed {ctx.seed}: done in {seconds} s", flush=True)


def peak_rss_mb() -> float:
    """Return this process's peak resident memory in megabytes."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def peak_gpu_mb() -> float:
    """Return the peak GPU memory in megabytes, or 0 without a GPU."""
    return torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0


def lag_times(*, valid_us: int, origin_us: int) -> list[datetime]:
    """Return the times of the three origin lags of one row.

    Args:
        valid_us: The valid time in microseconds since the epoch.
        origin_us: The origin in microseconds since the epoch.

    Returns:
        The times of the newest, day, and week lags.
    """
    start = int(common.GRID_START.timestamp() * 1e6)
    target = (valid_us - start) // 1_800_000_000
    newest = (origin_us - start) // 1_800_000_000 - common.NEWEST_AGE
    gap = target - newest
    per_day = common.STEPS_PER_DAY
    idx = [
        newest,
        target - per_day * math.ceil(gap / per_day),
        target - 7 * per_day * math.ceil(gap / (7 * per_day)),
    ]
    return [common.GRID_START + i * common.STEP for i in idx]


def print_example(*, ctx: Context) -> None:
    """Print one training row and one scored row with their weather and lag times.

    The print gives the Primary label, the run's initialisation time, the origin, the valid time,
    the lead, the three lag times, the row's age at the refit, and its weight under each rule.
    The run time is the origin minus the publication delay of 9 hours by construction. The check
    that a row's stored run is issued at least 9 hours before its origin is
    `prep.assert_origin_after_run`, which `load_site` and `run_inputs` call.

    Args:
        ctx: A group whose window-F rows can be built.
    """
    rt = tensors_for(ctx=ctx, which="F")
    i = int(np.argmin(np.abs(rt.lead - 128.5) + np.where(rt.is_val_np, 1e9, 0.0)))
    origin_us, valid_us = int(rt.origin_us[i]), int(rt.valid_us[i])
    run = datetime.fromtimestamp(origin_us / 1e6, tz=UTC) - common.PUBLICATION_DELAY
    label = prep.primaries()["site"][int(rt.primary_np[i])]
    age = rt.age[i : i + 1]
    permuted = weights.permuted_age_days(
        valid_us=rt.valid_us,
        refit_us=int(ctx.refit.timestamp() * 1e6),
        floor_us=int(arms707.LONG_WINDOW_FLOOR.timestamp() * 1e6),
        fold_yyyymm=int(ctx.month.replace("-", "")),
    )[i : i + 1]
    floor_weight = weights.floor_exponential(age=age, half_life_years=0.5, floor=0.1)
    print(f"\n== a training row of fold {ctx.month} (window F) ==")
    print(
        f"Primary {label}; run initialised {run:%F %H:%M} UTC; origin "
        f"{datetime.fromtimestamp(origin_us / 1e6, tz=UTC):%F %H:%M}; valid time "
        f"{datetime.fromtimestamp(valid_us / 1e6, tz=UTC):%F %H:%M}; lead {rt.lead[i]:.1f} h; "
        f"age at the refit {float(age[0]):.1f} days"
    )
    print(
        "lag times:", [f"{t:%F %H:%M}" for t in lag_times(valid_us=valid_us, origin_us=origin_us)]
    )
    print(
        "weights: "
        + ", ".join(
            f"h{h:g}={float(weights.exponential(age=age, half_life_years=h)[0]):.4f}"
            for h in (0.5, 1.0, 2.0)
        )
        + f", floor={float(floor_weight[0]):.4f}"
        + f", HW730 keeps={bool(weights.within_window(age=age, window_days=730)[0])}"
        + f", HW365 keeps={bool(weights.within_window(age=age, window_days=365)[0])}"
        + f", placebo(h={ctx.h_t:g}) age {float(permuted[0]):.1f} days -> "
        f"{float(weights.exponential(age=permuted, half_life_years=ctx.h_t)[0]):.4f}"
    )
    origin = prep.test_origins(month=ctx.month)[0] if ctx.month in arms.SCORED_MONTHS else None
    if origin is not None:
        run_scored = origin - common.PUBLICATION_DELAY
        print(
            f"== a scored row of fold {ctx.month} ==\nthe first scored origin is {origin:%F %H:%M} "
            f"UTC and reads the 00Z run of {run_scored:%F %H:%M} UTC (issued 9 hours earlier); "
            f"its lags are power readings at least 3.5 hours before the origin"
        )


def group_jobs(
    *, jobs: list[arms.Job], h_t: float, b_star: str, o_star: str | None
) -> list[tuple[int, tuple[str, int, str], list[str]]]:
    """Return the jobs grouped by (tier, seed, fold, state), with the arms in plan order.

    A group holds the arms of one priority tier only, so the groups run in the order of the
    tiers: seed 0 of tier 1 covers all 12 folds before seed 1 of tier 1 starts.

    Args:
        jobs: The jobs.
        h_t: The half-life in years of `W-h_T`.
        b_star: The ENS-only recipe `B*`.
        o_star: The one-shot recipe `O*`, or `None`.

    Returns:
        Each group's tier, its (fold, seed, state) key, and its arm names, sorted by tier, seed,
        fold, and state.
    """
    grouped: dict[tuple[int, int, str, str], list[str]] = defaultdict(list)
    for job in jobs:
        tier = arms.job_tier(job=job, h_t=h_t, b_star=b_star, o_star=o_star)
        grouped[tier, job.seed, job.month, job.state].append(job.arm)
    return [
        (tier, (month, seed, state), list(dict.fromkeys(names)))
        for (tier, seed, month, state), names in sorted(grouped.items())
    ]


def measured_seconds(*, out_dir: Path) -> dict[tuple[str, str], float]:
    """Return the mean seconds per fit of each (arm, state) from the finished fits' logs.

    Args:
        out_dir: The output folder.

    Returns:
        The mean `seconds_total` by (arm, state); empty if there is no log.
    """
    totals: dict[tuple[str, str], list[float]] = defaultdict(list)
    for path in (out_dir / "logs").glob("*-*/*.json"):
        log = json.loads(path.read_text())
        totals[log["arm"], log["state"]].append(float(log["seconds_total"]))
    return {key: float(np.mean(values)) for key, values in totals.items()}


def hours_report(
    *,
    jobs: list[arms.Job],
    out_dir: Path,
    args: argparse.Namespace,
) -> None:
    """Print cumulative planned and finished GPU-hours per tier, and the contrasts at a budget.

    The seconds of a job are the mean of the logged fits of the same arm and state, else the mean
    of every logged fit, else `--assume-seconds` (60 s for `M3`, which only forecasts).
    The planned hours of a tier are the estimates of all its jobs. The finished hours of a tier
    are the estimates of the jobs whose forecast file exists. The last block walks the unfinished
    jobs in plan order up to `--budget-hours` and says, for each tested contrast, on how many folds
    both arms would hold each seed by then.

    Args:
        jobs: The jobs in plan order.
        out_dir: The output folder.
        args: The parsed arguments (`h_t`, `b_star`, `o_star`, `assume_seconds`, `budget_hours`).
    """
    measured = measured_seconds(out_dir=out_dir)
    overall = float(np.mean(list(measured.values()))) if measured else None

    def seconds(job: arms.Job) -> float:
        if (job.arm, job.state) in measured:
            return measured[job.arm, job.state]
        if arms.ARMS[job.arm].free_from is not None:
            return 60.0
        return overall if overall is not None else float(args.assume_seconds)

    def done(job: arms.Job) -> bool:
        return forecast_path(
            out_dir=out_dir, arm=job.arm, state=job.state, month=job.month, seed=job.seed
        ).exists()

    source = "measured from logs" if measured else f"assumed {args.assume_seconds:g} s per fit"
    print(f"\nGPU-hours per tier ({source}); cumulative over tiers")
    cumulative_planned = cumulative_finished = 0.0
    tiers = sorted(
        {arms.job_tier(job=j, h_t=args.h_t, b_star=args.b_star, o_star=args.o_star) for j in jobs}
    )
    for tier in tiers:
        part = [
            j
            for j in jobs
            if arms.job_tier(job=j, h_t=args.h_t, b_star=args.b_star, o_star=args.o_star) == tier
        ]
        planned = sum(seconds(j) for j in part) / 3600.0
        finished = sum(seconds(j) for j in part if done(j)) / 3600.0
        cumulative_planned += planned
        cumulative_finished += finished
        print(
            f"tier {tier}: {len(part)} jobs ({sum(done(j) for j in part)} finished); planned "
            f"{planned:.1f} h, finished {finished:.1f} h; cumulative planned "
            f"{cumulative_planned:.1f} h, cumulative finished {cumulative_finished:.1f} h"
        )
    target = scale_seconds_path(out_dir=out_dir)
    built = json.loads(target.read_text()) if target.exists() else {}
    unbuilt = {j.month for j in jobs if not done(j)} - set(built)
    mean_build = float(np.mean(list(built.values()))) if built else float(args.assume_seconds) / 6
    print(
        f"fold scale builds are outside the fit seconds: {len(unbuilt)} still to build at "
        f"{mean_build:.0f} s each = {len(unbuilt) * mean_build / 3600.0:.2f} h on top of the above"
    )
    held: set[tuple[str, str, str, int]] = {
        (j.arm, j.state, j.month, j.seed) for j in jobs if done(j)
    }
    spent = 0.0
    for job in jobs:
        if (job.arm, job.state, job.month, job.seed) in held:
            continue
        if spent + seconds(job) / 3600.0 > args.budget_hours:
            break
        spent += seconds(job) / 3600.0
        held.add((job.arm, job.state, job.month, job.seed))
    print(
        f"\nat {args.budget_hours:g} GPU-hours of new fits ({spent:.1f} h planned), "
        f"folds held by both arms"
    )

    def holds(arm: str, state: str, month: str, seed: int) -> bool:
        if state == "n" and not arms.needs_new_n_fit(arm):
            return True
        return (arm, state, month, seed) in held

    def resolve(name: str) -> str:
        return {"B*": args.b_star, "O*": args.o_star or "O*", "W-h_T": f"W-h{args.h_t:g}"}.get(
            name, name
        )

    pairs = {
        label: (resolve(t), "i", resolve(r), "i") for label, (t, r) in arms.CONTRAST_NAMES.items()
    }
    pairs |= {f"I{n}": (x, "i", x, "n") for n, x in enumerate(("M1", "M2", args.b_star), start=1)}
    seeds = sorted({j.seed for j in jobs})
    for label, (ta, ts, ra, rs) in pairs.items():
        counts = {
            seed: sum(holds(ta, ts, m, seed) and holds(ra, rs, m, seed) for m in arms.SCORED_MONTHS)
            for seed in seeds
        }
        text = ", ".join(f"seed {s}: {c}/{len(arms.SCORED_MONTHS)}" for s, c in counts.items())
        print(f"{label:3s} {ta}-{ts} minus {ra}-{rs}: {text}")


def read_pp_half_lives(*, path: Path | None) -> np.ndarray | None:
    """Return the learned per-Primary half-lives in years from a JSON file keyed by site label.

    Args:
        path: A JSON file such as `{"D2": 1.0, "D3": "inf"}`, or `None`.

    Returns:
        The half-lives indexed by Primary, or `None` if no file is given.
    """
    if path is None:
        return None
    values = json.loads(path.read_text())
    sites = prep.primaries()["site"].to_list()
    return np.array([float(values[s]) for s in sites], dtype=np.float64)


def scale_seconds_path(*, out_dir: Path) -> Path:
    """Return the file that holds each fold's measured scale-build seconds."""
    return out_dir / "logs" / "month_scale_seconds.json"


def record_scale_seconds(*, out_dir: Path, month: str, seconds: float) -> None:
    """Store the seconds that building a fold's scale took, so that `--gpu-hours` can count them.

    Args:
        out_dir: The output folder.
        month: The fold.
        seconds: The measured seconds.
    """
    target = scale_seconds_path(out_dir=out_dir)
    known = json.loads(target.read_text()) if target.exists() else {}
    known[month] = seconds
    write_json(payload=known, target=target)


def run_group(
    *,
    key: tuple[str, int, str],
    arm_names: list[str],
    scales: dict[str, MonthScale],
    args: argparse.Namespace,
    out_dir: Path,
    show_example: bool,
) -> bool:
    """Fit every pending job of one (fold, seed, state) group.

    Args:
        key: The fold, the seed, and the state.
        arm_names: The arms to fit.
        scales: The cache of per-fold scales. A scale holds only the standardiser, the target
            statistics, the spread table, and one integer, so every fold stays cached.
        args: The command-line arguments.
        out_dir: The output folder.
        show_example: Whether to print the origin-time weather example (the pilot always does).

    Returns:
        Whether the group had pending work, so that the caller prints the example only once.

    Raises:
        SystemExit: If a stored placebo forecast was fitted with another `h_T`.
    """
    month, seed, state = key
    if "P" in arm_names and state == "i":
        check_placebo_h_t(out_dir=out_dir, month=month, seed=seed, h_t=args.h_t)
    pending = [
        a
        for a in arm_names
        if not forecast_path(out_dir=out_dir, arm=a, state=state, month=month, seed=seed).exists()
    ]
    if "PP" in pending and args.pp_half_lives is None:
        print(
            f"{month} s{seed} {state}: skipping PP, which needs --pp-half-lives (write the file "
            "with e00710_score.py --tuning --write-pp-half-lives, then run --arms PP)",
            flush=True,
        )
        pending.remove("PP")
    if not pending:
        print(f"{month} s{seed} {state}: all {len(arm_names)} jobs done", flush=True)
        return False
    if month not in scales:
        scale_started = time.monotonic()
        scales[month] = month_scale(month=month)
        scale_seconds = round(time.monotonic() - scale_started, 1)
        record_scale_seconds(out_dir=out_dir, month=month, seconds=scale_seconds)
        print(f"{month}: fold scale built in {scale_seconds:.0f} s (kept for every later group)")
    world = inject.build_world(month=month, seed=seed) if state == "i" else None
    ctx = Context(
        month=month,
        seed=seed,
        state=state,  # ty: ignore[invalid-argument-type]
        refit=prep.refit_of(month=month),
        month_scale=scales[month],
        world=world,
        device=args.device,
        h_t=args.h_t,
        pp_half_lives=read_pp_half_lives(path=args.pp_half_lives),
        out_dir=out_dir,
        use_stored=not args.no_stored_states,
        shift_share_range=None
        if args.shift_share_range is None
        else (args.shift_share_range[0], args.shift_share_range[1]),
    )
    if world is not None:
        print(
            f"{month} s{seed}: world seed {world.world_seed}, {world.n_events} events",
            flush=True,
        )
    if args.pilot or show_example:
        print_example(ctx=ctx)
    for name in pending:
        fit_job(arm=arms.ARMS[name], ctx=ctx, bank=not args.no_bank)
        torch.cuda.empty_cache()
    ctx.tensors = None
    torch.cuda.empty_cache()
    return True


def check_placebo_h_t(*, out_dir: Path, month: str, seed: int, h_t: float) -> None:
    """Refuse to reuse a stored placebo forecast that was fitted with another `h_T`.

    Args:
        out_dir: The output folder.
        month: The fold.
        seed: The network seed.
        h_t: The half-life in years of this run.

    Raises:
        SystemExit: If the placebo's log records another half-life.
    """
    log = out_dir / "logs" / "P-i" / f"{month}_s{seed}.json"
    if not log.exists():
        return
    stored = json.loads(log.read_text())["h_t"]
    if abs(float(stored) - h_t) > 1e-9:
        msg = (
            f"STOP: the stored placebo P-i {month} s{seed} used h_T={stored}, this run has "
            f"h_T={h_t}; move the P-i files away or pass the same --h-t"
        )
        raise SystemExit(msg)


def run_stored(
    *, name: str, months: list[str], seeds: list[int], device: str, out_dir: Path
) -> None:
    """Forecast from stored networks of experiment 707: the bank passes, and `M3-n`'s forecast.

    Args:
        name: `M1-n`, `M2-n`, or `M3-n`.
        months: The scored folds.
        seeds: The stored seeds.
        device: The torch device.
        out_dir: The output folder.
    """
    scale_folder, state_folder = STORED_ARMS[name]
    arm_name, state = name.split("-")
    for month, seed in ((m, s) for m in months for s in seeds):
        model_dir = STORED_707_WORK / "models" / scale_folder / month / f"s{seed}"
        scale = net707.load_scale(folder=model_dir)
        fitted = net707.load_fitted(folder=model_dir, device=device)
        if state_folder is not None:
            fitted.model.load_state_dict(
                torch.load(
                    STORED_707_WORK / "pretrained" / state_folder / f"{month}_s{seed}.pt",
                    map_location=device,
                )
            )
            fitted.model.eval()
        job = arms.Job(arm=arm_name, state="n", month=month, seed=seed)
        write_bank(
            fitted=fitted, scale=scale, job=job, out_dir=out_dir, device=device, overwrite=False
        )
        target = forecast_path(out_dir=out_dir, arm=arm_name, state=state, month=month, seed=seed)
        if state_folder is not None and not target.exists():
            frame = net.forecast_month(fitted=fitted, scale=scale, month=month, device=device)
            common.atomic_write(frame=frame, target=target)
        print(f"stored {name} {month} seed {seed}: bank written", flush=True)


def code_hash() -> str:
    """Return the SHA-256 of this experiment's scripts, which the known-answer marker records."""
    folder = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted([*folder.glob("e00710_*.py"), *folder.glob("run_phase.sh")]):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def refuse_without_marker(*, allowed: bool) -> None:
    """Exit with status 1 unless all seven known-answer tests passed on the current code.

    Args:
        allowed: Whether `--no-marker` was passed.
    """
    if allowed:
        return
    passed = json.loads(MARKER.read_text()) if MARKER.exists() else {}
    missing = [n for n in range(1, 8) if str(n) not in passed]
    if passed.get("code_sha256") != code_hash():
        missing = list(range(1, 8))
    if missing:
        print(
            f"refusing to fit: known-answer tests {missing} have not passed on this code "
            f"(marker {MARKER}); run `e00710_known_answer.py` or pass --no-marker"
        )
        raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--arms", nargs="+", choices=list(arms.ARMS), default=None)
    parser.add_argument("--state", choices=["i", "n", "both"], default="i")
    parser.add_argument("--months", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--phase", choices=["T", "S", "C", "SC"], default=None)
    parser.add_argument(
        "--tier",
        nargs="+",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="priority tiers to run (default all): 1 headline contrasts, 2 the rest of the i "
        "state, 3 every n state",
    )
    parser.add_argument("--minimal", action="store_true", help="the minimal plan of the phase")
    parser.add_argument(
        "--finalists",
        nargs="*",
        choices=[a.name for a in arms.ARMS.values() if a.scored == "finalist"],
        default=None,
        help="creative arms of Phase S (default for Phases S, C, SC: the tuning choice)",
    )
    parser.add_argument(
        "--b-star",
        default=None,
        help="the ENS-only recipe B* (default for Phases S, C, SC: the tuning choice)",
    )
    parser.add_argument(
        "--o-star",
        default=None,
        help="the one-shot recipe O* (default for Phases S, C, SC: the tuning choice)",
    )
    parser.add_argument(
        "--h-t",
        type=float,
        default=None,
        help="half-life in years of W-h_T and P (default for Phases S, C, SC: the tuning choice)",
    )
    parser.add_argument(
        "--shift-share-range",
        nargs=2,
        type=float,
        metavar=("LOW", "HIGH"),
        default=None,
        help="stop if the share of rows with a shifted lag leaves this range (set after the pilot)",
    )
    parser.add_argument(
        "--gpu-hours", action="store_true", help="print planned and finished GPU-hours per tier"
    )
    parser.add_argument("--budget-hours", type=float, default=10.0)
    parser.add_argument("--assume-seconds", type=float, default=1800.0)
    parser.add_argument("--pp-half-lives", type=Path, default=None, help="JSON of PP's half-lives")
    parser.add_argument("--stored", choices=list(STORED_ARMS), default=None)
    parser.add_argument("--pilot", action="store_true", help="one fold, timing, output under pilot")
    parser.add_argument("--no-bank", action="store_true")
    parser.add_argument(
        "--no-stored-states", action="store_true", help="retrain stages that 707 stored"
    )
    parser.add_argument("--no-marker", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def plan_from_args(*, args: argparse.Namespace, out_dir: Path) -> list[arms.Job]:
    """Return the jobs that the command-line arguments name.

    Args:
        args: The parsed arguments.
        out_dir: The output folder (unused for planning; kept so the pilot reads one place).

    Returns:
        The jobs.

    Raises:
        SystemExit: If the arguments name no way to choose jobs.
    """
    del out_dir
    if args.pilot:
        names = args.arms or ["M1", "W-h0.5"]
        months, seeds = args.months or ["2026-01"], args.seeds or [0]
        return [arms.Job(a, "i", m, s) for m in months for s in seeds for a in names]
    if args.phase:
        jobs = arms.plan_jobs(
            include_pp=args.pp_half_lives is not None,
            phase=args.phase,
            minimal=args.minimal,
            h_t=args.h_t,
            finalists=tuple(args.finalists),
            b_star=args.b_star,
            o_star=args.o_star,
        )
        keep_arms, keep_months, keep_seeds, keep_tiers = (
            args.arms,
            args.months,
            args.seeds,
            args.tier,
        )
        return [
            j
            for j in jobs
            if (not keep_arms or j.arm in keep_arms)
            and (not keep_months or j.month in keep_months)
            and (not keep_seeds or j.seed in keep_seeds)
            and (
                not keep_tiers
                or arms.job_tier(job=j, h_t=args.h_t, b_star=args.b_star, o_star=args.o_star)
                in keep_tiers
            )
        ]
    if not args.arms or not args.months:
        msg = "give --phase, or --arms and --months, or --stored, or --pilot"
        raise SystemExit(msg)
    states = ("i", "n") if args.state == "both" else (args.state,)
    return [
        arms.Job(a, s, m, seed)
        for m in args.months
        for seed in (args.seeds or [0])
        for s in states
        for a in args.arms
    ]


def require_stop_checks() -> None:
    """Refuse Phases S, C, and SC unless both tuning stop checks exist and passed.

    Raises:
        SystemExit: If `work/stop_checks.json` is missing, or a check is missing or failed.
    """
    if not arms.STOP_CHECKS_FILE.exists():
        msg = f"{arms.STOP_CHECKS_FILE} is missing: run e00710_score.py --tuning first"
        raise SystemExit(msg)
    checks = json.loads(arms.STOP_CHECKS_FILE.read_text())
    failed = [name for name, check in checks.items() if not check["passed"]]
    if len(checks) < 2 or failed:
        msg = f"the tuning stop checks did not all pass: {failed or 'a check is missing'}"
        raise SystemExit(msg)


def resolve_choices(*, args: argparse.Namespace) -> None:
    """Fill `b_star`, `o_star`, `h_t`, and `finalists` from the tuning choice, or Phase T values.

    Phases S, C, and SC take every value the operator did not give from the file that the tuning
    scorer wrote, and stop if there is no file. Phase T and the other modes use `M2`, no `O*`, and
    0.5 years.

    Args:
        args: The parsed arguments, changed in place.

    Raises:
        SystemExit: If a value is missing and the tuning scorer has not written its choice.
    """
    if args.phase in ("S", "C", "SC") and not args.pilot:
        if not (args.dry_run or args.gpu_hours):
            require_stop_checks()
        choices = (
            arms.read_choices()
            if None in (args.b_star, args.o_star, args.h_t, args.finalists)
            else {}
        )
        args.b_star = args.b_star or choices["b_star"]
        args.o_star = args.o_star or choices["o_star"]
        args.h_t = choices["h_t"] if args.h_t is None else args.h_t
        args.finalists = choices["finalists"] if args.finalists is None else args.finalists
        print(
            f"choices: B*={args.b_star} O*={args.o_star} h_T={args.h_t:g} "
            f"finalists={args.finalists}",
            flush=True,
        )
        if "PP" in args.finalists and args.pp_half_lives is None:
            msg = "the finalist PP needs --pp-half-lives"
            raise SystemExit(msg)
        return
    args.b_star = args.b_star or "M2"
    args.h_t = 0.5 if args.h_t is None else args.h_t
    args.finalists = args.finalists or []


def main() -> None:
    """Plan the jobs from the arguments and run the groups."""
    parser = build_parser()
    args = parser.parse_args()
    resolve_choices(args=args)
    out_dir = PILOT_DIR if args.pilot else OUT_DIR
    if args.stored:
        run_stored(
            name=args.stored,
            months=args.months or list(arms.SCORED_MONTHS),
            seeds=args.seeds or list(arms.STORED_SEEDS),
            device=args.device,
            out_dir=out_dir,
        )
        return
    jobs = plan_from_args(args=args, out_dir=out_dir)
    groups = group_jobs(jobs=jobs, h_t=args.h_t, b_star=args.b_star, o_star=args.o_star)
    pending = sum(
        not forecast_path(
            out_dir=out_dir, arm=j.arm, state=j.state, month=j.month, seed=j.seed
        ).exists()
        for j in jobs
    )
    print(f"{len(jobs)} jobs in {len(groups)} groups; {pending} not yet fitted; out {out_dir}")
    if args.gpu_hours:
        hours_report(jobs=jobs, out_dir=out_dir, args=args)
    if args.dry_run:
        for tier, key, names in groups:
            print(f"  tier {tier} seed {key[1]} fold {key[0]} state {key[2]}: {names}")
        return
    if args.gpu_hours:
        return
    if not args.pilot:
        refuse_without_marker(allowed=args.no_marker)
        if args.shift_share_range is None:
            parser.error("--shift-share-range LOW HIGH is required outside the pilot")
    started = time.monotonic()
    scales: dict[str, MonthScale] = {}
    shown = False
    for _tier, key, names in groups:
        worked = run_group(
            key=key,
            arm_names=names,
            scales=scales,
            args=args,
            out_dir=out_dir,
            show_example=not shown,
        )
        shown = shown or worked
    if args.pilot:
        print(
            f"\npilot: wall-clock {time.monotonic() - started:.0f} s, peak host memory "
            f"{peak_rss_mb() / 1024:.1f} GB, peak GPU memory {peak_gpu_mb() / 1024:.1f} GB"
        )


if __name__ == "__main__":
    main()
