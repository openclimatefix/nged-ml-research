r"""Fit one (arm, fold, seed) job of experiment 707 and write its raw forecasts.

Every arm is experiment 425's pooled plain network (setting `g09`, an 8-wide Primary embedding)
trained through up to three stages (see `e00707_arms`):

- `S_era`: from scratch on the ERA5-and-CAMS rows, unweighted, at the learning rate of `g09`
  (5e-4). With the placebo, the weather days of those rows are permuted (`e00707_placebo`).
- `S_ens(W)`: continues from the `S_era` weights (or starts from scratch for the `R` arms), on the
  unweighted ENS rows of window `W`, at the same learning rate.
- `FT(W)`: continues from the last earlier stage on the ENS rows of window `W` with a 0.5-year
  half-life, at a learning rate of 2e-4 (as experiment 425 fine-tunes `A`). An arm with no earlier
  stage (`R_24`, `R_12`) trains from scratch at 5e-4, as experiment 425's `R`.

Every stage stops early on the held-out weeks (one week in 12). The stage that follows another stage
starts from the stored weights of the stage before, so arms that share a chain share its states
(`work/pretrained/<chain>/<month>_s<seed>.pt`).

**The outputs have experiment 425's layout** under `<data>/00707-*/work/`:

- `seed_preds/<arm>/<month>_s<seed>.parquet`: raw quantiles as a fraction of capacity, unwidened,
  keyed on `time_series_id`, `power_fcst_init_time`, `valid_time`; the scorer widens them by 1.15.
  This file is the job's completion marker and is written last, through a temporary name.
- `models/<arm>/<month>/s<seed>/`: `pooled.pt`, `meta.json`, and `standardiser.npz`.
- `pretrained/<chain>/<month>_s<seed>.pt` and `.json`: the weights after `S_era` or `S_ens`.
- `logs/<arm>/<month>_s<seed>.json`: rows, dropped rows, epochs, validation loss, and seconds.

**A job is resumable and fixed-seed.** A finished job is skipped. `torch.manual_seed` sets the
network's initial weights and the order of each epoch; the placebo's draw is fixed by the seed, the
Primary, and the fold. Like experiment 425, the code does not force deterministic GPU kernels, so a
refit matches a stored fit only to the seed-to-seed noise (`e00707_gate`).

**The arms with an `S_ens` stage need the gates.** `A_E24`, `R_E24`, `A_F`, `R_F`, `A_C`, `R_C`, and
`R_F+flags` are refused unless `e00707_gate.py` has written a passing marker for both gates
(`arms.GATE_MARKER`) or `--no-gate-marker` is passed.

**Jobs that share a fold run in one worker.** A group is the jobs of one fold and one fine-tune
window, which share the fold's rows and standardiser. `--workers` groups run at once, each with
`--threads` threads, and share one GPU.

```bash
export PYTHONPATH=$PWD/ml-experiments/00707-ens-backfill-pretraining-gain-against-ens-history-length
export DATA_PATH_INTERNAL=<path to the private data folder>
export NGED_S3_BUCKET_URL=https://example.com NGED_S3_BUCKET_ACCESS_KEY=dummy
export NGED_S3_BUCKET_SECRET=dummy
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 systemd-run --user --scope \
    -p MemoryMax=14G -p MemorySwapMax=0 \
    uv run --no-sync python ml-experiments/00707-*/e00707_fit.py --screen --workers 1 --threads 1
```
"""

import argparse
import json
import multiprocessing
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00707_arms as arms  # ty: ignore[unresolved-import]
import e00707_common as common  # ty: ignore[unresolved-import]
import e00707_folds as folds  # ty: ignore[unresolved-import]
import e00707_net as net  # ty: ignore[unresolved-import]
import e00707_placebo as placebo  # ty: ignore[unresolved-import]
import e00707_prep as prep  # ty: ignore[unresolved-import]
import numpy as np
import torch  # ty: ignore[unresolved-import]

OUT_DIR: Final[Path] = arms.WORK_DIR / "work"
SMOKE_DIR: Final[Path] = arms.WORK_DIR / "work" / "smoke"
RUN_MARGIN_DAYS: Final[int] = 16
"""A run issued this long before a trailing window starts can still reach into the window."""


@dataclass(frozen=True)
class Job:
    """One fit: an arm on a fold with a seed."""

    arm: str
    month: str
    seed: int


@dataclass(frozen=True)
class Group:
    """The jobs of one fold and one fine-tune window, run by one worker."""

    month: str
    ft_window: str
    jobs: tuple[Job, ...]


def final_path(*, out_dir: Path, job: Job) -> Path:
    """Return where a job's raw quantile forecast is stored."""
    folder = arms.ARMS[job.arm].folder
    return out_dir / "seed_preds" / folder / f"{job.month}_s{job.seed}.parquet"


def state_path(*, out_dir: Path, key: str, month: str, seed: int) -> Path:
    """Return where the weights after a stage are stored."""
    return out_dir / "pretrained" / key / f"{month}_s{seed}.pt"


def plan_jobs(
    *, arm_names: list[str], months: list[str], seeds: tuple[int, ...] | None, screen: bool
) -> list[Job]:
    """Return every job of the named arms and folds, with each arm's own seeds unless overridden.

    Args:
        arm_names: Keys of `arms.ARMS`.
        months: The folds.
        seeds: Seeds for every arm, or `None` for each arm's own (cut to three on the screen).
        screen: Whether this is the screen, which fits at most seeds 0 to 2.
    """
    out = []
    for name in arm_names:
        arm = arms.ARMS[name]
        arm_seeds = seeds if seeds is not None else arms.seeds_for(arm=arm, screen=screen)
        out.extend(Job(arm=name, month=m, seed=s) for m in months for s in arm_seeds)
    return out


def group_jobs(*, jobs: list[Job]) -> list[Group]:
    """Return the jobs grouped by fold and fine-tune window, folds in time order."""
    grouped: dict[tuple[str, str], list[Job]] = defaultdict(list)
    for job in jobs:
        grouped[job.month, arms.ARMS[job.arm].ft.name].append(job)
    return [Group(month=m, ft_window=w, jobs=tuple(js)) for (m, w), js in sorted(grouped.items())]


def check_window_coverage(*, window: arms.Window, refit: datetime, allow_partial: bool) -> dict:
    """Return the back-fill coverage of a window; raise if it is incomplete and not allowed.

    A window needs every daily run from its start to 2024-03-31 (the runs from 2024-04-01 are
    experiment 397's rows). A floor starts at the floor; a trailing window starts at the refit
    minus `days`, less 16 days because a run reaches 15 days ahead.

    Raises:
        ValueError: If runs are missing from the back-filled cache and `allow_partial` is false.
    """
    if window.floor is not None:
        start = window.floor
    elif window.days is not None:
        start = refit - timedelta(days=window.days + RUN_MARGIN_DAYS)
    else:
        start = prep.BACKFILL_FIRST_RUN
    start = max(start, prep.BACKFILL_FIRST_RUN)
    wanted = [r for r in prep.backfill_runs() if r >= start]
    manifest = prep.NEW_RAW_DIR / "manifest.json"
    have = set(json.loads(manifest.read_text())["runs"]) if manifest.exists() else set()
    missing = [r for r in wanted if f"{r:%Y-%m-%d}" not in have]
    info = {
        "window": window.name,
        "runs_wanted_before_2024_04": len(wanted),
        "runs_missing": len(missing),
        "missing_ranges": prep.as_ranges(days=missing),
    }
    if missing:
        text = (
            f"window {window.name} at {refit:%Y-%m}: "
            f"{len(missing)} of {len(wanted)} back-filled runs missing"
        )
        if not allow_partial:
            msg = (
                f"{text}; run `e00707_prep.py summaries` and `raw`, or pass --allow-partial-window"
            )
            raise ValueError(msg)
        print(f"WARNING PARTIAL WINDOW: {text}", flush=True)
    return info


def tensors(
    *,
    x: np.ndarray,
    y: np.ndarray,
    primary: np.ndarray,
    weight: np.ndarray | None,
    is_val: np.ndarray,
    origin: np.ndarray,
    flags: bool,
    fold: folds.Fold,
    device: str,
) -> dict[str, torch.Tensor]:
    """Return the training tensors on the device, with the target standardised per Primary.

    With `flags`, the two indicator columns of `folds.flag_columns` are appended to the inputs. A
    stage with no weights gets weights of one.
    """
    z = (y - fold.target_mean[primary]) / fold.target_std[primary]
    inputs = np.concatenate([x, folds.flag_columns(origin=origin)], axis=1) if flags else x
    ones = np.ones(len(y), dtype=np.float32)
    return {
        "x": torch.from_numpy(inputs).to(device),
        "z": torch.from_numpy(z.astype(np.float32)).to(device),
        "w": torch.from_numpy(ones if weight is None else weight.astype(np.float32)).to(device),
        "p": torch.from_numpy(primary).to(device),
        "v": torch.from_numpy(is_val).to(device),
    }


def train(
    *, model: torch.nn.Module, setting: net.Setting, t: dict[str, torch.Tensor], seed: int
) -> tuple[int, float]:
    """Train a network in place with experiment 397's loop; return epochs and the best loss."""
    return net.train_network(
        model=model,
        setting=setting,
        x=t["x"],
        z=t["z"],
        w=t["w"],
        primary=t["p"],
        is_val=t["v"],
        seed=seed,
    )


def write_json(*, payload: dict, target: Path) -> None:
    """Write a JSON file through a temporary name."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=1))
    temporary.replace(target)


def width_of(*, arm: arms.Arm) -> int:
    """Return the input width of an arm's network: the 46 inputs, plus the two flags."""
    return prep.feature_names()[2].n_inputs + (net.N_FLAGS if arm.flags else 0)


def last_state_key(*, arm: arms.Arm) -> str | None:
    """Return the key of the stage that the fine-tune starts from, or `None` if there is none."""
    if arm.ens is not None:
        return arm.stage_key(stage="ens")
    if arm.era:
        return arm.stage_key(stage="era")
    return None


def era_inputs(
    *, fold: folds.Fold, refit: datetime, permute: placebo.Permuter | None, device: str
) -> tuple[dict[str, torch.Tensor], dict]:
    """Return the `S_era` tensors of a fold and a description of the rows for the stage's log."""
    rows, dropped = folds.fold_pretrain(fold=fold, refit=refit, permute=permute)
    t = tensors(
        x=rows.x,
        y=rows.y,
        primary=rows.primary,
        weight=None,
        is_val=rows.is_val,
        origin=rows.origin,
        flags=False,
        fold=fold,
        device=device,
    )
    return t, {
        "rows": len(rows.x),
        "val_rows": int(rows.is_val.sum()),
        "dropped": dropped.as_dict(),
    }


def train_era_stage(
    *, todo: list[Job], fold: folds.Fold, refit: datetime, out_dir: Path, device: str
) -> None:
    """Train and store the `S_era` state of every (chain, seed) that a pending job needs.

    The unshuffled rows are built once and shared by the seeds; a placebo seed builds its own.
    """
    needed: dict[tuple[str, int], arms.Arm] = {}
    for job in todo:
        arm = arms.ARMS[job.arm]
        last = last_state_key(arm=arm)
        if not arm.era or (
            last is not None
            and state_path(out_dir=out_dir, key=last, month=job.month, seed=job.seed).exists()
        ):
            continue
        key = arm.stage_key(stage="era")
        if not state_path(out_dir=out_dir, key=key, month=job.month, seed=job.seed).exists():
            needed[key, job.seed] = arm
    shared: tuple[dict[str, torch.Tensor], dict] | None = None
    for (key, seed), arm in sorted(needed.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        if arm.placebo:
            t, info = era_inputs(
                fold=fold, refit=refit, permute=placebo.Permuter(seed=seed), device=device
            )
        else:
            if shared is None:
                shared = era_inputs(fold=fold, refit=refit, permute=None, device=device)
            t, info = shared
        started = time.monotonic()
        torch.manual_seed(seed)
        model = net.new_model(n_inputs=width_of(arm=arm), n_primaries=len(fold.target_mean))
        model = model.to(device)
        epochs, best = train(model=model, setting=net.G09, t=t, seed=seed)
        target = state_path(out_dir=out_dir, key=key, month=f"{refit:%Y-%m}", seed=seed)
        net.atomic_save(state=model.state_dict(), target=target)
        seconds = round(time.monotonic() - started)
        write_json(
            payload={
                "stage": "S_era",
                **info,
                "epochs": epochs,
                "val_nll": best,
                "seconds": seconds,
            },
            target=target.with_suffix(".json"),
        )
        print(
            f"  S_era {key} {refit:%Y-%m} seed {seed}: {info['rows']:,} rows, {epochs} epochs, "
            f"val NLL {best:.4f}, {seconds} s",
            flush=True,
        )
        if arm.placebo:
            del t
            torch.cuda.empty_cache()
    shared = None
    torch.cuda.empty_cache()


def train_ens_stage(
    *,
    todo: list[Job],
    fold: folds.Fold,
    refit: datetime,
    out_dir: Path,
    device: str,
    allow_partial: bool,
) -> dict[str, dict]:
    """Train and store the `S_ens` state of every (chain, seed) that a pending job needs.

    Returns:
        The window coverage of each `S_ens` window used, keyed by window name, for the job logs.
    """
    by_window: dict[str, list[Job]] = defaultdict(list)
    for job in todo:
        arm = arms.ARMS[job.arm]
        if arm.ens is None:
            continue
        if state_path(
            out_dir=out_dir, key=arm.stage_key(stage="ens"), month=job.month, seed=job.seed
        ).exists():
            continue
        by_window[arm.ens.name].append(job)
    coverage: dict[str, dict] = {}
    for window_name, jobs in by_window.items():
        window = arms.ARMS[jobs[0].arm].ens
        assert window is not None
        coverage[window_name] = check_window_coverage(
            window=window, refit=refit, allow_partial=allow_partial
        )
        rows, dropped = folds.ens_rows(fold=fold, refit=refit, window=window)
        first_origin = datetime.fromtimestamp(int(rows.origin.min()) / 1e6, tz=UTC)
        coverage[window_name] |= {"first_origin": f"{first_origin:%Y-%m-%d}", "rows": len(rows.x)}
        print(
            f"  S_ens window {window_name}: {len(rows.x):,} rows from {first_origin:%Y-%m-%d} "
            f"({int(rows.is_val.sum()):,} held out); dropped {dropped.as_dict()}",
            flush=True,
        )
        cache: dict[bool, dict[str, torch.Tensor]] = {}
        for job in sorted(jobs, key=lambda j: (arms.ARMS[j.arm].flags, j.seed, j.arm)):
            arm = arms.ARMS[job.arm]
            if arm.flags not in cache:
                # Sorted by flags, so the other width's tensors are never needed again.
                cache.clear()
                torch.cuda.empty_cache()
                cache[arm.flags] = tensors(
                    x=rows.x,
                    y=rows.y,
                    primary=rows.primary,
                    weight=None,
                    is_val=rows.is_val,
                    origin=rows.origin,
                    flags=arm.flags,
                    fold=fold,
                    device=device,
                )
            key = arm.stage_key(stage="ens")
            target = state_path(out_dir=out_dir, key=key, month=job.month, seed=job.seed)
            if target.exists():
                continue
            started = time.monotonic()
            torch.manual_seed(job.seed)
            model = net.new_model(n_inputs=width_of(arm=arm), n_primaries=len(fold.target_mean))
            if arm.era:
                era_state = state_path(
                    out_dir=out_dir, key=arm.stage_key(stage="era"), month=job.month, seed=job.seed
                )
                model.load_state_dict(torch.load(era_state))
            model = model.to(device)
            epochs, best = train(model=model, setting=net.G09, t=cache[arm.flags], seed=job.seed)
            net.atomic_save(state=model.state_dict(), target=target)
            write_json(
                payload={
                    "stage": "S_ens",
                    "window": window_name,
                    "rows": len(rows.x),
                    "val_rows": int(rows.is_val.sum()),
                    "dropped": dropped.as_dict(),
                    "epochs": epochs,
                    "val_nll": best,
                    "seconds": round(time.monotonic() - started),
                    "coverage": coverage[window_name],
                },
                target=target.with_suffix(".json"),
            )
            print(
                f"  S_ens {key} {job.month} seed {job.seed}: {epochs} epochs, val NLL {best:.4f}, "
                f"{time.monotonic() - started:.0f} s",
                flush=True,
            )
        del rows, cache
        torch.cuda.empty_cache()
    return coverage


def fine_tune_stage(
    *, todo: list[Job], fold: folds.Fold, out_dir: Path, device: str, coverage: dict[str, dict]
) -> None:
    """Fine-tune, forecast, and store every pending job; the forecast file is written last."""
    scale = net.Scale(
        standardiser=fold.standardiser, target_mean=fold.target_mean, target_std=fold.target_std
    )
    cache: dict[bool, dict[str, torch.Tensor]] = {}
    for job in sorted(todo, key=lambda j: (arms.ARMS[j.arm].flags, j.seed, j.arm)):
        arm = arms.ARMS[job.arm]
        if arm.flags not in cache:
            # Sorted by flags, so the other width's tensors are never needed again.
            cache.clear()
            torch.cuda.empty_cache()
            cache[arm.flags] = tensors(
                x=fold.x,
                y=fold.y,
                primary=fold.primary,
                weight=fold.weight,
                is_val=fold.is_val,
                origin=fold.origin,
                flags=arm.flags,
                fold=fold,
                device=device,
            )
        started = time.monotonic()
        torch.manual_seed(job.seed)
        model = net.new_model(n_inputs=width_of(arm=arm), n_primaries=len(fold.target_mean))
        setting = net.G09
        last = last_state_key(arm=arm)
        if last is not None:
            model.load_state_dict(
                torch.load(state_path(out_dir=out_dir, key=last, month=job.month, seed=job.seed))
            )
            setting = replace(net.G09, lr=net.FINE_TUNE_LR)
        model = model.to(device)
        epochs, best = train(model=model, setting=setting, t=cache[arm.flags], seed=job.seed)
        fitted = net.Fitted(
            model=model, setting=setting, seed=job.seed, epochs=epochs, val_nll=best
        )
        net.save_fit(
            fitted=fitted,
            scale=scale,
            target=out_dir / "models" / arm.folder / job.month / f"s{job.seed}",
        )
        forecast = net.forecast_month(
            fitted=fitted, scale=scale, month=job.month, device=device, flags=arm.flags
        )
        stages = {}
        for stage, present in (("era", arm.era), ("ens", arm.ens is not None)):
            if present:
                key = arm.stage_key(stage=stage)
                sidecar = state_path(out_dir=out_dir, key=key, month=job.month, seed=job.seed)
                stages[stage] = json.loads(sidecar.with_suffix(".json").read_text())
        seconds = round(time.monotonic() - started)
        first_origin = datetime.fromtimestamp(int(fold.origin.min()) / 1e6, tz=UTC)
        write_json(
            payload={
                "arm": job.arm,
                "month": job.month,
                "seed": job.seed,
                "stages_of_arm": arm.stages,
                "ft_window": arm.ft.name,
                "fine_tune_rows": len(fold.x),
                "fine_tune_val_rows": int(fold.is_val.sum()),
                "fine_tune_rows_dropped": fold.dropped.as_dict(),
                "fine_tune_first_origin": f"{first_origin:%Y-%m-%d}",
                "fine_tune_learning_rate": setting.lr,
                "epochs": epochs,
                "val_nll": best,
                "seconds_fine_tune_and_forecast": seconds,
                "earlier_stages": stages,
                "ens_window_coverage": coverage.get(arm.ens.name) if arm.ens is not None else None,
                "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
            },
            target=out_dir / "logs" / arm.folder / f"{job.month}_s{job.seed}.json",
        )
        common.atomic_write(frame=forecast, target=final_path(out_dir=out_dir, job=job))
        print(
            f"{arm.name} {job.month} seed {job.seed}: {epochs} epochs, val NLL {best:.4f}, "
            f"{seconds} s (fine-tune and forecast)",
            flush=True,
        )
    del cache
    torch.cuda.empty_cache()


def run_group(*, group: Group, out_dir: Path, device: str, allow_partial: bool) -> str:
    """Fit every pending job of a group: the `S_era` states, the `S_ens` states, the fine-tunes.

    Returns:
        A progress line.
    """
    todo = [j for j in group.jobs if not final_path(out_dir=out_dir, job=j).exists()]
    if not todo:
        return f"{group.month} {group.ft_window}: all {len(group.jobs)} jobs done"
    started = time.monotonic()
    refit = prep.refit_of(month=group.month)
    ft_window = arms.ARMS[todo[0].arm].ft
    check_window_coverage(window=ft_window, refit=refit, allow_partial=allow_partial)
    fold = folds.build_fold(refit=refit, window=ft_window)
    print(
        f"fold {group.month} window {ft_window.name}: {len(fold.x):,} fine-tuning rows "
        f"({int(fold.is_val.sum()):,} held out); dropped {fold.dropped.as_dict()}; "
        f"{len(todo)} jobs to fit",
        flush=True,
    )
    train_era_stage(todo=todo, fold=fold, refit=refit, out_dir=out_dir, device=device)
    coverage = train_ens_stage(
        todo=todo,
        fold=fold,
        refit=refit,
        out_dir=out_dir,
        device=device,
        allow_partial=allow_partial,
    )
    fine_tune_stage(todo=todo, fold=fold, out_dir=out_dir, device=device, coverage=coverage)
    return (
        f"{group.month} {group.ft_window}: {len(todo)} jobs in {time.monotonic() - started:.0f} s"
    )


def _initialise_worker(threads: int, max_epochs: int | None) -> None:
    """Set the thread counts and, for the smoke run, the epoch limit, in a worker process."""
    torch.set_num_threads(threads)
    if max_epochs is not None:
        net.MAX_EPOCHS = max_epochs


def _run_group_kw(args: tuple[Group, Path, str, bool]) -> str:
    """Return `run_group` for a tuple of its arguments, as a picklable function of one argument."""
    group, out_dir, device, allow_partial = args
    return run_group(group=group, out_dir=out_dir, device=device, allow_partial=allow_partial)


def refuse_without_gate_marker(*, arm_names: list[str], allowed: bool) -> None:
    """Exit with status 1 if an arm with an `S_ens` stage is named and the gates have not passed.

    The arms with an `S_ens` stage are `A_E24`, `R_E24`, `A_F`, `R_F`, `A_C`, `R_C`, and
    `R_F+flags`. They read the back-filled runs, which the two gates have not yet cleared unless
    `arms.GATE_MARKER` holds both gate names.

    Args:
        arm_names: The arms to fit.
        allowed: Whether `--no-gate-marker` was passed.
    """
    long_arms = [n for n in arm_names if arms.ARMS[n].ens is not None]
    if not long_arms or allowed:
        return
    marker = json.loads(arms.GATE_MARKER.read_text()) if arms.GATE_MARKER.exists() else {}
    missing = [g for g in arms.GATE_NAMES if g not in marker]
    if missing:
        print(
            f"refusing to fit {long_arms}: gate marker {arms.GATE_MARKER} lacks {missing}; run "
            f"`e00707_gate.py` or pass --no-gate-marker",
            flush=True,
        )
        raise SystemExit(1)


def main() -> None:
    """Plan the jobs from the arguments and run the groups."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--arms", nargs="+", choices=list(arms.ARMS), default=None)
    parser.add_argument("--months", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--screen", action="store_true", help="the 4 screen folds, seeds 0 to 2")
    parser.add_argument("--gate", action="store_true", help="A_24 and R_24 on the 2 gate folds")
    parser.add_argument("--workers", type=int, default=1, help="groups fitted at once")
    parser.add_argument("--threads", type=int, default=1, help="CPU threads per worker")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true", help="one epoch, outputs under work/smoke")
    parser.add_argument("--allow-partial-window", action="store_true")
    parser.add_argument(
        "--no-gate-marker",
        action="store_true",
        help="fit arms with an S_ens stage although the gates have not both passed",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the groups and fit nothing")
    args = parser.parse_args()
    if args.gate:
        arm_names = args.arms or ["A_24", "R_24"]
        months = args.months or list(arms.GATE_MONTHS)
    else:
        arm_names = args.arms or [n for n, a in arms.ARMS.items() if a.in_full_run]
        months = args.months or list(arms.SCREEN_MONTHS if args.screen else arms.ALL_MONTHS)
    seeds = tuple(args.seeds) if args.seeds else None
    jobs = plan_jobs(arm_names=arm_names, months=months, seeds=seeds, screen=args.screen)
    out_dir = SMOKE_DIR if args.smoke else OUT_DIR
    groups = group_jobs(jobs=jobs)
    pending = [j for j in jobs if not final_path(out_dir=out_dir, job=j).exists()]
    print(f"{len(jobs)} jobs in {len(groups)} groups; {len(pending)} not yet fitted; out {out_dir}")
    if args.dry_run:
        for g in groups:
            names = sorted({f"{j.arm}" for j in g.jobs})
            print(f"  {g.month} window {g.ft_window}: {len(g.jobs)} jobs, arms {names}")
        return
    refuse_without_gate_marker(arm_names=arm_names, allowed=args.no_gate_marker)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(args.threads)
    os.environ["POLARS_MAX_THREADS"] = str(args.threads)
    max_epochs = 1 if args.smoke else None
    work = [(g, out_dir, args.device, args.allow_partial_window) for g in groups]
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialise_worker,
        initargs=(args.threads, max_epochs),
    ) as pool:
        for line in pool.map(_run_group_kw, work):
            print(line, flush=True)


if __name__ == "__main__":
    main()
