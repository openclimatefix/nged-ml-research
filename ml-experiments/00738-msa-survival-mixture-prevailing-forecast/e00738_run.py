# ruff: noqa: E501
"""The one entry point of experiment 738: one function per stage, started by `run.sh` or `run_workers.sh`.

Stages in run order: `phase0` (worlds, curves, the A1 table, the stratum check, the gallery),
`review-mark`, `smoke-fit`, `smoke-predict`, `smoke-collect`, `screen-fits` (fits, then predictions
of the 4 screen folds at seed 0), `gate`, `gate-remedy`, `screen-score`, `full-fits`, `predict`,
`score`, `members`, `protocol`, `report`, and `figures`. A stage needs the marker of the stage before
it, and the review marker must hold the hash of the current code. A failed task writes
`failed/task_<name>.json` with the message and the traceback, and a later successful run of the task
moves that record into `failed/archive/`. A failed CPU stage exits with code 3, a failed GPU chain with
code 5 (and the gate's second failure), the gate's first failure with code 6, and a launcher that
finds the GPU busy with code 75.

Every fit and every prediction job runs in a child process of its own (`one-task`), because
experiment 713's pipeline reads its month and phase once, when it is imported. The parent process
keeps the ledger, the failure records, and the done markers.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

import e00738_common as common
import e00738_phases as phases
import numpy as np

SMOKE_STAGES: Final[frozenset[str]] = frozenset({"smoke-fit", "smoke-predict", "smoke-collect"})
"""The stages that read their prerequisite markers from the data folder and write the rest to a
scratch folder."""
GPU_STAGES: Final[frozenset[str]] = frozenset(
    {"smoke-fit", "smoke-predict", "screen-fits", "full-fits", "predict", "gate-remedy"}
)
STAGES: Final[tuple[str, ...]] = (
    "phase0",
    "review-mark",
    "smoke-fit",
    "smoke-predict",
    "smoke-collect",
    "screen-fits",
    "gate",
    "gate-remedy",
    "screen-score",
    "full-fits",
    "predict",
    "score",
    "members",
    "protocol",
    "report",
    "figures",
    "one-task",
)
PROJECTED_GPU_HOURS: Final[float] = 6.3


# ---------------------------------------------------------------------------------------------
# Job lists (pure)
# ---------------------------------------------------------------------------------------------
def fit_jobs(
    *, screen: bool, drop_ld: bool = False, seeds: tuple[int, ...] = common.SEEDS
) -> list[tuple[str, dict[str, Any]]]:
    """Return the fit jobs in the order seed, then fold, then arm.

    Args:
        screen: True for the 20 screen fits (4 folds, seed 0, 5 arms); False for the other fits.
        drop_ld: Cut LD from the fits outside the screen (the first cut of the cost plan).
        seeds: The seeds of the full set (the second cut keeps seeds 0 and 1).

    Returns:
        The names and keyword arguments of the tasks.
    """
    arms = tuple(a for a in common.FIT_ARMS if not (drop_ld and not screen and a == "LD"))
    jobs: list[tuple[str, dict[str, Any]]] = []
    if screen:
        jobs.extend(
            (
                f"fit_{a}_{m}_s{common.SCREEN_SEED}",
                {"arm": a, "month": m, "seed": common.SCREEN_SEED},
            )
            for m in common.SCREEN_FOLDS
            for a in arms
        )
        return jobs
    for s in seeds:
        for m in common.FOLDS:
            if s == common.SCREEN_SEED and m in common.SCREEN_FOLDS:
                continue
            jobs.extend((f"fit_{a}_{m}_s{s}", {"arm": a, "month": m, "seed": s}) for a in arms)
    return jobs


def predict_jobs(
    *, screen: bool, seeds: tuple[int, ...] = common.SEEDS
) -> list[tuple[str, dict[str, Any]]]:
    """Return the prediction jobs, one per fold and seed (the 4 screen jobs, or the other 20)."""
    out = []
    for s in seeds:
        for m in common.FOLDS:
            is_screen = s == common.SCREEN_SEED and m in common.SCREEN_FOLDS
            if is_screen == screen:
                out.append((f"predict_{m}_s{s}", {"month": m, "seed": s}))
    return out


def count_fits() -> int:
    """Return the number of fits of the full plan: 120."""
    return len(fit_jobs(screen=True)) + len(fit_jobs(screen=False))


# ---------------------------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------------------------
def task_fit(*, arm: str, month: str, seed: int, device: str) -> None:
    """Fit one arm in this process; the gate's remedy flag, once set, applies to every head fit.

    Run only in a child process (`one-task`), because the pipeline of experiment 713 is imported for
    one month and one phase per process.
    """
    import e00738_fit as fit
    import e00738_gate as gate

    fit.fit_job(
        arm=arm,
        month=month,
        seed=seed,
        device=device,
        remedy=arm != common.ARM_W0 and gate.remedy_active(),
    )


def task_predict(
    *,
    month: str,
    seed: int,
    device: str,
    arms: tuple[str, ...] | None = None,
    only: tuple[str, ...] | None = None,
) -> None:
    """Predict the arms and panels of one fold and seed in this process (a child process only)."""
    import e00738_predict as pred

    pred.run_job(month=month, seed=seed, device=device, arms=arms, only=only)


def child_argv(
    *,
    kind: str,
    month: str,
    seed: int,
    device: str,
    arm: str | None = None,
    arms: tuple[str, ...] | None = None,
    only: tuple[str, ...] | None = None,
) -> list[str]:
    """Return the command line of the child process that runs one fit or prediction job."""
    argv = [
        sys.executable,
        str(Path(__file__).resolve()),
        "one-task",
        "--kind",
        kind,
        "--month",
        month,
        "--seed",
        str(seed),
        "--device",
        device,
    ]
    if arm is not None:
        argv += ["--arm", arm]
    if arms is not None:
        argv += ["--arms", ",".join(arms)]
    if only is not None:
        argv += ["--only", ",".join(only)]
    return argv


def sub_fit(
    *, arm: str, month: str, seed: int, device: str, extra_env: dict[str, str] | None = None
) -> None:
    """Fit one arm in a child process of its own, in the training phase of its month."""
    phases.run_child(
        argv=child_argv(kind="fit", arm=arm, month=month, seed=seed, device=device),
        env=phases.child_env(month=month, phase="train", extra=extra_env),
    )


def sub_predict(
    *,
    month: str,
    seed: int,
    device: str,
    arms: tuple[str, ...] | None = None,
    only: tuple[str, ...] | None = None,
    extra_env: dict[str, str] | None = None,
) -> None:
    """Predict the arms and panels of one fold and seed in a child process of its own, in the prediction phase."""
    phases.run_child(
        argv=child_argv(
            kind="predict", arms=arms, only=only, month=month, seed=seed, device=device
        ),
        env=phases.child_env(month=month, phase="predict", extra=extra_env),
    )


def stage_one_task(*, args: argparse.Namespace) -> None:
    """Run one fit or prediction job in this process; the parent process started it for this job alone."""
    if args.kind == "fit":
        task_fit(arm=args.arm, month=args.month, seed=args.seed, device=args.device)
    else:
        task_predict(
            month=args.month,
            seed=args.seed,
            device=args.device,
            arms=tuple(args.arms.split(",")) if args.arms else None,
            only=tuple(args.only.split(",")) if args.only else None,
        )


def load_fold(*, month: str, phase: str, device: str) -> tuple[Any, Any, np.ndarray]:
    """Return (data, 713 world, is_mw) of a fold in the given phase; the phase variable is read at import."""
    os.environ["E00667_PHASE"] = phase
    import e00713_train as tr  # ty: ignore[unresolved-import]
    import e00738_fit as fit

    data = fit.load_fold_data(month=month, phase=phase, seed=0, device=device)
    tw = tr.World(
        source=data.source,
        cutoff_idx=data.cutoff_idx,
        clean=data.clean,
        target_mean=data.target_mean,
        target_std=data.target_std,
        month=month,
        placebo_pool=tr.block_pool(clean=data.clean),
    )
    return data, tw, fit.fold_world()[2]


def task_curves(*, month: str) -> None:
    """Build the fold's empirical survival curve and record its error against the generator's.

    The arms use the empirical curve of the training worlds' drawn durations. A fitted Weibull (or
    the best of the four families) is recorded beside it for the report only. The task fails when the
    empirical curve is more than `CURVE_ERROR_MAX` from the generator's at a checkpoint.
    """
    import e00650_common as c650  # ty: ignore[unresolved-import]
    import e00738_fit as fit
    import e00738_survival as surv
    import numpy as np

    is_mw = fit.fold_world()[2]
    cutoff = c650.month_start_idx(month=month)
    durations = fit.fold_durations(month=month, seed=0, cutoff_idx=cutoff, is_mw=is_mw)
    curve = surv.EmpiricalCurve.of(durations_days=durations)
    weibull = surv.fit_curve(durations_days=durations)
    err = surv.curve_error(curve=curve)
    weibull_err = surv.curve_error(curve=weibull)
    result = {
        "family": "empirical",
        "n_durations": int(durations.size),
        "error": err,
        "error_ok": surv.curve_error_ok(curve=curve),
        "worst": float(np.max(list(err.values()))),
        "report_only_fitted_family": {
            "family": weibull.family,
            "params": list(weibull.params),
            "error": weibull_err,
            "worst": float(np.max(list(weibull_err.values()))),
        },
    }
    common.atomic_json(obj=result, path=common.OUT / "curves" / f"{month}.json")
    if not result["error_ok"]:
        msg = f"the empirical survival curve of fold {month} is off the generator's: {err}"
        raise AssertionError(msg)


def task_strata_check() -> None:
    """Write every fold's injected shifts to disk, read them back, and check strata and holdout.

    The stratum shares come from the files that were written (the drawn end and the drawn shift),
    not from a simulation. The same check confirms that no shift reaches 2026-07-01 and that the
    forecast's valid-time filter drops rows of the 2026-06 fold and of no other fold, which needs no
    fit. The shares at a lead of 10 days (the plan's first figures) come from the simulation, beside
    the file-based pooled shares.
    """
    import e00738_world as world
    import e00738_worldcheck as wc

    for month in common.FOLDS:
        wc.write_worlds(month=month)
    found = wc.check_worlds(months=common.FOLDS)
    result = wc.verdict(found=found)
    common.atomic_json(
        obj={
            "pooled_leads_from_files": found["shares"],
            "simulated_lead_10_days": world.simulate_stratum_shares(lead_days=(10.0,)),
            "expected": common.EXPECTED_SHARES,
            "expected_lead_10_days": common.EXPECTED_SHARES_LEAD_10,
            "dropped_rows_by_fold": found["dropped_rows"],
            "latest_shift_end_idx": found["latest_shift_end_idx"],
            "holdout_idx": found["holdout_idx"],
            **result,
        },
        path=common.OUT / "checks" / "strata_shares.json",
    )
    if result["off"] or result["faults"]:
        msg = f"injected worlds differ from the plan: {result}"
        raise AssertionError(msg)


def task_a1_table(*, month: str) -> None:  # noqa: PLR0915 - one linear Phase 0 report
    """Print A1's accuracy by true age band and its false-alarm share, on a training world and on the real record (no fit needed)."""
    import e00738_survival as surv
    import e00738_train as train
    import e00738_world as world
    import numpy as np
    import torch  # ty: ignore[unresolved-import]

    data, tw, is_mw = load_fold(month=month, phase="train", device="cpu")
    z = (data.clean - data.target_mean[:, None]) / data.target_std[:, None]
    noise = surv.noise_unit_from_panel(z=z, cutoff_idx=data.cutoff_idx)
    own_shifts = []
    for p in range(16):
        own_shifts.extend(
            world.renewal_shifts(
                space="train",
                fit_seed=0,
                month_idx=common.month_index(month=month),
                pass_no=0,
                primary=p,
                window_start_idx=0,
                window_end_idx=data.cutoff_idx,
                is_mw=is_mw,
            )
        )
    own = world.build_own(clean=data.clean, shifts=own_shifts, is_mw=is_mw)
    sources = {
        "injected": train.world_source(
            base=tw.source,
            clean=data.clean,
            own=own,
            target_mean=data.target_mean,
            target_std=data.target_std,
            device="cpu",
        ),
        "real": tw.source,
    }
    import e00713_train as tr  # ty: ignore[unresolved-import]

    origins = np.arange(48 * 20 + 24, data.cutoff_idx - 48 * 3, 48 * 3)
    prim, cc = np.meshgrid(np.arange(16), origins, indexing="ij")
    prim, cc = prim.ravel(), cc.ravel()
    samples = tr.SampleSet(primary=prim, c_idx=cc, is_val=np.zeros(len(prim), bool))
    hist: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, src in sources.items():
        power_rows, visible_rows = [], []
        for start in range(0, len(prim), 512):
            sel = np.arange(start, min(start + 512, len(prim)))
            b = tr.batch_for(
                source=src,
                samples=samples,
                index=sel,
                lead=np.zeros(len(sel), int),
                ensemble=False,
                cut=10**9,
                dropout=None,
            )
            power_rows.append(b.inputs.hist_power.numpy())
            visible_rows.append(b.inputs.hist_visible.numpy())
        hist[name] = (np.concatenate(power_rows), np.concatenate(visible_rows))
    # A1's threshold is tuned on the real rows of the training period (this fold's cutoff is the
    # earliest of the eight, so the rows lie in every fold's training window), never on a scoring
    # month. These rows include the logged events, which overstates the false alarms slightly.
    z_tuned, share_tuned = surv.tune_a1_z(
        hist_power=hist["real"][0], visible=hist["real"][1], noise=noise
    )
    share_default = float(
        (
            surv.age_rule_a1(hist_power=hist["real"][0], visible=hist["real"][1], noise=noise) > 0
        ).mean()
    )
    ages = {
        name: surv.age_rule_a1(
            hist_power=hp, visible=vis, noise=noise, z=z_tuned if z_tuned is not None else None
        )
        for name, (hp, vis) in hist.items()
    }
    by_prim = {p: [s for s in own_shifts if s.primary == p] for p in range(16)}
    true_age = np.full(len(prim), -1.0)
    since_end = np.full(len(prim), 1e9)
    for i, (p, c) in enumerate(zip(prim, cc, strict=True)):
        for s in by_prim[int(p)]:
            if s.onset_idx <= c < s.end_idx:
                true_age[i] = (c - s.onset_idx) / world.PER_DAY
            elif c >= s.end_idx:
                since_end[i] = min(since_end[i], (c - s.end_idx) / world.PER_DAY)
    rows = {}
    for band, lo, hi in common.A1_BANDS:
        sel = (true_age >= lo) & (true_age < hi)
        found = ages["injected"][sel]
        rows[band] = {
            "n": int(sel.sum()),
            "share_zero": float((found == 0).mean()) if sel.any() else None,
            "mean_abs_age_error": float(np.abs(found[found > 0] - true_age[sel][found > 0]).mean())
            if (found > 0).any()
            else None,
            "mean_signed_age_error": float((found[found > 0] - true_age[sel][found > 0]).mean())
            if (found > 0).any()
            else None,
        }
    free = (true_age < 0) & (since_end > 15.0)
    rows["false_alarm_injected_event_free"] = float((ages["injected"][free] > 0).mean())
    rows["false_alarm_real_all_rows"] = float((ages["real"] > 0).mean())
    rows["false_alarm_real_at_default_z"] = share_default
    rows["default_z"] = common.A1_Z
    rows["tuned_z"] = z_tuned
    rows["bar_ok"] = z_tuned is not None and share_tuned <= common.A1_FALSE_ALARM_MAX
    seen = (true_age >= 0) & (ages["injected"] > 0)
    bias = float((ages["injected"][seen] - true_age[seen]).mean()) if seen.any() else None
    rows["age_bias_days"] = bias
    rows["age_bias_bar_days"] = common.A1_BIAS_MAX_DAYS
    rows["age_bias_exceeds_bar"] = bias is not None and abs(bias) > common.A1_BIAS_MAX_DAYS
    rows["noise_unit"] = noise
    del torch
    pooled = (true_age >= 3.0) & (true_age < 11.0)
    miss_3_10 = float((ages["injected"][pooled] == 0).mean()) if pooled.any() else float("nan")
    rows["miss_rate_by_true_age_band"] = {
        band: rows[band]["share_zero"] for band, _, _ in common.A1_BANDS
    }
    rows["miss_rate_true_age_3_to_10_days"] = miss_3_10
    common.atomic_json(obj=rows, path=common.OUT / "a1" / f"{month}.json")
    print(json.dumps(rows, indent=1), flush=True)
    if z_tuned is None:
        msg = (
            f"A1 cannot reach a false-alarm share of {common.A1_FALSE_ALARM_MAX} on the training "
            f"rows at any threshold of the grid (share {share_tuned:.3f} at the largest). "
            "Stop and report to the coordinator."
        )
        raise RuntimeError(msg)
    if not miss_3_10 <= common.A1_MISS_MAX_3_10:
        msg = (
            f"A1 at the tuned threshold {z_tuned} misses {miss_3_10:.2f} of the shifts of 3 to 10 "
            f"days, over the {common.A1_MISS_MAX_3_10} limit. Stop and report to the coordinator."
        )
        raise RuntimeError(msg)
    common.atomic_json(
        obj={
            "z": z_tuned,
            "false_alarm_share": share_tuned,
            "tuned_on_fold": month,
            "noise_unit": noise,
            "miss_rate_true_age_3_to_10_days": miss_3_10,
        },
        path=common.OUT / common.A1_THRESHOLD_FILE,
    )


def task_gallery(*, month: str) -> None:
    """Draw the gallery from the cleaned power of one fold's first Primary other than D8."""
    import e00738_gallery as gallery

    data, _, _ = load_fold(month=month, phase="train", device="cpu")
    import numpy as np

    row = data.clean[0]
    start = int(np.argmax(np.isfinite(row).astype(int).cumsum() > 48 * 60))
    gallery.draw_gallery(clean_row=row[start : start + 48 * 45])


# ---------------------------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------------------------
def run_cpu_tasks(
    *,
    jobs: list[tuple[str, dict[str, Any]]],
    fn: Callable[..., None],
    stage: str,
    args: argparse.Namespace,
) -> None:
    """Run a worker's share of CPU tasks with the CPU budget check between tasks."""
    mine = phases.shard(items=jobs, worker=args.worker, of=args.of)
    phases.run_chain(jobs=mine, fn=fn, device="cpu", stage=stage, exit_code=common.EXIT_FAILED)


def run_gpu_tasks(
    *,
    jobs: list[tuple[str, dict[str, Any]]],
    fn: Callable[..., None],
    stage: str,
    args: argparse.Namespace,
) -> None:
    """Run the GPU chain: one GPU process at a time, stopping between fits at the stop line."""
    mine = phases.shard(items=jobs, worker=args.worker, of=args.of)
    phases.run_chain(
        jobs=[(n, {**kw, "device": args.device}) for n, kw in mine],
        fn=fn,
        device="gpu",
        stage=stage,
        exit_code=common.EXIT_CHAIN_FAILED,
    )


def mark(*, name: str, out: Path | None = None) -> None:
    """Write a stage marker holding the code hash (under `OUT` unless `out` is given)."""
    path = (out if out is not None else common.OUT) / phases.STAGE_MARKERS.get(name, f"{name}.done")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"code_hash={phases.code_hash()} {time.strftime('%FT%TZ', time.gmtime())}\n")


def finish_stage(*, name: str, tasks: list[str]) -> None:
    """Write the stage marker when every task of the stage has finished, whichever worker finishes last.

    Each worker calls this after its own share. The marker is written by the first call that finds
    every task done and no failure record left, so a run with several workers writes it too.

    Args:
        name: The stage's key in `phases.STAGE_MARKERS`.
        tasks: The names of every task of the stage, not only this worker's.
    """
    try:
        phases.require_finished(names=tasks)
    except RuntimeError as exc:
        print(f"stage marker {name} not written yet: {exc}", flush=True)
        return
    mark(name=name)


def stage_phase0(*, args: argparse.Namespace) -> None:
    """Curves, the A1 table, the stratum check, and the gallery; then print the projection."""
    jobs: list[tuple[str, dict[str, Any]]] = [("strata_check", {})]
    jobs += [(f"curves_{m}", {"month": m}) for m in common.FOLDS]
    jobs += [("a1_table_2025-11", {"month": "2025-11"}), ("gallery_2025-11", {"month": "2025-11"})]

    # a task's name selects its function, so each job carries its own callable
    failed = 0
    for name, kw in phases.shard(items=jobs, worker=args.worker, of=args.of):
        phases.require_budget(device="cpu")
        fn = (
            task_strata_check
            if name == "strata_check"
            else task_a1_table
            if name.startswith("a1")
            else task_gallery
            if name.startswith("gallery")
            else task_curves
        )
        failed += not phases.run_task(name=name, fn=fn, kwargs=kw, device="cpu", stage="phase0")
    if failed:
        sys.exit(common.EXIT_FAILED)
    finish_stage(name="phase0", tasks=[n for n, _ in jobs])
    print(
        f"projection: {count_fits()} fits, {PROJECTED_GPU_HOURS} GPU-process-hours, stop at {common.STOP_GPU_HOURS}",
        flush=True,
    )


def stage_review_mark() -> None:
    """Write the review marker with the code hash, once the reviewer has passed the code."""
    phases.require_stage(stage="review")
    print("review marker written for", phases.write_review_marker(), flush=True)


def smoke_marker_root() -> Path:
    """Return the folder whose markers the smoke stages read: the data folder, read-only."""
    return common.REAL_OUT


def stage_smoke_fit(*, args: argparse.Namespace) -> None:
    """One pass of every arm on the real device and the real files; scratch output only.

    The prerequisite markers (`phase0.done` and the review marker) are read from the data folder.
    Every fit runs in a child process whose environment alone holds the one-pass cap.
    """
    require_scratch()
    phases.require_stage(stage="smoke", out=smoke_marker_root())
    jobs = [
        (
            f"smoke_fit_{arm}",
            {
                "arm": arm,
                "month": common.SCREEN_FOLDS[0],
                "seed": 0,
                "extra_env": {common.PASS_CAP_ENV: "1"},
            },
        )
        for arm in common.FIT_ARMS
    ]
    run_gpu_tasks(jobs=jobs, fn=sub_fit, stage="smoke", args=args)


def stage_smoke_predict(*, args: argparse.Namespace) -> None:
    """Predict the real record and one world pass of the first screen fold, reproduce stored `M1`, and read the files with the scoring code."""
    require_scratch()
    phases.require_stage(stage="smoke", out=smoke_marker_root())
    jobs = [
        (
            "smoke_predict",
            {"month": common.SCREEN_FOLDS[0], "seed": 0, "only": ("real", "pass00")},
        )
    ]
    run_gpu_tasks(jobs=jobs, fn=sub_predict, stage="smoke", args=args)


def stage_smoke_collect() -> None:
    """Read the smoke predictions with the scoring code, write `smoke.ok` to the scratch folder, and write it to the data folder.

    The marker in the data folder is the one file the smoke test writes there. `screen-fits` needs it.
    """
    require_scratch()
    phases.require_stage(stage="smoke", out=smoke_marker_root())
    import e00738_predict as pred
    import e00738_score as score
    import numpy as np
    import polars as pl

    month = common.SCREEN_FOLDS[0]
    for arm in common.ALL_ARMS:
        for name in ("real", "pass00"):
            f = pl.read_parquet(pred.panel_path(arm=arm, month=month, seed=0, name=name))
            origin, valid, lead = score.row_indices(frame=f)
            y = f["actual"].to_numpy().astype(float)
            q = f.select(*score._QCOLS).to_numpy().astype(float)
            ok = np.isfinite(y)
            assert ok.any()
            assert np.isfinite(score.crps_points(y=y[ok], q=q[ok])).all()
            assert len(origin) == len(valid) == len(lead)
    assert (common.OUT / "checks" / f"reproduce_m1_{month}_s0.json").exists(), (
        "stored M1 was not reproduced"
    )
    book_smoke_ledger()
    mark(name="smoke")
    mark(name="smoke", out=smoke_marker_root())
    print("smoke ok", flush=True)


def book_smoke_ledger() -> int:
    """Copy the smoke lines of the scratch ledger into the real ledger, once, and return how many.

    The smoke stages run on the real GPU, so their hours count against the stop line.
    """
    scratch = common.OUT / "time_log.csv"
    real = common.REAL_OUT / "time_log.csv"
    if not scratch.exists():
        return 0
    have = set(real.read_text().splitlines()) if real.exists() else set()
    new = [
        line
        for line in scratch.read_text().splitlines()
        if line.split(",")[0] == "smoke" and line not in have
    ]
    if new:
        real.parent.mkdir(parents=True, exist_ok=True)
        with real.open("a") as f:
            f.writelines(line + "\n" for line in new)
    return len(new)


def require_scratch() -> None:
    """Stop unless `E00738_OUT` names a scratch folder (the smoke test never writes its output to the data folder)."""
    if "E00738_OUT" not in os.environ or not common.is_scratch():
        sys.exit("refusing to run the smoke test outside a scratch folder: set E00738_OUT")


def stage_screen_fits(*, args: argparse.Namespace) -> None:
    """The 20 screen fits, then the 4 screen predictions."""
    phases.require_stage(stage="screen_fits")
    print(
        f"fits starting: {len(fit_jobs(screen=True))} screen fits, {count_fits()} in all, projected {PROJECTED_GPU_HOURS} GPU-process-hours, memory cap {common.MEMORY_MAX_GB} GB, cores from SLOT_CPUS",
        flush=True,
    )
    run_gpu_tasks(jobs=fit_jobs(screen=True), fn=sub_fit, stage="screen_fit", args=args)
    run_gpu_tasks(jobs=predict_jobs(screen=True), fn=sub_predict, stage="screen_predict", args=args)
    names = [n for n, _ in (*fit_jobs(screen=True), *predict_jobs(screen=True))]
    if all(phases.is_done(name=n) for n in names):
        common.atomic_json(obj=recompute_projection(), path=common.OUT / "projection.json")
    finish_stage(name="screen_fits", tasks=names)


def recompute_projection() -> dict[str, float]:
    """Recompute the projection from the mean and the 90th percentile of the 20 screen fit times, not from one fit."""
    import e00738_fit as fit
    import numpy as np

    seconds = np.array(
        [
            fit.read_log(arm=a, month=m, seed=0)["seconds"]
            for m in common.SCREEN_FOLDS
            for a in common.FIT_ARMS
        ]
    )
    n_remaining = count_fits() - len(seconds)
    mean, p90 = float(seconds.mean()), float(np.percentile(seconds, 90))
    gpu = phases.spent_hours(device="gpu")
    return {
        "mean_seconds": mean,
        "p90_seconds": p90,
        "projected_remaining_hours_mean": n_remaining * mean / 3600.0,
        "projected_remaining_hours_p90": n_remaining * p90 / 3600.0,
        "spent_gpu_hours": gpu,
        "projected_total_p90": gpu + n_remaining * p90 / 3600.0 + 1.8,
        "cut_list_fires": gpu + n_remaining * p90 / 3600.0 + 1.8 > common.STOP_GPU_HOURS,
    }


def screen_shifts(*, month: str):  # noqa: ANN201
    """Return the shifts of every scoring panel of a fold, with the fold's draws keyed as experiment 650 keys them."""
    import e00650_common as c650  # ty: ignore[unresolved-import]
    import e00738_fit as fit
    import e00738_predict as pred
    import e00738_world as world

    is_mw = fit.fold_world()[2]
    return pred.all_panel_shifts(
        month=month,
        month_start=int(c650.month_start_idx(month=month)),
        window_end=fit.window_end_idx(),
        is_mw=is_mw,
        real_days=world.load_real_durations_days(),
    )


def a1_age_bias() -> dict[str, object]:
    """Return A1's mean signed age error that Phase 0 recorded, and whether it is over the 1-day bar.

    A1's age is not changed. A bias over the bar is stated here so that a reader of the gate and of
    every reading that rests on the survival ratio sees it.

    Returns:
        The bias in days by fold (None when Phase 0 wrote no table), and `exceeds_bar`.
    """
    found: dict[str, float | None] = {}
    for path in sorted((common.OUT / "a1").glob("*.json")):
        found[path.stem] = json.loads(path.read_text()).get("age_bias_days")
    exceeds = any(v is not None and abs(v) > common.A1_BIAS_MAX_DAYS for v in found.values())
    return {"by_fold": found, "bar_days": common.A1_BIAS_MAX_DAYS, "exceeds_bar": exceeds}


def stage_gate() -> None:
    """The known-answer gate on the screen fits and predictions; sets the remedy flag on a first failure, exits with code 5 on a second."""
    phases.require_stage(stage="gate")
    phases.require_all_recipes()
    import e00738_gate as gate
    import e00738_predict as pred
    import e00738_score as score
    import numpy as np
    import polars as pl

    total = {
        "num": 0.0, "den": 0.0, "num_seen": 0.0, "den_seen": 0.0,
        "m1_num": 0.0, "m1_num_seen": 0.0, "rows": 0, "rows_seen": 0,
    }  # fmt: skip
    ratios: list[float] = []
    by_band: dict[str, list[float]] = {}
    ids = site_labels()[1]
    for m in common.SCREEN_FOLDS:
        shifts = screen_shifts(month=m)
        real = pl.read_parquet(pred.panel_path(arm="M1", month=m, seed=0, name="real"))
        for name in (n for n in pred.panel_names() if score.world_of(name=n) == "train_family"):
            lp_path = pred.panel_path(arm="LP", month=m, seed=0, name=name)
            diag_path = lp_path.with_name(f"{name}_diag.parquet")
            got = gate.panel_readings(
                m1=pl.read_parquet(pred.panel_path(arm="M1", month=m, seed=0, name=name)),
                lp=pl.read_parquet(lp_path),
                real=real,
                diag=pl.read_parquet(diag_path) if diag_path.exists() else None,
                shifts=shifts[name],
                ids=ids,
            )
            for k in total:
                total[k] += got[k]
            ratios.extend(got["depth_ratio"])
            for band, vals in got["depth_bands"].items():
                by_band.setdefault(band, []).extend(vals)
    m1_follow = total["m1_num_seen"] / total["den_seen"] if total["den_seen"] > 0 else 0.0
    record = gate.evaluate(
        oracle={
            "midday": gate.oracle_follow_synthetic(onset_half_hour_of_day=24),
            "midnight": gate.oracle_follow_synthetic(onset_half_hour_of_day=0),
        },
        lp_follow=total["num_seen"] / total["den_seen"] if total["den_seen"] > 0 else float("nan"),
        lp_follow_all_far=total["num"] / total["den"] if total["den"] > 0 else float("nan"),
        depth_ratio=float(np.median(ratios)) if ratios else float("nan"),
        m1_follow=m1_follow,
        diagnostics={
            "a1_age_bias_days": a1_age_bias(),
            "m1_follow_all_far_rows": total["m1_num"] / total["den"] if total["den"] > 0 else None,
            "depth_ratio_by_band": {k: float(np.median(v)) for k, v in by_band.items()},
            "far_rows_lead_1_2": total["rows"],
            "far_rows_a1_sees": total["rows_seen"],
            "share_of_far_rows_a1_sees": total["rows_seen"] / max(total["rows"], 1),
        },
    )
    decision = gate.write_gate(
        record=record, remedy_done=gate.remedy_complete(), code_hash=phases.code_hash()
    )
    print(json.dumps(record, indent=1), "decision:", decision, flush=True)
    if decision == "remedy":
        sys.exit(common.EXIT_REMEDY)
    if decision == "stop":
        sys.exit(common.EXIT_CHAIN_FAILED)


def archive_first_attempt() -> bool:
    """Move the 16 first-attempt screen head fits and their forecasts aside, once, and return whether it moved them.

    A marker records that the move happened, so a rerun after a crash in the remedy resumes the
    remedy fits (each has its own done marker) and never moves the finished remedy fits away. The
    first attempt is kept under `gate_first_attempt/`, because the gate's first reading rests on it.
    """
    import shutil

    marker = common.OUT / "gate_remedy_started.ok"
    if marker.exists():
        return False
    keep = common.OUT / "gate_first_attempt"
    for m in common.SCREEN_FOLDS:
        for arm in common.HEAD_ARMS:
            paths = list(common.fit_paths(arm=arm, month=m, seed=0).values())
            paths.append(phases.recipe_path(path=paths[0]))
            for path in paths:
                if path.exists():
                    target = keep / path.relative_to(common.OUT)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(path, target)
            folder = common.OUT / "predictions" / arm / f"{m}_s0"
            if folder.exists():
                target = keep / folder.relative_to(common.OUT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(folder, target)
    marker.write_text(f"code_hash={phases.code_hash()}\n")
    return True


def stage_gate_remedy(*, args: argparse.Namespace) -> None:
    """Refit the 16 screen head fits (LP, LD, LM, LR on 4 folds) with the remedy, predict them, and run the gate again.

    The remedy (cap 120, patience 15) applies to every head arm alike, so every screen head fit is
    refitted under it. Then S* is chosen, and C2 is compared, between fits made with one recipe. The
    screen fits of W0 are not refitted, because the remedy does not apply to W0. A rerun after a crash
    resumes: the first attempt is moved aside once, and each finished remedy task is skipped.
    """
    import e00738_gate as gate

    assert gate.remedy_active(), "the gate has not set the remedy flag"
    arms = common.HEAD_ARMS
    archive_first_attempt()
    run_gpu_tasks(
        jobs=[
            (f"fit_{arm}_{m}_s0_remedy", {"arm": arm, "month": m, "seed": 0})
            for m in common.SCREEN_FOLDS
            for arm in arms
        ],
        fn=sub_fit,
        stage="gate_remedy",
        args=args,
    )
    run_gpu_tasks(
        jobs=[
            (f"predict_{m}_s0_remedy", {"month": m, "seed": 0, "arms": arms})
            for m in common.SCREEN_FOLDS
        ],
        fn=sub_predict,
        stage="gate_remedy",
        args=args,
    )
    stage_gate()


def stage_screen_score(*, args: argparse.Namespace) -> None:
    """Score the screen folds at seed 0, choose S* by the decision rule (interim), and print the seed spread of stored M1."""
    phases.require_stage(stage="screen_score")
    phases.require_all_recipes()
    import e00738_fit as fit
    import e00738_report as report
    import e00738_score as score

    labels, ids = site_labels()
    for m in common.SCREEN_FOLDS:
        score.score_fold_seed(
            month=m, seed=0, labels=labels, ids=ids, shifts_of=screen_shifts(month=m)
        )
    tables = report.load_tables(months=common.SCREEN_FOLDS, seeds=(0,))
    unscaled = report.screen_in_force(tables=tables, months=common.SCREEN_FOLDS, labels=labels)
    scaled = report.screen_in_force(
        tables=tables, months=common.SCREEN_FOLDS, labels=labels, metric="crps_scaled"
    )
    star = score.choose_s_star(in_force=unscaled)
    common.atomic_json(
        obj={
            "s_star": star,
            "in_force_unscaled": unscaled,
            "in_force_scaled_report_only": scaled,
            "interim": True,
            "tie_margin": common.TIE_MARGIN,
        },
        path=common.OUT / "sstar.json",
    )
    print(
        "interim screen in-force unscaled CRPS:", unscaled, "(scaled, report only:", scaled, ") S* =", star,
        flush=True,
    )  # fmt: skip
    print(
        "seed spread of stored M1 on one screen fold:",
        m1_seed_spread(month=common.SCREEN_FOLDS[0]),
        flush=True,
    )
    del fit
    mark(name="screen_score")


def m1_seed_spread(*, month: str) -> float:
    """Return the standard deviation, over the stored seeds of `M1`, of the fold's mean CRPS on the real record (points of capacity)."""
    import e00738_score as score
    import numpy as np
    import polars as pl

    vals = []
    for s in common.SEEDS:
        f = pl.read_parquet(common.ROOT_713 / "work" / "fits" / "M1" / f"{month}_s{s}.parquet")
        y = f["actual"].to_numpy().astype(float)
        q = f.select(*score._QCOLS).to_numpy().astype(float)
        ok = np.isfinite(y)
        vals.append(float(score.crps_points(y=y[ok], q=q[ok]).mean()))
    return float(np.std(vals, ddof=1))


def site_labels():  # noqa: ANN201
    """Return the 16 Primaries' labels and `time_series_id`s in index order."""
    import e00738_fit as fit

    sites = fit.site_table()
    return sites["label"].to_list(), sites["time_series_id"].to_numpy()


def stage_full_fits(*, args: argparse.Namespace) -> None:
    """The other 100 fits, with the manager's cuts applied if the screen projection fired them."""
    phases.require_stage(stage="full_fits")
    cuts = common.read_cuts()
    if cuts["drop_ld"]:
        star = json.loads((common.OUT / "sstar.json").read_text())["s_star"]
        assert star != "LD", "the cut list drops LD, but the screen chose LD as S*"
    jobs = fit_jobs(screen=False, drop_ld=cuts["drop_ld"], seeds=common.active_seeds())
    run_gpu_tasks(jobs=jobs, fn=sub_fit, stage="full_fit", args=args)
    finish_stage(name="full_fits", tasks=[n for n, _ in jobs])


def stage_predict(*, args: argparse.Namespace) -> None:
    """The other prediction jobs (20, fewer after the seed cut)."""
    phases.require_stage(stage="predict")
    jobs = predict_jobs(screen=False, seeds=common.active_seeds())
    run_gpu_tasks(jobs=jobs, fn=sub_predict, stage="predict", args=args)
    finish_stage(name="predict", tasks=[n for n, _ in jobs])


def stage_score(*, args: argparse.Namespace) -> None:
    """Score each fold and seed, streaming, and write `score.done`."""
    phases.require_stage(stage="score")
    phases.require_all_recipes()
    import e00738_score as score

    labels, ids = site_labels()
    jobs = [
        (f"score_{m}_s{s}", {"month": m, "seed": s, "labels": labels, "ids": ids})
        for s in common.active_seeds()
        for m in common.FOLDS
    ]

    def one(*, month: str, seed: int, labels: list[str], ids: Any) -> None:
        score.score_fold_seed(
            month=month, seed=seed, labels=labels, ids=ids, shifts_of=screen_shifts(month=month)
        )

    run_cpu_tasks(jobs=jobs, fn=one, stage="score", args=args)
    finish_stage(name="score", tasks=[n for n, _ in jobs])


MEMBER_PANELS: Final[tuple[str, ...]] = ("pass00", "pass01", "pass02", "pass03")


def stage_members(*, args: argparse.Namespace) -> None:
    """Run the 51-path check on `LM` over the screen folds and write `checks/members.json`.

    The check needs a mixture arm with one fitted curve, so it runs on `LM`. `LR` learns its hazards
    from the features of each sample, which no stored file holds, and `LD` is not a mixture. The
    result applies to the chosen arm only when that arm is `LM`; the file says which.
    """
    del args
    phases.require_stage(stage="score")
    phases.require_all_recipes()
    import e00738_fit as fit
    import e00738_members as members
    import e00738_predict as pred
    import e00738_survival as surv
    import polars as pl
    import torch  # ty: ignore[unresolved-import]

    star = json.loads((common.OUT / "sstar.json").read_text())["s_star"]
    per_fold: dict[str, dict[str, float]] = {}
    for month in common.SCREEN_FOLDS:
        saved = torch.load(
            common.fit_paths(arm="LM", month=month, seed=0)["state"], map_location="cpu"
        )
        curve = surv.curve_from_record(record=saved["curve"])
        rows = []
        for name in MEMBER_PANELS:
            arm_path = pred.panel_path(arm="LM", month=month, seed=0, name=name)
            result = members.check_frames(
                arm=pl.read_parquet(arm_path),
                base=pl.read_parquet(
                    pred.panel_path(arm=common.ARM_M1, month=month, seed=0, name=name)
                ),
                diag=pl.read_parquet(arm_path.with_name(f"{name}_diag.parquet")),
                curve=curve,
            )
            rows.append(result)
        weight = np.array([r["rows"] for r in rows])
        per_fold[month] = {
            k: float(np.average([r[k] for r in rows], weights=weight))
            for k in rows[0]
            if k != "rows"
        } | {"rows": float(weight.sum())}
    pooled = {
        k: float(
            np.average(
                [v[k] for v in per_fold.values()], weights=[v["rows"] for v in per_fold.values()]
            )
        )
        for k in next(iter(per_fold.values()))
        if k != "rows"
    }
    passed = members.passes(
        ended_gap=pooled["ended_gap"],
        quantile_gap=pooled["quantile_gap"],
        crps_gap=pooled["crps_gap"],
    )
    common.atomic_json(
        obj={
            "arm_checked": "LM",
            "s_star": star,
            "applies_to_s_star": star == "LM",
            "per_fold": per_fold,
            "pooled": pooled,
            "passed": passed,
        },
        path=common.OUT / "checks" / "members.json",
    )
    print("member-path check on LM:", json.dumps(pooled, indent=1), "passed:", passed, flush=True)
    del fit
    if not passed:
        sys.exit(common.EXIT_FAILED)


def stage_protocol(*, args: argparse.Namespace) -> None:
    """Score the real record of every arm by experiment 713's evaluation protocol (four scenarios)."""
    del args
    phases.require_stage(stage="score")
    phases.require_all_recipes()
    import e00738_protocol as protocol

    folder = protocol.run()
    print("protocol tables written to", folder, flush=True)


def stage_report() -> None:
    """Print every number of the page, with the verdict, and write `report.done`."""
    phases.require_stage(stage="report")
    phases.require_all_recipes()
    import e00738_report as report

    labels, _ = site_labels()
    star = json.loads((common.OUT / "sstar.json").read_text())["s_star"]
    invalid = not (common.OUT / "gate_passed.json").exists()
    seeds = common.active_seeds()
    tables = report.load_tables(months=common.FOLDS, seeds=seeds)
    rep = report.compute_report(
        tables=tables,
        labels=labels,
        s_star=star,
        seeds=seeds,
        invalid=invalid,
        drop_ld=common.read_cuts()["drop_ld"],
    )
    rep["audit_triggers"] = report.audit_triggers(report=rep)
    report.write_report(report=rep)
    print(report.render(report=rep), flush=True)
    mark(name="report")


def stage_figures() -> None:
    """Draw the result figures from the report."""
    phases.require_stage(stage="figures")
    import e00738_figures as figures
    import e00738_report as report

    rep = report.read_report()
    figures.forest_plot(report=rep)
    figures.stratum_plot(report=rep)
    figures.age_band_plot(report=rep)
    figures.loss_plots(month=common.SCREEN_FOLDS[0], seed=0)
    mark(name="figures")


# ---------------------------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------------------------
def parse_args(*, argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=STAGES)
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--of", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--kind", choices=("fit", "predict"), help="one-task only")
    p.add_argument("--arm", help="one-task fit only")
    p.add_argument("--month", help="one-task only")
    p.add_argument("--seed", type=int, default=0, help="one-task only")
    p.add_argument("--arms", help="one-task predict only: comma-separated arms")
    p.add_argument("--only", help="one-task predict only: comma-separated panel names")
    return p.parse_args(argv)


def dispatch(*, args: argparse.Namespace) -> None:
    """Run the stage named in `args`, after the guards that need no review marker."""
    s = args.stage
    if s not in {"phase0", "review-mark", "one-task"}:
        # the parent process of a `one-task` child has already checked the marker
        phases.require_review(out=smoke_marker_root() if s in SMOKE_STAGES else None)
    table: dict[str, Callable[[], None]] = {
        "one-task": lambda: stage_one_task(args=args),
        "phase0": lambda: stage_phase0(args=args),
        "review-mark": stage_review_mark,
        "smoke-fit": lambda: stage_smoke_fit(args=args),
        "smoke-predict": lambda: stage_smoke_predict(args=args),
        "smoke-collect": stage_smoke_collect,
        "screen-fits": lambda: stage_screen_fits(args=args),
        "gate": stage_gate,
        "gate-remedy": lambda: stage_gate_remedy(args=args),
        "screen-score": lambda: stage_screen_score(args=args),
        "full-fits": lambda: stage_full_fits(args=args),
        "predict": lambda: stage_predict(args=args),
        "score": lambda: stage_score(args=args),
        "members": lambda: stage_members(args=args),
        "protocol": lambda: stage_protocol(args=args),
        "report": stage_report,
        "figures": stage_figures,
    }
    table[s]()


def main(*, argv: list[str] | None = None) -> None:
    """Run a stage; a failure writes `failed/stage_<stage>.json` with the message and the traceback, and exits with code 3 (CPU) or 5 (GPU)."""
    args = parse_args(argv=argv)
    phases.assert_blas_pinned()
    gpu = args.stage in GPU_STAGES
    if gpu and args.of != 1:
        sys.exit(
            f"refusing to run: {args.stage} is a GPU stage, which runs as one worker "
            f"(got --of {args.of}); two workers would both train on the one GPU"
        )
    if args.stage != "one-task" and common.PASS_CAP_ENV in os.environ:
        sys.exit(
            f"refusing to run: {common.PASS_CAP_ENV} is set in the environment (only the smoke "
            "stages set it, and only for their child processes)"
        )
    if gpu and args.device.startswith("cuda"):
        phases.require_gpu_free()
    if args.stage == "one-task":
        dispatch(args=args)  # the parent process records a failure
        return
    try:
        dispatch(args=args)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - the failure record is the point of this boundary
        phases.write_failure(name=f"{args.stage}_worker_{args.worker}", exc=exc, kind="stage")
        print(f"FAILED stage {args.stage}: {exc}", file=sys.stderr, flush=True)
        sys.exit(common.EXIT_CHAIN_FAILED if gpu else common.EXIT_FAILED)
    phases.archive_stage_failures(stage=args.stage)


if __name__ == "__main__":
    main()
