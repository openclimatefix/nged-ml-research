# ruff: noqa: E501
"""The one entry point of experiment 738: one function per stage, started by `run.sh` or `run_workers.sh`.

Stages in run order: `phase0` (worlds, curves, the A1 table, the stratum check, the gallery),
`review-mark`, `smoke-fit`, `smoke-predict`, `smoke-collect`, `screen-fits` (fits, then predictions
of the 4 screen folds at seed 0), `gate`, `gate-remedy`, `screen-score`, `full-fits`, `predict`,
`score`, `members`, `report`, and `figures`. A stage needs the marker of the stage before it, and the
review marker must hold the hash of the current code. A failed task writes `failed/task_<name>.json`
with the message and the traceback. A failed CPU stage exits with code 3, a failed GPU chain with code
5, and a launcher that finds the GPU busy exits with code 75.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable
from typing import Any, Final

import e00738_common as common
import e00738_phases as phases
import numpy as np

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
)
PROJECTED_GPU_HOURS: Final[float] = 5.9


# ---------------------------------------------------------------------------------------------
# Job lists (pure)
# ---------------------------------------------------------------------------------------------
def fit_jobs(
    *, screen: bool, drop_ld: bool = False, seeds: tuple[int, ...] = common.SEEDS
) -> list[tuple[str, dict[str, Any]]]:
    """Return the fit jobs in the order seed, then fold, then arm.

    Args:
        screen: True for the 20 screen fits (4 folds, seed 0, 5 arms); False for the other fits.
        drop_ld: Cut LD (the first cut of the cost plan).
        seeds: The seeds of the full set (the second cut keeps seeds 0 and 1).

    Returns:
        The names and keyword arguments of the tasks.
    """
    arms = tuple(a for a in common.FIT_ARMS if not (drop_ld and a == "LD"))
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
    """Fit one arm; the gate's remedy flag, once set, applies to every head fit."""
    import e00738_fit as fit
    import e00738_gate as gate

    fit.fit_job(
        arm=arm,
        month=month,
        seed=seed,
        device=device,
        remedy=arm != common.ARM_W0 and gate.remedy_active(),
    )


def task_predict(*, month: str, seed: int, device: str) -> None:
    """Predict every arm and panel of one fold and seed."""
    import e00738_predict as pred

    pred.run_job(month=month, seed=seed, device=device)


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
    """Fit the survival curve of a fold's training worlds and record its error against the generator's."""
    import e00650_common as c650  # ty: ignore[unresolved-import]
    import e00738_fit as fit
    import e00738_survival as surv
    import numpy as np

    is_mw = fit.fold_world()[2]
    cutoff = c650.month_start_idx(month=month)
    curve = fit.fold_curve(month=month, seed=0, cutoff_idx=cutoff, is_mw=is_mw)
    err = surv.curve_error(curve=curve)
    common.atomic_json(
        obj={
            "family": curve.family,
            "params": list(curve.params),
            "error": err,
            "error_ok": surv.curve_error_ok(curve=curve),
            "worst": float(np.max(list(err.values()))),
        },
        path=common.OUT / "curves" / f"{month}.json",
    )


def task_strata_check() -> None:
    """Compare the simulated stratum shares with the expected ones, at the pooled leads and at 10 days.

    The scored rows span leads of 1 to 10 days, so the expected shares are those at the pooled leads.
    The event-free and onset-ahead shares depend on the lead, because the valid time moves with it.
    The shares at a lead of 10 days (the plan's first figures) are written beside them.
    """
    import e00738_world as world

    pooled = world.simulate_stratum_shares()
    at_10 = world.simulate_stratum_shares(lead_days=(10.0,))
    bad = {}
    for k, want in common.EXPECTED_SHARES.items():
        tol = (
            common.STRATUM_TOLERANCE_END if k in {"near", "ended_24h"} else common.STRATUM_TOLERANCE
        )
        if abs(pooled[k] - want) > tol:
            bad[k] = (pooled[k], want)
    common.atomic_json(
        obj={
            "simulated_pooled_leads": pooled,
            "simulated_lead_10_days": at_10,
            "expected": common.EXPECTED_SHARES,
            "expected_lead_10_days": common.EXPECTED_SHARES_LEAD_10,
            "off": bad,
        },
        path=common.OUT / "checks" / "strata_shares.json",
    )
    if bad:
        msg = f"stratum shares differ from the plan: {bad}"
        raise AssertionError(msg)


def task_a1_table(*, month: str) -> None:
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
    ages = {}
    for name, src in sources.items():
        found = []
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
            found.append(
                surv.age_rule_a1(
                    hist_power=b.inputs.hist_power.numpy(),
                    visible=b.inputs.hist_visible.numpy(),
                    noise=noise,
                )
            )
        ages[name] = np.concatenate(found)
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
        }
    free = (true_age < 0) & (since_end > 15.0)
    rows["false_alarm_injected_event_free"] = float((ages["injected"][free] > 0).mean())
    rows["false_alarm_real_all_rows"] = float((ages["real"] > 0).mean())
    rows["bar_ok"] = rows["false_alarm_real_all_rows"] <= common.A1_FALSE_ALARM_MAX
    rows["noise_unit"] = noise
    del torch
    common.atomic_json(obj=rows, path=common.OUT / "a1" / f"{month}.json")
    print(json.dumps(rows, indent=1), flush=True)


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


def mark(*, name: str) -> None:
    """Write a stage marker holding the code hash."""
    path = common.OUT / phases.STAGE_MARKERS.get(name, f"{name}.done")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"code_hash={phases.code_hash()} {time.strftime('%FT%TZ', time.gmtime())}\n")


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
    if args.of == 1:
        mark(name="phase0")
    print(
        f"projection: {count_fits()} fits, {PROJECTED_GPU_HOURS} GPU-process-hours, stop at {common.STOP_GPU_HOURS}",
        flush=True,
    )


def stage_review_mark() -> None:
    """Write the review marker with the code hash, once the reviewer has passed the code."""
    phases.require_stage(stage="review")
    print("review marker written for", phases.write_review_marker(), flush=True)


def stage_smoke_fit(*, args: argparse.Namespace) -> None:
    """One pass of every arm on the real device and the real files; scratch output only."""
    require_scratch()
    phases.require_stage(stage="smoke")
    os.environ[common.PASS_CAP_ENV] = "1"  # one pass of every arm
    import e00738_fit as fit

    for arm in common.FIT_ARMS:
        fit.fit_job(arm=arm, month=common.SCREEN_FOLDS[0], seed=0, device=args.device)


def stage_smoke_predict(*, args: argparse.Namespace) -> None:
    """Predict the real record and one world pass of the first screen fold, reproduce stored `M1`, and read the files with the scoring code."""
    require_scratch()
    phases.require_stage(stage="smoke")
    import e00738_predict as pred

    pred.run_job(month=common.SCREEN_FOLDS[0], seed=0, device=args.device, only=("real", "pass00"))


def stage_smoke_collect() -> None:
    """Read the smoke predictions with the scoring code and write `smoke.ok`."""
    require_scratch()
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
    mark(name="smoke")
    print("smoke ok", flush=True)


def require_scratch() -> None:
    """Stop unless `E00738_OUT` names a scratch folder (the smoke test never writes to the data folder)."""
    if "E00738_OUT" not in os.environ:
        sys.exit("refusing to run the smoke test outside a scratch folder: set E00738_OUT")


def stage_screen_fits(*, args: argparse.Namespace) -> None:
    """The 20 screen fits, then the 4 screen predictions."""
    phases.require_stage(stage="screen_fits")
    print(
        f"fits starting: {len(fit_jobs(screen=True))} screen fits, {count_fits()} in all, projected {PROJECTED_GPU_HOURS} GPU-process-hours, memory cap {common.MEMORY_MAX_GB} GB, cores from SLOT_CPUS",
        flush=True,
    )
    run_gpu_tasks(jobs=fit_jobs(screen=True), fn=task_fit, stage="screen_fit", args=args)
    run_gpu_tasks(
        jobs=predict_jobs(screen=True), fn=task_predict, stage="screen_predict", args=args
    )
    if args.of == 1:
        projection = recompute_projection()
        common.atomic_json(obj=projection, path=common.OUT / "projection.json")
        mark(name="screen_fits")


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


def stage_gate() -> None:
    """The known-answer gate on the screen fits and predictions; sets the remedy flag on a first failure, exits with code 5 on a second."""
    phases.require_stage(stage="gate")
    import e00738_gate as gate
    import e00738_predict as pred
    import e00738_score as score
    import numpy as np
    import polars as pl

    num = den = 0.0
    ratios: list[float] = []
    for m in common.SCREEN_FOLDS:
        shifts = screen_shifts(month=m)
        for name in (n for n in pred.panel_names() if score.world_of(name=n) == "train_family"):
            m1 = pl.read_parquet(pred.panel_path(arm="M1", month=m, seed=0, name=name))
            lp = pl.read_parquet(pred.panel_path(arm="LP", month=m, seed=0, name=name))
            real = pl.read_parquet(pred.panel_path(arm="M1", month=m, seed=0, name="real"))
            ids = sorted(m1["time_series_id"].unique().to_list())
            prim = np.array([ids.index(t) for t in m1["time_series_id"].to_list()])
            origin, valid, lead = score.row_indices(frame=m1)
            sh = shifts[name]
            masks = score.row_masks(
                prim=prim,
                origin_idx=origin,
                valid_idx=valid,
                lead_day=lead,
                onset=np.array([s.onset_idx for s in sh]),
                end=np.array([s.end_idx for s in sh]),
                clipped=np.array([s.clipped for s in sh]),
            )
            sel = masks["far"] & np.isin(lead, (1, 2))
            delta = np.nan_to_num(m1["actual"].to_numpy() - real["actual"].to_numpy())
            num += float(((lp["q50"].to_numpy() - m1["q50"].to_numpy())[sel] * delta[sel]).sum())
            den += float((delta[sel] ** 2).sum())
            diag_path = pred.panel_path(arm="LP", month=m, seed=0, name=name).with_name(
                f"{name}_diag.parquet"
            )
            if diag_path.exists():
                d = pl.read_parquet(diag_path).filter(pl.col("lead_day") == 1)
                dprim = np.array([ids.index(t) for t in d["time_series_id"].to_list()])
                c = d["origin_us"].to_numpy()
                c_idx = (c - common.GRID_START_US) // common.STEP_US + common.NATIVE_OFFSET_STEPS
                onset = np.array([s.onset_idx for s in sh])[dprim]
                end = np.array([s.end_idx for s in sh])[dprim]
                age = (c_idx - onset) / 48.0
                true = np.array([s.sign * s.depth for s in sh])[dprim]
                seen = d["age"].to_numpy() > 0
                sel2 = (c_idx >= onset) & (c_idx < end)
                tab = gate.depth_ratio_table(
                    depth_share=d["depth_share"].to_numpy()[sel2],
                    true_depth=true[sel2],
                    age=age[sel2],
                    seen=seen[sel2],
                )
                if not np.isnan(tab["3-10"]):
                    ratios.append(tab["3-10"])
    record = gate.evaluate(
        oracle={
            "midday": gate.oracle_follow_synthetic(onset_half_hour_of_day=24),
            "midnight": gate.oracle_follow_synthetic(onset_half_hour_of_day=0),
        },
        lp_follow=num / den if den > 0 else float("nan"),
        depth_ratio=float(np.median(ratios)) if ratios else float("nan"),
    )
    decision = gate.write_gate(
        record=record, remedy_done=gate.remedy_active(), code_hash=phases.code_hash()
    )
    print(json.dumps(record, indent=1), "decision:", decision, flush=True)
    if decision == "remedy":
        sys.exit(common.EXIT_REMEDY)
    if decision == "stop":
        sys.exit(common.EXIT_CHAIN_FAILED)


def stage_gate_remedy(*, args: argparse.Namespace) -> None:
    """Refit the 4 screen LP fits with the pre-registered remedy (cap 120, patience 15), predict them, and run the gate again."""
    import e00738_gate as gate
    import e00738_predict as pred

    assert gate.remedy_active(), "the gate has not set the remedy flag"
    for m in common.SCREEN_FOLDS:
        for p in (common.fit_paths(arm="LP", month=m, seed=0),):
            for path in p.values():
                path.unlink(missing_ok=True)
        import shutil

        shutil.rmtree(common.OUT / "predictions" / "LP" / f"{m}_s0", ignore_errors=True)
    jobs = [
        (f"fit_LP_{m}_s0_remedy", {"arm": "LP", "month": m, "seed": 0}) for m in common.SCREEN_FOLDS
    ]
    run_gpu_tasks(jobs=jobs, fn=task_fit, stage="gate_remedy", args=args)
    for m in common.SCREEN_FOLDS:
        pred.run_job(month=m, seed=0, device=args.device, arms=("LP",))
    stage_gate()


def stage_screen_score(*, args: argparse.Namespace) -> None:
    """Score the screen folds at seed 0, choose S* by the decision rule (interim), and print the seed spread of stored M1."""
    phases.require_stage(stage="screen_score")
    import e00738_fit as fit
    import e00738_report as report
    import e00738_score as score

    labels, ids = site_labels()
    for m in common.SCREEN_FOLDS:
        score.score_fold_seed(
            month=m, seed=0, labels=labels, ids=ids, shifts_of=screen_shifts(month=m)
        )
    tables = report.load_tables(months=common.SCREEN_FOLDS, seeds=(0,))
    scaled = report.screen_in_force_scaled(tables=tables, months=common.SCREEN_FOLDS, labels=labels)
    star = score.choose_s_star(in_force_scaled=scaled)
    common.atomic_json(
        obj={
            "s_star": star,
            "in_force_scaled": scaled,
            "interim": True,
            "tie_margin": common.TIE_MARGIN,
        },
        path=common.OUT / "sstar.json",
    )
    print("interim screen in-force scaled CRPS:", scaled, "S* =", star, flush=True)
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
    cuts = common.OUT / "cuts.json"
    spec = json.loads(cuts.read_text()) if cuts.exists() else {}
    seeds = (0, 1) if spec.get("seeds01") else common.SEEDS
    run_gpu_tasks(
        jobs=fit_jobs(screen=False, drop_ld=bool(spec.get("drop_ld")), seeds=seeds),
        fn=task_fit,
        stage="full_fit",
        args=args,
    )
    if args.of == 1:
        mark(name="full_fits")


def stage_predict(*, args: argparse.Namespace) -> None:
    """The other 20 prediction jobs."""
    phases.require_stage(stage="predict")
    run_gpu_tasks(jobs=predict_jobs(screen=False), fn=task_predict, stage="predict", args=args)
    if args.of == 1:
        mark(name="predict")


def stage_score(*, args: argparse.Namespace) -> None:
    """Score each fold and seed, streaming, and write `score.done`."""
    phases.require_stage(stage="score")
    import e00738_score as score

    labels, ids = site_labels()
    jobs = [
        (f"score_{m}_s{s}", {"month": m, "seed": s, "labels": labels, "ids": ids})
        for s in common.SEEDS
        for m in common.FOLDS
    ]

    def one(*, month: str, seed: int, labels: list[str], ids: Any) -> None:
        score.score_fold_seed(
            month=month, seed=seed, labels=labels, ids=ids, shifts_of=screen_shifts(month=month)
        )

    run_cpu_tasks(jobs=jobs, fn=one, stage="score", args=args)
    if args.of == 1:
        phases.require_finished(names=[n for n, _ in jobs])
        mark(name="score")


MEMBER_PANELS: Final[tuple[str, ...]] = ("pass00", "pass01", "pass02", "pass03")


def stage_members(*, args: argparse.Namespace) -> None:
    """Run the 51-path check on `LM` over the screen folds and write `checks/members.json`.

    The check needs a mixture arm with one fitted curve, so it runs on `LM`. `LR` learns its hazards
    from the features of each sample, which no stored file holds, and `LD` is not a mixture. The
    result applies to the chosen arm only when that arm is `LM`; the file says which.
    """
    del args
    phases.require_stage(stage="score")
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
        c = saved["curve"]
        curve = surv.Curve(family=c["family"], params=tuple(c["params"]), nll=float(c["nll"]))
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
    import e00738_protocol as protocol

    folder = protocol.run()
    print("protocol tables written to", folder, flush=True)


def stage_report() -> None:
    """Print every number of the page, with the verdict, and write `report.done`."""
    phases.require_stage(stage="report")
    import e00738_report as report

    labels, _ = site_labels()
    star = json.loads((common.OUT / "sstar.json").read_text())["s_star"]
    invalid = not (common.OUT / "gate_passed.json").exists()
    tables = report.load_tables(months=common.FOLDS, seeds=common.SEEDS)
    rep = report.compute_report(tables=tables, labels=labels, s_star=star, invalid=invalid)
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
    return p.parse_args(argv)


def dispatch(*, args: argparse.Namespace) -> None:
    """Run the stage named in `args`, after the guards that need no review marker."""
    s = args.stage
    if s not in {"phase0", "review-mark"}:
        phases.require_review()
    table: dict[str, Callable[[], None]] = {
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
    gpu = args.stage in {
        "smoke-fit",
        "smoke-predict",
        "screen-fits",
        "full-fits",
        "predict",
        "gate-remedy",
    }
    try:
        dispatch(args=args)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - the failure record is the point of this boundary
        phases.write_failure(name=f"{args.stage}_worker_{args.worker}", exc=exc, kind="stage")
        print(f"FAILED stage {args.stage}: {exc}", file=sys.stderr, flush=True)
        sys.exit(common.EXIT_CHAIN_FAILED if gpu else common.EXIT_FAILED)


if __name__ == "__main__":
    main()
