# ruff: noqa: E501
"""Experiment 721: the command-line entry of the real run (fit, predict, and score as separate processes).

Each subcommand is one process, started through `run.sh --go e00721_cli.py ...`:

- `fit` runs the fit phase of a wave (or of one job) with `E00667_PHASE=train`: the world is built
  from the fold's training pipeline, so nothing at or after the fold's cutoff is read.
- `predict` runs the predict phase of the same jobs with `E00667_PHASE=predict`, in a new process,
  because experiment 713's pipeline is built for one phase per process.
- `stage0` runs the checks before any fit (the flag test, the reachability and noise checks) and
  writes `synthetic_world.json` and `stage0.json`.
- `probe` runs the 8 capacity probes of D5 and D10 (GPU pretraining, CPU probe).
- `quicklook` scores the quick look and prints the harm stop and the interim contrasts.
- `predict` also writes the MDE file as soon as `SG-0` seed 1 on fold 2026-05 has been predicted.
- `score` runs the power check of G3 over experiment 713's stored `M1` seeds, scores the stored
fits, and
  writes `summary.json` and `report.md`.

Examples:
    run.sh --go e00721_cli.py stage0
    run.sh --go e00721_cli.py probe --device cuda
    run.sh --go e00721_cli.py fit --wave 1 --device cuda
    run.sh --go e00721_cli.py predict --wave 1 --device cuda
    run.sh --go e00721_cli.py score
"""

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Final

import e00721_paths

# isort: split
import e00721_core as core
import e00721_fit as fit

M1_FOLDER: Final = (
    Path("ml-experiments")
    / "00713-msa-axial-attention-demand-network-part-1"
    / "work"
    / "fits"
    / "M1"
)
"""Folder, under the data folder, of experiment 713's stored `M1` forecasts: `<fold>_s<seed>.parquet`."""
PHASES: Final = {"fit": "train", "predict": "predict"}
"""The `E00667_PHASE` of each process kind."""


def jobs_of(
    *, wave: int | None, arm: str | None, world: str | None, fold: str | None, seed: int | None,
    capacity_scale: float = 1.0,
) -> list[fit.Job]:  # fmt: skip
    """Return the jobs that the arguments name: a wave, or one job (arm, world, fold, and seed all given)."""
    if wave is not None:
        return [replace(j, capacity_scale=capacity_scale) for j in fit.wave_jobs(wave=wave)]
    if None in (arm, world, fold, seed):
        msg = "name a --wave, or all of --arm, --world, --fold, and --seed"
        raise ValueError(msg)
    assert arm is not None
    assert world is not None
    assert fold is not None
    assert seed is not None
    return [fit.Job(arm=arm, world=world, fold=fold, seed=seed, capacity_scale=capacity_scale)]


def spent_hours(*, data_dir: Path) -> float:
    """Return the GPU-process-hours that the stored fits used (the sum of `hours` over the curve files)."""
    return fit.Budget.from_files(data_dir=data_dir).spent_hours


def site_ids(*, table: Sequence[dict]) -> tuple[list[str], dict[str, int]]:
    """Return the 16 Primary labels in panel order and the `time_series_id` of every Primary label.

    Args:
        table: Rows with `site`, `time_series_id`, and `substation_type` (experiment 87's series table).

    Returns:
        The 16 labels sorted by number, and each label's `time_series_id`.
    """
    primaries = sorted(
        (r for r in table if r["substation_type"] == "Primary"), key=lambda r: int(r["site"][1:])
    )
    return [r["site"] for r in primaries], {r["site"]: int(r["time_series_id"]) for r in primaries}


def pipeline_fold(*, job: fit.Job) -> str:
    """Return the fold month whose pipeline serves the job (the synthetic world rides on `SYNTH_PIPELINE_FOLD`)."""
    return core.SYNTH_PIPELINE_FOLD if job.world == "synthetic" else job.fold


def group_by_pipeline_fold(*, jobs: Sequence[fit.Job]) -> dict[str, list[fit.Job]]:
    """Return the jobs grouped by pipeline fold, in the order of each group's first job.

    Experiment 713's pipeline reads its fold month once, at import, so a process serves one fold month
    (and one phase). Jobs of another fold month run in a process of their own.
    """
    groups: dict[str, list[fit.Job]] = {}
    for job in jobs:
        groups.setdefault(pipeline_fold(job=job), []).append(job)
    return groups


def run_in_processes(*, argv: Sequence[str], groups: dict[str, list[fit.Job]]) -> int:
    """Run each fold month's jobs in its own process (`--only-fold`) and return the first non-zero exit status."""
    for month in groups:
        status = subprocess.run(
            [sys.executable, str(Path(__file__)), *argv, "--only-fold", month], check=False
        ).returncode
        if status != 0:
            return status
    return 0


def write_mde_if_ready(*, data_dir: Path, hidden_ids: dict[str, int]) -> None:
    """Write the MDE file once `SG-0` seed 1 on fold 2026-05 has been predicted (Stage 5 opens with this fit).

    The file refuses to be overwritten, and it is written before any other seed-1 fit is read.
    """
    import e00721_driver as drv

    root = fit.data_root(data_dir=data_dir)
    path = root / "mde.json"
    done = fit.job_paths(
        data_dir=data_dir, job=fit.Job(arm="SG-0", world="W2", fold=core.MDE_FOLD, seed=1)
    )["done"]
    if path.exists() or not done.exists():
        return
    diff = drv.seed_difference_points(data_dir=data_dir, hidden_ids=hidden_ids)
    core.write_mde_file(path=path, seed_difference_points=diff)


def hidden_series_ids() -> dict[str, int]:
    """Return experiment 87's `time_series_id` of D5 and D10 (the id that the prediction files carry)."""
    import e00087_common as common  # ty: ignore[unresolved-import]

    _, ids = site_ids(table=common.series_table().to_dicts())
    return {s: ids[s] for s in core.HIDDEN_PAIR}


def run_phase(*, kind: str, jobs: list[fit.Job], data_dir: Path, device: str) -> None:
    """Run the fit or the predict phase of `jobs` in this process.

    A correctness failure (the holdout guard, the no-two-worlds guard, an assertion) is recorded in
    `correctness.json` before it is raised again, because verdict row 1 reads that file.
    """
    import e00721_driver as drv
    import e00721_jobs as jobs_mod
    import e00721_real as real
    import torch  # ty: ignore[unresolved-import]

    phase = PHASES[kind]
    os.environ["E00667_PHASE"] = phase
    e00721_paths.pin_threads()
    if device.startswith("cuda"):
        fit.limit_gpu_memory(torch=torch)
    inputs = jobs_mod.WorldInputs(
        make_world=real.make_world_builder(data_dir=data_dir, device=device, phase=phase),
        device=device,
    )
    for job in jobs:
        try:
            if kind == "fit":
                # the tallies of every process are read from the stored curve files before each fit
                budget = fit.Budget.from_files(data_dir=data_dir)
                fit.fit_job(
                    job=job, inputs=inputs, data_dir=data_dir, budget=budget, continuation=True
                )
            else:
                fit.predict_job(job=job, inputs=inputs, data_dir=data_dir)
                write_mde_if_ready(data_dir=data_dir, hidden_ids=hidden_series_ids())
        except (core.CorrectnessFailureError, AssertionError) as error:
            drv.record_correctness_failure(
                data_dir=data_dir, name=type(error).__name__, detail=str(error)[:200]
            )
            raise


def run_stage0(*, data_dir: Path) -> dict:
    """Run Stage 0 (flag test, reachability and noise checks, the stage of the sign head) and write its files."""
    import e00721_stage0 as stage0

    return stage0.run_stage0(data_dir=data_dir)


def run_probe(*, data_dir: Path, device: str, folds: tuple[str, ...]) -> dict:
    """Run the capacity probes of the hidden pair for `folds` (one fold month per process)."""
    import e00721_probe as probe

    return probe.run_probes(data_dir=data_dir, device=device, folds=folds)


def run_quicklook(*, data_dir: Path) -> dict:
    """Score the quick look (`W2`, fold 2026-05, seed 0) and write `quicklook.json`."""
    import e00721_quicklook as quick

    result = quick.quick_look(data_dir=data_dir, hidden_ids=hidden_series_ids())
    fit.atomic_write_json(path=fit.data_root(data_dir=data_dir) / "quicklook.json", body=result)
    print(quick.format_quick_look(result=result))
    return result


def run_score(*, data_dir: Path, n_resamples: int) -> dict:
    """Score every stored fit and write the summary and report.

    The MDE file is not written here: the predict phase writes it at Stage 5. The power check of
    G3 runs first.
    """
    import e00087_common as common  # ty: ignore[unresolved-import]
    import e00721_driver as drv
    import e00721_report as rep

    root = fit.data_root(data_dir=data_dir)
    sites, ids = site_ids(table=common.series_table().to_dicts())
    hidden = {s: ids[s] for s in core.HIDDEN_PAIR}
    mva = [ids[s] for s in core.MVA_ONLY]
    power = drv.power_check(m1_dir=data_dir / M1_FOLDER, folds=core.FOLDS, series_ids=hidden)
    del sites
    stage0_path = root / "stage0.json"
    mde_path = root / "mde.json"
    summary = drv.score_all(
        data_dir=data_dir, folds=core.FOLDS, seeds=(0, 1), hidden_ids=hidden, mva_ids=mva, n_resamples=n_resamples,
        stage0=json.loads(stage0_path.read_text()) if stage0_path.exists() else None,
        mde_path=mde_path, power=power,
    )  # fmt: skip
    drv.write_summary(summary=summary, path=root / "summary.json")
    (root / "report.md").write_text(
        rep.format_report(summary=summary, mde=core.read_mde_file(path=mde_path))
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:  # noqa: PLR0911
    """Run one subcommand."""
    e00721_paths.pin_threads()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in PHASES:
        p = sub.add_parser(name)
        p.add_argument("--wave", type=int)
        p.add_argument("--arm")
        p.add_argument("--world")
        p.add_argument("--fold")
        p.add_argument("--seed", type=int)
        p.add_argument("--device", default="cuda")
        p.add_argument(
            "--only-fold",
            help="internal: run only the jobs of this pipeline fold month in this process",
        )
        if name == "predict":
            p.add_argument("--capacity-scale", type=float, choices=(1.0, *core.CAPACITY_SENSITIVITY), default=1.0,
                           help="prediction-only capacity sensitivity: 0.7 or 1.3 times the oracle capacity")  # fmt: skip
    for name in ("probe",):
        sp = sub.add_parser(name)
        sp.add_argument("--device", default="cuda")
        sp.add_argument("--only-fold", help="internal: probe only this fold month in this process")
    sub.add_parser("stage0")
    sub.add_parser("quicklook")
    q = sub.add_parser("score")
    q.add_argument("--n-resamples", type=int, default=core.N_RESAMPLES)
    args = parser.parse_args(argv)
    data_dir = Path(os.environ["DATA_PATH_INTERNAL"])
    if args.command == "stage0":
        run_stage0(data_dir=data_dir)
        return 0
    if args.command == "probe":
        if args.only_fold is None:
            return run_in_processes(
                argv=list(argv if argv is not None else sys.argv[1:]),
                groups={key: [] for key in core.FOLDS},
            )
        run_probe(data_dir=data_dir, device=args.device, folds=(args.only_fold,))
        return 0
    if args.command == "quicklook":
        run_quicklook(data_dir=data_dir)
        return 0
    if args.command == "score":
        run_score(data_dir=data_dir, n_resamples=args.n_resamples)
        return 0
    jobs = jobs_of(
        wave=args.wave,
        arm=args.arm,
        world=args.world,
        fold=args.fold,
        seed=args.seed,
        capacity_scale=getattr(args, "capacity_scale", 1.0),
    )
    groups = group_by_pipeline_fold(jobs=jobs)
    if args.only_fold is None and len(groups) > 1:
        return run_in_processes(
            argv=list(argv if argv is not None else sys.argv[1:]), groups=groups
        )
    if args.only_fold is not None:
        jobs = groups.get(args.only_fold, [])
    run_phase(kind=args.command, jobs=jobs, data_dir=data_dir, device=args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
