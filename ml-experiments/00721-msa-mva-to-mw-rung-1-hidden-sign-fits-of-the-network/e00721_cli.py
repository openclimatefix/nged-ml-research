# ruff: noqa: E501
"""Experiment 721: the command-line entry of the real run (fit, predict, and score as separate processes).

Each subcommand is one process, started through `run.sh --go e00721_cli.py ...`:

- `fit` runs the fit phase of a wave (or of one job) with `E00667_PHASE=train`: the world is built
  from the fold's training pipeline, so nothing at or after the fold's cutoff is read.
- `predict` runs the predict phase of the same jobs with `E00667_PHASE=predict`, in a new process,
  because experiment 713's pipeline is built for one phase per process.
- `score` writes the MDE file (from the seed difference of `SG-0` on fold 2026-05), runs the power check
  of G3 over experiment 713's stored `M1` seeds, scores the stored fits, and writes `summary.json` and
  `report.md`.

Examples:
    run.sh --go e00721_cli.py fit --wave 1 --device cuda
    run.sh --go e00721_cli.py predict --wave 1 --device cuda
    run.sh --go e00721_cli.py score
"""

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import e00721_paths  # noqa: F401

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
    *, wave: int | None, arm: str | None, world: str | None, fold: str | None, seed: int | None
) -> list[fit.Job]:
    """Return the jobs that the arguments name: a wave, or one job (arm, world, fold, and seed all given)."""
    if wave is not None:
        return fit.wave_jobs(wave=wave)
    if None in (arm, world, fold, seed):
        msg = "name a --wave, or all of --arm, --world, --fold, and --seed"
        raise ValueError(msg)
    assert arm is not None
    assert world is not None
    assert fold is not None
    assert seed is not None
    return [fit.Job(arm=arm, world=world, fold=fold, seed=seed)]


def spent_hours(*, data_dir: Path) -> float:
    """Return the GPU-process-hours that the stored fits used (the sum of `hours` over the curve files)."""
    folder = fit.data_root(data_dir=data_dir) / "fits"
    return sum(
        float(json.loads(p.read_text()).get("hours", 0.0))
        for p in sorted(folder.glob("*.curve.json"))
    )


def site_ids(*, table: Sequence[dict]) -> tuple[list[str], dict[str, int]]:
    """Return the 16 Primary labels in panel order and the `time_series_id` of every Primary label.

    Args:
        table: Rows with `site`, `time_series_id`, and `substation_type` (experiment 87's series table).
    """
    primaries = sorted(
        (r for r in table if r["substation_type"] == "Primary"), key=lambda r: int(r["site"][1:])
    )
    return [r["site"] for r in primaries], {r["site"]: int(r["time_series_id"]) for r in primaries}


def run_phase(*, kind: str, jobs: list[fit.Job], data_dir: Path, device: str) -> None:
    """Run the fit or the predict phase of `jobs` in this process."""
    import e00721_jobs as jobs_mod
    import e00721_real as real
    import torch  # ty: ignore[unresolved-import]

    phase = PHASES[kind]
    os.environ["E00667_PHASE"] = phase
    torch.set_num_threads(1)
    if device.startswith("cuda"):
        fit.limit_gpu_memory(torch=torch)
    inputs = jobs_mod.WorldInputs(
        make_world=real.make_world_builder(data_dir=data_dir, device=device, phase=phase),
        device=device,
    )
    budget = fit.Budget(spent_hours=spent_hours(data_dir=data_dir))
    for job in jobs:
        if kind == "fit":
            fit.fit_job(job=job, inputs=inputs, data_dir=data_dir, budget=budget, continuation=True)
        else:
            fit.predict_job(job=job, inputs=inputs, data_dir=data_dir)


def run_score(*, data_dir: Path, n_resamples: int) -> dict:
    """Write the MDE file, run the power check, score every stored fit, and write the summary and report."""
    import e00087_common as common  # ty: ignore[unresolved-import]
    import e00721_driver as drv
    import e00721_report as rep

    root = fit.data_root(data_dir=data_dir)
    sites, ids = site_ids(table=common.series_table().to_dicts())
    hidden = {s: sites.index(s) for s in core.HIDDEN_PAIR}
    mva = [sites.index(s) for s in core.MVA_ONLY]
    power = drv.power_check(
        m1_dir=data_dir / M1_FOLDER,
        folds=core.FOLDS,
        series_ids={s: ids[s] for s in core.HIDDEN_PAIR},
    )
    stage0_path = root / "stage0.json"
    summary = drv.score_all(
        data_dir=data_dir, folds=core.FOLDS, seeds=(0, 1), hidden_ids=hidden, mva_ids=mva, n_resamples=n_resamples,
        stage0=json.loads(stage0_path.read_text()) if stage0_path.exists() else None, mde_path=root / "mde.json", power=power,
    )  # fmt: skip
    drv.write_summary(summary=summary, path=root / "summary.json")
    (root / "report.md").write_text(
        rep.format_report(summary=summary, mde=core.read_mde_file(path=root / "mde.json"))
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """Run one subcommand."""
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
    q = sub.add_parser("score")
    q.add_argument("--n-resamples", type=int, default=core.N_RESAMPLES)
    args = parser.parse_args(argv)
    data_dir = Path(os.environ["DATA_PATH_INTERNAL"])
    if args.command == "score":
        run_score(data_dir=data_dir, n_resamples=args.n_resamples)
        return 0
    jobs = jobs_of(wave=args.wave, arm=args.arm, world=args.world, fold=args.fold, seed=args.seed)
    run_phase(kind=args.command, jobs=jobs, data_dir=data_dir, device=args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
