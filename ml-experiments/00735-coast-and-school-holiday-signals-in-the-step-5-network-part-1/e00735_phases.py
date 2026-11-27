# ruff: noqa: E501, PLR0912, PLR0915
"""Experiment 735: run the phases of the experiment behind review markers, with a GPU-hours stop.

README. This script runs the phases of experiment 735 by calling `e00735_run.sh` (the GPU jobs),
`e00735_tests_known_answer.py`, `e00735_gallery.py`, `e00735_regime.py` (CPU), and `e00735_score.py`
(CPU). It needs no PYTHONPATH itself, because the shell driver and the child commands source
`ml-experiments/00524*/env.sh`; set `DATA_PATH_INTERNAL`.

A phase starts only when the file `<root>/review_markers/<phase>` exists, so that a reviewer has
signed off on the code first (the manager or the reviewer creates the marker; this script never
does). The phases, in order:

- `columns` (CPU): the known-answer tests, the column gallery with the training-row and weather
  counts, and the per-year regime check; no fit;
- `smoke` (GPU): every arm at one refit month (2026-01), seed 0, one epoch per stage, then a forecast
  of the stage-3 validation rows only, scored by `e00735_score.py --smoke`; output goes to
  `.claude/scratch/e735_smoke/`, never to the experiment's data folder;
- `gate` (GPU): R0 at the gate month (2026-10), seeds 0 to 2, compared with experiment 677's stored
  `k1_5w_sp1` forecasts by `e00735_score.py --gate`;
- `fit` (GPU): every arm at every month (stage 2, stage 3, prediction pass), one job at a time;
- `score` (CPU): `e00735_score.py --stage all`.

Every job's stage times go to `<root>/logs/phase_times.csv` (arm, month, phase, device, start, end).
Before each GPU job this script sums the device-`cuda` rows of that file and of the smoke file into
GPU-process-hours, prints the running total, and refuses to start a job when the total reaches the
stop line (5.5 hours) or when the total plus the mean job time would pass it. A failed job writes a
record with the exception message and the traceback to `<root>/failures/` and ends the chain with
exit code 20. The stop line ends the chain with exit code 21, a failed replay gate with 22, a busy GPU
with 75 (nothing failed: run the same command again later), and a missing marker or a bad argument
with 2. The full list is at the top of `e00735_run.sh`. The `smoke`, `gate`, and `fit` phases compare
the hashes of experiment 677's stage-1 files with the recorded ones before the first job, and the
`fit` phase compares them again after the last job.

Usage: python e00735_phases.py PHASE [--arms S SP ...] [--months 2025-11 ...] [--dry-run]
[--accept-over-budget]. Environment: `CORES` (default 22,23), `CAP` (memory cap per process, default
12G), `SEEDS`, `R735` (the data folder, which must contain 00735).
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

HERE: Final = Path(__file__).resolve().parent
REPO: Final = HERE.parents[1]
NAME: Final = "00735-coast-and-school-holiday-signals-in-the-step-5-network-part-1"
ARMS: Final = ("R0", "S", "SP", "SKW", "SKWP", "SE", "SEP")
NEW_ARMS: Final = ARMS[1:]
MONTHS: Final = (
    "2025-11", "2025-12", "2026-01", "2026-02", "2026-03", "2026-04",
    "2026-05", "2026-06", "2026-07", "2026-08", "2026-09", "2026-10",
)  # fmt: skip
PHASES: Final = ("columns", "smoke", "gate", "fit", "score")
SMOKE_MONTH: Final = "2026-01"
GATE_MONTH: Final = "2026-10"
STOP_HOURS: Final = 5.5
"""The maintainer's line: more than this many GPU-process-hours needs a new request."""
EXIT_FAILED: Final = 20
EXIT_STOP: Final = 21
EXIT_USAGE: Final = 2
EXIT_GATE_FAILED: Final = 22
EXIT_BUSY: Final = 75
"""All exit codes are listed at the top of `e00735_run.sh`."""
SMOKE_ROOT: Final = REPO / ".claude" / "scratch" / "e735_smoke" / "00735-smoke"
DROP_ORDER: Final = ("SEP", "SE")
"""If the measured time takes the total over the line, SEP is dropped first, then SE."""
SEEDS: Final = os.environ.get("SEEDS", "0 1 2").split()
JOB_STAGES: Final = ("step2", "fit", "pfit")


def root_folder() -> Path:
    """Return experiment 735's data folder and refuse any other."""
    data = os.environ.get("DATA_PATH_INTERNAL")
    assert data, "set DATA_PATH_INTERNAL to the private data folder"
    root = Path(os.environ.get("R735", Path(data) / "ml-experiments" / NAME))
    assert "00735" in str(root), root
    assert "00677" not in str(root), root
    return root


def gpu_hours(*, csv_paths: list[Path]) -> float:
    """Return the GPU-process-hours in the time logs: the sum of end minus start over rows with device cuda."""
    total = 0.0
    for path in csv_paths:
        if not path.exists():
            continue
        with path.open() as handle:
            for row in csv.DictReader(handle):
                if row["device"] != "cuda":
                    continue
                start = datetime.fromisoformat(row["start_utc"])
                end = datetime.fromisoformat(row["end_utc"])
                total += (end - start).total_seconds() / 3600.0
    return total


def job_minutes(*, path: Path, arm: str, month: str) -> float:
    """Return the minutes that the three stages of one arm-month took, from the time log."""
    total = 0.0
    if not path.exists():
        return total
    with path.open() as handle:
        for row in csv.DictReader(handle):
            if row["arm"] == arm and row["month"] == month and row["phase"] in JOB_STAGES:
                start = datetime.fromisoformat(row["start_utc"])
                end = datetime.fromisoformat(row["end_utc"])
                total += (end - start).total_seconds() / 60.0
    return total


def log_cpu_phase(*, root: Path, phase: str, start: datetime, end: datetime) -> None:
    """Append one device-`cpu` row to the time log."""
    path = root / "logs" / "phase_times.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a") as handle:
        if new:
            handle.write("arm,month,phase,device,start_utc,end_utc\n")
        handle.write(f"-,-,{phase},cpu,{start:%Y-%m-%dT%H:%M:%SZ},{end:%Y-%m-%dT%H:%M:%SZ}\n")


def failure_record(*, root: Path, arm: str, month: str, stage: str, code: int) -> Path:
    """Write the exception message and traceback of a failed stage to `<root>/failures/` and return its path."""
    log = root / "logs" / f"{month}_{arm}_{stage}.log"
    smoke_log = SMOKE_ROOT / "logs" / f"smoke_{month}_{arm}_{stage}.log"
    source = log if log.exists() else smoke_log
    text = source.read_text(errors="replace") if source.exists() else ""
    at = text.rfind("Traceback (most recent call last)")
    trace = text[at:] if at >= 0 else text[-4000:]
    message = trace.strip().splitlines()[-1] if trace.strip() else "no log"
    path = root / "failures" / f"{month}_{arm}_{stage}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "arm": arm, "month": month, "stage": stage, "exit_code": code,
                "when_utc": datetime.now(UTC).isoformat(), "exception_message": message,
                "traceback": trace, "log": str(source),
            },
            indent=1,
        )
    )  # fmt: skip
    return path


def run_job(*, arm: str, month: str, stage: str, root: Path, smoke: bool, cores: str) -> int:
    """Run one arm-month stage through `e00735_run.sh` and return its exit code."""
    env = {**os.environ, "CORES": cores, "R735": str(root), "SMOKE": "1" if smoke else "0"}
    if smoke:
        env["SEEDS"] = "0"
    cmd = [str(HERE / "e00735_run.sh"), arm, month, stage]
    print("start", cmd, "cores", cores, flush=True)
    return subprocess.call(cmd, env=env)


def run_cpu(*, cmd: list[str], root: Path, cores: str, cap: str) -> int:
    """Run a CPU command in a capped scope on the slot's cores, with one BLAS thread."""
    scoped = [
        "systemd-run", "--user", "--scope", "-q",
        "-p", f"MemoryMax={cap}", "-p", "MemorySwapMax=0", "-p", "OOMPolicy=continue",
        "bash", "-c",
        f'cd {REPO} && source "$(ls -d ml-experiments/00524*)/env.sh" && '
        f"OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 POLARS_MAX_THREADS=2 "
        f"NGED_S3_BUCKET_URL=x NGED_S3_BUCKET_ACCESS_KEY=x NGED_S3_BUCKET_SECRET=x "
        f"R735={root} taskset -c {cores} {REPO}/.venv/bin/python " + " ".join(cmd),
    ]  # fmt: skip
    print("cpu:", " ".join(cmd), flush=True)
    return subprocess.call(scoped, env=os.environ)


def hashes_unchanged(*, root: Path, cores: str, cap: str) -> bool:
    """Return True when the files of experiment 677's stage 1 still have the recorded hashes."""
    cmd = [str(HERE / "e00735_tests_known_answer.py"), "--hashes-only"]
    code = run_cpu(cmd=cmd, root=root, cores=cores, cap=cap)
    if code != 0:
        print(
            "experiment 677's stage-1 files differ from the recorded hashes (or the check failed)",
            flush=True,
        )
    return code == 0


def main() -> int:  # noqa: PLR0911
    """Check the review marker, then run the phase's jobs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=PHASES)
    parser.add_argument("--arms", nargs="+", default=None, choices=ARMS)
    parser.add_argument("--months", nargs="+", default=None, choices=MONTHS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--accept-over-budget", action="store_true")
    args = parser.parse_args()
    root = root_folder()
    cores = os.environ.get("CORES", "22,23")
    cap = os.environ.get("CAP", "12G")
    marker = root / "review_markers" / args.phase
    if not marker.exists() and not args.dry_run:
        print(f"refusing to start phase {args.phase}: no review marker {marker}")
        return EXIT_USAGE
    csvs = [root / "logs" / "phase_times.csv", SMOKE_ROOT / "logs" / "phase_times.csv"]
    if args.phase in {"columns", "score"}:
        start = datetime.now(UTC)
        if args.phase == "columns":
            cmds = [
                [str(HERE / "e00735_tests_known_answer.py")],
                [str(HERE / "e00735_gallery.py")],
                [str(HERE / "e00735_regime.py")],
            ]
        else:
            gate = root / "scores" / "gate.json"
            gate_passed = gate.exists() and bool(json.loads(gate.read_text()).get("passed"))
            cmds = [
                [
                    str(HERE / "e00735_score.py"),
                    "--stage",
                    "all",
                    *(["--stored"] if gate_passed else []),
                ]
            ]
            print(
                "replay gate passed, scoring with experiment 677's stored rows as R0:"
                if gate_passed
                else "no passed replay gate: R0 comes from this experiment's own fits",
                flush=True,
            )
        if args.dry_run:
            print("cpu phase:", cmds)
            return 0
        for cmd in cmds:
            code = run_cpu(cmd=cmd, root=root, cores=cores, cap=cap)
            if code != 0:
                print("FAILED", cmd, code)
                return EXIT_FAILED
        log_cpu_phase(root=root, phase=args.phase, start=start, end=datetime.now(UTC))
        return 0
    if args.phase == "smoke":
        jobs = [(a, SMOKE_MONTH) for a in (args.arms or list(ARMS))]
        stages = ("step2", "fit", "valpred")
        smoke, work_root = True, SMOKE_ROOT
    elif args.phase == "gate":
        jobs = [("R0", GATE_MONTH)]
        stages = ("step2", "fit", "pfit")
        smoke, work_root = False, root
    else:
        arms = args.arms or list(NEW_ARMS)
        months = args.months or list(MONTHS)
        jobs = [(a, m) for m in months for a in arms]  # month-major, so a stop leaves whole months
        stages = ("step2", "fit", "pfit")
        smoke, work_root = False, root
    if args.phase == "fit":
        gate_minutes = job_minutes(path=csvs[0], arm="R0", month=GATE_MONTH)
        spent = gpu_hours(csv_paths=csvs)
        if gate_minutes > 0:
            projected = spent + gate_minutes / 60.0 * len([j for j in jobs if not _done(root, *j)])
            print(
                f"gate measured {gate_minutes:.2f} min per arm-month; spent {spent:.2f} GPU-process-hours;"
                f" projected total {projected:.2f} (stop line {STOP_HOURS})"
            )
            if projected > STOP_HOURS and not args.accept_over_budget:
                print(
                    f"over the line: drop {' then '.join(DROP_ORDER)} (never SP or SKWP while their arm runs) and ask"
                )
                return EXIT_STOP
        else:
            print("no gate time in the log: the projection cannot be checked", flush=True)
            if not args.accept_over_budget and not args.dry_run:
                return EXIT_STOP
    if not args.dry_run and not hashes_unchanged(root=root, cores=cores, cap=cap):
        return EXIT_FAILED
    print(f"{args.phase}: {len(jobs)} jobs, 1 GPU process at a time, cores {cores}", flush=True)
    print(
        f"fits starting: {len(jobs)} arm-months x {len(SEEDS)} seeds, memory cap {cap}, cores {cores},"
        f" spent {gpu_hours(csv_paths=csvs):.2f} GPU-process-hours",
        flush=True,
    )  # fmt: skip
    for arm, month in jobs:
        spent = gpu_hours(csv_paths=csvs)
        done_minutes = [job_minutes(path=csvs[0], arm=a, month=m) for a, m in jobs]
        mean_minutes = sum(done_minutes) / max(1, sum(1 for x in done_minutes if x > 0))
        if (
            mean_minutes == 0.0
        ):  # no job of this phase has finished yet: use the gate's measured time
            mean_minutes = job_minutes(path=csvs[0], arm="R0", month=GATE_MONTH)
        print(f"running total {spent:.3f} GPU-process-hours (stop line {STOP_HOURS})", flush=True)
        if spent >= STOP_HOURS or spent + mean_minutes / 60.0 > STOP_HOURS:
            print(f"STOP: {spent:.3f} GPU-process-hours, the next job would pass {STOP_HOURS}: ask")
            return EXIT_STOP
        if args.dry_run:
            print("dry run:", arm, month, stages)
            continue
        for stage in stages:
            code = run_job(
                arm=arm, month=month, stage=stage, root=work_root, smoke=smoke, cores=cores
            )
            if code == EXIT_BUSY:
                print("GPU busy: nothing failed; run the same command again later", flush=True)
                return EXIT_BUSY
            if code != 0:
                record = failure_record(
                    root=work_root, arm=arm, month=month, stage=stage, code=code
                )
                print("FAILED", arm, month, stage, code, "record", record, flush=True)
                return EXIT_FAILED
    if args.dry_run:
        return 0
    spent = gpu_hours(csv_paths=csvs)
    print(f"{args.phase} done: {spent:.3f} GPU-process-hours in total", flush=True)
    if args.phase == "smoke":
        return run_cpu(
            cmd=[str(HERE / "e00735_score.py"), "--smoke"], root=root, cores=cores, cap=cap
        )
    if args.phase == "gate":
        return run_cpu(
            cmd=[str(HERE / "e00735_score.py"), "--gate"], root=root, cores=cores, cap=cap
        )
    if args.phase == "fit" and not hashes_unchanged(root=root, cores=cores, cap=cap):
        return EXIT_FAILED
    return 0


def _done(root: Path, arm: str, month: str) -> bool:
    """Return True when the arm-month's three seeds already have a prediction file."""
    base = root / "arms" / f"k1_{arm}_sp1" / "work" / "fits" / "P5"
    return all((base / f"{month}_d0_s{s}.parquet").exists() for s in SEEDS)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001  # a crash of this script itself is a failed chain, never the launcher's code
        traceback.print_exc()
        sys.exit(EXIT_FAILED)
