# ruff: noqa: E501, PLR0912, PLR0915
"""Experiment 735 part 2: run the phases behind review markers, with a GPU-hours stop.

README. This script runs the phases of experiment 735 part 2 by calling `e00735b_run.sh` (the GPU
jobs of the four new arms), part 1's `e00735_run.sh` (the gate month's R0 job), `e00735b_tests.py`,
`e00735b_check_stages.py`, and `e00735b_score.py` (CPU). It follows part 1's `e00735_phases.py`. It
needs no PYTHONPATH itself, because the shell driver and the child commands source
`ml-experiments/00524*/env.sh`; set `DATA_PATH_INTERNAL`.

A phase starts only when the file `<root>/review_markers/<phase>` exists, so that a reviewer has
signed off on the code first (the manager or the reviewer creates the marker; this script never
does). The phases, in order:

- `columns` (CPU): the known-answer tests and the "every stage" gate of the 8 columns; no fit;
- `smoke` (GPU): every new arm at one refit month (2026-01), seed 0, one epoch per stage, then a
  forecast of the stage-3 validation rows only, scored by `e00735b_score.py --smoke`; output goes to
  `.claude/scratch/e735b_smoke/`, never to the experiment's data folder;
- `gate` (GPU): part 1's R0 job (experiment 677's stage-1 weights, no new column) at the gate month
  (2026-03), seeds 0 to 2, compared with experiment 677's stored `k1_5w_sp1` forecasts by
  `e00735b_score.py --gate`. The new arms do not run before the gate has passed;
- `fit` (GPU): every new arm at every refit month (stages 1, 2, 3, and the prediction pass), one
  job at a time, month by month;
- `score` (CPU): `e00735b_score.py --stage all`.

Every job's stage times go to `<root>/logs/phase_times.csv` (arm, month, phase, device, start, end).
Before each GPU job this script sums the device-`cuda` rows of that file and of the smoke file into
GPU-process-hours, prints the running total, and refuses to start a job when the total reaches the
stop line (6.5 hours, agreed with the manager) or when the total plus the mean job time would pass
it. A failed job writes a record with the exception message and the traceback to `<root>/failures/`
and ends the chain with exit code 20. The stop line ends the chain with exit code 21, a failed replay
gate with 22, a busy GPU with 75 (nothing failed: run the same command again later), and a missing
marker or a bad argument with 2. The full list is at the top of `e00735b_run.sh`.

If the measured time takes the projection over the line, the plan drops SKE-CP from the months from
2026-06 backwards, in that order; this script only stops and asks.

Usage: python e00735b_phases.py PHASE [--arms R1 SKE ...] [--months 2025-11 ...] [--dry-run]
[--accept-over-budget]. Environment: `CORES` (default 22,23), `CAP` (memory cap per process, default
12G), `SEEDS`, `R735` (the data folder, which must contain 00735).
"""

import argparse
import ast
import csv
import json
import os
import re
import subprocess
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

HERE: Final = Path(__file__).resolve().parent
REPO: Final = HERE.parents[1]
NAME: Final = "00735-school-columns-in-the-date-time-encoder-part-2"
PART_1_RUN: Final = next(REPO.glob("ml-experiments/00735-coast*/e00735_run.sh"))
NEW_ARMS: Final = ("R1", "SKE", "SKE-S21", "SKE-CP")
MONTHS: Final = (
    "2025-11", "2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06",
)  # fmt: skip
PHASES: Final = ("columns", "smoke", "gate", "fit", "score")
SMOKE_MONTH: Final = "2026-01"
GATE_MONTH: Final = "2026-03"
GATE_ARM: Final = "R0"
STOP_HOURS: Final = min(float(os.environ.get("STOP_HOURS", "6.5")), 6.5)
"""The line, in GPU-process-hours, past which the runner stops and asks (6.5, agreed with the manager).

The environment variable `STOP_HOURS` may lower the line and cannot raise it.
"""
PLANNED_MINUTES: Final = 9.5
"""The plan's estimate of the minutes one arm-month takes (three stages and the prediction pass), used until a job has been measured."""
EXIT_FAILED: Final = 20
EXIT_STOP: Final = 21
EXIT_USAGE: Final = 2
EXIT_GATE_FAILED: Final = 22
EXIT_BUSY: Final = 75
"""All exit codes are listed at the top of `e00735b_run.sh`."""
SMOKE_ROOT: Final = REPO / ".claude" / "scratch" / "e735b_smoke" / "00735-smoke"
SEEDS: Final = os.environ.get("SEEDS", "0 1 2").split()
JOB_STAGES: Final = ("pretrain", "step2", "fit", "pfit")
GATE_STAGES: Final = ("step2", "fit", "pfit")
"""The gate's R0 job reads experiment 677's stage 1, so it has no pre-training stage."""


def root_folder() -> Path:
    """Return experiment 735 part 2's data folder and refuse any other."""
    data = os.environ.get("DATA_PATH_INTERNAL")
    assert data, "set DATA_PATH_INTERNAL to the private data folder"
    root = Path(os.environ.get("R735", Path(data) / "ml-experiments" / NAME))
    assert "school-columns-in-the-date-time-encoder-part-2" in str(root) or "00735-smoke" in str(
        root
    ), root
    assert "coast-and-school" not in str(root), root
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
    """Return the minutes that the stages of one arm-month took, from the time log."""
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


PRETRAIN_LINE: Final = re.compile(
    r"pretrain (?P<arm>\S+) (?P<month>\d{4}-\d{2}) s(?P<seed>\d+): (?P<epochs>\d+) epochs, "
    r"val NLL (?P<nll>[0-9.]+), \d+ s, dropout (?P<dropout>\{.*\})"
)
"""A line of a stage-1 log, from experiment 677's `e00677_k1.py pretrain` or this experiment's runner."""
NLL_TOLERANCE: Final = 1e-3
NLL_TOLERANCE_ONE_EPOCH: Final = 2e-3


def parse_pretrain_log(*, text: str) -> dict[int, dict]:
    """Return, by seed, the epochs, validation NLL, and dropout-share dictionary of each stage-1 line in a log."""
    out: dict[int, dict] = {}
    for line in text.splitlines():
        found = PRETRAIN_LINE.search(line)
        if found:
            out[int(found["seed"])] = {
                "epochs": int(found["epochs"]),
                "nll": float(found["nll"]),
                "dropout": ast.literal_eval(found["dropout"]),
            }
    return out


def compare_pretrain(
    *, mine: dict[int, dict], reference: dict[int, dict], seeds: list[int]
) -> tuple[str, list[str]]:
    """Compare R1's stage 1 with experiment 677's, seed by seed.

    Returns `ok`, `flag`, or `mismatch`, and the messages. The dropout shares must be equal; the
    epochs and the validation NLL must agree to 1e-3. A difference of one epoch with NLL within 2e-3
    is flagged and does not stop the phase.
    """
    status, messages = "ok", []
    for seed in seeds:
        if seed not in mine or seed not in reference:
            return "mismatch", [
                f"seed {seed}: missing from {'R1' if seed not in mine else 'experiment 677'}'s log"
            ]
        a, b = mine[seed], reference[seed]
        gap = abs(a["nll"] - b["nll"])
        if a["dropout"] != b["dropout"]:
            return "mismatch", [
                f"seed {seed}: dropout shares differ: {a['dropout']} against {b['dropout']}"
            ]
        if a["epochs"] == b["epochs"] and gap <= NLL_TOLERANCE:
            continue
        if abs(a["epochs"] - b["epochs"]) == 1 and gap <= NLL_TOLERANCE_ONE_EPOCH:
            status = "flag"
            messages.append(
                f"seed {seed}: epochs {a['epochs']} against {b['epochs']}, val NLL {a['nll']} against {b['nll']}"
            )
            continue
        return "mismatch", [
            f"seed {seed}: epochs {a['epochs']} against {b['epochs']}, val NLL {a['nll']} against {b['nll']}"
        ]
    return status, messages


def r1_reproduction_check(*, root: Path, month: str) -> int:
    """Compare R1's stage 1 of a month with experiment 677's log; return 0, or `EXIT_FAILED` after writing the failure record."""
    reference_logs = sorted(
        (Path(os.environ["DATA_PATH_INTERNAL"]) / "ml-experiments").glob(
            f"00677-*/logs/{month}_5w_pretrain.log"
        )
    )
    if not reference_logs:
        print(
            f"R1 reproduction check skipped: experiment 677 has no stage-1 log for {month}",
            flush=True,
        )
        return 0
    path = root / "logs" / f"{month}_R1_pretrain.log"
    try:
        assert path.exists(), f"R1's stage-1 log is missing: {path}"
        status, messages = compare_pretrain(
            mine=parse_pretrain_log(text=path.read_text(errors="replace")),
            reference=parse_pretrain_log(text=reference_logs[0].read_text(errors="replace")),
            seeds=[int(s) for s in SEEDS],
        )
        assert status != "mismatch", (
            f"R1's stage 1 does not reproduce experiment 677's for {month}: {messages}"
        )
    except AssertionError as error:
        record = root / "failures" / f"{month}_R1_reproduction.json"
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(
            json.dumps(
                {"arm": "R1", "month": month, "stage": "reproduction", "exit_code": EXIT_FAILED,
                 "when_utc": datetime.now(UTC).isoformat(), "exception_message": f"AssertionError: {error}",
                 "traceback": traceback.format_exc(), "log": str(path)},
                indent=1,
            )
        )  # fmt: skip
        print("R1 REPRODUCTION FAILED:", error, "record", record, flush=True)
        return EXIT_FAILED
    print(f"R1 reproduction check {status} for {month}", *messages, flush=True)
    return 0


def run_job(*, arm: str, month: str, stage: str, root: Path, smoke: bool, cores: str) -> int:
    """Run one arm-month stage through the shell driver and return its exit code.

    The gate's R0 job uses part 1's driver (it reads experiment 677's stage 1); every new arm uses
    this experiment's driver.
    """
    env = {**os.environ, "CORES": cores, "R735": str(root), "SMOKE": "1" if smoke else "0"}
    if smoke:
        env["SEEDS"] = "0"
    driver = PART_1_RUN if arm == GATE_ARM else HERE / "e00735b_run.sh"
    cmd = [str(driver), arm, month, stage]
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


def gate_passed(*, root: Path) -> bool:
    """Return True when `scores/gate.json` records a passed replay gate."""
    gate = root / "scores" / "gate.json"
    return gate.exists() and bool(json.loads(gate.read_text()).get("passed"))


def main() -> int:  # noqa: PLR0911
    """Check the review marker, then run the phase's jobs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=PHASES)
    parser.add_argument("--arms", nargs="+", default=None, choices=NEW_ARMS)
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
            cmds = [[str(HERE / "e00735b_tests.py")], [str(HERE / "e00735b_check_stages.py")]]
        else:
            passed = gate_passed(root=root)
            cmds = [
                [
                    str(HERE / "e00735b_score.py"),
                    "--stage",
                    "all",
                    *(["--stored"] if passed else []),
                ]
            ]
            print(
                "replay gate passed, scoring with experiment 677's stored rows as R0:"
                if passed
                else "no passed replay gate: R0 is not scored",
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
        jobs = [(a, SMOKE_MONTH) for a in (args.arms or list(NEW_ARMS))]
        stages = ("pretrain", "step2", "fit", "valpred")
        smoke, work_root = True, SMOKE_ROOT
    elif args.phase == "gate":
        jobs = [(GATE_ARM, GATE_MONTH)]
        stages = GATE_STAGES
        smoke, work_root = False, root
    else:
        if not gate_passed(root=root) and not args.dry_run:
            print("refusing to start phase fit: the replay gate has not passed")
            return EXIT_USAGE
        arms = args.arms or list(NEW_ARMS)
        months = args.months or list(MONTHS)
        arms = sorted(
            arms, key=lambda a: a != "R1"
        )  # R1 first in every month: its stage 1 is the reproduction check
        jobs = [(a, m) for m in months for a in arms]  # month-major, so a stop leaves whole months
        stages = JOB_STAGES
        smoke, work_root = False, root
    if args.phase == "fit":
        spent = gpu_hours(csv_paths=csvs)
        todo = len([j for j in jobs if not _done(root, *j)])
        projected = spent + PLANNED_MINUTES / 60.0 * todo
        print(
            f"planned {PLANNED_MINUTES} min per arm-month; spent {spent:.2f} GPU-process-hours;"
            f" projected total {projected:.2f} (stop line {STOP_HOURS})"
        )
        if projected > STOP_HOURS and not args.accept_over_budget:
            print(
                "over the line: the plan drops SKE-CP from the months from 2026-06 backwards, and asks"
            )
            return EXIT_STOP
    print(f"{args.phase}: {len(jobs)} jobs, 1 GPU process at a time, cores {cores}", flush=True)
    print(
        f"fits starting: {len(jobs)} arm-months x {len(SEEDS)} seeds, memory cap {cap}, cores {cores},"
        f" spent {gpu_hours(csv_paths=csvs):.2f} GPU-process-hours",
        flush=True,
    )  # fmt: skip
    checked_months: set[str] = set()
    for arm, month in jobs:
        spent = gpu_hours(csv_paths=csvs)
        done_minutes = [job_minutes(path=csvs[0], arm=a, month=m) for a, m in jobs]
        measured = [x for x in done_minutes if x > 0]
        mean_minutes = sum(measured) / len(measured) if measured else PLANNED_MINUTES
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
            if (
                args.phase == "fit"
                and arm == "R1"
                and stage == "pretrain"
                and month not in checked_months
            ):
                checked_months.add(month)
                if r1_reproduction_check(root=work_root, month=month) != 0:
                    return EXIT_FAILED  # before step 2 of R1 and before any SKE job
    if args.dry_run:
        return 0
    spent = gpu_hours(csv_paths=csvs)
    print(f"{args.phase} done: {spent:.3f} GPU-process-hours in total", flush=True)
    if args.phase == "smoke":
        return run_cpu(
            cmd=[str(HERE / "e00735b_score.py"), "--smoke"], root=root, cores=cores, cap=cap
        )
    if args.phase == "gate":
        return run_cpu(
            cmd=[str(HERE / "e00735b_score.py"), "--gate"], root=root, cores=cores, cap=cap
        )
    return 0


def _done(root: Path, arm: str, month: str) -> bool:
    """Return True when the arm-month's seeds already have a prediction file."""
    base = root / "arms" / f"k1_{arm}_sp1" / "work" / "fits" / "P5"
    return all((base / f"{month}_d0_s{s}.parquet").exists() for s in SEEDS)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001  # a crash of this script itself is a failed chain, never the launcher's code
        traceback.print_exc()
        sys.exit(EXIT_FAILED)
