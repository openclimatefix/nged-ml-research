"""Experiment 731: run a queue of (arm, month) jobs behind review markers, at most 3 at a time.

README. This script runs the fit, bank, and score steps of experiment 731 by calling
`e00731_run.sh` (which runs `e00731_k1.py` and 677's `e00677_cand.py bank`; relies on 677, 574,
581, 591, 710) and `e00731_score.py`. It needs no PYTHONPATH itself, because the shell driver
sources `ml-experiments/00524*/env.sh`; set `DATA_PATH_INTERNAL` and, for the score phase, source
the same file before running it.

A phase starts only when the file `<root>/review_markers/<phase>` exists, so that a reviewer has
signed off on the code first. The phases are:

- `smoke`: one refit month (default 2026-01) and one seed (0) of every arm, stage 3 and prediction;
- `fit`: every arm at every month, stage 3 then prediction (`e00731_run.sh ARM MONTH all`);
- `bank`: the held-out bank pass of every arm and month;
- `score`: `e00731_score.py` (CPU, one process).

Usage: python e00731_phases.py PHASE [--arms R0 W0 ...] [--months 2025-11 ...] [--parallel 3]
[--dry-run]. Environment: `CORES` (default 22-29, split between the parallel slots), `CAP`
(memory cap per process, default 10G), `SEEDS`.
"""

import argparse
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final

HERE: Final = Path(__file__).resolve().parent
NAME: Final = "00731-step-5-recency-and-injection-ablations-monthly-refits"
ARMS: Final = ("R0", "W0", "W30", "W270", "WP", "IP", "B0")
MONTHS: Final = (
    "2025-11", "2025-12", "2026-01", "2026-02", "2026-03", "2026-04",
    "2026-05", "2026-06", "2026-07", "2026-08", "2026-09", "2026-10",
)  # fmt: skip
PHASES: Final = ("smoke", "fit", "bank", "score")
MAX_PARALLEL: Final = 3
SMOKE_MONTH: Final = "2026-01"


def root_folder() -> Path:
    """Return experiment 731's data folder and refuse any other."""
    data = os.environ.get("DATA_PATH_INTERNAL")
    assert data, "set DATA_PATH_INTERNAL to the private data folder"
    root = Path(os.environ.get("R731", Path(data) / "ml-experiments" / NAME))
    assert "00731" in str(root), root
    assert "00677" not in str(root), root
    return root


def core_slots(*, cores: str, parallel: int) -> list[str]:
    """Split a taskset core list such as `22-29` into `parallel` slices (or keep it whole)."""
    ids: list[int] = []
    for part in cores.split(","):
        lo, _, hi = part.partition("-")
        ids.extend(range(int(lo), int(hi or lo) + 1))
    if len(ids) < parallel:
        return [cores] * parallel
    size = len(ids) // parallel
    return [",".join(map(str, ids[i * size : (i + 1) * size])) for i in range(parallel)]


def main() -> int:
    """Check the review marker, then run the phase's jobs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=PHASES)
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    parser.add_argument("--months", nargs="+", default=None)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    assert 1 <= args.parallel <= MAX_PARALLEL, f"at most {MAX_PARALLEL} concurrent GPU processes"
    root = root_folder()
    marker = root / "review_markers" / args.phase
    if not marker.exists() and not args.dry_run:
        print(f"refusing to start phase {args.phase}: no review marker {marker}")
        return 2
    months = args.months or ([SMOKE_MONTH] if args.phase == "smoke" else list(MONTHS))
    assert set(months) <= set(MONTHS), months
    env = {**os.environ, "R731": str(root)}
    if args.phase == "smoke":
        env["SEEDS"] = "0"
    if args.phase == "score":
        cmd = [sys.executable, str(HERE / "e00731_score.py")]
        print("score:", " ".join(cmd))
        return 0 if args.dry_run else subprocess.call(cmd, env=env)
    run_phase = {"smoke": "all", "fit": "all", "bank": "bank"}[args.phase]
    jobs = [(arm, month) for month in months for arm in args.arms]  # R0 first within each month
    slots = core_slots(cores=os.environ.get("CORES", "22-29"), parallel=args.parallel)
    free = list(range(args.parallel))
    lock = threading.Lock()
    failed: list[tuple[str, str]] = []
    print(f"{args.phase}: {len(jobs)} jobs, {args.parallel} parallel, cores {slots}", flush=True)

    def work(job: tuple[str, str]) -> None:
        arm, month = job
        with lock:
            slot = free.pop()
        try:
            cmd = [str(HERE / "e00731_run.sh"), arm, month, run_phase]
            print("start", cmd, "cores", slots[slot], flush=True)
            if args.dry_run:
                return
            code = subprocess.call(cmd, env={**env, "CORES": slots[slot]})
            if code != 0:
                failed.append(job)
                print("FAILED", job, code, flush=True)
        finally:
            with lock:
                free.append(slot)

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        list(pool.map(work, jobs))
    print("failed jobs:", failed or "none")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
