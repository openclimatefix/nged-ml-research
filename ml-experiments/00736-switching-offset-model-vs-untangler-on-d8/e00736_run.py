"""Stages of experiment 736, one command per stage.

    e00736_run.py stage0 [--allow-fallback]   G1, G2, G4, the attribute check, the design probe
    e00736_run.py review-mark                 write the code hash into review_stage1.ok (after the
                                              review of that code; a person or the manager runs it)
    e00736_run.py freeze-code                 write code.sha256 (needs a review marker of this code)
    e00736_run.py gate3                       G3, one decode of the un-injected record
    e00736_run.py world [--draw k]            build world k (all 8 without --draw)
    e00736_run.py decode --worker i --of n    one world and arm per task, sharded
    e00736_run.py score
    e00736_run.py report                      prints and stores report_output.txt
    e00736_run.py figures                     private figures under the data folder

`run.sh` sets the environment and the memory-capped scope. Every output goes under `OUT`. Every
stage prints `OUT` first. The stages `world` and later refuse to start without `review_stage1.ok`
holding the current code hash, and the stages that decode or score also need `code.sha256` to hold
the same hash. A stage that fails writes `OUT/failed/stage_<stage>.json` and exits with code 3.
"""

from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import os
import resource
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Final

import e00736_decode as decode
import e00736_gates as gates
from e00736_common import (
    ARMS,
    EXIT_FAILED,
    N_DRAWS,
    OUT,
    atomic_json,
    sha256_of,
    time_log,
)
from e00736_world import (
    D8,
    DESIGN_FULL,
    GRID_LENGTH,
    build_world_power,
    load_env,
    probe_design_a,
    world_additions,
    write_world,
)

CODE_SHA: Final[Path] = OUT / "code.sha256"
"""Hash of the scoring code, written by `freeze-code`."""
STOP_CPU_HOURS: Final[float] = 8.0
BUDGET_MARKER: Final[str] = "cpu_budget_extended.ok"
"""Written by the manager to allow a run past `STOP_CPU_HOURS`."""
N_TASKS: Final[int] = N_DRAWS * len(ARMS)
"""Decode tasks: one per world and arm."""
MARKER_REVIEW: Final[str] = "review_stage1.ok"


BLAS_VARIABLES: Final[tuple[str, ...]] = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "POLARS_MAX_THREADS",
)
"""The variables that `run.sh` sets to 1, one thread per numerical library."""


def assert_blas_pinned() -> None:
    """Stop unless every numerical library is limited to one thread by the environment."""
    wrong = {k: os.environ.get(k) for k in BLAS_VARIABLES if os.environ.get(k) != "1"}
    if wrong:
        sys.exit(f"refusing to run: these variables must equal 1 (use run.sh): {wrong}")


def code_hash(*, folder: Path | None = None) -> str:
    """Return the SHA-256 over the names and bytes of every non-test module and `run*.sh` script.

    Args:
        folder: The folder to hash (this file's folder by default).

    Returns:
        The hex digest.
    """
    here = folder if folder is not None else Path(__file__).parent
    h = hashlib.sha256()
    for f in [*sorted(here.glob("e00736_*.py")), *sorted(here.glob("run*.sh"))]:
        h.update(f.name.encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()


def marker_hash(*, marker: str) -> str | None:
    """Return the code hash a marker file holds, or None when the file does not exist.

    Args:
        marker: The marker's file name under `OUT`.

    Returns:
        The first word of the file.
    """
    path = OUT / marker
    text = path.read_text().split() if path.exists() else []
    return text[0] if text else None


def require_review(*, marker: str) -> None:
    """Stop unless the review marker holds the hash of the current code.

    Args:
        marker: The marker's file name under `OUT`.
    """
    held = marker_hash(marker=marker)
    if held is None:
        sys.exit(f"refusing to run: {OUT / marker} is missing (the review has not passed)")
    if held != code_hash():
        sys.exit(
            f"refusing to run: {OUT / marker} holds the hash of different code (run review-mark)"
        )


def require_code_frozen() -> None:
    """Stop unless the code hash equals the one frozen with the stage 1 review."""
    if not CODE_SHA.exists() or code_hash() != CODE_SHA.read_text().split()[0]:
        sys.exit("refusing to run: the code differs from code.sha256 (run freeze-code)")


def require_gates() -> None:
    """Stop unless `gates.json` records a pass for G1 to G4."""
    path = OUT / "gates.json"
    if not path.exists():
        sys.exit("refusing to run: gates.json does not exist (run stage0 and gate3 first)")
    record = json.loads(path.read_text())
    if record.get("code_hash") != code_hash():
        sys.exit("refusing to run: gates.json was written by different code (rerun stage0, gate3)")
    failed = [g for g in ("G1", "G2", "G3", "G4") if record.get(g, {}).get("passed") is not True]
    if failed:
        sys.exit(f"refusing to run: gates {failed} have not passed")


def shard(*, items: list, worker: int, of: int) -> list:
    """Return the items of worker `worker` out of `of` workers.

    Args:
        items: Every item.
        worker: This worker's index.
        of: The number of workers.

    Returns:
        This worker's items.
    """
    return items[worker::of]


def spent_cpu_hours(*, log: Path | None = None, stage: str | None = None) -> float:
    """Return the CPU-hours in the per-phase time log (one thread per task).

    Args:
        log: The log file (`OUT/time_log.csv` by default).
        stage: Count only this phase when given.

    Returns:
        Hours between each line's start and end, summed.
    """
    path = log if log is not None else OUT / "time_log.csv"
    if not path.exists():
        return 0.0
    total = 0.0
    for line in path.read_text().splitlines():
        parts = line.split(",")
        if len(parts) == 5 and (stage is None or parts[0] == stage):
            a, b = (calendar.timegm(time.strptime(v, "%Y-%m-%dT%H:%M:%S")) for v in parts[3:5])
            total += (b - a) / 3600.0
    return total


def done_tasks(*, log: Path | None = None) -> int:
    """Return the number of decode tasks in the time log.

    Args:
        log: The log file.

    Returns:
        The number of lines of the `decode` phase.
    """
    path = log if log is not None else OUT / "time_log.csv"
    if not path.exists():
        return 0
    return sum(1 for ln in path.read_text().splitlines() if ln.startswith("decode,"))


def projected_cpu_hours(*, log: Path | None = None) -> float | None:
    """Project the decode phase's CPU-hours from the tasks finished so far.

    Args:
        log: The log file.

    Returns:
        The mean hours of a finished task times the number of tasks, or None before the first.
    """
    n = done_tasks(log=log)
    if n == 0:
        return None
    return spent_cpu_hours(log=log, stage="decode") / n * N_TASKS


def require_budget() -> None:
    """Stop at the CPU-hour line, spent or projected from the finished tasks.

    The marker `cpu_budget_extended.ok` in `OUT` lets a run past the line (the manager writes it).
    """
    if (OUT / BUDGET_MARKER).exists():
        return
    spent = spent_cpu_hours()
    if spent > STOP_CPU_HOURS:
        sys.exit(
            f"refusing to run: {spent:.1f} CPU-hours spent, over the {STOP_CPU_HOURS:.0f} stop "
            f"(ask the manager; marker {BUDGET_MARKER})"
        )
    projected = projected_cpu_hours()
    if projected is not None and projected > STOP_CPU_HOURS:
        sys.exit(
            f"refusing to run: the finished tasks project {projected:.1f} CPU-hours, over the "
            f"{STOP_CPU_HOURS:.0f} stop (ask the manager; marker {BUDGET_MARKER})"
        )


def run_task(*, name: str, fn: Callable[..., None], **kwargs: object) -> bool:
    """Run one task unless its done marker exists; record a failure and return False on error.

    A finished task writes `OUT/done/<name>.ok`, holding the code hash, the time, and the peak
    resident memory of this process. A re-run after a crash skips every task whose marker holds
    the current code hash, and runs again a task whose marker holds another hash.

    Args:
        name: The task's name, unique within the experiment.
        fn: The task.
        **kwargs: The task's keyword arguments.

    Returns:
        True when the task finished now or earlier, False when it failed.
    """
    done = OUT / "done" / f"{name}.ok"
    here = code_hash()
    if done.exists() and done.read_text().startswith(f"code_hash={here} "):
        print(f"skip {name}: already done", flush=True)
        return True
    try:
        fn(**kwargs)
    except Exception as exc:  # noqa: BLE001 - the failure record is the point of this boundary
        atomic_json(
            obj={"task": name, "message": str(exc), "traceback": traceback.format_exc()},
            path=OUT / "failed" / f"{name}.json",
        )
        print(f"FAILED {name}: {exc}", file=sys.stderr, flush=True)
        return False
    done.parent.mkdir(parents=True, exist_ok=True)
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    stamp = time.strftime("%FT%TZ", time.gmtime())
    done.write_text(f"code_hash={here} {stamp} peak_rss_gb={peak_gb:.2f}\n")
    print(f"peak resident memory of this worker so far: {peak_gb:.2f} GB", flush=True)
    return True


def run_shard(*, names_and_kwargs: list[tuple[str, dict]], fn: Callable[..., None]) -> None:
    """Run a worker's tasks in turn, stopping at the CPU budget; exit with code 3 on any failure.

    Args:
        names_and_kwargs: Each task's name and keyword arguments.
        fn: The task function.
    """
    failed = 0
    for name, kwargs in names_and_kwargs:
        require_budget()
        failed += not run_task(name=name, fn=fn, **kwargs)
    if failed:
        sys.exit(EXIT_FAILED)


def gate_record(*, out: dict, gate: str) -> dict:
    """Return one gate's record from the gates dictionary.

    Args:
        out: The gates dictionary.
        gate: The gate's name.

    Returns:
        The record.
    """
    return out[gate]


def stage0(*, allow_fallback: bool = False) -> None:
    """Run the gates that need no decode (G1, G2, G4), the attribute check, and the design probe.

    Args:
        allow_fallback: Whether a failed design probe may select the transfers-only design.

    Raises:
        RuntimeError: When the world design is not the full design and no fallback was allowed, or
            when the panel length differs from experiment 653's.
    """
    import e00653_common as common653  # ty: ignore[unresolved-import]
    import e00734_decode as dec  # ty: ignore[unresolved-import]

    start = time.time()
    missing = gates.missing_attributes()
    if missing:
        sys.exit(f"refusing to run: attributes missing: {missing}")
    if GRID_LENGTH != common653.GRID_LENGTH:
        msg = f"GRID_LENGTH is {GRID_LENGTH}, but experiment 653 has {common653.GRID_LENGTH}"
        raise RuntimeError(msg)
    design = probe_design_a(allow_fallback=allow_fallback)
    if design["design"] != DESIGN_FULL and not allow_fallback:
        msg = f"the world design is {design['design']}, not {DESIGN_FULL}"
        raise RuntimeError(msg)
    atomic_json(obj=design, path=OUT / "world_design.json")
    env = load_env()
    ctx = dec.load_context()
    rig = ctx.rig
    out = {
        "attributes_missing": missing,
        "design": design["design"],
        "G4": gates.g4(env=env),
        "G1": gates.g1(env=env),
        "G2": gates.g2(env=env, power=rig.panel.power, is_mva=rig.panel.is_mva),
    }
    existing = OUT / "gates.json"
    if existing.exists():
        old = json.loads(existing.read_text())
        if old.get("code_hash") == code_hash():
            out["G3"] = old.get("G3", {})
    out["code_hash"] = code_hash()
    atomic_json(obj=out, path=existing)
    time_log(stage="stage0", task="gates", start=start)
    print(json.dumps(out, indent=1, default=float))
    failed = [g for g in ("G1", "G2", "G4") if not gate_record(out=out, gate=g)["passed"]]
    if failed:
        sys.exit(f"gates failed: {failed}")


def gate3() -> None:
    """Run G3, which decodes the un-injected record once, and add it to `gates.json`."""
    require_review(marker=MARKER_REVIEW)
    require_code_frozen()
    record = gates.g3()
    path = OUT / "gates.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    out["G3"] = record
    out["code_hash"] = code_hash()
    atomic_json(obj=out, path=path)
    print(json.dumps(record, indent=1, default=float))
    if not record["passed"]:
        sys.exit("gate G3 failed")


def build_world(*, draw: int) -> None:
    """Build one world's additions and write them with a record.

    Args:
        draw: The draw (world) index.
    """
    import e00734_decode as dec  # ty: ignore[unresolved-import]

    start = time.time()
    design_file = OUT / "world_design.json"
    if design_file.exists():
        design = json.loads(design_file.read_text())
    else:
        design = probe_design_a()
        atomic_json(obj=design, path=design_file)
    env = load_env()
    rig = dec.load_context().rig
    sites = list(rig.panel.power)
    add = world_additions(
        segments=env.segments,
        overlays=env.overlays,
        draw=draw,
        design=design["design"],
        sites=sites,
    )
    power = build_world_power(power=rig.panel.power, additions=add, is_mva=rig.panel.is_mva)
    moved = {s: int((power[s] != rig.panel.power[s]).sum()) for s in sites}
    write_world(
        additions=add,
        draw=draw,
        record={
            "draw": draw,
            "design": design["design"],
            "half_hours_moved_by_site": moved,
            "d8_half_hours_with_addition": int((add[D8] != 0).sum()),
            "full_design_ran": design["design"] == DESIGN_FULL,
        },
    )
    time_log(stage="world", task=f"world_{draw}", start=start)


def review_mark() -> None:
    """Write the hash of the current code into the review marker.

    A person or the manager runs this after the review of exactly this code.
    """
    marker = OUT / MARKER_REVIEW
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(f"{code_hash()}  reviewed code\n")
    print(marker.read_text(), end="")


def parse_args(*, argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    Args:
        argv: The arguments (the process's by default).

    Returns:
        The parsed arguments.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "stage",
        choices=(
            "stage0",
            "review-mark",
            "freeze-code",
            "gate3",
            "world",
            "decode",
            "score",
            "report",
            "figures",
        ),
    )
    ap.add_argument("--draw", type=int, default=None)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--allow-fallback", action="store_true")
    return ap.parse_args(argv)


def dispatch(*, args: argparse.Namespace) -> None:
    """Run the stage that the arguments name.

    Args:
        args: The parsed arguments.
    """
    if args.stage == "stage0":
        stage0(allow_fallback=args.allow_fallback)
    elif args.stage == "review-mark":
        review_mark()
    elif args.stage == "freeze-code":
        require_review(marker=MARKER_REVIEW)
        CODE_SHA.parent.mkdir(parents=True, exist_ok=True)
        CODE_SHA.write_text(f"{code_hash()}  code\n")
        print(CODE_SHA.read_text(), end="")
    elif args.stage == "gate3":
        gate3()
    elif args.stage == "world":
        require_review(marker=MARKER_REVIEW)
        for k in [args.draw] if args.draw is not None else range(N_DRAWS):
            build_world(draw=k)
    elif args.stage == "decode":
        require_review(marker=MARKER_REVIEW)
        require_code_frozen()
        require_gates()
        tasks = [
            (f"decode_world_{k}_{arm}", {"draw": k, "arm": arm})
            for k in range(N_DRAWS)
            for arm in ARMS
        ]
        run_shard(
            names_and_kwargs=shard(items=tasks, worker=args.worker, of=args.of),
            fn=decode.decode_task,
        )
    elif args.stage == "score":
        require_review(marker=MARKER_REVIEW)
        require_code_frozen()
        require_gates()
        import e00736_score as score

        score.run_score()
    elif args.stage == "report":
        require_code_frozen()
        import e00736_score as score

        score.report()
    else:
        require_code_frozen()
        import e00736_figures as figs

        figs.make_figures()


def write_stage_failure(*, stage: str, worker: int, exc: BaseException) -> None:
    """Write a failed stage's message and traceback under `OUT/failed/`.

    Args:
        stage: The stage's name.
        worker: The worker's index.
        exc: The exception that ended the stage.
    """
    atomic_json(
        obj={
            "stage": stage,
            "worker": worker,
            "message": str(exc),
            "traceback": "".join(traceback.format_exception(exc)),
        },
        path=OUT / "failed" / f"stage_{stage}_worker_{worker}.json",
    )


def main(*, argv: list[str] | None = None) -> None:
    """Parse the command line and run one stage, recording a failure and exiting with code 3.

    Args:
        argv: The arguments (the process's by default).
    """
    assert_blas_pinned()
    args = parse_args(argv=argv)
    print(f"output folder: {OUT}", flush=True)
    try:
        dispatch(args=args)
    except Exception as exc:  # noqa: BLE001 - the failure record is the point of this boundary
        write_stage_failure(stage=args.stage, worker=args.worker, exc=exc)
        print(f"FAILED stage {args.stage}: {exc}", file=sys.stderr, flush=True)
        sys.exit(EXIT_FAILED)


if __name__ == "__main__":
    main()


__all__ = ["code_hash", "main", "sha256_of"]
