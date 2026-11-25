# ruff: noqa: D205, E501, PT018, PLR0911, PLR0912, PERF401
"""Experiment 725: the phase runner (design sections 7 and 9).

Usage:
    e00725_phases.py list PHASE
        # print the jobs of a phase and whether each is done
    e00725_phases.py run PHASE [--parallel 1|2] [--dry-run] [--manager-notified] [--device cuda]
    e00725_phases.py project [--from-phase1]
        # the compute projection, and the cut that it implies
    e00725_phases.py hours
        # GPU-process-hours spent so far, from the job logs

The runner holds no GPU or torch code. It launches each job as its own process through `run.sh`, so that
every fit runs in a capped scope of its own, and it never runs more than two GPU jobs at a time.

**Phases** (design section 9; PHASE is 0 to 7):

- 0: the two ENS caches (CPU), the wiring checks KL1 to KL10, the smoke fit of `L` (never scored),
  the reproduction fit of `Z` (fold 2025-11, seed 0) and its prediction, the reproduction check, and
  the projection.
- 1: `L` on the quick-look folds, seeds 0 and 1, and the `S` predictions; then the quick look.
- 2: `L` on the other 4 folds, seeds 0 and 1, and the `S` predictions.
- 3: `Z` on the quick-look folds, seeds 0 and 1, less the phase 0 fit; then the rule on `Z` and the
  two-seed reading.
- 4: the `T3` predictions of the stored `M*`, of `L`, and of the stored Candidate (target-row
  summary moved to D - 3).
- 5: the budget probe, only if `decisions/probe_flag.json` says so (cut to fold 2025-11 if the
  projection says so).
- 6: `L` on 8 folds, seeds 2 to 4, unless `decisions/futility.json` says the futility rule fired.
- 7: `E` on the quick-look folds, seeds 0 and 1, unless the projection cut it.

**Guards the runner applies.** A job whose output exists is skipped. Phase 0 and phase 1 refuse to
start a GPU job without `--manager-notified`, which records that "fits starting" went to the
manager. Before and after a phase the runner hashes experiment 713's `work` folder and refuses to
continue if the hash changed. After each job it adds the job's seconds to `work/gpu_hours.json`.
"""

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final

import e00725_common as cm
import e00725_t3_candidate as t3c
from studies.sources import ML_EXPERIMENTS_DATA_DIR

HERE: Final = Path(__file__).resolve().parent
RUN_SH: Final = HERE / "run.sh"
MAX_PARALLEL: Final = 2
DESIGN_ROWS: Final = (
    ("0", 0.4, "fit"), ("1", 1.3, "fit"), ("2", 1.3, "fit"), ("3", 1.0, "fit"), ("4", 0.3, "fixed"),
    ("5", 1.6, "fit"), ("6", 3.4, "fit"), ("7", 1.1, "fit"), ("8", 0.5, "fixed"), ("9", 0.8, "fixed"),
)  # fmt: skip
"""(phase, GPU-process-hours at 0.14 per fit, whether the row scales with the time per fit)."""
FIT_HOURS_DESIGN: Final = 0.14


@dataclass(frozen=True)
class Job:
    """One process of a phase.

    Attributes:
        kind: `fit`, `predict`, `cpu`, or `check` (a `cpu` or `check` job needs no GPU).
        script: The script under this folder that `run.sh` runs.
        args: The script's arguments.
        output: The file whose existence marks the job as done, or None if the job always runs.
        gpu: Whether the job uses the GPU.
    """

    kind: str
    script: str
    args: tuple[str, ...]
    output: Path | None
    gpu: bool

    @property
    def name(self) -> str:
        """Return a short label of the job."""
        return f"{self.script} {' '.join(self.args)}"

    def done(self) -> bool:
        """Return whether the job's output exists, and for a check job whether the code that wrote it is current."""
        if self.output is None:
            return False
        if self.kind == "check":
            return cm.check_output_current(path=self.output)
        return self.output.exists()


def fit_job(*, arm: str, month: str, seed: int, smoke: bool = False) -> Job:
    """Return the fit job of one arm, fold, and seed."""
    out = cm.log_file(
        data_dir=ML_EXPERIMENTS_DATA_DIR, arm=arm, month=month, seed=seed, kind="fit", smoke=smoke
    )
    args = (arm, month, str(seed), *(("--smoke",) if smoke else ()))
    return Job(kind="fit", script="e00725_fit.py", args=args, output=out, gpu=True)


def predict_job(*, arm: str, month: str, seed: int) -> Job:
    """Return the prediction job of one arm, fold, and seed."""
    out = cm.fit_file(data_dir=ML_EXPERIMENTS_DATA_DIR, arm=arm, month=month, seed=seed)
    return Job(
        kind="predict",
        script="e00725_predict.py",
        args=(arm, month, str(seed)),
        output=out,
        gpu=True,
    )


def arm_jobs(*, arm: str, months: tuple[str, ...], seeds: tuple[int, ...]) -> list[Job]:
    """Return a fit and a prediction job for every fold and seed of one arm."""
    jobs = []
    for month in months:
        for seed in seeds:
            jobs += [
                fit_job(arm=arm, month=month, seed=seed),
                predict_job(arm=arm, month=month, seed=seed),
            ]
    return jobs


def predict_jobs(*, arm: str, months: tuple[str, ...], seeds: tuple[int, ...]) -> list[Job]:
    """Return a prediction job for every fold and seed of one prediction-only reading."""
    return [predict_job(arm=arm, month=m, seed=s) for m in months for s in seeds]


def t3_candidate_jobs(*, months: tuple[str, ...], seeds: tuple[int, ...]) -> list[Job]:
    """Return, for every fold and seed, the `table` job and then the `fit` job that predict the
    stored Candidate with the target-row summary moved to D - 3.

    The `fit` job's output is the log that the job writes after its checks passed, not the forecast file.
    """
    jobs = []
    for month in months:
        for seed in seeds:
            for step, output in (
                ("table", t3c.table_file(month=month, seed=seed)),
                ("fit", t3c.log_file(month=month, seed=seed)),
            ):
                jobs.append(
                    Job(
                        kind="fit" if step == "table" else "predict",
                        script="e00725_t3_candidate.py",
                        args=(t3c.ARM, month, str(seed), "--step", step),
                        output=output,
                        gpu=True,
                    )
                )
    return jobs


def check_jobs() -> list[Job]:
    """Return the wiring-check jobs of phase 0, in order."""
    checks = cm.work(data_dir=ML_EXPERIMENTS_DATA_DIR) / "checks"
    jobs = [
        Job(
            "cpu",
            "e00725_sources.py",
            ("train",),
            cm.cache_dir(phase="train") / "meta.json",
            gpu=False,
        ),
        Job(
            "cpu",
            "e00725_sources.py",
            ("predict",),
            cm.cache_dir(phase="predict") / "meta.json",
            gpu=False,
        ),
        Job("check", "e00725_checks.py", ("pure",), checks / "pure.json", gpu=False),
        Job("check", "e00725_checks.py", ("kl5", "train"), checks / "kl5_train.json", gpu=False),
        Job(
            "check", "e00725_checks.py", ("kl5", "predict"), checks / "kl5_predict.json", gpu=False
        ),
        Job("check", "e00725_checks.py", ("kl8",), checks / "kl8.json", gpu=False),
    ]
    for month in cm.ALL_FOLDS:
        for phase in ("train", "predict"):
            jobs.append(
                Job("check", "e00725_checks.py", ("fold", month, phase), checks / f"{month}_{phase}.json", gpu=True)
            )  # fmt: skip
    jobs += [
        Job("check", "e00725_checks.py", ("kl3r", month), checks / f"{month}_kl3r.json", gpu=False)
        for month in cm.ALL_FOLDS
    ]
    jobs.append(
        Job(
            "check",
            "e00725_checks.py",
            ("collect",),
            cm.wiring_file(data_dir=ML_EXPERIMENTS_DATA_DIR),
            gpu=False,
        )
    )
    return jobs


def probe_months() -> tuple[str, ...]:
    """Return the folds of the probe: both, or fold 2025-11 alone when the projection cut the probe."""
    cut = read_json(path=projection_file()).get("cuts", [])
    return ("2025-11",) if "probe_to_one_fold" in cut else ("2025-11", "2026-06")


def phase_jobs(*, phase: int) -> list[Job]:
    """Return the jobs of one phase."""
    if phase == 0:
        return [
            *check_jobs(), fit_job(arm="L", month="2025-11", seed=0, smoke=True), fit_job(arm="Z", month="2025-11", seed=0),
            predict_job(arm="Z", month="2025-11", seed=0),
        ]  # fmt: skip
    if phase == 1:
        return [
            *arm_jobs(arm="L", months=cm.QUICK_FOLDS, seeds=cm.TWO_SEEDS),
            *predict_jobs(arm="S", months=cm.QUICK_FOLDS, seeds=cm.TWO_SEEDS),
        ]  # fmt: skip
    if phase == 2:
        return [
            *arm_jobs(arm="L", months=cm.OTHER_FOLDS, seeds=cm.TWO_SEEDS),
            *predict_jobs(arm="S", months=cm.OTHER_FOLDS, seeds=cm.TWO_SEEDS),
        ]  # fmt: skip
    if phase == 3:
        return arm_jobs(arm="Z", months=cm.QUICK_FOLDS, seeds=cm.TWO_SEEDS)
    if phase == 4:
        return [
            *predict_jobs(arm="T3-M", months=cm.ALL_FOLDS, seeds=cm.TWO_SEEDS),
            *predict_jobs(arm="T3-L", months=cm.ALL_FOLDS, seeds=cm.TWO_SEEDS),
            *t3_candidate_jobs(months=cm.ALL_FOLDS, seeds=cm.TWO_SEEDS),
        ]  # fmt: skip
    if phase == 5:
        months = probe_months()
        return [
            *arm_jobs(arm="L-P", months=months, seeds=(0,)),
            *arm_jobs(arm="Z-P", months=months, seeds=(0,)),
        ]
    if phase == 6:
        return arm_jobs(arm="L", months=cm.ALL_FOLDS, seeds=(2, 3, 4))
    assert phase == 7, f"phase {phase} does not exist: choose 0 to 7"
    return arm_jobs(arm="E", months=cm.QUICK_FOLDS, seeds=cm.TWO_SEEDS)


# ------------------------------------------------------------------------------------------ helpers


def read_json(*, path: Path) -> dict:
    """Return a JSON file's content, or an empty dict if the file is missing."""
    return json.loads(path.read_text()) if path.exists() else {}


def projection_file() -> Path:
    """Return the file that holds the compute projection."""
    return cm.work(data_dir=ML_EXPERIMENTS_DATA_DIR) / "projection.json"


def hours_file() -> Path:
    """Return the file that accumulates the GPU-process-seconds of finished jobs."""
    return cm.work(data_dir=ML_EXPERIMENTS_DATA_DIR) / "gpu_hours.json"


def gate_for(*, phase: int) -> None:
    """Refuse to run a phase whose gate says no (reproduction, projection, audit, probe, futility, cuts)."""
    decisions = cm.ROOT / "decisions"
    cut_arms = read_json(path=projection_file()).get("cut_arms", [])
    if phase == 1:
        repro = read_json(path=decisions / "repro.json")
        assert repro, "run `e00725_score.py repro` first (the reproduction check of the Z fit)"
        assert repro["outcome"] == "pass" or (decisions / "repro_cleared.json").exists(), (
            f"the reproduction check ended `{repro['outcome']}`: the experiment stops until the manager "
            "writes decisions/repro_cleared.json"
        )
        projection = read_json(path=projection_file())
        assert projection, "run `e00725_phases.py project` first (arm E is cut before phase 1)"
        assert float(projection["total"]) <= cm.CEILING_GPU_PROCESS_HOURS, (
            f"the projection of {projection['total']} GPU-process-hours is above the ceiling of "
            f"{cm.CEILING_GPU_PROCESS_HOURS}: ask the manager"
        )
    if phase == 2:
        audit = read_json(path=decisions / "quick_audit.json")
        assert audit, "run `e00725_score.py quick` first"
        assert not audit["fires"] or (decisions / "audit_cleared.json").exists(), (
            "the audit rule of the quick look fired: wait for the manager to write decisions/audit_cleared.json"
        )
    if phase == 3:
        assert "Z" not in cut_arms, (
            "the projection cut arm Z to the reproduction fit: phase 3 does not run"
        )
    if phase == 4:
        folder = cm.work(data_dir=ML_EXPERIMENTS_DATA_DIR) / "checks"
        for month in cm.ALL_FOLDS:
            for ph in ("train", "predict"):
                t3 = read_json(path=folder / f"{month}_{ph}.json").get("T3_known_answer", {})
                assert t3.get("pass"), (
                    f"the T3 known-answer check of {month} {ph} has not passed: {t3}"
                )
    if phase == 5:
        flag = read_json(path=decisions / "probe_flag.json")
        assert flag, "run `e00725_score.py probe-flag` first"
        assert flag["flag"], "the probe flag did not fire: the probe does not run"
    if phase == 6:
        futility = read_json(path=decisions / "futility.json")
        assert futility, "run `e00725_score.py two` first"
        assert not futility["fired"], "the futility rule fired: seeds 2 to 4 do not run"
        flag = read_json(path=decisions / "probe_flag.json")
        assert flag, "run `e00725_score.py probe-flag` first: the probe reports before phase 6"
        if flag["flag"]:
            assert all(j.done() for j in phase_jobs(phase=5)), (
                "the probe fired and has not finished: run phase 5"
            )
    if phase == 7:
        assert "E" not in cut_arms, "the projection cut arm E"


def run_job(*, job: Job, device: str) -> int:
    """Run one job through `run.sh` and return its exit status."""
    cmd = [str(RUN_SH), job.script, *job.args]
    if job.gpu and job.kind in {"fit", "predict"}:
        cmd += ["--device", device]
    if job.script == "e00725_checks.py" and job.args[:1] == ("fold",):
        cmd += ["--device", device]
    print("RUN", " ".join(cmd), flush=True)
    peak_dir = cm.work(data_dir=ML_EXPERIMENTS_DATA_DIR) / "memory_peak"
    peak_dir.mkdir(parents=True, exist_ok=True)
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", job.name)
    env = {**os.environ, "MEMORY_PEAK_FILE": str(peak_dir / f"{label}.txt")}
    return subprocess.run(cmd, check=False, env=env).returncode


def add_seconds(*, job: Job, seconds: float, lock: threading.Lock) -> None:
    """Record a finished GPU job's wall time."""
    with lock:
        data = read_json(path=hours_file())
        data[job.name] = round(seconds, 1)
        cm.write_json_atomic(obj=data, path=hours_file())


def run_chain(*, chain: list[Job], device: str, lock: threading.Lock, failures: list[str]) -> None:
    """Run the jobs of one chain in order (a fit, then its prediction), stopping at the first failure."""
    for job in chain:
        started = time.monotonic()
        status = run_job(job=job, device=device)
        if status != 0:
            failures.append(f"{job.name}: exit status {status}")
            return
        if job.gpu:
            key_job = job if job.kind != "check" else replace(job, args=("CHECK", *job.args))
            add_seconds(job=key_job, seconds=time.monotonic() - started, lock=lock)


def chains_of(*, jobs: list[Job]) -> list[list[Job]]:
    """Group GPU fit and predict jobs into chains: one chain per (arm, fold, seed), the fit first
    (one chain per fold for `T3-C`).
    """
    chains: dict[tuple[str, ...], list[Job]] = {}
    for job in jobs:
        # the two seeds of a `T3-C` fold share one folder, so they form one chain and never run at once
        key = tuple(job.args[:2] if job.script == "e00725_t3_candidate.py" else job.args[:3])
        chains.setdefault(key, []).append(job)
    return list(chains.values())


def run_phase(*, phase: int, parallel: int, device: str, dry_run: bool, notified: bool) -> int:
    """Run the jobs of one phase that are not done, at most `parallel` GPU chains at a time."""
    import e00725_frozen as fz

    assert 1 <= parallel <= MAX_PARALLEL, f"at most {MAX_PARALLEL} GPU processes of F8 run at once"
    gate_for(phase=phase)
    jobs = phase_jobs(phase=phase)
    todo = [j for j in jobs if not j.done()]
    print(f"phase {phase}: {len(jobs)} jobs, {len(todo)} to run", flush=True)
    if dry_run or not todo:
        for j in todo:
            print("  TODO", j.name)
        return 0
    if phase in {0, 1}:
        assert notified, (
            '"fits starting" has to go to the manager first; rerun with --manager-notified'
        )
    before = fz.work_snapshot_713(data_dir=ML_EXPERIMENTS_DATA_DIR)
    failures: list[str] = []
    lock = threading.Lock()
    is_collect = lambda j: j.script == "e00725_checks.py" and j.args[0] == "collect"  # noqa: E731
    is_kl3r = lambda j: j.script == "e00725_checks.py" and j.args[0] == "kl3r"  # noqa: E731
    before_collect = [
        j for j in todo
        if (j.kind in {"cpu", "check"} or "--smoke" in j.args) and not is_collect(j) and not is_kl3r(j)
    ]  # fmt: skip
    for job in before_collect:
        run_chain(chain=[job], device=device, lock=lock, failures=failures)
        if failures:
            break
    kl3r_jobs = [j for j in todo if is_kl3r(j)]
    if kl3r_jobs and not failures:
        # KL3r is CPU only (about 40 minutes per fold), so it runs beside the GPU slot, one thread each
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            for f in [
                pool.submit(run_chain, chain=[j], device=device, lock=lock, failures=failures)
                for j in kl3r_jobs
            ]:
                f.result()
    for job in [j for j in todo if is_collect(j)]:
        if not failures:
            run_chain(chain=[job], device=device, lock=lock, failures=failures)
    serial = [*before_collect, *kl3r_jobs, *[j for j in todo if is_collect(j)]]
    if not failures:
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = [
                pool.submit(run_chain, chain=chain, device=device, lock=lock, failures=failures)
                for chain in chains_of(jobs=[j for j in todo if j not in serial])
            ]
            for f in futures:
                f.result()
    after = fz.work_snapshot_713(data_dir=ML_EXPERIMENTS_DATA_DIR)
    if before != after:
        failures.append("experiment 713's `work` folder changed during the phase")
    cm.write_json_atomic(
        obj={"phase": phase, "snapshot_713_before": before, "snapshot_713_after": after, "failures": failures},
        path=cm.work(data_dir=ML_EXPERIMENTS_DATA_DIR) / f"phase{phase}_run.json",
    )  # fmt: skip
    for line in failures:
        print("FAILED", line, flush=True)
    return 1 if failures else 0


# ----------------------------------------------------------------------------- projection and hours


def seconds_of(*, log: Path) -> float | None:
    """Return the `seconds` of a job log, or None if the log is missing."""
    data = read_json(path=log)
    return None if not data else float(data.get("seconds", 0.0))


def hours_spent() -> float:
    """Return the GPU-process-hours of the finished GPU jobs."""
    return sum(v for k, v in read_json(path=hours_file()).items() if " CHECK " not in k) / 3600.0


def projection(*, per_fit_hours: float, cut_arms: list[str], cuts: list[str]) -> dict:
    """Return the projected total GPU-process-hours at `per_fit_hours` per fit, with the given cuts applied."""
    scale = per_fit_hours / FIT_HOURS_DESIGN
    total, rows = 0.0, {}
    for phase, hours, kind in DESIGN_ROWS:
        value = hours * (scale if kind == "fit" else 1.0)
        if phase == "7" and "E" in cut_arms:
            value = 0.0
        if phase == "3" and "Z" in cut_arms:
            value = 0.0
        if phase == "5" and "probe_to_one_fold" in cuts:
            value *= 0.5
        rows[phase] = round(value, 2)
        total += value
    return {"per_fit_hours": per_fit_hours, "rows": rows, "total": round(total, 2)}


def choose_cuts(*, per_fit_hours: float) -> dict:
    """Return the cuts that bring the projection to the 12-hour ceiling, in the order E, then Z, then the probe."""
    cut_arms: list[str] = []
    cuts: list[str] = []
    steps = [
        ("E", cut_arms, "E"),
        ("Z", cut_arms, "Z"),
        ("probe_to_one_fold", cuts, "probe_to_one_fold"),
    ]
    result = projection(per_fit_hours=per_fit_hours, cut_arms=cut_arms, cuts=cuts)
    for label, target, value in steps:
        if result["total"] <= cm.CEILING_GPU_PROCESS_HOURS:
            break
        target.append(value)
        result = projection(per_fit_hours=per_fit_hours, cut_arms=cut_arms, cuts=cuts)
        result["last_cut"] = label
    return {"cut_arms": cut_arms, "cuts": cuts, **result}


def cmd_project(*, from_phase1: bool) -> int:
    """Write the projection from the phase 0 fits, or from the median of the first 3 fits of phase 1."""
    data_dir = ML_EXPERIMENTS_DATA_DIR
    if from_phase1:
        logs = [
            cm.log_file(data_dir=data_dir, arm="L", month=m, seed=s, kind="fit")
            for m in cm.QUICK_FOLDS for s in cm.TWO_SEEDS
        ]  # fmt: skip
        preds = [log.with_name(log.name.replace("_fit", "_predict")) for log in logs]
        secs = [(seconds_of(log=a), seconds_of(log=b)) for a, b in zip(logs, preds, strict=True)]
        done = [(a or 0.0) + (b or 0.0) for a, b in secs if a is not None][:3]
        assert len(done) >= 3, "fewer than 3 fits of phase 1 have finished"
        per_fit = statistics.median(done) / 3600.0
        label = "phase1_median_of_first_3"
    else:
        smoke = cm.log_file(
            data_dir=data_dir, arm="L", month="2025-11", seed=0, kind="fit", smoke=True
        )
        z_fit = cm.log_file(data_dir=data_dir, arm="Z", month="2025-11", seed=0, kind="fit")
        z_pred = cm.log_file(data_dir=data_dir, arm="Z", month="2025-11", seed=0, kind="predict")
        s_smoke, s_z, s_pred = seconds_of(log=smoke), seconds_of(log=z_fit), seconds_of(log=z_pred)
        assert s_smoke is not None and s_z is not None, (
            "the smoke fit and the Z fit have to finish first"
        )
        per_fit = max(s_smoke + (s_pred or 0.0), s_z + (s_pred or 0.0)) / 3600.0
        label = "phase0_slowest_of_smoke_and_Z_with_prediction"
    out = {
        "basis": label,
        **choose_cuts(per_fit_hours=per_fit),
        "spent_so_far_hours": round(hours_spent(), 3),
    }
    out["ceiling"] = cm.CEILING_GPU_PROCESS_HOURS
    cm.write_json_atomic(obj=out, path=projection_file())
    print(json.dumps(out, indent=1), flush=True)
    if float(out["total"]) > cm.CEILING_GPU_PROCESS_HOURS:
        print("THE PROJECTION CROSSES THE CEILING: stop and ask the manager.", flush=True)
        return 1
    return 0


def cmd_list(*, phase: int) -> int:
    """Print the jobs of a phase."""
    for job in phase_jobs(phase=phase):
        print(("done " if job.done() else "todo ") + job.name)
    return 0


def main() -> int:
    """Parse the command line and run a phase command."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p_list = sub.add_parser("list")
    p_list.add_argument("phase", type=int)
    p_run = sub.add_parser("run")
    p_run.add_argument("phase", type=int)
    p_run.add_argument("--parallel", type=int, default=1)
    p_run.add_argument("--dry-run", action="store_true")
    p_run.add_argument("--manager-notified", action="store_true")
    p_run.add_argument("--device", default="cuda")
    p_proj = sub.add_parser("project")
    p_proj.add_argument("--from-phase1", action="store_true")
    sub.add_parser("hours")
    args = parser.parse_args()
    handlers: dict[str, Callable[[], int]] = {
        "list": lambda: cmd_list(phase=args.phase),
        "run": lambda: run_phase(
            phase=args.phase,
            parallel=args.parallel,
            device=args.device,
            dry_run=args.dry_run,
            notified=args.manager_notified,
        ),
        "project": lambda: cmd_project(from_phase1=args.from_phase1),
        "hours": lambda: print(f"{hours_spent():.3f} GPU-process-hours") or 0,
    }
    return handlers[args.command]()


if __name__ == "__main__":
    sys.exit(main())
