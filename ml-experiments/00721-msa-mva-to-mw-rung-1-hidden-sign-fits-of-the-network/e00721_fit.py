# ruff: noqa: E501
"""Experiment 721: the job list, the fit loop with the flag, and the per-job files.

The module holds everything about running fits except the construction of a batch from real data,
which `JobInputs` abstracts. The fit loop is plain PyTorch and is tested on a tiny network.

Compute rules (the page's "Compute" section): 6 GB of GPU memory per process, one core, `uptime` at
most 24, `free -g` at least 25, and `systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0`
around every process (`run.sh` does the last four). The continuation floor of 0.6 GPU-process-hours
is
enforced between fits.
"""

import json
import os
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Final, Protocol

import e00721_core as core
import numpy as np

GPU_MEMORY_BYTES: Final = 6 * 1024**3
UPTIME_MAX: Final = 24.0
FREE_GB_MIN: Final = 25.0
WORLDS: Final = ("synthetic", "W2", "W0")
QUICK_LOOK: Final = ("W2", "2026-05", 0)
"""Wave 1's cell: the world, the fold, and the seed."""


# --------------------------------------------------------------------------- jobs and files


@dataclass(frozen=True)
class Job:
    """One fit: an arm in a world on a fold with a seed.

    The synthetic world has the one fold `synthetic`.
    """

    arm: str
    world: str
    fold: str
    seed: int

    @property
    def key(self) -> str:
        """Return the job's file stem."""
        return f"{self.arm}__{self.world}__{self.fold}__s{self.seed}"


def data_root(*, data_dir: Path) -> Path:
    """Return the experiment's private output folder."""
    return data_dir / "ml-experiments" / f"{core.EXPERIMENT_ID}-{core.SLUG}"


def job_paths(*, data_dir: Path, job: Job) -> dict[str, Path]:
    """Return the per-job files: state, curve, predictions, and the done marker."""
    root = data_root(data_dir=data_dir) / "fits" / job.key
    return {
        "state": root.with_suffix(".pt"),
        "curve": root.with_suffix(".curve.json"),
        "predictions": data_root(data_dir=data_dir) / "predictions" / f"{job.key}.parquet",
        "done": root.with_suffix(".done.json"),
    }


def atomic_write_bytes(*, path: Path, payload: bytes) -> None:
    """Write `payload` to `path` through a temporary file in the same folder, then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(payload)
    tmp.replace(path)


def atomic_write_json(*, path: Path, body: dict) -> None:
    """Write `body` as JSON atomically."""
    atomic_write_bytes(path=path, payload=json.dumps(body, indent=1, sort_keys=True).encode())


def synthetic_jobs(*, seed: int) -> list[Job]:
    """Return the synthetic-world fits of one seed (stage 1 only): `SG-A` and its placebo `SG-A-S`."""
    return [Job(arm=a, world="synthetic", fold="synthetic", seed=seed) for a in ("SG-A", "SG-A-S")]


def real_jobs(*, world: str, seed: int, folds: Iterable[str] = core.FOLDS) -> list[Job]:
    """Return the real-fold fits of one world and seed, in fold order (`SG-A-S` only in `W2`.

    `SG-A-open` only in `W0`).
    """
    out = []
    for arm, spec in core.ARM_SPECS.items():
        if world in spec.worlds:
            out.extend(Job(arm=arm, world=world, fold=f, seed=seed) for f in folds)
    return out


def wave_jobs(*, wave: int) -> list[Job]:
    """Return the jobs of one wave of the page's compute table.

    Wave 1 is the synthetic world at seed 0 and the quick look (`P0`, `SG-0`, `SG-A` in `W2`, fold
    2026-05,
    seed 0). Wave 2 is the synthetic world at seed 1 and the rest of `W2` at seed 0. Wave 3 is `W0`
    at
    seed 0. Wave 4 is seed 1 of `SG-0`, `SG-A`, and `SG-A-S` in `W2`, and of `SG-A` and `P0` in
    `W0`.
    """
    world, fold, seed = QUICK_LOOK
    quick_look = [Job(arm=a, world=world, fold=fold, seed=seed) for a in ("P0", "SG-0", "SG-A")]
    if wave == 1:
        return [*synthetic_jobs(seed=0), *quick_look]
    if wave == 2:
        quick = {j.key for j in quick_look}
        return synthetic_jobs(seed=1) + [
            j for j in real_jobs(world="W2", seed=0) if j.key not in quick
        ]
    if wave == 3:
        return real_jobs(world="W0", seed=0)
    if wave == 4:
        w2 = [j for j in real_jobs(world="W2", seed=1) if j.arm in ("SG-0", "SG-A", "SG-A-S")]
        w0 = [j for j in real_jobs(world="W0", seed=1) if j.arm in ("SG-A", "P0")]
        return w2 + w0
    msg = f"wave {wave} is not one of 1 to 4"
    raise ValueError(msg)


# --------------------------------------------------------------------------- resource guards


def read_uptime_load() -> float:
    """Return the one-minute load average."""
    return os.getloadavg()[0]


def read_free_gb() -> float:
    """Return the available memory in GiB from `/proc/meminfo`."""
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024**2
    return float("nan")


def resource_problems(*, load: float, free_gb: float) -> list[str]:
    """Return what keeps a fit from starting: a load above 24 or fewer than 25 GiB available."""
    problems = []
    if not load <= UPTIME_MAX:
        problems.append(f"one-minute load {load:.1f} is above {UPTIME_MAX:.0f}")
    if not free_gb >= FREE_GB_MIN:
        problems.append(f"{free_gb:.1f} GiB available is below {FREE_GB_MIN:.0f}")
    return problems


def limit_gpu_memory(*, torch: object, device: int = 0) -> None:
    """Cap this process's share of the GPU at 6 GB."""
    total = torch.cuda.get_device_properties(device).total_memory  # ty: ignore[unresolved-attribute]
    torch.cuda.set_per_process_memory_fraction(min(1.0, GPU_MEMORY_BYTES / total), device)  # ty: ignore[unresolved-attribute]


@dataclass
class Budget:
    """The continuation floor: 0.6 GPU-process-hours stay unspent for the flag's continuation.

    Attributes:
        spent_hours: GPU-process-hours used so far.
        cap_hours: The experiment's cap.
        floor_hours: The hours that fits other than continuations may not touch.
    """

    spent_hours: float = 0.0
    cap_hours: float = core.GPU_HOURS_CAP
    floor_hours: float = core.CONTINUATION_FLOOR_GPU_HOURS

    def may_start_fit(self, *, expected_hours: float = core.GPU_HOURS_PER_FIT) -> bool:
        """Return whether an ordinary fit may start and still leave the floor."""
        return self.spent_hours + expected_hours <= self.cap_hours - self.floor_hours

    def may_continue(self, *, expected_hours: float) -> bool:
        """Return whether a continuation may run (it may use the floor)."""
        return self.spent_hours + expected_hours <= self.cap_hours


# --------------------------------------------------------------------------- the fit loop


@dataclass
class FitResult:
    """A fit's validation curve and its two scored states.

    Attributes:
        val_history: Validation loss after each pass (the patience-8 stop included).
        stop_pass_p8: Pass at which a patience of 8 stops.
        best_pass_p8: Pass of the patience-8 best.
        best_state_p8: The state at that pass.
        continued: Whether a continuation ran (Branch A).
        flag: The flag result's dictionary, if the flag was evaluated.
    """

    val_history: list[float] = field(default_factory=list)
    stop_pass_p8: int = -1
    best_pass_p8: int = -1
    best_state_p8: dict | None = None
    continued: bool = False
    flag: dict | None = None


class Trainable(Protocol):
    """What the loop needs of a fit: a training pass, a validation, and a state to save and load."""

    def train_pass(self) -> None:
        """Run one pass of training."""

    def validate(self) -> float:
        """Return the validation loss."""

    def get_state(self) -> dict:
        """Return a copy of the state."""

    def set_state(self, state: dict) -> None:
        """Load a state."""


def fit_patience(
    *, fit: Trainable, max_passes: int, patience: int, history: list[float] | None = None
) -> tuple[list[float], int, dict | None]:  # fmt: skip
    """Train until `patience` passes bring no improvement, or until `max_passes`.

    The improvement threshold is the one of `core.replay_stop`. Returns the history, the best pass
    (0-based), and a copy of the state at the best pass.
    """
    hist = list(history or [])
    best, best_pass, best_state = (
        (min(hist), int(np.argmin(hist)), None) if hist else (float("inf"), -1, None)
    )
    start = len(hist)
    for _ in range(start, max_passes):
        fit.train_pass()
        loss = fit.validate()
        hist.append(loss)
        if loss < best - 1e-5:
            best, best_pass, best_state = loss, len(hist) - 1, fit.get_state()
        if len(hist) - 1 - best_pass >= patience:
            break
    return hist, best_pass, best_state


def run_fit_with_flag(
    *, fit: Trainable, max_passes: int, continuation: bool, budget: Budget | None = None,
    hours_per_pass: float = 0.0,
) -> FitResult:  # fmt: skip
    """Fit with patience 8 and, with `continuation`, continue a curve stopped while falling.

    The scored state is always the patience-8 best, so that a continuation changes the flag and no
    score.
    The continuation runs in stage 1 for `SG-A` in `W2` (Branch A) at patience 30 up to 150 passes
    more,
    and only when the budget's floor allows it.
    """
    hist, best_pass, best_state = fit_patience(
        fit=fit, max_passes=max_passes, patience=core.REAL_FOLD_PATIENCE
    )
    result = FitResult(
        val_history=hist,
        stop_pass_p8=len(hist) - 1,
        best_pass_p8=best_pass,
        best_state_p8=best_state,
    )
    if not continuation:
        return result
    expected = hours_per_pass * core.CONTINUATION_CAP_PASSES
    if budget is not None and not budget.may_continue(expected_hours=expected):
        result.flag = {"evaluated": False, "reason": "budget floor"}
        return result
    fit_hist, _, _ = fit_patience(
        fit=fit, max_passes=min(len(hist) + core.CONTINUATION_CAP_PASSES, max_passes + core.CONTINUATION_CAP_PASSES),
        patience=core.CONTINUATION_PATIENCE, history=hist,
    )  # fmt: skip
    result.continued = True
    result.val_history = fit_hist
    flag = core.stopped_while_falling(val_history=fit_hist)
    result.flag = {"evaluated": True, **asdict(flag)}
    return result


# --------------------------------------------------------------------------- the real-data boundary


class JobInputs(Protocol):
    """What a job needs from the data, built from real files and never in a unit test."""

    def build_trainable(self, *, job: Job) -> Trainable:
        """Return the job's trainable (network, optimiser, batches, validation)."""

    def predict(self, *, job: Job, state: dict) -> object:
        """Return the job's scored forecast table at the fold's rows."""


def run_job(
    *, job: Job, inputs: JobInputs, data_dir: Path, budget: Budget, continuation: bool, log: Callable[[str], None] = print
) -> dict:  # fmt: skip
    """Run one job end to end and write its files atomically.

    A job whose `done` file exists is skipped.
    """
    paths = job_paths(data_dir=data_dir, job=job)
    if paths["done"].exists():
        log(f"skip {job.key}: done")
        return json.loads(paths["done"].read_text())
    problems = resource_problems(load=read_uptime_load(), free_gb=read_free_gb())
    if problems:
        msg = f"{job.key} does not start: {'; '.join(problems)}"
        raise RuntimeError(msg)
    if not budget.may_start_fit():
        msg = f"{job.key} does not start: the continuation floor of {budget.floor_hours} hours would be spent"
        raise RuntimeError(msg)
    started = time.monotonic()
    trainable = inputs.build_trainable(job=job)
    is_real = job.world != "synthetic"
    max_passes = core.REAL_FOLD_CAP_FIRST if is_real else core.SYNTH_MAX_PASSES
    if is_real:
        result = run_fit_with_flag(
            fit=trainable, max_passes=max_passes, continuation=continuation and job.arm == "SG-A" and job.world == "W2",
            budget=budget,
        )  # fmt: skip
    else:
        hist, best_pass, best_state = fit_patience(
            fit=trainable, max_passes=max_passes, patience=core.SYNTH_PATIENCE
        )
        result = FitResult(
            val_history=hist,
            stop_pass_p8=len(hist) - 1,
            best_pass_p8=best_pass,
            best_state_p8=best_state,
        )
    state = result.best_state_p8 if result.best_state_p8 is not None else trainable.get_state()
    inputs.predict(job=job, state=state)
    hours = (time.monotonic() - started) / 3600.0
    budget.spent_hours += hours
    curve = {
        "job": job.key, "val_history": result.val_history, "best_pass_p8": result.best_pass_p8,
        "stop_pass_p8": result.stop_pass_p8, "continued": result.continued, "flag": result.flag, "hours": hours,
    }  # fmt: skip
    atomic_write_json(path=paths["curve"], body=curve)
    atomic_write_json(path=paths["done"], body={"job": job.key, "hours": hours})
    log(f"done {job.key}: {hours:.2f} h")
    return curve


def real_inputs_hole() -> None:
    """Raise: the builder of real batches is not in this module.

    The batch builder needs the real grid, the weather, the capacity probe of each hidden Primary
    per fold and
    world, and the farm series, which only a run on the workstation can read. `smoke_real_files.py`
    lists the
    files it needs.

    Raises:
        NotImplementedError: Always.
    """
    msg = "build a JobInputs from real files: see tests/smoke_real_files.py for the files it reads"
    raise NotImplementedError(msg)
