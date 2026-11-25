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
from typing import Any, Final, Protocol

import e00721_core as core
import numpy as np
import polars as pl

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
    capacity_scale: float = 1.0

    @property
    def base_key(self) -> str:
        """Return the file stem of the job's fit (the fit does not depend on the capacity sensitivity)."""
        return f"{self.arm}__{self.world}__{self.fold}__s{self.seed}"

    @property
    def key(self) -> str:
        """Return the job's file stem; the prediction-only capacity sensitivity (0.7 or 1.3) adds a suffix."""
        return (
            self.base_key
            if self.capacity_scale == 1.0
            else f"{self.base_key}__cap{self.capacity_scale:g}"
        )


def data_root(*, data_dir: Path) -> Path:
    """Return the experiment's private output folder."""
    return data_dir / "ml-experiments" / f"{core.EXPERIMENT_ID}-{core.SLUG}"


def job_paths(*, data_dir: Path, job: Job) -> dict[str, Path]:
    """Return the per-job files: state, curve, predictions, and the done marker."""
    root = data_root(data_dir=data_dir) / "fits" / job.base_key
    return {
        "state": root.with_suffix(".pt"),
        "curve": root.with_suffix(".curve.json"),
        "predictions": data_root(data_dir=data_dir) / "predictions" / f"{job.key}.parquet",
        "continuation_state": root.with_suffix(".continuation.pt"),
        "done": data_root(data_dir=data_dir) / "fits" / f"{job.key}.done.json",
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


def fit_key(*, arm: str, world: str, fold: str, seed: int) -> str:
    """Return the file stem of a job."""
    return Job(arm=arm, world=world, fold=fold, seed=seed).key


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
        # Stage 5 of the page: "`SG-0` seed 1 on fold 2026-05 runs first and writes the MDE file".
        first = fit_key(arm="SG-0", world="W2", fold=core.MDE_FOLD, seed=1)
        return sorted(w2 + w0, key=lambda j: j.key != first)
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


class ContinuationFloorReachedError(RuntimeError):
    """The continuation floor has been reached.

    The continuation time of the finished fits has reached the floor: the session asks the manager
    before the next `SG-A` fit in `W2`.
    """


@dataclass
class Budget:
    """The GPU-hour tallies that every process reads from the stored curve files.

    Attributes:
        spent_hours: GPU-process-hours of all finished fits.
        continuation_hours: GPU-process-hours of the continuations of the finished fits.
        cap_hours: The experiment's cap.
        floor_hours: The page's floor on the continuation: "before each `SG-A` fit in `W2` starts, the
            session adds up the continuation time of the fits that have finished. When that sum has
            reached 0.6 GPU-process-hours, the line stops at that fit". A continuation that is running
            when the sum passes the floor finishes.
    """

    spent_hours: float = 0.0
    continuation_hours: float = 0.0
    cap_hours: float = core.GPU_HOURS_CAP
    floor_hours: float = core.CONTINUATION_FLOOR_GPU_HOURS

    @classmethod
    def from_files(cls, *, data_dir: Path) -> Budget:
        """Return the tallies of every stored curve file (the processes of a wave share nothing else)."""
        folder = data_root(data_dir=data_dir) / "fits"
        spent = cont = 0.0
        for path in sorted(folder.glob("*.curve.json")):
            body = json.loads(path.read_text())
            spent += float(body.get("hours", 0.0))
            cont += float(body.get("continuation_hours", 0.0))
        return cls(spent_hours=spent, continuation_hours=cont)

    def may_start_fit(self, *, expected_hours: float = core.GPU_HOURS_PER_FIT) -> bool:
        """Return whether an ordinary fit may start under the cap."""
        return self.spent_hours + expected_hours <= self.cap_hours

    def floor_reached(self) -> bool:
        """Return whether the continuation time of the finished fits has reached the floor."""
        return self.continuation_hours >= self.floor_hours

    def check_before_continued_fit(self) -> None:
        """Raise when the floor is reached, before an `SG-A` fit in `W2` starts."""
        if self.floor_reached():
            msg = (
                f"the continuations of the finished fits took {self.continuation_hours:.2f} GPU-process-hours, "
                f"which reaches the floor of {self.floor_hours}: ask the manager before the next SG-A fit in W2"
            )
            raise ContinuationFloorReachedError(msg)


# --------------------------------------------------------------------------- the fit loop


@dataclass
class FitResult:
    """A fit's validation curve and its two scored states.

    Attributes:
        val_history: Validation loss after each pass (the continuation's passes included).
        stop_pass_p8: Last pass (0-based) of the patience-8 run.
        best_pass_p8: Pass of the patience-8 best.
        best_state_p8: The state at that pass (the state that is scored and passed on).
        best_state_p30: The state at the continuation's best pass, saved apart and never passed on.
        continued: Whether a continuation ran (Branch A).
        flag: The flag result's dictionary, if the flag was evaluated.
        rerun_of_cap: Whether the stage went on from the cap of 60 passes to the cap of 120.
        continuation_seconds: Measured seconds of the continuation's passes.
        seconds_per_pass: Measured seconds per pass of the patience-8 run.
    """

    val_history: list[float] = field(default_factory=list)
    stop_pass_p8: int = -1
    best_pass_p8: int = -1
    best_state_p8: dict | None = None
    best_state_p30: dict | None = None
    continued: bool = False
    flag: dict | None = None
    rerun_of_cap: bool = False
    continuation_seconds: float = 0.0
    seconds_per_pass: float = 0.0


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


class PatienceRun:
    """One run of a trainable that keeps its history, its best pass, and its best state.

    A pass is an improvement when it beats the best so far by more than 1e-5 (the rule of
    `core.replay_stop` and of experiment 713's loop), so the best pass and the best state are the
    ones the stop rule uses. `advance` can be called again with a larger cap or a longer patience, and
    the run goes on from where it stopped, so a stage that hit its cap is not trained twice.
    """

    def __init__(self, *, fit: Trainable) -> None:
        """Hold the trainable and an empty history."""
        self.fit = fit
        self.hist: list[float] = []
        self.best = float("inf")
        self.best_pass = -1
        self.best_state: dict | None = None

    def stale(self) -> int:
        """Return the passes since the best pass."""
        return len(self.hist) - 1 - self.best_pass

    def advance(self, *, max_passes: int, patience: int) -> None:
        """Train until `patience` passes bring no improvement, or until `max_passes` passes in all."""
        while len(self.hist) < max_passes and not (self.hist and self.stale() >= patience):
            self.fit.train_pass()
            loss = self.fit.validate()
            self.hist.append(loss)
            if loss < self.best - 1e-5:
                self.best, self.best_pass, self.best_state = (
                    loss,
                    len(self.hist) - 1,
                    self.fit.get_state(),
                )

    def hit_cap(self, *, cap: int, patience: int) -> bool:
        """Return whether the run ended at `cap` passes with its patience not yet used up."""
        return len(self.hist) >= cap and self.stale() < patience


def fit_patience(
    *, fit: Trainable, max_passes: int, patience: int
) -> tuple[list[float], int, dict | None]:  # fmt: skip
    """Train until `patience` passes bring no improvement, or until `max_passes`.

    Returns the history, the best pass (0-based, by the 1e-5 improvement rule), and a copy of the
    state at the best pass.
    """
    run = PatienceRun(fit=fit)
    run.advance(max_passes=max_passes, patience=patience)
    return run.hist, run.best_pass, run.best_state


def run_fit_with_flag(
    *, fit: Trainable, continuation: bool, first_cap: int | None = None, rerun_cap: int | None = None,
) -> FitResult:  # fmt: skip
    """Fit one real-fold stage with patience 8 and, with `continuation`, go on to patience 30.

    The page's rules, in one run: the stage trains for at most 60 passes, and "a stage that hits the
    cap of 60" goes on to at most 120 in the same run (the passes after pass 60 do not change the
    passes before it, so this is the rerun of the page without training the stage twice). The scored
    state is the patience-8 best. With `continuation` (Branch A: stage 1 of `SG-A` in `W2`), the same
    run then goes on at patience 30 up to 150 passes in all, and its better state is kept apart and
    never passed on. The flag reads the whole curve.
    """
    first_cap = (
        core.REAL_FOLD_CAP_FIRST if first_cap is None else first_cap
    )  # read at call time: the end-to-end pass overrides the constants
    rerun_cap = core.REAL_FOLD_CAP_RERUN if rerun_cap is None else rerun_cap
    run = PatienceRun(fit=fit)
    started = time.monotonic()
    run.advance(max_passes=first_cap, patience=core.REAL_FOLD_PATIENCE)
    raised = run.hit_cap(cap=first_cap, patience=core.REAL_FOLD_PATIENCE)
    if raised:
        run.advance(max_passes=rerun_cap, patience=core.REAL_FOLD_PATIENCE)
    seconds_per_pass = (time.monotonic() - started) / max(len(run.hist), 1)
    result = FitResult(
        val_history=list(run.hist), stop_pass_p8=len(run.hist) - 1, best_pass_p8=run.best_pass, best_state_p8=run.best_state,
        rerun_of_cap=raised, seconds_per_pass=seconds_per_pass,
    )  # fmt: skip
    if not continuation:
        return result
    cont_started = time.monotonic()
    run.advance(max_passes=core.CONTINUATION_CAP_PASSES, patience=core.CONTINUATION_PATIENCE)
    result.continuation_seconds = time.monotonic() - cont_started
    result.continued = True
    result.val_history = list(run.hist)
    result.best_state_p30 = run.best_state
    flag = core.stopped_while_falling(val_history=run.hist, p8_cap=rerun_cap)
    result.flag = {"evaluated": True, **asdict(flag)}
    return result


# --------------------------------------------------------------------------- the real-data boundary


class JobInputs(Protocol):
    """What a job needs from the data.

    `e00721_real` builds it from real files, and `e00721_e2e` from a synthetic mini world.
    """

    def stages(self, *, job: Job) -> tuple[int, ...]:
        """Return the training stages of the job (1 for the synthetic world, 1 to 3 for a real fold)."""

    def build_trainable(self, *, job: Job, stage: int) -> Trainable:
        """Return the stage's trainable on the job's network (the network carries from stage to stage)."""

    def predict(self, *, job: Job) -> pl.DataFrame:
        """Return the job's scored forecast table at the fold's rows, from the network's current state."""

    def network_state(self, *, job: Job) -> dict:
        """Return a copy of the job's network state (for the `.pt` file)."""

    def load_network_state(self, *, job: Job, state: dict) -> None:
        """Build the job's network for the predict phase and load `state` into it."""

    def sign_tokens(self, *, job: Job) -> dict[str, np.ndarray] | None:
        """Return the sign tokens of a synthetic fit, or `None` for any other job."""


def continuation_wanted(*, job: Job, stage: int) -> bool:
    """Return whether the continuation and the flag run: stage 1 of `SG-A` in `W2` (Branch A)."""
    return stage == core.SIGN_HEAD_STAGE and job.arm == "SG-A" and job.world == "W2"


def run_stage(
    *, job: Job, stage: int, inputs: JobInputs, continuation: bool
) -> tuple[FitResult, Trainable]:  # fmt: skip
    """Fit one stage and return the result with the trainable.

    A stage trains in one run: a real-fold stage that ends at the cap of 60 goes on to 120 in the same
    run, and the continuation, when it runs, goes on from there to 150 in all.
    """
    trainable = inputs.build_trainable(job=job, stage=stage)
    if job.world == "synthetic":
        hist, best_pass, best_state = fit_patience(
            fit=trainable, max_passes=core.SYNTH_MAX_PASSES, patience=core.SYNTH_PATIENCE
        )
        return FitResult(
            val_history=hist, stop_pass_p8=len(hist) - 1, best_pass_p8=best_pass, best_state_p8=best_state,
        ), trainable  # fmt: skip
    wanted = continuation_wanted(job=job, stage=stage)
    result = run_fit_with_flag(fit=trainable, continuation=continuation and wanted)
    if wanted and not continuation:
        result.flag = {
            "evaluated": False,
            "reason": "the manager allowed this fit without its continuation",
        }
    return result, trainable


def fit_job(
    *, job: Job, inputs: JobInputs, data_dir: Path, budget: Budget, continuation: bool, log: Callable[[str], None] = print
) -> dict:  # fmt: skip
    """Run the fit phase of one job and write its network state and curve atomically.

    A job whose curve file exists is skipped. Each stage starts from the network that the earlier
    stage left at its patience-8 best. The fit phase reads no data at or after the fold's cutoff.
    """
    paths = job_paths(data_dir=data_dir, job=job)
    if paths["curve"].exists() and paths["state"].exists():
        log(f"skip {job.key}: fitted")
        return json.loads(paths["curve"].read_text())
    problems = resource_problems(load=read_uptime_load(), free_gb=read_free_gb())
    if problems:
        msg = f"{job.key} does not start: {'; '.join(problems)}"
        raise RuntimeError(msg)
    if not budget.may_start_fit():
        msg = f"{job.key} does not start: the cap of {budget.cap_hours} GPU-process-hours would be passed"
        raise RuntimeError(msg)
    if continuation and any(
        continuation_wanted(job=job, stage=st) for st in inputs.stages(job=job)
    ):
        budget.check_before_continued_fit()
    (data_root(data_dir=data_dir) / "PRIVATE").mkdir(
        parents=True, exist_ok=True
    )  # dated detail goes here
    started = time.monotonic()
    stage_records: list[dict[str, Any]] = []
    continuation_hours = 0.0
    for stage in inputs.stages(job=job):
        t0 = time.monotonic()
        result, trainable = run_stage(
            job=job, stage=stage, inputs=inputs, continuation=continuation
        )
        if result.best_state_p8 is not None:
            trainable.set_state(result.best_state_p8)
        if result.best_state_p30 is not None:
            import io as _io

            import torch as _torch  # ty: ignore[unresolved-import]

            buf = _io.BytesIO()
            _torch.save({k: v.cpu() for k, v in result.best_state_p30.items()}, buf)
            atomic_write_bytes(path=paths["continuation_state"], payload=buf.getvalue())
        continuation_hours += result.continuation_seconds / 3600.0
        stage_records.append(
            {
                "stage": stage, "val_history": result.val_history, "best_pass_p8": result.best_pass_p8,
                "stop_pass_p8": result.stop_pass_p8, "continued": result.continued, "flag": result.flag,
                "seconds": time.monotonic() - t0, "passes": len(result.val_history), "rerun": result.rerun_of_cap,
                "seconds_per_pass": result.seconds_per_pass, "continuation_seconds": result.continuation_seconds,
            }
        )  # fmt: skip
        log(
            f"{job.key} stage {stage}: {len(result.val_history)} passes, {time.monotonic() - t0:.1f} s"
        )
    import io

    import torch  # ty: ignore[unresolved-import]

    buffer = io.BytesIO()
    torch.save(inputs.network_state(job=job), buffer)
    atomic_write_bytes(path=paths["state"], payload=buffer.getvalue())
    hours = (time.monotonic() - started) / 3600.0
    budget.spent_hours += hours
    budget.continuation_hours += continuation_hours
    curve = {
        "job": job.key, "stages": stage_records, "hours": hours, "continuation_hours": continuation_hours,
        "hours_per_pass": hours / max(sum(r["passes"] for r in stage_records), 1),
        "flag": next((r["flag"] for r in stage_records if r["flag"] is not None), None),
        "val_history": stage_records[0]["val_history"], "best_pass_p8": stage_records[0]["best_pass_p8"],
    }  # fmt: skip
    atomic_write_json(path=paths["curve"], body=curve)
    log(f"fitted {job.key}: {hours:.2f} h")
    return curve


def predict_job(
    *, job: Job, inputs: JobInputs, data_dir: Path, log: Callable[[str], None] = print
) -> None:
    """Run the predict phase of one job: load its stored state, write the scored rows, and mark the job done."""
    import torch  # ty: ignore[unresolved-import]

    paths = job_paths(data_dir=data_dir, job=job)
    if paths["done"].exists():
        log(f"skip {job.key}: done")
        return
    assert paths["state"].exists(), f"{paths['state']} is missing: run the fit phase first"
    inputs.load_network_state(job=job, state=torch.load(paths["state"], map_location="cpu"))
    table = inputs.predict(job=job)
    paths["predictions"].parent.mkdir(parents=True, exist_ok=True)
    tmp = paths["predictions"].with_name(f".{paths['predictions'].name}.{os.getpid()}.tmp")
    table.write_parquet(tmp)
    tmp.replace(paths["predictions"])
    tokens = inputs.sign_tokens(job=job)
    if tokens is not None:
        import io

        buffer = io.BytesIO()
        np.savez(buffer, **tokens)  # ty: ignore[invalid-argument-type]
        atomic_write_bytes(
            path=paths["predictions"].with_suffix(".tokens.npz"), payload=buffer.getvalue()
        )
    atomic_write_json(path=paths["done"], body={"job": job.key, "rows": table.height})
    log(f"predicted {job.key}: {table.height} rows")


def run_job(
    *, job: Job, inputs: JobInputs, data_dir: Path, budget: Budget, continuation: bool, log: Callable[[str], None] = print
) -> dict:  # fmt: skip
    """Run both phases of one job in this process (the end-to-end check; a real fold runs them in two processes)."""
    curve = fit_job(
        job=job, inputs=inputs, data_dir=data_dir, budget=budget, continuation=continuation, log=log
    )
    predict_job(job=job, inputs=inputs, data_dir=data_dir, log=log)
    return curve
