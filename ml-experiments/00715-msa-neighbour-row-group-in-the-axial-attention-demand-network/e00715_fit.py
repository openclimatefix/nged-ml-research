# ruff: noqa: E501, PT018, PLR0915, D205, D209
"""Experiment 715: fit one F4 network (one arm, one fold, one seed) in the training phase.

Usage: e00715_fit.py ARM MONTH SEED [--device cuda] [--accept-code-change OLD_HASH ...]
       [--accept-gate-change OLD_HASH]

Run it through `run.sh`. It sets `E00667_PHASE=train`, under which `e00667_fold.install` sets the cleaned
power to NaN from the fold month's start, so no code path can read a held-out reading, neighbour power
included. It refuses to run without `work/known_answers_passed.json`, which `e00715_known_answer.py`
writes. F4 skips stage 1 and runs stages 2 and 3 of experiment 713's recipe, starting from experiment
713's stage 1 state of `M*` (or of arm `E`, for `F4E`) of the same fold and seed. A stage that ends
at the
60-pass cap is rerun from its starting state with a cap of 120 passes, and the log records both. Each
stage's state is saved atomically, and a stage whose file exists is skipped, so a job that crashed
resumes from its last finished stage.

**Code policy.** Every stage log carries the code hash, the git commit, and a dirty flag. A resume, a
finished-job skip, the gate file, and the stage 1 state of experiment 713 are refused when their hash
differs from the current code's, unless `--accept-code-change OLD_HASH` (stage states, repeatable) or
`--accept-gate-change OLD_HASH` (the gate record) names the exact old hash. The hash covers every module
F4 imports (`e00715_arms.code_hash`).

**Nothing is written into experiment 713's data.** The dense-array cache is redirected to F4's own
folder before the first call. When the fit ends, the script asserts that the modification times of the
files it reads from experiment 713's data are unchanged: the stage 1 state and log, the standardiser
hash file, `selection.json`, and the `cache/` folder. **A write by F4 to any other file in experiment
713's folder is not detected.** A failed assertion marks the final log `watch_failed`, and a finished-job
skip refuses such a log.

**GPU memory.** `--checkpoint-neighbours` recomputes the neighbour path in the backward pass; the log
records the flag. Code review 2 estimated the peak at about 6.2 GB per process with the flag and
9.5 GB without. The first fit of phase 0 measures `peak_gpu_mb`; plan one process per slot, or 7 GB
per process.
The gate's synthetic fits cannot use the flag, so run the gate alone in its slot.
"""

import argparse
import copy
import hashlib
import json
import os
import resource
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

os.environ["E00667_PHASE"] = "train"

import e00713_arms as a713  # ty: ignore[unresolved-import]
import e00713_model as mdl  # ty: ignore[unresolved-import]
import e00713_train as tr713  # ty: ignore[unresolved-import]
import e00715_arms as arms
import e00715_model as f4
import e00715_neighbours as nb
import e00715_train as tr
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

torch.set_num_threads(1)
CAP_FIRST, CAP_RERUN = 60, 120
STAGES = (2, 3)


@dataclass
class Deps:
    """The two calls that a check replaces with a stub, so that no network is trained in a unit check."""

    load_world: Callable[..., tuple[tr.F4World, dict]]
    train_stage: Callable[..., tr713.StageLog] = tr.train_stage


def write_json_atomic(*, obj: dict, path: Path) -> None:
    """Write a JSON file under a temporary name that carries the process id, then rename it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    tmp.rename(path)


def mtimes_of(*, folder: Path) -> dict[str, int]:
    """Return the modification time (ns) of every file and folder under `folder`, for the read-only assertion."""
    out = {}
    if not folder.exists():
        return out
    for base, dirs, files in os.walk(folder):
        for name in (*dirs, *files):
            path = Path(base) / name
            out[str(path)] = path.stat().st_mtime_ns
    out[str(folder)] = folder.stat().st_mtime_ns
    return out


def read_files(*, data_dir: Path, month: str) -> list[Path]:
    """Return the files of experiment 713's data, other than the stage 1 state and log, that an F4 fit reads."""
    root = a713.root(data_dir=data_dir)
    return [
        root / "work" / "standardiser" / f"{month}.sha256",
        a713.selection_file(data_dir=data_dir),
    ]


def watched_mtimes(*, data_dir: Path, files: Sequence[Path]) -> dict[str, int]:
    """Return the modification times of `files` and of every file in experiment 713's `cache` folder.

    Experiment 713 writes its own logs, fits, and states into `work/` while F4 fits, so the assertion
    watches only what F4 reads, and the cache folder that the redirect protects.
    """
    out = {str(f): f.stat().st_mtime_ns for f in files if f.exists()}
    out.update(mtimes_of(folder=a713.root(data_dir=data_dir) / "cache"))
    return out


def real_world(
    *, month: str, seed: int, spec: arms.ArmSpec, device: str, data_dir: Path
) -> tuple[tr.F4World, dict]:
    """Load the fold's dense arrays and build the `F4World` (the real pipeline; needs the data folder)."""
    import e00713_inputs as inp  # ty: ignore[unresolved-import]
    import e00715_batch as fb
    from studies.eval_protocol.slices import MVA_ONLY_LABELS

    arms.install_own_cache()
    pipe = inp.load_pipeline(
        month=month, phase="train", chain_root=inp.chain_root_of_seed(seed=seed)
    )
    data = inp.build_fold_data(pipe=pipe, device=device)
    check_fingerprint(data=data, data_dir=data_dir, month=month)
    base = tr713.World(
        source=data.source, cutoff_idx=data.cutoff_idx, clean=data.clean, target_mean=data.target_mean,
        target_std=data.target_std, month=month, placebo_pool=tr713.block_pool(clean=data.clean),
    )  # fmt: skip
    cap = arms.capacity_mw()
    del fb
    return (
        tr.F4World(
            base=base, table=nb.load_table(), mw_scale=data.target_std.astype(float) * cap, spec=spec,
            rewire_seed=seed, capacity_mw=cap, mva_only=_mva_flags(labels=MVA_ONLY_LABELS),
        ),
        data.dropped_per_fold,
    )  # fmt: skip


def _mva_flags(*, labels: Sequence[str]):  # noqa: ANN202
    """Return the (16,) bool array that marks the MVA-only Primaries."""
    import numpy as np

    return np.array([lab in labels for lab in nb.LABELS])


def check_fingerprint(*, data, data_dir: Path, month: str) -> None:  # noqa: ANN001
    """Assert that the fold's standardiser equals the one that experiment 713's fits of the same
    fold used (read only)."""
    path = a713.root(data_dir=data_dir) / "work" / "standardiser" / f"{month}.sha256"
    assert path.exists(), f"{path} is missing: experiment 713 has not fitted fold {month} yet"
    assert path.read_text() == data.fingerprint, (
        "the standardiser differs from experiment 713's: F4 and M* would not share a scale"
    )


def new_model(
    *,
    spec: arms.ArmSpec,
    seed: int,
    device: str,
    state: dict,
    data_dir: Path,
    checkpoint_neighbours: bool = False,
) -> tuple[mdl.MsaNet, str]:
    """Build the network of `spec`, load the stage 1 state, and apply the encoder rule of the selected arm.

    Returns the network and the encoder mode (`frozen`, `trained`, or `random`) of experiment 713's arm
    that the stage 1 state belongs to; stages 2 and 3 follow that rule.
    """
    torch.manual_seed(seed)
    encoders = a713.resolve(arm=spec.start_arm(data_dir=data_dir), data_dir=data_dir).encoders
    if spec.neighbour_path:
        model: mdl.MsaNet = f4.F4Net(switches=spec.switches)
        model.load_mstar_state(state)  # ty: ignore[unresolved-attribute]
        model.checkpoint_neighbours = checkpoint_neighbours  # ty: ignore[unresolved-attribute]
    else:
        model = mdl.MsaNet(switches=spec.switches)
        model.load_state_dict(state)
    if encoders == "frozen":
        model.freeze_encoders()
    return model.to(device), encoders


def skip_finished(
    *, arm: str, month: str, seed: int, data_dir: Path, accept: Sequence[str], log: dict
) -> dict:
    """Handle a job whose stage 3 state exists: check the code, rebuild a missing final log, and return it."""
    final_log = arms.log_file(data_dir=data_dir, arm=arm, month=month, seed=seed, kind="fit")
    stages = []
    for stage in STAGES:
        path = arms.stage_log_file(data_dir=data_dir, arm=arm, month=month, seed=seed, stage=stage)
        assert path.exists(), f"{path} is missing: refit the job"
        earlier = json.loads(path.read_text())
        arms.check_code(
            earlier=earlier, accept=accept, record=log, what=f"{arm} {month} s{seed} stage {stage}"
        )
        stages.append(earlier)
    if final_log.exists():
        existing = json.loads(final_log.read_text())
        assert not existing.get("watch_failed"), (
            f"{final_log} is marked `watch_failed`: a file read from experiment 713's data changed during the fit; refit the job"
        )
        new = [
            o for o in log.get("code_overrides", []) if o not in existing.get("code_overrides", [])
        ]
        if new:
            existing.setdefault("code_overrides", []).extend(new)
            write_json_atomic(obj=existing, path=final_log)
        print(f"{arm} {month} s{seed}: done, skipped", flush=True)
        return existing
    rebuilt = {
        **log, "stages": stages, "dropped": None, "rebuilt_from_stage_logs": True,
        "rebuilt_by_code_hash": arms.code_hash(), "code_hash": stages[-1].get("code_hash"),
        "git_commit": stages[-1].get("git_commit"),
    }  # fmt: skip
    write_json_atomic(obj=rebuilt, path=final_log)
    print(f"{arm} {month} s{seed}: done, final log rebuilt from the stage logs", flush=True)
    return rebuilt


def fit(
    *, arm: str, month: str, seed: int, device: str, data_dir: Path, accept: Sequence[str] = (),
    accept_gate: str | None = None, deps: Deps | None = None, checkpoint_neighbours: bool = False,
) -> dict:  # fmt: skip
    """Fit stages 2 and 3 and return the log."""
    gate = arms.gate_file(data_dir=data_dir)
    assert gate.exists(), f"{gate} is missing: run e00715_known_answer.py first"
    spec = arms.resolve(arm=arm)
    gate_sha = json.loads(gate.read_text()).get("known_answer_sha256")
    own_sha = hashlib.sha256(
        (Path(__file__).parent / "e00715_known_answer.py").read_bytes()
    ).hexdigest()
    print(
        f"known-answer file sha256: gate {gate_sha}, current {own_sha}, {'equal' if gate_sha == own_sha else 'DIFFERENT'}",
        flush=True,
    )
    assert month in spec.folds, f"arm {arm} is not fitted on fold {month}: {spec.folds}"
    gate713 = arms.gate713_summary(data_dir=data_dir)
    print(
        f"experiment 713's gate: waived {gate713['waived']}, failed checks {gate713['failed_checks']}, code hash {gate713['code_hash']}",
        flush=True,
    )
    log: dict = {
        "arm": arm, "month": month, "seed": seed, "checkpoint_neighbours": checkpoint_neighbours,
        "e00713_gate": gate713,
    }  # fmt: skip
    arms.check_code(
        earlier=json.loads(gate.read_text()).get("code", {}), accept=[accept_gate] if accept_gate else [],
        record=log, what="the known-answer gate file", kind="gate",
    )  # fmt: skip
    final = arms.model_file(data_dir=data_dir, arm=arm, month=month, seed=seed, stage=3)
    if final.exists():
        return skip_finished(
            arm=arm, month=month, seed=seed, data_dir=data_dir, accept=accept, log=log
        )
    deps = deps or Deps(
        load_world=lambda *, month, seed, spec, device, data_dir: real_world(
            month=month, seed=seed, spec=spec, device=device, data_dir=data_dir
        )
    )
    started = time.monotonic()
    start_arm = spec.start_arm(data_dir=data_dir)
    state_path = arms.stage1_state_file(
        data_dir=data_dir, start_arm=start_arm, month=month, seed=seed
    )
    log_path_1 = arms.stage1_log_file(
        data_dir=data_dir, start_arm=start_arm, month=month, seed=seed
    )
    assert state_path.exists() and log_path_1.exists(), (
        f"experiment 713's stage 1 state or log of {start_arm} {month} s{seed} is missing: {state_path}"
    )
    before = watched_mtimes(
        data_dir=data_dir,
        files=[state_path, log_path_1, *read_files(data_dir=data_dir, month=month)],
    )
    stage1_log = json.loads(log_path_1.read_text())
    arms.check_warm_start(
        stage1_log=stage1_log,
        accept=accept,
        record=log,
        what=f"stage 1 state of {start_arm} {month} s{seed}",
    )
    world, dropped = deps.load_world(
        month=month, seed=seed, spec=spec, device=device, data_dir=data_dir
    )
    model, encoders = new_model(
        spec=spec, seed=seed, device=device, state=torch.load(state_path, map_location=device), data_dir=data_dir,
        checkpoint_neighbours=checkpoint_neighbours,
    )  # fmt: skip
    log.update(
        stages=[], dropped=dropped, start_arm=start_arm, stage1_code_hash=stage1_log.get("code_hash"),
        parameters=mdl.parameter_count(model=model), **arms.code_record(),
    )  # fmt: skip
    for stage in STAGES:
        path = arms.model_file(data_dir=data_dir, arm=arm, month=month, seed=seed, stage=stage)
        stage_log = arms.stage_log_file(
            data_dir=data_dir, arm=arm, month=month, seed=seed, stage=stage
        )
        if path.exists():
            model.load_state_dict(torch.load(path, map_location=device))
            assert stage_log.exists(), f"{path} has no stage log: refit the stage"
            earlier = json.loads(stage_log.read_text())
            arms.check_code(earlier=earlier, accept=accept, record=log, what=f"{path.name}")
            log["stages"].append(earlier)
            continue
        samples = tr.make_samples(world=world, stage=stage)
        start_state = copy.deepcopy(model.state_dict())
        record = deps.train_stage(
            model=model, world=world, stage=stage, samples=samples, seed=seed, encoders=encoders,
            max_passes=CAP_FIRST,
        )  # fmt: skip
        entry = {"stage": stage, "pairs": len(samples), "first": vars(record).copy(), "rerun": None}
        if record.cap_hit:
            model.load_state_dict(start_state)
            again = deps.train_stage(
                model=model, world=world, stage=stage, samples=samples, seed=seed, encoders=encoders,
                max_passes=CAP_RERUN,
            )  # fmt: skip
            entry["rerun"] = vars(again).copy()
            record = again
        entry.update(
            passes=record.passes, best_val=record.best_val, cap_hit=record.cap_hit, **arms.code_record(),
            peak_gpu_mb=torch.cuda.max_memory_allocated() / 2**20 if device.startswith("cuda") else None,
            peak_host_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        )  # fmt: skip
        log["stages"].append(entry)
        write_json_atomic(
            obj=entry, path=stage_log
        )  # written before the state file, so a state file always has its log
        tr713.save_atomic(state=model.state_dict(), path=path)
        print(
            f"{arm} {month} s{seed} stage {stage}: {record.passes} passes, val {record.best_val:.4f}, "
            f"peak GPU {entry['peak_gpu_mb']} MB, peak host {entry['peak_host_mb']:.0f} MB",
            flush=True,
        )  # fmt: skip
    log["seconds"] = round(time.monotonic() - started, 1)
    log["peak_host_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    if device.startswith("cuda"):
        log["peak_gpu_mb"] = torch.cuda.max_memory_allocated() / 2**20
    write_json_atomic(
        obj=log, path=arms.log_file(data_dir=data_dir, arm=arm, month=month, seed=seed, kind="fit")
    )  # written before the read-only assertion, so a failed assertion never loses the log
    after = watched_mtimes(
        data_dir=data_dir,
        files=[state_path, log_path_1, *read_files(data_dir=data_dir, month=month)],
    )
    if before != after:
        log["watch_failed"] = True
        write_json_atomic(
            obj=log,
            path=arms.log_file(data_dir=data_dir, arm=arm, month=month, seed=seed, kind="fit"),
        )
        msg = "a file that F4 reads from experiment 713's data, or its cache folder, changed during the fit"
        raise AssertionError(msg)
    return log


def main() -> int:
    """Parse the command line and fit."""
    parser = argparse.ArgumentParser()
    parser.add_argument("arm")
    parser.add_argument("month")
    parser.add_argument("seed", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--accept-code-change", action="append", default=[], metavar="OLD_HASH")
    parser.add_argument("--accept-gate-change", default=None, metavar="OLD_HASH")
    parser.add_argument(
        "--checkpoint-neighbours", action="store_true",
        help="recompute the neighbour path in the backward pass, to lower the GPU peak; results are bit-identical",
    )  # fmt: skip
    args = parser.parse_args()
    fit(
        accept=args.accept_code_change, accept_gate=args.accept_gate_change, arm=args.arm,
        month=args.month, seed=args.seed, device=args.device, data_dir=ML_EXPERIMENTS_DATA_DIR,
        checkpoint_neighbours=args.checkpoint_neighbours,
    )  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
