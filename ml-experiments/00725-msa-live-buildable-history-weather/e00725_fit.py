# ruff: noqa: D205, E501, PT018, PLR0915
"""Experiment 725: fit one F8 network (one arm, one fold, one seed) in the training phase.

Usage: e00725_fit.py ARM MONTH SEED [--device cuda] [--smoke] [--accept-code-change OLD_HASH ...]

ARM is `L` (the live rule), `Z` (the rule switched off), `E` (ENS at every age), or the probe arms `L-P`
and `Z-P` (patience 30, cap 120). Run it through `run.sh`. It sets `E00667_PHASE=train`, under which
`e00667_fold.install` sets the cleaned power to NaN from the fold month's start.

**What a fit does.** It skips stage 1 and runs stages 2 and 3 of experiment 713's recipe from the stored
stage 1 state of `M*` (`M1`) of the same fold and seed, through experiment 713's own `train_stage`. The
only difference from the fit of `M*` is the source: a `LiveSource`, whose `with_power_in` override
carries the rule into stage 3 (check KL9). A stage that ends at the 60-pass cap is rerun from its
starting state with a cap of 120 passes, and the log records both. Each stage's state is saved
atomically, and a stage whose file exists is skipped.

**Gate.** A real fold needs `work/wiring_checks_passed.json`, which `e00725_checks.py` writes at F8's
current code hash. `--smoke` writes under `smoke/` and is never scored.

**Stopping-pass record.** Every stage log carries `stopping`, the record of `e00725_common.stopping_record`
(passes, best pass, optimiser steps at the best pass, losses at pass 1 and at the best pass, cap hit,
rerun).

**Read only.** Nothing is written into experiment 713's data. The fit asserts that the files it
reads from there (the stage 1 state and log, the standardiser hash, `selection.json`) are unchanged
when it ends.
"""

import argparse
import copy
import json
import os
import resource
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

os.environ["E00667_PHASE"] = "train"

import e00713_arms as arms713  # ty: ignore[unresolved-import]
import e00713_model as mdl  # ty: ignore[unresolved-import]
import e00713_train as tr  # ty: ignore[unresolved-import]
import e00725_common as cm
import e00725_frozen as fz
import e00725_world as world_mod
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

torch.set_num_threads(1)


def nvidia_smi_process_mb() -> int | None:
    """Return the GPU memory in MB that `nvidia-smi` reports for this process, or None."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        ).stdout  # fmt: skip
    except subprocess.CalledProcessError, FileNotFoundError:
        return None
    for line in out.splitlines():
        pid, _, mem = line.partition(",")
        if pid.strip() == str(os.getpid()):
            return int(mem.strip())
    return None


def gpu_gb_budget_ok(*, peak_mb: float | None) -> bool:
    """Return whether a process stays at or under 6 GB of GPU memory (design section 9)."""
    return peak_mb is None or peak_mb <= 6 * 1024


def new_model(*, seed: int, month: str, device: str, data_dir: Path) -> mdl.MsaNet:
    """Build the network of `M*`, load its stage 1 state, and freeze the encoders as experiment 713 does."""
    spec713 = arms713.resolve(arm=cm.M_STAR, data_dir=data_dir)
    torch.manual_seed(seed)
    model = mdl.MsaNet(switches=spec713.switches)
    state = torch.load(
        fz.stage1_state_file(data_dir=data_dir, month=month, seed=seed), map_location="cpu"
    )
    model.load_state_dict(state)  # strict: F8 adds no parameter, so no key is missing or extra
    if spec713.encoders == "frozen":
        model.freeze_encoders()
    return model.to(device)


def skip_finished(
    *,
    arm: str,
    month: str,
    seed: int,
    data_dir: Path,
    accept: Sequence[str],
    log: dict,
    smoke: bool,
) -> dict:
    """Handle a job whose stage 3 state exists: check the code, rebuild a missing final log, and return it."""
    final_log = cm.log_file(
        data_dir=data_dir, arm=arm, month=month, seed=seed, kind="fit", smoke=smoke
    )
    stages = []
    for stage in cm.STAGES:
        path = cm.stage_log_file(
            data_dir=data_dir, arm=arm, month=month, seed=seed, stage=stage, smoke=smoke
        )
        assert path.exists(), f"{path} is missing: refit the job"
        earlier = json.loads(path.read_text())
        fz.check_code(
            earlier=earlier, accept=accept, record=log, what=f"{arm} {month} s{seed} stage {stage}"
        )
        stages.append(earlier)
    if final_log.exists():
        existing = json.loads(final_log.read_text())
        assert not existing.get("watch_failed"), (
            f"{final_log} is marked `watch_failed`: a file read from experiment 713's data changed during the fit"
        )
        print(f"{arm} {month} s{seed}: done, skipped", flush=True)
        return existing
    rebuilt = {
        **log, "stages": stages, "rebuilt_from_stage_logs": True,
        "rebuilt_by_code_hash": fz.code_hash(), "code_hash": stages[-1].get("code_hash"),
    }  # fmt: skip
    cm.write_json_atomic(obj=rebuilt, path=final_log)
    print(f"{arm} {month} s{seed}: done, final log rebuilt from the stage logs", flush=True)
    return rebuilt


def run_stage(
    *, model: mdl.MsaNet, world: tr.World, stage: int, seed: int, encoders: str, spec: cm.ArmSpec,
    train_stage: Callable[..., tr.StageLog] = tr.train_stage,
) -> tuple[dict, tr.StageLog]:  # fmt: skip
    """Train one stage with the arm's patience and cap, rerun at 120 passes if the cap is hit, and
    return the entry.
    """
    samples = tr.make_samples(world=world, stage=stage, placebo=False)
    start_state = copy.deepcopy(model.state_dict())
    record = train_stage(
        model=model, world=world, stage=stage, samples=samples, seed=seed, encoders=encoders,
        max_passes=spec.cap, patience=spec.patience,
    )  # fmt: skip
    entry: dict = {
        "stage": stage,
        "pairs": len(samples),
        "first": vars(record).copy(),
        "rerun": None,
    }
    if record.cap_hit and not spec.probe:
        model.load_state_dict(start_state)
        again = train_stage(
            model=model, world=world, stage=stage, samples=samples, seed=seed, encoders=encoders,
            max_passes=cm.CAP_RERUN, patience=spec.patience,
        )  # fmt: skip
        entry["rerun"] = vars(again).copy()
        record = again
    entry.update(
        passes=record.passes, best_val=record.best_val, cap_hit=record.cap_hit,
        seconds=record.seconds, patience=spec.patience, cap=spec.cap,
    )  # fmt: skip
    entry["stopping"] = cm.stopping_record(entry=entry)
    return entry, record


def fit(
    *, arm: str, month: str, seed: int, device: str, data_dir: Path, accept: Sequence[str] = (),
    smoke: bool = False,
) -> dict:  # fmt: skip
    """Fit stages 2 and 3 of one F8 arm and return the log."""
    spec = cm.resolve(arm=arm)
    fz.assert_frozen_checkout()
    fz.limit_gpu_memory(device=device)
    log: dict = {"arm": arm, "month": month, "seed": seed, "smoke": smoke, "rule": spec.rule}
    gate = cm.wiring_file(data_dir=data_dir)
    if not smoke:
        assert gate.exists(), f"{gate} is missing: run e00725_checks.py first"
        cm.assert_gate_checks_hash(gate=json.loads(gate.read_text()))
        fz.check_code(
            earlier=json.loads(gate.read_text()).get("code", {}), accept=[], record=log,
            what="the wiring-check gate file",
        )  # fmt: skip
    final = cm.model_file(data_dir=data_dir, arm=arm, month=month, seed=seed, stage=3, smoke=smoke)
    if final.exists():
        return skip_finished(
            arm=arm, month=month, seed=seed, data_dir=data_dir, accept=accept, log=log, smoke=smoke
        )
    started = time.monotonic()
    state_path = fz.stage1_state_file(data_dir=data_dir, month=month, seed=seed)
    stage1_log_path = fz.stage_log_713(data_dir=data_dir, month=month, seed=seed, stage=1)
    assert state_path.exists() and stage1_log_path.exists(), (
        f"experiment 713's stage 1 state or log of {cm.M_STAR} {month} s{seed} is missing"
    )
    watched = fz.read_files_713(data_dir=data_dir, month=month, seed=seed)
    before = fz.watched_mtimes(files=watched)
    stage1_log = json.loads(stage1_log_path.read_text())
    fz.check_warm_start(stage1_log=stage1_log, what=f"stage 1 state of {cm.M_STAR} {month} s{seed}")
    log["e00713_gate"] = fz.gate713_summary(data_dir=data_dir)
    lw = world_mod.load_world(
        month=month, seed=seed, rule_name=spec.rule, phase="train", device=device, data_dir=data_dir
    )  # fmt: skip
    spec713 = arms713.resolve(arm=cm.M_STAR, data_dir=data_dir)
    model = new_model(seed=seed, month=month, device=device, data_dir=data_dir)
    log.update(
        stages=[], dropped=lw.data.dropped_per_fold, stage1_code_hash=stage1_log.get("code_hash"),
        parameters=sum(p.numel() for p in model.parameters()), **fz.code_record(),
    )  # fmt: skip
    for stage in cm.STAGES:
        path = cm.model_file(
            data_dir=data_dir, arm=arm, month=month, seed=seed, stage=stage, smoke=smoke
        )
        stage_log = cm.stage_log_file(
            data_dir=data_dir, arm=arm, month=month, seed=seed, stage=stage, smoke=smoke
        )
        if path.exists():
            model.load_state_dict(torch.load(path, map_location=device))
            assert stage_log.exists(), f"{path} has no stage log: refit the stage"
            earlier = json.loads(stage_log.read_text())
            fz.check_code(earlier=earlier, accept=accept, record=log, what=path.name)
            log["stages"].append(earlier)
            continue
        entry, record = run_stage(
            model=model, world=lw.world, stage=stage, seed=seed, encoders=spec713.encoders, spec=spec
        )  # fmt: skip
        entry.update(
            **fz.code_record(),
            counters=dict(lw.live.counters),
            peak_gpu_mb=torch.cuda.max_memory_allocated() / 2**20 if device.startswith("cuda") else None,
            peak_host_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        )  # fmt: skip
        log["stages"].append(entry)
        cm.write_json_atomic(obj=entry, path=stage_log)  # written before the state file
        tr.save_atomic(state=model.state_dict(), path=path)
        print(
            f"{arm} {month} s{seed} stage {stage}: {record.passes} passes, best pass "
            f"{entry['stopping']['best_pass']}, val {record.best_val:.4f}, peak host "
            f"{entry['peak_host_mb']:.0f} MB, peak GPU {entry['peak_gpu_mb']} MB",
            flush=True,
        )  # fmt: skip
    log["seconds"] = round(time.monotonic() - started, 1)
    log["peak_host_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    if device.startswith("cuda"):
        log["peak_gpu_mb"] = torch.cuda.max_memory_allocated() / 2**20
        log["nvidia_smi_process_mb"] = nvidia_smi_process_mb()
        assert gpu_gb_budget_ok(peak_mb=log["nvidia_smi_process_mb"]), (
            f"the process holds {log['nvidia_smi_process_mb']} MB of GPU memory, over the 6 GB budget"
        )
    log["counters"] = dict(lw.live.counters)
    log_path = cm.log_file(
        data_dir=data_dir, arm=arm, month=month, seed=seed, kind="fit", smoke=smoke
    )
    cm.write_json_atomic(obj=log, path=log_path)  # written before the read-only assertion
    if before != fz.watched_mtimes(files=watched):
        log["watch_failed"] = True
        cm.write_json_atomic(obj=log, path=log_path)
        msg = "a file that F8 reads from experiment 713's data changed during the fit"
        raise AssertionError(msg)
    return log


def main() -> int:
    """Parse the command line and fit."""
    parser = argparse.ArgumentParser()
    parser.add_argument("arm")
    parser.add_argument("month")
    parser.add_argument("seed", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--accept-code-change", action="append", default=[], metavar="OLD_HASH")
    args = parser.parse_args()
    fit(
        arm=args.arm, month=args.month, seed=args.seed, device=args.device,
        data_dir=ML_EXPERIMENTS_DATA_DIR, accept=args.accept_code_change, smoke=args.smoke,
    )  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
