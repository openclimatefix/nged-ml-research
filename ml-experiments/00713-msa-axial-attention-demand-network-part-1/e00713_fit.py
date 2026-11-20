# ruff: noqa: E501
"""Experiment 713: fit one MSA network (one arm, one fold, one seed) in the training phase.

Usage: e00713_fit.py ARM MONTH SEED [--device cuda] [--max-passes 60]

Run it through `run.sh`. It sets `E00667_PHASE=train`, under which `e00667_fold.install` sets the
cleaned power to NaN from the fold month's start, so no code path can read a held-out reading. It
refuses to run without `work/known_answers_passed.json`, which `e00713_known_answer.py` writes. A stage
that ends at the 60-pass cap is rerun from its starting state with a cap of 120 passes, and the log
records both. Each stage's state is saved atomically, and a stage whose file exists is skipped.
"""

import argparse
import copy
import json
import os
import resource
import sys
import time
from pathlib import Path

os.environ["E00667_PHASE"] = "train"

import e00713_arms as arms
import e00713_inputs as inp
import e00713_model as mdl
import e00713_train as tr
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

torch.set_num_threads(1)
CAP_FIRST, CAP_RERUN = 60, 120


def new_model(*, spec: arms.ArmSpec, seed: int, month: str, device: str) -> mdl.MsaNet:
    """Build the network of `spec` for (`seed`, `month`) and load or freeze its encoders."""
    torch.manual_seed(seed)
    model = mdl.MsaNet(switches=spec.switches)
    if spec.encoders in {"frozen", "trained"}:
        path = inp.pretrained_encoder_file(
            chain_root=inp.chain_root_of_seed(seed=seed), month=month, seed=seed
        )
        model.load_encoders(path=path)
    if spec.encoders == "frozen":
        model.freeze_encoders()
    return model.to(device)


def fit(*, arm: str, month: str, seed: int, device: str, data_dir: Path) -> dict:
    """Fit the three stages and return the log."""
    gate = arms.root(data_dir=data_dir) / "work" / "known_answers_passed.json"
    assert gate.exists(), f"{gate} is missing: run e00713_known_answer.py first"
    spec = arms.resolve(arm=arm, data_dir=data_dir)
    final = arms.model_file(data_dir=data_dir, arm=arm, month=month, seed=seed, stage=3)
    if final.exists():
        print(f"{arm} {month} s{seed}: done, skipped", flush=True)
        return {}
    started = time.monotonic()
    pipe = inp.load_pipeline(
        month=month, phase="train", chain_root=inp.chain_root_of_seed(seed=seed)
    )
    data = inp.build_fold_data(pipe=pipe, device=device)
    check_fingerprint(data=data, data_dir=data_dir, month=month)
    world = tr.World(
        source=data.source, cutoff_idx=data.cutoff_idx, clean=data.clean,
        target_mean=data.target_mean, target_std=data.target_std, month=month,
        placebo_pool=tr.block_pool(clean=data.clean),
    )  # fmt: skip
    model = new_model(spec=spec, seed=seed, month=month, device=device)
    log: dict = {
        "arm": arm,
        "month": month,
        "seed": seed,
        "stages": [],
        "dropped": data.dropped_per_fold,
    }
    for stage in (1, 2, 3):
        path = arms.model_file(data_dir=data_dir, arm=arm, month=month, seed=seed, stage=stage)
        stage_log = arms.log_file(data_dir=data_dir, arm=arm, month=month, seed=seed, kind="fit")
        stage_log = stage_log.with_name(f"{stage_log.stem}_stage{stage}.json")
        if path.exists():
            model.load_state_dict(torch.load(path, map_location=device))
            assert stage_log.exists(), f"{path} has no stage log: refit the stage"
            earlier = json.loads(stage_log.read_text())
            assert earlier.get("code_hash") == arms.code_hash(), (
                f"{path} was fitted by other code (hash {earlier.get('code_hash')}, now {arms.code_hash()}): refit it"
            )
            log["stages"].append(earlier)
            continue
        samples = tr.make_samples(world=world, stage=stage, placebo=spec.placebo)
        start_state = copy.deepcopy(model.state_dict())
        record = tr.train_stage(
            model=model, world=world, stage=stage, samples=samples, seed=seed,
            encoders=spec.encoders, max_passes=CAP_FIRST,
        )  # fmt: skip
        entry = {"stage": stage, "pairs": len(samples), "first": vars(record).copy(), "rerun": None}
        if record.cap_hit:
            model.load_state_dict(start_state)
            again = tr.train_stage(
                model=model, world=world, stage=stage, samples=samples, seed=seed,
                encoders=spec.encoders, max_passes=CAP_RERUN,
            )  # fmt: skip
            entry["rerun"] = vars(again).copy()
            record = again
        entry.update(
            passes=record.passes,
            best_val=record.best_val,
            cap_hit=record.cap_hit,
            code_hash=arms.code_hash(),
        )
        log["stages"].append(entry)
        stage_log.parent.mkdir(parents=True, exist_ok=True)
        tmp_stage = stage_log.with_name(stage_log.name + f".{os.getpid()}.tmp")
        tmp_stage.write_text(json.dumps(entry, indent=1, default=str))
        tmp_stage.rename(
            stage_log
        )  # written before the state file, so a state file always has its log
        tr.save_atomic(state=model.state_dict(), path=path)
        print(
            f"{arm} {month} s{seed} stage {stage}: {record.passes} passes, val {record.best_val:.4f}",
            flush=True,
        )
    log["seconds"] = round(time.monotonic() - started, 1)
    log["peak_host_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    if device.startswith("cuda"):
        log["peak_gpu_mb"] = torch.cuda.max_memory_allocated() / 2**20
    log_path = arms.log_file(data_dir=data_dir, arm=arm, month=month, seed=seed, kind="fit")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = log_path.with_name(log_path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(log, indent=1, default=str))
    tmp.rename(log_path)
    return log


def check_fingerprint(*, data: inp.FoldData, data_dir: Path, month: str) -> None:
    """Write the fold's standardiser hash, or assert it is unchanged (the predict phase compares it)."""
    path = arms.root(data_dir=data_dir) / "work" / "standardiser" / f"{month}.sha256"
    if path.exists():
        assert path.read_text() == data.fingerprint, "the standardiser changed between two fits"
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(data.fingerprint)
    tmp.rename(path)


def main() -> int:
    """Parse the command line and fit."""
    parser = argparse.ArgumentParser()
    parser.add_argument("arm")
    parser.add_argument("month")
    parser.add_argument("seed", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    fit(
        arm=args.arm,
        month=args.month,
        seed=args.seed,
        device=args.device,
        data_dir=ML_EXPERIMENTS_DATA_DIR,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
