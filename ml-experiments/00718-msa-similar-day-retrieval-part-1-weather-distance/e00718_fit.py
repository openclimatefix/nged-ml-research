# ruff: noqa: E501, PLR0915
"""Experiment 718: fit one network (one arm, one fold, one seed) in the training phase (design sections 6 and 11).

Usage: e00718_fit.py ARM MONTH SEED [--device cuda] [--smoke] [--accumulate N] [--accept-code-change OLD_HASH ...] [--accept-gate-change OLD_HASH]

Run it through `run.sh`. It sets `E00667_PHASE=train`, under which the fold month's cleaned power is
NaN, so no code path can read a held-out reading. The row tables are read from
`work/tables/<arm>/<month>_{era5,ens}.npz`, which `e00718_tables.py` builds in the prediction phase,
and their hashes go into the log. Every arm is fitted with all three stages from the seed's own
initialisation (design section 6.1); the encoders of `M*` are loaded from the Candidate chain and
frozen. A real fit refuses to start without `work/known_answers_passed.json` (part 1's gate). The
`--smoke` path skips the gate, writes under `work/smoke/`, stops each stage after 2 passes, and its
outputs are never scored. A stage that ends at the 60-pass cap is rerun from its starting state with
a cap of 120 passes. After the fit the modification times of experiment 713's `work` and `cache`
files are asserted unchanged.

**Code policy.** Every stage log carries the code hash, the git commit, and a dirty flag. A resume, a
finished-job skip, and the gate file are refused when their hash differs from the current code's,
unless `--accept-code-change OLD_HASH` (stage states, repeatable) or `--accept-gate-change OLD_HASH`
(the gate record) names the exact old hash.
"""

import argparse
import copy
import json
import os
import resource
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("E00667_PHASE", "train")

import e00713_inputs as inp  # ty: ignore[unresolved-import]
import e00713_train as tr  # ty: ignore[unresolved-import]
import e00718_arms as arms
import e00718_batch as fb
import e00718_model as f9
import e00718_retrieval as ret
import e00718_train as ftr
import numpy as np
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

torch.set_num_threads(1)
CAP_FIRST, CAP_RERUN = 60, 120
SMOKE_PASSES = 2


@dataclass
class Loaded:
    """What a fit reads: the world, the standardiser fingerprint, and the dropped counts."""

    world: tr.World
    fingerprint: str
    dropped: dict


Loader = Callable[[str, int, str], Loaded]


def real_loader(month: str, seed: int, device: str) -> Loaded:
    """Load the fold in the training phase from the Candidate pipeline (needs the stored data)."""
    pipe = inp.load_pipeline(
        month=month, phase="train", chain_root=inp.chain_root_of_seed(seed=seed)
    )
    data = inp.build_fold_data(pipe=pipe, device=device)
    world = tr.World(
        source=data.source, cutoff_idx=data.cutoff_idx, clean=data.clean, target_mean=data.target_mean,
        target_std=data.target_std, month=month, placebo_pool=None,
    )  # fmt: skip
    return Loaded(world=world, fingerprint=data.fingerprint, dropped=data.dropped_per_fold)


def save_table(*, table: ret.RowTable, path: Path) -> None:
    """Write a row table atomically, with its digest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp.npz")
    np.savez(
        tmp,
        arm=table.arm,
        primaries=table.primaries,
        days=table.days,
        ages=table.ages,
        measure=table.measure,
        no_weather=table.no_weather,
        digest=table.digest(),
    )
    tmp.rename(path)


def load_table(*, path: Path) -> ret.RowTable:
    """Read a row table and check its digest."""
    z = np.load(path)
    table = ret.RowTable(
        arm=str(z["arm"]),
        primaries=z["primaries"],
        days=z["days"],
        ages=z["ages"],
        measure=z["measure"],
        no_weather=z["no_weather"],
    )
    assert table.digest() == str(z["digest"]), f"{path} does not match its digest"
    return table


def new_model(
    *, seed: int, month: str, device: str, data_dir: Path, load_encoders: bool = True
) -> f9.F9Net:
    """Build the network for `seed` and load and freeze the encoders of `M*`."""
    torch.manual_seed(seed)
    model = f9.F9Net(seed=seed)
    mode = arms.encoders_mode(data_dir=data_dir) if load_encoders else "random"
    if mode in {"frozen", "trained"}:
        model.load_encoders(
            path=inp.pretrained_encoder_file(
                chain_root=inp.chain_root_of_seed(seed=seed), month=month, seed=seed
            )
        )
    if mode == "frozen":
        model.freeze_encoders()
    return model.to(device)


def check_fingerprint(*, fingerprint: str, data_dir: Path, month: str) -> None:
    """Write the fold's standardiser hash (under part 1's folder), or assert it is unchanged; the prediction phase compares it."""
    path = arms.root(data_dir=data_dir, part=1) / "work" / "standardiser" / f"{month}.sha256"
    if path.exists():
        assert path.read_text() == fingerprint, "the standardiser changed between two fits"
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(fingerprint)
    tmp.rename(path)


def write_json_atomic(*, obj: dict, path: Path) -> None:
    """Write a JSON file under a temporary name that carries the process id, then rename it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    tmp.rename(path)


def stage_log_file(
    *, data_dir: Path, arm: str, month: str, seed: int, stage: int, smoke: bool
) -> Path:
    """Return the log of one stage."""
    path = arms.log_file(
        data_dir=data_dir, arm=arm, month=month, seed=seed, kind="fit", smoke=smoke
    )
    return path.with_name(f"{path.stem}_stage{stage}.json")


def fit(
    *, arm: str, month: str, seed: int, device: str, data_dir: Path, accept: Sequence[str] = (),
    accept_gate: str | None = None, smoke: bool = False, accumulate: int = 1,
    loader: Loader = real_loader, make_samples: Callable = tr.make_samples,
    encoders: str | None = None, tables: dict[str, ret.RowTable] | None = None,
) -> dict:  # fmt: skip
    """Fit the three stages and return the log. `loader`, `make_samples`, `encoders`, and `tables` exist for the checks."""
    spec = arms.resolve(arm=arm, accumulate=accumulate)
    log: dict = {"arm": arm, "month": month, "seed": seed, "smoke": smoke, "accumulate": accumulate}
    if not smoke:
        gate = arms.gate_file(data_dir=data_dir)
        assert gate.exists(), f"{gate} is missing: run e00718_known_answer.py first"
        arms.check_code(
            earlier=json.loads(gate.read_text()).get("code", {}),
            accept=[accept_gate] if accept_gate else [],
            record=log,
            what="the known-answer gate file",
            kind="gate",
        )
    final = arms.model_file(
        data_dir=data_dir, arm=arm, month=month, seed=seed, stage=3, smoke=smoke
    )
    stage_logs = {
        s: stage_log_file(data_dir=data_dir, arm=arm, month=month, seed=seed, stage=s, smoke=smoke)
        for s in (1, 2, 3)
    }
    final_log = arms.log_file(
        data_dir=data_dir, arm=arm, month=month, seed=seed, kind="fit", smoke=smoke
    )
    if final.exists():
        stages = []
        for s, path in stage_logs.items():
            assert path.exists(), f"{path} is missing: refit the job"
            earlier = json.loads(path.read_text())
            arms.check_code(
                earlier=earlier, accept=accept, record=log, what=f"{arm} {month} s{seed} stage {s}"
            )
            stages.append(earlier)
        if not final_log.exists():
            write_json_atomic(
                obj={
                    **log,
                    "stages": stages,
                    "rebuilt_from_stage_logs": True,
                    **arms.code_record(),
                },
                path=final_log,
            )
        print(f"{arm} {month} s{seed}: done, skipped", flush=True)
        return json.loads(final_log.read_text())
    started = time.monotonic()
    before = arms.watched_713_files(data_dir=data_dir)
    loaded = loader(month, seed, device)
    world = loaded.world
    if not smoke:
        check_fingerprint(fingerprint=loaded.fingerprint, data_dir=data_dir, month=month)
    if tables is None:
        tables = {
            v: load_table(path=arms.table_file(data_dir=data_dir, arm=arm, month=month, version=v))
            for v in (fb.TABLE_ERA5, fb.TABLE_ENS)
        }
    log.update(
        stages=[],
        dropped=loaded.dropped,
        table_digests={v: t.digest() for v, t in tables.items()},
        recipe=arms.recipe_hashes(),
        **arms.code_record(),
    )
    mode = encoders or arms.encoders_mode(data_dir=data_dir)
    model = new_model(
        seed=seed, month=month, device=device, data_dir=data_dir, load_encoders=encoders is None
    )
    for stage in (1, 2, 3):
        path = arms.model_file(
            data_dir=data_dir, arm=arm, month=month, seed=seed, stage=stage, smoke=smoke
        )
        if path.exists():
            model.load_state_dict(torch.load(path, map_location=device))
            assert stage_logs[stage].exists(), f"{path} has no stage log: refit the stage"
            earlier = json.loads(stage_logs[stage].read_text())
            arms.check_code(earlier=earlier, accept=accept, record=log, what=path.name)
            log["stages"].append(earlier)
            continue
        samples = make_samples(world=world, stage=stage, placebo=False)
        start_state = copy.deepcopy(model.state_dict())
        cap = SMOKE_PASSES if smoke else CAP_FIRST
        record = ftr.train_stage(
            model=model,
            world=world,
            tables=tables,
            arm=arm,
            stage=stage,
            samples=samples,
            seed=seed,
            encoders=mode,
            max_passes=cap,
            accumulate=spec.accumulate,
        )
        entry = {"stage": stage, "pairs": len(samples), "first": vars(record).copy(), "rerun": None}
        if record.cap_hit and not smoke:
            model.load_state_dict(start_state)
            record = ftr.train_stage(
                model=model,
                world=world,
                tables=tables,
                arm=arm,
                stage=stage,
                samples=samples,
                seed=seed,
                encoders=mode,
                max_passes=CAP_RERUN,
                accumulate=spec.accumulate,
            )
            entry["rerun"] = vars(record).copy()
        entry.update(
            passes=record.passes,
            best_val=record.best_val,
            cap_hit=record.cap_hit,
            smoke=smoke,
            **arms.code_record(),
            recipe=arms.recipe_hashes(),
        )
        log["stages"].append(entry)
        write_json_atomic(
            obj=entry, path=stage_logs[stage]
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
    write_json_atomic(obj=log, path=final_log)
    after = arms.watched_713_files(data_dir=data_dir)
    assert before == after, "a file of experiment 713's work or cache folder changed during the fit"
    return log


def main() -> int:
    """Parse the command line and fit."""
    parser = argparse.ArgumentParser()
    parser.add_argument("arm")
    parser.add_argument("month")
    parser.add_argument("seed", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--accept-code-change", action="append", default=[], metavar="OLD_HASH")
    parser.add_argument("--accept-gate-change", default=None, metavar="OLD_HASH")
    args = parser.parse_args()
    data_dir = ML_EXPERIMENTS_DATA_DIR
    arms.install_own_cache(data_dir=data_dir)
    fit(
        arm=args.arm, month=args.month, seed=args.seed, device=args.device, data_dir=data_dir,
        accept=args.accept_code_change, accept_gate=args.accept_gate_change, smoke=args.smoke,
        accumulate=args.accumulate,
    )  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
