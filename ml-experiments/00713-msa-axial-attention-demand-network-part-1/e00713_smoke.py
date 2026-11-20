# ruff: noqa: E501, ANN001, ANN202, D103
"""Experiment 713 smoke job (code review 3): train, crash, resume, rebuild, predict, and score one tiny fit.

The script is part of the hashed fit modules (`arms.FIT_CODE_FILES`). It imports the experiment's own modules unchanged and narrows
three things only: the training pairs to the six signed-MW Primaries (`make_samples(primaries=...)`),
the pass caps (`CAP_FIRST`, `CAP_RERUN`), and the data folder (everything under
`00713-*/smoke/`, never `00713-*/work/`). Run each mode in its own process through `run.sh e00713_smoke.py MODE`; the real fit's commands follow
the same path. Delete `00713-*/smoke/` afterwards.

Modes:
    count             build the fold (train phase) and print the pair and step counts per stage; no training
    fit               fit the three stages; `--crash-after 2` exits after the stage 2 state is saved
    drop-final-log    delete the final fit log (the stage logs stay), to exercise the rebuild
    predict           forecast the fold (predict phase) from the smoke stage 3 state
    score             score the smoke M1 against the Candidate (seed 0, one fold)
"""

import argparse
import functools
import json
import sys
from pathlib import Path

MONTH, SEED, ARM = "2025-11", 0, "M1"
CAPS = (
    3,
    4,
)  # (CAP_FIRST, CAP_RERUN): 3 passes, then a rerun of 4 when the cap is hit (always, patience is 8)
GPU_CAP_GB = 3.0
"""`torch.cuda.set_per_process_memory_fraction` makes a job that needs more than this fail with an out-of-memory error."""


def cap_gpu(device: str) -> None:
    import torch  # ty: ignore[unresolved-import]

    if device.startswith("cuda"):
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(GPU_CAP_GB * 2**30 / total)


def smoke_dir() -> Path:
    import e00713_arms as arms
    from studies.sources import ML_EXPERIMENTS_DATA_DIR

    return ML_EXPERIMENTS_DATA_DIR / arms.NAME / "smoke"


def six_indices() -> tuple[int, ...]:
    import e00578_common as k578  # ty: ignore[unresolved-import]
    import e00713_score as score

    labels = k578.sites().sort("primary")["label"].to_list()
    return tuple(i for i, label in enumerate(labels) if label in score.SIX)


def narrow_training(*, tr) -> None:
    tr.make_samples = functools.partial(tr.make_samples, primaries=six_indices())


def write_gate() -> None:
    import e00713_arms as arms

    gate = arms.root(data_dir=smoke_dir()) / "work" / "known_answers_passed.json"
    gate.parent.mkdir(parents=True, exist_ok=True)
    gate.write_text(
        json.dumps({"SMOKE_GATE_NOT_A_KNOWN_ANSWER_RUN": True, "code": arms.code_record()})
    )


def mode_count(device: str) -> None:
    import e00713_fit  # noqa: F401  (sets E00667_PHASE=train)
    import e00713_inputs as inp
    import e00713_train as tr

    narrow_training(tr=tr)
    pipe = inp.load_pipeline(
        month=MONTH, phase="train", chain_root=inp.chain_root_of_seed(seed=SEED)
    )
    data = inp.build_fold_data(pipe=pipe, device=device)
    world = tr.World(
        source=data.source,
        cutoff_idx=data.cutoff_idx,
        clean=data.clean,
        target_mean=data.target_mean,
        target_std=data.target_std,
        month=MONTH,
        placebo_pool=tr.block_pool(clean=data.clean),
    )
    for stage in (1, 2, 3):
        s = tr.make_samples(world=world, stage=stage, placebo=False)
        n_train = int((~s.is_val).sum())
        print(
            f"stage {stage}: {len(s.primary)} pairs, {n_train} training, {-(-n_train // tr.BATCH)} steps per pass"
        )


def mode_fit(device: str, crash_after: int | None) -> None:
    import e00713_fit as f
    import e00713_train as tr

    narrow_training(tr=tr)
    f.CAP_FIRST, f.CAP_RERUN = CAPS  # ty: ignore[invalid-assignment, invalid-argument-type]
    write_gate()
    if crash_after is not None:
        save = tr.save_atomic

        def save_then_crash(*, state, path):
            save(state=state, path=path)
            if path.name.endswith(f"_stage{crash_after}.pt"):
                print(f"SMOKE: simulated crash after {path.name}", flush=True)
                sys.exit(3)

        tr.save_atomic = save_then_crash  # ty: ignore[invalid-assignment, invalid-argument-type]
    log = f.fit(arm=ARM, month=MONTH, seed=SEED, device=device, data_dir=smoke_dir())
    print(
        "SMOKE fit log:",
        json.dumps({k: v for k, v in log.items() if k != "stages"}, default=str)[:800],
    )
    print(
        "SMOKE stages:",
        [
            (s.get("stage"), s.get("passes"), s.get("cap_hit"), s.get("code_hash"))
            for s in log.get("stages", [])
        ],
    )


def mode_drop_final_log() -> None:
    import e00713_arms as arms

    path = arms.log_file(data_dir=smoke_dir(), arm=ARM, month=MONTH, seed=SEED, kind="fit")
    assert "smoke" in path.parts, path
    path.unlink()
    print("SMOKE: removed", path)


def mode_predict(device: str) -> None:
    import e00713_predict as p

    out = p.predict(
        arm=ARM, month=MONTH, seed=SEED, device=device, mask_history=False, data_dir=smoke_dir()
    )
    import polars as pl

    frame = pl.read_parquet(out)
    print("SMOKE predict:", out, frame.height, "rows; columns", frame.columns)
    print(frame.select("q01", "q50", "q99", "actual").describe())


def mode_score() -> None:
    import e00713_arms as arms
    import e00713_score as score

    original = arms.fit_file
    arms.fit_file = lambda **kw: original(**{**kw, "data_dir": smoke_dir()})  # ty: ignore[invalid-assignment, invalid-argument-type]
    res = score.run_contrasts(
        star=ARM, months=(MONTH,), seeds=(SEED,), pairs=[(ARM, "C")], level=95.0
    )
    for g in ("six", "eight"):
        c = res[f"{ARM}-C:{g}"]
        print(
            f"SMOKE score {g}: point {c['point']:+.3f} points of CRPS, rows {c['n_rows']}, common keys {res['n_common_keys']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("count", "fit", "drop-final-log", "predict", "score"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--crash-after", type=int, default=None)
    args = parser.parse_args()
    if args.mode in {"count", "fit", "predict"}:
        cap_gpu(args.device)
    if args.mode == "count":
        mode_count(args.device)
    elif args.mode == "fit":
        mode_fit(args.device, args.crash_after)
    elif args.mode == "drop-final-log":
        mode_drop_final_log()
    elif args.mode == "predict":
        mode_predict(args.device)
    else:
        mode_score()
    return 0


if __name__ == "__main__":
    sys.exit(main())
