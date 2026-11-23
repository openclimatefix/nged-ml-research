# ruff: noqa: PT017, E501, PLR0915, PT018
"""Experiment 718, WP5b: the known-answer gate (design section 10.2): checks KS1 to KS3 on worlds WS and WN.

Usage: e00718_known_answer.py MONTH [--device cuda]   # the real gate: needs the fold's stored data and the GPU
       e00718_known_answer.py --selftest              # CPU only: the checks' logic, and a two-pass fit in a tiny world

Each gating fit is stage 1 alone on a world's samples, for a fixed 120 passes with no early stopping. The
metric is the MAE of the median on the 60 held-out origin days after the fold's cutoff (all 14 lead days).
For each seed the metric is the median over the evaluations after passes 80, 90, 100, 110, and 120, and a
check reads the median over seeds 0, 1, and 2. The pre-check computes the NumPy references' gaps under the
era layout before any fit, and stops with no fit if one gap is below 0.010. The script writes
`work/known_answers_passed.json` only if every check that it can read passes. A KS3 precondition that
fails leaves the two difference tests unread and sends them to the manager; the floor test stays
unconditional. WS+, `U27` in WS, and the frozen-encoder fit are not part of this script.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final

import e00713_layout as lay713  # ty: ignore[unresolved-import]
import e00713_model as mdl  # ty: ignore[unresolved-import]
import e00713_train as tr  # ty: ignore[unresolved-import]
import e00718_arms as arms
import e00718_batch as fb
import e00718_layout as lay
import e00718_model as f9
import e00718_retrieval as ret
import e00718_train as ftr
import e00718_worlds as worlds
import numpy as np
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

torch.set_num_threads(1)
PASSES: Final = 120
EVAL_PASSES: Final = (80, 90, 100, 110, 120)
SEEDS: Final = (0, 1, 2)
THRESHOLDS: Final = {
    "ks1_fraction": 0.5,
    "ks2_fraction": 0.5,
    "ks3_difference": 0.005,
    "ks3_precondition_slack": 0.02,
    "ks3_floor_fraction": 0.95,
}
WS_ARMS: Final = ("Z", "R", "P90", "U", "U-A")
WN_ARMS: Final = ("Z", "R", "U", "N")
"""`Z` is the last 14 days (`M` in the design's table of checks)."""


def real_budget_pass(
    *, val_history: Sequence[float], patience: int = tr.PATIENCE, cap: int = 60
) -> int:
    """Return the pass (1-based) that experiment 713's rule (patience 8, at most 60 passes) would have kept."""
    best, best_pass, stale = float("inf"), 1, 0
    for i, v in enumerate(val_history[:cap], start=1):
        if v < best - 1e-5:
            best, best_pass, stale = v, i, 0
        else:
            stale += 1
            if stale >= patience:
                break
    return best_pass


def median_mae(
    *,
    model: torch.nn.Module,
    batch_fn: Callable,
    source: object,
    held: tr.SampleSet,
    mean: np.ndarray,
    std: np.ndarray,
) -> float:
    """Return the MAE of the median (shares of capacity) on the held-out samples at all 14 lead days."""
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for lead in range(lay713.N_LEAD_DAYS):
            for start in range(0, len(held.primary), 512):
                sel = np.arange(start, min(start + 512, len(held.primary)))
                b = batch_fn(
                    source=source,
                    samples=held,
                    index=sel,
                    lead=np.full(len(sel), lead),
                    ensemble=False,
                    cut=10**9,
                    dropout=None,
                )
                q = mdl.standardised_quantiles(out=model(b.inputs).out)[..., 5].cpu().numpy()
                p = held.primary[sel]
                med = q * std[p][:, None] + mean[p][:, None]
                act = b.target.cpu().numpy() * std[p][:, None] + mean[p][:, None]
                ok = np.isfinite(act)
                total += float(np.abs(med[ok] - act[ok]).sum())
                count += int(ok.sum())
    return total / max(count, 1)


def fit_world_arm(
    *, arm: str, world: tr.World, mean: np.ndarray, std: np.ndarray, held: tr.SampleSet, table: ret.RowTable | None,
    seed: int, passes: int = PASSES, eval_passes: Sequence[int] = EVAL_PASSES, device: str = "cpu", max_steps: int | None = None,
    samples: tr.SampleSet | None = None,
) -> dict:  # fmt: skip
    """Fit stage 1 of `arm` for a fixed number of passes with no early stopping and return the MAE trajectory and the metric."""
    torch.manual_seed(seed)
    if samples is None:
        samples = tr.make_samples(world=world, stage=1, placebo=False, primaries=worlds.PRIMARIES)
    trajectory: list[float] = []
    if arm == "N":
        model = mdl.MsaNet(switches=mdl.ARM_SWITCHES["N"]).to(device)
        source, batch_fn = world.source, tr.batch_for
    else:
        assert table is not None
        model = f9.F9Net(seed=seed).to(device)
        source, batch_fn = (
            fb.F9Source(base=world.source, tables={fb.TABLE_ERA5: table, fb.TABLE_ENS: table}),
            ftr.batch_for,
        )

    def on_pass(n: int, net: torch.nn.Module) -> None:
        trajectory.append(
            median_mae(model=net, batch_fn=batch_fn, source=source, held=held, mean=mean, std=std)
        )

    if arm == "N":
        log = tr.train_stage(
            model=model,
            world=world,
            stage=1,
            samples=samples,
            seed=seed,
            encoders="random",
            max_passes=passes,
            patience=10**6,
            on_pass=on_pass,
        )
    else:
        log = ftr.train_stage(
            model=model,
            world=world,
            tables={fb.TABLE_ERA5: table, fb.TABLE_ENS: table},
            arm=arm,
            stage=1,
            samples=samples,
            seed=seed,
            encoders="random",
            max_passes=passes,
            patience=10**6,
            on_pass=on_pass,
            max_steps=max_steps,
        )  # ty: ignore[invalid-argument-type]
    picks = [trajectory[p - 1] for p in eval_passes if p <= len(trajectory)]
    return {
        "arm": arm, "seed": seed, "trajectory": trajectory, "val_history": list(log.val_history),
        "metric": float(np.median(picks)) if picks else float("nan"),
        "real_budget_pass": real_budget_pass(val_history=log.val_history),
        "metric_at_real_budget_pass": float(trajectory[real_budget_pass(val_history=log.val_history) - 1]) if trajectory else float("nan"),
    }  # fmt: skip


def arm_score(*, fits: dict[str, list[dict]], arm: str) -> float:
    """Return the median over seeds of an arm's per-seed metric."""
    return float(np.median([f["metric"] for f in fits[arm]]))


def evaluate_checks(
    *, ws: dict[str, float], wn: dict[str, float], gaps: dict[str, float], floor: float
) -> dict:
    """Return KS1 to KS3 from the scores of the arms in worlds WS and WN (`Z` is `M`), the references' gaps, and WN's floor."""
    t = THRESHOLDS
    out: dict = {}
    ks1_a, ks1_b = ws["P90"] - ws["R"], ws["Z"] - ws["R"]
    out["KS1"] = {
        "P90_minus_R": ks1_a, "need_P90_minus_R": t["ks1_fraction"] * gaps["gap_R"], "M_minus_R": ks1_b, "need_M_minus_R": t["ks1_fraction"] * gaps["gap_RM"],
        "pass": bool(ks1_a >= t["ks1_fraction"] * gaps["gap_R"] and ks1_b >= t["ks1_fraction"] * gaps["gap_RM"]),
    }  # fmt: skip
    ks2 = ws["U-A"] - ws["U"]
    out["KS2"] = {
        "UA_minus_U": ks2,
        "need": t["ks2_fraction"] * gaps["gap_U"],
        "pass": bool(ks2 >= t["ks2_fraction"] * gaps["gap_U"]),
    }
    precondition = wn["N"] <= t["ks3_precondition_slack"] + floor
    floor_ok = all(v >= t["ks3_floor_fraction"] * floor for v in wn.values())
    diffs = {"R_minus_M": abs(wn["R"] - wn["Z"]), "U_minus_M": abs(wn["U"] - wn["Z"])}
    read = bool(precondition)
    out["KS3"] = {
        "precondition_N_within_0.02_of_floor": bool(precondition), "N": wn["N"], "floor": floor, "floor_test_pass": bool(floor_ok),
        "differences": diffs if read else "unread: the precondition failed, send to the manager",
        "differences_pass": bool(all(d <= t["ks3_difference"] for d in diffs.values())) if read else None,
        "pass": bool(floor_ok and (not read or all(d <= t["ks3_difference"] for d in diffs.values()))),
    }  # fmt: skip
    out["all_pass"] = bool(out["KS1"]["pass"] and out["KS2"]["pass"] and out["KS3"]["pass"])
    out["ks3_unread"] = not read
    return out


def write_record(*, path: Path, result: dict) -> Path:
    """Write a record of the gate's result atomically, with the code record and the gate hash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    code = {**arms.code_record(), "gate_hash": arms.gate_hash()}
    tmp.write_text(
        json.dumps(
            {"code": code, "recipe": arms.recipe_hashes(), "result": result},
            indent=1,
            default=float,
        )
    )
    tmp.rename(path)
    return path


def write_gate(*, data_dir: Path, result: dict) -> Path:
    """Write the gate file that every real fit requires; the fit reads its `code` record."""
    return write_record(path=arms.gate_file(data_dir=data_dir), result=result)


def finish(*, data_dir: Path, result: dict, manager_note: Path | None = None) -> int:
    """Write the result record in every case, and the gate file only when every check passes and KS3 was read.

    A passing run whose KS3 precondition failed writes the gate file only when
    `manager_note` names the manager's written decision; the note's path and sha256 go into the result
    record, the gate file, and every fit log.
    """
    checks = result.get("checks")
    ok = bool(checks and checks["all_pass"])
    read = bool(checks and not checks["ks3_unread"])
    accepted = manager_note is not None
    if manager_note is not None:
        assert manager_note.is_file(), f"{manager_note} is not a file"
        result["manager_note"] = {
            "path": str(manager_note),
            "sha256": hashlib.sha256(manager_note.read_bytes()).hexdigest(),
        }
    result["gate_written"] = bool(ok and (read or accepted))
    result["manager_accepts_unread_ks3"] = accepted
    path = write_record(path=arms.result_file(data_dir=data_dir), result=result)
    print("result record:", path)
    if not ok:
        print("a known-answer check failed or the run stopped: no gate file; go to the manager")
        return 1
    if not read and not accepted:
        print(
            "KS3's difference tests are unread (the precondition failed): no gate file; send them to the manager"
        )
        return 1
    print("gate file:", write_gate(data_dir=data_dir, result=result))
    return 0


def tables_for(
    *, arm_list: Sequence[str], template: object, arrays: worlds.WorldArrays, days: np.ndarray
) -> dict[str, ret.RowTable]:
    """Return the stage 1 row table of every F9 arm in `arm_list` for a world (the placebo `U-A` needs the table of its own arm)."""
    return {
        a: worlds.world_tables(arm=a, template=template, arrays=arrays, days=days)
        for a in arm_list
        if a != "N"
    }  # ty: ignore[invalid-argument-type]


def gaps_of(
    *, arrays: worlds.WorldArrays, tables: dict[str, ret.RowTable], held: tr.SampleSet
) -> dict:
    """Return the three gaps of the pre-check from the NumPy references (computed before any fit)."""
    pi = {p: i for i, p in enumerate(tables["R"].primaries.tolist())}
    di = {d: i for i, d in enumerate(tables["R"].days.tolist())}

    def rows(arm: str) -> np.ndarray:
        t = tables[arm]
        return np.stack(
            [
                t.ages[pi[int(p)], di[int(c) // lay713.STEPS_PER_DAY]]
                for p, c in zip(held.primary, held.c_idx, strict=True)
            ]
        )

    mae = {
        a: worlds.reference_gap(arrays=arrays, rows_a=rows(a), rows_b=rows(a), samples=held)[0]
        for a in ("R", "P90", "Z", "U", "U-A")
    }
    return {
        "gap_R": mae["P90"] - mae["R"],
        "gap_RM": mae["Z"] - mae["R"],
        "gap_U": mae["U-A"] - mae["U"],
        "reference_mae": mae,
    }


def _check_finish(*, good: dict, bad_wn: dict, good_ws: dict, gaps: dict) -> None:
    """Check that the result record is always written and the gate file only for a read, passing run."""
    import tempfile

    unread = evaluate_checks(ws=good_ws, wn=bad_wn, gaps=gaps, floor=0.04)
    assert unread["ks3_unread"], unread
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        gate, record = arms.gate_file(data_dir=root), arms.result_file(data_dir=root)
        note = root / "note.txt"
        note.write_text("the manager accepts KS3 unread, test")
        assert finish(data_dir=root, result={"checks": unread}) == 1
        assert record.exists()
        assert not gate.exists(), "the gate file was written with KS3 unread"
        assert finish(data_dir=root, result={"checks": unread}, manager_note=note) == 0
        assert json.loads(record.read_text())["result"]["manager_note"]["sha256"]
        assert json.loads(gate.read_text())["result"]["manager_note"]["path"] == str(note)
        assert json.loads(gate.read_text())["code"]["gate_hash"] == arms.gate_hash()
        gate.unlink()
        assert finish(data_dir=root, result={"checks": good}) == 0 and gate.exists()
        gate.unlink()
        assert finish(data_dir=root, result={"stopped": "pre-check"}) == 1 and not gate.exists()
        bad = {**good, "all_pass": False}
        assert finish(data_dir=root, result={"checks": bad}) == 1 and not gate.exists()
        stale = {"gate_hash": "old"}
        try:
            arms.check_code(earlier=stale, accept=[], record={}, what="gate", kind="gate")
            raise SystemExit("a gate file with another gate hash was accepted")
        except AssertionError as err:
            assert "--accept-gate-change" in str(err)


def selftest() -> int:
    """Check the checks' logic, the real-budget rule, and a two-pass fit of arms `Z`, `R`, and `N` in a tiny world on the CPU."""
    good_ws = {"Z": 0.12, "R": 0.03, "P90": 0.10, "U": 0.03, "U-A": 0.09, "N": 0.2}
    good_wn = {"Z": 0.045, "R": 0.047, "U": 0.046, "N": 0.05}
    gaps = {"gap_R": 0.033, "gap_RM": 0.023, "gap_U": 0.05}
    res = evaluate_checks(ws=good_ws, wn=good_wn, gaps=gaps, floor=0.04)
    assert res["all_pass"] and not res["ks3_unread"], res
    _check_finish(good=res, bad_wn={**good_wn, "N": 0.2}, good_ws=good_ws, gaps=gaps)
    assert not evaluate_checks(ws={**good_ws, "R": 0.099}, wn=good_wn, gaps=gaps, floor=0.04)[
        "KS1"
    ]["pass"]
    assert not evaluate_checks(ws={**good_ws, "U": 0.085}, wn=good_wn, gaps=gaps, floor=0.04)[
        "KS2"
    ]["pass"]
    bad_floor = evaluate_checks(ws=good_ws, wn={**good_wn, "R": 0.03}, gaps=gaps, floor=0.04)
    assert not bad_floor["KS3"]["floor_test_pass"], "an arm below 0.95 x the floor passed"
    leak = evaluate_checks(ws=good_ws, wn={**good_wn, "R": 0.060}, gaps=gaps, floor=0.04)
    assert not leak["KS3"]["pass"], "a difference of 0.015 passed"
    unread = evaluate_checks(ws=good_ws, wn={**good_wn, "N": 0.12}, gaps=gaps, floor=0.04)
    assert (
        unread["ks3_unread"]
        and unread["KS3"]["differences_pass"] is None
        and unread["KS3"]["floor_test_pass"]
    )
    assert real_budget_pass(val_history=[3, 2, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3]) == 3
    assert real_budget_pass(val_history=[3.0, 2.9999995, 2.99, 2.98]) == 4
    import e00718_check_batch as cb

    template = cb.synthetic_source()
    n_grid = template.n_grid
    cutoff = 450 * lay713.STEPS_PER_DAY + 24
    arr = worlds.build_ws(template=template, cutoff=cutoff)
    assert arr.y.shape == (16, n_grid) and np.isfinite(arr.y[2]).all() and np.isnan(arr.y[0]).all()
    assert abs(float(np.nanmean(arr.y[2])) - 0.5) < 0.1
    assert (
        arr.weather[2, 24 : 24 + 48, 0].unique().numel() == 1
        and arr.weather[2, 24 : 24 + 48, 1].unique().numel() == 1
    )
    assert torch.equal(
        arr.weather[2, 24:72, worlds.TEMPERATURE_COLUMN],
        template.dense_weather[2, 24:72, worlds.TEMPERATURE_COLUMN],
    )
    assert float(arr.weather[..., 0].float().abs().max()) <= 15.5
    q_block = int(arr.regime[2, worlds.block_of(t=np.array([100]))[0]])
    assert float(arr.weather[2, 100, 0]) == q_block - 15.5
    assert float(arr.datetime[..., worlds.DATETIME_NAMES.index("covid")].abs().max()) == 0.0
    assert worlds.datetime_names_match_prep()
    # the last era starts 400 days before the cutoff and the targets of the last era are masked in training
    world, mean, std = worlds.to_world(arrays=arr, template=template, device="cpu")
    assert tr.target_end(world=world, stage=1) == arr.boundary == cutoff - worlds.LAST_ERA_DAYS * 48
    held = worlds.held_out_samples(arrays=arr)
    assert int(held.c_idx.min()) // 48 == cutoff // 48 and len(held.primary) == 4 * 60
    # the weather path keeps the 32 regimes apart (no clipping or saturation up to +-15.5)
    net = mdl.MsaNet()
    x = torch.zeros(worlds.N_REGIMES, mdl.N_WEATHER + 8 + mdl.N_B1)
    for col in range(mdl.N_WEATHER):
        if col != worlds.TEMPERATURE_COLUMN:
            x[:, col] = torch.arange(worlds.N_REGIMES, dtype=torch.float32) - 15.5
    net.eval()
    with torch.no_grad():
        out = net.weather(x)
    gaps_between = (out[1:] - out[:-1]).abs().sum(dim=1)
    assert bool((gaps_between > 1e-6).all()), (
        "two neighbouring regimes give the same encoder output"
    )
    # tables, references, and a two-pass fit
    tabs = tables_for(
        arm_list=("Z", "R", "P90", "U", "U-A"),
        template=template,
        arrays=arr,
        days=np.concatenate([np.arange(300, 350), np.arange(450, 458)]),
    )
    small_held = tr.SampleSet(
        primary=held.primary[:8], c_idx=held.c_idx[:8], is_val=np.zeros(8, bool)
    )
    pre = gaps_of(arrays=arr, tables=tabs, held=small_held)
    assert all(np.isfinite(pre[k]) for k in ("gap_R", "gap_RM", "gap_U")), pre
    print(
        "synthetic gaps (shares of capacity):",
        {k: round(float(pre[k]), 4) for k in ("gap_R", "gap_RM", "gap_U")},
    )
    prim, dd = np.meshgrid(np.array(worlds.PRIMARIES), np.arange(300, 350), indexing="ij")
    train = tr.SampleSet(
        primary=prim.ravel(), c_idx=dd.ravel() * 48 + 24, is_val=(np.arange(prim.size) % 10 == 0)
    )
    tr.BATCH = 32  # the CPU's memory cap; the real gate keeps experiment 713's batch
    for arm in ("Z", "R", "N"):
        r = fit_world_arm(
            samples=train,
            arm=arm,
            world=world,
            mean=mean,
            std=std,
            held=small_held,
            table=tabs.get(arm),
            seed=0,
            passes=2,
            eval_passes=(1, 2),
        )
        assert len(r["trajectory"]) == 2 and np.isfinite(r["metric"]), r
    print(
        "known-answer self-test: the checks' logic, the real-budget rule, the era layout, the regimes in the weather path, and a two-pass fit of Z, R, and N pass"
    )
    print("e00718_known_answer: all checks passed")
    return 0


def precheck(*, month: str, last_era_days: int) -> int:
    """Run the pre-check of world WS on the real grid on the CPU (no fit): print the three gaps and the gap of `U` with its rows older than 91 days hidden."""
    import e00422_prep as prep422  # ty: ignore[unresolved-import]  # noqa: F401
    import e00713_inputs as inp  # ty: ignore[unresolved-import]
    import e00718_check_real as cr

    cr.cpu_only_cuda_calls()
    pipe = inp.load_pipeline(month=month, phase="train", chain_root=inp.chain_root_of_seed(seed=0))
    data = inp.build_fold_data(pipe=pipe, device="cpu")
    template = data.source
    arrays = worlds.build_ws(template=template, cutoff=data.cutoff_idx, last_era_days=last_era_days)
    held = worlds.held_out_samples(arrays=arrays)
    days = np.unique(held.c_idx // lay713.STEPS_PER_DAY)
    arm_list = ("Z", "R", "P90", "U", "U-A")
    tabs = tables_for(arm_list=arm_list, template=template, arrays=arrays, days=days)
    gaps = gaps_of(arrays=arrays, tables=tabs, held=held)
    pi = {p: i for i, p in enumerate(tabs["U"].primaries.tolist())}
    di = {d: i for i, d in enumerate(tabs["U"].days.tolist())}
    rows_u = np.stack(
        [
            tabs["U"].ages[pi[int(p)], di[int(c) // lay713.STEPS_PER_DAY]]
            for p, c in zip(held.primary, held.c_idx, strict=True)
        ]
    )
    young = np.where(rows_u > lay.SHORT_POOL_MAX_AGE, -1, rows_u)
    mae_young = worlds.reference_gap(arrays=arrays, rows_a=young, rows_b=young, samples=held)[0]
    out = {
        "month": month, "last_era_days": last_era_days, "held_out_samples": len(held.primary), "gaps": {k: float(gaps[k]) for k in ("gap_R", "gap_RM", "gap_U")},
        "reference_mae": gaps["reference_mae"], "mae_U_rows_up_to_91_days_only": mae_young,
        "gap_U_with_rows_up_to_91_days_only": gaps["reference_mae"]["U-A"] - mae_young,
        "minimum_gap": worlds.GAP_MINIMUM,
    }  # fmt: skip
    print(json.dumps(out, indent=1, default=float))
    return 0


def run(*, month: str, device: str, manager_note: Path | None = None) -> int:
    """Run the pre-check and the gating fits for the fold `month` and write the gate file (needs the stored data and the GPU)."""
    import e00422_prep as prep422  # ty: ignore[unresolved-import]
    import e00713_inputs as inp  # ty: ignore[unresolved-import]

    for stale in (
        arms.gate_file(data_dir=ML_EXPERIMENTS_DATA_DIR),
        arms.result_file(data_dir=ML_EXPERIMENTS_DATA_DIR),
    ):
        stale.unlink(
            missing_ok=True
        )  # a rerun that fails must not leave an earlier gate file valid
    arms.install_own_cache(data_dir=ML_EXPERIMENTS_DATA_DIR)
    pipe = inp.load_pipeline(month=month, phase="train", chain_root=inp.chain_root_of_seed(seed=0))
    data = inp.build_fold_data(pipe=pipe, device=device)
    template = data.source
    raw = pipe.d632.dense_raw()
    temperature = np.asarray(
        raw[:, :, pipe.d632.prep.WEATHER.index("temperature_2m")], dtype=np.float64
    )
    if np.nanmedian(temperature) > 100:
        temperature = temperature - 273.15
    n_grid = temperature.shape[1]
    half_hour = lay713.local_half_hours_for_grid(grid_start_us=prep422.GRID_START_US, n_grid=n_grid)
    year = 2019 + np.arange(n_grid) / (365.25 * 48)
    trending = worlds.trending_columns(dense_datetime=template.dense_datetime, year=year)
    print(
        "date-time columns that increase with the year (set to zero besides `covid`):",
        trending or "none",
    )
    started = time.monotonic()
    result: dict = {
        "month": month,
        "trending_columns": trending,
        "datetime_names_match": worlds.datetime_names_match_prep(),
    }
    scores: dict[str, dict[str, float]] = {}
    fits_all: dict[str, dict[str, list[dict]]] = {}
    gaps: dict = {}
    floor = float("nan")
    for kind in ("WS", "WN"):
        if kind == "WS":
            arrays = worlds.build_ws(template=template, cutoff=data.cutoff_idx)
        else:
            arrays = worlds.build_wn(
                template=template,
                temperature=temperature,
                half_hour=half_hour,
                cutoff=data.cutoff_idx,
            )
        arrays.datetime = worlds.neutralised_datetime(
            dense_datetime=template.dense_datetime, extra=trending
        )
        world, mean, std = worlds.to_world(arrays=arrays, template=template, device=device)
        held = worlds.held_out_samples(arrays=arrays)
        days = np.arange(
            (lay713.STEPS_PER_DAY * (lay713.MAX_AGE + 1) + 24) // lay713.STEPS_PER_DAY,
            arrays.cutoff // lay713.STEPS_PER_DAY + worlds.HELD_OUT_DAYS + 1,
        )
        arm_list = WS_ARMS if kind == "WS" else WN_ARMS
        tabs = tables_for(
            arm_list=("Z", "R", "P90", "U", "U-A") if kind == "WS" else ("Z", "R", "U"),
            template=template,
            arrays=arrays,
            days=days,
        )
        if kind == "WS":
            gaps = gaps_of(arrays=arrays, tables=tabs, held=held)
            print("pre-check gaps:", gaps)
            result["gaps"] = gaps
            if min(gaps["gap_R"], gaps["gap_RM"], gaps["gap_U"]) < worlds.GAP_MINIMUM:
                result["stopped"] = (
                    "pre-check: a gap is below 0.010; the world is mis-specified, no world fit runs"
                )
                print(result["stopped"])
                return finish(data_dir=ML_EXPERIMENTS_DATA_DIR, result=result)
        else:
            floor = worlds.held_out_floor(arrays=arrays)
        fits: dict[str, list[dict]] = {}
        for arm in arm_list:
            fits[arm] = [
                fit_world_arm(
                    arm=arm,
                    world=world,
                    mean=mean,
                    std=std,
                    held=held,
                    table=tabs.get(arm),
                    seed=s,
                    device=device,
                )
                for s in SEEDS
            ]
            print(kind, arm, [round(f["metric"], 4) for f in fits[arm]], flush=True)
        fits_all[kind] = fits
        scores[kind] = {a: arm_score(fits=fits, arm=a) for a in arm_list}
    result["scores"] = scores
    result["checks"] = evaluate_checks(ws=scores["WS"], wn=scores["WN"], gaps=gaps, floor=floor)
    result["fits"] = (
        fits_all  # every per-pass trajectory, for the real-budget reading and the traces
    )
    result["secondary_real_budget_pass"] = {
        k: {a: [f["real_budget_pass"] for f in v] for a, v in fits.items()}
        for k, fits in fits_all.items()
    }
    result["seconds"] = round(time.monotonic() - started, 1)
    print(json.dumps(result["checks"], indent=1, default=float))
    return finish(
        data_dir=ML_EXPERIMENTS_DATA_DIR,
        result=result,
        manager_note=manager_note,
    )


def main() -> int:
    """Run the gate or the self-test."""
    parser = argparse.ArgumentParser()
    parser.add_argument("month", nargs="?")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument(
        "--precheck",
        action="store_true",
        help="CPU only: the pre-check gaps of world WS on the real grid, no fit",
    )
    parser.add_argument("--last-era-days", type=int, default=worlds.LAST_ERA_DAYS)
    parser.add_argument(
        "--manager-accepts-unread-ks3",
        type=Path,
        default=None,
        metavar="NOTE",
        help="path of the manager's written decision; write the gate file although KS3's difference tests were unread; the note's hash goes into every record",
    )
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    assert args.month, "give the fold month"
    if args.precheck:
        return precheck(month=args.month, last_era_days=args.last_era_days)
    return run(
        month=args.month,
        device=args.device,
        manager_note=args.manager_accepts_unread_ks3,
    )


if __name__ == "__main__":
    sys.exit(main())
