# ruff: noqa: D205, E501, PT018
"""Experiment 725: forecast one fold with a stored or fitted state on the live rule (prediction phase).

Usage: e00725_predict.py ARM MONTH SEED [--device cuda] [--accept-code-change OLD_HASH ...]

ARM is a fitted arm of F8 (`L`, `Z`, `E`; its stage 3 state predicts on its own rule), or one of the
prediction-only readings of `e00725_common.PREDICT_SPECS`:

- `S`: the stored stage 3 state of `M*` (experiment 713's `M1`, read only) predicts on the live rule,
  with no retraining.
- `T3-M` and `T3-L`: `M*` on the analysis weather and `L` on the live rule, with the target row's
  CAMS day summary moved from UTC day D - 1 to D - 3 (report-only read-out `T3`; no arm is trained
  on the moved summary).

Run it through `run.sh`, in a separate process from the fit: it sets `E00667_PHASE=predict`, which
opens the fold month's scored window. It asserts that the fold's standardiser equals the training
phase's, forecasts every (Primary, scored origin, lead day) through experiment 713's
`forecast_frame`, drops valid times on or after 2026-07-01, and writes
`work/fits/<arm>/<month>_s<seed>.parquet` atomically with experiment 713's columns. It refuses a
stage 3 state whose log carries another code hash unless `--accept-code-change` names the old hash.
"""

import argparse
import json
import os
import resource
import sys
import time
from collections.abc import Sequence
from pathlib import Path

os.environ["E00667_PHASE"] = "predict"

import e00713_arms as arms713  # ty: ignore[unresolved-import]
import e00713_model as mdl  # ty: ignore[unresolved-import]
import e00713_predict as pred713  # ty: ignore[unresolved-import]
import e00725_common as cm
import e00725_frozen as fz
import e00725_world as world_mod
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

torch.set_num_threads(1)


def resolve_state(
    *, arm: str, month: str, seed: int, data_dir: Path
) -> tuple[Path, Path, str, int | None, bool]:
    """Return (state file, its stage log, the rule, the target-summary shift, is-713-state) of a
    prediction arm.
    """
    if arm in cm.PREDICT_SPECS:
        spec = cm.PREDICT_SPECS[arm]
        if spec.state == "m_star":
            state = arms713.model_file(
                data_dir=data_dir, arm=cm.M_STAR, month=month, seed=seed, stage=3
            )
            log = fz.stage_log_713(data_dir=data_dir, month=month, seed=seed, stage=3)
            return state, log, spec.rule, spec.target_summary_back_days, True
        own = spec.state.removeprefix("own:")
        state = cm.model_file(data_dir=data_dir, arm=own, month=month, seed=seed, stage=3)
        log = cm.stage_log_file(data_dir=data_dir, arm=own, month=month, seed=seed, stage=3)
        return state, log, spec.rule, spec.target_summary_back_days, False
    arm_spec = cm.resolve(arm=arm)
    state = cm.model_file(data_dir=data_dir, arm=arm, month=month, seed=seed, stage=3)
    log = cm.stage_log_file(data_dir=data_dir, arm=arm, month=month, seed=seed, stage=3)
    return state, log, arm_spec.rule, None, False


def predict(
    *, arm: str, month: str, seed: int, device: str, data_dir: Path, accept: Sequence[str] = ()
) -> Path:  # fmt: skip
    """Forecast one fold and write the per-seed file."""
    import e00422_prep as prep422  # ty: ignore[unresolved-import]
    import e00578_common as k578  # ty: ignore[unresolved-import]

    fz.assert_frozen_checkout()
    fz.limit_gpu_memory(device=device)
    out = cm.fit_file(data_dir=data_dir, arm=arm, month=month, seed=seed)
    if out.exists():
        print(f"{arm} {month} s{seed}: done, skipped", flush=True)
        return out
    gate = cm.wiring_file(data_dir=data_dir)
    assert gate.exists(), f"{gate} is missing: run e00725_checks.py first"
    cm.assert_gate_checks_hash(gate=json.loads(gate.read_text()))
    predict_log: dict = {"arm": arm, "month": month, "seed": seed, **fz.code_record()}
    fz.check_code(
        earlier=json.loads(gate.read_text()).get("code", {}), accept=[], record=predict_log,
        what="the wiring-check gate file",
    )  # fmt: skip
    state, stage_log, rule_name, back_days, is_713 = resolve_state(
        arm=arm, month=month, seed=seed, data_dir=data_dir
    )  # fmt: skip
    assert state.exists() and stage_log.exists(), f"{state} or its log is missing"
    log_in = json.loads(stage_log.read_text())
    if is_713:
        fz.check_warm_start(stage1_log=log_in, what=f"stage 3 state of {cm.M_STAR} {month} s{seed}")
    else:
        fz.check_code(earlier=log_in, accept=accept, record=predict_log, what=state.name)
    started = time.monotonic()
    lw = world_mod.load_world(
        month=month, seed=seed, rule_name=rule_name, phase="predict", device=device,
        data_dir=data_dir, target_summary_back_days=back_days,
    )  # fmt: skip
    spec713 = arms713.resolve(arm=cm.M_STAR, data_dir=data_dir)
    model = mdl.MsaNet(switches=spec713.switches).to(device)
    model.load_state_dict(torch.load(state, map_location=device))
    sites = k578.sites().sort("primary")
    frame = pred713.forecast_frame(
        model=model, data=lw.data, spec=spec713, world=lw.world, seed=seed, mask_history=False,
        grid_start_us=prep422.GRID_START_US, time_series_ids=sites["time_series_id"].to_numpy(),
    )  # fmt: skip
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + f".{os.getpid()}.tmp")
    frame.write_parquet(tmp)
    tmp.rename(out)
    predict_log.update(
        rows=frame.height, rule=rule_name, target_summary_back_days=back_days,
        seconds=round(time.monotonic() - started, 1), counters=dict(lw.live.counters),
        peak_host_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        peak_gpu_mb=torch.cuda.max_memory_allocated() / 2**20 if device.startswith("cuda") else None,
        nvidia_smi_process_mb=fz.nvidia_smi_process_mb() if device.startswith("cuda") else None,
    )  # fmt: skip
    cm.write_json_atomic(
        obj=predict_log,
        path=cm.log_file(data_dir=data_dir, arm=arm, month=month, seed=seed, kind="predict"),
    )  # fmt: skip
    print(f"{arm} {month} s{seed}: {frame.height} rows", flush=True)
    return out


def main() -> int:
    """Parse the command line and predict."""
    parser = argparse.ArgumentParser()
    parser.add_argument("arm")
    parser.add_argument("month")
    parser.add_argument("seed", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--accept-code-change", action="append", default=[], metavar="OLD_HASH")
    args = parser.parse_args()
    predict(
        arm=args.arm, month=args.month, seed=args.seed, device=args.device,
        data_dir=ML_EXPERIMENTS_DATA_DIR, accept=args.accept_code_change,
    )  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
