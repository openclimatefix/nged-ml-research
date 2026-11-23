# ruff: noqa: E501, PT018
"""Experiment 716: forecast one fold with a fitted PV arm (prediction phase; WP4).

Usage: e00716_predict.py ARM MONTH SEED [--device cuda] [--readout NAME] [--accept-code-change OLD_HASH ...]
       e00716_predict.py --kf7

Run it through `run.sh`, in a separate process from the fit: it sets `E00667_PHASE=predict`. It loads
the stage 3 state, asserts that the fold's standardiser equals the training phase's, forecasts every
(Primary, scored origin, lead day), drops valid times on or after 2026-07-01, and writes
`work/fits/<arm>[-readout]/<month>_s<seed>.parquet` atomically with experiment 713's columns.

Read-outs with no refit (`--readout`): `cams` (the CAMS-rule mask and the monthly mean temperature),
`convention` (ages 2 to 5 masked), `nopv` (every PV flag false), `scale0.7`, `scale1.3`, and
`lookahead` (the six read the 2026-03 capacity in folds 2025-11 to 2026-02).

`--kf7` runs the check KF7: this module's `forecast_frame` on the stored stage 3 state of `M*` (fold
2025-11, seed 0, as arm `Z`) against the stored forecast file of `M*`, with identical keys and a
difference of at most 1e-6 of capacity. It reads real data on the CPU and writes nothing.
"""

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

os.environ.setdefault("E00667_PHASE", "predict")

import e00713_arms as arms713  # ty: ignore[unresolved-import]
import e00713_layout as lay  # ty: ignore[unresolved-import]
import e00713_model as mdl  # ty: ignore[unresolved-import]
import e00713_train as tr  # ty: ignore[unresolved-import]
import e00716_arms as arms
import e00716_batch as pvb
import e00716_model as pvm
import e00716_train as pvt
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.experiment_store import HOLDOUT_START
from studies.sources import ML_EXPERIMENTS_DATA_DIR

torch.set_num_threads(1)
QCOLS = ("q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99")
PREDICT_BATCH = 512
READOUTS = ("cams", "convention", "nopv", "scale0.7", "scale1.3", "lookahead")


def forecast_frame(
    *, model: pvm.PvNet, test_origin_idx: np.ndarray, source: pvb.PvSource, target_mean: np.ndarray,
    target_std: np.ndarray, seed: int, grid_start_us: int, time_series_ids: np.ndarray, readout: str | None = None,
) -> pl.DataFrame:  # fmt: skip
    """Return the forecast frame of all scored origins, 16 Primaries, and 14 lead days (713's columns)."""
    c_all = test_origin_idx + lay.NATIVE_OFFSET_STEPS
    prim, cc, lead = np.meshgrid(np.arange(16), c_all, np.arange(14), indexing="ij")
    prim, cc, lead = prim.ravel(), cc.ravel(), lead.ravel()
    samples = tr.SampleSet(primary=prim, c_idx=cc, is_val=np.zeros(len(prim), bool))
    model.eval()
    parts = []
    scale = {"scale0.7": 0.7, "scale1.3": 1.3}.get(readout or "", 1.0)
    with torch.no_grad():
        for start in range(0, len(prim), PREDICT_BATCH):
            sel = np.arange(start, min(start + PREDICT_BATCH, len(prim)))
            ones = np.ones(len(sel), bool)
            masks = (
                (ones, np.zeros(len(sel), bool))
                if readout == "cams"
                else (np.zeros(len(sel), bool), ones)
                if readout == "convention"
                else None
            )
            b = pvt.batch_for(
                source=source, samples=samples, index=sel, lead=lead[sel], cut=10**9, pv_masks=masks
            )
            inp = b.inputs
            if readout == "nopv":
                inp = inp.with_pv(
                    hist_pv_ok=torch.zeros_like(inp.hist_pv_ok),
                    top_pv_ok=torch.zeros_like(inp.top_pv_ok),
                    hist_pv=torch.zeros_like(inp.hist_pv),
                    top_pv=torch.zeros_like(inp.top_pv),
                )
            if scale != 1.0:
                inp = inp.with_pv(hist_pv=inp.hist_pv * scale, top_pv=inp.top_pv * scale)
            ens = source.ens
            assert ens is not None
            assert bool(
                ens.filled[
                    inp.primary[:, None],
                    (b.c_idx // 48 - ens.first_day)[:, None],
                    (48 * inp.lead_day[:, None] + source.steps),
                ].all()
            ), "a scored row has no ensemble inputs"
            out = model(inp).out
            dev = out.device
            tstd, tmean = (
                torch.from_numpy(target_std).to(dev)[inp.primary],
                torch.from_numpy(target_mean).to(dev)[inp.primary],
            )
            m = (
                (mdl.standardised_mixture_mean(out=out) * tstd[:, None] + tmean[:, None])
                .cpu()
                .numpy()
            )
            q = mdl.standardised_quantiles(out=out) * tstd[:, None, None] + tmean[:, None, None]
            q = torch.sort(q, dim=-1).values.cpu().numpy()
            actual = source.power_z[inp.primary[:, None], b.target_idx].cpu().numpy()
            parts.append((sel, q, b.target_idx.cpu().numpy(), actual, m))
    rows = []
    for sel, q, tidx, actual, m in parts:
        pr = np.repeat(prim[sel], 48)
        c = np.repeat(cc[sel], 48)
        rows.append(
            pl.DataFrame({
                "time_series_id": time_series_ids[pr].astype(np.int32),
                "origin_us": grid_start_us + (c - lay.NATIVE_OFFSET_STEPS) * lay.HALF_HOUR_US,
                "valid_us": grid_start_us + tidx.reshape(-1) * lay.HALF_HOUR_US,
                **{name: q[:, :, i].reshape(-1).astype(np.float32) for i, name in enumerate(QCOLS)},
                "mean": m.reshape(-1).astype(np.float32),
                "actual": (actual.reshape(-1) * target_std[pr] + target_mean[pr]).astype(np.float32),
            })
        )  # fmt: skip
    utc = pl.Datetime("us", "UTC")
    return (
        pl.concat(rows)
        .with_columns(
            power_fcst_init_time=pl.col("origin_us").cast(pl.Datetime("us")).dt.replace_time_zone("UTC"),
            valid_time=pl.col("valid_us").cast(pl.Datetime("us")).dt.replace_time_zone("UTC"),
        )
        .with_columns(
            nwp_init_time=pl.col("power_fcst_init_time").dt.truncate("1d"),
            lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes() / 60.0,
            seed=pl.lit(seed, dtype=pl.Int64),
        )
        .filter(pl.col("valid_time") < pl.lit(HOLDOUT_START).cast(utc))
        .drop("origin_us", "valid_us")
        .select("time_series_id", "power_fcst_init_time", "valid_time", *QCOLS, "seed", "mean", "actual", "lead_hours", "nwp_init_time")
        .sort("time_series_id", "power_fcst_init_time", "valid_time")
    )  # fmt: skip


def compare_forecasts(*, new: pl.DataFrame, stored: pl.DataFrame, tolerance: float = 1e-6) -> str:
    """Check KF7: identical keys, and every quantile and the mean within `tolerance` of capacity."""
    keys = ["time_series_id", "power_fcst_init_time", "valid_time"]
    assert new.height == stored.height, (new.height, stored.height)
    assert new.select(keys).equals(stored.select(keys)), "the forecast keys differ"
    worst = max(float((new[c] - stored[c]).abs().max()) for c in (*QCOLS, "mean"))  # ty: ignore[invalid-argument-type]
    assert worst <= tolerance, f"KF7: the largest difference is {worst}"
    return f"KF7: {new.height} rows, identical keys, largest difference {worst:.2e} of capacity"


def predict(
    *, arm: str, month: str, seed: int, device: str, data_dir: Path, readout: str | None = None,
    accept: Sequence[str] = (),
) -> Path:  # fmt: skip
    """Forecast one fold and write the per-seed file."""
    import e00422_prep as prep422  # ty: ignore[unresolved-import]
    import e00578_common as k578  # ty: ignore[unresolved-import]
    import e00713_inputs as inp  # ty: ignore[unresolved-import]
    import e00716_fit as fit

    name = f"{arm}-{readout}" if readout else arm
    out = (
        arms.fit_file(data_dir=data_dir, arm=arm, month=month, seed=seed).parent.parent
        / name
        / f"{month}_s{seed}.parquet"
    )
    if out.exists():
        print(f"{name} {month} s{seed}: done, skipped", flush=True)
        return out
    spec = arms.ARMS[arm]
    state = arms.model_file(data_dir=data_dir, arm=arm, month=month, seed=seed, stage=3)
    assert state.exists(), state
    stage_log = fit.stage_log_file(data_dir=data_dir, arm=arm, month=month, seed=seed, stage=3)
    assert stage_log.exists(), f"{state} has no stage log: refit it"
    log: dict = {
        "arm": name,
        "month": month,
        "seed": seed,
        "readout": readout,
        **arms.code_record(),
    }
    arms.check_code(
        earlier=json.loads(stage_log.read_text()), accept=accept, record=log, what=state.name
    )
    arms.install_own_cache()
    pipe = inp.load_pipeline(
        month=month, phase="predict", chain_root=inp.chain_root_of_seed(seed=seed)
    )
    data = inp.build_fold_data(pipe=pipe, device=device, keep_rows=True)
    saved = arms.root(data_dir=data_dir, arm=arm) / "work" / "standardiser" / f"{month}.sha256"
    assert saved.exists() and saved.read_text() == data.fingerprint, (
        "standardiser differs from the training phase"
    )
    tables = fit.build_pv_tables(pipe=pipe, data=data, arm=arm, device=device, readout=readout)
    source = pvb.PvSource(source=data.source, tables=tables)
    model = pvm.PvNet(arm=spec.net, seed=seed).to(device)  # ty: ignore[invalid-argument-type]
    model.load_state_dict(torch.load(state, map_location=device))
    assert data.test_origin_idx is not None
    ids = k578.sites().sort("primary")["time_series_id"].to_numpy()
    frame = forecast_frame(
        model=model, test_origin_idx=data.test_origin_idx, source=source, target_mean=data.target_mean,
        target_std=data.target_std, seed=seed, grid_start_us=prep422.GRID_START_US, time_series_ids=ids, readout=readout,
    )  # fmt: skip
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + f".{os.getpid()}.tmp")
    frame.write_parquet(tmp)
    tmp.rename(out)
    log_path = arms.log_file(
        data_dir=data_dir,
        arm=arm,
        month=month,
        seed=seed,
        kind="predict" if readout is None else f"predict_{readout}",
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps({**log, "rows": frame.height}, indent=1, default=str))
    print(f"{name} {month} s{seed}: {frame.height} rows", flush=True)
    return out


def kf7(*, data_dir: Path, device: str = "cpu") -> str:
    """KF7: predict fold 2025-11, seed 0 from the stored stage 3 state of `M*` as arm `Z` and compare with `M*`'s stored forecast.

    The stored forecast was written on the GPU, and the CPU path differs from it by 8.4e-6 of capacity at
    the 99th percentile (experiment 713's own CPU path gives the same difference), so KF7 runs with `--device cuda`.
    """
    import e00422_prep as prep422  # ty: ignore[unresolved-import]
    import e00578_common as k578  # ty: ignore[unresolved-import]
    import e00713_inputs as inp  # ty: ignore[unresolved-import]
    import e00716_fit as fit

    month, seed = "2025-11", 0
    star = arms713.m_star(data_dir=data_dir)
    state = arms713.model_file(data_dir=data_dir, arm=star, month=month, seed=seed, stage=3)
    stored = pl.read_parquet(arms713.fit_file(data_dir=data_dir, arm=star, month=month, seed=seed))
    arms.install_own_cache()
    pipe = inp.load_pipeline(
        month=month, phase="predict", chain_root=inp.chain_root_of_seed(seed=seed)
    )
    data = inp.build_fold_data(pipe=pipe, device=device, keep_rows=True)
    tables = fit.build_pv_tables(pipe=pipe, data=data, arm="Z", device=device)
    source = pvb.PvSource(source=data.source, tables=tables)
    model = pvm.PvNet(arm="Z").to(device)
    model.load_state_dict(torch.load(state, map_location=device))
    ids = k578.sites().sort("primary")["time_series_id"].to_numpy()
    assert data.test_origin_idx is not None
    frame = forecast_frame(
        model=model, test_origin_idx=data.test_origin_idx, source=source, target_mean=data.target_mean,
        target_std=data.target_std, seed=seed, grid_start_us=prep422.GRID_START_US, time_series_ids=ids,
    )  # fmt: skip
    return compare_forecasts(new=frame, stored=stored)


def main() -> int:
    """Parse the command line and predict."""
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", nargs="?")
    parser.add_argument("month", nargs="?")
    parser.add_argument("seed", type=int, nargs="?")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--readout", choices=READOUTS, default=None)
    parser.add_argument("--kf7", action="store_true")
    parser.add_argument(
        "--record", action="store_true", help="with --kf7: write the wiring record of KF7"
    )
    parser.add_argument("--accept-code-change", action="append", default=[], metavar="OLD_HASH")
    args = parser.parse_args()
    if args.kf7:
        message = kf7(data_dir=ML_EXPERIMENTS_DATA_DIR, device=args.device)
        print(message)
        if args.record:
            arms.write_wiring_record(
                data_dir=ML_EXPERIMENTS_DATA_DIR,
                name="KF7",
                passed=True,
                fold="2025-11",
                detail={"message": str(message)},
            )
        return 0
    predict(
        arm=args.arm,
        month=args.month,
        seed=args.seed,
        device=args.device,
        data_dir=ML_EXPERIMENTS_DATA_DIR,
        readout=args.readout,
        accept=args.accept_code_change,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
