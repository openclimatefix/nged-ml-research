# ruff: noqa: E501, PT018
"""Experiment 713: forecast one fold with a fitted MSA network (prediction phase).

Usage: e00713_predict.py ARM MONTH SEED [--device cuda] [--mask-history]

Run it through `run.sh`, in a separate process from the fit: it sets `E00667_PHASE=predict`, which
opens the fold month's scored window. It loads the stage 3 state, asserts that the fold's standardiser
equals the training phase's, forecasts every (Primary, scored origin, lead day), drops valid times on
or after 2026-07-01, and writes `work/fits/<arm>/<month>_s<seed>.parquet` atomically. The columns are
the Candidate's (`time_series_id`, `power_fcst_init_time`, `valid_time`, `q01` to `q99`, in shares of
capacity) plus `seed`, `actual`, `lead_hours`, and `nwp_init_time`. `--mask-history` forecasts with
every history power token masked (the free read-out of section 2.7) under the arm name `<arm>-nohist`.
"""

import argparse
import os
import sys
from pathlib import Path

os.environ["E00667_PHASE"] = "predict"

import e00713_arms as arms
import e00713_inputs as inp
import e00713_layout as lay
import e00713_model as mdl
import e00713_train as tr
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.experiment_store import HOLDOUT_START
from studies.sources import ML_EXPERIMENTS_DATA_DIR

torch.set_num_threads(1)
QCOLS = ("q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99")
PREDICT_BATCH = 512


def forecast_frame(
    *, model: mdl.MsaNet, data: inp.FoldData, spec: arms.ArmSpec, world: tr.World, seed: int,
    mask_history: bool, grid_start_us: int, time_series_ids: np.ndarray,
) -> pl.DataFrame:  # fmt: skip
    """Return the forecast frame of all scored origins, 16 Primaries, and 14 lead days."""
    assert data.test_origin_idx is not None
    c_all = data.test_origin_idx + lay.NATIVE_OFFSET_STEPS
    prim, cc, lead = np.meshgrid(np.arange(16), c_all, np.arange(14), indexing="ij")
    prim, cc, lead = prim.ravel(), cc.ravel(), lead.ravel()
    samples = tr.SampleSet(primary=prim, c_idx=cc, is_val=np.zeros(len(prim), bool))
    if spec.placebo:
        assert world.placebo_pool is not None
        samples.ages = tr.placebo_ages_for(primary=prim, c_idx=cc, pool=world.placebo_pool)
    parts = []
    model.eval()
    src = data.source
    with torch.no_grad():
        for start in range(0, len(prim), PREDICT_BATCH):
            sel = np.arange(start, min(start + PREDICT_BATCH, len(prim)))
            ones = np.ones(len(sel), bool)
            drop = (ones, np.zeros(len(sel), bool)) if mask_history else None
            b = tr.batch_for(
                source=src, samples=samples, index=sel, lead=lead[sel], ensemble=True, cut=10**9, dropout=drop,
            )  # fmt: skip
            assert src.ens is not None
            assert bool(
                src.ens.filled[
                    b.inputs.primary[:, None],
                    (b.c_idx // 48 - src.ens.first_day)[:, None],
                    (48 * b.inputs.lead_day[:, None] + src.steps),
                ].all()
            ), "a scored row has no ensemble inputs"  # ty: ignore[unresolved-attribute]
            q = mdl.standardised_quantiles(out=model(b.inputs).out)
            q = q * torch.from_numpy(data.target_std).to(q.device)[b.inputs.primary][:, None, None]
            q = q + torch.from_numpy(data.target_mean).to(q.device)[b.inputs.primary][:, None, None]
            q = torch.sort(q, dim=-1).values.cpu().numpy()
            actual = data.source.power_z[b.inputs.primary[:, None], b.target_idx].cpu().numpy()
            tidx = b.target_idx.cpu().numpy()
            parts.append((sel, q, tidx, actual))
    rows = []
    for sel, q, tidx, actual in parts:
        n = len(sel)
        pr = np.repeat(prim[sel], 48)
        c = np.repeat(cc[sel], 48)
        flat_t = tidx.reshape(-1)
        mean = data.target_mean[pr]
        std = data.target_std[pr]
        rows.append(
            pl.DataFrame(
                {
                    "time_series_id": time_series_ids[pr].astype(np.int32),
                    "origin_us": grid_start_us + (c - lay.NATIVE_OFFSET_STEPS) * lay.HALF_HOUR_US,
                    "valid_us": grid_start_us + flat_t * lay.HALF_HOUR_US,
                    **{
                        name: q[:, :, i].reshape(-1).astype(np.float32)
                        for i, name in enumerate(QCOLS)
                    },
                    "actual": (actual.reshape(-1) * std + mean).astype(np.float32),
                }
            )
        )
        del n
    frame = pl.concat(rows)
    utc = pl.Datetime("us", "UTC")
    return (
        frame.with_columns(
            power_fcst_init_time=pl.col("origin_us")
            .cast(pl.Datetime("us"))
            .dt.replace_time_zone("UTC"),
            valid_time=pl.col("valid_us").cast(pl.Datetime("us")).dt.replace_time_zone("UTC"),
        )
        .with_columns(
            nwp_init_time=pl.col("power_fcst_init_time").dt.truncate("1d"),
            lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes()
            / 60.0,
            seed=pl.lit(seed, dtype=pl.Int64),
        )
        .filter(pl.col("valid_time") < pl.lit(HOLDOUT_START).cast(utc))
        .drop("origin_us", "valid_us")
        .select(
            "time_series_id",
            "power_fcst_init_time",
            "valid_time",
            *QCOLS,
            "seed",
            "actual",
            "lead_hours",
            "nwp_init_time",
        )
        .sort("time_series_id", "power_fcst_init_time", "valid_time")
    )


def predict(
    *, arm: str, month: str, seed: int, device: str, mask_history: bool, data_dir: Path
) -> Path:
    """Forecast one fold and write the per-seed file."""
    import e00422_prep as prep422  # ty: ignore[unresolved-import]
    import e00578_common as k578  # ty: ignore[unresolved-import]

    name = f"{arm}-nohist" if mask_history else arm
    out = arms.fit_file(data_dir=data_dir, arm=name, month=month, seed=seed)
    if out.exists():
        print(f"{name} {month} s{seed}: done, skipped", flush=True)
        return out
    spec = arms.resolve(arm=arm, data_dir=data_dir)
    state = arms.model_file(data_dir=data_dir, arm=arm, month=month, seed=seed, stage=3)
    assert state.exists(), state
    pipe = inp.load_pipeline(
        month=month, phase="predict", chain_root=inp.chain_root_of_seed(seed=seed)
    )
    data = inp.build_fold_data(pipe=pipe, device=device)
    saved = arms.root(data_dir=data_dir) / "work" / "standardiser" / f"{month}.sha256"
    assert saved.exists() and saved.read_text() == data.fingerprint, (
        "standardiser differs from the training phase"
    )
    world = tr.World(
        source=data.source, cutoff_idx=data.cutoff_idx, clean=data.clean, target_mean=data.target_mean,
        target_std=data.target_std, month=month, placebo_pool=tr.block_pool(clean=data.clean),
    )  # fmt: skip
    model = mdl.MsaNet(switches=spec.switches).to(device)
    model.load_state_dict(torch.load(state, map_location=device))
    sites = k578.sites().sort("primary")
    ids = sites["time_series_id"].to_numpy()
    frame = forecast_frame(
        model=model, data=data, spec=spec, world=world, seed=seed, mask_history=mask_history,
        grid_start_us=prep422.GRID_START_US, time_series_ids=ids,
    )  # fmt: skip
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    frame.write_parquet(tmp)
    tmp.rename(out)
    print(f"{name} {month} s{seed}: {frame.height} rows", flush=True)
    return out


def main() -> int:
    """Parse the command line and predict."""
    parser = argparse.ArgumentParser()
    parser.add_argument("arm")
    parser.add_argument("month")
    parser.add_argument("seed", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mask-history", action="store_true")
    args = parser.parse_args()
    predict(
        arm=args.arm, month=args.month, seed=args.seed, device=args.device,
        mask_history=args.mask_history, data_dir=ML_EXPERIMENTS_DATA_DIR,
    )  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
