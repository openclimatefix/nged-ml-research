# ruff: noqa: E501, PT018
"""Experiment 718: forecast one fold with a fitted network (prediction phase; design sections 6, 9, and 12).

Usage: e00718_predict.py ARM MONTH SEED [--device cuda] [--mask-history] [--tables VERSION] [--smoke] [--accept-code-change OLD_HASH ...]

Run it through `run.sh`, in a separate process from the fit: it sets `E00667_PHASE=predict`, which
opens the fold month's scored window. It loads the stage 3 state, asserts that the fold's
standardiser equals the training phase's, forecasts every (Primary, scored origin, lead day) with the
rows of the arm's `ens` table, drops valid times on or after 2026-07-01, and writes
`work/fits/<arm>/<month>_s<seed>.parquet` atomically with the columns of experiment 713's forecasts.
`--mask-history` forecasts with all history power masked, under the name `<arm>-nohist`. `--tables
swap` reads the rebuilt live-fair tables (the selection swap of design section 9) under the name
`<arm>-swap`; the tokens still read ERA5.
"""

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

os.environ["E00667_PHASE"] = "predict"

import e00713_inputs as inp  # ty: ignore[unresolved-import]
import e00713_layout as lay713  # ty: ignore[unresolved-import]
import e00713_model as mdl  # ty: ignore[unresolved-import]
import e00713_predict as p713  # ty: ignore[unresolved-import]
import e00718_arms as arms
import e00718_batch as fb
import e00718_fit as fit_mod
import e00718_model as f9
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.experiment_store import HOLDOUT_START
from studies.sources import ML_EXPERIMENTS_DATA_DIR

torch.set_num_threads(1)
QCOLS = p713.QCOLS
PREDICT_BATCH = 512


def forecast_frame(
    *, model: f9.F9Net, data: inp.FoldData, source: fb.F9Source, seed: int, mask_history: bool, grid_start_us: int,
    time_series_ids: np.ndarray,
) -> pl.DataFrame:  # fmt: skip
    """Return the forecast frame of all scored origins, 16 Primaries, and 14 lead days.

    `data` supplies `test_origin_idx`, `target_mean`, and `target_std`. The rows of every scored
    sample are asserted equal to the row table's.
    """
    assert data.test_origin_idx is not None
    c_all = data.test_origin_idx + lay713.NATIVE_OFFSET_STEPS
    prim, cc, lead = np.meshgrid(np.arange(16), c_all, np.arange(14), indexing="ij")
    prim, cc, lead = prim.ravel(), cc.ravel(), lead.ravel()
    parts = []
    model.eval()
    ens = source.ens
    assert ens is not None
    with torch.no_grad():
        for start in range(0, len(prim), PREDICT_BATCH):
            sel = np.arange(start, min(start + PREDICT_BATCH, len(prim)))
            to = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(source.device)  # noqa: E731
            ones = to(np.ones(len(sel), bool))
            b = source.assemble(
                primary=to(prim[sel]), c_idx=to(cc[sel]), lead_day=to(lead[sel]), ensemble=True,
                mask_all_power=ones if mask_history else None, target_end=10**9,
            )  # fmt: skip
            assert bool(
                ens.filled[
                    b.inputs.primary[:, None],
                    (b.c_idx // 48 - ens.first_day)[:, None],
                    (48 * b.inputs.lead_day[:, None] + source.base.steps),
                ].all()
            ), "a scored row has no ensemble inputs"
            table_ages = source.tables[fb.TABLE_ENS].lookup(
                primary=prim[sel], day=cc[sel] // 48, lead_day=lead[sel]
            )
            got = np.where(b.inputs.row_exists.cpu().numpy(), b.ages.cpu().numpy(), -1)
            assert np.array_equal(got[table_ages >= 0], table_ages[table_ages >= 0]), (
                "the rows of a scored forecast differ from the row table"
            )
            out = model(b.inputs).out
            std = torch.from_numpy(data.target_std).to(out.device)[b.inputs.primary]
            mean = torch.from_numpy(data.target_mean).to(out.device)[b.inputs.primary]
            m = mdl.standardised_mixture_mean(out=out) * std[:, None] + mean[:, None]
            q = mdl.standardised_quantiles(out=out) * std[:, None, None] + mean[:, None, None]
            actual = source.power_z[b.inputs.primary[:, None], b.target_idx].cpu().numpy()
            parts.append(
                (
                    sel,
                    torch.sort(q, dim=-1).values.cpu().numpy(),
                    b.target_idx.cpu().numpy(),
                    actual,
                    m.cpu().numpy(),
                )
            )
    rows = []
    for sel, q, tidx, actual, m in parts:
        pr = np.repeat(prim[sel], 48)
        c = np.repeat(cc[sel], 48)
        rows.append(
            pl.DataFrame(
                {
                    "time_series_id": time_series_ids[pr].astype(np.int32),
                    "origin_us": grid_start_us + (c - lay713.NATIVE_OFFSET_STEPS) * lay713.HALF_HOUR_US,
                    "valid_us": grid_start_us + tidx.reshape(-1) * lay713.HALF_HOUR_US,
                    **{name: q[:, :, i].reshape(-1).astype(np.float32) for i, name in enumerate(QCOLS)},
                    "mean": m.reshape(-1).astype(np.float32),
                    "actual": (actual.reshape(-1) * data.target_std[pr] + data.target_mean[pr]).astype(np.float32),
                }
            )
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


def predict(
    *, arm: str, month: str, seed: int, device: str, mask_history: bool, data_dir: Path,
    tables_version: str = "ens", smoke: bool = False, accept: Sequence[str] = (),
) -> Path:  # fmt: skip
    """Forecast one fold and write the per-seed file."""
    import e00422_prep as prep422  # ty: ignore[unresolved-import]
    import e00578_common as k578  # ty: ignore[unresolved-import]

    name = arm + ("-nohist" if mask_history else "") + ("-swap" if tables_version == "swap" else "")
    out = arms.fit_file(data_dir=data_dir, arm=name, month=month, seed=seed, smoke=smoke)
    if out.exists():
        print(f"{name} {month} s{seed}: done, skipped", flush=True)
        return out
    state = arms.model_file(
        data_dir=data_dir, arm=arm, month=month, seed=seed, stage=3, smoke=smoke
    )
    assert state.exists(), state
    stage_log = arms.log_file(
        data_dir=data_dir, arm=arm, month=month, seed=seed, kind="fit", smoke=smoke
    )
    stage_log = stage_log.with_name(f"{stage_log.stem}_stage3.json")
    assert stage_log.exists(), f"{state} has no stage log: refit it"
    predict_log: dict = {
        "arm": name,
        "month": month,
        "seed": seed,
        "smoke": smoke,
        **arms.code_record(),
        "recipe": arms.recipe_hashes(),
    }
    arms.check_code(
        earlier=json.loads(stage_log.read_text()),
        accept=accept,
        record=predict_log,
        what=state.name,
    )
    pipe = inp.load_pipeline(
        month=month, phase="predict", chain_root=inp.chain_root_of_seed(seed=seed)
    )
    data = inp.build_fold_data(pipe=pipe, device=device)
    saved = arms.root(data_dir=data_dir, part=1) / "work" / "standardiser" / f"{month}.sha256"
    assert saved.exists() and saved.read_text() == data.fingerprint, (
        "the standardiser differs from the training phase"
    )
    expect = fit_mod.table_meta(
        arm=arm, month=month, fingerprint=data.fingerprint, phase="predict",
        digest=fit_mod.inputs_digest(source=data.source, cutoff_idx=data.cutoff_idx),
    )  # fmt: skip
    tables = {
        fb.TABLE_ERA5: fit_mod.load_table(
            path=arms.table_file(data_dir=data_dir, arm=arm, month=month, version="era5"),
            expect=expect,
        ),
        fb.TABLE_ENS: fit_mod.load_table(
            path=arms.table_file(data_dir=data_dir, arm=arm, month=month, version=tables_version),
            expect=expect,
        ),
    }
    source = fb.F9Source(base=data.source, tables=tables)
    model = f9.F9Net(seed=seed).to(device)
    model.load_state_dict(torch.load(state, map_location=device))
    ids = k578.sites().sort("primary")["time_series_id"].to_numpy()
    frame = forecast_frame(
        model=model,
        data=data,
        source=source,
        seed=seed,
        mask_history=mask_history,
        grid_start_us=prep422.GRID_START_US,
        time_series_ids=ids,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + f".{os.getpid()}.tmp")
    frame.write_parquet(tmp)
    tmp.rename(out)
    fit_mod.write_json_atomic(
        obj={
            **predict_log,
            "rows": frame.height,
            "table_digests": {v: t.digest() for v, t in tables.items()},
        },
        path=arms.log_file(
            data_dir=data_dir, arm=name, month=month, seed=seed, kind="predict", smoke=smoke
        ),
    )
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
    parser.add_argument("--tables", default="ens", choices=["ens", "swap"])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--accept-code-change", action="append", default=[], metavar="OLD_HASH")
    args = parser.parse_args()
    arms.install_own_cache(data_dir=ML_EXPERIMENTS_DATA_DIR)
    predict(
        arm=args.arm,
        month=args.month,
        seed=args.seed,
        device=args.device,
        mask_history=args.mask_history,
        data_dir=ML_EXPERIMENTS_DATA_DIR,
        tables_version=args.tables,
        smoke=args.smoke,
        accept=args.accept_code_change,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
