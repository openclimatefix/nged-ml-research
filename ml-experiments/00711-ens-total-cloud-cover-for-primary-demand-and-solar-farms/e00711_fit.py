r"""Fit one (arm, fold, seed) job of experiment 711 and write its predictions.

**One invocation fits one job.** The job has a fixed seed, its own output files, and a completion
marker, so a crashed job reruns without touching any other. A job whose prediction set exists is
skipped. The predictions go through `studies.experiment_store.write_predictions`, which refuses a
valid time on or after 2026-07-01 and never overwrites a set. The code removes those valid times
(the 10-day horizons of June's runs reach into July) before it writes.

- **Primary demand arms** follow experiment 707's arm `R_F`: the pooled mixture density network of
  experiment 425 (setting `g09`), an unweighted `S_ens` stage on every ENS row from 2021-07-20 to
  the fold's cut, and a fine-tune on the last 730 days at the learning rate of 2e-4 with a
  half-year weight half-life. The arm's cloud columns, the `cloud_missing` flag, the placebo, and
  the gap handling come from `e00711_folds`. `torch.manual_seed` fixes the weights and the epoch
  order. Like experiment 707, the code does not force deterministic GPU kernels.
- **Solar farm arms** follow experiments 659 and 666: PLP with its 18 inputs, a three-component
  Gaussian mixture centred on the mean physics-lite power, 20 epochs, no farm column, and the refit
  of the first of the month. The training window starts at the first row of
  `e00711_solar_inputs.py`, not 2024-04-01. `e00666_network` chooses the GPU if one is visible, so
  this script ignores `--device` for the solar arms (set `CUDA_VISIBLE_DEVICES=` to force the CPU).

**Outputs** under the experiment's data folder: `predictions/<set>/` (the store's set, written
last), `work/logs/<arm>/<month>_s<seed>.json` (rows, epochs, validation loss, input columns, wall
time, peak resident memory, and peak GPU memory), and `work/states/<arm>/<month>/` (the fitted
network).

**The gate is not enforced here.** `e00711_score.py gate` compares `D0n` with the stored `R_F`.

The environment as in `e00711_common.py`; run with the folders of experiments 707 and 666 on
`PYTHONPATH`. The memory-measuring first fit (arm `D1`, fold 2026-06, seed 0):

```bash
export DATA_PATH_INTERNAL=<path to the internal data folder>
export NGED_S3_BUCKET_URL=https://example.com NGED_S3_BUCKET_ACCESS_KEY=dummy
export NGED_S3_BUCKET_SECRET=dummy
E=$PWD/ml-experiments
export PYTHONPATH=$E/00707-ens-backfill-pretraining-gain-against-ens-history-length:\
$E/00666-one-year-backtest-of-simple-wind-and-solar-networks:\
$E/00711-ens-total-cloud-cover-for-primary-demand-and-solar-farms
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 POLARS_MAX_THREADS=1 \
    systemd-run --user --scope -p MemoryMax=16G -p MemorySwapMax=0 -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00711-*/e00711_fit.py --arm D1 --fold 2026-06 --seed 0
```
"""

import argparse
import os
import resource
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00707_arms as arms707  # ty: ignore[unresolved-import]
import e00707_fit as fit707  # ty: ignore[unresolved-import]
import e00707_net as net  # ty: ignore[unresolved-import]
import e00707_prep as prep  # ty: ignore[unresolved-import]
import e00711_arms as arms
import e00711_common as common
import e00711_folds as folds
import e00711_solar as solar
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

from studies import experiment_store

LEVEL_COLUMNS: Final[tuple[str, ...]] = solar.LEVEL_COLUMNS
PREDICTIONS_DIR: Final[Path] = common.WORK_DIR / "predictions"
LOG_DIR: Final[Path] = common.WORK_DIR / "work" / "logs"
STATE_DIR: Final[Path] = common.WORK_DIR / "work" / "states"
PRIMARY_ENS_WINDOW: Final[arms707.Window] = arms707.W_F
"""The `S_ens` window of `R_F`: every ENS row from 2021-07-20 to the cut."""
PRIMARY_FT_WINDOW: Final[arms707.Window] = arms707.W24
SOLAR_EPOCHS: Final[int] = 20


def prediction_path(*, arm: str, month: str, seed: int) -> Path:
    """Return the folder of a job's prediction set, which is the job's completion marker."""
    return PREDICTIONS_DIR / arms.set_name(arm=arm, month=month, seed=seed)


def peak_rss_gb() -> float:
    """Return this process's peak resident memory in gigabytes."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def peak_gpu_gb() -> float | None:
    """Return the peak GPU memory that torch allocated, or `None` without a GPU."""
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / 1e9


def write_log(*, payload: dict, arm: str, month: str, seed: int) -> None:
    """Write a job's log through a temporary name."""
    fit707.write_json(payload=payload, target=LOG_DIR / arm / f"{month}_s{seed}.json")


def write_set(*, arm: arms.Arm, month: str, seed: int, frame: pl.DataFrame) -> Path:
    """Write the job's predictions to the store, after removing valid times from the holdout.

    Args:
        arm: The arm.
        month: The fold.
        seed: The fitting seed.
        frame: The predictions with `valid_time` and `seed`.

    Returns:
        The folder of the written set.
    """
    kept = frame.filter(pl.col("valid_time") < experiment_store.HOLDOUT_START)
    if kept.is_empty():
        msg = "no prediction before 2026-07-01"
        raise ValueError(msg)
    return experiment_store.write_predictions(
        experiment_id=arms.EXPERIMENT_ID,
        name=arms.set_name(arm=arm.name, month=month, seed=seed),
        frame=kept,
        kind="quantiles",
        row_set_id=arm.row_set_id,
    )


# --- Primary demand ---------------------------------------------------------------------------


def primary_forecast(
    *,
    arm: arms.Arm,
    month: str,
    seed: int,
    fitted: net.Fitted,
    scale: net.Scale,
    device: str,
) -> pl.DataFrame:
    """Return a fold's raw quantiles (fraction of capacity, unwidened) for every scored row.

    Args:
        arm: The arm.
        month: The fold.
        seed: The fitting seed.
        fitted: The fine-tuned network.
        scale: The fold's standardiser and target statistics.
        device: The torch device.

    Returns:
        One row per series, origin, and valid time, with the 11 quantiles and `seed`.
    """
    pieces = []
    for scored in folds.scored_rows(arm=arm, month=month):
        row = scored[0]
        keys, x = folds.scored_input(
            arm=arm, scored=scored, month=month, seed=seed, standardiser=scale.standardiser
        )
        p = int(row["primary"])
        q = net.predict_z_quantiles(fitted=fitted, x=x, primary=p, device=device)
        q = np.sort(q * scale.target_std[p] + scale.target_mean[p], axis=1).astype(np.float32)
        pieces.append(keys.hstack(pl.DataFrame(q, schema=list(LEVEL_COLUMNS), orient="row")))
    return pl.concat(pieces).with_columns(seed=pl.lit(seed).cast(pl.Int64))


def fit_primary(*, arm: arms.Arm, month: str, seed: int, device: str, max_epochs: int) -> dict:
    """Fit a Primary demand job: `S_ens`, the fine-tune, and the forecast; return its log.

    Args:
        arm: The arm.
        month: The fold.
        seed: The fitting seed.
        device: The torch device.
        max_epochs: The most epochs of each stage.

    Returns:
        The log payload, and the predictions are written to the store as the last step.
    """
    refit = prep.refit_of(month=month)
    started = time.monotonic()
    fold = folds.build_fold(arm=arm, refit=refit, seed=seed, window=PRIMARY_FT_WINDOW)
    names, _, _ = folds.input_plan(arm=arm)
    n_inputs = fold.layout.n_inputs
    n_primaries = len(fold.target_mean)
    print(
        f"{arm.name} {month} seed {seed}: {len(fold.x):,} fine-tuning rows with {n_inputs} inputs "
        f"({int(fold.is_val.sum()):,} held out); dropped {fold.dropped.as_dict()}",
        flush=True,
    )
    rows, ens_dropped = folds.ens_rows(
        arm=arm, fold=fold, refit=refit, seed=seed, window=PRIMARY_ENS_WINDOW
    )
    print(f"  S_ens: {len(rows.x):,} rows; dropped {ens_dropped.as_dict()}", flush=True)
    t_ens = fit707.tensors(
        x=rows.x,
        y=rows.y,
        primary=rows.primary,
        weight=None,
        is_val=rows.is_val,
        origin=rows.origin,
        flags=False,
        fold=fold,
        device=device,
    )
    n_ens_rows, n_ens_val = len(rows.x), int(rows.is_val.sum())
    del rows
    torch.manual_seed(seed)
    model = net.new_model(n_inputs=n_inputs, n_primaries=n_primaries).to(device)
    ens_epochs, ens_best = fit707.train(
        model=model, setting=net.G09, t=t_ens, seed=seed, max_epochs=max_epochs
    )
    del t_ens
    torch.cuda.empty_cache()
    t_ft = fit707.tensors(
        x=fold.x,
        y=fold.y,
        primary=fold.primary,
        weight=fold.weight,
        is_val=fold.is_val,
        origin=fold.origin,
        flags=False,
        fold=fold,
        device=device,
    )
    setting = replace(net.G09, lr=net.FINE_TUNE_LR)
    epochs, best = fit707.train(
        model=model, setting=setting, t=t_ft, seed=seed, max_epochs=max_epochs
    )
    del t_ft
    torch.cuda.empty_cache()
    scale = net.Scale(
        standardiser=fold.standardiser, target_mean=fold.target_mean, target_std=fold.target_std
    )
    fitted = net.Fitted(model=model, setting=setting, seed=seed, epochs=epochs, val_nll=best)
    net.save_fit(fitted=fitted, scale=scale, target=STATE_DIR / arm.name / month / f"s{seed}")
    forecast = primary_forecast(
        arm=arm, month=month, seed=seed, fitted=fitted, scale=scale, device=device
    )
    log = {
        "arm": arm.name,
        "month": month,
        "seed": seed,
        "target": "primary",
        "input_columns": names,
        "fine_tune_rows": len(fold.x),
        "fine_tune_val_rows": int(fold.is_val.sum()),
        "fine_tune_dropped": fold.dropped.as_dict(),
        "s_ens_rows": n_ens_rows,
        "s_ens_val_rows": n_ens_val,
        "s_ens_dropped": ens_dropped.as_dict(),
        "s_ens_epochs": ens_epochs,
        "s_ens_val_nll": ens_best,
        "epochs": epochs,
        "val_nll": best,
        "forecast_rows": forecast.height,
    }
    return log | {"started": started, "frame": forecast}


# --- solar farms ------------------------------------------------------------------------------


def solar_frame(*, arm: arms.Arm, refit: datetime, seed: int) -> pl.DataFrame:
    """Return every farm's daylight rows with the arm's cloud columns and its capacity ratio.

    The ratio is the refit's capacity (the 99.9th percentile of the farm's cleaned power over 730
    days before the refit) over the farm's stored effective capacity, as in experiment 666.
    """
    from e00666_common import capacity_before  # ty: ignore[unresolved-import]
    from studies.pv_dataset import pv_sites

    month = f"{refit:%Y-%m}"
    sites = pv_sites()
    parts = []
    ratios: dict[str, float] = {}
    for index, row in enumerate(sites.iter_rows(named=True)):
        full = solar.load_farm(site=row["site"])
        rows = solar.daylight_rows(table=full)
        parts.append(
            solar.with_arm_columns(
                rows=rows, full=full, arm=arm, month=month, seed=seed, farm_index=index
            )
        )
        ratios[row["site"]] = capacity_before(
            time_series_id=row["time_series_id"], refit=refit, old_mw=row["effective_capacity_mw"]
        )["ratio"]
    frame = pl.concat(parts).sort("series", "init_time", "time")
    return frame.with_columns(
        ratio=pl.col("series").replace_strict(ratios, return_dtype=pl.Float64)
    )


def fit_solar(*, arm: arms.Arm, month: str, seed: int) -> dict:
    """Fit a solar farm job: PLP on the refit's training rows, and forecast the month's origins.

    Args:
        arm: The arm.
        month: The fold.
        seed: The fitting seed.

    Returns:
        The log payload with the forecast frame.
    """
    import e00666_network as network  # ty: ignore[unresolved-import]

    refit = solar.refit_of(month=month)
    started = time.monotonic()
    frame = solar_frame(arm=arm, refit=refit, seed=seed)
    sel = solar.select_rows(frame=frame, refit=refit, arm=arm)
    columns = solar.input_columns(arm=arm)
    cutoff = refit - timedelta(days=solar.REFIT_GAP_DAYS)
    train = sel.train.with_columns(scaled=pl.col("actual").cast(pl.Float64) / pl.col("ratio"))
    is_stop = (train["time"] >= cutoff - timedelta(days=solar.VALIDATION_DAYS)).to_numpy()
    is_fit = ~is_stop
    x_raw = solar.raw_matrix(frame=train, columns=columns)
    mean, std, x = solar.standardisation(x=x_raw, is_fit=is_fit, columns=columns)
    model, state = network.fit_network(
        x=x,
        mu=train["pl_mean"].to_numpy().astype(np.float64),
        y=train["scaled"].to_numpy(),
        is_fit=is_fit,
        is_stop=is_stop,
        mean=mean,
        std=std,
        epochs=SOLAR_EPOCHS,
        seed=seed,
    )
    target_raw = solar.raw_matrix(frame=sel.target, columns=columns)
    filled = np.where(np.isnan(target_raw), mean[None, :], target_raw)
    fraction = network.predict_quantiles(
        model=model,
        x=filled,
        mu=sel.target["pl_mean"].to_numpy().astype(np.float64),
        mean=mean,
        std=std,
    )
    forecast = solar.prediction_frame(
        target=sel.target, fraction=fraction, ratio=sel.target["ratio"], seed=seed
    )
    target_path = STATE_DIR / arm.name / month / f"s{seed}.pt"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = target_path.with_name(target_path.name + ".tmp")
    torch.save({"state": state, "mean": mean.tolist(), "std": std.tolist()}, temporary)
    temporary.rename(target_path)
    log = {
        "arm": arm.name,
        "month": month,
        "seed": seed,
        "target": "solar",
        "input_columns": columns,
        "train_rows": sel.train.height,
        "stop_rows": int(is_stop.sum()),
        "target_rows": sel.target.height,
        "masked_train_rows": int(sel.train["cloud_missing"].sum()),
        "forecast_rows": forecast.height,
    }
    return log | {"started": started, "frame": forecast}


# --- the prediction gate ----------------------------------------------------------------------


PREDICT_GATE_TOLERANCE: Final[float] = 1e-4
"""The most that a quantile may differ from the stored forecast, as a fraction of capacity."""


def predict_gate(*, month: str, seed: int, device: str) -> float:
    """Reload experiment 707's stored `R_F` state, forecast with this code, and compare.

    The new row builder (`e00711_folds.scored_rows` and `scored_input`, arm `D0n` with no extra
    input) and the stored standardiser must reproduce the stored forecast of `R_F` to 1e-4.

    Args:
        month: The fold.
        seed: The stored seed.
        device: The torch device.

    Returns:
        The largest absolute difference in any quantile.

    Raises:
        ValueError: If the largest difference exceeds `PREDICT_GATE_TOLERANCE` or the rows differ.
    """
    folder = arms707.WORK_DIR / "work" / "models" / "R_F" / month / f"s{seed}"
    fitted = net.load_fitted(folder=folder, device=device)
    scale = net.load_scale(folder=folder)
    arm = arms.ARMS["D0n"]
    new = primary_forecast(
        arm=arm, month=month, seed=seed, fitted=fitted, scale=scale, device=device
    )
    stored = pl.read_parquet(
        arms707.WORK_DIR / "work" / "seed_preds" / "R_F" / f"{month}_s{seed}.parquet"
    )
    keys = ["time_series_id", "power_fcst_init_time", "valid_time"]
    joined = new.join(stored, on=keys, how="inner", suffix="_stored")
    if joined.height != stored.height:
        msg = f"{stored.height - joined.height} stored rows are absent from the new forecast"
        raise ValueError(msg)
    worst = max(
        float((joined[q] - joined[f"{q}_stored"]).abs().max())  # ty: ignore[invalid-argument-type]
        for q in LEVEL_COLUMNS
    )
    if worst > PREDICT_GATE_TOLERANCE:
        msg = f"the prediction gate fails at {month} seed {seed}: largest difference {worst:.2e}"
        raise ValueError(msg)
    return worst


# --- driver -----------------------------------------------------------------------------------


def run_job(*, arm: arms.Arm, month: str, seed: int, device: str, max_epochs: int) -> str:
    """Run one job and return a progress line; skip it if the prediction set exists.

    Args:
        arm: The arm; a stored arm is refused.
        month: The fold, one of the arm's folds.
        seed: The fitting seed, one of the arm's seeds.
        device: The torch device (Primary demand arms).
        max_epochs: The most epochs of each Primary demand stage.

    Returns:
        A progress line.

    Raises:
        ValueError: If the arm is stored, or the fold or seed is not the arm's.
    """
    if arm.stored:
        msg = f"arm {arm.name} is stored and has no fit"
        raise ValueError(msg)
    if month not in arm.months or seed not in arm.seeds:
        msg = f"{arm.name} has folds {arm.months} and seeds {arm.seeds}, not {month} and {seed}"
        raise ValueError(msg)
    if prediction_path(arm=arm.name, month=month, seed=seed).exists():
        return f"{arm.name} {month} seed {seed}: prediction set exists, skipped"
    if arm.target == "primary":
        result = fit_primary(arm=arm, month=month, seed=seed, device=device, max_epochs=max_epochs)
    else:
        result = fit_solar(arm=arm, month=month, seed=seed)
    started = result.pop("started")
    frame = result.pop("frame")
    seconds = round(time.monotonic() - started)
    log = result | {
        "seconds": seconds,
        "peak_rss_gb": round(peak_rss_gb(), 2),
        "peak_gpu_gb": peak_gpu_gb(),
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    write_log(payload=log, arm=arm.name, month=month, seed=seed)
    write_set(arm=arm, month=month, seed=seed, frame=frame)
    return (
        f"{arm.name} {month} seed {seed}: {seconds} s, peak resident {log['peak_rss_gb']} GB, "
        f"peak GPU {log['peak_gpu_gb']} GB, {frame.height:,} rows written"
    )


def main() -> None:
    """Parse the job and run it."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--arm", required=True, choices=sorted(arms.ARMS))
    parser.add_argument("--fold", required=True, help="the fold's month, such as 2026-06")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="validate the job; fit nothing")
    parser.add_argument(
        "--gate-predict",
        action="store_true",
        help="reload the stored R_F state of this fold and seed and compare forecasts; no fit",
    )
    args = parser.parse_args()
    arm = arms.ARMS[args.arm]
    if args.gate_predict:
        worst = predict_gate(month=args.fold, seed=args.seed, device=args.device)
        print(f"prediction gate {args.fold} seed {args.seed}: largest difference {worst:.2e}")
        return
    if args.dry_run:
        done = prediction_path(arm=arm.name, month=args.fold, seed=args.seed).exists()
        print(
            f"{arm.name} {args.fold} seed {args.seed}: columns {arm.extra_columns}, "
            f"{'finished' if done else 'pending'}"
        )
        return
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = str(args.threads)
    torch.set_num_threads(args.threads)
    line = run_job(
        arm=arm, month=args.fold, seed=args.seed, device=args.device, max_epochs=net.MAX_EPOCHS
    )
    print(line, flush=True)


if __name__ == "__main__":
    main()
