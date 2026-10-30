# ruff: noqa: E501
"""Fit the known-answer arms of experiment 409 (the score stages are added only if the gate holds).

Experiment 405's `train`, `make_model`, and `capacity_table` are reused unchanged. Two arm names are
aliases of experiment 405 specs: `A0-noirr-doy` is `A0-noirr` and `C2-nodoy` is `C2` fitted without
the day-of-year inputs (every other arm has them).

```bash
export DATA_PATH_INTERNAL=<the private data folder>
export PYTHONPATH=ml-experiments/00087-...:ml-experiments/00397-...:ml-experiments/00405-...:ml-experiments/00409-...
OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00409-*/e00409_fit.py kat --device cuda
```
"""

import argparse
import json
import time
from datetime import datetime
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_fit_demand as fd  # ty: ignore[unresolved-import]
import e00397_models as base_models  # ty: ignore[unresolved-import]
import e00397_prep as base  # ty: ignore[unresolved-import]
import e00405_fit as f5  # ty: ignore[unresolved-import]
import e00405_models as m5  # ty: ignore[unresolved-import]
import e00405_prep as p5  # ty: ignore[unresolved-import]
import e00405_pv as pv  # ty: ignore[unresolved-import]
import e00409_prep as prep  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

m5.SPECS["A0-noirr-doy"] = m5.SPECS["A0-noirr"]
m5.SPECS["C2-nodoy"] = m5.SPECS["C2"]
OUT: Final = prep.WORK
KAT_REFIT: Final[str] = "2026-06"
FARM_CONDITIONS: Final[tuple[str, ...]] = (
    "control",
    "held_profile",
    "placebo_profile",
    "held_anomaly",
    "placebo_anomaly",
)
PLAN: Final[dict[str, tuple[tuple[str, ...], tuple[int, ...]]]] = {
    "C2": (prep.CONDITIONS, (0, 1, 2, 3, 4)),
    "C4": (FARM_CONDITIONS, (0, 1, 2)),
    "C2-nodoy": (("control", "pos_profile"), (0, 1, 2)),
}
"""Arm -> (conditions, seeds): 7x5 + 5x3 + 2x3 = 56 fits."""


def uses_doy(*, arm: str) -> bool:
    """Return whether the arm's demand module has the day-of-year inputs."""
    return arm != "C2-nodoy"


def fit_one(
    *, arm: str, fold: prep.Fold, refit: datetime, seed: int, device: str
) -> tuple[m5.PvMdn, int, float]:
    """Fit one arm on a fold; return the network, its epochs, and its validation loss."""
    model = f5.make_model(arm=arm, fold=fold, table=base.primaries(), refit=refit, seed=seed)
    epochs, val = f5.train(model=model, arm=arm, fold=fold, seed=seed, device=device)
    return model, epochs, val


def run_kat(
    *, device: str, arms: list[str], conditions: list[str] | None, seeds: list[int] | None
) -> None:
    """Fit the known-answer arms under each condition; store capacities and the saved network."""
    refit = base.refit_of(month=KAT_REFIT)
    for arm in arms:
        conds, plan_seeds = PLAN[arm]
        for cond in conditions or conds:
            fold = None
            for seed in seeds or plan_seeds:
                target = OUT / "kat" / arm / f"{cond}_s{seed}.json"
                if target.exists():
                    continue
                if fold is None:
                    fold = prep.build_fold(refit=refit, injection=cond, use_doy=uses_doy(arm=arm))
                started = time.monotonic()
                model, epochs, val = fit_one(
                    arm=arm, fold=fold, refit=refit, seed=seed, device=device
                )
                cap = f5.capacity_table(model=model, table=base.primaries())
                target.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), target.with_suffix(".pt"))
                target.write_text(
                    json.dumps(
                        {
                            "capacity_mw": cap,
                            "epochs": epochs,
                            "val_nll": val,
                            "as_of_year_cell": fold.as_of_year_cell,
                            "rows": len(fold.x),
                        }
                    )
                )
                print(
                    f"kat {arm} {cond} seed {seed}: {epochs} epochs, val {val:.4f}, {time.monotonic() - started:.0f} s",
                    flush=True,
                )
            del fold


SCORE_ARMS: Final[tuple[str, ...]] = ("A0-noirr-doy", "C1", "C2", "C3", "C4", "C5a", "C5b")
SCREEN_SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
SEED_DIR: Final = prep.WORK / "seed_preds"


def run_inputs(
    *, origin: datetime, fold: prep.Fold
) -> list[tuple[dict, pl.DataFrame, np.ndarray, np.ndarray]]:
    """Return, per Primary, the keys, the demand inputs (with the day-of-year inputs), and the extras."""
    _series, arrays = fd._series_and_arrays()
    frame = pl.read_parquet(common.demand_frame_path(run=origin - common.PUBLICATION_DELAY))
    out = []
    for row in base.primaries().iter_rows(named=True):
        prepared = fd.prepared_run(
            run_rows=frame.filter(pl.col("time_series_id") == row["time_series_id"]),
            row=row,
            arrays=arrays[row["site"]],
            a7=False,
        )
        summary = pl.read_parquet(
            base.SUMMARY_DIR / f"init={origin - common.PUBLICATION_DELAY:%Y-%m-%d}.parquet"
        ).filter(pl.col("time_series_id") == row["time_series_id"])
        first = prepared.filter(pl.col("ensemble_member") == pl.col("ensemble_member").min())
        prepared = first.join(
            summary.drop("time_series_id"),
            on=["power_fcst_init_time", "valid_time"],
            how="inner",
            validate="1:1",
            maintain_order="left",
        )
        keys = prepared.select(*common.KEYS, "lead_hours").unique(
            subset=list(common.KEYS), maintain_order=True
        )
        raw = base.feature_matrix(frame=prepared, scheme="summary")
        valid_us = prepared["valid_time"].dt.epoch("us").to_numpy()
        p5.mask_short_history_lags(x=raw, valid_us=valid_us, site=row["site"])
        extras = p5.extras_from_raw(x=raw, valid_us=valid_us)
        x = np.concatenate(
            [
                fold.standardiser.apply(x=raw)[:, fold.keep],
                prep.doy_hour_inputs(x_raw=raw, valid_us=valid_us),
            ],
            axis=1,
        )
        out.append((row, keys, x, extras))
    return out


def solar_mw(*, model: m5.PvMdn, p: int, cell: int, extras: torch.Tensor) -> torch.Tensor:
    """Return the learned solar output in MW: capacity times the (adapted) unit shape, clamped."""
    u = extras[:, 0]
    if model.adapter is not None:
        g = torch.exp(m5.ADAPTER_LOG_RANGE * torch.tanh(model.adapter(extras[:, 2:]).squeeze(-1)))
    else:
        g = torch.ones_like(u)
    return model.capacity()[p, cell] * torch.clamp(u * g, 0.0, 1.0)


def forecast_month(
    *, model: m5.PvMdn, arm: str, fold: prep.Fold, month: str, device: str
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return the test month's quantile forecast (fraction of capacity) and the solar output (MW)."""
    spec = m5.SPECS[arm]
    cell = fold.as_of_month_cell if spec.cells == "month" else fold.as_of_year_cell
    pieces, solar = [], []
    for origin in base.test_origins(month=month):
        for row, keys, x, extras in run_inputs(origin=origin, fold=fold):
            p = int(row["primary"])
            quantiles, mw = [], []
            with torch.no_grad():
                for start in range(0, len(x), f5.SCORE_CHUNK_ROWS):
                    sl = slice(start, start + f5.SCORE_CHUNK_ROWS)
                    xb = torch.from_numpy(x[sl]).to(device)
                    eb = torch.from_numpy(extras[sl]).to(device)
                    pb = torch.full((len(xb),), p, dtype=torch.int64, device=device)
                    cb = torch.full((len(xb),), cell, dtype=torch.int64, device=device)
                    log_w, mean, std = model.parts(xb, pb, eb, cb)
                    quantiles.append(
                        base_models.mixture_quantiles(
                            log_w=log_w.double(), mean=mean.double(), std=std.double()
                        )
                        .cpu()
                        .numpy()
                    )
                    if spec.solar and row["site"] in pv.SIGNED_SITES:
                        mw.append(solar_mw(model=model, p=p, cell=cell, extras=eb).cpu().numpy())
            q = np.sort(
                np.concatenate(quantiles) * fold.target_std[p] + fold.target_mean[p], axis=1
            )
            k = keys.drop("lead_hours")
            pieces.append(
                k.hstack(
                    pl.DataFrame(
                        q.astype(np.float32), schema=list(common.LEVEL_COLUMNS), orient="row"
                    )
                )
            )
            if mw:
                solar.append(
                    k.with_columns(solar_mw=pl.Series(np.concatenate(mw).astype(np.float32)))
                )
    return pl.concat(pieces), (pl.concat(solar) if solar else pl.DataFrame())


def run_screen(*, device: str, arms: list[str], months: list[str], seeds: list[int]) -> None:
    """Fit each arm on each screen month and seed; store forecasts, solar output, and networks."""
    for month in months:
        refit = base.refit_of(month=month)
        fold = None
        for arm in arms:
            for seed in seeds:
                target = SEED_DIR / arm / f"{month}_s{seed}.parquet"
                if target.exists():
                    continue
                if fold is None:
                    fold = prep.build_fold(refit=refit, use_doy=True)
                started = time.monotonic()
                model, epochs, val = fit_one(
                    arm=arm, fold=fold, refit=refit, seed=seed, device=device
                )
                frame, solar = forecast_month(
                    model=model, arm=arm, fold=fold, month=month, device=device
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), target.with_suffix(".pt"))
                if solar.height:
                    solar.write_parquet(target.with_suffix(".solar.parquet"))
                (target.with_suffix(".capacity.json")).write_text(
                    json.dumps(f5.capacity_table(model=model, table=base.primaries()))
                )
                common.atomic_write(frame=frame, target=target)
                print(
                    f"{arm} {month} seed {seed}: {epochs} epochs, val {val:.4f}, {time.monotonic() - started:.0f} s",
                    flush=True,
                )
        del fold


def main() -> None:
    """Run a stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("kat", "screen"))
    parser.add_argument("--arms", nargs="+", default=None)
    parser.add_argument("--months", nargs="+", default=list(base.SCREEN_MONTHS))
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.stage == "screen":
        run_screen(
            device=args.device,
            arms=args.arms or list(SCORE_ARMS),
            months=args.months,
            seeds=args.seeds or list(SCREEN_SEEDS),
        )
        return
    run_kat(
        device=args.device,
        arms=args.arms or list(PLAN),
        conditions=args.conditions,
        seeds=args.seeds,
    )


if __name__ == "__main__":
    main()
