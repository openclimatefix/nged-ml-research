# ruff: noqa: E501
"""Fit and forecast the solar-term mixture density networks of experiment 405.

Stages (each writes only under this experiment's private data folder):

- `smoke`: fit one arm for a few epochs on the 2026-06 fold and print the loss and the capacity.
- `screen`: fit and forecast the named arms on the screen folds and seeds.
- `kat`: the known-answer test. Fit C2 and C4 (and C2 with the increment penalty) on the 2026-06
  fold under each injection condition, and write the learned capacities.

```bash
source scripts_env.sh   # PYTHONPATH of the folders this file imports, DATA_PATH_INTERNAL
OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00405-*/e00405_fit.py kat --device cuda
```
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_fit_demand as fd  # ty: ignore[unresolved-import]
import e00397_models as base_models  # ty: ignore[unresolved-import]
import e00397_prep as base  # ty: ignore[unresolved-import]
import e00405_models as models  # ty: ignore[unresolved-import]
import e00405_prep as prep  # ty: ignore[unresolved-import]
import e00405_pv as pv  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

BATCH: Final[int] = 16_384
MAX_EPOCHS: Final[int] = 40
PATIENCE: Final[int] = 6
GRAD_CLIP: Final[float] = 1.0
LR: Final[float] = 5e-4
LR_CAPACITY: Final[float] = 5e-3
WEIGHT_DECAY: Final[float] = 1e-3
SCORE_CHUNK_ROWS: Final[int] = 8_192
OUT_DIR: Final[Path] = pv.WORK_DIR
KAT_REFIT: Final[str] = "2026-06"
KAT_SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
KAT_CONDITIONS: Final[dict[str, str]] = {
    "control": "farmC_share000",
    "held": "farmC_share050",
    "divisor_A": "farmA_share050",
    "divisor_B": "farmB_share050",
    "divisor_D": "farmD_share050",
    "placebo": "farmC_share050_placebo",
}
KAT_ARMS: Final[tuple[str, ...]] = ("C2", "C4")


def arm_tables(*, arm: str, fold: prep.Fold, table: pl.DataFrame) -> dict[str, np.ndarray]:
    """Return the per-Primary, per-cell arrays that build a `PvMdn`.

    Args:
        arm: The arm name.
        fold: The fold (for the target standard deviation).
        table: The Primary table of experiment 397.

    Returns:
        `solar_mask`, `capacity_mw`, `prior_centre`, `prior_sd` (P, T) and `scale` (P,).
    """
    spec = models.SPECS[arm]
    n = len(table)
    n_cells = prep.N_MONTH_CELLS if spec.cells == "month" else prep.N_YEAR_CELLS
    month_year = [
        int(np.clip((2024 + (3 + m) // 12 if (3 + m) % 12 + 1 >= 9 else 2024 + (3 + m) // 12 - 1) - 2023, 0, 2))
        for m in range(prep.N_MONTH_CELLS)
    ]
    solar = np.zeros(n, dtype=bool)
    capacity = np.ones((n, n_cells))
    centre = np.ones((n, n_cells))
    sd = np.ones((n, n_cells))
    scale = np.zeros(n)
    for row in table.iter_rows(named=True):
        p, site = int(row["primary"]), row["site"]
        scale[p] = 1.0 / (float(row["effective_capacity_mw"]) * float(fold.target_std[p]))
        if site not in pv.SIGNED_SITES:
            continue
        solar[p] = True
        c_year, s_year = pv.studies_prior(site=site, n_cells=prep.N_YEAR_CELLS)
        # Experiment 269 covers the six only; the report-only D15 and D18 pin to the studies page.
        pinned = (
            pv.pinned_e269(site=site, n_cells=prep.N_YEAR_CELLS)
            if site in pv.SIGNED_SIX
            else c_year
        )
        pick = month_year if spec.cells == "month" else list(range(prep.N_YEAR_CELLS))
        centre[p], sd[p] = c_year[pick], s_year[pick]
        capacity[p] = (pinned if arm == "C1" else c_year)[pick]
    return {
        "solar_mask": solar,
        "capacity_mw": capacity,
        "prior_centre": centre,
        "prior_sd": sd,
        "scale": scale,
    }


def make_model(*, arm: str, fold: prep.Fold, table: pl.DataFrame) -> models.PvMdn:
    """Return an untrained network for an arm."""
    return models.PvMdn(
        spec=models.SPECS[arm],
        n_inputs=fold.x.shape[1],
        n_primaries=len(table),
        **arm_tables(arm=arm, fold=fold, table=table),
    )


def cells_of(*, arm: str, fold: prep.Fold) -> np.ndarray:
    """Return each row's capacity cell for the arm."""
    return fold.month_cell if models.SPECS[arm].cells == "month" else fold.year_cell


def train(
    *, model: models.PvMdn, arm: str, fold: prep.Fold, seed: int, device: str, max_epochs: int = MAX_EPOCHS
) -> tuple[int, float]:
    """Train with weighted net-demand negative log-likelihood and early stopping."""
    torch.manual_seed(seed)
    z_all = (fold.y - fold.target_mean[fold.primary]) / fold.target_std[fold.primary]
    t = {
        "x": torch.from_numpy(fold.x).to(device),
        "z": torch.from_numpy(z_all.astype(np.float32)).to(device),
        "w": torch.from_numpy(fold.weight).to(device),
        "p": torch.from_numpy(fold.primary).to(device),
        "e": torch.from_numpy(fold.extras).to(device),
        "c": torch.from_numpy(cells_of(arm=arm, fold=fold)).to(device),
        "v": torch.from_numpy(fold.is_val).to(device),
    }
    model.to(device)
    train_idx = torch.nonzero(~t["v"]).squeeze(1)
    val_idx = torch.nonzero(t["v"]).squeeze(1)
    n_eff = float(t["w"][train_idx].sum())
    groups = [
        {"params": [p for n, p in model.named_parameters() if n != "log_c"], "lr": LR, "weight_decay": WEIGHT_DECAY}
    ]
    if models.SPECS[arm].learn_capacity:
        groups.append({"params": [model.log_c], "lr": LR_CAPACITY, "weight_decay": 0.0})
    optimiser = torch.optim.AdamW(groups)

    def batch_nll(b: torch.Tensor) -> torch.Tensor:
        log_w, mean, std = model.parts(t["x"][b], t["p"][b], t["e"][b], t["c"][b])
        return models.nll_parts(log_w=log_w, mean=mean, std=std, z=t["z"][b])

    best, best_state, stale, epochs = float("inf"), None, 0, 0
    for epoch in range(max_epochs):
        model.train()
        order = train_idx[torch.randperm(len(train_idx), device=device)]
        for start in range(0, len(order), BATCH):
            b = order[start : start + BATCH]
            nll = batch_nll(b)
            loss = (t["w"][b] * nll).sum() / t["w"][b].sum() + model.penalty() / n_eff
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimiser.step()
        model.eval()
        with torch.no_grad():
            val = float(torch.cat([batch_nll(b) for b in val_idx.split(BATCH * 4)]).mean())
        epochs = epoch + 1
        if val < best - 1e-5:
            best, stale = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return epochs, best


def capacity_table(*, model: models.PvMdn, table: pl.DataFrame) -> dict[str, list[float]]:
    """Return each signed Primary's learned capacity (MW) by cell."""
    c = model.capacity().detach().cpu().numpy()
    return {r["site"]: c[int(r["primary"])].tolist() for r in table.iter_rows(named=True) if r["site"] in pv.SIGNED_SITES}


# --- forecasting ---------------------------------------------------------------------------


def run_inputs(
    *, origin: datetime, fold: prep.Fold
) -> list[tuple[dict, pl.DataFrame, np.ndarray, np.ndarray]]:
    """Return, per Primary, the keys, the standardised demand inputs, and the solar extras of a run."""
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
        keys = prepared.select(*common.KEYS, "lead_hours").unique(subset=list(common.KEYS), maintain_order=True)
        raw = base.feature_matrix(frame=prepared, scheme="summary")
        valid_us = prepared["valid_time"].dt.epoch("us").to_numpy()
        prep.mask_short_history_lags(x=raw, valid_us=valid_us, site=row["site"])
        extras = prep.extras_from_raw(x=raw, valid_us=valid_us)
        x = fold.standardiser.apply(x=raw)[:, fold.keep]
        out.append((row, keys, x, extras))
    return out


def forecast_month(
    *, model: models.PvMdn, arm: str, fold: prep.Fold, month: str, device: str
) -> pl.DataFrame:
    """Return the test month's quantiles (fraction of capacity) for the arm."""
    spec = models.SPECS[arm]
    cell = fold.as_of_month_cell if spec.cells == "month" else fold.as_of_year_cell
    pieces = []
    for origin in base.test_origins(month=month):
        for row, keys, x, extras in run_inputs(origin=origin, fold=fold):
            p = int(row["primary"])
            quantiles = []
            with torch.no_grad():
                for start in range(0, len(x), SCORE_CHUNK_ROWS):
                    xb = torch.from_numpy(x[start : start + SCORE_CHUNK_ROWS]).to(device)
                    eb = torch.from_numpy(extras[start : start + SCORE_CHUNK_ROWS]).to(device)
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
            q = np.sort(np.concatenate(quantiles) * fold.target_std[p] + fold.target_mean[p], axis=1)
            pieces.append(
                keys.drop("lead_hours").hstack(
                    pl.DataFrame(q.astype(np.float32), schema=list(common.LEVEL_COLUMNS), orient="row")
                )
            )
    return pl.concat(pieces)


# --- stages --------------------------------------------------------------------------------


def fit_one(
    *, arm: str, fold: prep.Fold, seed: int, device: str, max_epochs: int = MAX_EPOCHS
) -> tuple[models.PvMdn, int, float]:
    """Fit one arm on a fold and return the network, its epochs, and its validation loss."""
    table = base.primaries()
    model = make_model(arm=arm, fold=fold, table=table)
    epochs, val = train(model=model, arm=arm, fold=fold, seed=seed, device=device, max_epochs=max_epochs)
    return model, epochs, val


def run_smoke(*, arm: str, device: str) -> None:
    """Fit an arm for 2 epochs on the 2026-06 fold and print what happened."""
    started = time.monotonic()
    fold = prep.build_fold(
        refit=base.refit_of(month=KAT_REFIT), irradiance_free=models.SPECS[arm].irradiance_free
    )
    print(f"fold: {len(fold.x):,} rows x {fold.x.shape[1]} inputs ({time.monotonic() - started:.0f} s)", flush=True)
    model, epochs, val = fit_one(arm=arm, fold=fold, seed=0, device=device, max_epochs=2)
    cap = capacity_table(model=model, table=base.primaries())
    print(f"{arm}: {epochs} epochs, val nll {val:.4f}, {time.monotonic() - started:.0f} s")
    print({k: [round(x, 2) for x in v] for k, v in cap.items()})
    print(f"gpu max memory {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


def run_screen(*, arms: list[str], seeds: list[int], months: list[str], device: str) -> None:
    """Fit each arm on each fold and seed and forecast the test month."""
    for month in months:
        for arm in arms:
            fold = prep.build_fold(
                refit=base.refit_of(month=month), irradiance_free=models.SPECS[arm].irradiance_free
            )
            for seed in seeds:
                target = OUT_DIR / "seed_preds" / "summary" / arm / f"{month}_s{seed}.parquet"
                if target.exists():
                    continue
                started = time.monotonic()
                model, epochs, val = fit_one(arm=arm, fold=fold, seed=seed, device=device)
                frame = forecast_month(model=model, arm=arm, fold=fold, month=month, device=device)
                common.atomic_write(frame=frame, target=target)
                cap = capacity_table(model=model, table=base.primaries())
                (target.with_suffix(".capacity.json")).write_text(json.dumps(cap))
                print(f"{arm} {month} seed {seed}: {epochs} epochs, val {val:.4f}, {time.monotonic() - started:.0f} s", flush=True)
            del fold


def run_kat(*, device: str, conditions: list[str], arms: list[str]) -> None:
    """Fit the known-answer arms under each injection condition and store the learned capacities."""
    refit = base.refit_of(month=KAT_REFIT)
    for arm in arms:
        fold_for: dict[str, prep.Fold] = {}
        for cond in conditions:
            for seed in KAT_SEEDS:
                target = OUT_DIR / "kat" / arm / f"{cond}_s{seed}.json"
                if target.exists():
                    continue
                if cond not in fold_for:
                    fold_for.clear()
                    fold_for[cond] = prep.build_fold(
                        refit=refit,
                        irradiance_free=True,
                        injection=KAT_CONDITIONS[cond],
                    )
                started = time.monotonic()
                model, epochs, val = fit_one(arm=arm, fold=fold_for[cond], seed=seed, device=device)
                cap = capacity_table(model=model, table=base.primaries())
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(
                    json.dumps({"capacity_mw": cap, "epochs": epochs, "val_nll": val, "as_of_year_cell": fold_for[cond].as_of_year_cell})
                )
                print(f"kat {arm} {cond} seed {seed}: {epochs} epochs, val {val:.4f}, {time.monotonic() - started:.0f} s", flush=True)


def main() -> None:
    """Run a stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "screen", "kat"))
    parser.add_argument("--arms", nargs="+", default=["C2"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--months", nargs="+", default=list(base.SCREEN_MONTHS))
    parser.add_argument("--conditions", nargs="+", default=list(KAT_CONDITIONS))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.stage == "smoke":
        run_smoke(arm=args.arms[0], device=args.device)
    elif args.stage == "screen":
        run_screen(arms=args.arms, seeds=args.seeds, months=args.months, device=args.device)
    else:
        run_kat(device=args.device, conditions=args.conditions, arms=args.arms)


if __name__ == "__main__":
    main()
