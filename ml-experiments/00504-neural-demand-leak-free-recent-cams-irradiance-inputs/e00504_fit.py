# ruff: noqa: E501
r"""Fine-tune experiment 425's arm A with leak-free CAMS columns (experiment 504).

Each arm's pooled plain network starts from the stored pretrained arm A weights of the same month
and seed, with zero weights for the new input columns inserted before the Primary embedding, and
is fine-tuned on the fold's ENS rows with experiment 425's loop (learning rate 2e-4). The fold is
built once per month with all five CAMS columns (`e00504_cams.COLUMNS`) after the base columns;
an arm keeps the columns it names. The new columns are standardised on the fold's training rows.

Arms (see `e00504_cams.ARM_COLUMNS`): `A-cams` (three leak-free columns), `A-cams-real` (realised
unit PV shape and clear-sky index of the target half-hour; report only), and `A-cams-lag-only`
(the daily cloudiness column). After each `A-cams` fit two prediction-only forecasts are written
with no refit: `A-cams-mean` (the new columns set to their training mean) and `A-cams-stale1`
(the leak-free day one day earlier).

```bash
export PYTHONPATH=$(ls -d $PWD/ml-experiments/*/ | tr '\n' ':')
export DATA_PATH_INTERNAL=<path to the private data folder>
OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00504-*/e00504_fit.py --device cuda
```
"""

import argparse
import json
import time
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_regimes as regimes  # ty: ignore[unresolved-import]
import e00397_fit as fit397  # ty: ignore[unresolved-import]
import e00397_prep as base  # ty: ignore[unresolved-import]
import e00407_fit as fit407  # ty: ignore[unresolved-import]
import e00425_fit as fit425  # ty: ignore[unresolved-import]
import e00425_prep as prep425  # ty: ignore[unresolved-import]
import e00504_cams as cams_mod  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00504-neural-demand-leak-free-recent-cams-irradiance-inputs" / "work"
)
MONTHS: Final[tuple[str, ...]] = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
ARMS: Final[tuple[str, ...]] = tuple(cams_mod.ARM_COLUMNS)
FIRST_LAYER_KEY: Final[str] = "net.0.weight"
US_PER_DAY: Final[int] = 86_400 * 1_000_000
CHECK_ARMS: Final[dict[str, str]] = {"A-cams-mean": "mean", "A-cams-stale1": "stale1"}
OUT_DIR: Path = WORK_DIR


def _us(*, t: datetime) -> int:
    """Return a UTC datetime as microseconds since the epoch."""
    return int(t.timestamp() * 1_000_000)


def site_cams_of(*, row: dict) -> cams_mod.SiteCams:
    """Return a Primary's CAMS arrays."""
    return cams_mod.site_cams(
        site=row["site"], latitude=float(row["latitude"]), longitude=float(row["longitude"])
    )


def build_fold(*, refit: datetime) -> prep425.EnsFold:
    """Return experiment 425's fold with the five CAMS columns appended after the base columns.

    Experiment 425's `build_ens_fold`, unchanged except that the CAMS columns are computed from
    each row's own origin and valid time before the standardiser is fitted. The base columns, rows,
    cuts, weights, and early-stopping weeks are therefore the same as the stored arm A's.
    """
    _, mask, layout = base.feature_names(scheme="summary")
    mask = np.concatenate([mask, np.ones(len(cams_mod.COLUMNS), dtype=bool)])
    flagged = fit407.flagged_times()
    dropped = prep425.DroppedRows()
    parts = []
    for row in base.primaries().iter_rows(named=True):
        z = np.load(base.RAW_DIR / "summary" / f"{row['site']}.npz")
        keep = (
            (z["valid"] <= _us(t=refit - regimes.REFIT_CUTOFF_AGE))
            & (z["origin"] <= _us(t=refit))
            & (z["valid"] > _us(t=refit - timedelta(days=base.WINDOW_DAYS)))
        )
        keep, counts = prep425.apply_cuts(
            keep=keep,
            valid=z["valid"],
            lead=z["lead"],
            site=row["site"],
            series_id=int(row["time_series_id"]),
            flagged=flagged,
        )
        dropped += counts
        if not keep.any():
            continue
        extra = cams_mod.extra_columns(
            cams=site_cams_of(row=row), origin_us=z["origin"][keep], valid_us=z["valid"][keep]
        )
        assert np.isfinite(extra).all(), f"{row['site']}: a CAMS column is not finite"
        parts.append(
            {"x": np.concatenate([z["x"][keep], extra], axis=1)}
            | {k: z[k][keep] for k in ("y", "origin", "valid", "lead")}
            | {"primary": np.full(int(keep.sum()), row["primary"], dtype=np.int64)}
        )
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    del parts
    origin_day = ((cat["origin"] - _us(t=base.FIRST_RUN)) // US_PER_DAY).astype(np.int64)
    held, purged = base.held_out_weeks(origin_day=origin_day)
    train = ~held & ~purged
    standardiser = base.Standardiser.fit(x=cat["x"][train], mask=mask)
    age_days = (_us(t=refit) - cat["valid"]) / US_PER_DAY
    weight = 0.5 ** (age_days / common.SHORT_HALF_LIFE_DAYS)
    n_primaries = len(base.primaries())
    target_mean, target_std = np.zeros(n_primaries), np.ones(n_primaries)
    for p in range(n_primaries):
        sel = train & (cat["primary"] == p)
        if sel.any():
            target_mean[p] = cat["y"][sel].mean()
            target_std[p] = max(float(cat["y"][sel].std()), 1e-3)
    keep = train | held
    return prep425.EnsFold(
        x=standardiser.apply(x=cat["x"][keep]),
        y=cat["y"][keep],
        primary=cat["primary"][keep],
        weight=weight[keep].astype(np.float32),
        is_val=held[keep],
        lead=cat["lead"][keep],
        origin_day=origin_day[keep],
        standardiser=standardiser,
        target_mean=target_mean,
        target_std=target_std,
        layout=layout,
        valid=cat["valid"][keep],
        dropped=dropped,
    )


def arm_columns(*, arm: str, n_base: int) -> np.ndarray:
    """Return the indices of the base columns and the arm's CAMS columns in the fold's matrix."""
    extra = [n_base + cams_mod.COLUMNS.index(c) for c in cams_mod.ARM_COLUMNS[arm]]
    return np.array([*range(n_base), *extra])


def arm_standardiser(*, fold: prep425.EnsFold, cols: np.ndarray) -> base.Standardiser:
    """Return the fold's standardiser restricted to an arm's columns."""
    s = fold.standardiser
    return base.Standardiser(mean=s.mean[cols], std=s.std[cols], mask=s.mask[cols])


def padded_state(*, state: dict, n_base: int, n_new: int) -> dict:
    """Return a pretrained state dict with zero first-layer columns inserted after the base columns.

    The network's first layer reads the input row and then the Primary embedding, so the new
    columns go between them.
    """
    state = dict(state)
    weight = state[FIRST_LAYER_KEY]
    zeros = torch.zeros(weight.shape[0], n_new, dtype=weight.dtype)
    state[FIRST_LAYER_KEY] = torch.cat([weight[:, :n_base], zeros, weight[:, n_base:]], dim=1)
    return state


def fine_tune(
    *,
    arm: str,
    month: str,
    seed: int,
    n_base: int,
    n_new: int,
    n_primaries: int,
    layout: object,
    t: dict[str, torch.Tensor],
    device: str,
) -> fit397.Fitted:
    """Fine-tune the stored pretrained arm A network of this month and seed with the new columns."""
    torch.manual_seed(seed)
    model = fit397.make_model(
        arm="plain_pooled",
        setting=fit425.SETTING,
        layout=layout,  # ty: ignore[invalid-argument-type]
        n_primaries=n_primaries,
        n_inputs=n_base + n_new,
    ).to(device)
    stored = fit425.pretrained_path(arm=fit425.ARMS["A"], cut=True, month=month, seed=seed)
    state = torch.load(stored, map_location="cpu")
    model.load_state_dict(padded_state(state=state, n_base=n_base, n_new=n_new))
    setting = replace(fit425.SETTING, lr=fit425.FINE_TUNE_LR)
    epochs, best = fit425.train(model=model, setting=setting, t=t, seed=seed)
    return fit397.Fitted(
        model=model, setting=setting, arm="plain_pooled", seed=seed, epochs=epochs, val_nll=best
    )


def forecast(
    *,
    fitted: fit397.Fitted,
    fold: prep425.EnsFold,
    standardiser: base.Standardiser,
    cols: np.ndarray,
    n_base: int,
    month: str,
    device: str,
    mode: str,
) -> pl.DataFrame:
    """Return the month's raw quantiles from experiment 397's forecast step with CAMS columns.

    `mode` is `normal` (the arm's columns from the leak-free or realised rule), `mean` (the CAMS
    columns of the standardised input set to 0, their training mean), or `stale1` (the leak-free
    day moved one day earlier).
    """
    original = base.run_inputs
    sites = {r["time_series_id"]: r for r in base.primaries().iter_rows(named=True)}
    extra_cols = cols[n_base:] - n_base
    base_standardiser = base.Standardiser(
        mean=standardiser.mean[:n_base],
        std=standardiser.std[:n_base],
        mask=standardiser.mask[:n_base],
    )
    extra_standardiser = base.Standardiser(
        mean=standardiser.mean[n_base:],
        std=standardiser.std[n_base:],
        mask=standardiser.mask[n_base:],
    )

    def with_cams(*, origin: datetime, scheme: str, standardiser: object) -> list:
        """Append the CAMS columns to experiment 397's standardised run inputs."""
        del standardiser
        out = []
        for row, keys, x in original(origin=origin, scheme=scheme, standardiser=base_standardiser):
            if mode == "mean":
                extra = np.zeros((len(x), len(extra_cols)), dtype=np.float32)
            else:
                raw = cams_mod.extra_columns(
                    cams=site_cams_of(row=sites[row["time_series_id"]]),
                    origin_us=keys["power_fcst_init_time"].dt.epoch("us").to_numpy(),
                    valid_us=keys["valid_time"].dt.epoch("us").to_numpy(),
                    stale_days=1 if mode == "stale1" else 0,
                )
                extra = extra_standardiser.apply(x=raw[:, extra_cols])
            out.append((row, keys, np.concatenate([x, extra], axis=1)))
        return out

    base.run_inputs = with_cams  # ty: ignore[invalid-assignment]
    try:
        stand = SimpleNamespace(
            standardiser=standardiser, target_mean=fold.target_mean, target_std=fold.target_std
        )
        q, _ = fit397.forecast_month(
            fits={None: fitted},
            fold=stand,  # ty: ignore[invalid-argument-type]
            scheme="summary",
            month=month,
            device=device,  # ty: ignore[invalid-argument-type]
        )
    finally:
        base.run_inputs = original
    return q


def final_path(*, arm: str, month: str, seed: int) -> Path:
    """Return where an arm's seed-level raw quantile forecast is stored."""
    return OUT_DIR / "seed_preds" / arm / f"{month}_s{seed}.parquet"


def run(*, months: list[str], arms: list[str], seeds: list[int], device: str) -> None:
    """Fine-tune each arm on each fold and seed, store the forecasts and the networks."""
    for month in months:
        todo = [
            (a, s)
            for a in arms
            for s in seeds
            if not final_path(arm=a, month=month, seed=s).exists()
        ]
        if not todo:
            continue
        started = time.monotonic()
        fold = build_fold(refit=base.refit_of(month=month))
        n_base = fold.x.shape[1] - len(cams_mod.COLUMNS)
        log_path = OUT_DIR / "logs" / f"{month}.json"
        log: dict = json.loads(log_path.read_text()) if log_path.exists() else {}
        log |= {
            "month": month,
            "rows": len(fold.x),
            "val_rows": int(fold.is_val.sum()),
            "rows_dropped": fold.dropped.as_dict(),
            "fold_build_seconds": round(time.monotonic() - started),
        }
        print(f"fold {month}: {len(fold.x):,} rows; dropped {fold.dropped.as_dict()}", flush=True)
        for arm in arms:
            cols = arm_columns(arm=arm, n_base=n_base)
            standardiser = arm_standardiser(fold=fold, cols=cols)
            seeds_todo = [s for a, s in todo if a == arm]
            if not seeds_todo:
                continue
            t = fit425.tensors(
                x=np.ascontiguousarray(fold.x[:, cols]),
                y=fold.y,
                primary=fold.primary,
                weight=fold.weight,
                is_val=fold.is_val,
                fold=fold,
                device=device,
            )
            for seed in seeds_todo:
                began = time.monotonic()
                fitted = fine_tune(
                    arm=arm, month=month, seed=seed, n_base=n_base, n_new=len(cols) - n_base,
                    n_primaries=len(fold.target_mean), layout=fold.layout, t=t, device=device,
                )  # fmt: skip
                stand = SimpleNamespace(
                    standardiser=standardiser,
                    target_mean=fold.target_mean,
                    target_std=fold.target_std,
                )
                fit397.save_fits(
                    fits={None: fitted},
                    fold=stand,  # ty: ignore[invalid-argument-type]
                    target=OUT_DIR / "models" / arm / month / f"s{seed}",
                )
                outputs = (CHECK_ARMS if arm == "A-cams" else {}) | {
                    arm: "normal"
                }  # the arm itself last: it marks the fit done
                for name, mode in outputs.items():
                    q = forecast(
                        fitted=fitted, fold=fold, standardiser=standardiser, cols=cols,
                        n_base=n_base, month=month, device=device, mode=mode,
                    )  # fmt: skip
                    common.atomic_write(
                        frame=q, target=final_path(arm=name, month=month, seed=seed)
                    )
                seconds = round(time.monotonic() - began)
                log[f"{arm}_s{seed}"] = {
                    "epochs": fitted.epochs,
                    "val_nll": fitted.val_nll,
                    "seconds": seconds,
                }
                print(
                    f"{arm} {month} seed {seed}: {fitted.epochs} epochs, val NLL {fitted.val_nll:.4f}, "
                    f"{seconds} s (fine-tune and forecast)", flush=True,
                )  # fmt: skip
                log_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = log_path.with_suffix(".tmp")
                temporary.write_text(json.dumps(log, indent=1))
                temporary.replace(log_path)
            del t
            torch.cuda.empty_cache()
        del fold


def main() -> None:
    """Fine-tune the named arms on the named folds and seeds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", nargs="+", default=list(MONTHS))
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true", help="one epoch, outputs under work/smoke")
    args = parser.parse_args()
    if args.smoke:
        global OUT_DIR  # noqa: PLW0603
        OUT_DIR = WORK_DIR / "smoke"
        fit397.MAX_EPOCHS = 1
    run(months=args.months, arms=args.arms, seeds=args.seeds, device=args.device)


if __name__ == "__main__":
    main()
