# ruff: noqa: E501
r"""Fine-tune experiment 425's arm A with one extra explicit solar-output input column.

The column is the capacity of the Primary at the fold (`e00501_capacity.py`, out of fold, megawatts
of alternating current) times experiment 405's unit shape of the ensemble-mean weather (the stored
`extras` column 0, the shape divided by `u99`), divided by `PV_SCALE`. It is zero at the Primaries
other than the signed-MW six. Experiment 425's pretrained weights are loaded with a zero first-layer
weight for the new column, so the fine-tuning starts from arm A's pretrained network. Everything
else (rows, cuts, learning rate 2e-4, seeds, widening) is experiment 425's.

- `check`: load experiment 425's stored fine-tuned arm A for one fold and seed, add the zero column,
  forecast with the replica forecast function, and compare with the stored forecast.
- `fit`: fine-tune and forecast (arm `APV`).

```bash
export PYTHONPATH=<as e00425_fit.py shows>:ml-experiments/00405-mdn-explicit-pv-term-learned-capacity-known-answer:ml-experiments/00501-explicit-pv-column-from-the-474-capacity-table
OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00501-*/e00501_fit.py fit --device cuda
```
"""

import argparse
import json
import time
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00397_fit as fit397  # ty: ignore[unresolved-import]
import e00397_prep as base  # ty: ignore[unresolved-import]
import e00405_prep as p405  # ty: ignore[unresolved-import]
import e00407_fit as fit407  # ty: ignore[unresolved-import]
import e00425_fit as fit425  # ty: ignore[unresolved-import]
import e00425_prep as prep425  # ty: ignore[unresolved-import]
import e00501_capacity as capacity
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

ARM: Final[str] = "APV"
PV_SCALE: Final[float] = 3.0
"""Megawatts of the typical signed-MW Primary's capacity; a fixed constant, not fitted."""
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
WORK_DIR: Final[Path] = capacity.WORK_DIR
SEED_DIR: Final[Path] = WORK_DIR / "seed_preds" / ARM
MODEL_DIR: Final[Path] = WORK_DIR / "models" / ARM
US_PER_DAY: Final[int] = prep425.US_PER_DAY


@dataclass
class PvFold(prep425.EnsFold):
    """Experiment 425's fold rows, plus the solar-output column per row (before the scale)."""

    pv_unit: np.ndarray = None  # ty: ignore[invalid-assignment]


_CAPACITY: dict[tuple[str, str], float] = {}


def capacity_table() -> dict[tuple[str, str], float]:
    """Return the out-of-fold AC capacity in MW by (fold, site), read once and checked complete."""
    if not _CAPACITY:
        frame = pl.concat(
            [pl.read_parquet(p) for p in sorted(WORK_DIR.glob("capacity_by_fold_*.parquet"))]
        )
        assert frame.height == len(frame.select("fold", "site").unique()), "duplicate capacity rows"
        assert frame.height == len(capacity.FOLDS) * len(capacity.SIGNED_SIX), frame.height
        _CAPACITY.update({(r["fold"], r["site"]): r["ac_mw"] for r in frame.iter_rows(named=True)})
    return _CAPACITY


def pv_column(*, site: str, month: str, unit_shape: np.ndarray) -> np.ndarray:
    """Return capacity times the unit shape, divided by the scale; zero outside the signed six."""
    if site not in capacity.SIGNED_SIX:
        return np.zeros(len(unit_shape), dtype=np.float32)
    mw = capacity_table()[(month, site)]
    return (mw * unit_shape / PV_SCALE).astype(np.float32)


def build_pv_fold(*, refit: datetime, month: str) -> PvFold:
    """Return experiment 425's fold with the solar-output column added (row order unchanged)."""
    _, mask, layout = base.feature_names(scheme="summary")
    flagged = fit407.flagged_times()
    dropped = prep425.DroppedRows()
    parts = []
    for row in base.primaries().iter_rows(named=True):
        z = np.load(base.RAW_DIR / "summary" / f"{row['site']}.npz")
        keep = (
            (z["valid"] <= prep425._us(t=refit - prep425.regimes.REFIT_CUTOFF_AGE))
            & (z["origin"] <= prep425._us(t=refit))
            & (z["valid"] > prep425._us(t=refit - timedelta(days=base.WINDOW_DAYS)))
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
        shape = np.load(p405.EXTRAS_DIR / f"{row['site']}.npz")["e"][keep][:, 0]
        parts.append(
            {k: z[k][keep] for k in ("x", "y", "origin", "valid", "lead")}
            | {
                "primary": np.full(int(keep.sum()), row["primary"], dtype=np.int64),
                "pv": pv_column(site=row["site"], month=month, unit_shape=shape),
            }
        )
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    del parts
    origin_day = ((cat["origin"] - prep425._us(t=base.FIRST_RUN)) // US_PER_DAY).astype(np.int64)
    held, purged = base.held_out_weeks(origin_day=origin_day)
    train = ~held & ~purged
    standardiser = base.Standardiser.fit(x=cat["x"][train], mask=mask)
    age_days = (prep425._us(t=refit) - cat["valid"]) / US_PER_DAY
    weight = 0.5 ** (age_days / common.SHORT_HALF_LIFE_DAYS)
    n_primaries = len(base.primaries())
    target_mean, target_std = np.zeros(n_primaries), np.ones(n_primaries)
    for p in range(n_primaries):
        sel = train & (cat["primary"] == p)
        if sel.any():
            target_mean[p] = cat["y"][sel].mean()
            target_std[p] = max(float(cat["y"][sel].std()), 1e-3)
    keep = train | held
    return PvFold(
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
        pv_unit=cat["pv"][keep],
    )


def expand_state(*, state: dict, n_inputs: int) -> dict:
    """Return a state dict whose first layer has a zero-weight column for the new input.

    The network concatenates the Primary embedding after the inputs, so the new column is inserted
    at position `n_inputs` (the old input width), before the embedding columns.
    """
    new = dict(state)
    weight = state["net.0.weight"]
    zero = torch.zeros(weight.shape[0], 1, dtype=weight.dtype, device=weight.device)
    new["net.0.weight"] = torch.cat([weight[:, :n_inputs], zero, weight[:, n_inputs:]], dim=1)
    return new


def new_model(*, fold: PvFold) -> torch.nn.Module:
    """Return an untrained pooled plain network with one more input than arm A."""
    return fit397.make_model(
        arm="plain_pooled",
        setting=fit425.SETTING,
        layout=fold.layout,
        n_primaries=len(fold.target_mean),
        n_inputs=fold.x.shape[1] + 1,
    )


class RecordingStandardiser(base.Standardiser):
    """A standardiser that keeps the raw matrix of each call, for the solar-output column."""

    raw: list

    def apply(self, *, x: np.ndarray) -> np.ndarray:  # ty: ignore[invalid-method-override]
        """Record `x`, then standardise it."""
        if not hasattr(self, "raw"):
            self.raw = []
        self.raw.append(x.copy())
        return super().apply(x=x)


def forecast_month_pv(
    *, fitted: fit397.Fitted, fold: PvFold, month: str, device: str
) -> pl.DataFrame:
    """Return the test month's quantiles (fraction of capacity) with the solar-output column."""
    recorder = RecordingStandardiser(
        mean=fold.standardiser.mean, std=fold.standardiser.std, mask=fold.standardiser.mask
    )
    pieces = []
    for origin in base.test_origins(month=month):
        recorder.raw = []
        for (row, keys, x), raw in zip(
            base.run_inputs(origin=origin, scheme="summary", standardiser=recorder),
            recorder.raw,
            strict=True,
        ):
            p = int(row["primary"])
            valid_us = keys["valid_time"].dt.epoch("us").to_numpy()
            shape = p405.extras_from_raw(x=raw, valid_us=valid_us)[:, 0]
            column = pv_column(site=row["site"], month=month, unit_shape=shape)
            q, _, _ = fit397.predict_z_quantiles(
                fitted=fitted,
                x=np.hstack([x, column[:, None]]).astype(np.float32),
                primary=p,
                members=1,
                device=device,
            )
            q = np.sort(q * fold.target_std[p] + fold.target_mean[p], axis=1).astype(np.float32)
            pieces.append(
                keys.drop("lead_hours").hstack(
                    pl.DataFrame(q, schema=list(common.LEVEL_COLUMNS), orient="row")
                )
            )
    return pl.concat(pieces)


def stored_a(*, month: str, seed: int) -> Path:
    """Return the folder of experiment 425's fine-tuned arm A."""
    return fit425.OUT_DIR / "models" / "A" / month / f"s{seed}"


def run_check(*, month: str, seed: int, device: str) -> None:
    """Forecast with experiment 425's stored arm A plus a zero column; compare with its forecast."""
    refit = base.refit_of(month=month)
    fold = build_pv_fold(refit=refit, month=month)
    reference = prep425.build_ens_fold(refit=refit)
    assert fold.x.shape == reference.x.shape
    assert np.array_equal(fold.x, reference.x)
    assert np.array_equal(fold.y, reference.y)
    print(f"fold rows identical to experiment 425's: {len(fold.x):,}")
    model = new_model(fold=fold).to(device)
    state = torch.load(stored_a(month=month, seed=seed) / "pooled.pt", map_location=device)
    model.load_state_dict(expand_state(state=state, n_inputs=fold.x.shape[1]))
    model.eval()
    fitted = fit397.Fitted(
        model=model, setting=fit425.SETTING, arm="plain_pooled", seed=seed, epochs=0, val_nll=0.0
    )
    ours = forecast_month_pv(fitted=fitted, fold=fold, month=month, device=device)
    stored = pl.read_parquet(
        fit425.final_path(arm=fit425.ARMS["A"], cut=True, month=month, seed=seed)
    )
    keys = list(common.KEYS)
    joined = ours.join(stored, on=keys, suffix="_stored")
    print(f"rows ours {ours.height}, stored {stored.height}, joined {joined.height}")
    diff = max(float((joined[c] - joined[f"{c}_stored"]).abs().max()) for c in common.LEVEL_COLUMNS)
    print(f"largest absolute difference over the 11 quantile columns: {diff:.2e}")
    pv = fold.pv_unit
    print(f"solar column: nonzero share {float((pv != 0).mean()):.3f}, max {float(pv.max()):.3f}")


def fine_tune(
    *, fold: PvFold, month: str, seed: int, device: str, t: dict[str, torch.Tensor]
) -> fit397.Fitted:
    """Fine-tune arm A's pretrained network with the new column and return it."""
    torch.manual_seed(seed)
    model = new_model(fold=fold).to(device)
    pretrained = torch.load(
        fit425.pretrained_path(arm=fit425.ARMS["A"], cut=True, month=month, seed=seed),
        map_location=device,
    )
    model.load_state_dict(expand_state(state=pretrained, n_inputs=fold.x.shape[1]))
    setting = replace(fit425.SETTING, lr=fit425.FINE_TUNE_LR)
    epochs, best = fit425.train(model=model, setting=setting, t=t, seed=seed)
    return fit397.Fitted(
        model=model, setting=setting, arm="plain_pooled", seed=seed, epochs=epochs, val_nll=best
    )


def run_fit(*, months: list[str], seeds: list[int], device: str) -> None:
    """Fine-tune the arm on each fold and seed and store forecasts, networks, and logs."""
    for month in months:
        todo = [s for s in seeds if not (SEED_DIR / f"{month}_s{s}.parquet").exists()]
        if not todo:
            continue
        refit = base.refit_of(month=month)
        fold = build_pv_fold(refit=refit, month=month)
        print(f"fold {month}: {len(fold.x):,} rows; dropped {fold.dropped.as_dict()}", flush=True)
        t = fit425.tensors(
            x=np.hstack([fold.x, fold.pv_unit[:, None]]).astype(np.float32),
            y=fold.y,
            primary=fold.primary,
            weight=fold.weight,
            is_val=fold.is_val,
            fold=fold,
            device=device,
        )
        for seed in todo:
            started = time.monotonic()
            fitted = fine_tune(fold=fold, month=month, seed=seed, device=device, t=t)
            target = MODEL_DIR / month / f"s{seed}"
            target.mkdir(parents=True, exist_ok=True)
            torch.save(fitted.model.state_dict(), target / "pooled.pt")
            (target / "meta.json").write_text(
                json.dumps(
                    {
                        "epochs": fitted.epochs,
                        "val_nll": fitted.val_nll,
                        "pv_scale": PV_SCALE,
                        "dropped": fold.dropped.as_dict(),
                        "capacity_mw": {
                            s: capacity_table()[(month, s)] for s in capacity.SIGNED_SIX
                        },
                    }
                )
            )
            frame = forecast_month_pv(fitted=fitted, fold=fold, month=month, device=device)
            common.atomic_write(frame=frame, target=SEED_DIR / f"{month}_s{seed}.parquet")
            print(
                f"{ARM} {month} seed {seed}: {fitted.epochs} epochs, val NLL {fitted.val_nll:.4f}, "
                f"{time.monotonic() - started:.0f} s",
                flush=True,
            )
        del t
        torch.cuda.empty_cache()


def main() -> None:
    """Run the check or the fit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "fit"))
    parser.add_argument("--months", nargs="+", default=list(capacity.FOLDS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "check":
        run_check(month=args.months[0], seed=args.seeds[0], device=args.device)
    else:
        run_fit(months=args.months, seeds=args.seeds, device=args.device)


if __name__ == "__main__":
    main()
