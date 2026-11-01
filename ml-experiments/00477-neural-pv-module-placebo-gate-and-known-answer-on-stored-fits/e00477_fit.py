# ruff: noqa: E501
"""Fit the new arms of experiment 477: the known-answer injections and the lag-masked PV-split.

Experiment 416's `fit_one` (which registers `A0-doy`, `H1`, and `A0-noirr-doy`) and experiment 409's
`forecast_month` are reused unchanged. `C3-nolag` is `C3` with the three power-lag values and their
missing flags held at a constant in training and at inference.

Stages:

- `kat`: fit `C3`, `A0-doy`, and `C3-nolag` on the 2026-06 fold under the injected conditions of
  `e00477_inject`. For each fit the median forecast on the last 60 days of the fold's own daylight
  rows is stored (in MW), so the score script can read out an equivalent capacity from any arm.
- `screen`: fit `C3-nolag` on the five fold months of experiment 449 and store its forecasts.

```bash
. <the environment script that sets DATA_PATH_INTERNAL and PYTHONPATH>
systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00477-*/e00477_fit.py kat --device cuda
```
"""

import argparse
import json
import time
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00397_models as base_models  # ty: ignore[unresolved-import]
import e00397_prep as base  # ty: ignore[unresolved-import]
import e00405_fit as f5  # ty: ignore[unresolved-import]
import e00405_models as m5  # ty: ignore[unresolved-import]
import e00405_prep as p5  # ty: ignore[unresolved-import]
import e00405_pv as pv  # ty: ignore[unresolved-import]
import e00409_fit as f9  # ty: ignore[unresolved-import]
import e00409_prep as p9  # ty: ignore[unresolved-import]
import e00416_fit as f16  # ty: ignore[unresolved-import]
import e00477_inject as inj  # ty: ignore[unresolved-import]
import numpy as np
import torch  # ty: ignore[unresolved-import]

p9.INJ_DIR = inj.INJ_DIR
m5.SPECS["C3-nolag"] = m5.SPECS["C3"]
WORK: Final = inj.WORK
KAT_REFIT: Final[str] = "2026-06"
KAT_SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
PLAN: Final[dict[str, tuple[str, ...]]] = {
    "C3": inj.CONDITIONS,
    "A0-doy": ("control", "placebo_C_mw4", "profile_C_mw4"),
    "C3-nolag": ("control", "held_C_mw4"),
}
"""Arm -> injected conditions: 7 x 3 + 3 x 3 + 2 x 3 = 36 fits."""
READOUT_DAYS: Final[int] = 60
SEED_DIR: Final = WORK / "seed_preds"
LAG_RAW_COLUMNS: Final[tuple[str, ...]] = tuple(
    f"lag_{tag}{suffix}" for tag in ("origin", "day", "week") for suffix in ("", "_missing")
)


def lag_positions(*, keep: np.ndarray) -> np.ndarray:
    """Return the positions, among a fold's kept columns, of the lag values and missing flags."""
    wanted = {p5.col(name=n) for n in LAG_RAW_COLUMNS}
    pos = np.array([j for j, c in enumerate(keep.tolist()) if c in wanted], dtype=np.int64)
    assert len(pos) == len(LAG_RAW_COLUMNS)
    return pos


def mask_lags(*, x: np.ndarray, keep: np.ndarray) -> None:
    """Hold the lag values and missing flags at a constant (0), in place."""
    x[:, lag_positions(keep=keep)] = 0.0


def masked_run_inputs(
    original: Callable[..., list],
) -> Callable[..., list]:
    """Return `run_inputs` with the lag columns held constant."""

    def run_inputs(*, origin: datetime, fold: p9.Fold) -> list:
        out = original(origin=origin, fold=fold)
        for _row, _keys, x, _extras in out:
            mask_lags(x=x, keep=fold.keep)
        return out

    return run_inputs


def build_masked_fold(
    *, refit: datetime, irradiance_free: bool, injection: str | None, lag_masked: bool
) -> p9.Fold:
    """Return a fold with optional injection and optionally masked lags."""
    fold = p9.build_fold(
        refit=refit, irradiance_free=irradiance_free, injection=injection, use_doy=True
    )
    if lag_masked:
        mask_lags(x=fold.x, keep=fold.keep)
    return fold


def readout(*, model: m5.PvMdn, fold: p9.Fold, refit: datetime, device: str) -> dict:
    """Return the median forecast (MW) on the last days' daylight rows of the signed Primaries."""
    table = base.primaries()
    eff = np.zeros(len(table))
    signed = np.zeros(len(table), dtype=bool)
    for r in table.iter_rows(named=True):
        eff[int(r["primary"])] = float(r["effective_capacity_mw"])
        signed[int(r["primary"])] = r["site"] in pv.SIGNED_SITES
    since = p5._us(t=refit - timedelta(days=READOUT_DAYS))
    sel = np.nonzero((fold.valid >= since) & (fold.extras[:, 0] > 0.01) & signed[fold.primary])[0]
    cell = fold.as_of_year_cell
    median = np.empty(len(sel))
    with torch.no_grad():
        for start in range(0, len(sel), f5.SCORE_CHUNK_ROWS):
            ix = sel[start : start + f5.SCORE_CHUNK_ROWS]
            xb = torch.from_numpy(fold.x[ix]).to(device)
            eb = torch.from_numpy(fold.extras[ix]).to(device)
            pr = fold.primary[ix]
            pb = torch.from_numpy(pr).to(device)
            cb = torch.full((len(ix),), cell, dtype=torch.int64, device=device)
            log_w, mean, std = model.parts(xb, pb, eb, cb)
            q = (
                base_models.mixture_quantiles(
                    log_w=log_w.double(), mean=mean.double(), std=std.double()
                )
                .cpu()
                .numpy()
            )
            q50 = q[:, common.LEVEL_COLUMNS.index("q50")]
            median[start : start + len(ix)] = (
                q50 * fold.target_std[pr] + fold.target_mean[pr]
            ) * eff[pr]
    return {
        "median_mw": median.astype(np.float32),
        "unit_shape": fold.extras[sel, 0].astype(np.float32),
        "primary": fold.primary[sel],
        "valid": fold.valid[sel],
        "lead": fold.lead[sel],
        "is_val": fold.is_val[sel],
    }


def run_kat(*, device: str, arms: list[str], conditions: list[str] | None) -> None:
    """Fit each arm under each injected condition on the 2026-06 fold; store the read-outs."""
    refit = base.refit_of(month=KAT_REFIT)
    for arm in arms:
        spec = m5.SPECS[arm]
        for cond in conditions or PLAN[arm]:
            fold = None
            for seed in KAT_SEEDS:
                target = WORK / "kat" / arm / f"{cond}_s{seed}.json"
                if target.exists():
                    continue
                if fold is None:
                    fold = build_masked_fold(
                        refit=refit,
                        irradiance_free=spec.irradiance_free,
                        injection=cond,
                        lag_masked=arm == "C3-nolag",
                    )
                started = time.monotonic()
                model, epochs, val = f16.fit_one(
                    arm=arm, fold=fold, refit=refit, seed=seed, device=device
                )
                cap = f5.capacity_table(model=model, table=base.primaries())
                out = readout(model=model, fold=fold, refit=refit, device=device)
                target.parent.mkdir(parents=True, exist_ok=True)
                np.savez(target.with_suffix(".npz"), **out)
                target.write_text(
                    json.dumps(
                        {
                            "capacity_mw": cap if spec.solar else None,
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


def run_screen(*, device: str, months: list[str], seeds: list[int]) -> None:
    """Fit `C3-nolag` on each screen month and seed; store forecasts, solar output, and networks."""
    original = f9.run_inputs
    f9.run_inputs = masked_run_inputs(original)  # ty: ignore[invalid-assignment]
    try:
        for month in months:
            refit = base.refit_of(month=month)
            fold = None
            for seed in seeds:
                target = SEED_DIR / "C3-nolag" / f"{month}_s{seed}.parquet"
                if target.exists():
                    continue
                if fold is None:
                    fold = build_masked_fold(
                        refit=refit, irradiance_free=True, injection=None, lag_masked=True
                    )
                started = time.monotonic()
                model, epochs, val = f16.fit_one(
                    arm="C3-nolag", fold=fold, refit=refit, seed=seed, device=device
                )
                frame, solar = f9.forecast_month(
                    model=model, arm="C3-nolag", fold=fold, month=month, device=device
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), target.with_suffix(".pt"))
                if solar.height:
                    solar.write_parquet(target.with_suffix(".solar.parquet"))
                target.with_suffix(".capacity.json").write_text(
                    json.dumps(f5.capacity_table(model=model, table=base.primaries()))
                )
                common.atomic_write(frame=frame, target=target)
                print(
                    f"C3-nolag {month} seed {seed}: {epochs} epochs, val {val:.4f}, {time.monotonic() - started:.0f} s",
                    flush=True,
                )
            del fold
    finally:
        f9.run_inputs = original


def main() -> None:
    """Run a stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("kat", "screen"))
    parser.add_argument("--arms", nargs="+", default=None)
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--months", nargs="+", default=list(base.SCREEN_MONTHS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(KAT_SEEDS))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.stage == "kat":
        run_kat(device=args.device, arms=args.arms or list(PLAN), conditions=args.conditions)
    else:
        run_screen(device=args.device, months=args.months, seeds=args.seeds)


if __name__ == "__main__":
    main()
