# ruff: noqa: E501
"""Fit and forecast the four arms of experiment 416 on the five screen months.

Experiment 409's `fit_one` and `forecast_month` are reused unchanged. Two arms are new: `A0-doy` is
experiment 405's `A0` (free irradiance input, no solar term) with the day-of-year inputs, and `H1`
is `C3`'s solar module with the irradiance and solar-geometry inputs kept. `C3` and `A0-noirr-doy`
come from experiment 409's specs. The forecasts of `C3` and `A0-noirr-doy` for seeds 0 to 2 are
experiment 409's stored files, which the score script reads; this script fits only the missing
seeds.

```bash
export DATA_PATH_INTERNAL=<the private data folder>
export PYTHONPATH=ml-experiments/00087-...:ml-experiments/00397-...:ml-experiments/00405-...:ml-experiments/00409-...
OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00416-*/e00416_fit.py screen --device cuda
```
"""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00397_prep as base  # ty: ignore[unresolved-import]
import e00405_fit as f5  # ty: ignore[unresolved-import]
import e00405_models as m5  # ty: ignore[unresolved-import]
import e00409_fit as f9  # ty: ignore[unresolved-import]
import e00409_prep as prep  # ty: ignore[unresolved-import]
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00416-mdn-explicit-solar-term-better-powered-follow-up" / "work"
)
SEED_DIR: Final[Path] = WORK / "seed_preds"
SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
REUSED_SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
"""Seeds of `C3` and `A0-noirr-doy` that experiment 409 already fitted with this recipe."""
ARMS: Final[tuple[str, ...]] = ("A0-doy", "C3", "H1", "A0-noirr-doy")
FROM_409: Final[tuple[str, ...]] = ("C3", "A0-noirr-doy")

m5.SPECS["A0-doy"] = m5.SPECS["A0"]
m5.SPECS["H1"] = replace(m5.SPECS["C3"], irradiance_free=False)
m5.SPECS["A0-noirr-doy"] = m5.SPECS["A0-noirr"]


def seeds_to_fit(*, arm: str) -> tuple[int, ...]:
    """Return the seeds this experiment fits for an arm (experiment 409's seeds are reused)."""
    return tuple(s for s in SEEDS if arm not in FROM_409 or s not in REUSED_SEEDS)


def run_screen(*, device: str, arms: list[str], months: list[str], seeds: list[int] | None) -> None:
    """Fit each arm on each screen month and seed; store forecasts, solar output, and networks."""
    for month in months:
        refit = base.refit_of(month=month)
        folds: dict[bool, prep.Fold] = {}
        for arm in arms:
            irradiance_free = m5.SPECS[arm].irradiance_free
            for seed in seeds or seeds_to_fit(arm=arm):
                target = SEED_DIR / arm / f"{month}_s{seed}.parquet"
                if target.exists():
                    continue
                if irradiance_free not in folds:
                    folds.clear()
                    folds[irradiance_free] = prep.build_fold(
                        refit=refit, irradiance_free=irradiance_free, use_doy=True
                    )
                fold = folds[irradiance_free]
                started = time.monotonic()
                model, epochs, val = f9.fit_one(
                    arm=arm, fold=fold, refit=refit, seed=seed, device=device
                )
                frame, solar = f9.forecast_month(
                    model=model, arm=arm, fold=fold, month=month, device=device
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
                    f"{arm} {month} seed {seed}: {epochs} epochs, val {val:.4f}, "
                    f"{fold.x.shape[1]} inputs, {time.monotonic() - started:.0f} s",
                    flush=True,
                )


def main() -> None:
    """Run the screen stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("screen",))
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--months", nargs="+", default=list(base.SCREEN_MONTHS))
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run_screen(device=args.device, arms=args.arms, months=args.months, seeds=args.seeds)


if __name__ == "__main__":
    main()
