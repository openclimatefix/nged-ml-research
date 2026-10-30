# ruff: noqa: E501
"""Fit the known-answer arms of experiment 409 (the score stages are added only if the gate holds).

Experiment 405's `train`, `make_model`, and `capacity_table` are reused unchanged. Two arm names are
aliases of experiment 405 specs: `A0-noirr-doy` is `A0-noirr` and `C2-nodoy` is `C2` fitted without
the day-of-year inputs (every other arm has them).

```bash
export DATA_PATH_INTERNAL=/home/jack/dev/nged-substation-forecast/data
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

import e00397_prep as base  # ty: ignore[unresolved-import]
import e00405_fit as f5  # ty: ignore[unresolved-import]
import e00405_models as m5  # ty: ignore[unresolved-import]
import e00409_prep as prep  # ty: ignore[unresolved-import]
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
) -> tuple[object, int, float]:
    """Fit one arm on a fold; return the network, its epochs, and its validation loss."""
    model = f5.make_model(arm=arm, fold=fold, table=base.primaries(), refit=refit, seed=seed)
    epochs, val = f5.train(model=model, arm=arm, fold=fold, seed=seed, device=device)
    return model, epochs, val


def run_kat(*, device: str, arms: list[str], conditions: list[str] | None) -> None:
    """Fit the known-answer arms under each condition; store capacities and the saved network."""
    refit = base.refit_of(month=KAT_REFIT)
    for arm in arms:
        conds, seeds = PLAN[arm]
        for cond in conditions or conds:
            fold = None
            for seed in seeds:
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


def main() -> None:
    """Run a stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("kat",))
    parser.add_argument("--arms", nargs="+", default=list(PLAN))
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run_kat(device=args.device, arms=args.arms, conditions=args.conditions)


if __name__ == "__main__":
    main()
