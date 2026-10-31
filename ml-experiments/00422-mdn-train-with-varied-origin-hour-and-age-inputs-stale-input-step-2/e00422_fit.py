"""Fit the pooled plain network under each training arm of experiment 422.

For each arm, fold (test month), and seed the script builds the arm's training rows
(`e00422_prep.build_train`), fits experiment 397's pooled plain network with its setting and
training loop unchanged, and saves the network, the standardiser, and the target statistics under
this experiment's private folder. Forecasting is `e00422_predict.py`.

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\
ml-experiments/00048-weekly-retraining-confirmation-and-extension-demand-generators:\
ml-experiments/00397-plain-and-attention-mixture-density-network-for-primary-demand:\
ml-experiments/00410-mdn-stale-weather-and-delayed-power-rescore:\
ml-experiments/00422-mdn-train-with-varied-origin-hour-and-age-inputs-stale-input-step-2 \
OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00422-*/e00422_fit.py --arms T0 T1 T2 T3 T4
```
"""

import argparse
import time
from pathlib import Path
from typing import Final

import e00397_fit as fit397  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00422_prep as prep422  # ty: ignore[unresolved-import]
import numpy as np
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00422-mdn-train-with-varied-origin-hour-and-age-inputs-stale-input-step-2"
    / "work"
)
MODELS_DIR: Final[Path] = WORK_DIR / "models"
SETTING_NAME: Final[str] = "g09"
"""The setting stored with experiment 397's pooled plain networks."""
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
FOLD_COMPLETE: Final[str] = "meta.json"


def fit_one(
    *, arm_name: str, month: str, seed: int, clean: dict[str, np.ndarray], device: str
) -> str:
    """Fit and save one arm, fold, and seed; return a progress line."""
    target = MODELS_DIR / arm_name / month / f"s{seed}"
    if (target / FOLD_COMPLETE).exists():
        return f"{arm_name} {month} s{seed}: exists"
    arm = prep422.arm_by_name(name=arm_name)
    started = time.monotonic()
    fold = prep422.build_train(arm=arm, refit=prep.refit_of(month=month), seed=seed, clean=clean)
    built = time.monotonic() - started
    print(
        f"{arm_name} {month} s{seed}: {len(fold.x):,} rows ({int(fold.is_val.sum()):,} held out), "
        f"build {built:.0f} s",
        flush=True,
    )
    fits = fit397.fit_arm(
        arm="plain_pooled",
        setting=fit397.setting_by_name(name=SETTING_NAME),
        fold=fold,
        seed=seed,
        device=device,
    )
    fit397.save_fits(fits=fits, fold=fold, target=target)
    epochs = next(iter(fits.values())).epochs
    return f"{arm_name} {month} s{seed}: {epochs} epochs, total {time.monotonic() - started:.0f} s"


def main() -> None:
    """Fit the requested arms, months, and seeds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", default=[a.name for a in prep422.TRAIN_ARMS])
    parser.add_argument("--months", nargs="+", default=list(prep.SCREEN_MONTHS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    clean = prep422.cleaned_arrays()
    for month in args.months:
        for seed in args.seeds:
            for arm_name in args.arms:
                print(
                    fit_one(
                        arm_name=arm_name, month=month, seed=seed, clean=clean, device=args.device
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
