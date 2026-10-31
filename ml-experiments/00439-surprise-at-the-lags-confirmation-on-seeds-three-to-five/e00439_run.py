# ruff: noqa: E501
"""Fit and forecast S1 and R-inj of experiment 428 on seeds 3 to 5 (the S1 confirmation).

This module reuses experiment 428's fit loop, rows, and prediction unchanged and redirects the
modules' output folders to this experiment's own work folder. S1 reads the cross-fitted lagless
table of its own seed. Experiment 404 froze tables for seeds 0 to 2 only, so tables for seeds 3 to 5
are built with the frozen code (`e00404_lagless.py build`, new files `seed=3|4|5.parquet` beside
the frozen ones); with `--lagless-mod3` S1 instead reads the
table of seed `seed % 3`.

```bash
export DATA_PATH_INTERNAL=/home/jack/dev/nged-substation-forecast/data
source <paths>; PYTHONPATH=$PP:ml-experiments/00439-* OMP_NUM_THREADS=1 \
  systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
  uv run --no-sync python ml-experiments/00439-*/e00439_run.py fit --arms S1 R-inj --seeds 3
  uv run --no-sync python ml-experiments/00439-*/e00439_run.py predict --arms S1 R-inj --seeds 3
```
"""

import argparse
from pathlib import Path
from typing import Final

import e00428_data as data  # ty: ignore[unresolved-import]
import e00428_fit as fit  # ty: ignore[unresolved-import]
import e00428_predict as pr  # ty: ignore[unresolved-import]
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00439-surprise-at-the-lags-confirmation-on-seeds-three-to-five"
    / "work"
)
ARMS: Final[tuple[str, ...]] = ("S1", "R-inj")
SEEDS: Final[tuple[int, ...]] = (3, 4, 5)
FROZEN_SEEDS: Final[int] = 3
"""Experiment 404 froze lagless tables for seeds 0 to 2."""


def install(*, lagless_mod3: bool) -> None:
    """Redirect experiment 428's modules to this experiment's folders (and optionally mod-3 tables)."""
    fit.MODELS_DIR = WORK_DIR / "models"
    pr.PRED_DIR = WORK_DIR / "predictions"
    original = fit.surprise_for

    def surprise_for(*, arm: fit.Arm, month: str, seed: int, device: str):  # noqa: ANN202
        return original(
            arm=arm, month=month, seed=seed % FROZEN_SEEDS if lagless_mod3 else seed, device=device
        )

    fit.surprise_for = surprise_for


def main() -> None:
    """Fit or forecast the arms."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fit", "predict"))
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--months", nargs="+", default=list(fit.SCREEN_MONTHS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--lagless-mod3", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    install(lagless_mod3=args.lagless_mod3)
    if args.command == "fit":
        power = fit.power_tensor(device=args.device)
        stack, _ = data.load_stack(include_eval=False)
        cache: dict = {}
        for month in args.months:
            for seed in args.seeds:
                for name in args.arms:
                    print(
                        fit.fit_one(
                            arm_name=name, month=month, seed=seed, power=power, stack=stack,
                            device=args.device, rows_cache=cache,
                        ),
                        flush=True,
                    )  # fmt: skip
        return
    power_np, code = data.load_stack(include_eval=True)
    power = torch.from_numpy(power_np).to(args.device)
    flags = pr.Flags(code=code, real=data.real_event_mask(), power=power_np, device=args.device)
    for month in args.months:
        for seed in args.seeds:
            for name in args.arms:
                for line in pr.predict_month(
                    arm_name=name, month=month, seed=seed, device=args.device, power=power,
                    flags=flags, y_clean=power_np[0],
                ):  # fmt: skip
                    print(line, flush=True)


if __name__ == "__main__":
    main()
