r"""Fit bundle T of experiment 437 on six winter-weighted folds, seeds 0 to 4 (experiment 442).

The network, the rows, the cuts, and the fold construction are experiment 437's
(`e00437_fit.run_arms`); this module only changes the folds, the seeds, and where the new forecasts
and networks are written. Experiment 437's folder is read and never written: its stored `T` seeds
0 to 2 on four folds are reused, and the new fits are `T` on 2025-11 and 2026-02 with seeds 0 to 4
and `T` on the four older folds with seeds 3 and 4. The control `R` is experiment 425's stored
forecast for every month with seeds 0 to 4, so it is not refitted. `plan` prints the fits that are
missing.

```bash
source <(printf 'export PYTHONPATH=%s\n' "$(cat ml-experiments/00442-*/pythonpath.txt)")
export DATA_PATH_INTERNAL=/home/jack/dev/nged-substation-forecast/data OMP_NUM_THREADS=1
uv run --no-sync python ml-experiments/00442-*/e00442_fit.py plan
systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00442-*/e00442_fit.py fit --device cuda
```
"""

import argparse
from pathlib import Path
from typing import Final

import e00437_fit as fit437  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_DIR_NAME: Final[str] = "00442-thermal-lag-cold-night-test-on-six-winter-folds"
OUT_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / EXPERIMENT_DIR_NAME / "work"
EXPERIMENT_437_DIR: Final[Path] = fit437.OUT_DIR
"""Experiment 437's work folder, read only."""
MONTHS: Final[tuple[str, ...]] = ("2025-10", "2025-11", "2025-12", "2026-01", "2026-02", "2026-03")
"""The six folds: experiment 437's four, plus the two other winter months of the scored year."""
OLD_MONTHS: Final[tuple[str, ...]] = fit437.SCREEN_MONTHS
NEW_MONTHS: Final[tuple[str, ...]] = ("2025-11", "2026-02")
SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
OLD_SEEDS: Final[tuple[int, ...]] = fit437.SEEDS
ARM: Final[str] = "T"
CONTROL: Final[str] = "R"
"""The base network. Experiment 425's `R`; the one line to change if S1 is later adopted."""


def seed_path(*, arm: str, month: str, seed: int) -> Path:
    """Return where a seed's forecast is stored: experiment 437's folder for its three seeds."""
    stored = EXPERIMENT_437_DIR / "seed_preds" / arm / f"{month}_s{seed}.parquet"
    if month in OLD_MONTHS and seed in OLD_SEEDS and stored.exists():
        return stored
    return OUT_DIR / "seed_preds" / arm / f"{month}_s{seed}.parquet"


def missing_fits() -> list[tuple[str, int]]:
    """Return the (month, seed) pairs of `T` with no stored forecast."""
    return [
        (m, s) for m in MONTHS for s in SEEDS if not seed_path(arm=ARM, month=m, seed=s).exists()
    ]


def main() -> None:
    """Print the missing fits, or fit them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "fit"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    todo = missing_fits()
    print(f"{len(todo)} fits missing: {todo}")
    if args.command == "plan":
        return
    # Experiment 437's `run_arms` writes through its module-level OUT_DIR and checks its seed_path.
    original = fit437.OUT_DIR
    fit437.OUT_DIR = OUT_DIR
    try:
        for month in MONTHS:
            seeds = [s for m, s in todo if m == month]
            if seeds:
                fit437.run_arms(months=[month], arms=[ARM], seeds=seeds, device=args.device)
    finally:
        fit437.OUT_DIR = original


if __name__ == "__main__":
    main()
