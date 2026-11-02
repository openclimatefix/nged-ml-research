r"""Fit the Christmas weight alone (WO) and the ERA5 replay alone (RO) from experiment 425's arm A.

Experiment 485's H2 bundled a window-row weight of 5 with a 25% ERA5 replay. This script reuses
experiment 485's loop (`e00485_fit.run`, `train_ext`, `pretrain_rows`) unchanged and defines the
two components as separate arms. Every arm starts from experiment 425's stored pretrained arm A
weights and fine-tunes at the same learning rate, with the same widened first layer (three extra
zero columns left at zero), so the network has H2's shape. Outputs go to this experiment's folder.

- `WO`: window rows weighted 5, no replay;
- `RO`: 25% replay of ERA5 and CAMS window rows of every earlier Christmas, no weight.

```bash
source .xmasenv.sh
systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00497-*/e00497_fit.py --arms WO RO
```
"""

import argparse
from typing import Final

import e00485_fit as fit485  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_DIR_NAME: Final[str] = "00497-christmas-weight-only-and-replay-only-arms"
WORK_DIR = ML_EXPERIMENTS_DATA_DIR / EXPERIMENT_DIR_NAME
"""This experiment's private data folder."""
OUT_DIR = WORK_DIR / "work"
ARMS: Final[dict[str, fit485.Arm]] = {
    "WO": fit485.Arm("WO", weight=True),
    "RO": fit485.Arm("RO", replay=True),
}


def main() -> None:
    """Fit the named arms on the named folds and seeds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", nargs="+", default=list(fit485.MONTHS5))
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(fit485.SEEDS3))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true", help="one epoch, outputs under work/smoke")
    args = parser.parse_args()
    fit485.OUT_DIR = OUT_DIR / "smoke" if args.smoke else OUT_DIR
    if args.smoke:
        fit485.fit397.MAX_EPOCHS = 1
    fit485.run(
        months=args.months,
        arms=[ARMS[a] for a in args.arms],
        seeds=args.seeds,
        device=args.device,
    )


if __name__ == "__main__":
    main()
