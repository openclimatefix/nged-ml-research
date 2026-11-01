"""Fit the lag-dropout S1 network (arm S1d) for the five scored months and fit seeds 0 to 2.

This is experiment 404's `fit_arm` (rows, six epochs of Adam on stride-2 rows, injected training
world of seed 100 + the fit seed, cross-fitted lagless table of the fit seed) with the network
replaced by `DropoutS1Mdn` and the surprise inputs switched on for the arm. Networks are saved
under this experiment's own data folder. The first fit's time is printed so it can be reported.

```bash
source /tmp/e460_env.sh  # PYTHONPATH of experiments 397, 87, 48, 404, 435, 460; DATA_PATH_INTERNAL
systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00460-*/e00460_fit.py --months 2026-06 --seeds 0
```
"""

import argparse
from pathlib import Path
from typing import Final

import e00404_fit as fit  # ty: ignore[unresolved-import]
import e00460_models as net_mod  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ARM: Final[str] = "S1d"
WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00460-lag-dropout-network-forecast-disagreement-as-a-trust-and-width-signal"
    / "work"
)
MODEL_DIR: Final[Path] = WORK_DIR / "models"
MONTHS: Final[tuple[str, ...]] = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
_ORIGINAL_BUILD = fit.build_network


def build_network(*, arm: str, n_base: int, n_trust: int) -> object:
    """Return the lag-dropout network for arm S1d and experiment 404's network otherwise."""
    if arm == ARM:
        return net_mod.DropoutS1Mdn(n_base=n_base, n_primaries=16)
    return _ORIGINAL_BUILD(arm=arm, n_base=n_base, n_trust=n_trust)


def patch_fit_module() -> None:
    """Point experiment 404's fit code at this arm, the surprise inputs, and the model folder."""
    fit.build_network = build_network  # ty: ignore[invalid-assignment]
    fit.SURPRISE_ARM = ARM
    fit.MODEL_DIR = MODEL_DIR


def main() -> None:
    """Fit the arm for the given months and seeds, skipping networks that already exist."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", nargs="+", default=list(MONTHS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    patch_fit_module()
    for month in args.months:
        for seed in args.seeds:
            if (MODEL_DIR / ARM / f"{month}_s{seed}.pt").exists():
                continue
            fit.fit_arm(
                arm=ARM, month=month, seed=seed, device=args.device, stride=2, epochs=fit.EPOCHS
            )  # fmt: skip


if __name__ == "__main__":
    main()
