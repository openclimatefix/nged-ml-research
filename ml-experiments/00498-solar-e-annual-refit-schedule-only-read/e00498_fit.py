# ruff: noqa: E501
"""Fit seeds 6 to 9 of `a2_member` for experiment 498: one annual refit and the three later quarterly refits.

The first quarterly refit (2025-09) is the same fit as the annual one (experiment 493's control), so
the quarterly arm takes its first quarter from the annual file. That makes 4 annual fits plus 12
later quarterly fits, 16 in all. Experiment 430's and 493's jobs are called unchanged, with their
output folders redirected to this experiment's own folder so that their stored seeds stay untouched.

    PYTHONPATH=ml-experiments/00310-metered-farm-long-lead-scoring:ml-experiments/00430-unified-forecaster-solar-and-pretraining-ablations:ml-experiments/00493-solar-annual-against-quarterly-refits \
    OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00498-*/e00498_fit.py
"""

import os
from typing import Final

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00310_fit as f310  # noqa: E402  # ty: ignore[unresolved-import]
import e00430_common as common430  # noqa: E402  # ty: ignore[unresolved-import]
import e00430_fit as fit430  # noqa: E402  # ty: ignore[unresolved-import]
import e00493_common as common493  # noqa: E402  # ty: ignore[unresolved-import]
import e00493_fit as fit493  # noqa: E402  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR  # noqa: E402

SLUG: Final[str] = "00498-solar-e-annual-refit-schedule-only-read"
DATA_DIR: Final = ML_EXPERIMENTS_DATA_DIR / SLUG
SEEDS: Final[tuple[int, ...]] = (6, 7, 8, 9)
ARM: Final[str] = "a2_member"


def main() -> None:
    """Fit the annual job and the three later quarterly jobs for each new seed."""
    common493.PRED_DIR = DATA_DIR / "predictions_annual"
    common493.DATA_DIR = DATA_DIR
    common430.PRED_DIR = DATA_DIR / "predictions_quarterly"
    later = f310.REFIT_STARTS[1:]
    for seed in SEEDS:
        print(fit493.run_job(arm=ARM, seed=seed), flush=True)
        for start in later:
            print(
                fit430.run_job(arm=ARM, config=common430.ARMS[ARM], seed=seed, start=start),
                flush=True,
            )


if __name__ == "__main__":
    main()
