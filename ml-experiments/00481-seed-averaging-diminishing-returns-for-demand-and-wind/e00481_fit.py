# ruff: noqa: E501
r"""Fit the extra seeds of experiment 481: arm A (425), wind `member_plain` (429), and S1 (454).

Arm A and wind seeds go to this experiment's own data folder; S1 seed 5 goes to experiment 404's
model folder, as experiment 454 did, because experiment 404's scorer reads them there. Every
writer skips an output that already exists, so no stored file is overwritten.

```bash
E=$PWD/ml-experiments
export PYTHONPATH=<the 425 PYTHONPATH of e00425_fit.py>:$E/00481-seed-averaging-diminishing-returns-for-demand-and-wind
DATA_PATH_INTERNAL=... OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00481-*/e00481_fit.py demand --seeds 5 6 7 8 9
```
Wind needs the 429 `PYTHONPATH` (`00429/env.sh`) instead; S1 needs 454's (`e00404_fit.py`).
"""

import sys
from pathlib import Path
from typing import Final

from studies.sources import ML_EXPERIMENTS_DATA_DIR  # ty: ignore[unresolved-import]

OWN_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00481-seed-averaging-diminishing-returns-for-demand-and-wind"
MONTHS: Final[list[str]] = ["2025-10", "2025-12", "2026-01", "2026-03", "2026-06"]


def demand(*, seeds: list[int]) -> None:
    """Fit arm A of experiment 425 for the five months, writing under this experiment's folder."""
    import e00425_fit as fit  # noqa: PLC0415  # ty: ignore[unresolved-import]

    fit.OUT_DIR = OWN_DIR / "work"
    fit.run_arms(months=MONTHS, arms=[fit.ARMS["A"]], seeds=seeds, cut=True, device="cuda")


def wind(*, seeds: list[int]) -> None:
    """Fit `member_plain` of experiment 429, writing under this experiment's folder."""
    import e00429_run as run  # noqa: PLC0415  # ty: ignore[unresolved-import]

    assert min(seeds) >= 5, "seeds 0 to 4 are stored in experiment 429's folder"

    run.PARTS_DIR = OWN_DIR / "wind" / "parts"
    run.MODELS_DIR = OWN_DIR / "wind" / "models"
    sys.argv = [sys.argv[0], "member_plain", "--seeds", *map(str, seeds)]
    run.main()


def s1(*, seeds: list[int]) -> None:
    """Fit S1 of experiment 404 for extra seeds (experiment 454's mapping of training worlds)."""
    import e00454_fit as fit  # noqa: PLC0415  # ty: ignore[unresolved-import]

    assert set(seeds) <= {5}, "experiment 404 stores the cross-fitted lagless tables for seeds 0 to 5 only"

    sys.argv = [sys.argv[0], "--arms", "S1", "--seeds", *map(str, seeds)]
    fit.main()


if __name__ == "__main__":
    which = sys.argv[1]
    at = sys.argv.index("--seeds")
    chosen = [int(s) for s in sys.argv[at + 1 :]]
    {"demand": demand, "wind": wind, "s1": s1}[which](seeds=chosen)
