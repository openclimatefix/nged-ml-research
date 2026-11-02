# ruff: noqa: E501
r"""Run experiment 510's lag-free fine-tuning harness on the single fold 2024-12 (experiment 511).

Re-points experiment 510's modules at this experiment's own work folder and at the month 2024-12,
then runs one of its commands (`leak`, `pretrain`, `fit`, `assemble`) unchanged.

```bash
source <PYTHONPATH of every folder under ml-experiments/>; export DATA_PATH_INTERNAL=<private data folder>
OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00511-*/e00511_run.py leak
```
"""

import sys
from pathlib import Path
from typing import Final

import e00510_common as c  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

MONTH: Final[str] = "2024-12"
WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00511-lagless-weekly-finetune-december-2024-sensitivity" / "work"
)


def main() -> None:
    """Point experiment 510's harness at 2024-12 and this folder, then run its command."""
    c.WORK_DIR = WORK_DIR
    c.MONTHS = (MONTH,)
    import e00510_fit as fit  # ty: ignore[unresolved-import]

    fit.W = WORK_DIR
    sys.argv[1:] = [*sys.argv[1:], "--months", MONTH] if sys.argv[1:2] == ["fit"] else sys.argv[1:]
    fit.main()


if __name__ == "__main__":
    main()
