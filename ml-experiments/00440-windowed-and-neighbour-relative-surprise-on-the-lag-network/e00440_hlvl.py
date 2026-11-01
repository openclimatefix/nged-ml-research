"""Fit or forecast experiment 428's H-lvl arm into this experiment's folders.

H-lvl is experiment 428's S1 lag network plus one learned level offset per origin that fades with
the lead, and a four-way soft regime vector (`HLevelMdn` in `e00428_history`). Its code, rows,
worlds, and training loop are experiment 428's, unchanged; this module only redirects the output
folders, so 428's folders are read and never written. The seeds 0 to 2 lagless tables are
experiment 404's, as for S1 in experiment 428.

```bash
export DATA_PATH_INTERNAL=<path to the private data folder>
source <paths>; PYTHONPATH=$PP:ml-experiments/00440-* OMP_NUM_THREADS=1 \
  systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
  uv run --no-sync python ml-experiments/00440-*/e00440_hlvl.py fit --arms H-lvl
  uv run --no-sync python ml-experiments/00440-*/e00440_hlvl.py predict --arms H-lvl
```
"""

import sys

import e00428_fit as fit  # ty: ignore[unresolved-import]
import e00428_predict as pr  # ty: ignore[unresolved-import]
import e00440_run as run  # ty: ignore[unresolved-import]


def main() -> None:
    """Redirect 428's output folders to this experiment's and run 428's fit or predict command."""
    fit.MODELS_DIR = run.WORK_DIR / "models"
    pr.PRED_DIR = run.WORK_DIR / "predictions"
    command = sys.argv.pop(1)
    {"fit": fit.main, "predict": pr.main}[command]()


if __name__ == "__main__":
    main()
