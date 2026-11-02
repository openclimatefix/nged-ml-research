# ruff: noqa: E501
r"""Compute experiment 404's S1 network heads for fit seed 5, the seed experiment 454 never scored.

Experiment 454 cached the S1 mixture heads of fit seeds 0 to 4 under its own `work/heads/S/` (read
here, never written). This script builds the seed 5 head of each of the five months the same way
(`e00454_build.lagged_head`) and writes it under this experiment's `work/s1_heads/`, with the row
keys, so the scorer can average S1 over seeds 0 to 5.

```bash
E=$PWD/ml-experiments
export PYTHONPATH=$E/00087-bg2-best-guess-assembled-with-drop-one-ablations:$E/00048-weekly-retraining-confirmation-and-extension-demand-generators:$E/00404-mdn-lagless-and-calibration-module-trust-weights:$E/00454-lag-trust-mixture-and-ensemble-constructs-for-lagged-and-lagless-heads:$E/00435-stage-0-recalibration-table-for-untrusted-inputs-and-tails:$E/00397-plain-and-attention-mixture-density-network-for-primary-demand
DATA_PATH_INTERNAL=... OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00481-*/e00481_s1_heads.py
```
"""

import time
from pathlib import Path
from typing import Final

import e00454_build as build  # ty: ignore[unresolved-import]
import numpy as np
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OWN_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00481-seed-averaging-diminishing-returns-for-demand-and-wind"
)
HEAD_DIR: Final[Path] = OWN_DIR / "work" / "s1_heads"
NEW_SEED: Final[int] = 5


def new_head_path(*, month: str) -> Path:
    """Return where the seed 5 head of a month is stored."""
    return HEAD_DIR / f"{month}_s{NEW_SEED}.pt"


def main() -> None:
    """Build the missing seed 5 heads and check their keys against seed 0's stored head."""
    torch.set_num_threads(1)
    HEAD_DIR.mkdir(parents=True, exist_ok=True)
    for month in build.MONTHS:
        path = new_head_path(month=month)
        if path.exists():
            continue
        started = time.time()
        head, keys = build.lagged_head(arm="S1", month=month, seed=NEW_SEED)
        stored = torch.load(build.head_path(kind="S", month=month, seed=0), weights_only=False)[
            "keys"
        ]
        for k in build.KEY_COLUMNS:
            assert np.array_equal(stored[k], keys[k]), f"{month}: key {k} differs from seed 0"
        torch.save({"head": head, "keys": keys}, path)
        print(
            f"{month} seed {NEW_SEED}: {len(keys['primary']):,} rows, {time.time() - started:.0f} s",
            flush=True,
        )


if __name__ == "__main__":
    main()
