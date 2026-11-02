# ruff: noqa: E501
r"""S2-0 CPU gate of experiment 524: stored L0PT weights scored on ENS rows with no fine-tune.

The stored pretrained networks (experiment 448's `L0PT` files, and experiment 510's own for 2025-12)
are loaded and used to forecast every ENS test row of the month, with no fine-tune. They keep
`lead`, `lead_sqrt`, and the `std_*` columns, so the arm bounds a frozen stage 1 from below. Output
layout is experiment 510's `seed_preds/<arm>/<month>_s<seed>.parquet`, under this experiment's work
folder, so experiment 510's scorer reads it.

```bash
cd <worktree>; source ml-experiments/00524-*/env.sh
systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00524-*/e00524_gate.py --months 2025-12
```
"""

import argparse
import time
from pathlib import Path
from typing import Final

import e00428_data as data  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import e00510_fit as fit510  # ty: ignore[unresolved-import]
import numpy as np
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00524-two-stage-lagless-forecaster-stage-1-and-ensemble-calibrator"
    / "work"
)
ARM: Final[str] = "S2-0"


def run(*, months: list[str], seeds: list[int]) -> None:
    """Forecast every ENS test row of each month with the stored pretrained weights."""
    torch.set_num_threads(2)
    clean = c.clean_stack()
    for month in months:
        todo = [
            s
            for s in seeds
            if not (WORK_DIR / "seed_preds" / ARM / f"{month}_s{s}.parquet").exists()
        ]
        if not todo:
            continue
        rows = data.build_fold(refit=c.month_start(month=month), clean=clean)
        test = data.build_test(month=month, standardiser=rows.standardiser)
        select = np.ones(len(test.primary), dtype=bool)
        for seed in todo:
            started = time.monotonic()
            model = c.new_model(
                outlier=False, n_inputs=test.x.shape[1], n_primaries=len(rows.target_mean)
            )
            c.load_pretrained(
                model=model,
                path=fit510.pretrained_path(v=fit510.VARIANTS["hl90"], month=month, seed=seed),
            )
            model.eval()
            frame = c.forecast_frame(
                model=model,
                outlier=False,
                test=test,
                select=select,
                rows=rows,
                widen=1.0,
                device="cpu",
            )
            assert frame.height == int(select.sum()), (
                "row count differs from the selected test rows"
            )
            target = WORK_DIR / "seed_preds" / ARM / f"{month}_s{seed}.parquet"
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(".tmp")
            frame.write_parquet(tmp)
            tmp.rename(target)
            print(
                f"{month} s{seed}: {frame.height:,} rows, {time.monotonic() - started:.0f} s",
                flush=True,
            )
        del rows, test


def main() -> None:
    """Run the gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", nargs="+", default=list(c.MONTHS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    args = parser.parse_args()
    run(months=args.months, seeds=args.seeds)


if __name__ == "__main__":
    main()
