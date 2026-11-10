# ruff: noqa: E501
r"""Experiment 527, `e00527_fit.py`: kept for the record.

Fit L-1d seeds 2 and 3 (experiment 511's daily-refit lag-free network) into this experiment's
folder.

Reuses experiment 511's and 510's code unchanged; only the output folder is redirected, so no
stored file of 511 or 510 is touched. Seeds 0 and 1 stay in 511's folder.

Commands: `pretrain` (5 folds x seeds 2 and 3, same recipe as experiment 510's `L0PT`), `fit`,
`assemble`.

```bash
PYTHONPATH=<every folder under ml-experiments/> DATA_PATH_INTERNAL=<private data folder> OMP_NUM_THREADS=1 \
  systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
  uv run --no-sync python ml-experiments/00527-*/e00527_fit.py fit
```
"""

import argparse
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00428_data as data  # ty: ignore[unresolved-import]
import e00448_run as r448  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import e00510_fit as f510  # ty: ignore[unresolved-import]
import e00511_common as k  # ty: ignore[unresolved-import]
import e00511_fit as f511  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

W: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00527-lag-gap-head-count-test-k2-s1-vs-four-seed-daily-lag-free"
    / "work"
)
SEEDS: Final[tuple[int, ...]] = (2, 3)
# Redirect every write of experiments 510 and 511 to this experiment's folder.
f510.W = W
f511.W = W


def pretrain(*, device: str) -> None:
    """Pretrain the three-component lag-free network for seeds 2 and 3 on each fold (510's recipe)."""
    clean = c.clean_stack()
    for month in c.MONTHS:
        todo = [s for s in SEEDS if not (W / "pretrained" / "L0PT" / f"{month}_s{s}.pt").exists()]
        if not todo:
            continue
        refit_t = c.month_start(month=month)
        rows = data.build_fold(refit=refit_t, clean=clean)
        pre = r448.pretrain_rows(rows=rows, refit=refit_t, clean=clean)
        prim = pre.primary
        z = (pre.y - rows.target_mean[prim]) / rows.target_std[prim]
        t = {
            "x": torch.from_numpy(pre.x).to(device),
            "z": torch.from_numpy(z.astype(np.float32)).to(device),
            "w": torch.ones(len(pre.y), device=device),
            "p": torch.from_numpy(prim).to(device),
            "v": torch.from_numpy(pre.is_val).to(device),
        }
        for seed in todo:
            started = time.monotonic()
            torch.manual_seed(seed)
            model = c.new_model(
                outlier=False, n_inputs=pre.x.shape[1], n_primaries=len(rows.target_mean)
            ).to(device)
            epochs, best = c.train(
                model=model, x=t["x"], z=t["z"], w=t["w"], p=t["p"], is_val=t["v"], seed=seed,
                lr=c.PRETRAIN_LR, outlier=False, prior=False, device=device,
            )  # fmt: skip
            target = W / "pretrained" / "L0PT" / f"{month}_s{seed}.pt"
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(".tmp")
            torch.save(model.state_dict(), tmp)
            tmp.rename(target)
            print(
                f"  pretrain {month} s{seed}: {epochs} epochs, val NLL {best:.4f}, {time.monotonic() - started:.0f} s ({datetime.now(UTC):%H:%M})",
                flush=True,
            )
        del t, pre, rows
        torch.cuda.empty_cache()


def assemble() -> None:
    """Concatenate each month's daily window forecasts into one table per month and seed."""
    arm = f511.ARMS["L-1d"]
    for month in c.MONTHS:
        cuts = k.cutoffs(month=month, cadence=arm.cadence)
        for seed in SEEDS:
            parts = [
                W
                / "fits"
                / arm.tag
                / f"{month}_d{f511.day_of(month=month, cut=cut)}_s{seed}.parquet"
                for cut in cuts
            ]
            if all(p.exists() or p.with_suffix(".empty").exists() for p in parts):
                frame = pl.concat([pl.read_parquet(p) for p in parts if p.exists()])
                f511.atomic_frame(
                    frame=frame, target=W / "seed_preds" / arm.tag / f"{month}_s{seed}.parquet"
                )
                print("wrote", month, seed, frame.height, flush=True)
            else:
                print("incomplete", month, seed, flush=True)


def main() -> None:
    """Run a command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("pretrain", "fit", "assemble"))
    parser.add_argument("--months", nargs="+", default=list(c.MONTHS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "pretrain":
        pretrain(device=args.device)
    elif args.command == "fit":
        f511.fit_arm(
            arm=f511.ARMS["L-1d"], months=args.months, seeds=list(SEEDS), stress=False,
            limit=args.limit, device=args.device,
        )  # fmt: skip
    else:
        assemble()


if __name__ == "__main__":
    main()
