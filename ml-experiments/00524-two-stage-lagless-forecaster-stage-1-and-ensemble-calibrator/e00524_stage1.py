# ruff: noqa: E501
r"""Stage 1 of experiment 524: the lead-free, spread-free weather-to-power network.

Same recipe as experiment 510's `L0PT` pretraining (experiment 448's ERA5-and-CAMS rows, pooled
`PlainMdn` of 64 x 2 with 3 components, learning rate 5e-4, early stopping on the held-out weeks),
with the columns `lead`, `lead_sqrt`, and the seven `std_*` removed from the inputs. Rows are those
of `e00448_run.pretrain_rows`, so the ERA5 and CAMS pseudo-runs stop at the month's refit time
minus the cutoff age. The kept column positions are stored beside the weights.

```bash
cd <worktree>; source ml-experiments/00524-*/env.sh
systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00524-*/e00524_stage1.py
```
"""

import argparse
import json
import time
from pathlib import Path
from typing import Final

import e00428_data as data  # ty: ignore[unresolved-import]
import e00448_run as r448  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import numpy as np
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00524-two-stage-lagless-forecaster-stage-1-and-ensemble-calibrator"
    / "work"
)
STAGE1: Final[Path] = WORK_DIR / "stage1"
CROSSFIT: Final[Path] = WORK_DIR / "stage1_crossfit"
BLOCKS: Final[int] = 6
"""Contiguous time blocks of the ERA5 and CAMS history, as in experiment 449's cross-fit."""
PAD_STEPS: Final[int] = 144
"""Three days of half-hours left out on each side of a held-out block."""


def block_edges(*, target_idx: np.ndarray) -> np.ndarray:
    """Return the BLOCKS + 1 edges (grid indices) that split the pretraining valid times equally."""
    return np.linspace(target_idx.min(), target_idx.max() + 1, BLOCKS + 1).astype(np.int64)


def stage1_columns(*, columns: list[str]) -> list[int]:
    """Return the positions of the inputs stage 1 keeps: everything but `lead`, `lead_sqrt`, `std_*`."""
    dropped = {
        i
        for i, name in enumerate(columns)
        if name in ("lead", "lead_sqrt") or name.startswith("std_")
    }
    assert len(dropped) == 9, (
        f"expected 9 dropped columns, found {sorted(columns[i] for i in dropped)}"
    )
    return [i for i in range(len(columns)) if i not in dropped]


def run(*, months: list[str], seeds: list[int], device: str, crossfit: bool = False) -> None:
    """Pretrain stage 1 for each month and seed."""
    clean = c.clean_stack()
    for month in months:
        folder = CROSSFIT if crossfit else STAGE1
        jobs = [
            (s, b)
            for s in seeds
            for b in (range(BLOCKS) if crossfit else [None])
            if not (folder / f"{month}_s{s}{'' if b is None else f'_b{b}'}.pt").exists()
        ]
        if not jobs:
            continue
        refit = c.month_start(month=month)
        rows = data.build_fold(refit=refit, clean=clean)
        keep = stage1_columns(columns=list(rows.columns))
        pre = r448.pretrain_rows(rows=rows, refit=refit, clean=clean)
        latest = (pre.target_idx.max() * c.prep422.STEP_US + c.prep422.GRID_START_US) / 1e6
        assert latest <= refit.timestamp() - 3.5 * 3600, "pretraining rows reach the refit cutoff"
        z = (pre.y - rows.target_mean[pre.primary]) / rows.target_std[pre.primary]
        t = {
            "x": torch.from_numpy(np.ascontiguousarray(pre.x[:, keep])).to(device),
            "z": torch.from_numpy(z.astype(np.float32)).to(device),
            "w": torch.ones(len(pre.y), device=device),
            "p": torch.from_numpy(pre.primary).to(device),
            "v": torch.from_numpy(pre.is_val).to(device),
        }
        print(f"{month}: {len(pre.y):,} rows, {len(keep)} inputs", flush=True)
        edges = block_edges(target_idx=pre.target_idx)
        for seed, block in jobs:
            started = time.monotonic()
            torch.manual_seed(seed)
            model = c.new_model(
                outlier=False, n_inputs=len(keep), n_primaries=len(rows.target_mean)
            ).to(device)
            if block is None:
                use = torch.ones(len(pre.y), dtype=torch.bool, device=device)
            else:
                lo, hi = edges[block] - PAD_STEPS, edges[block + 1] + PAD_STEPS
                tid = torch.from_numpy(pre.target_idx).to(device)
                use = ~((tid >= lo) & (tid < hi))
            epochs, best = c.train(
                model=model, x=t["x"][use], z=t["z"][use], w=t["w"][use], p=t["p"][use],
                is_val=t["v"][use], seed=seed, lr=c.PRETRAIN_LR, outlier=False, prior=False,
                device=device,
            )  # fmt: skip
            suffix = "" if block is None else f"_b{block}"
            folder.mkdir(parents=True, exist_ok=True)
            tmp = folder / f"{month}_s{seed}{suffix}.tmp"
            torch.save(model.state_dict(), tmp)
            tmp.rename(folder / f"{month}_s{seed}{suffix}.pt")
            c.write_json(
                obj={"month": month, "seed": seed, "block": block, "edges": edges.tolist(),
                     "epochs": epochs, "val_nll": best, "keep": keep,
                     "columns": [rows.columns[i] for i in keep], "rows": int(use.sum()),
                     "seconds": round(time.monotonic() - started)},
                path=folder / f"{month}_s{seed}{suffix}.json",
            )  # fmt: skip
            print(
                f"  {month} s{seed}{suffix}: {epochs} epochs, val NLL {best:.4f}, {time.monotonic() - started:.0f} s",
                flush=True,
            )
        del t, pre, rows
        torch.cuda.empty_cache()


def main() -> None:
    """Run the pretrains."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", nargs="+", default=list(c.MONTHS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--crossfit", action="store_true")
    args = parser.parse_args()
    _ = json
    run(months=args.months, seeds=args.seeds, device=args.device, crossfit=args.crossfit)


if __name__ == "__main__":
    main()
