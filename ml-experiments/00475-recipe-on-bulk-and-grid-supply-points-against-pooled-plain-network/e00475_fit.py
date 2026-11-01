"""Fit experiment 425's arm A on the 16 Primaries and 4 nodes, and forecast the nodes (475).

One pooled plain network (experiment 397's setting `g09`) with a series embedding, pretrained on
ERA5-and-CAMS rows of the 20 series and fine-tuned on ensemble-summary rows, as experiment 425's
arm A. Per fold (a test month) and seed: one pretraining, one fine-tuning, and the forecast of the
four nodes for the month, in the layout of experiment 415's `seed_preds`.

```bash
export DATA_PATH_INTERNAL=<path to the private data folder>
PYTHONPATH=$(ls -d ml-experiments/00{087,381,397,407,415,418,425,475}-* | paste -sd:) \
OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00475-*/e00475_fit.py --months 2026-06 --seeds 0 \
    --time-only --device cuda
```
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00397_fit as fit397  # ty: ignore[unresolved-import]
import e00397_prep as base  # ty: ignore[unresolved-import]
import e00425_fit as fit425  # ty: ignore[unresolved-import]
import e00425_prep as prep425  # ty: ignore[unresolved-import]
import e00475_prep as prep  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

ARM: Final[fit425.Arm] = fit425.ARMS["A"]
OUT_DIR: Final[Path] = prep.WORK_DIR
fit425.OUT_DIR = OUT_DIR  # the pretrained, model, and forecast paths of 425's helpers


def forecast_nodes(
    *, fitted: fit397.Fitted, fold: prep425.EnsFold, month: str, device: str
) -> pl.DataFrame:
    """Return the test month's quantiles (fraction of capacity) for the four nodes."""
    pieces = []
    for block in prep.node_test_rows(month=month, fold=fold):
        p = int(block["row"]["primary"])
        q, _, _ = fit397.predict_z_quantiles(
            fitted=fitted, x=block["x"], primary=p, members=1, device=device
        )
        q = np.sort(q * fold.target_std[p] + fold.target_mean[p], axis=1).astype(np.float32)
        keys = pl.DataFrame(
            {
                "time_series_id": pl.Series(
                    [block["row"]["time_series_id"]] * len(q), dtype=pl.Int32
                ),
                "power_fcst_init_time": pl.Series(block["origin"], dtype=pl.Int64).cast(
                    pl.Datetime("us", "UTC")
                ),
                "valid_time": pl.Series(block["valid"], dtype=pl.Int64).cast(
                    pl.Datetime("us", "UTC")
                ),
                "actual": block["y"],
            }
        )
        pieces.append(keys.hstack(pl.DataFrame(q, schema=list(common.LEVEL_COLUMNS), orient="row")))
    return pl.concat(pieces)


def pretrain(
    *, fold: prep425.EnsFold, refit: datetime, month: str, seeds: list[int], device: str
) -> None:
    """Pretrain and store one network per seed on the 20 series' ERA5-and-CAMS rows."""
    todo = [
        s
        for s in seeds
        if not fit425.pretrained_path(arm=ARM, cut=True, month=month, seed=s).exists()
        and not fit425.final_path(arm=ARM, cut=True, month=month, seed=s).exists()
    ]
    if not todo:
        return
    pre, dropped = prep.fold_pretrain(fold=fold, refit=refit)
    print(
        f"  pretraining rows {len(pre.x):,} ({int(pre.is_val.sum()):,} held out); "
        f"dropped {dropped.as_dict()}",
        flush=True,
    )
    t = fit425.tensors(
        x=pre.x,
        y=pre.y,
        primary=pre.primary,
        weight=np.ones(len(pre.y), dtype=np.float32),
        is_val=pre.is_val,
        fold=fold,
        device=device,
    )
    del pre
    for seed in todo:
        started = time.monotonic()
        torch.manual_seed(seed)
        model = fit425.new_model(arm=ARM, fold=fold).to(device)
        epochs, best = fit425.train(model=model, setting=fit425.SETTING, t=t, seed=seed)
        fit425.atomic_save(
            state=model.state_dict(),
            target=fit425.pretrained_path(arm=ARM, cut=True, month=month, seed=seed),
        )
        print(
            f"  pretrain {month} seed {seed}: {epochs} epochs, val NLL {best:.4f}, "
            f"{time.monotonic() - started:.0f} s",
            flush=True,
        )
    del t
    torch.cuda.empty_cache()


def run(*, months: list[str], seeds: list[int], device: str, time_only: bool) -> None:
    """Pretrain, fine-tune, and forecast the nodes for each fold and seed."""
    for month in months:
        todo = [
            s
            for s in seeds
            if time_only or not fit425.final_path(arm=ARM, cut=True, month=month, seed=s).exists()
        ]
        if not todo:
            continue
        refit = base.refit_of(month=month)
        started = time.monotonic()
        fold = prep.build_ens_fold(refit=refit)
        print(
            f"fold {month}: {len(fold.x):,} fine-tuning rows, {fold.x.shape[1]} inputs; dropped "
            f"{fold.dropped.as_dict()}; built in {time.monotonic() - started:.0f} s",
            flush=True,
        )
        log_path = OUT_DIR / "logs" / f"{month}.json"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            json.dumps({"month": month, "rows": len(fold.x), "dropped": fold.dropped.as_dict()})
        )
        pretrain(fold=fold, refit=refit, month=month, seeds=todo, device=device)
        t = fit425.tensors(
            x=fold.x,
            y=fold.y,
            primary=fold.primary,
            weight=fold.weight,
            is_val=fold.is_val,
            fold=fold,
            device=device,
        )
        for seed in todo:
            started = time.monotonic()
            fitted = fit425.fine_tune(
                arm=ARM, fold=fold, month=month, seed=seed, cut=True, device=device, t=t
            )
            if time_only:
                print(
                    f"TIMED {month} seed {seed}: fine-tune {time.monotonic() - started:.0f} s, "
                    f"{fitted.epochs} epochs, peak GPU "
                    f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB",
                    flush=True,
                )
                continue
            fit397.save_fits(
                fits={None: fitted},
                fold=fold,
                target=OUT_DIR / "models" / ARM.name / month / f"s{seed}",
            )
            frame = forecast_nodes(fitted=fitted, fold=fold, month=month, device=device)
            common.atomic_write(
                frame=frame, target=fit425.final_path(arm=ARM, cut=True, month=month, seed=seed)
            )
            print(
                f"{month} seed {seed}: {fitted.epochs} epochs, "
                f"{time.monotonic() - started:.0f} s, {frame.height:,} rows",
                flush=True,
            )
        del t
        torch.cuda.empty_cache()


def main() -> None:
    """Run the requested folds and seeds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", nargs="+", default=list(prep.SCREEN_MONTHS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--time-only",
        action="store_true",
        help="fit and print the time; writes pretrained weights and the log only",
    )
    args = parser.parse_args()
    run(months=args.months, seeds=args.seeds, device=args.device, time_only=args.time_only)


if __name__ == "__main__":
    main()
