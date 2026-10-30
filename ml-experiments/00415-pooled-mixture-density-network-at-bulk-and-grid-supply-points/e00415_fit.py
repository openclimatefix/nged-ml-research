"""Fit the pooled plain network at the Primaries and four nodes, and forecast the test months (415).

Arms (`e00415_prep.ARMS`): P0 fits at the 16 Primaries only; N1 adds the 4 nodes and a node-type
input; N2 adds the true neighbour columns; N2rw adds the rewired ones. Every arm is experiment
397's `plain_pooled` network with its chosen setting (`g09`), the `summary` weather scheme, one fit
per fold on the 24 months before the test month, and a seed. A P0 network has no embedding for a
node, so the node's embedding is set to the mean of the 16 Primary embeddings before forecasting.

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\
ml-experiments/00397-plain-and-attention-mixture-density-network-for-primary-demand:\
ml-experiments/00415-pooled-mixture-density-network-at-bulk-and-grid-supply-points:\
ml-experiments/00048-weekly-retraining-confirmation-and-extension-demand-generators \
OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00415-*/e00415_fit.py --arms P0 N1 \
    --months 2026-06 2026-01 --seeds 0 --device cuda
```
"""

import argparse
import time
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00397_fit as fit  # ty: ignore[unresolved-import]
import e00397_models as models  # ty: ignore[unresolved-import]
import e00397_prep as base  # ty: ignore[unresolved-import]
import e00415_prep as prep  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

SETTING_NAME: Final[str] = "g09"
"""The setting experiment 397 chose for the plain networks (width 64, 2 layers, dropout 0.1)."""
OUT_DIR: Final[Path] = prep.WORK_DIR


def forecast_month(
    *, fitted: fit.Fitted, fold: prep.Fold, arm: str, month: str, device: str
) -> pl.DataFrame:
    """Return the test month's quantiles (fraction of capacity) for every series."""
    train = fold.train
    model = fitted.model
    if arm == "P0":
        assert isinstance(model, models.PlainMdn)
        assert model.embedding is not None
        with torch.no_grad():
            weight = model.embedding.weight
            mean = weight[fold.fit_ids].mean(dim=0)
            for node_id in range(weight.shape[0]):
                if node_id not in fold.fit_ids:
                    weight[node_id] = mean
    pieces = []
    for block in prep.test_rows(arm=arm, month=month, standardiser=train.standardiser):
        p = int(block["row"]["primary"])
        q, _, _ = fit.predict_z_quantiles(
            fitted=fitted, x=block["x"], primary=p, members=1, device=device
        )
        q = np.sort(q * train.target_std[p] + train.target_mean[p], axis=1).astype(np.float32)
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


def run(
    *, arms: list[str], months: list[str], seeds: list[int], device: str, time_only: bool
) -> None:
    """Fit each arm at each fold and seed, save the network, and write the forecasts."""
    setting = fit.setting_by_name(name=SETTING_NAME)
    for month in months:
        for arm in arms:
            started = time.monotonic()
            fold = prep.build_fold(arm=arm, refit=base.refit_of(month=month))
            print(
                f"fold {month} arm {arm}: {len(fold.train.x):,} rows, "
                f"{fold.train.x.shape[1]} inputs, built in {time.monotonic() - started:.0f} s",
                flush=True,
            )
            for seed in seeds:
                target = OUT_DIR / "seed_preds" / arm / f"{month}_s{seed}.parquet"
                if target.exists() and not time_only:
                    continue
                started = time.monotonic()
                fits = fit.fit_arm(
                    arm="plain_pooled", setting=setting, fold=fold.train, seed=seed, device=device
                )
                fitted = fits[None]
                fit_seconds = time.monotonic() - started
                if time_only:
                    print(
                        f"TIMED {arm} {month} seed {seed}: fit {fit_seconds:.0f} s, "
                        f"{fitted.epochs} epochs, peak GPU "
                        f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB",
                        flush=True,
                    )
                    continue
                fit.save_fits(
                    fits=fits, fold=fold.train, target=OUT_DIR / "models" / arm / month / f"s{seed}"
                )
                frame = forecast_month(
                    fitted=fitted, fold=fold, arm=arm, month=month, device=device
                )
                common.atomic_write(frame=frame, target=target)
                print(
                    f"{arm} {month} seed {seed}: fit {fit_seconds:.0f} s, total "
                    f"{time.monotonic() - started:.0f} s, {frame.height:,} rows",
                    flush=True,
                )


def main() -> None:
    """Run the requested arms, folds, and seeds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", default=list(prep.ARMS), choices=prep.ARMS)
    parser.add_argument("--months", nargs="+", default=list(prep.SCREEN_MONTHS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--time-only", action="store_true", help="fit and print the time; no output"
    )
    args = parser.parse_args()
    run(
        arms=args.arms,
        months=args.months,
        seeds=args.seeds,
        device=args.device,
        time_only=args.time_only,
    )


if __name__ == "__main__":
    main()
