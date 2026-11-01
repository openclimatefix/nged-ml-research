# ruff: noqa: E501
"""Fit the annual-refit solar networks of experiment 493 on the GPU and store their quantiles.

Each job is one (arm, seed): experiment 430's configuration of the arm, trained once on rows whose
valid time is at least one day before 2025-09-01, predicting the runs of all 12 scored months.
Predictions go to this experiment's own folder, as `annual_<arm>/s<seed>_2025-09.parquet`, written
atomically and skipped when they exist.

    PYTHONPATH=ml-experiments/00310-metered-farm-long-lead-scoring:ml-experiments/00430-unified-forecaster-solar-and-pretraining-ablations \
    OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00493-*/e00493_fit.py [--seeds 0,1,2,3,4,5]
"""

import argparse
import os
import time
from datetime import timedelta
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00310_common as c310  # noqa: E402  # ty: ignore[unresolved-import]
import e00310_fit as f310  # noqa: E402  # ty: ignore[unresolved-import]
import e00430_common as common430  # noqa: E402  # ty: ignore[unresolved-import]
import e00430_fit as fit430  # noqa: E402  # ty: ignore[unresolved-import]
import e00493_common as common  # noqa: E402
import torch  # noqa: E402  # ty: ignore[unresolved-import]

REFIT = f310.REFIT_STARTS[0]


def pred_path(*, arm: str, seed: int) -> Path:
    """Return the prediction file of an annual-refit job."""
    return common.PRED_DIR / f"annual_{arm}" / f"s{seed}_{REFIT:%Y-%m}.parquet"


def run_job(*, arm: str, seed: int) -> str:
    """Fit one annual (arm, seed) and write its 12 months of forecasts."""
    path = pred_path(arm=arm, seed=seed)
    if path.exists():
        return f"annual_{arm} s{seed} skipped"
    began = time.time()
    config = common430.ARMS[arm]
    assert config.pretrain == "none"
    cutoff = REFIT - timedelta(days=f310.REFIT_GAP_DAYS)
    pieces = [
        fit430.ens_piece(farm=f, band=b, cutoff=cutoff, need_members=config.needs_members)
        for f in c310.FARMS
        for b in c310.BAND_ORDER
    ]
    stats = fit430.standardiser(pieces=pieces, columns=config.columns)
    model = fit430.train_network(
        pieces=pieces,
        config=config,
        seed=seed,
        stats=stats,
        learning_rate=common430.LEARNING_RATE,
    )
    table = fit430.predict_rows(
        model=model,
        config=config,
        stats=stats,
        start=REFIT,
        end=f310.month_add(start=REFIT, months=common.ANNUAL_MONTHS),
        cutoff=cutoff,
        seed=seed,
        arm=f"annual_{arm}",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    table.write_parquet(tmp)
    tmp.replace(path)
    state_path = common.DATA_DIR / "models" / f"annual_{arm}_s{seed}.pt"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), state_path)
    del model
    torch.cuda.empty_cache()
    return f"annual_{arm} s{seed} {table.height} rows {time.time() - began:.0f}s"


def main() -> None:
    """Run every arm and seed one after another in this process."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default=",".join(map(str, common.SEEDS)))
    args = parser.parse_args()
    for arm in common.BASE_ARMS:
        for seed in (int(s) for s in args.seeds.split(",")):
            print(run_job(arm=arm, seed=seed), flush=True)


if __name__ == "__main__":
    main()
