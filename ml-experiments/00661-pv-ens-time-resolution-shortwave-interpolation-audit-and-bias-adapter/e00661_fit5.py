# ruff: noqa: E501
"""Experiment 661, step 5: retrain experiment 639's free and calibrated explicit networks on degraded known weather.

Reuses experiment 639's code unchanged. The only changes are in this script: the raw input matrices
come from `e00661_degrade` (shortwave degraded to 3-hour or 6-hour windows), and the weights, test
rows, and quantiles are stored in this experiment's folder. The calibrated capacity table is
experiment 639's (`capacity_parameters`), read from its probes of the undegraded free networks, so
that capacity is the same number in every variant.

Usage: python e00661_fit5.py --variant w3 --cutoff 2026-06 --seeds 0 1 2
"""

import argparse
import json
import time
from pathlib import Path
from typing import Final

import e00418_prep as p418  # ty: ignore[unresolved-import]
import e00554_model as m  # ty: ignore[unresolved-import]
import e00634_fit as f634  # ty: ignore[unresolved-import]
import e00639_common as k  # ty: ignore[unresolved-import]
import e00639_data as d  # ty: ignore[unresolved-import]
import e00639_fit as f639  # ty: ignore[unresolved-import]
import e00639_net as net  # ty: ignore[unresolved-import]
import e00661_degrade as g
import numpy as np
import torch  # ty: ignore[unresolved-import]
from e00510_common import PRETRAIN_LR  # ty: ignore[unresolved-import]

WORK5: Final[Path] = g.WORK / "step5"


def _paths(*, variant: str) -> None:
    """Point experiment 639's path functions and the raw-matrix folder at this variant."""
    p418.RAW_DIR = g.raw_dir(variant=variant)
    root = WORK5 / variant
    k.free_weights_path = lambda *, cutoff, world, seed: (
        root / "pretrained" / cutoff / f"{world}_free_s{seed}.pt"
    )
    k.explicit_weights_path = lambda *, cutoff, world, arm, seed: (
        root / "pretrained" / cutoff / f"{world}_{arm}_s{seed}.pt"
    )
    k.prediction_path = lambda *, cutoff, world, arm, seed: (
        root / "predictions" / cutoff / f"{world}_{arm}_s{seed}.npz"
    )
    k.test_rows_path = lambda *, cutoff, world: root / "predictions" / cutoff / f"{world}_rows.npz"


def run_free_real(*, cutoff: str, seeds: list[int]) -> None:
    """Train the free network of the real world on degraded weather (experiment 639's `run_free`, one world)."""
    world = "real"
    train, rows, clean = d.build_train(cutoff=cutoff, world=world)
    test = d.build_test(cutoff=cutoff, world=world, rows=rows, clean=clean)
    f639._save_rows(cutoff=cutoff, world=world, test=test)
    mean, std = f639._stats(rows=rows)
    gpu = f639._gpu(train)
    z = torch.from_numpy(
        ((train.y - rows.target_mean[train.primary]) / rows.target_std[train.primary]).astype(
            np.float32
        )
    ).to(f639.DEVICE)
    w = torch.ones(len(z), device=f639.DEVICE)
    for seed in seeds:
        path = k.free_weights_path(cutoff=cutoff, world=world, seed=seed)
        target = k.prediction_path(cutoff=cutoff, world=world, arm="free", seed=seed)
        if target.exists():
            continue
        torch.manual_seed(seed)
        model = m.ModularMdn(n_primaries=16, stage="pre").to(f639.DEVICE)
        started = time.monotonic()
        epochs, best = f634.train(model=model, x=gpu["x"], z=z, w=w, p=gpu["p"], is_val=gpu["is_val"], seed=seed, lr=PRETRAIN_LR, device=f639.DEVICE)  # fmt: skip
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path.with_suffix(".tmp"))
        path.with_suffix(".tmp").rename(path)
        model.eval()
        f639._store(path=target, q=net.forecast(model=model, x=test.x, p=test.primary, mean=mean, std=std, subtract=None, device=f639.DEVICE))  # fmt: skip
        print(
            f"{cutoff} free s{seed}: {epochs} epochs, val NLL {best:.4f}, {time.monotonic() - started:.0f} s",
            flush=True,
        )
    del gpu, z
    torch.cuda.empty_cache()


def main() -> None:
    """Train and forecast one variant at one cutoff."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", required=True, choices=list(g.WIDTH_H))
    p.add_argument("--cutoff", required=True, choices=k.CUTOFFS)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    a = p.parse_args()
    _paths(variant=a.variant)
    started = time.monotonic()
    run_free_real(cutoff=a.cutoff, seeds=a.seeds)
    f639.run_explicit(cutoff=a.cutoff, arms=["cal"], worlds=["real"], seeds=a.seeds)
    print(
        json.dumps(
            {"done": a.variant, "cutoff": a.cutoff, "seconds": round(time.monotonic() - started)}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
