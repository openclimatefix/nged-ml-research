# ruff: noqa: E501
"""Experiment 638: pretrain experiment 554's network M0 with a step or a climatological placebo injected.

Identical to experiment 634's `e00634_fit.py` (same rows, loss, early stopping, seeds, farm-gap mask,
farm-to-Primary assignment, and `train`), except for the injected conditions of `e00638_common`. The
probe frames are experiment 634's, so no frame is written. The `control` and constant-injection
networks are experiment 634's too.

Usage: python e00638_fit.py --cutoff 2026-06 --conditions step_spring23_mw4 --seeds 0 1 2
"""

import argparse
import json
import time
from datetime import UTC, datetime

import e00428_data as data  # ty: ignore[unresolved-import]
import e00448_run as r448  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import e00554_model as m  # ty: ignore[unresolved-import]
import e00634_fit as f634  # ty: ignore[unresolved-import]
import e00638_common as k
import numpy as np
import torch  # ty: ignore[unresolved-import]


def main() -> None:
    """Pretrain the requested conditions and seeds for one cutoff."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", required=True, choices=k.CUTOFFS)
    parser.add_argument("--conditions", nargs="+", default=list(k.CONDITIONS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(k.SEEDS))
    args = parser.parse_args()
    device = "cuda"
    clean = c.clean_stack()
    refit_t = c.month_start(month=args.cutoff)
    rows = data.build_fold(refit=refit_t, clean=clean)
    pre = r448.pretrain_rows(rows=rows, refit=refit_t, clean=clean)
    latest = datetime.fromtimestamp(
        (pre.target_idx.max() * c.prep422.STEP_US + c.prep422.GRID_START_US) / 1e6, UTC
    )
    assert latest <= refit_t - c.regimes.REFIT_CUTOFF_AGE, (latest, refit_t)
    inj = k.Injector(
        primary=pre.primary, target_idx=pre.target_idx, x=pre.x, standardiser=rows.standardiser
    )
    keep = ~inj.gap
    print(
        f"{args.cutoff}: {len(pre.y):,} rows, {int(keep.sum()):,} kept, latest valid {latest.isoformat()}",
        flush=True,
    )
    tgt_mean, tgt_std = rows.target_mean[pre.primary], rows.target_std[pre.primary]
    x_gpu = torch.from_numpy(pre.x[keep]).to(device)
    p_gpu = torch.from_numpy(pre.primary[keep]).to(device)
    v_gpu = torch.from_numpy(pre.is_val[keep]).to(device)
    w_gpu = torch.ones(int(keep.sum()), device=device)
    for name in args.conditions:
        change = inj.change_points_step(condition=k.parse(name=name))
        if not np.any(change[keep]):
            print(
                f"  {args.cutoff} {name}: no post-step rows, identical to the control; skipped",
                flush=True,
            )
            continue
        z = (((pre.y.astype(np.float64) + change) - tgt_mean) / tgt_std)[keep].astype(np.float32)
        z_gpu = torch.from_numpy(z).to(device)
        for seed in args.seeds:
            target = k.weights_path(cutoff=args.cutoff, condition=name, seed=seed)
            if target.exists():
                continue
            started = time.monotonic()
            torch.manual_seed(seed)
            model = m.ModularMdn(n_primaries=len(rows.target_mean), stage="pre").to(device)
            epochs, best = f634.train(
                model=model, x=x_gpu, z=z_gpu, w=w_gpu, p=p_gpu, is_val=v_gpu, seed=seed,
                lr=c.PRETRAIN_LR, device=device,
            )  # fmt: skip
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(".tmp")
            torch.save(model.state_dict(), tmp)
            tmp.rename(target)
            seconds = time.monotonic() - started
            print(
                f"  {args.cutoff} {name} s{seed}: {epochs} epochs, val NLL {best:.4f}, {seconds:.0f} s",
                flush=True,
            )
            log = k.WORK / "logs" / args.cutoff / f"{name}_s{seed}.json"
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(json.dumps({"epochs": epochs, "val_nll": best, "seconds": seconds}))
        del z_gpu
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
