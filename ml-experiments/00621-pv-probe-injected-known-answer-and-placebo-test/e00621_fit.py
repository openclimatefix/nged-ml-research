# ruff: noqa: E501, ANN001
"""Experiment 621: pretrain experiment 554's network M0 on ERA5 and CAMS rows with a known injection.

The network, rows, loss, learning rate, early stopping and seeds are experiment 554's `pretrain`
(stage `pre`); the only change is the injected condition added to the target of the six signed-MW
Primaries (`e00621_common.Injector`). Rows with a gap in any of the farms A to F are dropped from
every arm, the uninjected `control` included.

Usage: python e00621_fit.py --cutoff 2026-06 --conditions control farm_mw4_r0 --seeds 0 1 2
"""

import argparse
import json
import time
from datetime import UTC, datetime

import e00397_models as base  # ty: ignore[unresolved-import]
import e00428_data as data  # ty: ignore[unresolved-import]
import e00448_run as r448  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import e00554_model as m  # ty: ignore[unresolved-import]
import e00621_common as k
import numpy as np
import torch  # ty: ignore[unresolved-import]

FRAME_HOURS = (8, 17)
"""Daylight hours (UTC valid-time hours) kept in the probe frame."""


def train(*, model, x, z, w, p, is_val, seed: int, lr: float, device: str) -> tuple[int, float]:
    """Experiment 554's loop for the stage `pre` network (no fresh mixer)."""
    torch.manual_seed(seed)
    train_idx = torch.nonzero(~is_val).squeeze(1)
    val_idx = torch.nonzero(is_val).squeeze(1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if len(val_idx) > c.VALIDATION_ROWS:
        pick = torch.randperm(len(val_idx), generator=generator)[: c.VALIDATION_ROWS]
        val_idx = val_idx[pick.to(device)]
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=c.WEIGHT_DECAY)
    best, best_state, stale, epochs = float("inf"), None, 0, 0
    for epoch in range(c.MAX_EPOCHS):
        model.train()
        order = train_idx[torch.randperm(len(train_idx), device=device)]
        order = order[: int(c.ROW_FRACTION * len(order))]
        for start in range(0, len(order), c.BATCH):
            b = order[start : start + c.BATCH]
            nll = base.mixture_nll(out=model(x[b], p[b]), z=z[b], components=m.COMPONENTS)
            loss = (w[b] * nll).sum() / w[b].sum()
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), c.GRAD_CLIP)
            optimiser.step()
        model.eval()
        with torch.no_grad():
            val = float(
                torch.cat(
                    [
                        base.mixture_nll(out=model(x[i], p[i]), z=z[i], components=m.COMPONENTS)
                        for i in val_idx.split(c.BATCH * 2)
                    ]
                ).mean()
            )
        epochs = epoch + 1
        if val < best - 1e-5:
            best, stale = val, 0
            best_state = {a: v.detach().clone() for a, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= c.PATIENCE:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return epochs, best


def write_frame(*, cutoff: str, pre, rows, keep: np.ndarray) -> None:
    """Save the probe frame: one row per (Primary, target time) of the six Primaries, daylight only."""
    six = k.six_indices()
    t_us = k.time_us(target_idx=pre.target_idx)
    hour = (t_us // 3_600_000_000) % 24
    sel = (
        keep
        & np.isin(pre.primary, list(six.values()))
        & (hour >= FRAME_HOURS[0])
        & (hour <= FRAME_HOURS[1])
    )
    key = pre.primary[sel].astype(np.int64) * 10_000_000 + pre.target_idx[sel]
    _, first = np.unique(key, return_index=True)
    idx = np.nonzero(sel)[0][first]
    out = k.frame_path(cutoff=cutoff)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp.npz")
    np.savez(
        tmp, x=pre.x[idx, :35], primary=pre.primary[idx], target_idx=pre.target_idx[idx],
        is_val=pre.is_val[idx], target_mean=rows.target_mean, target_std=rows.target_std,
        std_mean=rows.standardiser.mean[:7], std_std=rows.standardiser.std[:7],
    )  # fmt: skip
    tmp.rename(out)
    print(f"frame {cutoff}: {len(idx):,} rows", flush=True)


def main() -> None:
    """Pretrain the requested conditions and seeds for one cutoff."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", required=True, choices=k.CUTOFFS)
    parser.add_argument("--conditions", nargs="+", default=list(k.CONDITIONS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(k.SEEDS))
    parser.add_argument("--frame-only", action="store_true")
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
        f"{args.cutoff}: {len(pre.y):,} rows, {int(inj.gap.sum()):,} dropped for farm gaps "
        f"({inj.gap.sum() / inj.in_six.sum():.1%} of the six Primaries' rows); latest valid {latest.isoformat()}",
        flush=True,
    )
    write_frame(cutoff=args.cutoff, pre=pre, rows=rows, keep=keep)
    if args.frame_only:
        return
    tgt_mean, tgt_std = rows.target_mean[pre.primary], rows.target_std[pre.primary]
    x_gpu = torch.from_numpy(pre.x[keep]).to(device)
    p_gpu = torch.from_numpy(pre.primary[keep]).to(device)
    v_gpu = torch.from_numpy(pre.is_val[keep]).to(device)
    w_gpu = torch.ones(int(keep.sum()), device=device)
    for name in args.conditions:
        cond = k.parse(name=name)
        change = inj.change_points(condition=cond)
        y = pre.y.astype(np.float64) + change
        z = ((y - tgt_mean) / tgt_std)[keep].astype(np.float32)
        z_gpu = torch.from_numpy(z).to(device)
        for seed in args.seeds:
            target = k.weights_path(cutoff=args.cutoff, condition=name, seed=seed)
            if target.exists():
                continue
            started = time.monotonic()
            torch.manual_seed(seed)
            model = m.ModularMdn(n_primaries=len(rows.target_mean), stage="pre").to(device)
            epochs, best = train(
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
