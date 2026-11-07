# ruff: noqa: E501, ANN001
"""Experiment 635: pretrain the metered-input networks (arms b and c) with experiment 621's injections.

Everything but the extra input column is experiment 621's `e00621_fit.train`. After each fit the
network's validation CRPS (standardised target units, 64 draws per row) is stored beside the fit's
log. `--arm a` only scores experiment 621's stored control networks.

Usage: python e00635_fit.py --arm b --conditions control_r0 farm_mw4_r0 --seeds 0 1 2
"""

import argparse
import json
import time

import e00428_data as data  # ty: ignore[unresolved-import]
import e00448_run as r448  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import e00554_model as m  # ty: ignore[unresolved-import]
import e00621_common as k  # ty: ignore[unresolved-import]
import e00621_fit as f621  # ty: ignore[unresolved-import]
import e00635_common as k5
import e00635_model as mm
import numpy as np
import torch  # ty: ignore[unresolved-import]

import e00397_models as base  # ty: ignore[unresolved-import]  # isort: skip

CRPS_ROWS = 40_000
CRPS_DRAWS = 64


def crps(*, model, x, z, p, is_val, seed: int, six_only: bool = False) -> float:
    """Return the Monte Carlo CRPS of the mixture on a fixed validation sample (standardised units)."""
    g = torch.Generator(device="cpu").manual_seed(12345)
    mask = is_val & torch.isin(p, torch.tensor(list(k.six_indices().values()), device=p.device)) if six_only else is_val
    val = torch.nonzero(mask).squeeze(1).cpu()
    pick = val[torch.randperm(len(val), generator=g)[:CRPS_ROWS]].to(x.device)
    with torch.no_grad():
        out = model(x[pick], p[pick])
        log_w, mean, std = base.mixture_parts(out=out, components=m.COMPONENTS)
        gen = torch.Generator(device=x.device).manual_seed(777 + seed)
        draws = []
        for _ in range(2):
            comp = torch.multinomial(log_w.exp(), CRPS_DRAWS, replacement=True, generator=gen)
            mu, sd = mean.gather(1, comp), std.gather(1, comp)
            draws.append(mu + sd * torch.randn(mu.shape, device=x.device, generator=gen))
        a, b = draws
        zz = z[pick][:, None]
        score = (a - zz).abs().mean(1) - 0.5 * (a - b).abs().mean(1)
    return float(score.mean())


def main() -> None:
    """Pretrain (or, for arm a, score) the requested conditions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=("a", "b", "c"))
    parser.add_argument("--conditions", nargs="+", default=list(k5.CONDITIONS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(k5.SEEDS))
    args = parser.parse_args()
    device = "cuda"
    clean = c.clean_stack()
    refit_t = c.month_start(month=k5.CUTOFF)
    rows = data.build_fold(refit=refit_t, clean=clean)
    pre = r448.pretrain_rows(rows=rows, refit=refit_t, clean=clean)
    inj = k.Injector(primary=pre.primary, target_idx=pre.target_idx, x=pre.x, standardiser=rows.standardiser)
    keep = ~inj.gap
    latest = pre.target_idx.max() * c.prep422.STEP_US + c.prep422.GRID_START_US
    assert latest <= refit_t.timestamp() * 1e6 - c.regimes.REFIT_CUTOFF_AGE.total_seconds() * 1e6, latest
    tgt_mean, tgt_std = rows.target_mean[pre.primary], rows.target_std[pre.primary]
    x_np = pre.x[keep]
    p_gpu = torch.from_numpy(pre.primary[keep]).to(device)
    v_gpu = torch.from_numpy(pre.is_val[keep]).to(device)
    w_gpu = torch.ones(int(keep.sum()), device=device)
    t_keep = k.time_us(target_idx=pre.target_idx)[keep]
    for name in args.conditions:
        if args.arm == "a" and name != "control_r0":
            continue
        cond, rotation, shuffled = k5.split(name=name)
        change = inj.change_points(condition=cond)
        y = pre.y.astype(np.float64) + change
        z = ((y - tgt_mean) / tgt_std)[keep].astype(np.float32)
        z_gpu = torch.from_numpy(z).to(device)
        x_np[:, mm.METERED_COL] = k5.metered_column(
            primary=pre.primary[keep], t_us=t_keep, rotation=rotation, shuffled=shuffled
        )
        x_gpu = torch.from_numpy(x_np).to(device)
        for seed in args.seeds:
            log = k5.WORK / "logs" / args.arm / f"{name}_s{seed}.json"
            if log.exists():
                continue
            started = time.monotonic()
            if args.arm == "a":
                model = m.ModularMdn(n_primaries=len(rows.target_mean), stage="pre").to(device)
                model.load_state_dict(torch.load(k.weights_path(cutoff=k5.CUTOFF, condition="control", seed=seed), map_location=device))
                model.eval()
                epochs, best = 0, float("nan")
            else:
                target = k5.weights_path(arm=args.arm, condition=name, seed=seed)
                torch.manual_seed(seed)
                model = mm.MeteredMdn(n_primaries=len(rows.target_mean), arm=args.arm).to(device)
                epochs, best = f621.train(
                    model=model, x=x_gpu, z=z_gpu, w=w_gpu, p=p_gpu, is_val=v_gpu, seed=seed,
                    lr=c.PRETRAIN_LR, device=device,
                )  # fmt: skip
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_suffix(".tmp")
                torch.save(model.state_dict(), tmp)
                tmp.rename(target)
            score = crps(model=model, x=x_gpu, z=z_gpu, p=p_gpu, is_val=v_gpu, seed=seed)
            score6 = crps(model=model, x=x_gpu, z=z_gpu, p=p_gpu, is_val=v_gpu, seed=seed, six_only=True)
            seconds = time.monotonic() - started
            print(f"  {args.arm} {name} s{seed}: {epochs} epochs, val NLL {best:.4f}, CRPS {score:.4f} (six {score6:.4f}), {seconds:.0f} s", flush=True)
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(json.dumps({"epochs": epochs, "val_nll": best, "crps": score, "crps_six": score6, "seconds": seconds}))
        del z_gpu, x_gpu
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
