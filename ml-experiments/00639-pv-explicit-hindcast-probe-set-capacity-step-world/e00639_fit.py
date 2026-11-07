# ruff: noqa: E501, ANN001
"""Experiment 639: train the free and explicit networks and forecast the test windows.

Phases, run in this order for each cutoff:

1. `free`: train missing free networks of both worlds (experiment 634's `train`, unchanged),
   forecast the
   test window with all five seeds, and write each world's test rows.
2. (run `e00639_probe.py` on the CPU.)
3. `explicit`: train the explicit arms and forecast the test window.

Usage: python e00639_fit.py free --cutoff 2026-06
       python e00639_fit.py explicit --cutoff 2026-06 --arms cal raw
"""

import argparse
import json
import time

import e00554_model as m  # ty: ignore[unresolved-import]
import e00634_common as k634  # ty: ignore[unresolved-import]
import e00634_fit as f634  # ty: ignore[unresolved-import]
import e00639_common as k
import e00639_data as d
import e00639_net as net
import numpy as np
import torch  # ty: ignore[unresolved-import]
from e00510_common import PRETRAIN_LR  # ty: ignore[unresolved-import]

DEVICE = "cuda"


def _gpu(block: d.Block) -> dict[str, torch.Tensor]:
    """Return the per-row tensors of a training block."""
    f32 = torch.float32
    return {
        "x": torch.from_numpy(block.x).to(DEVICE), "p": torch.from_numpy(block.primary).to(DEVICE),
        "is_val": torch.from_numpy(block.is_val).to(DEVICE),
        "y": torch.from_numpy(block.y).to(DEVICE, f32), "r4": torch.from_numpy(block.r4).to(DEVICE, f32),
        "after": torch.from_numpy(block.after).to(DEVICE, f32),
        "u": torch.from_numpy(block.u).to(DEVICE, f32), "u_shuf": torch.from_numpy(block.u_shuf).to(DEVICE, f32),
        "inc": torch.from_numpy(block.inc).to(DEVICE, f32),
    }  # fmt: skip


def _save_rows(*, cutoff: str, world: str, test: d.Block) -> None:
    """Write a test window's row keys, actuals, and masks."""
    path = k.test_rows_path(cutoff=cutoff, world=world)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path.with_name(path.name + ".tmp.npz"), primary=test.primary, target_idx=test.target_idx,
        y=test.y, kt=test.kt, hour=test.hour, t_us=test.t_us, inc=test.inc,
    )  # fmt: skip
    path.with_name(path.name + ".tmp.npz").rename(path)


def _store(*, path, q: np.ndarray) -> None:
    """Write quantiles atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, q=q.astype(np.float32))
    tmp.rename(path)


def _stats(*, rows) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the per-Primary net-target mean and standard deviation of the fold."""
    return (
        torch.from_numpy(rows.target_mean).to(DEVICE, torch.float32),
        torch.from_numpy(rows.target_std).to(DEVICE, torch.float32),
    )


def run_free(*, cutoff: str) -> None:
    """Train the missing free networks and forecast the test window with all seeds."""
    for world in k.WORLDS:
        train, rows, clean = d.build_train(cutoff=cutoff, world=world)
        test = d.build_test(cutoff=cutoff, world=world, rows=rows, clean=clean)
        _save_rows(cutoff=cutoff, world=world, test=test)
        mean, std = _stats(rows=rows)
        g = _gpu(train)
        z = torch.from_numpy(
            ((train.y - rows.target_mean[train.primary]) / rows.target_std[train.primary]).astype(
                np.float32
            )
        ).to(DEVICE)
        w = torch.ones(len(z), device=DEVICE)
        for seed in k.SEEDS:
            path = k.free_weights_path(cutoff=cutoff, world=world, seed=seed)
            torch.manual_seed(seed)
            model = m.ModularMdn(n_primaries=16, stage="pre").to(DEVICE)
            if not path.exists():
                started = time.monotonic()
                epochs, best = f634.train(
                    model=model, x=g["x"], z=z, w=w, p=g["p"], is_val=g["is_val"], seed=seed, lr=PRETRAIN_LR, device=DEVICE,
                )  # fmt: skip
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".tmp")
                torch.save(model.state_dict(), tmp)
                tmp.rename(path)
                print(
                    f"{cutoff} {world} free s{seed}: {epochs} epochs, val NLL {best:.4f}, {time.monotonic() - started:.0f} s",
                    flush=True,
                )
            else:
                model.load_state_dict(torch.load(path, map_location=DEVICE))
            model.eval()
            target = k.prediction_path(cutoff=cutoff, world=world, arm="free", seed=seed)
            if not target.exists():
                q = net.forecast(
                    model=model,
                    x=test.x,
                    p=test.primary,
                    mean=mean,
                    std=std,
                    subtract=None,
                    device=DEVICE,
                )
                _store(path=target, q=q)
        del g, z
        torch.cuda.empty_cache()


def run_explicit(*, cutoff: str, arms: list[str], worlds: list[str], seeds: list[int]) -> None:
    """Train the explicit arms and forecast the test window."""
    six = k634.six_indices()
    for world in worlds:
        train, rows, clean = d.build_train(cutoff=cutoff, world=world)
        test = d.build_test(cutoff=cutoff, world=world, rows=rows, clean=clean)
        mean, std = _stats(rows=rows)
        g = _gpu(train)
        for arm in arms:
            if arm not in k.ARMS_BY_WORLD[world]:
                continue
            params = k.capacity_parameters(cutoff=cutoff, arm=arm)
            tensors = net.parameter_tensors(params=params, six=six, device=DEVICE)
            shuffled = arm == "shuf"
            u_train = (g["u_shuf"] if shuffled else g["u"]) * g["inc"]
            u_test = torch.from_numpy((test.u_shuf if shuffled else test.u) * test.inc).to(
                DEVICE, torch.float32
            )
            for seed in seeds:
                target = k.prediction_path(cutoff=cutoff, world=world, arm=arm, seed=seed)
                if target.exists():
                    continue
                started = time.monotonic()
                torch.manual_seed(seed)
                model = m.ModularMdn(n_primaries=16, stage="pre").to(DEVICE)
                epochs, best, delta = net.train_explicit(
                    model=model, x=g["x"], y=g["y"], u_inc=u_train, r4=g["r4"], after=g["after"], p=g["p"],
                    is_val=g["is_val"], mean=mean, std=std, tensors=tensors, seed=seed, lr=PRETRAIN_LR, device=DEVICE,
                )  # fmt: skip
                weights = k.explicit_weights_path(cutoff=cutoff, world=world, arm=arm, seed=seed)
                weights.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), weights.with_suffix(".tmp"))
                weights.with_suffix(".tmp").rename(weights)
                k.save_json(
                    path=weights.with_suffix(".json"),
                    value={
                        "epochs": epochs,
                        "val_nll": best,
                        "delta_mw": delta.cpu().tolist(),
                        "params": {s: vars(v) for s, v in params.items()},
                    },
                )
                p_test = torch.from_numpy(test.primary).to(DEVICE)
                cap = net.capacity_mw(
                    t=tensors, delta=delta,
                    p=p_test, r4=torch.from_numpy(test.r4).to(DEVICE, torch.float32),
                    after=torch.from_numpy(test.after).to(DEVICE, torch.float32),
                )  # fmt: skip
                q = net.forecast(
                    model=model,
                    x=test.x,
                    p=test.primary,
                    mean=mean,
                    std=std,
                    subtract=(cap * u_test).cpu(),
                    device=DEVICE,
                )
                _store(path=target, q=q)
                print(
                    f"{cutoff} {world} {arm} s{seed}: {epochs} epochs, val NLL {best:.4f}, {time.monotonic() - started:.0f} s",
                    flush=True,
                )
        del g
        torch.cuda.empty_cache()


def main() -> None:
    """Run one phase for one cutoff."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("free", "explicit"))
    parser.add_argument("--cutoff", required=True, choices=k.CUTOFFS)
    parser.add_argument("--arms", nargs="+", default=list(k.EXPLICIT_ARMS))
    parser.add_argument("--worlds", nargs="+", default=list(k.WORLDS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(k.SEEDS))
    args = parser.parse_args()
    if args.phase == "free":
        run_free(cutoff=args.cutoff)
    else:
        run_explicit(cutoff=args.cutoff, arms=args.arms, worlds=args.worlds, seeds=args.seeds)
    print(json.dumps({"done": args.phase, "cutoff": args.cutoff}), flush=True)


if __name__ == "__main__":
    main()
