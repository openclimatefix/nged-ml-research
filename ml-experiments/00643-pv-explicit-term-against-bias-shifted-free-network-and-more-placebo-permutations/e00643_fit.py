# ruff: noqa: E501, ANN001, ANN202
"""Experiment 643: fit the new arms and write their test-window quantiles.

Phases (one cutoff each): `fbias` (no fitting: shift stored free forecasts by a training-month constant),
`foff` (free network with a trainable sunny-midday offset), `perm` (explicit network with shifted
capacity timelines; `--arms cal perm1 perm2 ...`).

Usage: python e00643_fit.py foff --cutoff 2026-06
       python e00643_fit.py perm --cutoff 2026-06 --worlds step --arms perm1 perm3 --seeds 0 1 2 3 4
"""

import argparse
import time

import e00554_model as m  # ty: ignore[unresolved-import]
import e00634_common as k634  # ty: ignore[unresolved-import]
import e00639_common as k639  # ty: ignore[unresolved-import]
import e00639_data as d  # ty: ignore[unresolved-import]
import e00639_net as net  # ty: ignore[unresolved-import]
import e00643_common as k
import numpy as np
import torch  # ty: ignore[unresolved-import]
from e00510_common import PRETRAIN_LR  # ty: ignore[unresolved-import]

DEVICE = "cuda"
F32 = torch.float32


def _sunny(block: d.Block) -> np.ndarray:
    """Return 1.0 on sunny-midday rows (as experiment 639 defines them) and 0.0 elsewhere."""
    return (
        (block.kt >= k639.SUNNY_KT)
        & (block.hour >= k639.SUNNY_HOURS[0])
        & (block.hour <= k639.SUNNY_HOURS[1])
    ).astype(np.float64)


def _store(*, path, q: np.ndarray) -> None:
    """Write quantiles atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, q=q.astype(np.float32))
    tmp.rename(path)


def _stats(*, rows):
    """Return the per-Primary net-target mean and standard deviation of the fold."""
    return (
        torch.from_numpy(rows.target_mean).to(DEVICE, F32),
        torch.from_numpy(rows.target_std).to(DEVICE, F32),
    )


def _tensors(*, block: d.Block) -> dict[str, torch.Tensor]:
    """Return the per-row tensors of a training block."""
    return {
        "x": torch.from_numpy(block.x).to(DEVICE), "p": torch.from_numpy(block.primary).to(DEVICE),
        "is_val": torch.from_numpy(block.is_val).to(DEVICE),
        "y": torch.from_numpy(block.y).to(DEVICE, F32), "r4": torch.from_numpy(block.r4).to(DEVICE, F32),
        "after": torch.from_numpy(block.after).to(DEVICE, F32),
        "u": torch.from_numpy(block.u).to(DEVICE, F32), "inc": torch.from_numpy(block.inc).to(DEVICE, F32),
    }  # fmt: skip


def run_fbias(*, cutoff: str, worlds: list[str]) -> None:
    """Shift each free network's sunny-midday test forecasts by its mean median error on the held-out training rows."""
    for world in worlds:
        train, rows, clean = d.build_train(cutoff=cutoff, world=world)
        test = d.build_test(cutoff=cutoff, world=world, rows=rows, clean=clean)
        mean, std = _stats(rows=rows)
        val = train.is_val & (train.inc > 0) & (_sunny(train) > 0)
        test_sunny = (_sunny(test) > 0) & (test.inc > 0)
        shifts = {}
        for seed in k.STAGE1_SEEDS:
            model = m.ModularMdn(n_primaries=16, stage="pre").to(DEVICE)
            model.load_state_dict(
                torch.load(
                    k639.free_weights_path(cutoff=cutoff, world=world, seed=seed),
                    map_location=DEVICE,
                )
            )
            model.eval()
            q = net.forecast(
                model=model,
                x=train.x[val],
                p=train.primary[val],
                mean=mean,
                std=std,
                subtract=None,
                device=DEVICE,
            )
            err = q[:, 5] - train.y[val]
            shift = np.zeros(16)
            for p in np.unique(train.primary[val]):
                shift[p] = err[train.primary[val] == p].mean()
            shifts[seed] = shift
            q_test = np.load(
                k639.prediction_path(cutoff=cutoff, world=world, arm="free", seed=seed)
            )["q"].astype(np.float64)
            q_test[test_sunny] -= shift[test.primary[test_sunny]][:, None]
            _store(
                path=k.prediction_path(cutoff=cutoff, world=world, arm="fbias", seed=seed), q=q_test
            )
        k639.save_json(
            path=k.WORK / "analysis" / f"fbias_shift_{cutoff}_{world}.json",
            value={
                "rows": int(val.sum()),
                "shift_points_by_seed": {s: v.tolist() for s, v in shifts.items()},
            },
        )
        print(f"{cutoff} {world} fbias: {int(val.sum())} held-out sunny rows", flush=True)


def _train_and_store(
    *, cutoff, world, arm, seed, g, train_u, test, tensors, mean, std, u_test, r4_test, after_test, extra: dict
) -> None:  # fmt: skip
    """Train one explicit-form network, write its weights, offsets, and test quantiles."""
    target = k.prediction_path(cutoff=cutoff, world=world, arm=arm, seed=seed)
    if target.exists():
        return
    started = time.monotonic()
    torch.manual_seed(seed)
    model = m.ModularMdn(n_primaries=16, stage="pre").to(DEVICE)
    epochs, best, delta = net.train_explicit(
        model=model, x=g["x"], y=g["y"], u_inc=train_u, r4=g["r4"], after=g["after"], p=g["p"],
        is_val=g["is_val"], mean=mean, std=std, tensors=tensors, seed=seed, lr=PRETRAIN_LR, device=DEVICE,
    )  # fmt: skip
    weights = k.weights_path(cutoff=cutoff, world=world, arm=arm, seed=seed)
    weights.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights.with_suffix(".tmp"))
    weights.with_suffix(".tmp").rename(weights)
    k639.save_json(
        path=weights.with_suffix(".json"),
        value={"epochs": epochs, "val_nll": best, "delta_mw": delta.cpu().tolist(), **extra},
    )
    cap = net.capacity_mw(
        t=tensors, delta=delta, p=torch.from_numpy(test.primary).to(DEVICE), r4=r4_test, after=after_test,
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


def run_networks(
    *, phase: str, cutoff: str, arms: list[str], worlds: list[str], seeds: list[int]
) -> None:
    """Train `foff` (phase `foff`) or explicit arms `cal`, `perm<shift>` (phase `perm`)."""
    six = k634.six_indices()
    zero = {s: k639.SiteCapacity(k=0.0, d=0.0, rel_sd_k=0.0, rel_sd_d=0.0) for s in six}
    for world in worlds:
        train, rows, clean = d.build_train(cutoff=cutoff, world=world)
        test = d.build_test(cutoff=cutoff, world=world, rows=rows, clean=clean)
        mean, std = _stats(rows=rows)
        g = _tensors(train)
        for arm in arms:
            if phase == "foff":
                tensors = net.parameter_tensors(params=zero, six=six, device=DEVICE)
                train_u = torch.from_numpy(_sunny(train) * train.inc).to(DEVICE, F32)
                u_test = torch.from_numpy(_sunny(test) * test.inc).to(DEVICE, F32)
                r4_t = torch.zeros(len(test.primary), device=DEVICE)
                after_t = torch.zeros(len(test.primary), device=DEVICE)
                g_use = {
                    **g,
                    "r4": torch.zeros_like(g["r4"]),
                    "after": torch.zeros_like(g["after"]),
                }
                extra: dict = {}
            else:
                params = (
                    k639.capacity_parameters(cutoff=cutoff, arm="cal")
                    if arm == "cal"
                    else k.shifted_parameters(cutoff=cutoff, shift=int(arm.removeprefix("perm")))
                )
                tensors = net.parameter_tensors(params=params, six=six, device=DEVICE)
                train_u = g["u"] * g["inc"]
                u_test = torch.from_numpy(test.u * test.inc).to(DEVICE, F32)
                r4_t = torch.from_numpy(test.r4).to(DEVICE, F32)
                after_t = torch.from_numpy(test.after).to(DEVICE, F32)
                g_use = g
                extra = {"params": {s: vars(v) for s, v in params.items()}}
            for seed in seeds:
                _train_and_store(
                    cutoff=cutoff, world=world, arm=arm, seed=seed, g=g_use, train_u=train_u, test=test, tensors=tensors,
                    mean=mean, std=std, u_test=u_test, r4_test=r4_t, after_test=after_t, extra=extra,
                )  # fmt: skip
        del g
        torch.cuda.empty_cache()


def main() -> None:
    """Run one phase for one cutoff."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fbias", "foff", "perm"))
    parser.add_argument("--cutoff", required=True, choices=k639.CUTOFFS)
    parser.add_argument("--arms", nargs="+", default=["foff"])
    parser.add_argument("--worlds", nargs="+", default=list(k639.WORLDS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(k.STAGE1_SEEDS))
    a = parser.parse_args()
    if a.phase == "fbias":
        run_fbias(cutoff=a.cutoff, worlds=a.worlds)
    else:
        arms = ["foff"] if a.phase == "foff" else a.arms
        run_networks(phase=a.phase, cutoff=a.cutoff, arms=arms, worlds=a.worlds, seeds=a.seeds)
    print(f"done {a.phase} {a.cutoff}", flush=True)


if __name__ == "__main__":
    main()
