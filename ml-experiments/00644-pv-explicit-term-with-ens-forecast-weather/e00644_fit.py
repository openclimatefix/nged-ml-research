# ruff: noqa: E501, ANN001, PLR0912, PLR0915, PT018
"""Experiment 644: fine-tune the free and explicit solar networks on ensemble-forecast weather and forecast.

For each pretraining cutoff, each month of its 60-day test window, and each world, a network starts
from experiment 639's (or 643's) stored ERA5-and-CAMS pretraining, is fine-tuned on the 24 months of
ensemble rows before the month, and forecasts every run of the month at lead up to 240 hours. The
recipe of the fine-tune is experiment 554's arm M0 (modular network, fresh `mixer_ft`, 90-day
recency half-life), and the explicit term is experiment 639's: the target is `y + C * inc * u`, and
the same term is subtracted from every forecast quantile. `u` is computed from the ensemble-mean
shortwave and temperature of the row.

Usage: python e00644_fit.py fit --cutoff 2026-06 [--arms a b] [--worlds real step] [--seeds 0 1 2] [--smoke]
"""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00397_models as base  # ty: ignore[unresolved-import]
import e00428_data as data  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import e00554_model as m  # ty: ignore[unresolved-import]
import e00634_common as k634  # ty: ignore[unresolved-import]
import e00638_common as k638  # ty: ignore[unresolved-import]
import e00639_common as k639  # ty: ignore[unresolved-import]
import e00639_data as d639  # ty: ignore[unresolved-import]
import e00639_net as net  # ty: ignore[unresolved-import]
import numpy as np
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final = "00644-pv-explicit-term-with-ens-forecast-weather"
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / NAME / "work"
DEVICE = "cuda"
CUTOFFS: Final[tuple[str, ...]] = k639.CUTOFFS
MONTHS: Final[dict[str, tuple[str, ...]]] = {
    "2025-07": ("2025-07", "2025-08"),
    "2026-03": ("2026-03", "2026-04"),
    "2026-06": ("2026-06",),
}
"""Calendar months of each cutoff's test window (experiment 639's 60 days, rounded to whole months)."""
ARMS: Final[dict[str, str]] = {
    "a": "free", "b": "cal", "c": "raw", "f": "r4", "p": "perm", "s": "shuf",
}  # fmt: skip
"""Fit arms: letter in the plan -> experiment 639's pretrained arm name (`p` is the capacity-permuted placebo)."""
HALF_LIFE_DAYS: Final = 90.0
FRESH_LR: Final = 5e-4
FRESH_SEED: Final = 12345
MIN_LEAD_H: Final = 0.0
"""Rows below this lead (hours from the power-forecast origin) are not scored."""


def fits_path(*, cutoff: str, month: str, world: str, arm: str, seed: int) -> Path:
    """Return where a fine-tuned network's weights are stored (its quantiles sit beside it, `.npz`)."""
    return WORK / "fits" / cutoff / f"{month}_{world}_{arm}_s{seed}.pt"


def rows_path(*, month: str, world: str) -> Path:
    """Return where a month's test-row keys, actuals, and the physics curve are stored."""
    return WORK / "rows" / f"{month}_{world}.npz"


def _pretrained(*, cutoff: str, world: str, arm: str, seed: int) -> Path:
    name = ARMS[arm]
    if name == "free":
        return k639.free_weights_path(cutoff=cutoff, world=world, seed=seed)
    return k639.explicit_weights_path(cutoff=cutoff, world=world, arm=name, seed=seed)


def _load_for_finetune(*, model: torch.nn.Module, path: Path) -> None:
    """Load pretrained weights without `mixer_ft`, then give `mixer_ft` experiment 554's seed-12345 initialisation."""
    raw = torch.load(path, map_location="cpu")
    state = {a: v for a, v in raw.items() if not a.startswith("mixer_ft.")}
    missing = model.load_state_dict(state, strict=False)
    assert all(a.startswith("mixer_ft.") for a in missing.missing_keys), missing
    assert not missing.unexpected_keys, missing
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(FRESH_SEED)
        fresh = base.mlp(
            n_in=model.mixer_ft[0].in_features, hidden=m.MIXER_HIDDEN, layers=2,  # ty: ignore[not-subscriptable,unresolved-attribute]
            n_out=3 * m.COMPONENTS, dropout=m.DROPOUT,
        )  # fmt: skip
        base.init_head(head=fresh, components=m.COMPONENTS)
    model.mixer_ft.load_state_dict(fresh.state_dict())  # ty: ignore[unresolved-attribute]


def _train(
    *, model, x, y, u_inc, r4, after, p, w, is_val, mean, std, tensors, seed: int, init_delta=None,
) -> tuple[int, float, torch.Tensor]:  # fmt: skip
    """Return experiment 639's `train_explicit` fit with recency weights and the fresh mixer group.

    Returns:
        The epochs run, the best validation NLL, and the best per-Primary offsets.
    """
    torch.manual_seed(seed)
    delta = (
        init_delta.clone()
        if init_delta is not None
        else torch.zeros(net.N_PRIMARIES, device=DEVICE)
    ).requires_grad_(True)
    train_idx = torch.nonzero(~is_val).squeeze(1)
    val_idx = torch.nonzero(is_val).squeeze(1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if len(val_idx) > c.VALIDATION_ROWS:
        pick = torch.randperm(len(val_idx), generator=generator)[: c.VALIDATION_ROWS]
        val_idx = val_idx[pick.to(DEVICE)]
    fresh = list(model.mixer_ft.parameters())
    fresh_ids = {id(q) for q in fresh}
    optimiser = torch.optim.AdamW(
        [
            {"params": [q for q in model.parameters() if id(q) not in fresh_ids], "lr": c.FINE_TUNE_LR, "weight_decay": c.WEIGHT_DECAY},
            {"params": fresh, "lr": FRESH_LR, "weight_decay": c.WEIGHT_DECAY},
            {"params": [delta], "lr": c.FINE_TUNE_LR, "weight_decay": 0.0},
        ]
    )  # fmt: skip
    prior = delta.detach().clone()

    def nll_of(idx: torch.Tensor, ek, ed) -> torch.Tensor:
        cap = net.capacity_mw(
            t=tensors, delta=delta, p=p[idx], r4=r4[idx], after=after[idx], ek=ek, ed=ed
        )
        z = (y[idx] + cap * u_inc[idx] - mean[p[idx]]) / std[p[idx]]
        return base.mixture_nll(out=model(x[idx], p[idx]), z=z, components=m.COMPONENTS)

    best, best_state, stale, epochs = float("inf"), None, 0, 0
    for epoch in range(c.MAX_EPOCHS):
        model.train()
        ek = torch.randn(net.N_PRIMARIES, device=DEVICE) * tensors["sd_k"]
        ed = torch.randn(net.N_PRIMARIES, device=DEVICE) * tensors["sd_d"]
        order = train_idx[torch.randperm(len(train_idx), device=DEVICE)]
        order = order[: int(c.ROW_FRACTION * len(order))]
        for start in range(0, len(order), c.BATCH):
            b = order[start : start + c.BATCH]
            penalty = (
                ((delta - prior) ** 2).sum()
                / (2 * k639.DELTA_PRIOR_MW**2)
                / k639.DELTA_PENALTY_ROWS
            )
            nll = nll_of(b, ek, ed)
            loss = (w[b] * nll).sum() / w[b].sum() + penalty
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), c.GRAD_CLIP)
            optimiser.step()
        model.eval()
        with torch.no_grad():
            val = float(
                torch.cat([nll_of(i, None, None) for i in val_idx.split(c.BATCH * 2)]).mean()
            )
        epochs = epoch + 1
        if val < best - 1e-5:
            best, stale = val, 0
            best_state = (
                {a: v.detach().clone() for a, v in model.state_dict().items()},
                delta.detach().clone(),
            )
        else:
            stale += 1
            if stale >= c.PATIENCE:
                break
    assert best_state is not None
    model.load_state_dict(best_state[0])
    model.eval()
    return epochs, best, best_state[1]


def _f32(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a)).to(DEVICE, torch.float32)


def run_month(
    *, cutoff: str, month: str, arms: list[str], worlds: list[str], seeds: list[int], clean: np.ndarray, smoke: bool,
) -> None:  # fmt: skip
    """Fine-tune and forecast one month of one cutoff's test window."""
    six = k634.six_indices()
    refit = c.month_start(month=month)
    mr = c.build_month(month=month, clean=clean)
    test = data.build_test(month=month, standardiser=mr.base.standardiser)
    mean = torch.from_numpy(mr.base.target_mean).to(DEVICE, torch.float32)
    std = torch.from_numpy(mr.base.target_std).to(DEVICE, torch.float32)
    # Fine-tuning rows: experiment 510's rule (valid time at least the cutoff age before the refit, 730 days back).
    sel = c.select_rows(rows=mr, cutoff=refit, half_life_days=HALF_LIFE_DAYS)
    assert sel.report["rows_past_valid_cut"] == 0 and sel.report["rows_past_origin_cut"] == 0
    idx = sel.index
    if smoke:
        idx = idx[::20]
    for world in worlds:
        inj = k638.Injector(
            primary=mr.primary[idx],
            target_idx=mr.target_idx[idx],
            x=mr.x[idx],
            standardiser=mr.base.standardiser,
        )
        keep = ~inj.gap
        change = (
            inj.change_points_step(condition=k638.parse(name="step_spring23_mw4"))
            if world == "step"
            else 0.0
        )
        y = mr.y[idx].astype(np.float64) + change
        blk = d639._decorate(
            cutoff=cutoff, world=world, x=mr.x[idx], y=y, primary=mr.primary[idx],
            target_idx=mr.target_idx[idx], is_val=mr.is_val[idx], inj=inj,
        )  # fmt: skip
        tr = d639._select(block=blk, keep=keep)
        w_all = (sel.weight[::20] if smoke else sel.weight)[keep]
        # Test rows: leads from MIN_LEAD_H to 240 hours, finite cleaned power, the six Primaries only.
        ok = np.isin(test.primary, list(six.values())) & (test.lead >= MIN_LEAD_H)
        y_test = clean[test.primary, test.target_idx]
        ok &= np.isfinite(y_test)
        tid = np.flatnonzero(ok)
        tinj = k638.Injector(
            primary=test.primary[tid],
            target_idx=test.target_idx[tid],
            x=test.x[tid],
            standardiser=mr.base.standardiser,
        )
        tchange = (
            tinj.change_points_step(condition=k638.parse(name="step_spring23_mw4"))
            if world == "step"
            else 0.0
        )
        tblk = d639._decorate(
            cutoff=cutoff, world=world, x=test.x[tid], y=y_test[tid].astype(np.float64) + tchange,
            primary=test.primary[tid], target_idx=test.target_idx[tid],
            is_val=np.zeros(len(tid), dtype=bool), inj=tinj,
        )  # fmt: skip
        tkeep = ~tinj.gap
        tblk = d639._select(block=tblk, keep=tkeep)
        tid = tid[tkeep]
        rp = rows_path(month=month, world=world)
        if smoke:
            rp = rp.with_name("smoke_" + rp.name)
        rp.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            rp.with_name(rp.name + ".tmp.npz"), primary=tblk.primary, target_idx=tblk.target_idx, y=tblk.y,
            t_us=tblk.t_us, origin_us=test.origin[tid], lead=test.lead[tid], inc=tblk.inc, u=tblk.u, kt_forecast=tblk.kt,
        )  # fmt: skip
        rp.with_name(rp.name + ".tmp.npz").rename(rp)
        g = {
            "x": torch.from_numpy(tr.x).to(DEVICE), "p": torch.from_numpy(tr.primary).to(DEVICE),
            "is_val": torch.from_numpy(tr.is_val).to(DEVICE), "y": _f32(tr.y), "r4": _f32(tr.r4),
            "after": _f32(tr.after), "u": _f32(tr.u), "u_shuf": _f32(tr.u_shuf), "inc": _f32(tr.inc),
            "w": torch.from_numpy(w_all.astype(np.float32)).to(DEVICE),
        }  # fmt: skip
        for arm in arms:
            name = ARMS[arm]
            if name == "perm":
                params = k639.capacity_parameters(cutoff=cutoff, arm="perm")
            elif name == "free":
                params = k639.capacity_parameters(cutoff=cutoff, arm="r4")
            else:
                params = k639.capacity_parameters(cutoff=cutoff, arm=name)
            tensors = net.parameter_tensors(params=params, six=six, device=DEVICE)
            if name == "free":
                tensors = {a: torch.zeros_like(v) for a, v in tensors.items()}
            shuffled = name == "shuf"
            u_train = (g["u_shuf"] if shuffled else g["u"]) * g["inc"]
            if name == "free":
                u_train = u_train * 0.0
            u_test = (
                _f32((tblk.u_shuf if shuffled else tblk.u) * tblk.inc)
                if name != "free"
                else _f32(tblk.u * 0.0)
            )
            for seed in seeds:
                out = fits_path(cutoff=cutoff, month=month, world=world, arm=arm, seed=seed)
                target = out.with_suffix(".npz")
                if target.exists():
                    continue
                started = time.monotonic()
                torch.manual_seed(seed)
                model = m.ModularMdn(n_primaries=16, stage="ft").to(DEVICE)
                _load_for_finetune(
                    model=model, path=_pretrained(cutoff=cutoff, world=world, arm=arm, seed=seed)
                )
                model.to(DEVICE)
                init = None
                js = _pretrained(cutoff=cutoff, world=world, arm=arm, seed=seed).with_suffix(
                    ".json"
                )
                if name != "free" and js.exists():
                    init = torch.tensor(
                        json.loads(js.read_text())["delta_mw"], dtype=torch.float32, device=DEVICE
                    )
                epochs, best, delta = _train(
                    model=model, x=g["x"], y=g["y"], u_inc=u_train, r4=g["r4"], after=g["after"], p=g["p"], w=g["w"],
                    is_val=g["is_val"], mean=mean, std=std, tensors=tensors, seed=seed, init_delta=init,
                )  # fmt: skip
                p_test = torch.from_numpy(tblk.primary).to(DEVICE)
                cap = net.capacity_mw(
                    t=tensors, delta=delta, p=p_test, r4=_f32(tblk.r4), after=_f32(tblk.after),
                )  # fmt: skip
                q = net.forecast(
                    model=model, x=tblk.x, p=tblk.primary, mean=mean, std=std, subtract=(cap * u_test).cpu(), device=DEVICE,
                )  # fmt: skip
                if smoke:
                    print(
                        f"SMOKE {cutoff} {month} {world} {arm} s{seed}: {epochs} epochs, val NLL {best:.4f}, {time.monotonic() - started:.0f} s, {len(tr.x):,} rows",
                        flush=True,
                    )
                    continue
                out.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), out.with_suffix(".tmp"))
                out.with_suffix(".tmp").rename(out)
                extra = {}
                if name == "free":
                    # Arm e's constant: the mean median error on held-out fine-tuning rows that the ensemble forecast calls sunny.
                    v = (
                        tr.is_val
                        & (tr.kt >= k639.SUNNY_KT)
                        & (tr.hour >= k639.SUNNY_HOURS[0])
                        & (tr.hour <= k639.SUNNY_HOURS[1])
                    )
                    qv = net.forecast(
                        model=model,
                        x=tr.x[v],
                        p=tr.primary[v],
                        mean=mean,
                        std=std,
                        subtract=None,
                        device=DEVICE,
                    )
                    err = qv[:, 5] - tr.y[v]
                    shift = np.zeros(16)
                    for pr in np.unique(tr.primary[v]):
                        shift[pr] = err[tr.primary[v] == pr].mean()
                    extra["shift"] = shift
                tmp = target.with_name(target.name + ".tmp.npz")
                np.savez(tmp, q=q.astype(np.float32), **extra)  # ty: ignore[invalid-argument-type]
                tmp.rename(target)
                out.with_suffix(".json").write_text(
                    json.dumps(
                        {
                            "epochs": epochs,
                            "val_nll": best,
                            "seconds": round(time.monotonic() - started),
                            "rows": len(tr.x),
                            "delta_mw": delta.cpu().tolist(),
                            "train_val_rows": int(tr.is_val.sum()),
                        }
                    )
                )
                print(
                    f"{cutoff} {month} {world} {arm} s{seed}: {epochs} epochs, val NLL {best:.4f}, {time.monotonic() - started:.0f} s, {len(tr.x):,} rows",
                    flush=True,
                )
                del model
        del g
        torch.cuda.empty_cache()


def main() -> None:
    """Run the fine-tunes of one cutoff."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit",))
    parser.add_argument("--cutoff", required=True, choices=CUTOFFS)
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--worlds", nargs="+", default=list(k639.WORLDS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(k639.SEEDS))
    parser.add_argument(
        "--smoke", action="store_true", help="Time one fine-tune on every 20th row; store nothing."
    )
    a = parser.parse_args()
    clean = c.clean_stack()
    for month in MONTHS[a.cutoff]:
        run_month(
            cutoff=a.cutoff,
            month=month,
            arms=a.arms,
            worlds=a.worlds,
            seeds=a.seeds,
            clean=clean,
            smoke=a.smoke,
        )
    print(
        json.dumps({"done": a.phase, "cutoff": a.cutoff, "at": datetime.now(UTC).isoformat()}),
        flush=True,
    )


if __name__ == "__main__":
    main()
