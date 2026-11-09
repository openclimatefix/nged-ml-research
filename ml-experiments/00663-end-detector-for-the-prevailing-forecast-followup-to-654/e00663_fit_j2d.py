# ruff: noqa: ANN001, ANN202, D103, E501, PLR0912, PLR0915
"""Experiment 663, step 4: the end detector on frozen J2 outputs (CPU).

Arms `J2d` (real neighbour graph), `J2d_rw0` to `J2d_rw2` (rewired graphs), and `J2d_noneigh`. The
saved J2 networks of experiment 654 are not changed: `e00663_j2_base.py` stored their NRA
quantiles, depth, spread, and survival on every scored row. The detector (two layers of 64 units
reading the origin-level detector inputs) outputs `q_end`, and the final survival is `p_base * (1 -
q_end)`, used in experiment 647's mixture on the NRA quantiles. Loss: experiment 652's (weighted
pinball of the mixture quantiles plus 0.05 times the survival cross-entropy on in-force rows) plus the
detector's own cross-entropy against "the shift has ended" (`LAMBDA_END`, as in `e00663_fit_e0.py`).

The scored month's detector trains on the other four months' rows, taken from each month's own J2
network (the network whose training excluded that month). Hard negatives enter 30% of the training
and early-stopping origins.

Output `fits_j2/<arm>_<month>_s<seed>.npz`: per set the row index, NRA and prevailing quantiles,
final survival, and `q_end`.

Usage: python e00663_fit_j2d.py [--arms ...] [--months ...] [--seeds 0 1] [--workers 4]
"""

import argparse
import multiprocessing as mp
import time

import e00650_common as c650  # ty: ignore[unresolved-import]
import e00650_fit as f650  # ty: ignore[unresolved-import]
import e00652_fit as fit652  # ty: ignore[unresolved-import]
import e00663_common as c663
import e00663_fit_e0 as fe0
import e00663_panel as pan
import numpy as np
import polars as pl

ARMS = ("J2d", "J2d_rw0", "J2d_rw1", "J2d_rw2", "J2d_noneigh")
OUT = c663.WORK / "fits_j2"
BASE = c663.WORK / "j2_base"


def base_rows(*, month: str, seed: int, setname: str) -> dict[str, np.ndarray]:
    parts = [np.load(BASE / f"J2_{month}_d{d}_s{seed}.npz") for d in (0, 14)]
    idx = np.concatenate([p[f"{setname}_idx"] for p in parts])
    o = np.argsort(idx)
    return {
        k: np.concatenate([p[f"{setname}_{k}"] for p in parts])[o]
        for k in ("idx", "nra", "d", "sigma", "p")
    } | {"idx": idx[o]}


def variant_of(*, arm: str) -> str:
    return {"J2d": "g", "J2d_noneigh": "g"}.get(arm, arm.removeprefix("J2d_"))


def gather(*, fold: str, data_month: str, seed: int, kind: str, z: dict, i: np.ndarray, setname: str,
           variant: str, neg_share: float, rng, noneigh: bool, base: dict | None = None) -> dict[str, np.ndarray]:  # fmt: skip
    """Return the arrays of rows `i` of `z`: base outputs, detector inputs, truth, and weights."""
    b = base if base is not None else base_rows(month=data_month, seed=seed, setname=setname)
    pos = np.searchsorted(b["idx"], i)
    assert np.array_equal(b["idx"][np.clip(pos, 0, len(b["idx"]) - 1)], i), (
        "rows missing from j2_base"
    )
    d = np.load(c663.WORK / "det_inputs_v2" / f"{fold}_{data_month}.npz")
    org = z["org"][i]
    clean = d[f"{kind}_{variant}"][org]
    neg = d[f"{kind}_neg_{variant}"][org] if neg_share > 0 else clean
    uniq, inv = np.unique(org, return_inverse=True)
    use = (rng.random(len(uniq)) < neg_share)[inv] if neg_share > 0 else np.zeros(len(i), bool)
    det = np.where(use[:, None], neg, clean)
    if noneigh:
        det = det.copy()
        det[:, list(pan.NB_COLUMNS)] = 0.0
    return {
        "nra": b["nra"][pos],
        "d": b["d"][pos],
        "sigma": b["sigma"][pos],
        "p": b["p"][pos],
        "det": det.astype(np.float32),
    }


def fit_one(job: tuple[str, str, int]) -> str:
    import e00647_fit as f647  # ty: ignore[unresolved-import]
    import e00647_head as h  # ty: ignore[unresolved-import]
    import e00647_head as h647  # ty: ignore[unresolved-import]
    import torch  # ty: ignore[unresolved-import]
    from torch import nn  # ty: ignore[unresolved-import]

    torch.set_num_threads(1)
    arm_name, month, seed = job
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{arm_name}_{month}_s{seed}.npz"
    if out.exists():
        return f"{out.name}: exists"
    t0 = time.time()
    variant, noneigh = variant_of(arm=arm_name), arm_name == "J2d_noneigh"
    final = month == "final"
    key = 999 if final else int(month.replace("-", "")) % 997
    rng = np.random.default_rng(1000 * seed + key)
    rng_neg = np.random.default_rng(7000 * seed + key)
    tr: list[dict] = []
    va: list[dict] = []
    for train_month in [m for m in c650.MONTHS if m != month]:
        src = f650.sources(month=train_month, bank="new")
        fit652.with_state(z=src["bank"], month=train_month)
        for kind, n_tr, n_va in (
            ("real", f650.N_REAL, f650.N_VAL),
            ("bank", f650.N_BANK_PER_MONTH, f650.N_VAL_BANK // 2),
        ):
            z = src[kind]
            setname = "real" if kind == "real" else "fam"
            for lst, lo, hi, n in ((tr, 0, 18, n_tr), (va, 21, 99, n_va)):
                i = f650.pick(
                    z=z, kind=kind, month=train_month, lo=lo, hi=hi, n=n, rng=rng, train_family=True
                )
                g = gather(fold=month, data_month=train_month, seed=seed, kind=kind, z=z, i=i, setname=setname, variant=variant,
                           neg_share=c663.HARD_NEGATIVE_SHARE, rng=rng_neg, noneigh=noneigh)  # fmt: skip
                w, lab, inf, _prim = fit652.extras(z=z, idx=i, kind=kind, mixture="m50")
                y = z["target"][i] if "target" in z else z["y"][i]
                el, em = fe0.end_labels(z=z, idx=i, kind=kind)
                g.update(
                    w=w,
                    lab=lab,
                    inf=inf,
                    y=y.astype(np.float32),
                    lead=z["lead"][i].astype(np.float32),
                    el=el,
                    em=em,
                )
                lst.append(g)

    def cat(parts: list[dict]) -> dict[str, np.ndarray]:
        return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}

    train, val = cat(tr), cat(va)
    mu, sd = train["det"].mean(0), np.maximum(train["det"].std(0), 1e-3)

    def tens(d: dict) -> dict:
        t = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in d.items()}
        t["det"] = torch.from_numpy(np.clip((d["det"] - mu) / sd, -10, 10).astype(np.float32))
        return t

    ttr, tva = tens(train), tens(val)
    torch.manual_seed(seed)
    det_net = nn.Sequential(
        nn.Linear(train["det"].shape[1], 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )
    with torch.no_grad():
        det_net[-1].weight.mul_(0.1)
        det_net[-1].bias.fill_(-3.0)
    opt = torch.optim.Adam(det_net.parameters(), lr=f647.LR, weight_decay=f647.WD)

    def forward(t: dict, ix) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qe = torch.sigmoid(det_net(t["det"][ix])[:, 0])
        p = t["p"][ix] * (1.0 - qe)
        q = h647.mixture_quantiles(q=t["nra"][ix], d=t["d"][ix], sigma=t["sigma"][ix], p=p)
        return q, p, qe

    def loss_of(t: dict, ix):
        q, p, qe = forward(t, ix)
        wt = t["w"][ix]
        pin = (h.pinball(y=t["y"][ix], q=q) * wt).sum() / wt.sum()
        m = t["inf"][ix]
        bce = nn.functional.binary_cross_entropy(
            torch.clamp(p, 1e-4, 1 - 1e-4), t["lab"][ix], reduction="none"
        )
        bce = (bce * m * wt).sum() / torch.clamp((m * wt).sum(), min=1e-6)
        wm = t["em"][ix] * wt
        aux = nn.functional.binary_cross_entropy(
            torch.clamp(qe, 1e-4, 1 - 1e-4), t["el"][ix], reduction="none"
        )
        return (
            pin
            + fit652.LAMBDA * bce
            + fe0.LAMBDA_END * (aux * wm).sum() / torch.clamp(wm.sum(), min=1e-6)
        )

    best, best_state, bad, log = np.inf, None, 0, []
    n = len(train["y"])
    for epoch in range(f647.EPOCHS):
        det_net.train()
        perm = torch.from_numpy(np.random.default_rng(seed * 100 + epoch).permutation(n))
        for a in range(0, n, f647.BATCH):
            loss = loss_of(ttr, perm[a : a + f647.BATCH])
            opt.zero_grad()
            loss.backward()
            opt.step()
        det_net.eval()
        with torch.no_grad():
            vl = sum(
                float(loss_of(tva, torch.arange(a, min(a + 20000, len(val["y"])))))
                * min(20000, len(val["y"]) - a)
                for a in range(0, len(val["y"]), 20000)
            ) / len(val["y"])
        log.append(vl)
        if vl < best - 1e-7:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in det_net.state_dict().items()}
        else:
            bad += 1
            if bad >= f647.PATIENCE:
                break
    assert best_state is not None
    det_net.load_state_dict(best_state)
    det_net.eval()
    res: dict[str, np.ndarray] = {}
    if (
        final
    ):  # the window: the frozen J2 networks' forward pass (`e00663_j2_final.py`) on the window rows
        zw = f650.load(path=c663.WORK / "window" / "real_window.npz", kind="real")
        idx_w = np.flatnonzero(zw["usable"])
        rows = pl.read_parquet(c663.WORK / "j2_final" / "forecasts" / f"J2_s{seed}.parquet")
        key_rows = (
            rows["primary"].to_numpy().astype(np.int64) * 4_000_000 + rows["origin_idx"].to_numpy()
        ) * 1024 + (rows["target_idx"].to_numpy() - rows["origin_idx"].to_numpy())
        key_w = (zw["prim"][idx_w].astype(np.int64) * 4_000_000 + zw["oi"][idx_w]) * 1024 + (
            zw["vi"][idx_w] - zw["oi"][idx_w]
        )
        o = np.argsort(key_rows)
        pos = np.searchsorted(key_rows[o], key_w)
        assert np.array_equal(key_rows[o][np.clip(pos, 0, len(o) - 1)], key_w), (
            "window rows missing from the J2 forward pass"
        )
        sel = o[pos]
        win_base = {
            "idx": idx_w,
            "nra": rows.select([f"nra_{i}" for i in range(11)]).to_numpy()[sel],
            "d": rows["depth"].to_numpy()[sel],
            "sigma": rows["spread"].to_numpy()[sel],
            "p": rows["survival_probability"].to_numpy()[sel],
        }
        sets = {"window": ("real", zw, idx_w)}
    else:
        test = f650.sources(month=month, bank="new")
        sets = {
            "real": ("real", test["real"], np.flatnonzero(test["real"]["usable"])),
            "held": (
                "bank",
                test["bank"],
                np.flatnonzero(test["bank"]["usable"] & (test["bank"]["pass_"] >= 200)),
            ),
            "fam": (
                "bank",
                test["bank"],
                np.flatnonzero(test["bank"]["usable"] & (test["bank"]["pass_"] < 200)),
            ),
        }
    for name, (kind, z, idx) in sets.items():
        g = gather(
            fold=month,
            data_month="window" if final else month,
            seed=seed,
            kind=kind,
            z=z,
            i=idx,
            setname=name,
            variant=variant,
            neg_share=0.0,
            rng=None,
            noneigh=noneigh,
            base=win_base if final else None,
        )
        t = tens(g)
        qs, ps, es = [], [], []
        with torch.no_grad():
            for a in range(0, len(idx), 50000):
                q, p, qe = forward(t, torch.arange(a, min(a + 50000, len(idx))))
                qs.append(q.numpy())
                ps.append(p.numpy())
                es.append(qe.numpy())
        res[f"{name}_idx"] = idx.astype(np.int32)
        res[f"{name}_nra"] = g["nra"]
        res[f"{name}_q"] = np.concatenate(qs).astype(np.float32)
        res[f"{name}_p"] = np.concatenate(ps).astype(np.float32)
        res[f"{name}_q_end"] = np.concatenate(es).astype(np.float32)
    np.savez(
        out.with_suffix(".tmp.npz"),
        **res,
        val_log=np.array(log),
        n_train=n,
        best=best,
        seconds=time.time() - t0,
    )  # ty: ignore[invalid-argument-type]
    out.with_suffix(".tmp.npz").rename(out)
    torch.save({"det": det_net.state_dict(), "mu": mu, "sd": sd}, OUT / f"{out.stem}.pt")
    return f"{out.name}: {len(log)} epochs, val {best:.5f}, {time.time() - t0:.0f}s"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--months", nargs="+", default=list(c663.MONTHS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    jobs = [(arm, m, s) for m in a.months for arm in a.arms for s in a.seeds]
    print(f"{len(jobs)} fits", flush=True)
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for msg in pool.imap_unordered(fit_one, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
