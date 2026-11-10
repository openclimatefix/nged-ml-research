# ruff: noqa: ANN001, D103, PLR0912, PLR0915, E501
"""Experiment 687, step 2: fit the hazard head of experiment 652 with detector features or extra lags (CPU).

Arms (`ARMS`): `B0` (experiment 652's R-haz, refitted), `B0x` (B0 with L1's age rule), `B0g` (B0 with F1's age rule and no extra columns),
`L1` (experiment 681's lag block), `F1` (features D1 to D6 at the day lag, week lag, and the 6-hour
cutoff), `F1d46` (D4 to D6 at the 6-hour cutoff only), `F2` (F1 plus the explicit flag), `P` (F1 with
every feature from an unrelated Primary), `F1h4` (F1 with the 4-hour cutoff), and `F1nd` (F1 without
dropout). Training data, folds, optimiser, loss, and early stopping are those of experiment 652's
`e00652_fit.py`; the extra inputs are appended to the origin-level raw inputs, and the age input
(which gates the survival probability) is the larger of the day-lag age and the arm's own newest-reading age.

Output per fit, `fits/<arm>_<month>_s<seed>.npz` (same keys as experiment 652) and `fits/<...>.pt`.

Usage: python e00687_fit.py [--arms ...] [--months ...] [--seeds 0 1] [--workers 4]
"""

import argparse
import multiprocessing as mp
import time

import e00650_common as c650  # ty: ignore[unresolved-import]
import e00650_fit as f650  # ty: ignore[unresolved-import]
import e00652_common as c652  # ty: ignore[unresolved-import]
import e00652_fit as f652  # ty: ignore[unresolved-import]
import e00687_features as feats
import e00687_separability as sep
import e00687_tables as t
import numpy as np

LAMBDA = f652.LAMBDA
OUT = t.WORK / "fits"
N_FEAT = 58
DROPOUT = 0.5
LAG_AGE_THRESHOLD = 0.05
FN = feats.FEATURE_NAMES
D46_COLS = [FN.index(n) for n in ("step_12", "step_48", "step_144", "cstep_12", "cstep_48", "cstep_144", "cu_depth", "cu_age", "cu_ratio", "cu_ret")]  # fmt: skip
ARMS = ("B0", "B0x", "B0g", "L1", "F1", "F1d46", "F2", "P", "F1h4", "F1nd")
NEW_CUT = {"F1h4": 2}  # cutoff index: 0 day, 1 week, 2 new4, 3 new6; default 3
_PMAP: np.ndarray | None = None
_TABLES: dict = {}


def tables(*, month: str, kind: str) -> dict:
    key = (month, kind)
    if key not in _TABLES:
        z = dict(np.load(t.WORK / "feats" / f"{month}_{'M' if kind == 'bank' else 'real'}.npz"))
        z["index"] = {(int(p), int(o), int(r)): i for i, (p, o, r) in enumerate(zip(z["prim"], z["oi"], z["pass_"], strict=True))}  # fmt: skip
        _TABLES[key] = z
    return _TABLES[key]


def extra_block(*, arm: str, z: dict, month: str, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (extra columns (n_org, k), newest-reading age in days (n_org,)) for an arm, in origin order."""
    global _PMAP  # noqa: PLW0603
    tb = tables(month=month, kind=kind)
    pas = z["org_pass"] if kind == "bank" else np.zeros(len(z["org_oi"]), np.int16)
    rows = np.array([tb["index"][(int(p), int(o), int(r))] for p, o, r in zip(z["org_prim"], z["org_oi"], pas, strict=True)])  # fmt: skip
    n = len(rows)
    new = NEW_CUT.get(arm, 3)
    if arm in ("L1", "B0x"):
        lag, miss = tb["lags"][rows], tb["lags_miss"][rows]
        mean = lag[:, :5].mean(axis=1) / feats_lag_scale()
        age_new = np.where((np.abs(mean) > LAG_AGE_THRESHOLD) & ~miss[:, :5].any(axis=1), 1.0, 0.0).astype(np.float32)
        extra = np.concatenate([lag, miss.astype(np.float32)], axis=1) if arm == "L1" else np.zeros((n, 0), np.float32)  # fmt: skip
        return extra.astype(np.float32), age_new
    if arm == "B0":
        return np.zeros((n, 0), np.float32), np.zeros(n, np.float32)
    gate_only = arm == "B0g"  # the F1 age rule with no extra columns
    if arm == "P":
        if _PMAP is None:
            _PMAP = sep.placebo_map()
        rows = np.array([tb["index"][(int(_PMAP[p]), int(o), int(r))] for p, o, r in zip(z["org_prim"], z["org_oi"], pas, strict=True)])  # fmt: skip
    ft, fl = tb["feat"][rows], tb["flag"][rows]
    if gate_only:
        new = 3
        age_new = np.where(fl[:, new], ft[:, new, FN.index("cu_age")] * 6.0, 0.0).astype(np.float32)
        return np.zeros((n, 0), np.float32), age_new
    if arm == "F1d46":
        extra = ft[:, new, :][:, D46_COLS]
    else:
        extra = np.concatenate([ft[:, 0], ft[:, 1], ft[:, new]], axis=1)
        if arm == "F2":
            extra = np.concatenate([extra, fl[:, [0, 1, new]].astype(np.float32)], axis=1)
    cu_age_days = ft[:, new, FN.index("cu_age")] * 6.0
    age_new = np.where(fl[:, new], cu_age_days, 0.0).astype(np.float32)
    return extra.astype(np.float32), age_new


def feats_lag_scale() -> float:
    return t.LAG_SCALE


def sources(*, month: str, arm: str) -> dict:
    """Return the real and bank arrays of `month` with the arm's extra inputs and gate age appended."""
    src = f650.sources(month=month, bank="new")
    out = {}
    for kind in ("real", "bank"):
        z = dict(src[kind])
        extra, age_new = extra_block(arm=arm, z=z, month=month, kind=kind)
        raw = z["raw"].copy()
        if arm != "B0":
            age = np.maximum(z["age"].astype(np.float32), age_new)
            raw[:, N_FEAT] = np.log1p(age)
            raw[:, N_FEAT + 1] = age / 30.0
        z["raw"] = np.concatenate([raw, extra], axis=1)
        z["n_extra"] = extra.shape[1]
        out[kind] = z
    return out


def fit_one(job: tuple[str, str, int]) -> str:
    import e00647_fit as f647  # ty: ignore[unresolved-import]
    import e00647_head as h  # ty: ignore[unresolved-import]
    import e00647_head as h647  # ty: ignore[unresolved-import]
    import e00652_head as h652  # ty: ignore[unresolved-import]
    import torch  # ty: ignore[unresolved-import]
    from torch import nn  # ty: ignore[unresolved-import]

    torch.set_num_threads(1)
    arm_name, month, seed = job
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{arm_name}_{month}_s{seed}.npz"
    if out.exists():
        return f"{out.name}: exists"
    t0 = time.time()
    rng = np.random.default_rng(1000 * seed + int(month.replace("-", "")) % 997)
    tr, va, ex_tr, ex_va = [], [], [], []
    for train_month in [m for m in c650.MONTHS if m != month]:
        src = sources(month=train_month, arm=arm_name)
        f652.with_state(z=src["bank"], month=train_month)
        for kind, n_tr, n_va in (("real", f650.N_REAL, f650.N_VAL), ("bank", f650.N_BANK_PER_MONTH, f650.N_VAL_BANK // 2)):  # fmt: skip
            z = src[kind]
            i_tr = f650.pick(z=z, kind=kind, month=train_month, lo=0, hi=18, n=n_tr, rng=rng, train_family=True)  # fmt: skip
            i_va = f650.pick(z=z, kind=kind, month=train_month, lo=21, hi=99, n=n_va, rng=rng, train_family=True)  # fmt: skip
            tr.append((z, i_tr, None))
            va.append((z, i_va, None))
            ex_tr.append(f652.extras(z=z, idx=i_tr, kind=kind, mixture="m50"))
            ex_va.append(f652.extras(z=z, idx=i_va, kind=kind, mixture="m50"))
    train, val = f650.Pool(parts=tr), f650.Pool(parts=va)

    def stack(ex: list) -> tuple:
        return tuple(np.concatenate([e[k] for e in ex]) for k in range(4))

    w_tr, lab_tr, inf_tr, _ = stack(ex_tr)
    w_va, lab_va, inf_va, _ = stack(ex_va)
    arm = {"kind": "mix", "inputs": "all", "short": 0.5}
    x_tr, age_tr, stats = f647.make_inputs(raw=train.raw, arm=arm, stats=None, seed=seed)
    x_va, age_va, _ = f647.make_inputs(raw=val.raw, arm=arm, stats=stats, seed=seed + 1)
    n_extra = tr[0][0]["n_extra"]
    drop = n_extra > 0 and arm_name not in ("B0", "B0x", "B0g", "F1nd")
    first_extra = N_FEAT + 4
    torch.manual_seed(seed)
    head = h652.HazardHead(n_in=x_tr.shape[1] + 3)
    opt = torch.optim.Adam(head.parameters(), lr=f647.LR, weight_decay=f647.WD)
    gen = torch.Generator().manual_seed(seed)

    def loss_of(*, pool, x_org, age_org, ix, w, lab, inf, training) -> torch.Tensor:
        x, q, age, lead, y = f647.batch_inputs(x_org=x_org, age_org=age_org, pool=pool, ix=ix)
        if drop and training:
            keep = (torch.rand(len(ix), 1, generator=gen) >= DROPOUT).to(x.dtype)
            x = x.clone()
            x[:, first_extra : first_extra + n_extra] *= keep
        qq, p = head(x=x, q=q, age=age, lead_h=lead)
        wt = torch.from_numpy(w[ix])
        pin = (h.pinball(y=y, q=qq) * wt).sum() / wt.sum()
        m = torch.from_numpy(inf[ix]) * (age >= h647.AGE_FLOOR_DAYS).to(torch.float32)
        bce = nn.functional.binary_cross_entropy(torch.clamp(p, 1e-4, 1 - 1e-4), torch.from_numpy(lab[ix]), reduction="none")  # fmt: skip
        bce = (bce * m * wt).sum() / torch.clamp((m * wt).sum(), min=1e-6)
        return pin + LAMBDA * bce

    best, best_state, bad, log = np.inf, None, 0, []
    n = len(train.y)
    for epoch in range(f647.EPOCHS):
        head.train()
        perm = np.random.default_rng(seed * 100 + epoch).permutation(n)
        for a in range(0, n, f647.BATCH):
            ix = perm[a : a + f647.BATCH]
            loss = loss_of(pool=train, x_org=x_tr, age_org=age_tr, ix=ix, w=w_tr, lab=lab_tr, inf=inf_tr, training=True)  # fmt: skip
            opt.zero_grad()
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            vl = 0.0
            for a in range(0, len(val.y), 20000):
                ix = np.arange(a, min(a + 20000, len(val.y)))
                vl += float(loss_of(pool=val, x_org=x_va, age_org=age_va, ix=ix, w=w_va, lab=lab_va, inf=inf_va, training=False)) * len(ix)  # fmt: skip
            vl /= len(val.y)
        log.append(vl)
        if vl < best - 1e-7:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
        else:
            bad += 1
            if bad >= f647.PATIENCE:
                break
    assert best_state is not None
    head.load_state_dict(best_state)
    head.eval()
    test = sources(month=month, arm=arm_name)
    res: dict[str, np.ndarray] = {}
    sets = {
        "real": (test["real"], np.flatnonzero(test["real"]["usable"])),
        "held": (test["bank"], np.flatnonzero(test["bank"]["usable"] & (test["bank"]["pass_"] >= 200))),
        "fam": (test["bank"], np.flatnonzero(test["bank"]["usable"] & (test["bank"]["pass_"] < 200))),
    }  # fmt: skip
    for name, (z, idx) in sets.items():
        pool = f650.Pool(parts=[(z, idx, None)])
        x_te, age_te, _ = f647.make_inputs(raw=pool.raw, arm=arm, stats=stats, seed=seed + 3)
        qs, ps = [], []
        with torch.no_grad():
            for a in range(0, len(idx), 50000):
                ix = np.arange(a, min(a + 50000, len(idx)))
                x, q, age, lead, _y = f647.batch_inputs(x_org=x_te, age_org=age_te, pool=pool, ix=ix)
                qq, p = head(x=x, q=q, age=age, lead_h=lead)
                qs.append(qq.numpy())
                ps.append(p.numpy())
        res[f"{name}_idx"] = idx.astype(np.int32)
        res[f"{name}_q"] = np.concatenate(qs).astype(np.float32)
        res[f"{name}_p"] = np.concatenate(ps).astype(np.float32)
    torch.save(best_state, OUT / f"{arm_name}_{month}_s{seed}.pt")
    np.savez(out, **res, val_log=np.array(log), n_train=n, best=best, seconds=time.time() - t0)  # ty: ignore[invalid-argument-type]
    return f"{out.name}: {len(log)} epochs, val {best:.5f}, {time.time() - t0:.0f}s"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--months", nargs="+", default=list(c650.MONTHS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(c652.SEEDS))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    jobs = [(arm, m, s) for m in a.months for arm in a.arms for s in a.seeds]
    print(f"{len(jobs)} fits", flush=True)
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for msg in pool.imap_unordered(fit_one, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
