# ruff: noqa: E501, ANN001, D103, PLR0912, PLR0915
"""Experiment 652, step 1: fit R-haz and its placebo, one head per scored month (CPU).

Arms: `Rhaz_<mixture>` (hazard head, rows weighted to the duration mixture) for the mixtures
`m50`, `mshort`, `mlong`, and `Placebo` (the `m50` head refitted with each bank row's target,
survival label, in-force flag, and weight permuted among the bank rows of the same Primary).
Training data, early stopping, optimiser, batch, and epochs are those of experiment 650's
`e00650_fit.py` (imported), with the scored month held out and the other four development months
used for training. Loss: weighted pinball of the mixture quantiles plus `LAMBDA` times the binary
cross-entropy of the survival probability against the still-on label on in-force bank rows.

Output per fit, `fits/<arm>_<month>_s<seed>.npz`: the head's quantiles and survival probability on
the scored month's real, held-out, and training-family rows (the index sets of experiment 650).

Usage: python e00652_fit.py [--arms ...] [--months ...] [--seeds 0 1] [--workers 8]
"""

import argparse
import multiprocessing as mp
import time

import e00650_common as c650  # ty: ignore[unresolved-import]
import e00650_fit as f650  # ty: ignore[unresolved-import]
import e00652_common as c652  # ty: ignore[unresolved-import]
import numpy as np

LAMBDA = 0.05
ARMS = ("Rhaz_m50", "Rhaz_mshort", "Rhaz_mlong", "Placebo")
OUT = c652.WORK / "fits"


def mixture_of(*, arm: str) -> str:
    return "m50" if arm == "Placebo" else arm.removeprefix("Rhaz_")


def with_state(*, z: dict, month: str) -> dict:
    """Add the shift state of a bank file (once) and the Primary of each row."""
    if "stratum" not in z:
        z.update(c652.row_state(z=z, month=month))
    return z


def extras(*, z: dict, idx: np.ndarray, kind: str, mixture: str) -> tuple[np.ndarray, ...]:
    """Return (weight, label, in-force, Primary) of the selected rows; real rows have unit weight and no label."""
    n = len(idx)
    if kind == "real":
        return (
            np.ones(n, np.float32),
            np.zeros(n, np.float32),
            np.zeros(n, np.float32),
            z["prim"][idx].astype(np.int64),
        )
    w = (c652.class_weight(short=z["short"][idx], mixture=mixture) * z["event_w"][idx]).astype(
        np.float32
    )
    return (
        w,
        z["still_on"][idx].astype(np.float32),
        (z["in_force"][idx] & ~z["censored"][idx]).astype(np.float32),
        z["prim"][idx].astype(np.int64),
    )


def fit_one(job: tuple[str, str, int]) -> str:
    import e00647_fit as f647  # ty: ignore[unresolved-import]
    import e00647_head as h  # ty: ignore[unresolved-import]
    import e00647_head as h647  # ty: ignore[unresolved-import]
    import e00652_head as h652  # ty: ignore[unresolved-import]
    import torch  # ty: ignore[unresolved-import]

    torch.set_num_threads(1)
    arm_name, month, seed = job
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{arm_name}_{month}_s{seed}.npz"
    if out.exists():
        return f"{out.name}: exists"
    t0 = time.time()
    mixture = mixture_of(arm=arm_name)
    rng = np.random.default_rng(1000 * seed + int(month.replace("-", "")) % 997)
    tr, va, ex_tr, ex_va = [], [], [], []
    for train_month in [m for m in c650.MONTHS if m != month]:
        src = f650.sources(month=train_month, bank="new")
        with_state(z=src["bank"], month=train_month)
        for kind, n_tr, n_va in (
            ("real", f650.N_REAL, f650.N_VAL),
            ("bank", f650.N_BANK_PER_MONTH, f650.N_VAL_BANK // 2),
        ):
            z = src[kind]
            i_tr = f650.pick(
                z=z, kind=kind, month=train_month, lo=0, hi=18, n=n_tr, rng=rng, train_family=True
            )
            i_va = f650.pick(
                z=z, kind=kind, month=train_month, lo=21, hi=99, n=n_va, rng=rng, train_family=True
            )
            tr.append((z, i_tr, None))
            va.append((z, i_va, None))
            ex_tr.append(extras(z=z, idx=i_tr, kind=kind, mixture=mixture))
            ex_va.append(extras(z=z, idx=i_va, kind=kind, mixture=mixture))
    train, val = f650.Pool(parts=tr), f650.Pool(parts=va)

    def stack(ex: list) -> tuple:
        return tuple(np.concatenate([e[k] for e in ex]) for k in range(4))

    w_tr, lab_tr, inf_tr, prim_tr = stack(ex_tr)
    w_va, lab_va, inf_va, prim_va = stack(ex_va)
    if (
        arm_name == "Placebo"
    ):  # permute target, label, flag, and weight among the bank rows of the same Primary
        for pool, w, lab, inf, prim, is_bank in (
            (
                train,
                w_tr,
                lab_tr,
                inf_tr,
                prim_tr,
                np.concatenate([np.full(len(e[0]), k % 2 == 1) for k, e in enumerate(ex_tr)]),
            ),
            (
                val,
                w_va,
                lab_va,
                inf_va,
                prim_va,
                np.concatenate([np.full(len(e[0]), k % 2 == 1) for k, e in enumerate(ex_va)]),
            ),
        ):
            for p in np.unique(prim[is_bank]):
                rows = np.flatnonzero(is_bank & (prim == p))
                perm = rows[rng.permutation(len(rows))]
                pool.y[rows], w[rows], lab[rows], inf[rows] = (
                    pool.y[perm],
                    w[perm].copy(),
                    lab[perm].copy(),
                    inf[perm].copy(),
                )
    arm = {"kind": "mix", "inputs": "all", "short": 0.5}
    x_tr, age_tr, stats = f647.make_inputs(raw=train.raw, arm=arm, stats=None, seed=seed)
    x_va, age_va, _ = f647.make_inputs(raw=val.raw, arm=arm, stats=stats, seed=seed + 1)
    torch.manual_seed(seed)
    head = h652.HazardHead(n_in=x_tr.shape[1] + 3)
    opt = torch.optim.Adam(head.parameters(), lr=f647.LR, weight_decay=f647.WD)

    def loss_of(*, pool, x_org, age_org, ix, w, lab, inf) -> torch.Tensor:
        x, q, age, lead, y = f647.batch_inputs(x_org=x_org, age_org=age_org, pool=pool, ix=ix)
        qq, p = head(x=x, q=q, age=age, lead_h=lead)
        wt = torch.from_numpy(w[ix])
        pin = (h.pinball(y=y, q=qq) * wt).sum() / wt.sum()
        m = torch.from_numpy(inf[ix]) * (age >= h647.AGE_FLOOR_DAYS).to(torch.float32)
        bce = nn.functional.binary_cross_entropy(
            torch.clamp(p, 1e-4, 1 - 1e-4), torch.from_numpy(lab[ix]), reduction="none"
        )
        bce = (bce * m * wt).sum() / torch.clamp((m * wt).sum(), min=1e-6)
        return pin + LAMBDA * bce

    from torch import nn  # ty: ignore[unresolved-import]

    best, best_state, bad, log = np.inf, None, 0, []
    n = len(train.y)
    for epoch in range(f647.EPOCHS):
        head.train()
        perm = np.random.default_rng(seed * 100 + epoch).permutation(n)
        for a in range(0, n, f647.BATCH):
            ix = perm[a : a + f647.BATCH]
            loss = loss_of(
                pool=train, x_org=x_tr, age_org=age_tr, ix=ix, w=w_tr, lab=lab_tr, inf=inf_tr
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            vl = 0.0
            for a in range(0, len(val.y), 20000):
                ix = np.arange(a, min(a + 20000, len(val.y)))
                vl += float(
                    loss_of(
                        pool=val, x_org=x_va, age_org=age_va, ix=ix, w=w_va, lab=lab_va, inf=inf_va
                    )
                ) * len(ix)
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
    test = f650.sources(month=month, bank="new")
    res: dict[str, np.ndarray] = {}
    sets = {
        "real": (test["real"], np.flatnonzero(test["real"]["usable"])),
        "held": (
            test["bank"],
            np.flatnonzero(test["bank"]["usable"] & (test["bank"]["pass_"] >= 200)),
        ),
        "fam": (
            test["bank"],
            np.flatnonzero(test["bank"]["usable"] & (test["bank"]["pass_"] < 200)),
        ),
    }
    for name, (z, idx) in sets.items():
        pool = f650.Pool(parts=[(z, idx, None)])
        x_te, age_te, _ = f647.make_inputs(raw=pool.raw, arm=arm, stats=stats, seed=seed + 3)
        qs, ps = [], []
        with torch.no_grad():
            for a in range(0, len(idx), 50000):
                ix = np.arange(a, min(a + 50000, len(idx)))
                x, q, age, lead, _y = f647.batch_inputs(
                    x_org=x_te, age_org=age_te, pool=pool, ix=ix
                )
                qq, p = head(x=x, q=q, age=age, lead_h=lead)
                qs.append(qq.numpy())
                ps.append(p.numpy())
        res[f"{name}_idx"] = idx.astype(np.int32)
        res[f"{name}_q"] = np.concatenate(qs).astype(np.float32)
        res[f"{name}_p"] = np.concatenate(ps).astype(np.float32)
    np.savez(out, **res, val_log=np.array(log), n_train=n, best=best, seconds=time.time() - t0)  # ty: ignore[invalid-argument-type]
    return f"{out.name}: {len(log)} epochs, val {best:.5f}, {time.time() - t0:.0f}s"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--months", nargs="+", default=list(c650.MONTHS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(c652.SEEDS))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    jobs = [(arm, m, s) for m in a.months for arm in a.arms for s in a.seeds]
    print(f"{len(jobs)} fits", flush=True)
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for msg in pool.imap_unordered(fit_one, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
