# ruff: noqa: ANN001, ANN202, D103, E501, PLR0912, PLR0915
"""Experiment 663, step 2: refit E0 (R-haz of experiment 652) and its detector arms (CPU).

Arms: `E0` (experiment 652's `Rhaz_m50` recipe and data selection, refitted so that the head's depth,
spread, and survival are kept), `E0d` (plus the end detector on the real neighbour graph),
`E0d_rw0` to `E0d_rw2` (rewired graphs), and `E0d_noneigh` (neighbour inputs zeroed). The detector
is a two-layer network with 64 units whose output `q_end` multiplies the survival: `p = p_haz * (1 -
q_end)`. The loss is experiment 652's (weighted pinball of the mixture quantiles plus 0.05 times the
survival cross-entropy on in-force rows) plus `LAMBDA_END` times the cross-entropy of `q_end`
against "the shift has ended" on in-force and just-ended origins of the bank. Hard negatives enter
the detector inputs of 30% of the training and early-stopping origins. The scored month's rows never enter training.

Output `fits_e0/<arm>_<month>_s<seed>.npz`: per set (`real`, `held`, `fam`) the row index, the
prevailing quantiles, the final survival, the head's depth and spread, the base survival, and `q_end`.

Usage: python e00663_fit_e0.py [--arms ...] [--months ...] [--seeds 0 1] [--workers 4]
"""

import argparse
import multiprocessing as mp
import time

import e00650_common as c650  # ty: ignore[unresolved-import]
import e00650_fit as f650  # ty: ignore[unresolved-import]
import e00652_common as c652  # ty: ignore[unresolved-import]
import e00652_fit as fit652  # ty: ignore[unresolved-import]
import e00663_common as c663
import e00663_panel as pan
import numpy as np

ARMS = ("E0", "E0d", "E0d_rw0", "E0d_rw1", "E0d_rw2", "E0d_noneigh")
OUT = c663.WORK / "fits_e0"


LAMBDA_END = 0.5
"""Weight of the detector's own cross-entropy: `q_end` against "the shift has ended" on in-force and just-ended origins."""


def end_labels(*, z: dict, idx: np.ndarray, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (ended label, mask): 1 on rows whose origin lies after the shift's end (within 7 days), 0 in force."""
    if kind == "real":
        return np.zeros(len(idx), np.float32), np.zeros(len(idx), np.float32)
    st = z["stratum"][idx]
    ended = st == "just_ended"
    in_force = np.isin(st, ["far", "middle", "near"])
    return ended.astype(np.float32), (ended | in_force).astype(np.float32)


def variant_of(*, arm: str) -> str | None:
    if arm == "E0":
        return None
    return {"E0d": "g", "E0d_noneigh": "g"}.get(arm, arm.removeprefix("E0d_"))


def det_part(*, fold: str, data_month: str, kind: str, variant: str, z: dict, idx: np.ndarray,
             neg_share: float, rng, noneigh: bool) -> np.ndarray:  # fmt: skip
    """Return the detector inputs of the origins of the part (`np.unique` order, as the pool)."""
    d = np.load(c663.WORK / "det_inputs_v2" / f"{fold}_{data_month}.npz")
    uniq = np.unique(z["org"][idx])
    clean = d[f"{kind}_{variant}"][uniq]
    if neg_share > 0:
        neg = d[f"{kind}_neg_{variant}"][uniq]
        use = rng.random(len(uniq)) < neg_share
        clean = np.where(use[:, None], neg, clean)
    if noneigh:
        clean = clean.copy()
        clean[:, list(pan.NB_COLUMNS)] = 0.0
    return clean


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
    variant = variant_of(arm=arm_name)
    noneigh = arm_name == "E0d_noneigh"
    final = month == "final"
    key = 999 if final else int(month.replace("-", "")) % 997
    rng = np.random.default_rng(1000 * seed + key)
    rng_neg = np.random.default_rng(7000 * seed + key)
    tr, va, ex_tr, ex_va, det_tr, det_va, en_tr, en_va = [], [], [], [], [], [], [], []
    for train_month in [m for m in c650.MONTHS if m != month]:  # `final`: all five months
        src = f650.sources(month=train_month, bank="new")
        fit652.with_state(z=src["bank"], month=train_month)
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
            ex_tr.append(fit652.extras(z=z, idx=i_tr, kind=kind, mixture="m50"))
            ex_va.append(fit652.extras(z=z, idx=i_va, kind=kind, mixture="m50"))
            en_tr.append(end_labels(z=z, idx=i_tr, kind=kind))
            en_va.append(end_labels(z=z, idx=i_va, kind=kind))
            if variant is not None:
                for lst, idx in ((det_tr, i_tr), (det_va, i_va)):
                    lst.append(det_part(fold=month, data_month=train_month, kind=kind, variant=variant, z=z, idx=idx,
                                        neg_share=c663.HARD_NEGATIVE_SHARE, rng=rng_neg, noneigh=noneigh))  # fmt: skip
    train, val = f650.Pool(parts=tr), f650.Pool(parts=va)

    def stack(ex: list) -> tuple:
        return tuple(np.concatenate([e[k] for e in ex]) for k in range(4))

    w_tr, lab_tr, inf_tr, _ = stack(ex_tr)
    w_va, lab_va, inf_va, _ = stack(ex_va)
    el_tr, em_tr = (np.concatenate([e[k] for e in en_tr]) for k in (0, 1))
    el_va, em_va = (np.concatenate([e[k] for e in en_va]) for k in (0, 1))
    arm = {"kind": "mix", "inputs": "all", "short": 0.5}
    x_tr, age_tr, stats = f647.make_inputs(raw=train.raw, arm=arm, stats=None, seed=seed)
    x_va, age_va, _ = f647.make_inputs(raw=val.raw, arm=arm, stats=stats, seed=seed + 1)
    dx_tr = dx_va = dstats = None
    if variant is not None:
        dtr, dva = np.concatenate(det_tr), np.concatenate(det_va)
        dstats = (dtr.mean(0), np.maximum(dtr.std(0), 1e-3))
        dx_tr = np.clip((dtr - dstats[0]) / dstats[1], -10, 10).astype(np.float32)
        dx_va = np.clip((dva - dstats[0]) / dstats[1], -10, 10).astype(np.float32)
    torch.manual_seed(seed)

    class Head(h652.HazardHead):
        def __init__(self, *, n_in: int, n_det: int) -> None:
            super().__init__(n_in=n_in)
            self.det = None
            if n_det:
                self.det = nn.Sequential(
                    nn.Linear(n_det, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1)
                )
                with torch.no_grad():
                    self.det[-1].weight.mul_(0.1)
                    self.det[-1].bias.fill_(-3.0)

        def forward(self, *, x, q, age, lead_h, det=None):  # ty: ignore[invalid-method-override]
            o = self.net(x)
            d = h647.DEPTH_SCALE * torch.tanh(o[:, 0])
            sigma = h647.SIGMA_FLOOR + h647.SIGMA_SCALE * nn.functional.softplus(o[:, 1])
            p_haz = h652.survival(hazard_logits=o[:, 2:], lead_h=lead_h, age_days=age)
            qe = (
                torch.sigmoid(self.det(det)[:, 0])
                if (self.det is not None and det is not None)
                else torch.zeros_like(p_haz)
            )
            p = p_haz * (1.0 - qe)
            return h647.mixture_quantiles(q=q, d=d, sigma=sigma, p=p), p, d, sigma, p_haz, qe

    head = Head(n_in=x_tr.shape[1] + 3, n_det=0 if variant is None else dx_tr.shape[1])
    opt = torch.optim.Adam(head.parameters(), lr=f647.LR, weight_decay=f647.WD)

    def loss_of(*, pool, x_org, age_org, dx, ix, w, lab, inf, el, em):
        x, q, age, lead, y = f647.batch_inputs(x_org=x_org, age_org=age_org, pool=pool, ix=ix)
        det = None if dx is None else torch.from_numpy(dx[pool.org[ix]])
        qq, p, _d, _s, _ph, qe = head(x=x, q=q, age=age, lead_h=lead, det=det)
        wt = torch.from_numpy(w[ix])
        pin = (h.pinball(y=y, q=qq) * wt).sum() / wt.sum()
        m = torch.from_numpy(inf[ix]) * (age >= h647.AGE_FLOOR_DAYS).to(torch.float32)
        bce = nn.functional.binary_cross_entropy(
            torch.clamp(p, 1e-4, 1 - 1e-4), torch.from_numpy(lab[ix]), reduction="none"
        )
        bce = (bce * m * wt).sum() / torch.clamp((m * wt).sum(), min=1e-6)
        loss = pin + fit652.LAMBDA * bce
        if dx is not None:
            wm = torch.from_numpy(em[ix]) * wt
            aux = nn.functional.binary_cross_entropy(
                torch.clamp(qe, 1e-4, 1 - 1e-4), torch.from_numpy(el[ix]), reduction="none"
            )
            loss = loss + LAMBDA_END * (aux * wm).sum() / torch.clamp(wm.sum(), min=1e-6)
        return loss

    best, best_state, bad, log = np.inf, None, 0, []
    n = len(train.y)
    for epoch in range(f647.EPOCHS):
        head.train()
        perm = np.random.default_rng(seed * 100 + epoch).permutation(n)
        for a in range(0, n, f647.BATCH):
            ix = perm[a : a + f647.BATCH]
            loss = loss_of(
                pool=train,
                x_org=x_tr,
                age_org=age_tr,
                dx=dx_tr,
                ix=ix,
                w=w_tr,
                lab=lab_tr,
                inf=inf_tr,
                el=el_tr,
                em=em_tr,
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
                        pool=val,
                        x_org=x_va,
                        age_org=age_va,
                        dx=dx_va,
                        ix=ix,
                        w=w_va,
                        lab=lab_va,
                        inf=inf_va,
                        el=el_va,
                        em=em_va,
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
    res: dict[str, np.ndarray] = {}
    if final:  # the window: experiment 620's K1 rows, built by `e00663_window_data.py`
        zw = f650.load(path=c663.WORK / "window" / "real_window.npz", kind="real")
        sets = {"window": ("real", zw, np.flatnonzero(zw["usable"]))}
        data_month = "window"
    else:
        test = f650.sources(month=month, bank="new")
        data_month = month
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
        pool = f650.Pool(parts=[(z, idx, None)])
        x_te, age_te, _ = f647.make_inputs(raw=pool.raw, arm=arm, stats=stats, seed=seed + 3)
        dx = None
        if variant is not None:
            dte = det_part(
                fold=month,
                data_month=data_month,
                kind=kind,
                variant=variant,
                z=z,
                idx=idx,
                neg_share=0.0,
                rng=None,
                noneigh=noneigh,
            )
            dx = np.clip((dte - dstats[0]) / dstats[1], -10, 10).astype(np.float32)
        cols: list[list] = [[] for _ in range(6)]
        with torch.no_grad():
            for a in range(0, len(idx), 50000):
                ix = np.arange(a, min(a + 50000, len(idx)))
                x, q, age, lead, _y = f647.batch_inputs(
                    x_org=x_te, age_org=age_te, pool=pool, ix=ix
                )
                det = None if dx is None else torch.from_numpy(dx[pool.org[ix]])
                for c, v in zip(cols, head(x=x, q=q, age=age, lead_h=lead, det=det), strict=True):
                    c.append(v.numpy())
        res[f"{name}_idx"] = idx.astype(np.int32)
        for key, c in zip(("q", "p", "d", "sigma", "p_haz", "q_end"), cols, strict=True):
            res[f"{name}_{key}"] = np.concatenate(c).astype(np.float32)
    np.savez(
        out.with_suffix(".tmp.npz"),
        **res,
        val_log=np.array(log),
        n_train=n,
        best=best,
        seconds=time.time() - t0,
    )  # ty: ignore[invalid-argument-type]
    out.with_suffix(".tmp.npz").rename(out)
    torch.save(
        {"head": head.state_dict(), "stats": stats, "dstats": dstats}, OUT / f"{out.stem}.pt"
    )
    return f"{out.name}: {len(log)} epochs, val {best:.5f}, {time.time() - t0:.0f}s"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument(
        "--months",
        nargs="+",
        default=list(c652.MONTHS),
        help="scored months, or `final` for all five months and the window",
    )
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
