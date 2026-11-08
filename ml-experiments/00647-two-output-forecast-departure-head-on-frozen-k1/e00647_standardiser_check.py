# ruff: noqa: E501, D415
"""Experiment 647, stage 3 follow-up: does a later month's K1 suffer from earlier-month inputs standardised differently? (GPU)

K1 of the scored month is applied to rows of an earlier month whose input columns were standardised
with that earlier month's standardiser (as the E2 and E3 training rows are). The check compares the
mixture negative log-likelihood, in capacity units, of K1(earlier month) and K1(later month) on the
same earlier-month rows, per K1 seed.

Usage: python e00647_standardiser_check.py EARLY LATE
"""

import sys

import e00397_models as base  # ty: ignore[unresolved-import]
import e00559_model as m559  # ty: ignore[unresolved-import]
import e00647_common as c647  # ty: ignore[unresolved-import]
import e00647_enc as enc  # ty: ignore[unresolved-import]
import e00647_fit as fit  # ty: ignore[unresolved-import]
import numpy as np
import torch  # ty: ignore[unresolved-import]

early, late = sys.argv[1], sys.argv[2]
lines = []
for kind in ("real", "bank"):
    z = fit.load(month=early)[kind]
    idx = np.flatnonzero(z["train_ok"])
    idx = np.sort(np.random.default_rng(0).choice(idx, size=min(60_000, len(idx)), replace=False))
    rows = enc.Rows(parts=[(early, kind, idx)], stats=None)
    for seed in (0, 1):
        res = {}
        for month in (early, late):
            mt = enc.meta(month)
            mu = torch.from_numpy(mt["target_mean"]).float().cuda()[rows.prim]
            sd = torch.from_numpy(mt["target_std"]).float().cuda()[rows.prim]
            nll = torch.empty(rows.n, device="cuda")
            with torch.no_grad():
                for r in (0, 1):
                    w = rows.refit == r
                    if not w.any():
                        continue
                    mdl = enc.k1_model(month=month, seed=seed, refit=r)
                    s = torch.nonzero(w).squeeze(1)
                    hl = enc.lag_hl(mdl=mdl, x=rows.x[s], prim=rows.prim[s], tab=rows.tab[s])
                    out = enc.nra_forward(mdl=mdl, x=rows.x[s], prim=rows.prim[s], hl=hl)
                    zt = (rows.yobs[s] - mu[s]) / sd[s]
                    nll[s] = base.mixture_nll(
                        out=out, z=zt, components=m559.COMPONENTS
                    ) + torch.log(sd[s])
            res[month] = float(nll.mean())
        lines.append(
            f"{kind} rows of {early} ({rows.n} sampled), K1 seed {seed}: NLL of K1({early}) = {res[early]:.4f}, NLL of K1({late}) = {res[late]:.4f}, difference = {res[late] - res[early]:+.4f} nats per row (capacity units)"
        )
print("\n".join(lines))
(c647.WORK / "cache" / f"standardiser_check_{early}_{late}.txt").write_text("\n".join(lines) + "\n")
