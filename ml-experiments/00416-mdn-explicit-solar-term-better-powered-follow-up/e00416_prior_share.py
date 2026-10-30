"""Print the capacity prior's share of the loss gradient for C3, at the start and end of training.

The training loss is the weighted mean negative log-likelihood of a batch plus the capacity penalty
divided by the effective number of training rows (`train` in experiment 405). This script computes,
on random batches of training rows of one fold, the gradient of each part with respect to the
learned log-capacities, and reports the prior's share: the norm of the prior gradient over the sum
of the two norms. The start is a freshly built C3 (capacity at the prior centre); the end is a
stored C3 network of experiment 409 (same fold, seed 0).

```bash
export PYTHONPATH=<as in e00416_fit.py>
uv run --no-sync python ml-experiments/00416-*/e00416_prior_share.py
```
"""

import e00397_prep as base  # ty: ignore[unresolved-import]
import e00405_fit as f5  # ty: ignore[unresolved-import]
import e00405_models as m5  # ty: ignore[unresolved-import]
import e00409_prep as prep  # ty: ignore[unresolved-import]
import numpy as np
import torch  # ty: ignore[unresolved-import]

MONTH = "2026-06"
ARM = "C3"
N_BATCHES = 10


def share(*, model: m5.PvMdn, fold: prep.Fold, seed: int) -> dict[str, float]:
    """Return the gradient norms of the likelihood and the prior, and the prior's share."""
    z = (fold.y - fold.target_mean[fold.primary]) / fold.target_std[fold.primary]
    t = {
        "x": torch.from_numpy(fold.x),
        "z": torch.from_numpy(z.astype(np.float32)),
        "w": torch.from_numpy(fold.weight),
        "p": torch.from_numpy(fold.primary),
        "e": torch.from_numpy(fold.extras),
        "c": torch.from_numpy(f5.cells_of(arm=ARM, fold=fold)),
    }
    train_idx = np.flatnonzero(~fold.is_val)
    n_eff = float(fold.weight[train_idx].sum())
    rng = np.random.default_rng(seed)
    model.eval()
    nll_norms, pen_norms = [], []
    for _ in range(N_BATCHES):
        b = torch.from_numpy(rng.choice(train_idx, size=f5.BATCH, replace=False))
        log_w, mean, std = model.parts(t["x"][b], t["p"][b], t["e"][b], t["c"][b])
        nll = m5.nll_parts(log_w=log_w, mean=mean, std=std, z=t["z"][b])
        nll_loss = (t["w"][b] * nll).sum() / t["w"][b].sum()
        (g_nll,) = torch.autograd.grad(nll_loss, model.log_c, retain_graph=True)
        (g_pen,) = torch.autograd.grad(model.penalty() / n_eff, model.log_c)
        nll_norms.append(float(g_nll.norm()))
        pen_norms.append(float(g_pen.norm()))
    nll_norm, pen_norm = float(np.mean(nll_norms)), float(np.mean(pen_norms))
    return {
        "n_eff_rows": n_eff,
        "likelihood_gradient_norm": nll_norm,
        "prior_gradient_norm": pen_norm,
        "prior_share": pen_norm / (nll_norm + pen_norm),
    }


def main() -> None:
    """Print the share at the start and the end of training."""
    refit = base.refit_of(month=MONTH)
    fold = prep.build_fold(refit=refit, irradiance_free=True, use_doy=True)
    model = f5.make_model(arm=ARM, fold=fold, table=base.primaries(), refit=refit, seed=0)
    print("start (capacity at the prior centre):", share(model=model, fold=fold, seed=0))
    stored = prep.WORK / "seed_preds" / ARM / f"{MONTH}_s0.pt"
    model.load_state_dict(torch.load(stored, map_location="cpu"))
    print("end (experiment 409's stored C3, seed 0):", share(model=model, fold=fold, seed=0))
    cap = model.capacity().detach()
    centre = model.prior_centre
    mask = model.solar_mask[:, None].bool().expand_as(cap)
    print(
        "mean |capacity - prior centre| / prior sd at the end:",
        float(((cap - centre).abs() / model.prior_sd)[mask].mean()),
    )


if __name__ == "__main__":
    main()
