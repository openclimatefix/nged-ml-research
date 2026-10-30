# ruff: noqa: E501
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

import argparse
from pathlib import Path

import e00397_prep as base  # ty: ignore[unresolved-import]
import e00405_fit as f5  # ty: ignore[unresolved-import]
import e00405_models as m5  # ty: ignore[unresolved-import]
import e00409_prep as prep  # ty: ignore[unresolved-import]
import e00416_models as models  # ty: ignore[unresolved-import]
import numpy as np
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_416 = (
    ML_EXPERIMENTS_DATA_DIR / "00416-mdn-explicit-solar-term-better-powered-follow-up" / "work"
)
SHIFT_LOG_UNITS = 0.7
"""One prior standard deviation of the log-scale arms: the start is shifted by it to show the penalty."""
MONTH = "2026-06"
ARM = "C3"
N_BATCHES = 10


def share(*, model: m5.PvMdn, fold: prep.Fold, seed: int, arm: str) -> dict[str, float]:
    """Return the gradient norms of the likelihood and the prior, and the prior's share."""
    z = (fold.y - fold.target_mean[fold.primary]) / fold.target_std[fold.primary]
    t = {
        "x": torch.from_numpy(fold.x),
        "z": torch.from_numpy(z.astype(np.float32)),
        "w": torch.from_numpy(fold.weight),
        "p": torch.from_numpy(fold.primary),
        "e": torch.from_numpy(fold.extras),
        "c": torch.from_numpy(f5.cells_of(arm=arm, fold=fold)),
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
    """Print the share at the start and the end of training of one arm."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", default=ARM)
    parser.add_argument("--month", default=MONTH)
    args = parser.parse_args()
    refit = base.refit_of(month=args.month)
    fold = prep.build_fold(refit=refit, irradiance_free=True, use_doy=True)
    model = models.make_model(arm=args.arm, fold=fold, table=base.primaries(), refit=refit, seed=0)
    print("start:", share(model=model, fold=fold, seed=0, arm=args.arm))
    with torch.no_grad():
        model.log_c -= SHIFT_LOG_UNITS
    print(
        f"start, capacity {SHIFT_LOG_UNITS} log units below the prior centre:",
        share(model=model, fold=fold, seed=0, arm=args.arm),
    )
    with torch.no_grad():
        model.log_c += SHIFT_LOG_UNITS
    name = f"{args.month}_s0.pt"
    stored = f9_work(arm=args.arm) / name
    model.load_state_dict(torch.load(stored, map_location="cpu"))
    print(f"end ({stored}):", share(model=model, fold=fold, seed=0, arm=args.arm))
    cap = model.capacity().detach()
    mask = model.solar_mask[:, None].bool().expand_as(cap)
    print(
        "mean |capacity - prior centre| / prior sd at the end:",
        float(((cap - model.prior_centre).abs() / model.prior_sd)[mask].mean()),
    )


def f9_work(*, arm: str) -> Path:
    """Return the folder of an arm's stored networks: experiment 409's for C3, else this experiment's."""
    work = prep.WORK if arm == "C3" else WORK_416
    return work / "seed_preds" / arm


if __name__ == "__main__":
    main()
