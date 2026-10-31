# ruff: noqa: E501
"""Log the validation curve of the skeleton over 40 epochs (seed 0, all four refits) to find the best epoch.

PYTHONPATH=ml-experiments/00310-metered-farm-long-lead-scoring:ml-experiments/00430-unified-forecaster-solar-and-pretraining-ablations:ml-experiments/00443-solar-network-hyperparameter-and-architecture-sweep \
        uv run --no-sync python ml-experiments/00443-*/e00443_best_epoch.py
"""

from datetime import timedelta

import e00310_fit as f310  # ty: ignore[unresolved-import]
import e00430_fit as f430  # ty: ignore[unresolved-import]
import e00443_common as c  # ty: ignore[unresolved-import]
import e00443_fit as f  # ty: ignore[unresolved-import]
import numpy as np
import torch  # ty: ignore[unresolved-import]

for start in f430.REFITS:
    cutoff = start - timedelta(days=f310.REFIT_GAP_DAYS)
    stats = f.stats_for(cutoff=cutoff)
    data, v, a = f.training_data(cutoff=cutoff, stats=stats)
    cfg = c.Config(epochs=40)
    # replicate loop to log validation curve
    torch.manual_seed(0)
    m = f.SweepNet(n_static=data.x.shape[1], config=cfg).to(f.DEVICE)
    fit = torch.tensor(np.flatnonzero(~v), device=f.DEVICE)
    val = torch.tensor(np.flatnonzero(v), device=f.DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    g = torch.Generator(device=f.DEVICE).manual_seed(0)
    curve = []
    for _epoch in range(40):
        m.train()
        order = fit[torch.randperm(len(fit), generator=g, device=f.DEVICE)]
        for s in range(0, len(order), 2048):
            r = order[s : s + 2048]
            loss = m.nll_per_row(data=data, rows=r).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            curve.append(float(m.nll_per_row(data=data, rows=val).mean()))
    print(
        start.date(),
        "best epoch (1-based)",
        int(np.argmin(curve)) + 1,
        "val nll",
        round(min(curve), 4),
        "first",
        round(curve[0], 4),
        "last",
        round(curve[-1], 4),
    )
