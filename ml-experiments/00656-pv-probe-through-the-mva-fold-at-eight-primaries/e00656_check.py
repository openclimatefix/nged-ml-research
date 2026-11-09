# ruff: noqa: E501
"""Experiment 656: print the target change of every condition (a sanity check, written before any fit)."""

import sys

import e00428_data as data  # ty: ignore[unresolved-import]
import e00448_run as r448  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import e00656_common as k
import numpy as np

cutoff = sys.argv[1]
clean = c.clean_stack()
refit_t = c.month_start(month=cutoff)
rows = data.build_fold(refit=refit_t, clean=clean)
pre = r448.pretrain_rows(rows=rows, refit=refit_t, clean=clean)
world = k.WorldBuilder(
    primary=pre.primary,
    target_idx=pre.target_idx,
    x=pre.x,
    y=pre.y,
    standardiser=rows.standardiser,
    clean=clean,
)
keep = ~world.gap
print("rows", len(pre.y), "dropped", int(world.gap.sum()))
print("median midday", {int(p): round(v, 3) for p, v in world.median_midday.items()})
for name in k.CONDITIONS:
    cond = k.parse(name=name)
    y = world.target(cond=cond)
    d = (y - pre.y)[keep]
    assert np.isfinite(y[keep]).all(), name
    out = []
    for site, p in [*world.idx6.items(), *world.idxn.items()][::3]:
        s = (pre.primary == p)[keep]
        out.append(f"{site}:{d[s].mean():+.3f}/{d[s].min():+.2f}")
    print(f"{name:20s} min y {y[keep].min():+.3f}  mean change by site {' '.join(out)}", flush=True)
