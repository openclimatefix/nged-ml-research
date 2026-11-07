"""Experiment 625 pre-run check: K values at levels j/(K+1) from 51 values give an unbiased CRPS.

Draws 20,000 observations from a standard normal, takes the 51 values at levels k/52 of the same
normal, cuts them to K, and compares the CRPS of the K values with the exact CRPS (0.2337 *
...) of the normal, computed from the 51 values at fine resolution.
"""

import e00625_score as m
import numpy as np
from scipy.stats import norm

rng = np.random.default_rng(0)
y = rng.standard_normal(20000)
fine = norm.ppf((np.arange(1, 2001)) / 2001)
exact = np.mean(
    [np.abs(fine - v).mean() - 0.5 * np.abs(fine[:, None] - fine[None]).mean() for v in y[:300]]
)
v51 = norm.ppf(np.arange(1, 52) / 52)
rows = np.broadcast_to(v51[:, None, None], (51, 1, 1)).copy()
for k in (5, 10, 20, 51):
    cut = m.cut_paths(paths=np.broadcast_to(v51[:, None, None], (51, 1, 1)) * 1.0, k=k)
    # members keep distinct ranks only if paths differ: use the identity ordering
    vals = np.sort(cut[:, 0, 0])
    crps = np.mean(
        [m.plain_crps(paths=vals[:, None, None], actual=np.array([[x]]))[0, 0] for x in y[:2000]]
    )
    print(k, round(float(crps), 4), "exact", round(float(exact), 4))
