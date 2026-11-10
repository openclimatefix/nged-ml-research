# ruff: noqa: D103, E402
"""Experiment 684: known-answer checks of the positive output families (CPU, no data).

Usage: E00684_FAMILY={tgauss|gamma} python e00684_check.py. Checks that the negative log
likelihood integrates to one over the family's support, that the quantiles invert the cumulative
distribution, that Gaussian rows equal the original functions, and that no quantile is below zero.
"""

import e00684_family as fam
import numpy as np
import torch
from scipy import integrate

torch.manual_seed(0)
base = fam.base
K = 3
import e00578_common as k578  # ty: ignore[unresolved-import]

sm = k578.sites()
pos_primaries = fam.positive_primaries()
print("positive primaries:", pos_primaries.tolist(), "of", len(sm))
fam.STATE.mean = np.zeros(len(sm))
fam.STATE.std = np.full(len(sm), 0.5)
p_pos, p_neg = int(pos_primaries[0]), int(next(i for i in range(len(sm)) if i not in pos_primaries))
fam.STATE.mean[p_pos] = 2.0
fam.STATE.mean[p_neg] = 2.0
out = torch.randn(2, 3 * K, dtype=torch.float64) * 0.5
# integrate the density of a positive row over its support
fam.STATE.primary = torch.tensor([p_pos, p_neg])
z0 = -4.0


def density(z: float, row: int) -> float:
    return float(
        torch.exp(
            -fam.mixture_nll(out=out, z=torch.full((2,), z, dtype=torch.float64), components=K)
        )[row]
    )


total, _ = integrate.quad(lambda z: density(z, 0), z0, 60, limit=200)
print(f"positive row: density integrates to {total:.6f} over z >= z0")
assert abs(total - 1.0) < 1e-6, total
total_g, _ = integrate.quad(lambda z: density(z, 1), -60, 60, limit=200)
print(f"gaussian row: density integrates to {total_g:.6f}")
assert abs(total_g - 1.0) < 1e-6, total_g
log_w, mean, std = fam.mixture_parts(out=out, components=K)
q = fam.mixture_quantiles(log_w=log_w, mean=mean, std=std)
print("quantiles positive row:", np.round(q[0].numpy(), 4).tolist())
assert (q[0] >= z0 - 1e-9).all()
# the CDF at each quantile equals its level, by numeric integration
for j, level in enumerate(base.LEVELS):
    cdf, _ = integrate.quad(lambda z: density(z, 0), z0, float(q[0, j]), limit=200)
    assert abs(cdf - level) < 1e-5, (level, cdf)
ref = (
    fam._orig_quantiles(
        *fam._orig_parts(out=out, components=K)[:1], *fam._orig_parts(out=out, components=K)[1:]
    )
    if False
    else None
)
ref = fam._orig_quantiles(
    log_w=fam._orig_parts(out=out, components=K)[0],
    mean=fam._orig_parts(out=out, components=K)[1],
    std=fam._orig_parts(out=out, components=K)[2],
)
assert torch.allclose(q[1], ref[1], atol=1e-6), (q[1], ref[1])
print("gaussian row equals the original quantiles")
# the plumbing: constructing a real `Rows` records the fold's target statistics
import e00428_data as d428  # ty: ignore[unresolved-import]

if fam.FAMILY != "gauss":
    fam.STATE.mean = fam.STATE.std = None
    rows = d428.Rows(
        x=None, y=None, primary=None, origin_idx=None, target_idx=None, lead=None, weight=None,
        is_val=None, standardiser=None, target_mean=np.arange(3.0), target_std=np.ones(3),
        power_mean=0.0, power_std=1.0, columns=[], dropped=0,
    )  # fmt: skip
    assert np.array_equal(fam.STATE.mean, np.arange(3.0)), fam.STATE.mean  # ty: ignore[invalid-argument-type]
    print("Rows.__init__ records the target statistics")
print("all checks passed for", fam.FAMILY)
