# ruff: noqa: D100, I001, N812, PT018
import sys
import numpy as np

sys.path.insert(0, ".")
import e00558_lags as L

rng = np.random.default_rng(0)
origin = rng.integers(10_000, 20_000, 5000)
lead = rng.integers(1, 21, 5000) * 1  # steps up to 20
target = origin + rng.integers(1, 700, 5000)
d, w = L.capped_lag_indices(target_idx=target, origin_idx=origin)
assert ((target - d) % 48 == 0).all() and ((target - w) % 336 == 0).all()
assert (origin - d >= 48).all() and (origin - w >= 48).all() and (origin - d < 96).all()
# latest: one more period would be too young
assert (origin - (d + 48) < 48).all() and (origin - (w + 336) < 48).all()
ad, aw = L.capped_lag_ages_hours(lead_hours=(target - origin) / 2)
assert np.allclose(ad, (target - d) / 2) and np.allclose(aw, (target - w) / 2)
a, b = L.block_dropout_masks(origin_idx=origin, seed=1)
assert not (a & b).any()
print(a.mean(), b.mean())
print("ok")
