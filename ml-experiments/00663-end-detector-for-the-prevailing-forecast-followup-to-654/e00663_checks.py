# ruff: noqa
"""Experiment 663: leakage checks. No detector input may change when anything at or after the origin changes."""
import e00650_common as c650
import e00663_panel as pan
import numpy as np

panel = pan.load_month_panel(month="2025-10")
nb, _, _ = c650.world()
ms = c650.month_start_idx(month="2025-10")
o = np.array([ms + 10 * 48] * 3)
p = np.array([0, 3, 5])
T = o[0] - panel.lo
# 1. features read only the past
fold = pan.Fold(sigma=np.full(16, 0.01), const=np.zeros(16))
z = pan.world_z(panel=panel, fold=fold, own=None)
f1 = pan.features(z=z, panel=panel, origins=o, prims=p, nb_of=nb)
z2 = z.copy()
z2[:, T:] += 5.0
print("features leak max diff", float(np.abs(f1 - pan.features(z=z2, panel=panel, origins=o, prims=p, nb_of=nb)).max()))
# 2. fold statistics use only the given months (a held-out month perturbation changes nothing)
panels = {m: pan.load_month_panel(month=m) for m in ("2025-10", "2025-12", "2026-01")}
a = pan.fit_fold(panels=panels, train_months=["2025-12", "2026-01"])
panels["2025-10"].obs += 1.0
b = pan.fit_fold(panels=panels, train_months=["2025-12", "2026-01"])
print("fold stats depend on held-out month:", bool(np.abs(a.sigma - b.sigma).max() > 0 or np.abs(a.const - b.const).max() > 0))
# 3. hard negatives and the common-mode median do not carry the future into the past
neg, negc = pan.hard_negatives(shape=panel.obs.shape, nb_of=nb, seed=1)
zn = pan.world_z(panel=panel, fold=fold, own=None, extra=neg, extra_common=negc[None, :])
fa = pan.features(z=zn, panel=panel, origins=o, prims=p, nb_of=nb)
panel.obs[:, T:] += 0.3
zn2 = pan.world_z(panel=panel, fold=fold, own=None, extra=neg, extra_common=negc[None, :])
print("features after changing observations after the origin, max diff", float(np.abs(fa - pan.features(z=zn2, panel=panel, origins=o, prims=p, nb_of=nb)).max()))
