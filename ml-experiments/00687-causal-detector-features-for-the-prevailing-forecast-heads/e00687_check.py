# ruff: noqa: E501, D103
"""Experiment 687: causality check. Perturbing the gap after the cutoff must not change any feature at the cutoff."""

import e00687_features as f
import numpy as np


def main() -> None:
    rng = np.random.default_rng(0)
    n_p, n = 6, 6000
    gap = rng.normal(0, 0.02, (n_p, n)).cumsum(axis=1) * 0.05 + rng.normal(0, 0.02, (n_p, n))
    gap[:, 3000:3600] += 0.1
    gap[rng.random((n_p, n)) < 0.01] = np.nan
    sig, ref = gap.copy(), gap.copy()
    units = {w: np.full(n_p, 0.03 * np.sqrt(w / 12)) for w in f.W}
    nb = {p: [q for q in range(n_p) if q != p][:2] for p in range(n_p)}
    prim = np.arange(n_p).repeat(4)
    t = np.tile(np.array([2500, 3100, 3500, 4200]), n_p)
    a, fa = f.features_at(
        g=f.build_grids(gap=gap, gap_sig=sig, gap_ref=ref, units=units, start=0),
        prim=prim,
        t=t,
        neighbours=nb,
    )
    g2, s2, r2 = gap.copy(), sig.copy(), ref.copy()
    for arr in (g2, s2, r2):
        arr[:, 2501:] += rng.normal(
            0, 5.0, arr[:, 2501:].shape
        )  # perturb everything after the earliest cutoff
    sel = t == 2500
    b, fb = f.features_at(
        g=f.build_grids(gap=g2, gap_sig=s2, gap_ref=r2, units=units, start=0),
        prim=prim[sel],
        t=t[sel],
        neighbours=nb,
    )
    assert np.allclose(a[sel], b, equal_nan=True), np.abs(a[sel] - b).max()
    assert (fa[sel] == fb).all()
    print(
        "causality check passed: no feature at cutoff 2500 changed when later gaps were perturbed"
    )


if __name__ == "__main__":
    main()
