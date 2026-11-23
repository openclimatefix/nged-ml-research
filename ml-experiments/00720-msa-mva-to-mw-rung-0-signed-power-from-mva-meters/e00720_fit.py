"""Experiment 720: fitting the demand mean and the residual of the hindsight model from anchor rows.

All arrays are `(n_days, 48)` in points of capacity. The demand mean `f` is a ridge regression of
gross demand `g = p + v` on half-hour of day by day type and on temperature (a linear and a hinge
term in six blocks of the day), fitted on anchor rows only: reference days at half-hours with
`u < u_min`, where restriction R1 says the half-hour imported. The residual's autoregression
coefficient and innovation come from the same rows.
"""

from typing import Final

import e00720_hmm as hmm
import numpy as np

N_DAY_TYPES: Final[int] = 3
HINGE_C: Final[float] = 15.0
PHI_MAX: Final[float] = 0.98


def design_matrix(
    *, day_type: np.ndarray, temperature: np.ndarray, n_steps: int = 48
) -> np.ndarray:
    """Return the `(n_days × 48, n_features)` design matrix of the demand mean."""
    n_days = len(day_type)
    n_cell = n_steps * N_DAY_TYPES
    cols = np.zeros((n_days, n_steps, n_cell + 12))
    h = np.arange(n_steps)[None, :]
    cell = day_type[:, None] * n_steps + h
    for d in range(n_days):
        cols[d, np.arange(n_steps), cell[d]] = 1.0
    block = (np.arange(n_steps) // 8)[None, :]
    t = temperature[:, None] * np.ones((1, n_steps))
    for b in range(6):
        sel = block == b
        cols[:, :, n_cell + b] = np.where(sel, (t - 10.0) / 10.0, 0.0)
        cols[:, :, n_cell + 6 + b] = np.where(sel, np.maximum(t - HINGE_C, 0.0) / 10.0, 0.0)
    return cols.reshape(n_days * n_steps, -1)


def fit_demand_mean(
    *,
    g: np.ndarray,
    rows: np.ndarray,
    day_type: np.ndarray,
    temperature: np.ndarray,
    ridge: float = 1.0,
) -> np.ndarray:
    """Return `f` for every day: the ridge fit of `g` on the rows where `rows` is true.

    Args:
        g: Gross demand if imported, `p + v`, `(n_days, 48)`.
        rows: Boolean `(n_days, 48)` of anchor rows, finite in `g`.
        day_type: `(n_days,)` day type.
        temperature: `(n_days,)` temperature.
        ridge: The ridge penalty on the temperature terms and the cell intercepts.
    """
    x = design_matrix(day_type=day_type, temperature=temperature)
    y = g.reshape(-1)
    use = rows.reshape(-1) & np.isfinite(y)
    xs, ys = x[use], y[use]
    a = xs.T @ xs + ridge * np.eye(x.shape[1])
    w = np.linalg.solve(a, xs.T @ ys)
    return (x @ w).reshape(g.shape)


def fit_residual(*, g: np.ndarray, f: np.ndarray, rows: np.ndarray) -> tuple[float, float]:
    """Return `(phi, sigma_r)` from anchor residuals at consecutive half-hours of one day."""
    res = np.where(rows & np.isfinite(g), g - f, np.nan)
    a, b = res[:, :-1].reshape(-1), res[:, 1:].reshape(-1)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 50:
        return 0.9, 1.0
    phi = float(np.clip(np.corrcoef(a[ok], b[ok])[0, 1], 0.0, PHI_MAX))
    sd = float(np.nanstd(res))
    return phi, max(sd * float(np.sqrt(1.0 - phi**2)), 0.05)


def k0_spec(*, phi: float, sigma_r: float, q: float, n_bins: int = 41) -> hmm.HmmSpec:
    """Return the sign-only spec (no generator, `K = 0`) used at D12 and in the injected worlds."""
    if phi > PHI_MAX:
        n_bins = 81
        phi = PHI_MAX
    return hmm.HmmSpec(
        grid=hmm.residual_grid(phi=phi, sigma_r=sigma_r, n_bins=n_bins),
        phi=phi,
        sigma_r=sigma_r,
        unit_size=0.0,
        n_max=0,
        trans_n=np.ones((1, 1)),
        q=q,
        sigma_e=0.5,
    )
