# ruff: noqa: E501
"""Experiment 720, rung 0c: known-answer checks of the folded loss (CPU, seconds, no real data).

Run: `python e00720_check_loss.py`. Exit status 0 means every check passed.
"""

import sys
from typing import Final

import e00720_fold as fold
import numpy as np
from scipy.optimize import minimize

PI: Final = np.array([0.5, 0.3, 0.2])
MU: Final = np.array([3.0, -2.0, 8.0])
SIGMA: Final = np.array([1.0, 1.5, 2.0])
Q: Final = 2.0
N: Final = 120_000


def draws(*, seed: int) -> np.ndarray:
    """Return signed draws of the known mixture."""
    return fold.sample_mixture(rng=np.random.default_rng(seed), n=N, pi=PI, mu=MU, sigma=SIGMA)


def mean_nll(
    *, m: np.ndarray, pi: np.ndarray, mu: np.ndarray, sigma: np.ndarray, q: float
) -> float:
    """Return the mean folded loss of the readings under one mixture."""
    return float(np.mean(fold.folded_nll(m=m, pi=pi, mu=mu, sigma=sigma, q=q)))


def check_minimum_and_symmetry() -> None:
    """The loss is minimised at the truth and at every sign-reversed copy, and no lower elsewhere."""
    x = draws(seed=1)
    m = fold.magnitude_to_reading(y=np.abs(x), q=Q)
    base = mean_nll(m=m, pi=PI, mu=MU, sigma=SIGMA, q=Q)
    for k in range(3):
        mu = MU.copy()
        mu[k] = -mu[k]
        flipped = mean_nll(m=m, pi=PI, mu=mu, sigma=SIGMA, q=Q)
        assert abs(flipped - base) < 1e-9, (k, flipped, base)
    rng = np.random.default_rng(2)
    for _ in range(20):
        mu = MU + rng.normal(0.0, 0.3, 3)
        sigma = SIGMA * np.exp(rng.normal(0.0, 0.1, 3))
        assert mean_nll(m=m, pi=PI, mu=mu, sigma=sigma, q=Q) > base - 1e-3, (
            "a perturbation beats the truth"
        )
    print(f"PASS minimum and symmetry: mean loss {base:.4f}, flipped copies equal to 1e-9")


def check_quantile_coverage() -> None:
    """The folded quantiles cover at their nominal rates, as do the signed quantiles."""
    x = draws(seed=3)
    n = len(x)
    pi, mu, sigma = (np.tile(a, (n, 1)) for a in (PI, MU, SIGMA))
    fq = fold.folded_quantiles(pi=pi[:1], mu=mu[:1], sigma=sigma[:1])[0]
    sq = fold.signed_quantiles(pi=pi[:1], mu=mu[:1], sigma=sigma[:1])[0]
    for k, tau in enumerate(fold.TAUS_11):
        cover_f = float(np.mean(np.abs(x) <= fq[k]))
        cover_s = float(np.mean(x <= sq[k]))
        tol = 4.0 * np.sqrt(tau * (1 - tau) / n) + 1e-4
        assert abs(cover_f - tau) < tol, ("folded", tau, cover_f)
        assert abs(cover_s - tau) < tol, ("signed", tau, cover_s)
    abs_of_signed = np.abs(sq)
    assert not np.allclose(abs_of_signed, fq, atol=0.05), (
        "a folded quantile must not equal |signed quantile|"
    )
    print("PASS quantile coverage: 11 folded and 11 signed quantiles within 4 standard errors")


def check_fold_point() -> None:
    """Zero megawatts sits at `z = -mean/std`, and folding about `z = 0` is the wrong likelihood."""
    mean, std = 2.0, 1.5
    assert abs(fold.fold_point_z(target_mean=mean, target_std=std) + mean / std) < 1e-12
    p = draws(seed=4)  # signed power in the units of capacity percent, mean 2 is not zero

    m = np.abs(p)
    # Head in z units, shifted to v units, must reproduce the loss of a head that works in power units.
    mu_z, sigma_z = (MU - mean) / std, SIGMA / std
    mu_v = fold.mean_to_v(mu_z=mu_z, target_mean=mean, target_std=std)
    v = m / std
    right = mean_nll(m=v, pi=PI, mu=mu_v, sigma=sigma_z, q=0.0)
    in_power = mean_nll(m=m, pi=PI, mu=MU, sigma=SIGMA, q=0.0) - np.log(std)
    assert abs(right - in_power) < 1e-6, (right, in_power)
    wrong = mean_nll(m=v, pi=PI, mu=mu_z, sigma=sigma_z, q=0.0)  # forgot to add mean/std
    assert wrong > right + 0.05, ("folding about z = 0 should fit worse", wrong, right)
    zz = np.array([-9.0, 0.5, 9.0])
    v_hist = fold.history_z_to_v(z=zz, target_mean=mean, target_std=std)
    assert np.allclose(v_hist, [-5.0 + mean / std, 0.5 + mean / std, 5.0 + mean / std])
    print(
        f"PASS fold point: zero MW at z = {-mean / std:.3f}; loss in v units {right:.4f} equals power units; folding about z=0 loses {wrong - right:.3f}"
    )


def check_censored_branch() -> None:
    """The censored mass `P(|X| <= Q)` matches the simulated share, and the loss uses it."""
    x = draws(seed=5)
    n = len(x)
    pi, mu, sigma = (np.tile(a, (n, 1)) for a in (PI, MU, SIGMA))
    mass = float(np.exp(fold.censored_log_mass(q=Q, pi=pi[:1], mu=mu[:1], sigma=sigma[:1])[0]))
    share = float(np.mean(np.abs(x) <= Q))
    tol = 4.0 * np.sqrt(share * (1 - share) / n)
    assert abs(mass - share) < tol, (mass, share)
    m = np.full(5, 0.5 * Q)  # readings below the floor
    loss = fold.folded_nll(m=m, pi=PI, mu=MU, sigma=SIGMA, q=Q)
    assert np.allclose(loss, -np.log(mass)), (loss[0], -np.log(mass))
    print(f"PASS censored branch: analytic {mass:.5f}, simulated {share:.5f}")


def _unpack(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logits = np.concatenate([[0.0], theta[:2]])
    pi = np.exp(logits - logits.max())
    pi /= pi.sum()
    return pi, theta[2:5], np.exp(theta[5:8])


def check_reversed_start() -> None:
    """A fit from a sign-reversed start ends at the reversed mixture with the same loss."""
    x = draws(seed=6)[:20_000]
    m = fold.magnitude_to_reading(y=np.abs(x), q=0.0)

    def loss(theta: np.ndarray) -> float:
        pi, mu, sigma = _unpack(theta)
        return mean_nll(m=m, pi=pi, mu=mu, sigma=sigma, q=0.0)

    start = np.array([-0.4, -0.9, 2.5, -1.5, 7.0, 0.0, 0.3, 0.6])
    reversed_start = start.copy()
    reversed_start[2:5] *= -1.0
    a = minimize(loss, start, method="L-BFGS-B")
    b = minimize(loss, reversed_start, method="L-BFGS-B")
    mu_a, mu_b = _unpack(a.x)[1], _unpack(b.x)[1]
    assert abs(a.fun - b.fun) < 1e-5, (a.fun, b.fun)
    assert np.allclose(np.abs(mu_a), np.abs(mu_b), atol=0.05), (mu_a, mu_b)
    assert np.sign(mu_a[0]) == -np.sign(mu_b[0]), (
        "the reversed start must end with the reversed sign"
    )
    print(
        f"PASS reversed start: losses {a.fun:.5f} and {b.fun:.5f}; first mean {mu_a[0]:+.2f} against {mu_b[0]:+.2f}"
    )


def check_negative_control() -> None:
    """A loss that does not fold must NOT be symmetric (guards against a trivially-passing symmetry check)."""
    x = draws(seed=7)
    m = np.abs(x)
    mu = MU.copy()
    mu[1] = -mu[1]
    plain = np.mean(-fold.log_normal_pdf(x=m[:, None], mu=MU[None], sigma=SIGMA[None]).max(axis=1))
    plain_flipped = np.mean(
        -fold.log_normal_pdf(x=m[:, None], mu=mu[None], sigma=SIGMA[None]).max(axis=1)
    )
    assert abs(plain - plain_flipped) > 1e-3
    print("PASS negative control: an unfolded loss is not symmetric under a sign flip")


def main() -> int:
    """Run every check; return 0 when all pass."""
    for check in (
        check_minimum_and_symmetry,
        check_quantile_coverage,
        check_fold_point,
        check_censored_branch,
        check_reversed_start,
        check_negative_control,
    ):
        check()
    print("rung 0c: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
