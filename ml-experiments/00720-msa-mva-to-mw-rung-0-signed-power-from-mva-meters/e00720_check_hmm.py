# ruff: noqa: E501
"""Experiment 720, rung 0e and the aliasing toy of rung 0c: known-answer checks of the hidden Markov model.

Run: `python e00720_check_hmm.py`. Every world is synthetic; no real data is read.
"""

import itertools
import sys
from typing import Any

import e00720_hmm as hmm
import e00720_sign as sign
import e00720_worlds as worlds
import numpy as np


def make_spec(
    *,
    n_max: int,
    unit_size: float,
    n_bins: int = 15,
    phi: float = 0.9,
    sigma_r: float = 0.4,
    stay: float = 0.85,
    q: float = 0.2,
    sigma_e: float = 0.4,
) -> hmm.HmmSpec:
    """Return a spec with a persistent unit-count chain."""
    k = n_max + 1
    trans = np.full((k, k), (1.0 - stay) / max(k - 1, 1))
    np.fill_diagonal(trans, stay)
    grid = hmm.residual_grid(phi=phi, sigma_r=sigma_r, n_bins=n_bins)
    return hmm.HmmSpec(
        grid=grid,
        phi=phi,
        sigma_r=sigma_r,
        unit_size=unit_size,
        n_max=n_max,
        trans_n=trans,
        q=q,
        sigma_e=sigma_e,
    )


def check_brute_force() -> None:
    """On a tiny model the forward-backward loglik and smoothed marginals equal brute-force sums."""
    spec = make_spec(n_max=1, unit_size=3.0, n_bins=3, sigma_r=1.0)
    t_len = 5
    rng = np.random.default_rng(0)
    f = 4.0 + rng.normal(0, 0.3, t_len)
    p = rng.uniform(0, 2, t_len)
    m = np.abs(rng.normal(2.0, 1.0, t_len)) + 0.5
    post = hmm.forward_backward(spec=spec, m=m, f=f, p=p)
    trans = hmm.full_transition(spec=spec)
    x = hmm.state_power(spec=spec, f=f, p=p)
    pi0 = (
        hmm.stationary(trans=spec.trans_n)[:, None]
        * hmm.stationary(trans=hmm.residual_transition(spec=spec))[None, :]
    ).reshape(-1)
    s = spec.n_states
    total = 0.0
    marg = np.zeros((t_len, s))
    for path in itertools.product(range(s), repeat=t_len):
        w = pi0[path[0]] * np.exp(hmm.log_emission_first(spec=spec, m_0=m[0], x_0=x[0])[path[0]])
        for t in range(1, t_len):
            le = hmm.log_emission_pair(
                spec=spec, m_t=m[t], x_prev=x[t - 1], x_now=x[t], n_sub=hmm.N_SUBSTEPS_HMM
            )
            w *= trans[path[t - 1], path[t]] * np.exp(le[path[t - 1], path[t]])
        total += w
        for t in range(t_len):
            marg[t, path[t]] += w
    marg /= total
    assert abs(post.log_likelihood - np.log(total)) < 1e-8, (post.log_likelihood, np.log(total))
    assert np.abs(post.smoothed - marg).max() < 1e-8, np.abs(post.smoothed - marg).max()
    assert np.abs(post.xi_n.sum() - (t_len - 1)) < 1e-8
    print(
        f"PASS brute force: loglik {post.log_likelihood:.6f} and all {t_len * s} smoothed marginals match to 1e-8"
    )


def check_alias_toy() -> None:
    """One and two units give identical readings at net demand 1.5 g, and the posterior cannot separate them."""
    g = 10.0
    spec = make_spec(n_max=3, unit_size=g, n_bins=5, sigma_r=0.05, phi=0.5, q=0.0, sigma_e=0.3)
    for pair, net in (
        ((0, 1), 0.5 * g),
        ((0, 2), g),
        ((1, 2), 1.5 * g),
        ((1, 3), 2.0 * g),
        ((2, 3), 2.5 * g),
    ):
        a, b = pair
        assert abs(abs(net - a * g) - abs(net - b * g)) < 1e-12, pair
    t_len = 60
    f = np.full(t_len, 1.5 * g)
    p = np.zeros(t_len)
    m = np.full(t_len, 0.5 * g)
    post = hmm.forward_backward(spec=spec, m=m, f=f, p=p)
    pu = hmm.summarise(post=post, spec=spec)["p_units"]
    mid = pu[10:-10]
    assert np.abs(mid[:, 1] - mid[:, 2]).max() < 0.2, (
        "units 1 and 2 should be near-aliased at flat 1.5 g"
    )
    assert mid[:, 1].mean() + mid[:, 2].mean() > 0.9
    print(
        f"PASS alias toy: flat net demand 1.5 g gives P(n=1) = {mid[:, 1].mean():.2f}, P(n=2) = {mid[:, 2].mean():.2f}; the sign is ambiguous between them"
    )


def oracle_scores(*, f: np.ndarray, seed: int, g: float = 10.0) -> dict[str, float]:
    """Simulate 6,000 half-hours at oracle parameters and score the smoothed unit count."""
    spec = make_spec(
        n_max=3, unit_size=g, n_bins=15, sigma_r=0.3, phi=0.9, stay=0.85, q=0.2, sigma_e=0.4
    )
    n_steps = len(f)
    sample = worlds.simulate_switching(
        n_steps=n_steps, f=f, p=np.zeros(n_steps), trans_n=spec.trans_n, unit_size=g, pv_scale=1.0,
        phi=spec.phi, sigma_r=spec.sigma_r, q=spec.q, sigma_e=spec.sigma_e, df=spec.df, seed=seed,
    )  # fmt: skip
    post = hmm.forward_backward(spec=spec, m=sample.m, f=f, p=np.zeros(n_steps), want_xi=False)
    pu = hmm.summarise(post=post, spec=spec)["p_units"]
    est = pu.argmax(axis=1)
    run_true, run_est = sample.n >= 1, est >= 1
    tp = float((run_true & run_est).sum())
    f1 = 2 * tp / (run_true.sum() + run_est.sum())
    one_two = (sample.n == 1) | (sample.n == 2)
    confused = float(np.mean(est[one_two] != sample.n[one_two])) if one_two.any() else float("nan")
    q = hmm.signed_quantiles_of(post=post, spec=spec)
    soft = float(hmm.signed_crps(quantiles=q, truth=sample.x).mean())
    sign_hard = np.where(hmm.summarise(post=post, spec=spec)["p_negative"] > 0.5, -1.0, 1.0)
    hard = float(
        np.abs(sign_hard * np.sqrt(np.maximum(sample.m**2 - spec.q**2, 0.0)) - sample.x).mean()
    )
    return {
        "f1_any_running": f1,
        "count_accuracy": float(np.mean(est == sample.n)),
        "confused_1_2": confused,
        "crps_soft": soft,
        "crps_hard": hard,
    }


def check_oracle_simulation() -> None:
    """Flat demand at 1.5 g is the hard case; a daily swing and a D8-like level break the aliasing."""
    t = np.arange(6000)
    g = 10.0
    flat = np.full(6000, 1.5 * g)
    swing = 1.5 * g + 0.6 * g * np.sin(2 * np.pi * t / 48.0)
    high = 7.0 * g + 1.25 * g * np.sin(2 * np.pi * t / 48.0)
    res = {
        name: oracle_scores(f=f, seed=7 + i)
        for i, (name, f) in enumerate(
            (("flat 1.5g", flat), ("swing 0.6g", swing), ("D8-like level", high))
        )
    }
    for name, r in res.items():
        print(
            f"  {name}: F1 {r['f1_any_running']:.3f}, count accuracy {r['count_accuracy']:.3f}, 1/2 confused {r['confused_1_2']:.3f}, signed CRPS {r['crps_soft']:.3f} (two lobes) vs {r['crps_hard']:.3f} (hard sign)"
        )
    assert res["flat 1.5g"]["f1_any_running"] > 0.8
    assert res["swing 0.6g"]["count_accuracy"] > res["flat 1.5g"]["count_accuracy"] + 0.05
    assert res["D8-like level"]["count_accuracy"] > 0.95
    assert res["D8-like level"]["f1_any_running"] > 0.97
    assert res["flat 1.5g"]["crps_soft"] < res["flat 1.5g"]["crps_hard"], (
        "two lobes must beat the hard sign where the sign is ambiguous"
    )
    print(
        "PASS oracle simulation: aliasing lowers the unit count at flat 1.5 g and is broken by a swing or a high level"
    )


def check_hidden_sign_world() -> None:
    """With no generator (K = 0) the model recovers the sign of an exporting synthetic Primary."""
    w = worlds.synthetic_world(n_days=900, capacity=100.0, seed=21, pv_fraction=1.0)
    days = np.arange(600, 660)  # midsummer of year 2
    f = w.demand_mean[days].reshape(-1)
    p = w.p[days].reshape(-1)
    m = w.reading[days].reshape(-1)
    spec = make_spec(n_max=0, unit_size=0.0, n_bins=21, sigma_r=1.0, phi=0.9, q=2.0, sigma_e=0.6)
    post = hmm.forward_backward(spec=spec, m=m, f=f, p=p, want_xi=False)
    out = hmm.summarise(post=post, spec=spec)
    filt = hmm.summarise(post=post, spec=spec, version="filtered")
    x_true = w.x[days].reshape(-1)
    u = w.u[days].reshape(-1)
    smooth_eval = sign.evaluate_sign(rho=out["p_negative"], x_true=x_true, u=np.ones_like(u))
    filt_eval = sign.evaluate_sign(rho=filt["p_negative"], x_true=x_true, u=np.ones_like(u))
    pos = (x_true < 0).astype(float)
    ok = np.abs(x_true) >= sign.DEAD_BAND_POINTS
    rel = hmm.brier_reliability(prob=out["p_negative"][ok], outcome=pos[ok])
    assert smooth_eval["auc"] > 0.97, smooth_eval
    assert smooth_eval["n_reverse"] > 100
    print(
        f"PASS hidden-sign world: smoothed area {smooth_eval['auc']:.3f}, filtered {filt_eval['auc']:.3f} (gap {smooth_eval['auc'] - filt_eval['auc']:+.3f}); Brier reliability {rel['reliability']:.4f}, brier {rel['brier']:.4f}"
    )
    q = hmm.signed_quantiles_of(post=post, spec=spec)
    assert (np.diff(q, axis=1) >= -1e-9).all(), "quantiles must be monotone"
    crps = hmm.signed_crps(quantiles=q, truth=x_true)
    print(
        f"PASS two-lobe distribution: signed CRPS {crps[ok].mean():.3f} on the well-identified world; quantiles monotone"
    )


def check_em_and_cost_gate() -> None:
    """Baum-Welch steps do not lower the likelihood, and the cost gate refuses an over-cap projection."""
    g = 10.0
    true = make_spec(n_max=2, unit_size=g, n_bins=9, stay=0.9, sigma_r=0.3)
    t = np.arange(2500)
    f = 5.0 * g + 1.0 * g * np.sin(2 * np.pi * t / 48.0)
    sample = worlds.simulate_switching(
        n_steps=len(t), f=f, p=np.zeros(len(t)), trans_n=true.trans_n, unit_size=g, pv_scale=1.0,
        phi=true.phi, sigma_r=true.sigma_r, q=true.q, sigma_e=true.sigma_e, df=true.df, seed=5,
    )  # fmt: skip
    spec = make_spec(n_max=2, unit_size=g, n_bins=9, stay=0.6, sigma_r=0.3)
    last = -np.inf
    for it in range(4):
        post = hmm.forward_backward(spec=spec, m=sample.m, f=f, p=np.zeros(len(t)))
        assert post.log_likelihood >= last - 1e-6, (it, post.log_likelihood, last)
        last = post.log_likelihood
        spec.trans_n = hmm.update_trans_n(xi_n=post.xi_n)
    stay = float(np.mean(np.diag(spec.trans_n)))
    assert stay > 0.8, stay
    gate = hmm.CostGate(cap_hours=1e-6)
    gate.measure(
        run_one_pass=lambda: hmm.forward_backward(
            spec=spec, m=sample.m[:200], f=f[:200], p=np.zeros(200)
        )
    )
    try:
        gate.check(n_passes_total=10_000)
    except RuntimeError:
        print(
            f"PASS EM and cost gate: loglik rose to {last:.1f}, mean stay {stay:.2f} (true 0.9); gate refused {gate.projected_hours(n_passes_total=10_000):.3f} core-hours"
        )
        return
    msg = "the cost gate did not refuse"
    raise AssertionError(msg)


def check_gaussian_emission_differs() -> None:
    """The Student-t emission and the Gaussian check differ on an outlier reading."""
    spec = make_spec(n_max=0, unit_size=0.0, n_bins=7)
    gauss = make_spec(n_max=0, unit_size=0.0, n_bins=7)
    gauss.df = 0.0
    t_len = 40
    f = np.full(t_len, 10.0)
    p = np.zeros(t_len)
    m = np.full(t_len, 10.0)
    m[20] = 25.0
    a = hmm.forward_backward(spec=spec, m=m, f=f, p=p, want_xi=False).log_likelihood
    b = hmm.forward_backward(spec=gauss, m=m, f=f, p=p, want_xi=False).log_likelihood
    assert a > b + 5.0, (a, b)
    print(
        f"PASS emission: a 15-point outlier gives log-likelihood {a:.1f} (Student-t) against {b:.1f} (Gaussian)"
    )


def check_mixture_alignment_and_bins() -> None:
    """The mixture's own P(X < 0) equals the posterior's; the 81-bin rule is reachable."""
    from types import SimpleNamespace

    x = np.linspace(-3.0, 3.0, 13)
    rng = np.random.default_rng(3)
    gamma = rng.dirichlet(np.ones(13) * 0.3, size=40)
    post: Any = SimpleNamespace(smoothed=gamma, filtered=gamma, x_state=x)
    spec: Any = SimpleNamespace(spacing=0.5, sigma_e=0.4)
    pi, mu, sigma = hmm.two_lobe_mixture(post=post, spec=spec)
    target = (gamma * (x < 0)).sum(axis=1)
    assert (
        np.abs(hmm.mixture_p_negative(pi=pi, mu=mu, sigma=sigma) - target).max() > 0.01
    )  # untruncated lobes leak across zero
    assert np.abs(pi[:, 0] - target).max() < 1e-9
    q = hmm.truncated_signed_quantiles(pi=pi, mu=mu, sigma=sigma)
    tau = np.asarray(hmm.fold.TAUS_11)
    below = (q < 0.0).mean(axis=0)
    # at every level the share of rows with a negative quantile matches the posterior's mass below zero
    for k, t in enumerate(tau):
        expect = float((target > t).mean())
        assert abs(below[k] - expect) < 0.03, (t, below[k], expect)
    import e00720_fit as fit

    assert len(fit.k0_spec(phi=0.99, sigma_r=1.0, q=2.0).grid) == 81
    # the fit path: a residual with phi near 0.995 reaches the 81-bin grid through fit_residual
    rng_f = np.random.default_rng(5)
    ar_f = np.zeros((60, 48))
    ar_f[:, 0] = rng_f.normal(0, 1.0, 60)
    for t in range(1, 48):
        ar_f[:, t] = 0.995 * ar_f[:, t - 1] + rng_f.normal(0, 0.1, 60)
    phi_f, sigma_f = fit.fit_residual(
        g=ar_f, f=np.zeros_like(ar_f), rows=np.ones_like(ar_f, dtype=bool)
    )
    assert phi_f > fit.PHI_MAX, phi_f
    assert len(fit.k0_spec(phi=phi_f, sigma_r=sigma_f, q=2.0).grid) == 81
    assert len(fit.k0_spec(phi=0.9, sigma_r=1.0, q=2.0).grid) == 41


def main() -> int:
    """Run every check; return 0 when all pass."""
    for check in (
        check_brute_force,
        check_alias_toy,
        check_oracle_simulation,
        check_hidden_sign_world,
        check_em_and_cost_gate,
        check_gaussian_emission_differs,
        check_mixture_alignment_and_bins,
    ):
        check()
    print("rung 0e core (synthetic): all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
