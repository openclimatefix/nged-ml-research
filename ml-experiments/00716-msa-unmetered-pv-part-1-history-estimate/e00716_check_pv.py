# ruff: noqa: E501, PT018, ISC004, C416
"""Experiment 716: the checks of `e00716_pv.py` that need no fit (KF3, KF4, KF5; design section 8.1).

Usage: run.sh e00716_check_pv.py [synthetic|capacity|all]

`synthetic` runs on built arrays (no stored data except the constant `u99`); `capacity` reads the
stored capacity tables of experiments 649, 662, and 673 and prints KF5.
"""

import sys

import e00716_pv as pv
import numpy as np


def synthetic_raw(*, n_grid: int = 48 * 120, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return a raw table (16, G, 28) with a sunny-and-dull sky and a doy vector, for the checks."""
    rng = np.random.default_rng(seed)
    raw = np.zeros((16, n_grid, 28), dtype=np.float32)
    steps = np.arange(n_grid)
    hour = (steps % 48) / 2.0
    elevation = np.sin(np.pi * np.clip((hour - 4.0) / 16.0, 0.0, 1.0))
    cosz = np.clip(0.2 + 0.7 * elevation, -1.0, 1.0) * (elevation > 0) - 0.3 * (elevation <= 0)
    azimuth = np.pi + (hour - 12.0) / 12.0 * np.pi / 2
    sky = rng.uniform(0.2, 1.0, size=(16, n_grid // 48))
    sky = np.repeat(sky, 48, axis=1)
    for i in range(16):
        raw[i, :, 0] = (1000.0 * np.clip(cosz, 0.0, None) * sky[i]).astype(np.float32)
        raw[i, :, 2] = 8.0 + 6.0 * np.sin(steps / 48 * 2 * np.pi / 60)
        raw[i, :, 7] = cosz
        raw[i, :, 8] = np.sin(azimuth)
        raw[i, :, 9] = np.cos(azimuth)
    doy = (100.0 + steps // 48).astype(np.float64)
    return raw, doy


def kf4_estimate() -> list[str]:
    """KF4 on built arrays: `p` equals an independent computation, is zero at night, and the clipped share prints."""
    raw, doy = synthetic_raw()
    caps = np.array([3.0] * 6 + [2.0] * 8 + [0.0, 0.0])
    power_cap = np.full(16, 40.0)
    std = np.full(16, 0.05)
    est = pv.dense_share(
        raw=raw, doy=doy, capacities_mw=caps, power_capacity_mw=power_cap, u99=0.879
    )
    p, ok = pv.p_dense(estimate=est, target_std=std)
    u, _ = pv.yield_curve(
        ghi_mean=raw[0, :, 0].astype(np.float64), ghi_sd=np.zeros(raw.shape[1]),
        cos_zenith=raw[0, :, 7].astype(np.float64), sin_azimuth=raw[0, :, 8].astype(np.float64),
        cos_azimuth=raw[0, :, 9].astype(np.float64), doy=doy, temperature_c=raw[0, :, 2].astype(np.float64), u99=0.879,
    )  # fmt: skip
    independent = np.clip(3.0 * u / 40.0 / 0.05, 0.0, 10.0)  # NumPy, no module code
    assert np.abs(p[0] - independent).max() <= 1e-5, np.abs(p[0] - independent).max()
    night = raw[0, :, 7] <= 0
    assert (p[0][night] == 0).all(), "p is not zero where the sun is below the horizon"
    assert not ok[14].any() and not ok[15].any() and (p[14:] == 0).all(), (
        "absent Primaries must be zero and flagged"
    )
    assert ok[:14].all()
    clipped = (3.0 * u / 40.0 / 0.05 > pv.P_CLIP).mean()
    assert p[0].max() > 0.5, "the synthetic sky produced no PV"
    return [
        f"KF4: p matches an independent NumPy computation to 1e-5 (max p {p[0].max():.3f}); zero at night; "
        f"D15 and D18 absent; share of tokens clipped at 10 on the synthetic sky: {clipped:.4f}"
    ]  # fmt: skip


def kf3_placebos() -> list[str]:
    """KF3 on built arrays: the shifted sky, the target donor rule, the partner table, and determinism."""
    out = []
    raw, doy = synthetic_raw()
    caps = np.full(16, 3.0)
    power_cap = np.full(16, 40.0)
    std = np.full(16, 0.05)
    # a shift of 0 days reproduces H wherever neither the cap nor the low-sun rule binds
    g0, t0, missing0, binds0 = pv.shifted_sky(raw=raw, shift_days=0)
    assert not missing0[:, : raw.shape[1]].any()
    base = np.array(raw[:, :, 0], dtype=np.float64)
    free = ~binds0 & (
        pv.clear_sky(cos_zenith=np.asarray(raw[:, :, 7], dtype=np.float64)) >= pv.LOW_SUN_W_M2
    )
    assert np.abs((g0 - base)[free]).max() <= 1e-6, np.abs((g0 - base)[free]).max()
    est_h = pv.dense_share(
        raw=raw, doy=doy, capacities_mw=caps, power_capacity_mw=power_cap, u99=0.879
    )
    est_0 = pv.dense_share(
        raw=raw,
        doy=doy,
        capacities_mw=caps,
        power_capacity_mw=power_cap,
        ghi_override=g0,
        temperature_override=t0,
        u99=0.879,
    )
    p_h, _ = pv.p_dense(estimate=est_h, target_std=std)
    p_0, _ = pv.p_dense(estimate=est_0, target_std=std)
    out.append(
        f"KF3 identity: the H-S builder at a shift of 0 days reproduces H to {np.abs((p_0 - p_h)[:, :][free]).max():.1e} "
        f"where neither rule binds; the share of tokens where the cap or the low-sun rule binds: {binds0.mean():.4f}"
    )  # fmt: skip
    # a shift of 28 days: the donor of every token lies outside ages 2 to 15 (it is 28 days older)
    g28, _, missing28, binds28 = pv.shifted_sky(raw=raw, shift_days=pv.SHIFT_DAYS)
    first = pv.SHIFT_DAYS * 48
    assert missing28[:, :first].all() and not missing28[:, first : raw.shape[1]].any()
    again, _, _, _ = pv.shifted_sky(raw=raw, shift_days=pv.SHIFT_DAYS)
    assert np.array_equal(g28, again), "the shifted sky is not deterministic"
    donor_age = pv.SHIFT_DAYS
    assert donor_age > 15, "a donor must lie outside the history window (ages 2 to 15)"
    out.append(
        f"KF3 H-S: tokens before day 28 keep their own sky ({missing28.mean():.4f} of all tokens); cap or low-sun binds on {binds28.mean():.4f}"
    )
    # the HT-S donor rule
    days = np.array([100, 110, 127, 128, 150])
    donor, early = pv.donor_origin_day(origin_day=days, first_ens_day=100)
    assert donor.tolist() == [128, 138, 155, 100, 122] and early.tolist() == [
        True,
        True,
        True,
        False,
        False,
    ], (donor, early)
    assert (donor[~early] == days[~early] - 28).all() and (donor[early] == days[early] + 28).all()
    out.append(
        "KF3 HT-S: the donor run is 28 days before the origin's run, and 28 days later for the first 28 days of the record (listed by the flag)"
    )
    # the partner table
    table = pv.partner_table()
    expected = {
        "D12": "D14",
        "D22": "D11",
        "D5": "D19",
        "D10": "D21",
        "D2": "D8",
        "D16": "D3",
        "D9": "D4",
    }
    assert all(table[k] == v and table[v] == k for k, v in expected.items()) and len(table) == 14
    sites = [f"D{i}" for i in range(1, 17)]
    sites = [
        "D5",
        "D9",
        "D10",
        "D11",
        "D14",
        "D16",
        "D2",
        "D3",
        "D4",
        "D8",
        "D12",
        "D19",
        "D21",
        "D22",
        "D15",
        "D18",
    ]
    idx = pv.host_partner_index(sites=sites)
    assert idx[sites.index("D15")] == -1 and idx[sites.index("D18")] == -1
    assert sites[idx[sites.index("D12")]] == "D14"
    share = est_h.share
    p_x, ok_x = pv.partner_p(share=share, ok=est_h.ok, partner=idx, target_std=std)
    assert not ok_x[sites.index("D15")].any()
    assert np.array_equal(
        p_x, pv.partner_p(share=share, ok=est_h.ok, partner=idx, target_std=std)[0]
    )
    out.append(
        "KF3 H-X: the seven pairs equal section 5; the draw is identical on a second call; D15 and D18 have no partner"
    )
    # the transfer rule: a donor sky moved to another sun position keeps the clear-sky index
    mean = np.linspace(0, 600, 48)[None]
    cl_d = pv.clear_sky(cos_zenith=np.clip(np.sin(np.linspace(0, np.pi, 48)), -1, 1))[None]
    res = pv.transfer_sky(donor_mean=mean, donor_sd=0.1 * mean, clear_donor=cl_d, clear_own=cl_d)
    ok = ~res.cap_binds & (cl_d >= pv.LOW_SUN_W_M2)
    assert np.allclose(res.mean[ok], mean[ok], atol=1e-6) and np.allclose(
        res.sd[ok], 0.1 * mean[ok], atol=1e-6
    )
    return out


def kf5_capacity() -> list[str]:
    """KF5: the as-of rule, the printed capacities of every fold, and the reproduction of experiment 673's `probe` column."""
    import e00673_deliverable as dl  # ty: ignore[unresolved-import]

    assert pv.latest_cutoff(stored=pv.FOLD_CUTOFFS_649, fold="2025-11") == ("2025-10", False)
    assert pv.latest_cutoff(stored=pv.FOLD_CUTOFFS_649, fold="2026-02") == ("2025-10", False)
    assert pv.latest_cutoff(stored=pv.FOLD_CUTOFFS_649, fold="2026-05") == ("2026-03", False)
    assert pv.latest_cutoff(stored=pv.FOLD_CUTOFFS_649, fold="2026-06") == ("2026-06", False)
    assert pv.latest_cutoff(stored=pv.CUTOFFS_662, fold="2025-11") == ("2026-03", True)
    assert pv.latest_cutoff(stored=pv.CUTOFFS_662, fold="2026-05") == ("2026-03", False)
    t649, t662 = pv.read_tables()
    sites = [s for s in (*pv.SIX, *pv.EIGHT, *pv.ABSENT)]
    out = []
    ahead_six = 0
    for fold in (
        "2025-11",
        "2025-12",
        "2026-01",
        "2026-02",
        "2026-03",
        "2026-04",
        "2026-05",
        "2026-06",
    ):
        caps = [pv.as_of_capacity(site=s, fold=fold, table_649=t649, table_662=t662) for s in sites]
        out.extend(pv.lookahead_statement(caps=caps))
        ahead_six += sum(c.lookahead_probes for c in caps if c.site in pv.SIX)
        assert all(c.mw > 0 for c in caps if c.site not in pv.ABSENT)
        assert all(c.mw == 0 and c.source is None for c in caps if c.site in pv.ABSENT)
        eight = [c for c in caps if c.site in pv.EIGHT]
        assert all(c.lookahead_factors == (fold < "2026-06") for c in eight)
        assert all(c.lookahead_probes == (fold < "2026-03") for c in eight)
    assert ahead_six == 0, "the six look ahead nowhere"
    # reproduction of experiment 673's probe column with cutoff 2026-06
    theirs = dl.probe_table().select("site", "probe").to_dicts()
    copy = pv.probe_table_at_cutoff(cutoff="2026-06", table_649=t649, table_662=t662)
    worst = max(abs(copy[r["site"]] - r["probe"]) for r in theirs)
    assert worst <= 1e-9, worst
    # the as-of capacity at fold 2026-06 equals the probe column
    last = {
        c.site: c.mw
        for c in (
            pv.as_of_capacity(site=s, fold="2026-06", table_649=t649, table_662=t662)
            for s in sites[:14]
        )
    }
    assert max(abs(last[r["site"]] - r["probe"]) for r in theirs if r["site"] in last) <= 1e-9
    out.append(
        f"KF5: the copied probe_table at cutoff 2026-06 reproduces experiment 673's probe column to {worst:.1e}"
    )
    return out


def kf3c_target_donor() -> list[str]:
    """Check `target_estimate` and its `HT-S` donor: identity, donor-only dependence, and missing donors."""
    rng = np.random.default_rng(5)
    n_days, first_day = 40, 200
    n = pv.N_ENS_STEPS
    hour = (np.arange(n) % 48) / 2.0
    elevation = np.sin(np.pi * np.clip((hour - 4.0) / 16.0, 0.0, 1.0))
    day_factor = 0.5 + 0.4 * rng.uniform(size=(n_days, 1))
    cosz = np.clip(day_factor * elevation[None, :], 0.0, 1.0) - 0.2 * (elevation[None, :] <= 0)
    clear = pv.clear_sky(cos_zenith=cosz)
    values = np.zeros((16, n_days, n, 4), dtype=np.float32)
    azimuth = np.zeros((16, n_days, n, 2), dtype=np.float32)
    for i in range(16):
        values[i, ..., 0] = clear * rng.uniform(0.2, 1.0, size=(n_days, 1))
        values[i, ..., 1] = 0.1 * values[i, ..., 0]
        values[i, ..., 2] = 9.0
        values[i, ..., 3] = cosz
        azimuth[i, ..., 0] = 0.3
        azimuth[i, ..., 1] = -0.9
    filled = np.ones((16, n_days, n), dtype=bool)
    raw = pv.TargetRaw(
        values=values,
        azimuth=azimuth,
        filled=filled,
        first_day=first_day,
        run_us=np.ones((16, n_days), np.int64),
    )
    doy = pv.target_doy(first_day=first_day, n_days=n_days)
    scale = np.full(16, 0.8)
    scale[15] = 0.0
    std = np.full(16, 0.3)
    real = pv.target_estimate(raw=raw, doy_of_day=doy, share_scale=scale, target_std=std)
    same = pv.target_estimate(
        raw=raw, doy_of_day=doy, share_scale=scale, target_std=std, donor_day=np.arange(n_days)
    )
    assert np.array_equal(real.filled, same.filled)
    assert np.abs(real.p0 - same.p0).max() <= 1e-4, np.abs(real.p0 - same.p0).max()
    assert not real.filled[15].any() and real.filled[:15].all(), (
        "an absent Primary must have no target-row estimate"
    )
    donor = np.arange(n_days) - 28
    donor[:28] = np.arange(28) + 28  # the first 28 origin days read the run 28 days later
    shifted = pv.target_estimate(
        raw=raw, doy_of_day=doy, share_scale=scale, target_std=std, donor_day=donor
    )
    poisoned = raw.values.copy()
    poisoned[:, 30:, :, 0] = (
        1e6  # the own-day sky of days 30 and later; days 30 to 39 have donors 2 to 11
    )
    raw_p = pv.TargetRaw(
        values=poisoned, azimuth=azimuth, filled=filled, first_day=first_day, run_us=raw.run_us
    )
    shifted_p = pv.target_estimate(
        raw=raw_p, doy_of_day=doy, share_scale=scale, target_std=std, donor_day=donor
    )
    # Day 30's own sky is poisoned, but day 30 reads donor day 2, which is not poisoned: its estimate must not change.
    assert shifted.filled[:15, 30:].all() and np.array_equal(
        shifted.p0[:15, 30:], shifted_p.p0[:15, 30:]
    ), "the HT-S estimate depends on the own-day sky"
    assert shifted.filled[:15, :12].all() and not shifted.filled[:15, 12:28].any(), (
        "origins whose later donor lies past the table must be unfilled"
    )
    # Negative control: days 2 to 11 serve as donors of days 30 to 39 and are not poisoned; the real estimate of day 30 does change.
    real_p = pv.target_estimate(raw=raw_p, doy_of_day=doy, share_scale=scale, target_std=std)
    assert np.abs(real_p.p0[:15, 30:] - real.p0[:15, 30:]).max() > 1.0, (
        "negative control: poisoning must move the real estimate"
    )
    # Donors beyond the table are not filled.
    short = pv.target_estimate(
        raw=raw, doy_of_day=doy, share_scale=scale, target_std=std, donor_day=np.arange(n_days) + 35
    )
    assert not short.filled[:15, 5:].any() and short.filled[:15, :5].all(), short.filled[:15].sum(
        axis=(1, 2)
    )
    return [
        "KF3c: the target-row estimate equals itself under an identity donor to 1e-4; a poisoned own-day sky leaves the HT-S estimate unchanged; a missing donor leaves the row unfilled"
    ]


def main() -> int:
    """Run the requested checks and print every line."""
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    lines: list[str] = []
    if which in {"synthetic", "all"}:
        lines += kf4_estimate() + kf3_placebos() + kf3c_target_donor()
    if which in {"capacity", "all"}:
        lines += kf5_capacity()
    print("\n".join(lines))
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
