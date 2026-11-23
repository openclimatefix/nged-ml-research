# ruff: noqa: E501, PT018, PT017, ISC004, C416
"""Experiment 716: the checks of `e00716_pv.py` that need no fit (KF3, KF4, KF5; design section 8.1).

Usage: run.sh e00716_check_pv.py [synthetic|capacity|all|kf14|kf4real MONTH] [--record]

`kf14` and `kf4real` read stored data (CPU, read only); `--record` writes the wiring record of the check that ran.

`synthetic` runs on built arrays (no stored data except the constant `u99`); `capacity` reads the
stored capacity tables of experiments 649, 662, and 673 and prints KF5.
"""

import sys

import e00716_arms as arms
import e00716_pv as pv
import numpy as np
from studies.sources import ML_EXPERIMENTS_DATA_DIR


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


# ------------------------------------------------------------------ KF14, KF6, and KF4's real-data half


def kf14_check(
    *, ghi: np.ndarray, cos_zenith_at: dict[int, np.ndarray], n_steps: int, floor_w_m2: float = 5.0
) -> dict[str, int]:
    """Count the stamps that contradict a period-end labelling, and the stamps that contradict a period-start labelling.

    `cos_zenith_at[k]` is the cosine of the zenith `k` grid steps (half-hours) after each stamp, for `k`
    from `-n_steps` to `n_steps`; `n_steps` is the length of the CAMS period in grid steps. A positive
    irradiance under a period-end label means the sun was up at some point of `(t - period, t]`, so a
    stamp with `ghi` above the floor while the sun is down at every grid step of that period is a
    contradiction (`end_violations`); the same test on `[t, t + period)` counts the contradictions of a
    period-start label (`start_violations`). The labelling is period end when the first count is zero and
    the second is not (a series with no sunrise or sunset in the sample cannot tell the two apart).
    """
    lit = ghi > floor_w_m2
    end_dark = np.ones(len(ghi), dtype=bool)
    start_dark = np.ones(len(ghi), dtype=bool)
    for k in range(-n_steps, 1):
        end_dark &= cos_zenith_at[k] <= 0.0
    for k in range(n_steps + 1):
        start_dark &= cos_zenith_at[k] <= 0.0
    return {
        "end_violations": int((lit & end_dark).sum()),
        "start_violations": int((lit & start_dark).sum()),
        "n_lit": int(lit.sum()),
    }


def kf14_correlations(
    *, ghi: np.ndarray, cos_zenith_at: dict[int, np.ndarray], n_steps: int
) -> dict[int, float]:
    """Return the correlation of `ghi` with the mean positive cosine of the zenith over the period ending `shift` grid steps after each stamp, for shifts -1, 0 and +1.

    The two-sided violation counts of `kf14_check` cannot tell a period-end label from a stamp 30
    minutes later, because both give no end violations. The period-end label is right when shift 0
    correlates best. `cos_zenith_at` must hold steps `-n_steps - 1` to `n_steps + 1`.
    """
    out = {}
    for shift in (-1, 0, 1):
        window = [np.clip(cos_zenith_at[k + shift], 0.0, None) for k in range(-n_steps + 1, 1)]
        out[shift] = float(np.corrcoef(ghi, np.mean(window, axis=0))[0, 1])
    return out


def kf14_synthetic() -> list[str]:
    """KF14 on a toy series: a period-end series passes, and the same series stamped at the period start fails."""
    n = 48 * 60
    steps = np.arange(n)
    cz = (
        np.sin(np.pi * np.clip(((steps % 48) / 2.0 - 5.0) / 14.0, 0.0, 1.0)) - 0.001
    )  # at or below zero while the sun is down
    period = 2

    def cz_of(offset: np.ndarray | int) -> np.ndarray:
        return np.roll(cz, -offset)  # cos zenith `offset` steps after each stamp

    def mean_over(lo: int, hi: int) -> np.ndarray:
        return np.mean([np.clip(cz_of(k), 0, None) for k in range(lo, hi + 1)], axis=0) * 1000.0

    at = {k: cz_of(k) for k in range(-period - 1, period + 2)}
    end_labelled = mean_over(-period + 1, 0)  # the stamp is the end of the period
    start_labelled = mean_over(0, period - 1)  # the stamp is the start of the period
    later = mean_over(
        -period, -1
    )  # a stamp one grid step (30 minutes) after the period end: the period ended before the stamp
    good = kf14_check(ghi=end_labelled, cos_zenith_at=at, n_steps=period)
    bad = kf14_check(ghi=start_labelled, cos_zenith_at=at, n_steps=period)
    assert good["end_violations"] == 0 and good["start_violations"] > 0, good
    assert bad["end_violations"] > 0, ("a period-start series must fail the check", bad)
    # the +30 minute control: it has no end violations, so only the correlation tells it from the period end
    shifted = kf14_check(ghi=later, cos_zenith_at=at, n_steps=period)
    assert shifted["end_violations"] == 0, shifted
    right = kf14_correlations(ghi=end_labelled, cos_zenith_at=at, n_steps=period)
    wrong = kf14_correlations(ghi=later, cos_zenith_at=at, n_steps=period)
    assert max(right, key=right.get) == 0, right  # ty: ignore[no-matching-overload]
    assert max(wrong, key=wrong.get) == -1, ("a +30 minute series must fail the correlation", wrong)  # ty: ignore[no-matching-overload]
    return [
        f"KF14 logic: a period-end series gives {good}, a period-start series gives {bad} (fails, as it must); a series stamped 30 minutes after the period end has no end violations but its best correlation is at shift -1 (the period ends a step before the stamp), not 0 ({wrong})"
    ]


def check_origin_hour() -> list[str]:
    """`target_raw_from_rows` accepts native origins at 09:00 UTC and refuses any other hour."""
    x = np.zeros((2, 28), dtype=np.float32)
    kwargs = {
        "x": x, "primary": np.array([0, 0]), "origin_idx": np.array([100, 100]), "target_idx": np.array([106, 107]),
        "mean": np.zeros(28), "std": np.ones(28), "first_day": 0, "n_days": 10,
    }  # fmt: skip
    day_us = 24 * 3_600_000_000
    pv.target_raw_from_rows(
        origin_us=np.array([9 * 3_600_000_000, 9 * 3_600_000_000 + day_us]),
        **kwargs,  # ty: ignore[invalid-argument-type]
    )
    try:
        pv.target_raw_from_rows(
            origin_us=np.array([10 * 3_600_000_000, 9 * 3_600_000_000]),
            **kwargs,  # ty: ignore[invalid-argument-type]
        )
    except AssertionError as err:
        assert "09:00" in str(err), err
    else:
        raise AssertionError("an origin at 10:00 UTC was accepted")
    return ["origin hour: native origins must be 09:00 UTC; a 10:00 origin is refused"]


def kf14_real() -> list[str]:
    """KF14 on the CAMS source file of experiment 707 (read only), with the sun's geometry of the dense table at the same stamps."""
    import e00397_prep as prep397  # ty: ignore[unresolved-import]
    import e00707_prep as prep707  # ty: ignore[unresolved-import]
    import e00716_world_rung as wr
    import polars as pl

    raw = np.load(
        wr.ML_EXPERIMENTS_DATA_DIR / wr.NAME_713_CACHE / "cache" / "dense_raw.npy", mmap_mode="r"
    )
    sites = [r["site"] for r in prep397.primaries().iter_rows(named=True)]
    cams = pl.read_parquet(prep707.CAMS_PATH)
    out = []
    for pos, site in enumerate(sites):
        frame = cams.filter(pl.col("site") == site).sort("time")
        if frame.is_empty():
            continue
        stamps_us = frame["time"].dt.epoch("us").to_numpy()
        period_us = int(np.median(np.diff(stamps_us)))
        n_steps = max(1, period_us // pv.STEP_US)
        idx = (stamps_us - pv.GRID_START_US) // pv.STEP_US
        keep = (idx >= n_steps + 1) & (idx < raw.shape[1] - n_steps - 1)
        idx = idx[keep]
        at = {
            k: np.asarray(raw[pos, idx + k, 7], dtype=np.float64)
            for k in range(-n_steps - 1, n_steps + 2)
        }
        res = kf14_check(
            ghi=frame["ghi_w_m2"].to_numpy()[keep].astype(np.float64),
            cos_zenith_at=at,
            n_steps=n_steps,
        )
        assert res["end_violations"] == 0 and res["start_violations"] > 0, (site, res)
        corr = kf14_correlations(
            ghi=frame["ghi_w_m2"].to_numpy()[keep].astype(np.float64),
            cos_zenith_at=at,
            n_steps=n_steps,
        )
        assert max(corr, key=corr.get) == 0, (
            site,
            "period end is not the best-correlated label",
            corr,
        )  # ty: ignore[no-matching-overload]
        out.append(
            f"KF14 {site}: period {period_us // 60_000_000} min, {res}; correlation with the mean cosine of the zenith over the period ending at shift -1, 0, +1 steps: {corr}"
        )
        break  # one site is enough: the stamp convention is a property of the file
    assert out, "no Primary's site is in the CAMS file"
    return out


def kf6_statements() -> list[str]:
    """KF6: experiment 713's row-by-row statement, then the source and issue time of every PV value for each fit family."""
    from datetime import UTC, datetime

    import e00713_layout as lay  # ty: ignore[unresolved-import]
    import e00713_weather_lookahead as la  # ty: ignore[unresolved-import]

    contract = datetime(2026, 3, 10, 12, tzinfo=UTC)
    out = [*la.statement(contract=contract, primary="D5"), ""]
    newest_p = -(
        lay.STEPS_PER_DAY * (lay.MIN_AGE - 1) + 1
    )  # the last stamp of the age-2 row, in steps from the contract origin
    assert newest_p < 0, "a history PV value would be read at or after the contract origin"
    hist = (
        "history rows (ages 2 to 15): p at each token's own stamp from the CAMS irradiance and ERA5 temperature analysis at that stamp "
        f"(no issue time; a lab shortcut for ages 2 to 5, as in 713), the as-of capacity table of the fold; newest stamp {newest_p} steps from the contract origin: before it, PASS"
    )
    run = "target row: p from the ECMWF ENS run of 00Z of the origin day, issued 9 hours before the native origin (before: yes, by the layout alone), PASS"
    for arm, lines in {
        "Z": ["no PV value is read"],
        "H": [hist],
        "H-S": [
            hist.replace(
                "at each token's own stamp", "28 days before each token's stamp (CAMS), a placebo"
            )
        ],
        "H-X": [
            hist.replace(
                "the as-of capacity table of the fold",
                "the partner Primary's p at the same stamp (a placebo)",
            )
        ],
        "HT": [hist, run],
        "HT-S": [
            hist,
            run.replace(
                "of the origin day",
                "of the day 28 days earlier (a placebo; the first 28 days of the ENS record use the run 28 days later)",
            ),
        ],
        "T": [run],
    }.items():
        out += [f"fit family {arm}:", *[f"  {x}" for x in lines]]
    return [*out, "KF6: PASS"]


def kf4_real(*, month: str = "2025-11") -> list[str]:
    """KF4's real-data half: the dense `u` against `e00634_common.unit_curve` at 1e-6, and the dense geometry columns against the Candidate's rows."""
    import e00634_common as c634  # ty: ignore[unresolved-import]
    import e00713_inputs as inp  # ty: ignore[unresolved-import]
    import e00716_arms as arms

    arms.install_own_cache()
    pipe = inp.load_pipeline(month=month, phase="train", chain_root=inp.chain_root_of_seed(seed=0))
    raw = pipe.d632.dense_raw()
    n_grid = raw.shape[1]
    doy = pv.day_of_year(grid_start_us=pv.GRID_START_US, n_grid=n_grid)
    times_us = pv.GRID_START_US + np.arange(n_grid, dtype=np.int64) * pv.STEP_US
    pos = np.arange(0, n_grid, 7)  # every 7th stamp is enough for the 1e-6 comparison
    worst = 0.0
    for prim in range(16):
        u, _ = pv.yield_curve(
            ghi_mean=raw[prim, :, 0].astype(np.float64), ghi_sd=np.zeros(n_grid), cos_zenith=raw[prim, :, 7].astype(np.float64),
            sin_azimuth=raw[prim, :, 8].astype(np.float64), cos_azimuth=raw[prim, :, 9].astype(np.float64), doy=doy,
            temperature_c=raw[prim, :, 2].astype(np.float64),
        )  # fmt: skip
        x = np.zeros((len(pos), 17))
        x[:, 14], x[:, 15], x[:, 16] = raw[prim, pos, 7], raw[prim, pos, 8], raw[prim, pos, 9]
        ref = c634.unit_curve(
            ghi=raw[prim, pos, 0].astype(np.float64),
            x=x,
            times_us=times_us[pos],
            temp=raw[prim, pos, 2].astype(np.float64),
        )
        worst = max(worst, float(np.abs(u[pos] - ref).max()))
    assert worst <= 1e-6, f"the dense u differs from e00634_common.unit_curve by {worst}"
    mr, _ = pipe_rows(pipe=pipe)
    std = mr.base.standardiser  # ty: ignore[unresolved-attribute]
    return [
        f"KF4 real: the dense u equals e00634_common.unit_curve to {worst:.2e} at every 7th stamp of every Primary, given the same shortwave, temperature and geometry",
        pv.check_dense_columns(raw=raw, x_rows=mr.x, grid_index=mr.target_idx, primary=mr.primary),  # ty: ignore[unresolved-attribute]
        pv.check_dense_weather(
            raw=raw,
            x_rows=mr.x,  # ty: ignore[unresolved-attribute]
            mean=std.mean,  # ty: ignore[unresolved-attribute]
            std=std.std,  # ty: ignore[unresolved-attribute]
            grid_index=mr.target_idx,  # ty: ignore[unresolved-attribute]
            primary=mr.primary,  # ty: ignore[unresolved-attribute]
        ),
        "KF4 real cannot show that the dense shortwave and temperature values are right: both sides of the u comparison read the dense table, and the independent rows hold forecasts, so only the geometry columns are compared for equality and the weather columns for correlation",
    ]


def pipe_rows(*, pipe: object) -> tuple[object, object]:
    """Return the Candidate rows of the fold (`mr`, `test`) from 713's adapter."""
    import e00713_inputs as inp  # ty: ignore[unresolved-import]

    data = inp.build_fold_data(pipe=pipe, device="cpu", keep_rows=True)  # ty: ignore[invalid-argument-type]
    return data.rows


def main() -> int:
    """Run the requested checks and print every line."""
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    lines: list[str] = []
    if which in {"synthetic", "all"}:
        lines += (
            kf4_estimate()
            + kf3_placebos()
            + kf3c_target_donor()
            + kf14_synthetic()
            + kf6_statements()
            + check_origin_hour()
        )
    if which == "kf14":
        lines += kf14_real()
    if which == "kf4real":
        lines += kf4_real(
            month=sys.argv[2]
            if len(sys.argv) > 2 and not sys.argv[2].startswith("--")
            else "2025-11"
        )
    if which in {"capacity", "all"}:
        lines += kf5_capacity()
    print("\n".join(lines))
    print("PASS")
    if "--record" in sys.argv:
        names = {
            "synthetic": ("KF3",),
            "capacity": ("KF5",),
            "kf14": ("KF14",),
            "kf4real": ("KF4",),
        }
        if which not in names:
            print(f"--record: `{which}` is not a recorded check, so no wiring record was written")
        for name in names.get(which, ()):
            arms.write_wiring_record(
                data_dir=ML_EXPERIMENTS_DATA_DIR,
                name=name,
                passed=True,
                fold=sys.argv[2] if name in arms.FOLD_CHECKS else None,
            )
        if which == "synthetic":
            arms.write_wiring_record(data_dir=ML_EXPERIMENTS_DATA_DIR, name="KF6", passed=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
