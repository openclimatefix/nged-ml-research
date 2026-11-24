# ruff: noqa: ANN001
"""Sensors, stretches, concordance, injection, and closure fits on synthetic arrays."""

from collections.abc import Sequence

import e00722_core as core722  # ty: ignore[unresolved-import]
import e00723_core as core
import e00723_score as score
import e00723_world as world
import numpy as np
import polars as pl
import pytest

N_DAYS = 120
N = N_DAYS * 48


def lit_mask() -> np.ndarray:
    hh = np.arange(N) % 48
    return (hh >= 14) & (hh <= 34)


def u_clear_arc() -> np.ndarray:
    hh = np.arange(N) % 48
    return np.where(lit_mask(), np.sin(np.pi * (hh - 13) / 22.0).clip(0) * 800.0 + 1.0, 0.0)


def test_a_clear_day_leaves_no_fast_change_after_the_arc_is_removed():
    u = u_clear_arc()
    z = 0.7 * u / 800.0  # a farm that follows the arc at a constant clear-sky index
    dz = world.fast_sensor_change(z=z, u_clear=u / 800.0, lit=lit_mask())
    assert np.nanmax(np.abs(dz)) < 1e-9
    r, _ = world.arc_removal_r(
        dz=dz + 1e-6 * np.arange(N) % 1e-3, d_u_clear=world.arc_change(u_clear=u, lit=lit_mask())
    )
    assert not np.isfinite(r) or abs(r) < 0.2


def test_a_farm_that_ramps_with_the_arc_fails_the_arc_condition():
    lit = lit_mask()
    u = u_clear_arc()
    rng = np.random.default_rng(1)
    z = np.where(lit, 0.5 * u / 800.0 + 0.0, 0.0)
    # The raw change of a farm with a steeper ramp than the day ratio removes keeps the arc.
    steep = z * (1.0 + 0.6 * np.sin((np.arange(N) % 48) / 48.0 * np.pi))
    dz = world.fast_sensor_change(z=steep, u_clear=u / 800.0, lit=lit)
    r, _ = world.arc_removal_r(dz=dz, d_u_clear=world.arc_change(u_clear=u, lit=lit))
    assert not core.arc_removal_ok(r=r)
    noise = z + rng.normal(0, 0.05, N) * lit
    dz2 = world.fast_sensor_change(z=noise, u_clear=u / 800.0, lit=lit)
    r2, _ = world.arc_removal_r(dz=dz2, d_u_clear=world.arc_change(u_clear=u, lit=lit))
    assert core.arc_removal_ok(r=r2)


def test_resolution_shares_of_an_hourly_index_and_a_half_hourly_arc():
    lit = lit_mask()
    k = np.repeat(np.linspace(0.3, 0.9, N // 2), 2)
    u = u_clear_arc()
    res = world.resolution_shares(k=k, u_clear=u, lit=lit)
    assert res["share_index_unchanged"] == 1.0 or res["share_index_unchanged"] == 0.0
    assert res["share_u_clear_unchanged"] < core.ARC_VOID_SHARE


def test_the_trailing_capacity_is_the_95th_percentile_of_the_previous_91_days():
    lit = lit_mask()
    u = u_clear_arc()
    z = np.where(lit, 0.5 * u, 0.0) / 800.0
    c = world.trailing_capacity(z=z, u_clear=u / 800.0, lit=lit, n_days=N_DAYS)
    assert np.isnan(c[0])
    assert c[1:] == pytest.approx(0.5, rel=1e-6)


def test_coherence_drops_a_farm_that_does_not_follow_the_others():
    rng = np.random.default_rng(2)
    common = rng.normal(0, 1, N)
    dz_of = {
        "F1": common + rng.normal(0, 0.1, N),
        "F2": common + rng.normal(0, 0.1, N),
        "F3": rng.normal(0, 1, N),
    }
    keep, r_of = world.keep_coherent(dz_of=dz_of, in_season=np.ones(N, dtype=bool))
    assert keep == ["F1", "F2"]
    assert r_of["F1"] > core.HALF  # F1 against the mean of F2 and F3 is diluted by F3
    assert r_of["F3"] < core.HALF


def test_the_sensor_shifted_by_28_days_reads_the_sensor_of_28_days_earlier():
    sensor = world.SensorChange(dz=np.arange(N, dtype=float), dk=np.arange(N, dtype=float) * 2)
    shifted = world.shifted_sensor(sensor=sensor)
    assert shifted.dz[28 * 48] == 0.0
    assert shifted.dz[28 * 48 + 5] == 5.0
    assert np.isnan(shifted.dz[:48]).all()
    assert shifted.dk[28 * 48 + 3] == 6.0


def test_fold_is_even_in_the_sign_and_is_at_least_the_floor():
    x = np.array([np.nan, 5.0, -5.0, 5.0, -5.0, 0.0, 0.0])
    f = world.fold(signed_mw=x, capacity_mw=100.0)
    q = 2.0
    assert np.isnan(f[0])
    assert f[5] > np.sqrt(0 + q**2) * 0.99
    assert f[5] == pytest.approx(world.fold(signed_mw=-x, capacity_mw=100.0)[5])
    assert np.isnan(f[1])
    # A crossing half-hour (+5 to -5) averages the magnitude: far above the floor.
    crossing = world.fold(signed_mw=np.array([5.0, -5.0]), capacity_mw=100.0)[1]
    assert crossing > 1.5 * q
    assert crossing == pytest.approx(
        np.mean([np.sqrt(x**2 + q**2) for x in 5.0 - 10.0 * np.arange(1, 7) / 6])
    )


def test_injection_scale_matches_the_level_times_the_hosts_median_on_the_reference_rows():
    rng = np.random.default_rng(3)
    host = rng.normal(-3, 0.2, N)
    farm = np.abs(rng.normal(4, 0.3, N))
    rows = np.zeros(N, dtype=bool)
    rows[100:400] = True
    for level in core.LEVELS:
        a = world.injection_scale(host_signed_mw=host, farm_mw=farm, rows=rows, level=level)
        injected = world.inject(signed_mw=host, farm_mw=farm, scale=a)
        assert np.median(a * farm[rows]) == pytest.approx(level * np.median(np.abs(host[rows])))
        assert np.median(injected[rows]) < np.median(host[rows])


def test_level_reference_rows_are_sunny_summer_middays_before_the_cut():
    month = world.month_of_rows(n=N)
    hour = world.utc_hour_of_rows(n=N)
    k = np.full(N, 0.9)
    rows = world.level_reference_rows(k=k, month=month, hour=hour, before_row=N // 2)
    assert not rows[N // 2 :].any()
    assert (hour[rows] >= 11).all()
    assert (hour[rows] < 14).all()
    assert set(month[rows]) <= set(core.SUMMER_MONTHS)


def test_window_mask_drops_a_change_with_an_end_inside_the_window_or_within_one_hour():
    mask = world.window_mask(windows=[(100, 110)], n=300)
    assert mask[98:113].all()  # both ends of the change must be clear of [98, 112] to stay
    assert mask[98]  # row 98 is the end of the change 97 -> 98, inside the 1-hour margin
    assert not mask[96]
    assert mask[113]  # the change 112 -> 113 has the end 112 in the margin
    assert not mask[114]
    assert not world.window_mask(windows=[], n=50).any()


def test_hourly_mask_and_closure_rows():
    masked = np.zeros(20, dtype=bool)
    masked[7] = True
    hm = world.hourly_mask(masked=masked)
    assert hm[7]
    assert hm[8]
    assert not hm[9]
    zenith = np.full(96, 60.0)
    rows = world.closure_rows(zenith=zenith, masked=np.zeros(96, dtype=bool), start=48, kind="lit")
    assert rows[48:].sum() == 24
    assert not rows[:48].any()
    assert (np.flatnonzero(rows) % 2 == 1).all()
    assert not world.closure_rows(
        zenith=zenith, masked=np.zeros(96, dtype=bool), start=0, kind="dark"
    ).any()
    with_30 = world.closure_rows(
        zenith=zenith, masked=np.zeros(96, dtype=bool), start=0, kind="lit", min_elevation_deg=40.0
    )
    assert not with_30.any()


def test_trailing_overcast_median_uses_matching_overcast_days_only():
    key = np.arange(N) % 48
    key = (key * 13 + 1) * 2
    change = np.zeros(N)
    overcast = np.zeros(N, dtype=bool)
    for d in (3, 5, 8):
        overcast[d * 48 : (d + 1) * 48] = True
        change[d * 48 : (d + 1) * 48] = float(d)
    rows = np.array([10 * 48 + 20, 6 * 48 + 20])
    med = world.trailing_overcast_median(change=change, key=key, overcast_row=overcast, rows=rows)
    assert (
        med[0] == 5.0
    )  # ages 2 to 7 hold days 3, 5, 8 (age 7 is day 3, age 5 is day 5, age 2 is day 8)
    assert np.isnan(
        world.trailing_overcast_median(
            change=change, key=key, overcast_row=np.zeros(N, dtype=bool), rows=rows
        )
    ).all()


def synthetic_world(*, sign: float, seed: int = 4) -> tuple[dict, np.ndarray, world.SensorChange]:
    """A day of stretches: the reading's fast change follows the sensor with `sign` plus noise."""
    rng = np.random.default_rng(seed)
    n_pairs = 40
    dz = rng.normal(0, 1, n_pairs + 1)
    dk = np.where(rng.random(n_pairs + 1) > 0.5, 0.3, -0.3)
    dm = sign * 2.0 * dz + rng.normal(0, 0.05, n_pairs + 1)
    return {0: [(0, n_pairs, True)]}, dm, world.SensorChange(dz=dz, dk=dk)


@pytest.mark.parametrize(("sign", "expected"), [(1.0, "export"), (-1.0, "import")])
def test_concordance_has_the_sign_of_the_reading_against_the_sky(sign, expected):
    stretches, dm, sensor = synthetic_world(sign=sign)
    signed = np.full(len(dm), -5.0 * sign)
    table = world.stretch_table(
        stretches=stretches,
        signed_mw=signed,
        dm=dm,
        sensors={"farm": sensor},
        masked=np.zeros(len(dm), dtype=bool),
    )
    row = table.row(0, named=True)
    assert row["truth"] == expected
    assert core.evidence_of(c=row["c_farm"], n_usable=row["n_farm"]) == expected
    assert row["n_farm"] >= core.MIN_USABLE_PAIRS
    # A sensor that was not supplied reads NaN and no pairs.
    assert np.isnan(row["c_meter"])
    assert row["n_meter"] == 0


def test_the_usable_pair_rule_needs_both_sizes_and_no_mask():
    sensor = world.SensorChange(dz=np.ones(4), dk=np.array([0.1, 0.09, 0.5, 0.5]))
    dm = np.array([0.2, 5.0, 0.19, 1.0])
    masked = np.array([False, False, False, True])
    use = world.usable_pairs(dm=dm, sensor=sensor, masked=masked)
    assert use.tolist() == [True, False, False, False]
    diag = world.threshold_diagnostics(dm=dm, sensor=sensor, masked=masked)
    assert diag["n_daylight_pairs"] == 3
    assert diag["share_both"] == pytest.approx(1 / 3)
    assert diag["threshold_index"] == 0.10
    assert diag["threshold_reading_points"] == 0.2


def test_scorable_keeps_post_bounce_stretches_of_one_stretch_days_with_8_usable_pairs():
    table = pl.DataFrame(
        {
            "day": [0, 1, 2, 3],
            "first": [0] * 4,
            "last": [5] * 4,
            "is_post_bounce": [True, True, False, True],
            "n_post_bounce_in_day": [1, 2, 0, 1],
            "truth": ["export"] * 4,
            "n_farm": [8, 20, 20, 7],
        }
    )
    kept = world.scorable(table=table, sensor="farm")
    assert kept["day"].to_list() == [0]
    assert world.scorable(table=table, sensor="farm", post_bounce_one_stretch_days=False)[
        "day"
    ].to_list() == [0, 1]


def test_closure_fit_recovers_a_slope_and_a_second_regressor(rng):
    n_days = 200
    n = n_days * 48
    key = np.ones(n, dtype=np.int64)
    quiet = np.ones(n, dtype=bool)
    x = np.cumsum(rng.normal(0, 1, n))
    z = np.cumsum(rng.normal(0, 1, n))
    y = 1.5 * x + 0.5 * z + rng.normal(0, 0.1, n)
    day = np.arange(n) // 48
    weights = core.day_weights(n_days=n_days, n_boot=200)
    rows = world.hour_end_rows(n=n)
    fit = world.closure_fit(
        response=y,
        regressors={"x": x, "z": z},
        rows=rows,
        key=key,
        quiet=quiet,
        day=day,
        weights=weights,
    )
    assert fit.get("x")[0] == pytest.approx(1.5, abs=0.05)
    assert fit.get("z")[0] == pytest.approx(0.5, abs=0.05)
    power = world.power_row(
        response_change=world.hourly_demeaned(x=y, key=key, quiet=quiet),
        regressor_change=world.hourly_demeaned(x=x, key=key, quiet=quiet),
        rows=rows,
    )
    assert power["n_rows"] > 1000
    assert power["predicted_se"] == pytest.approx(
        core.SE_FACTOR * power["sd_response"] / (power["sd_regressor"] * np.sqrt(power["n_rows"]))
    )


def fake_score(
    *,
    name: str,
    role: str,
    host: str,
    level: float | None,
    c: Sequence[float],
    truth: list[str],
    farm: str | None = "F1",
) -> score.WorldScore:
    n = len(c)
    table = pl.DataFrame(
        {
            "day": list(range(n)), "first": [0] * n, "last": [9] * n,
            "is_post_bounce": [True] * n, "n_post_bounce_in_day": [1] * n, "truth": truth,
            **{f"c_{s}": c for s in world.SENSOR_NAMES},
            **{f"n_{s}": [10] * n for s in world.SENSOR_NAMES},
        },
        schema=world.STRETCH_SCHEMA,
    )  # fmt: skip
    return score.WorldScore(
        score.WorldSpec(name, role, host, farm, level), table, {}, {}, (0, 0.9), 1.0, 0.0
    )


def test_pooled_area_separates_export_from_import_and_the_bootstrap_resamples_days():
    rng = np.random.default_rng(5)
    c_exp = [float(v) for v in rng.uniform(0.3, 1.0, 60)]
    c_imp = [float(v) for v in rng.uniform(-1.0, -0.3, 60)]
    scores = [
        fake_score(
            name="a",
            role="injected",
            host="D10",
            level=1.0,
            c=c_exp + c_imp,
            truth=["export"] * 60 + ["import"] * 60,
        ),
        fake_score(
            name="b",
            role="injected",
            host="D14",
            level=2.0,
            c=c_exp + c_imp,
            truth=["export"] * 60 + ["import"] * 60,
        ),
    ]
    rows = score.tagged(scores=scores, sensor="farm", roles=("injected",))
    assert rows.height == 240
    out = score.area_of(rows=rows, weights=core.day_weights(n_days=120, n_boot=200))
    assert out["area"] == 1.0
    assert out["n_export"] == 120
    assert out["n_import"] == 120
    assert out["se"] == pytest.approx(0.0, abs=1e-9)
    flat = score.area_of(rows=rows.filter(pl.col("level") == 1.0), weights=None)
    assert flat["area"] == 1.0
    assert score.time_draws(rows=rows, weights=core.day_weights(n_days=120, n_boot=20)) >= 0.0


def test_dead_band_population_keeps_every_stretch_and_alignment_reads_lag_zero():
    d15 = fake_score(
        name="d15_folded",
        role="d15_folded",
        host="D15",
        level=None,
        c=[0.0, 0.1, 0.9, -0.8],
        truth=["import"] * 4,
        farm=None,
    )
    out = score.dead_band_population(scores=[d15])
    assert out["n_stretches"] == 4
    assert out["n_no_evidence"] == 2
    ok = score.alignment(scores=[d15])
    assert ok["ok"]
    bad = fake_score(
        name="u", role="uninjected", host="D9", level=None, c=[0.0], truth=["import"], farm=None
    )
    bad.lag = (1, 0.8)
    assert not score.alignment(scores=[d15, bad])["ok"]


def test_the_floor_rule_on_a_control_population():
    rows = pl.DataFrame({"c": [0.9] + [-0.5] * 99})
    assert score.evidence_counts(rows=rows) == {
        "n": 100,
        "n_export_evidence": 1,
        "n_import_evidence": 99,
        "n_no_evidence": 0,
    }
    assert core.false_export_bar(n_export_evidence=1, n_scorable_import=100) == "pass"


def test_bootstrap_share_has_a_standard_error_over_days():
    rng = np.random.default_rng(6)
    day = np.repeat(np.arange(100), 3)
    hit = rng.random(300) < 0.7
    share, se = core.bootstrap_share(
        hit=hit, day=day, weights=core.day_weights(n_days=100, n_boot=300)
    )
    assert share == pytest.approx(hit.mean())
    assert 0.01 < se < 0.08
    assert np.isnan(
        core.bootstrap_share(
            hit=np.array([], dtype=bool),
            day=np.array([], dtype=int),
            weights=core.day_weights(n_days=10, n_boot=5),
        )[0]
    )


def test_weighted_auc_is_imported_from_experiment_722():
    assert (
        core722.weighted_auc(score=np.array([0.0, 1.0]), label=np.array([0, 1]), w=np.ones(2))
        == 1.0
    )
