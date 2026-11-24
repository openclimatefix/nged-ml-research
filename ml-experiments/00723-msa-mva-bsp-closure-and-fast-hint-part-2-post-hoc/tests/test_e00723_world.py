# ruff: noqa: ANN001
"""Sensors, stretches, concordance, injection, and closure fits on synthetic arrays."""

from collections.abc import Sequence
from types import SimpleNamespace

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


def test_the_trailing_capacity_accepts_the_real_grid_that_ends_part_way_through_its_last_day():
    n_days = (core.N_GRID + 47) // 48
    assert core.N_GRID % 48 != 0  # the master grid is one half-hour short of whole days
    hh = np.arange(core.N_GRID) % 48
    lit = (hh >= 14) & (hh <= 34)
    u = np.where(lit, np.sin(np.pi * (hh - 13) / 22.0).clip(0) * 800.0 + 1.0, 0.0)
    z = np.where(lit, 0.5 * u, 0.0) / 800.0
    c = world.trailing_capacity(z=z, u_clear=u / 800.0, lit=lit, n_days=n_days)
    assert len(c) == n_days
    assert c[1:] == pytest.approx(0.5, rel=1e-6)
    sensor = world.farm_sensor(z=z, u_clear=u / 800.0, lit=lit)
    assert len(sensor.dz) == len(sensor.dk) == core.N_GRID


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
    # RULED: only dz is shifted; dk stays the real sensor's, so the usable pairs are the real ones.
    assert shifted.dk[28 * 48 + 3] == sensor.dk[28 * 48 + 3]
    assert (shifted.dk == sensor.dk).all()


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


def test_farm_norm_is_experiment_720s_scaling_by_the_farms_own_daylight_99th_percentile():
    n = 2000
    rng = np.random.default_rng(11)
    zenith = np.where(
        np.arange(n) % 48 < 12, 80.0, 40.0
    )  # the first 12 half-hours of a day are night
    ghi = np.where(zenith < 75.0, 600.0, 0.0)
    power = np.where(zenith < 75.0, rng.uniform(1.0, 5.0, n), 0.0)
    expected_p99 = float(np.percentile(power[zenith < 75.0], 99))
    out = world.farm_norm(power_mw=power, zenith_deg=zenith, ghi=ghi)
    day = zenith < 75.0
    # Daylight is the power over the farm's own daylight 99th percentile, not over its capacity.
    assert out[day] == pytest.approx(power[day] / expected_p99)
    assert (out[~day] == 0.0).all()  # night is zero
    assert (
        np.nanmax(out) > 1.0
    )  # the top 1% of daylight exceeds the percentile, with no clipping above
    # A different capacity changes nothing: the series has no capacity in it.
    assert world.farm_norm(power_mw=power * 3.0, zenith_deg=zenith, ghi=ghi) == pytest.approx(out)


def test_farm_norm_clips_negative_power_to_zero_and_sets_a_daylight_dropout_to_nan():
    n = 2000
    zenith = np.where(np.arange(n) % 48 < 12, 80.0, 40.0)
    ghi = np.where(zenith < 75.0, 600.0, 0.0)
    power = np.where(zenith < 75.0, 3.0, 0.0)
    power[12] = -0.5  # daylight, negative, irradiance above 100: a dropout
    power[13] = np.nan  # a gap
    ghi[14] = 50.0
    power[14] = -0.5  # daylight, negative, irradiance below 100: clipped to 0, not a dropout
    power[2] = -0.4  # night, negative: night is 0
    power[15] = 0.0  # daylight, zero, irradiance above 100: a dropout
    out = world.farm_norm(power_mw=power, zenith_deg=zenith, ghi=ghi)
    assert np.isnan(out[12])
    assert np.isnan(out[13])
    assert out[14] == 0.0
    assert out[2] == 0.0
    assert np.isnan(out[15])
    assert out[16] == pytest.approx(1.0)  # every other daylight value is the percentile


def test_farm_norm_is_nan_where_the_grid_has_no_frame_row_and_everywhere_without_daylight_power():
    zenith = np.array([40.0, np.nan, 40.0, 80.0])
    power = np.array([2.0, np.nan, 2.0, 0.0])
    out = world.farm_norm(power_mw=power, zenith_deg=zenith, ghi=np.full(4, 600.0))
    assert np.isnan(out[1])
    assert out[3] == 0.0
    assert out[0] == pytest.approx(1.0)
    none = world.farm_norm(power_mw=np.full(3, np.nan), zenith_deg=np.full(3, 40.0), ghi=np.ones(3))
    assert np.isnan(none).all()


def test_injection_scale_follows_experiment_720_level_times_the_median_times_the_norm():
    rng = np.random.default_rng(3)
    host = rng.normal(-3, 0.2, N)
    norm = np.abs(rng.normal(0.8, 0.1, N))
    rows = np.zeros(N, dtype=bool)
    rows[100:400] = True
    for level in core.LEVELS:
        a = world.injection_scale(
            host_signed_mw=host, farm_norm_series=norm, rows=rows, level=level
        )
        # The increment is level * median|X| times the farm's normalised series; no capacity enters.
        assert a * norm == pytest.approx(level * np.median(np.abs(host[rows])) * norm)
        injected = world.inject(signed_mw=host, farm_norm_series=norm, scale=a)
        assert np.median(injected[rows]) < np.median(host[rows])


def test_injection_scale_is_not_finite_with_too_few_reference_rows():
    host = np.full(N, -3.0)
    norm = np.full(N, 0.9)
    few = np.zeros(N, dtype=bool)
    few[: core.MIN_REFERENCE_ROWS - 1] = True
    assert np.isnan(
        world.injection_scale(host_signed_mw=host, farm_norm_series=norm, rows=few, level=1.0)
    )
    enough = np.zeros(N, dtype=bool)
    enough[: core.MIN_REFERENCE_ROWS] = True
    assert np.isfinite(
        world.injection_scale(host_signed_mw=host, farm_norm_series=norm, rows=enough, level=1.0)
    )
    # A farm gap removes reference rows.
    gap = norm.copy()
    gap[:100] = np.nan
    assert world.reference_row_count(host_signed_mw=host, farm_norm_series=gap, rows=enough) == 100


def test_a_farm_gap_or_dropout_makes_the_injected_host_series_nan_as_in_experiment_720():
    host = np.full(10, -3.0)
    norm = np.array([0.5, np.nan, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    injected = world.inject(signed_mw=host, farm_norm_series=norm, scale=2.0)
    assert np.isnan(injected[1])
    assert injected[0] == pytest.approx(-3.0 - 1.0)


def test_level_reference_rows_are_experiment_720s_sunny_middays_before_the_cut():
    month = world.month_of_rows(n=N)
    hour = world.utc_hour_of_rows(n=N)
    k = np.full(N, 0.9)
    rows = world.level_reference_rows(k=k, month=month, hour=hour, before_row=N // 2)
    assert rows.any()
    assert not rows[N // 2 :].any()
    assert (hour[rows] >= 10).all()
    assert (hour[rows] < 14).all()
    assert set(month[rows]) <= set(core.REFERENCE_MONTHS)
    assert not world.level_reference_rows(
        k=np.full(N, 0.69), month=month, hour=hour, before_row=N
    ).any()
    assert world.level_reference_rows(k=np.full(N, 0.7), month=month, hour=hour, before_row=N).any()


def test_the_cams_sensor_reads_the_hourly_change_of_an_hour_held_index():
    n = 48 * 4
    hh = np.arange(n) % 48
    lit = (hh >= 14) & (hh <= 34)
    hours = np.arange(n) // 2  # one value per pair of rows, so the odd rows change by 0
    k_true = 0.3 + 0.4 * ((hours % 5) / 4.0)
    u_clear = np.where(lit, 0.5, 0.0)
    s = SimpleNamespace(
        ghi_clear=np.full(n, 500.0),
        ghi=k_true * 500.0,
        u_clear=u_clear,
        u=k_true * u_clear,
        zenith=np.full(n, 40.0),
    )
    sensor = world.cams_sensor(s=s, lit=lit)
    assert sensor.hourly
    hour_rows = world.hour_end_rows(n=n)
    # On the hour-ending rows that the half-hour change would read as 0, the hourly change is not 0.
    half = np.full(n, np.nan)
    half[1:] = k_true[1:] - k_true[:-1]
    both = hour_rows & lit & np.roll(lit, 1) & np.roll(lit, 2)
    assert (np.abs(half[both]) < 1e-12).all()
    assert (np.abs(sensor.dk[both & np.isfinite(sensor.dk)]) > 0.05).any()
    assert np.isnan(sensor.dk[~hour_rows]).all()
    # The reading's change and mask are aligned to the same hourly step.
    dm = np.arange(n, dtype=float)
    masked = np.zeros(n, dtype=bool)
    masked[100] = True
    dm_h, masked_h = world.align_reading(dm=dm, masked=masked, sensor=sensor)
    assert dm_h[101] == 201.0
    assert masked_h[100]
    assert masked_h[101]
    dm_same, masked_same = world.align_reading(
        dm=dm, masked=masked, sensor=world.SensorChange(dz=dm, dk=dm)
    )
    assert dm_same is dm
    assert masked_same is masked


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


def test_the_placebo_is_read_on_the_stretches_and_pairs_of_the_area():
    # Two stretches; the shifted sensor has no finite pair on the second, so it scores c = 0 there.
    scores = [
        fake_score(
            name="a",
            role="injected",
            host="D10",
            level=1.0,
            c=[0.8, -0.8],
            truth=["export", "import"],
        )
    ]
    scores[0].table = scores[0].table.with_columns(
        pl.Series("c_shifted", [0.3, float("nan")], dtype=pl.Float64),
        pl.Series("n_shifted", [9, 0], dtype=pl.Int64),
    )
    real = score.tagged(scores=scores, sensor="farm", roles=("injected",))
    placebo = score.tagged(
        scores=scores, sensor="shifted", roles=("injected",), population_sensor="farm"
    )
    assert placebo.height == real.height == 2
    assert placebo["day"].to_list() == real["day"].to_list()
    assert placebo["c"].to_list() == [0.3, 0.0]
    assert placebo["n_pairs"].to_list() == [9, 0]


def test_a_world_that_cannot_be_scored_is_skipped_with_its_reason():
    membership = {"D10": ["F1", "F2", "F3"], "D14": ["F1", "F2"], "D16": []}
    spec = score.WorldSpec("w", "injected", "D10", "F1", 1.0)
    assert score.expected_to_score(spec=spec, membership=membership)
    assert not score.expected_to_score(
        spec=score.WorldSpec("w", "injected", "D14", "F1", 1.0), membership=membership
    )
    assert not score.expected_to_score(
        spec=score.WorldSpec("u", "uninjected", "D16", None, None), membership=membership
    )
    specs = score.world_specs(qualifying=[("D10", "F1"), ("D10", "F2"), ("D14", "F3")])
    ok = [s for s in specs if score.expected_to_score(spec=s, membership=membership)]
    assert len({s.name for s in ok}) == len(ok)


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
