# ruff: noqa: E501
"""The add-back rule, generator classes, stamps, the class audit, same-sky pairs, and concordance."""

import e00722_core as core
import e00722_sky as sky
import numpy as np
from e00722_testkit import calendar, make_weights


def _fit_label(*, series: np.ndarray, gen: np.ndarray, n_days: int) -> str:
    cal = calendar(n_days=n_days)
    w = make_weights(n_days=n_days)
    fit = core.bootstrap_ols(
        y=core.change(x=series, lag=1),
        x=core.change(x=gen, lag=1),
        day=cal["day"],
        weights=w,
        names=["c"],
    )
    return core.classify_addback(lo=float(fit.lo[0]), hi=float(fit.hi[0]))


def test_the_f1_rule_classifies_both_add_back_forms_and_the_embedded_form():
    n_days = 80
    n = n_days * 48
    rng = np.random.default_rng(6)
    demand = 60.0 + np.cumsum(rng.normal(0, 0.3, n))
    gen = np.clip(5.0 + np.cumsum(rng.normal(0, 0.5, n)), 0.0, 10.0)
    flow = demand - gen
    assert (flow > 0).all()
    assert _fit_label(series=np.abs(flow), gen=gen, n_days=n_days) == "embedded_not_added_back"
    assert (
        _fit_label(series=np.abs(flow) + gen, gen=gen, n_days=n_days) == "added_back_or_not_below"
    )
    assert (
        _fit_label(series=np.abs(flow + gen), gen=gen, n_days=n_days) == "added_back_or_not_below"
    )
    assert _fit_label(series=rng.normal(0, 1, n), gen=gen, n_days=n_days) in {
        "unresolved",
        "added_back_or_not_below",
    }


def test_the_below_minus_two_points_count_separates_the_two_forms():
    flow = np.array([-10.0, -10.0, 5.0, 5.0])
    engine = np.array([12.0, 12.0, 3.0, 3.0])
    form1 = np.abs(flow) + engine
    form2 = np.abs(flow + engine)
    first = core.below_count(reading_mw=form1, engine_mw=engine, capacity_mw=20.0)
    second = core.below_count(reading_mw=form2, engine_mw=engine, capacity_mw=20.0)
    assert first.n_below == 0
    assert first.form_hint == "no_discrimination"
    assert second.n_below == 2
    assert second.n_total == 4
    assert second.form_hint == "rules_out_abs_x_plus_a"


def test_generator_classes_and_the_subtraction_rule():
    emb, zero, unres = "embedded_not_added_back", "added_back_or_not_below", "unresolved"
    assert core.generator_class(labels=[zero, emb, zero]) == "embedded_not_added_back"
    assert core.generator_class(labels=[zero, zero]) == "outside_or_added_back"
    assert core.generator_class(labels=[zero, unres]) == "unresolved"
    assert core.generator_class(labels=[]) == "unresolved"

    def sub(
        *,
        cls: core.GeneratorClass = "outside_or_added_back",
        units: str = "MW",
        coverage: float = 0.95,
        is_engine: bool = False,
    ) -> bool:
        return core.subtract_generator(cls=cls, units=units, coverage=coverage, is_engine=is_engine)

    assert sub()
    assert not sub(units="MVA")
    assert not sub(coverage=0.5)
    assert not sub(is_engine=True)
    assert not sub(cls="embedded_not_added_back")
    assert not sub(cls="unresolved")


def test_a_known_thirty_minute_offset_is_found_by_the_stamp_check():
    rng = np.random.default_rng(7)
    y = rng.normal(0, 1, 3000)
    assert sky.peak_lag(x=y.copy(), y=y)[0] == 0
    delayed = core.shift_rows(x=y, rows=1)
    assert sky.peak_lag(x=delayed, y=y)[0] == 1
    assert sky.peak_lag(x=core.shift_rows(x=y, rows=-1), y=y)[0] == -1
    lag, corr = sky.peak_lag(x=-y, y=y)
    assert lag == 0
    assert corr < -0.99


def test_the_concordance_signs_and_the_no_evidence_rules():
    d_sky = np.tile([1.0, -1.0], 10)
    use = np.ones(20, dtype=bool)
    c, n, ev = sky.concordance(d_reading=d_sky * 2.0, d_sky=d_sky, usable=use)
    assert (c, n, ev) == (1.0, 20, True)
    c, n, ev = sky.concordance(d_reading=-d_sky * 2.0, d_sky=d_sky, usable=use)
    assert (c, n, ev) == (-1.0, 20, True)
    c, n, ev = sky.concordance(
        d_reading=d_sky[:7] * 2.0, d_sky=d_sky[:7], usable=np.ones(7, dtype=bool)
    )
    assert n == 7
    assert not ev
    mixed = np.concatenate([d_sky[:10] * 2.0, -d_sky[10:] * 2.0])
    mixed[:11] = d_sky[:11] * 2.0
    c, n, ev = sky.concordance(d_reading=mixed, d_sky=d_sky, usable=use)
    assert abs(c) < 0.25 or not ev


def test_touches_use_the_raw_reading_and_the_median_rule_would_merge_stretches():
    reading, lit = np.full(96, 8.0), np.zeros(96, dtype=bool)
    lit[12:37] = True
    reading[20] = 0.0
    reading[28] = 0.0
    level = 2.0
    raw = sky.touches(reading=reading, lit=lit, level=level)
    assert np.flatnonzero(raw).tolist() == [20, 28]
    smoothed = sky.touches(reading=sky.running_median3(x=reading), lit=lit, level=level)
    assert not smoothed.any()
    merged = sky.merged_by_median(
        reading=reading, lit=lit, level=level, usable=np.ones(96, dtype=bool)
    )
    assert merged == 2
    assert sky.touch_level(reading=np.full(96, 8.0), lit=lit, capacity_mw=50.0) == 5.0


def _sky_series(*, days: int, seed: int, cloud: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = days * 48
    half = np.arange(n) % 48
    zenith = 90.0 - np.clip(45.0 * np.sin(np.pi * (half - 12) / 24.0), -20.0, None)
    zenith = np.where((half < 12) | (half > 36), 120.0, zenith)
    k = np.clip(0.6 + cloud * rng.normal(0, 1, n), 0.05, 1.0)
    return k, zenith


def test_cloud_classes_follow_the_section_4_definitions():
    n_days = 30
    clear_k = np.full(n_days * 48, 0.95)
    over_k = np.full(n_days * 48, 0.2)
    _, zen = _sky_series(days=n_days, seed=1, cloud=0.0)
    lit = sky.daylight(zenith_deg=zen, min_elevation_deg=core.DAYLIGHT_ELEVATION_DEG)
    assert (sky.classify_days(k=clear_k, lit=lit, n_days=n_days) == sky.CLASS_CLEAR).all()
    assert (sky.classify_days(k=over_k, lit=lit, n_days=n_days) == sky.CLASS_OVERCAST).all()
    flicker = np.where(np.arange(n_days * 48) % 2 == 0, 0.35, 0.85)
    assert (sky.classify_days(k=flicker, lit=lit, n_days=n_days) == sky.CLASS_INTERMITTENT).all()


def test_the_class_audit_conditions_pass_on_a_matching_farm_and_fail_on_a_mismatch():
    n_days = 40
    classes = np.array([sky.CLASS_INTERMITTENT] * 20 + [sky.CLASS_CLEAR] * 20)
    coverage = np.ones(n_days)
    good = np.where(classes == sky.CLASS_INTERMITTENT, 6, 0)
    rows = sky.audit_farm(classes=classes, n_big=good, coverage=coverage, scope="F1")
    assert all(r.passed and r.readable for r in rows)
    bad = np.where(classes == sky.CLASS_INTERMITTENT, 0, 6)
    rows_bad = sky.audit_farm(classes=classes, n_big=bad, coverage=coverage, scope="F1")
    assert not any(r.passed for r in rows_bad)
    few = sky.audit_farm(classes=classes[:5], n_big=good[:5], coverage=coverage[:5], scope="F1")
    assert not any(r.readable for r in few)
    month = np.full(n_days, 6)
    season = sky.audit_season_share(
        classes=classes, month=month, has_data=np.ones(n_days, dtype=bool), scope="D1"
    )
    assert season.passed
    assert season.value == 0.5
    change = sky.audit_elevation_sensitivity(
        classes_10=classes,
        classes_15=classes,
        has_data=np.ones(n_days, dtype=bool),
        scope="all_sites",
    )
    assert change.passed
    assert sky.audit_passes(rows=[*rows, season, change])
    assert not sky.audit_passes(rows=[*rows_bad, season, change])
    assert not sky.audit_passes(rows=[*rows, season])


def test_big_changes_count_only_covered_lit_days():
    out = np.full(96, 5.0)
    out[14:30:2] = 0.0
    lit = np.zeros(96, dtype=bool)
    lit[12:36] = True
    n_big, coverage = sky.count_big_changes(output_mw=out, capacity_mw=10.0, lit=lit, n_days=2)
    assert n_big[0] >= core.AUDIT_MIN_BIG_CHANGES
    assert coverage[0] == 1.0
    assert coverage[1] == 0.0


def test_same_sky_correlation_needs_enough_rows_and_flags_matching_skies():
    rng = np.random.default_rng(8)
    a = rng.normal(0, 1, 2000)
    assert (
        sky.correlation_of_changes(a=a, b=a + rng.normal(0, 0.1, 2000))[0]
        > core.SAME_SKY_CORRELATION
    )
    assert sky.correlation_of_changes(a=a, b=rng.normal(0, 1, 2000))[0] < core.SAME_SKY_CORRELATION
    assert np.isnan(sky.correlation_of_changes(a=a[:100], b=a[:100])[0])
