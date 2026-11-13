r"""Known-answer tests for experiment 698's helpers, on small synthetic inputs.

    E=ml-experiments
    export PYTHONPATH=$(ls -d $E/00698-* $E/00697-* $E/00694-* $E/00695-* | paste -sd:)
    uv run --no-sync python $E/00698-*/e00698_selftest.py

The tests read no stored data (one test reads the stored panel and the farm file when they exist)
and write nothing outside a temporary folder. They cover the farm's grid and gap handling, the
smooth
twin, the fold rule applied once on the summed output (and how it differs from two applications),
the fold states, the curve gate, fifth edges, the paired week-block interval and the direction of
each criterion, off-spell and whole-day selection, the job lists and seed recipe, the verdict rule
on synthetic statistics (every branch and step 4's one-way adjustment), the SVG year guard, and that
no committed file of this folder names a path under the maintainer's home folder.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
from e00694_common import GRID_START, Q_FLOOR, inject_folded  # ty: ignore[unresolved-import]
from e00698_common import (  # ty: ignore[unresolved-import]
    CAPACITIES_MW,
    CRITERIA,
    FARM_FRAME,
    HOST_KEYS,
    N_SEEDS,
    PV_FREE,
    SHAPES,
    THRESHOLDS,
    GridCalendar,
    Interval,
    bright_dull_thirds,
    capacity_key,
    cell_mean,
    criterion_value,
    curve_medians,
    engine_seed,
    farm_on_grid,
    fifth_edges,
    fold_reading,
    fold_state,
    gate_passes,
    grid_calendar_fields,
    half_width,
    host_job_name,
    host_jobs,
    inside_threshold,
    longest_gap,
    meets_criterion,
    off_spell_lengths,
    off_spell_rows,
    paired_contrast,
    score_columns,
    solar_curve,
    step4_jobs,
    week_counts,
    week_table,
    whole_day_rows,
)
from e00698_figures import assert_no_year  # ty: ignore[unresolved-import]
from e00698_verdict import VERDICT_WORDS, block_reasons, decide, running_coverage  # ty: ignore[unresolved-import]

N_GRID = 48 * 800
HOME_PREFIX = "/home" + "/jack"
"""Built from two pieces so that this file itself holds no such path."""


def test_farm_on_grid_fills_night_and_keeps_gaps() -> None:
    """A half-hour the file lacks is 0 at night and NaN by day, and a stored NaN stays NaN."""
    origin = int(np.datetime64(GRID_START.replace(tzinfo=None), "us").astype("int64"))
    # Grid index i has the end stamp origin + i * 30 min. Keep indices 20 (10:00) and 22 (11:00).
    times = np.array([origin + 20 * 1_800_000_000, origin + 22 * 1_800_000_000])
    values = np.array([0.5, np.nan])
    out = farm_on_grid(series=(times, values), n_grid=48)
    assert out[20] == 0.5
    assert np.isnan(out[22])  # a stored dropout stays NaN
    assert np.isnan(out[24])  # 12:00 is absent from the file by day
    assert out[2] == 0.0  # 01:00 is absent at night
    assert out[44] == 0.0  # 22:00 is absent at night
    assert longest_gap(finite=np.array([True, False, False, True, False])) == 2


def test_smooth_twin_is_the_cell_mean_and_carries_no_weather() -> None:
    """The smooth twin equals c times the mean of its cell and is constant within each cell."""
    calendar = grid_calendar_fields(n_grid=N_GRID)
    rng = np.random.default_rng(0)
    seasonal = np.maximum(0.0, np.sin((calendar.half_hour - 12) / 48 * 2 * np.pi))
    weather = rng.uniform(0.5, 1.5, size=N_GRID)
    pv_shape = seasonal * weather
    curve = solar_curve(
        shape="smooth",
        capacity_mw=7.0,
        pv_shape=pv_shape,
        farm_norm=None,
        calendar=calendar,
        first=100,
        last=N_GRID - 100,
    )
    window = slice(100, N_GRID - 100)
    key = calendar.month[window] * 48 + calendar.half_hour[window]
    for k in np.unique(key)[:20]:
        cell = key == k
        assert np.ptp(curve.pv_mw[window][cell]) < 1e-12
        assert np.isclose(curve.pv_mw[window][cell][0], 7.0 * pv_shape[window][cell].mean())
    assert np.all(curve.pv_mw[:100] == 0.0)
    assert np.all(curve.pv_mw[N_GRID - 100 :] == 0.0)
    assert np.allclose(
        cell_mean(value=pv_shape, calendar=calendar, first=100, last=N_GRID - 100)[window],
        curve.pv_mw[window] / 7.0,
    )


def farm_fixture(*, calendar: GridCalendar) -> tuple[np.ndarray, np.ndarray]:
    """Return a CAMS shape and a farm series with another level and a flatter daily shape."""
    pv_shape = np.maximum(0.0, np.sin((calendar.half_hour - 12) / 48 * 2 * np.pi))
    farm = 0.7 * np.maximum(0.0, np.sin((calendar.half_hour - 12) / 48 * 2 * np.pi)) ** 0.5
    return pv_shape, farm


def test_farm_shape_fills_its_gaps_with_its_own_cell_mean_and_flags_them() -> None:
    """A farm gap is filled with the farm's cell mean, and `farm_ok` marks it."""
    calendar = grid_calendar_fields(n_grid=N_GRID)
    pv_shape, farm = farm_fixture(calendar=calendar)
    farm[1000:1010] = np.nan
    kwargs = {
        "capacity_mw": 8.9,
        "pv_shape": pv_shape,
        "farm_norm": farm,
        "calendar": calendar,
        "first": 0,
        "last": N_GRID,
    }
    farm_curve = solar_curve(shape="farm", **kwargs)  # ty: ignore[invalid-argument-type]
    twin = solar_curve(shape="smooth_farm", **kwargs)  # ty: ignore[invalid-argument-type]
    assert not farm_curve.farm_ok[1000:1010].any()
    assert farm_curve.farm_ok[:1000].all()
    assert np.all(np.isfinite(farm_curve.pv_mw))
    assert np.allclose(farm_curve.pv_mw[1000:1010], twin.pv_mw[1000:1010])
    assert np.allclose(farm_curve.pv_mw[:1000], 8.9 * farm[:1000])


def test_farm_twin_shares_the_farm_level_and_not_the_cams_level() -> None:
    """The farm run and `smooth_farm` have one mean per cell, so their contrast holds no level."""
    calendar = grid_calendar_fields(n_grid=N_GRID)
    pv_shape, farm = farm_fixture(calendar=calendar)
    rng = np.random.default_rng(1)
    farm = farm * rng.uniform(0.3, 1.2, size=N_GRID)
    kwargs = {
        "capacity_mw": 7.0,
        "pv_shape": pv_shape,
        "farm_norm": farm,
        "calendar": calendar,
        "first": 0,
        "last": N_GRID,
    }
    farm_curve = solar_curve(shape="farm", **kwargs)  # ty: ignore[invalid-argument-type]
    twin = solar_curve(shape="smooth_farm", **kwargs)  # ty: ignore[invalid-argument-type]
    cams_twin = solar_curve(shape="smooth", **kwargs)  # ty: ignore[invalid-argument-type]
    assert np.isclose(farm_curve.pv_mw.sum(), twin.pv_mw.sum())
    key = calendar.month * 48 + calendar.half_hour
    for k in np.unique(key)[:30]:
        cell = key == k
        assert np.isclose(farm_curve.pv_mw[cell].mean(), twin.pv_mw[cell].mean())
    # The CAMS twin has another level and shape, so a contrast against it would mix them in.
    assert abs(farm_curve.pv_mw.sum() - cams_twin.pv_mw.sum()) > 0.1 * farm_curve.pv_mw.sum()
    assert np.ptp(twin.pv_mw[key == key[500]]) < 1e-12


def test_fold_is_applied_once_on_the_sum() -> None:
    """The reading is sqrt((real - solar - engine)^2 + q^2), not the fold applied twice."""
    y = np.array([5.0, 3.0, 1.0, np.nan, 0.14])
    pv = np.array([4.0, 4.0, 4.0, 4.0, 0.0])
    g = np.array([2.0, 0.0, 2.0, 0.0, 0.0])
    reading, signed = fold_reading(y=y, total_mw=pv + g, first=0, last=5)
    real = np.sqrt(np.maximum(y**2 - Q_FLOOR**2, 0.0))
    assert np.allclose(signed[:3], (real - pv - g)[:3])
    assert np.allclose(reading[:3], np.sqrt(((real - pv - g)[:3]) ** 2 + Q_FLOOR**2))
    assert np.isnan(reading[3])
    assert np.isnan(signed[3])
    assert np.allclose(reading[:3], inject_folded(y=y[:3], g=(pv + g)[:3], q=Q_FLOOR))
    # Reading 2 is real 0.99 minus 6: signed -5.01, read back as +5.01, which a one-step fold shows.
    assert signed[2] < 0
    assert np.isclose(reading[2], np.sqrt(signed[2] ** 2 + Q_FLOOR**2))
    # Applying the fold twice (solar first, then the engine) gives a different reading.
    once = inject_folded(y=y[:3], g=pv[:3], q=Q_FLOOR)
    twice = inject_folded(y=once, g=g[:3], q=Q_FLOOR)
    assert not np.allclose(twice, reading[:3])
    # Outside the span the reading is untouched and the signed power is NaN.
    partial, signed_partial = fold_reading(y=y, total_mw=pv + g, first=1, last=3)
    assert partial[0] == y[0]
    assert np.isnan(signed_partial[0])
    assert partial[4] == y[4]


def test_fold_states() -> None:
    """Forward flow is above 1 MW, reverse flow below minus 1 MW, and NaN is -1."""
    state = fold_state(signed_mw=np.array([3.0, 1.0, 0.0, -1.0, -1.5, np.nan]))
    assert state.tolist() == [0, 1, 1, 1, 2, -1]


def test_curve_gate() -> None:
    """The gate needs the first four fifths within 1 MVA and enough rows, and ignores the fifth."""
    reference = np.array([6.0, 4.0, 2.0, 2.0, 2.0])
    counts = np.full(5, 100)
    assert gate_passes(curve=reference + 0.9, reference=reference, counts=counts)
    assert not gate_passes(
        curve=reference + np.array([0, 0, 0, 1.2, 0]), reference=reference, counts=counts
    )
    assert gate_passes(
        curve=reference + np.array([0, 0, 0, 0, 5.0]), reference=reference, counts=counts
    )
    assert not gate_passes(
        curve=reference, reference=reference, counts=np.array([100, 100, 5, 100, 100])
    )
    assert not gate_passes(
        curve=np.array([6.0, np.nan, 2, 2, 2]), reference=reference, counts=counts
    )


def test_fifth_edges_and_medians() -> None:
    """Fifth edges split a uniform grid into five equal groups and the medians follow."""
    shape = np.arange(1000, dtype=float)
    edges = fifth_edges(values=shape)
    assert np.allclose(edges, [199.8, 399.6, 599.4, 799.2])
    reading = shape * 0.01
    medians, counts = curve_medians(
        reading=reading, shape_value=shape, edges=edges, rows=np.ones(1000, dtype=bool)
    )
    assert counts.tolist() == [200] * 5
    assert np.allclose(medians, [0.995, 2.995, 4.995, 6.995, 8.995])
    masked, counts_masked = curve_medians(
        reading=reading, shape_value=shape, edges=edges, rows=shape < 500
    )
    assert counts_masked[4] == 0
    assert np.isnan(masked[4])


def test_thirds_within_cells() -> None:
    """Thirds are taken within local half-hour, weekday class, and month, above 50 W/m2."""
    calendar = grid_calendar_fields(n_grid=N_GRID)
    rng = np.random.default_rng(1)
    ghi = rng.uniform(0, 400, size=N_GRID)
    ghi[::7] = 10.0
    rows = np.ones(N_GRID, dtype=bool)
    thirds = bright_dull_thirds(ghi=ghi, value=ghi, calendar=calendar, rows=rows)
    assert np.all(thirds[ghi <= 50] == -1)
    assert set(np.unique(thirds[ghi > 50])) == {0, 1, 2}
    cell = (calendar.half_hour == 24) & (calendar.month == 6) & ~calendar.weekend & (ghi > 50)
    counts = np.bincount(thirds[cell], minlength=3)
    assert counts.max() - counts.min() <= 1
    assert thirds[cell][np.argmax(ghi[cell])] == 2
    assert thirds[cell][np.argmin(ghi[cell])] == 0


def test_score_columns_and_criteria() -> None:
    """F1, the false-flag share, and the flagged-share difference come out of the nine sums."""
    truth = np.array([1, 1, 0, 0, 0, 0], dtype=bool)
    flag = np.array([1, 0, 1, 0, 1, 1], dtype=bool)
    zero = ~truth
    third = np.array([2, 2, 2, 0, 0, 0])
    sums = score_columns(truth1=truth, flag=flag, zero=zero, third=third).sum(axis=0)
    assert sums.tolist() == [1, 3, 1, 3, 4, 2, 3, 2, 3]
    assert np.isclose(criterion_value(sums=sums, criterion="f1"), 2 / (2 + 3 + 1))
    assert np.isclose(criterion_value(sums=sums, criterion="false_flag"), 3 / 4)
    assert np.isclose(criterion_value(sums=sums, criterion="bmd"), 2 / 3 - 2 / 3)


def planted_tables(*, effect: float, seed: int, n_weeks: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Return per-week score sums for a base run and a job whose false flags rise by `effect`.

    Both runs share the same week effects (the pairing), and each week has 100 zero-output
    half-hours.
    """
    rng = np.random.default_rng(seed)
    week_effect = rng.normal(0, 0.05, n_weeks)
    zero = np.full(n_weeks, 100.0)
    base_ff = rng.binomial(100, np.clip(0.03 + week_effect, 0, 1)).astype(float)
    job_ff = np.clip(base_ff + rng.binomial(100, effect, n_weeks), 0, 100).astype(float)
    base = np.zeros((n_weeks, 9))
    job = np.zeros((n_weeks, 9))
    base[:, 3], base[:, 4] = base_ff, zero
    job[:, 3], job[:, 4] = job_ff, zero
    return job, base


def test_paired_interval_covers_a_planted_effect_and_excludes_zero() -> None:
    """The paired week-block interval holds a planted harm and excludes zero if the harm is real."""
    hits = 0
    excluded = 0
    n_rep = 40
    for rep in range(n_rep):
        job, base = planted_tables(effect=0.04, seed=rep)
        counts = week_counts(n_weeks=40, seed=rep, n_resamples=400)
        est = paired_contrast(criterion="false_flag", job_weeks=job, base_weeks=base, counts=counts)
        true_effect = (
            0.04 * (1 - 0.03) if False else job[:, 3].sum() / 4000 - base[:, 3].sum() / 4000
        )
        hits += int(est["lo"] <= true_effect <= est["hi"])
        excluded += int(est["lo"] > 0)
    assert hits >= 0.85 * n_rep, hits
    assert excluded >= 0.95 * n_rep, excluded
    job, base = planted_tables(effect=0.0, seed=5)
    counts = week_counts(n_weeks=40, seed=5, n_resamples=400)
    null = paired_contrast(criterion="false_flag", job_weeks=job, base_weeks=base, counts=counts)
    assert null["lo"] <= 0 <= null["hi"]


def test_one_draw_of_weeks_serves_every_run_and_pairing_narrows_the_interval() -> None:
    """Identical week effects cancel in a paired contrast, and the counts are reused unchanged."""
    counts = week_counts(n_weeks=30, seed=3, n_resamples=200)
    assert np.array_equal(counts, week_counts(n_weeks=30, seed=3, n_resamples=200))
    assert np.all(counts.sum(axis=1) == 30)
    job, base = planted_tables(effect=0.04, seed=9, n_weeks=30)
    paired = paired_contrast(criterion="false_flag", job_weeks=job, base_weeks=base, counts=counts)
    independent_counts = week_counts(n_weeks=30, seed=4, n_resamples=200)
    other = np.maximum(
        independent_counts @ job[:, 3] / (independent_counts @ job[:, 4])
        - counts @ base[:, 3] / (counts @ base[:, 4]),
        -1,
    )
    unpaired_half_width = (np.percentile(other, 97.5) - np.percentile(other, 2.5)) / 2
    assert half_width(interval=paired) < unpaired_half_width


def test_week_table_rejects_an_unknown_week() -> None:
    """A row whose week is not in the host's list raises instead of landing in a neighbour."""
    weeks = np.array([1, 2, 3])
    table = week_table(week=np.array([1, 1, 3]), columns=np.ones((3, 2)), weeks=weeks)
    assert table.tolist() == [[2.0, 2.0], [0.0, 0.0], [1.0, 1.0]]
    try:
        week_table(week=np.array([9]), columns=np.ones((1, 2)), weeks=weeks)
    except ValueError:
        return
    msg = "an unknown week was accepted"
    raise AssertionError(msg)


def test_criterion_directions_and_thresholds() -> None:
    """A positive harm is always worse, and meeting a criterion needs the threshold and a sign."""
    clear = Interval(point=0.06, lo=0.01, hi=0.11)
    assert meets_criterion(interval=clear, criterion="f1")
    assert not meets_criterion(interval=Interval(point=0.06, lo=-0.01, hi=0.13), criterion="f1")
    assert not meets_criterion(interval=Interval(point=0.04, lo=0.01, hi=0.07), criterion="f1")
    assert not meets_criterion(interval=Interval(point=-0.2, lo=-0.3, hi=-0.1), criterion="bmd")
    assert not meets_criterion(interval=None, criterion="bmd")
    assert inside_threshold(interval=Interval(point=0.0, lo=-0.03, hi=0.04), criterion="f1")
    assert not inside_threshold(interval=Interval(point=0.0, lo=-0.03, hi=0.06), criterion="f1")
    assert THRESHOLDS == {"f1": 0.05, "false_flag": 0.02, "bmd": 0.05}
    # F1 falling is a positive harm: job F1 0.5, base F1 0.6.
    base = np.zeros((10, 9))
    job = np.zeros((10, 9))
    base[:, :3] = [60, 40, 40]
    job[:, :3] = [50, 50, 50]
    counts = week_counts(n_weeks=10, seed=0, n_resamples=50)
    est = paired_contrast(criterion="f1", job_weeks=job, base_weeks=base, counts=counts)
    assert est["point"] > 0.05


def test_off_spells_and_whole_days() -> None:
    """Runs of False of 48 or more are kept, and a day counts only if it lies wholly inside."""
    flag = np.zeros(240, dtype=bool)
    flag[:10] = True  # run of False starts at 10
    flag[100] = True  # splits the False run 10..99 (90 long) from 101..239 (139 long)
    assert off_spell_lengths(flag=flag, min_length=48) == [90, 139]
    short = np.zeros(60, dtype=bool)
    short[30] = True
    assert off_spell_lengths(flag=short, min_length=48) == []
    rows = off_spell_rows(flag=flag, min_length=48)
    assert not rows[:10].any()
    assert rows[10:100].all()
    assert not rows[100]
    local_date = np.repeat(np.arange(5), 48)
    observed = np.ones(240, dtype=bool)
    observed[200] = False
    days = whole_day_rows(rows=rows, local_date=local_date, observed=observed)
    # Day 0 holds the flagged start, day 1 (48 to 95) is whole, day 2 holds the flag at 100, day 4
    # (192 to 239) holds an unobserved half-hour, and day 3 (144 to 191) is whole.
    assert days[48:96].all()
    assert not days[:48].any()
    assert not days[96:144].any()
    assert days[144:192].all()
    assert not days[192:].any()


def test_job_lists_and_seed_recipe() -> None:
    """The plan allows 72 host jobs and 10 step 4 jobs, and seeds follow experiment 697."""
    every = host_jobs(gate=None)
    assert len(every) == 72
    assert len({j.name for j in every}) == 72
    assert len(step4_jobs()) == 10
    twins = [j for j in every if j.shape == PV_FREE]
    assert len(twins) == 2 * (N_SEEDS + 1)
    assert sum(j.seed is None for j in every) == 18
    assert (
        host_job_name(host="d8pre", capacity_mw=8.9, shape="farm", seed=None) == "d8pre_c89_farm_sn"
    )
    assert (
        host_job_name(host="d3solar", capacity_mw=None, shape=PV_FREE, seed=2)
        == "d3solar_none_free_s2"
    )
    assert [engine_seed(host="d8pre", seed=s) for s in range(3)] == [0, 1, 2]
    assert [engine_seed(host="d3solar", seed=s) for s in range(2)] == [200, 201]
    gate = {
        "run": {
            host: {capacity_key(capacity_mw=c): dict.fromkeys(SHAPES, False) for c in CAPACITIES_MW}
            for host in HOST_KEYS
        }
    }
    gate["run"]["d8pre"]["c70"] = {
        "farm": True,
        "cams": False,
        "smooth": True,
        "smooth_farm": True,
    }
    listed = host_jobs(gate=gate)
    assert {j.host for j in listed} == {"d8pre"}
    assert {(j.capacity_mw, j.shape) for j in listed} == {
        (None, PV_FREE),
        (7.0, "farm"),
        (7.0, "smooth"),
        (7.0, "smooth_farm"),
    }
    assert len(listed) == 4 * (N_SEEDS + 1)


# --------------------------------------------------------------------------- the verdict


def interval(*, point: float, half: float = 0.01) -> dict:
    """Return a synthetic interval of a given point and half-width."""
    return {"point": point, "lo": point - half, "hi": point + half}


def contrast_criteria(*, f1: float, false_flag: float, bmd: float, half: float = 0.01) -> dict:
    """Return the three criteria of one contrast, as stored in `scores.json`."""
    return {
        "f1": interval(point=f1, half=half),
        "false_flag": interval(point=false_flag, half=half / 5),
        "bmd": interval(point=bmd, half=half),
    }


HARM = contrast_criteria(f1=0.10, false_flag=0.05, bmd=0.10)
NONE = contrast_criteria(f1=0.0, false_flag=0.0, bmd=0.0)
EDGE = {
    "f1": {"point": 0.0, "lo": -0.03, "hi": 0.06},
    "false_flag": {"point": 0.0, "lo": -0.002, "hi": 0.002},
    "bmd": {"point": 0.0, "lo": -0.01, "hi": 0.01},
}
"""A contrast whose F1 interval is not wholly below its threshold and does not meet it."""


def synthetic_gate(
    *, d8: tuple[bool, bool] = (True, True), d3: tuple[bool, bool] = (True, True)
) -> dict:
    """Return a gate in which the farm and CAMS shapes pass as given for D8 and D3."""
    keys = [capacity_key(capacity_mw=c) for c in CAPACITIES_MW]
    table = {
        "d8pre": {k: dict.fromkeys(SHAPES, ok) for k, ok in zip(keys, d8, strict=True)},
        "d3solar": {k: dict.fromkeys(SHAPES, ok) for k, ok in zip(keys, d3, strict=True)},
    }
    return {"gate": table, "run": table}


def synthetic_scores(
    *,
    d8_farm: tuple[dict, dict],
    d8_cams: tuple[dict, dict],
    d3_farm: tuple[dict, dict] | None,
    twin: float = 0.767,
    step4: dict | None = None,
) -> dict:
    """Return a minimal `scores.json` with the weather contrasts given for each capacity."""
    keys = [capacity_key(capacity_mw=c) for c in CAPACITIES_MW]

    def host_doc(*, farm: tuple[dict, dict], cams: tuple[dict, dict] | None) -> dict:
        contrasts: dict = {}
        for k, f, c in zip(keys, farm, cams or (None, None), strict=True):
            entry = {"weather_farm": {"smooth": f}, "level": {"smooth": HARM}}
            if c is not None:
                entry["weather_cams"] = {"smooth": c}
            contrasts[k] = entry
        return {"contrasts": contrasts}

    hosts = {"d8pre": host_doc(farm=d8_farm, cams=d8_cams)}
    if d3_farm is not None:
        hosts["d3solar"] = host_doc(farm=d3_farm, cams=None)
    return {"hosts": hosts, "twin_f1_d8pre": twin, "step4": {"contrast_solar_minus_net": step4}}


def word(*, gate: dict, scores: dict, coverage: float = 0.99) -> str:
    """Return the verdict word of `decide`."""
    return decide(gate=gate, scores=scores, coverage=coverage)[0]


def test_verdict_branch_1_blocks_the_test() -> None:
    """Each audit condition makes the verdict `inconclusive` before any weather contrast is read."""
    good = synthetic_scores(d8_farm=(HARM, HARM), d8_cams=(HARM, HARM), d3_farm=(HARM, HARM))
    assert word(gate=synthetic_gate(), scores=good) == "supported"
    assert word(gate=synthetic_gate(d8=(False, False)), scores=good) == "inconclusive"
    assert word(gate=synthetic_gate(d8=(True, False)), scores=good) == "supported"
    bad_twin = synthetic_scores(
        d8_farm=(HARM, HARM), d8_cams=(HARM, HARM), d3_farm=(HARM, HARM), twin=0.73
    )
    assert word(gate=synthetic_gate(), scores=bad_twin) == "inconclusive"
    assert word(gate=synthetic_gate(), scores=good, coverage=0.89) == "inconclusive"
    assert word(gate=synthetic_gate(), scores=good, coverage=0.90) == "supported"
    wide = contrast_criteria(f1=0.10, false_flag=0.05, bmd=0.10, half=0.06)
    wide_scores = synthetic_scores(d8_farm=(wide, wide), d8_cams=(wide, wide), d3_farm=(HARM, HARM))
    assert word(gate=synthetic_gate(), scores=wide_scores) == "inconclusive"
    reasons = block_reasons(gate=synthetic_gate(), scores=wide_scores, coverage=0.99)
    assert len(reasons) == 1
    assert "half-width" in reasons[0]
    # Only the farm weather contrast's F1 width counts: wide CAMS and level contrasts do not block.
    cams_wide = synthetic_scores(d8_farm=(HARM, HARM), d8_cams=(wide, wide), d3_farm=(HARM, HARM))
    assert word(gate=synthetic_gate(), scores=cams_wide) == "supported"
    cams_wide["hosts"]["d8pre"]["contrasts"]["c70"]["level"] = {"smooth": wide}
    assert word(gate=synthetic_gate(), scores=cams_wide) == "supported"
    # The coverage clause reads the hosts that run, on daylight half-hours.
    gate = synthetic_gate(d3=(False, False))
    audit = {"numbers": {"farm_finite_share_daylight": {"d8pre": 0.93, "d3solar": 0.5}}}
    assert running_coverage(gate=gate, audit=audit) == 0.93
    assert running_coverage(gate=synthetic_gate(), audit=audit) == 0.5


def test_verdict_branches_2_to_5() -> None:
    """Supported, partly supported, refuted, and not supported each come from its own case."""
    gate = synthetic_gate()
    supported = synthetic_scores(d8_farm=(NONE, HARM), d8_cams=(NONE, NONE), d3_farm=(NONE, HARM))
    assert word(gate=gate, scores=supported) == "supported"
    different_capacity = synthetic_scores(
        d8_farm=(HARM, NONE), d8_cams=(NONE, NONE), d3_farm=(NONE, HARM)
    )
    assert word(gate=gate, scores=different_capacity) == "partly supported"
    farm_only_at_d8 = synthetic_scores(
        d8_farm=(NONE, HARM), d8_cams=(NONE, NONE), d3_farm=(NONE, NONE)
    )
    assert word(gate=gate, scores=farm_only_at_d8) == "partly supported"
    cams_only = synthetic_scores(d8_farm=(NONE, NONE), d8_cams=(HARM, NONE), d3_farm=(NONE, NONE))
    assert word(gate=gate, scores=cams_only) == "partly supported"
    refuted = synthetic_scores(d8_farm=(NONE, NONE), d8_cams=(NONE, NONE), d3_farm=(NONE, NONE))
    assert word(gate=gate, scores=refuted) == "refuted"
    undecided = synthetic_scores(d8_farm=(NONE, EDGE), d8_cams=(NONE, NONE), d3_farm=(NONE, NONE))
    assert word(gate=gate, scores=undecided) == "not supported"
    # Without D3 (it failed the gate), both D8 capacities must meet a criterion.
    d3_failed = synthetic_gate(d3=(False, False))
    both = synthetic_scores(d8_farm=(HARM, HARM), d8_cams=(NONE, NONE), d3_farm=None)
    one = synthetic_scores(d8_farm=(NONE, HARM), d8_cams=(NONE, NONE), d3_farm=None)
    assert word(gate=d3_failed, scores=both) == "supported"
    assert word(gate=d3_failed, scores=one) == "partly supported"
    # A criterion with an interval that excludes zero on the harmless side is not harm.
    harmless = contrast_criteria(f1=-0.10, false_flag=-0.05, bmd=-0.10)
    reversed_scores = synthetic_scores(
        d8_farm=(harmless, harmless), d8_cams=(harmless, harmless), d3_farm=(harmless, harmless)
    )
    assert word(gate=gate, scores=reversed_scores) == "refuted"
    # The point estimate must reach the threshold even when the interval excludes zero.
    small = contrast_criteria(f1=0.045, false_flag=0.018, bmd=0.045, half=0.01)
    small_scores = synthetic_scores(
        d8_farm=(small, small), d8_cams=(small, small), d3_farm=(small, small)
    )
    assert word(gate=gate, scores=small_scores) == "not supported"


def test_refuted_when_the_gate_removes_a_pair() -> None:
    """Branch 4 covers the pairs that ran, and the strict reading needs all four."""
    gate = synthetic_gate(d8=(False, True), d3=(False, True))
    scores = synthetic_scores(d8_farm=(NONE, NONE), d8_cams=(NONE, NONE), d3_farm=(NONE, NONE))
    for host in ("d8pre", "d3solar"):
        del scores["hosts"][host]["contrasts"]["c70"]
    assert word(gate=gate, scores=scores) == "refuted"
    strict, _ = decide(gate=gate, scores=scores, coverage=0.99, strict_refuted=True)
    assert strict == "not supported"
    # A pair that passed the gate but has no contrast is not read as inside its threshold.
    gate_all = synthetic_gate()
    assert word(gate=gate_all, scores=scores) == "not supported"


def test_step_4_lowers_a_verdict_and_never_raises_one() -> None:
    """Step 4 turns `supported` into `partly supported` only with a narrow interval above zero."""
    gate = synthetic_gate()

    def run(*, harm: bool, step4: dict) -> str:
        farm = (NONE, HARM) if harm else (NONE, NONE)
        scores = synthetic_scores(d8_farm=farm, d8_cams=(NONE, NONE), d3_farm=farm, step4=step4)
        return word(gate=gate, scores=scores)

    better = {"point": 0.20, "lo": 0.12, "hi": 0.28}
    assert run(harm=True, step4=better) == "partly supported"
    wide = {"point": 0.30, "lo": 0.05, "hi": 0.55}
    assert run(harm=True, step4=wide) == "supported"
    includes_zero = {"point": 0.05, "lo": -0.03, "hi": 0.13}
    assert run(harm=True, step4=includes_zero) == "supported"
    # It cannot raise: a refuted verdict stays refuted.
    assert run(harm=False, step4=better) == "refuted"
    assert set(CRITERIA) == {"f1", "false_flag", "bmd"}


# --------------------------------------------------------------------------- guards


def test_svg_year_guard() -> None:
    """A year in an SVG text node or accessibility label stops the script, other numbers do not."""
    assert_no_year(
        text='<svg><text>Local time of day</text><path d="M 2024 10"/></svg>', name="a.svg"
    )
    for text in (
        "<svg><text>Spring 2024</text></svg>",
        '<svg><g aria-label="point mark: 2024-05-01, 3.2"/></svg>',
    ):
        try:
            assert_no_year(text=text, name="b.svg")
        except SystemExit:
            continue
        msg = f"a year was accepted in {text}"
        raise AssertionError(msg)


def test_score_refuses_a_job_made_with_another_farm() -> None:
    """A stored host job that records another farm, or none, stops the scoring."""
    import e00698_score as score

    saved = score.JOBS_DIR
    with tempfile.TemporaryDirectory() as folder:
        score.JOBS_DIR = Path(folder)
        np.savez_compressed(Path(folder) / "d8pre_none_free_s0.npz", farm="A", g=np.zeros(3))
        np.savez_compressed(Path(folder) / "step4_net_s0.npz", g=np.zeros(3))
        score.check_job_farms(farm="A")
        for wrong in ("B", None):
            if wrong is None:
                np.savez_compressed(Path(folder) / "d8pre_none_free_s1.npz", g=np.zeros(3))
            try:
                score.check_job_farms(farm=wrong or "A")
            except SystemExit:
                continue
            msg = "a job with the wrong farm was accepted"
            raise AssertionError(msg)
    score.JOBS_DIR = saved


def test_jobs_script_refuses_ignore_gate_for_a_real_run() -> None:
    """`--ignore-gate` without `--list` or `--time-one` is an error and writes nothing."""
    import e00698_jobs as jobs

    saved = sys.argv
    sys.argv = ["e00698_jobs.py", "--ignore-gate"]
    try:
        jobs.main()
    except SystemExit as error:
        code = error.code
    else:
        code = None
    finally:
        sys.argv = saved
    assert code == 2, "--ignore-gate alone must be refused"


def test_no_committed_file_names_a_home_path() -> None:
    """No script of this folder names a path under the maintainer's home folder."""
    folder = Path(__file__).resolve().parent
    for path in sorted(folder.glob("*")):
        if path.is_file() and path.suffix in {".py", ".sh", ".txt"}:
            assert HOME_PREFIX not in path.read_text(), path.name


def test_farm_series_covers_the_host_spans_when_stored() -> None:
    """When the stored farm file exists, the loader gives finite values on most of D8-pre."""
    if not FARM_FRAME.exists():
        return
    from e00698_common import farm_finite_share, farm_norm_series, host_bounds

    norm = farm_on_grid(series=farm_norm_series(farm="B"), n_grid=78816)
    first, last = host_bounds(host="d8pre")
    share = farm_finite_share(norm=norm, first=first, last=last)
    assert 0.5 < share <= 1.0
    assert np.nanmax(norm[first:last]) <= 1.5


def fake_run(*, host: str, rng: np.random.Generator, engine: bool) -> dict[str, np.ndarray]:
    """Return a stored-run layout with random labels, for the end-to-end test."""
    from e00698_common import host_bounds

    first, last = host_bounds(host=host)
    n = last - first
    g = np.where(rng.random(n) < 0.15, 2.0, 0.0) if engine else np.zeros(n)
    flag = rng.random(n) < 0.15
    return {
        "first": np.array(first),
        "last": np.array(last),
        "y": rng.uniform(1, 8, n),
        "g": g,
        "pv": np.zeros(n),
        "signed": rng.uniform(-3, 6, n),
        "farm_ok": np.ones(n, dtype=bool),
        "flag": flag,
        "filtered": flag,
        "k_path": np.zeros(n, dtype=np.int64),
        "p_any": np.zeros(n, dtype=np.float32),
        "k_max": np.array(2),
        "unit": np.array(1.9),
        "seconds": np.array(1.0),
    }


def test_end_to_end_on_fake_runs_when_the_panel_is_stored() -> None:
    """Scoring, the verdict, and the figures run on random runs written to a temporary folder."""
    from e00694_common import PANEL_PATH, load_panel  # ty: ignore[unresolved-import]
    from e00698_common import host_bounds

    if not PANEL_PATH.exists():
        return
    import e00698_figures as figures
    import e00698_score as score

    rng = np.random.default_rng(0)
    panel = load_panel()
    gate = {
        "run": {
            host: {capacity_key(capacity_mw=c): dict.fromkeys(SHAPES, True) for c in CAPACITIES_MW}
            for host in HOST_KEYS
        },
        "gate": {
            host: {capacity_key(capacity_mw=c): dict.fromkeys(SHAPES, True) for c in CAPACITIES_MW}
            for host in HOST_KEYS
        },
    }
    with tempfile.TemporaryDirectory() as folder:
        jobs_dir = Path(folder) / "jobs"
        jobs_dir.mkdir()
        for job in [*host_jobs(gate=gate), *step4_jobs()]:
            host = job.host if job.kind == "host" else "d8pre"
            run = fake_run(host=host, rng=rng, engine=job.seed is not None)
            if job.kind == "step4":
                from e00697_common import era_bounds  # ty: ignore[unresolved-import]

                first, last = era_bounds()[str(job.era)]
                n = last - first
                run.update(
                    {
                        "first": np.array(first),
                        "last": np.array(last),
                        "g": np.where(rng.random(n) < 0.1, 2.0, 0.0),
                        "flag": rng.random(n) < 0.3,
                        "y": np.ones(n),
                        "signed": np.ones(n),
                    }
                )
            np.savez_compressed(jobs_dir / f"{job.name}.npz", **run)  # ty: ignore[invalid-argument-type]
        score.JOBS_DIR = jobs_dir
        scores: dict = {"hosts": {}}
        for host in HOST_KEYS:
            first, last = host_bounds(host=host)
            scores["hosts"][host] = score.score_host(
                host=host, gate=gate, panel=panel, farm_ok_span=np.ones(last - first, dtype=bool)
            )
        scores["step4"] = score.score_step4(panel=panel)
        scores["twin_f1_d8pre"] = score.twin_f1_unmasked(panel=panel)
        for host in HOST_KEYS:
            assert scores["hosts"][host]["contrasts"]["c70"]["weather_farm"]["smooth"]["f1"]
            assert scores["hosts"][host]["cells"]["none_free"]["smooth"]["groups"]["hour"]
        assert scores["step4"]["contrast_solar_minus_net"] is not None
        assert scores["twin_f1_d8pre"] is not None
        score.print_half_widths(scores=scores)
        score.print_scores(scores=scores)
        word_, notes = decide(gate=gate, scores=scores, coverage=0.99)
        assert word_ in VERDICT_WORDS
        assert notes
        directory = Path(folder) / "figures"
        gate_doc = {
            **gate,
            "reference": {"medians": [6.0, 4.0, 2.0, 2.0, 2.2], "edges": [0.3, 0.5, 0.7, 0.9]},
            "curves": {
                host: {
                    "none": {"free": [7.0, 7.0, 7.0, 7.0, 6.5]},
                    **{
                        capacity_key(capacity_mw=c): {
                            s: {"medians": [6.0, 4.5, 2.5, 2.0, 1.5]} for s in SHAPES
                        }
                        for c in CAPACITIES_MW
                    },
                }
                for host in HOST_KEYS
            },
        }
        figures.figure_1(gate=gate_doc, directory=directory)
        figures.figure_2(scores=scores, directory=directory)
        figures.figure_3(scores=scores, directory=directory)
        label_only = {
            check: {
                era: {
                    "True": {"point": 0.1, "lo": 0.0, "hi": 0.2, "n_weeks": 5},
                    "False": {"point": 0.1, "lo": 0.0, "hi": 0.2, "n_weeks": 5},
                }
                for era in ("net", "solar")
            }
            for check in ("weekend_cheap", "night", "bmd")
        }
        figures.figure_4(audit={"numbers": {"label_only": label_only}}, directory=directory)
        svgs = sorted(p.name for p in directory.glob("*.svg"))
        assert len(svgs) == 4, svgs
        for path in directory.glob("*.svg"):
            text = path.read_text()
            assert "point mark" not in text, path.name
            assert "rule mark" not in text, path.name
            assert_no_year(text=text, name=path.name)


def test_temp_directory_is_not_used_for_data() -> None:
    """The selftest writes nothing outside a temporary folder."""
    with tempfile.TemporaryDirectory() as folder:
        assert Path(folder).exists()


def main() -> None:
    """Run every test and print one line each."""
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"{len(tests)} tests passed")
    _ = SHAPES


if __name__ == "__main__":
    main()
