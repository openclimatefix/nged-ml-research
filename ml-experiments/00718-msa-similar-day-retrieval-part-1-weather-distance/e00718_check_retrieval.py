# ruff: noqa: E501, PT018
"""Experiment 718, WP1: checks of the retrieval measures, the placebo draws (KR4), and the row
tables.

Every check runs on synthetic arrays on the CPU: a grid of 600 days, 3 Primaries with gaps,
random standardised weather, and a weekly day type. The independent recomputation of KR3 is
WP2's. Usage: e00718_check_retrieval.py
"""  # noqa: D205

import sys
from dataclasses import replace
from datetime import date

import e00718_layout as lay
import e00718_retrieval as ret
import numpy as np

N_DAYS_GRID = 600
N_PRIMARIES = 3
FIRST_DAY = 420
ORIGIN_DAYS = np.arange(FIRST_DAY, FIRST_DAY + 6)
ALL_ARMS = ("Z", "R", "R-A", "P90", "U", "U-A", "U27", "U27-A", "UX")


def synthetic_arrays(
    *, seed: int = 0, with_calendar: bool = True
) -> tuple[ret.PoolArrays, np.ndarray]:
    """Return synthetic retrieval arrays and the power they were built from."""
    rng = np.random.default_rng(seed)
    n_grid = N_DAYS_GRID * lay.STEPS_PER_DAY
    power = rng.normal(size=(N_PRIMARIES, n_grid))
    power[rng.random(power.shape) < 0.03] = np.nan
    power[0, 48 * 100 : 48 * 130] = np.nan  # a gap of 30 days at Primary 0
    power[:, : 48 * 5] = np.nan
    finite = np.isfinite(power).astype(np.int64)
    cumsum = np.concatenate([np.zeros((N_PRIMARIES, 1), np.int64), finite.cumsum(axis=1)], axis=1)
    weather = rng.normal(size=(N_PRIMARIES, n_grid, ret.N_WEATHER))
    day_type = ((np.arange(n_grid) // 48 + 1) % 7).astype(np.int64)
    day_class = None
    if with_calendar:
        day_class = day_type.copy()
    offsets, window, _ = ret.christmas_arrays(n_days=4000)
    arrays = ret.PoolArrays(
        finite_cumsum=cumsum, weather=weather, day_type=day_type, day_class=day_class,
        christmas_offset=offsets[:N_DAYS_GRID], in_window=window[:N_DAYS_GRID],
    )  # fmt: skip
    return arrays, power


def provider_for(
    *, arrays: ret.PoolArrays, filled_share: float = 1.0, seed: int = 1
) -> ret.TargetProvider:
    """Return a target provider that perturbs the stored weather of the target row, with gaps."""

    def provide(primary: int, day: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng([seed, primary, day])
        c = int(lay.contract_idx_of_day(day=day))
        idx = c + np.arange(lay.N_LEAD_DAYS * 48)
        idx = np.minimum(idx, arrays.n_grid - 1)
        w = arrays.weather[primary][idx].reshape(lay.N_LEAD_DAYS, 48, ret.N_WEATHER) + rng.normal(
            scale=0.3, size=(lay.N_LEAD_DAYS, 48, ret.N_WEATHER)
        )
        filled = rng.random((lay.N_LEAD_DAYS, 48)) < filled_share
        return w, filled

    return provide


def build(
    *,
    arm: str,
    arrays: ret.PoolArrays,
    days: np.ndarray = ORIGIN_DAYS,
    provider: ret.TargetProvider | None = None,
) -> ret.RowTable:
    """Build the table of `arm` for all Primaries."""
    return ret.build_table(
        arm=arm, arrays=arrays, primaries=np.arange(N_PRIMARIES), days=days,
        target_provider=provider or provider_for(arrays=arrays),
    )  # fmt: skip


def check_tables(*, tables: dict[str, ret.RowTable], arrays: ret.PoolArrays) -> list[str]:
    """Return the lines of the structural checks of every table (the asserted part of KR3)."""
    lines = []
    for arm, tab in tables.items():
        n_rows = lay.ROWS_PER_ARM[arm]
        assert tab.ages.shape[-1] == n_rows
        for pi, p in enumerate(tab.primaries):
            for di, day in enumerate(tab.days):
                c = int(lay.contract_idx_of_day(day=int(day)))
                for lead in range(lay.N_LEAD_DAYS):
                    a = tab.ages[pi, di, lead].astype(np.int64)
                    got = a[a >= 0]
                    assert bool((np.diff(got) > 0).all()), f"{arm}: ages not ascending or repeating"
                    assert (a[len(got) :] == lay.ABSENT).all(), (
                        f"{arm}: an absent row before a present row"
                    )
                    if arm == "Z":
                        assert (got == np.arange(2, 16)).all()
                        continue
                    if arm == "UX":
                        assert got.max(initial=0) <= int(day) + 1
                    else:
                        max_age = (
                            lay.SHORT_POOL_MAX_AGE
                            if arm in {"R", "R-A", "P90"}
                            else lay.LONG_POOL_MAX_AGE
                        )
                        assert (
                            got.min(initial=2) >= lay.MIN_AGE and got.max(initial=0) <= max_age
                        ), f"{arm}: ages outside the pool"
                    fixed = (
                        got[got <= lay.NEWEST_FIXED_MAX_AGE]
                        if arm in {"U27", "U27-A"}
                        else np.empty(0, np.int64)
                    )
                    rest = np.setdiff1d(got, fixed)
                    ok = ret.eligible_ages(arrays=arrays, primary=int(p), contract_idx=c, ages=rest)
                    assert ok.all(), f"{arm}: an ineligible block was chosen"
                    if arm in {"U27", "U27-A"}:
                        assert len(fixed) == 14, (
                            f"{arm}: the fixed rows of ages 2 to 15 are not all present"
                        )
                    if arm in {"U", "U-A", "UX"}:
                        meas = tab.measure[pi, di, lead][: len(got)]
                        recent = got[meas == lay.MEASURE_RECENCY]
                        pool = np.arange(2, 380)
                        pool = pool[c - 48 * pool >= 0]
                        elig = pool[
                            ret.eligible_ages(
                                arrays=arrays, primary=int(p), contract_idx=c, ages=pool
                            )
                        ]
                        if arm != "U-A":
                            chosen_other = got[meas == lay.MEASURE_WEATHER]
                            youngest = elig[~np.isin(elig, chosen_other)][: len(recent)]
                            assert (recent == youngest).all(), (
                                "recency rows are not the youngest eligible blocks"
                            )
                            wk = got[meas == lay.MEASURE_WEEKDAY]
                            assert ((wk + lead) % 7 == 0).all(), "a weekday row has another weekday"
        lines.append(
            f"{arm}: structure ok ({tab.ages.shape[0] * tab.ages.shape[1] * tab.ages.shape[2]} samples, hash {tab.digest()})"
        )
    return lines


def check_placebos(*, tables: dict[str, ret.RowTable], arrays: ret.PoolArrays) -> list[str]:
    """KR4: the placebo draws."""
    lines = []
    p90 = tables["P90"]
    again = build(arm="P90", arrays=arrays)
    assert p90.digest() == again.digest(), "P90 differs on a second call"
    for pi in range(N_PRIMARIES):
        for di in range(len(ORIGIN_DAYS)):
            rows = [tuple(p90.ages[pi, di, k]) for k in range(lay.N_LEAD_DAYS)]
            assert len(set(rows)) > 1, "P90 is identical across lead days"
            for r in rows:
                got = [a for a in r if a >= 0]
                assert len(got) == len(set(got)) and all(2 <= a <= 91 for a in got)
    lines.append(
        "P90: ages 2 to 91, no repeat, identical on a second call, different across lead days"
    )
    for placebo, treat, keep in (
        ("R-A", "R", None),
        ("U-A", "U", lay.MEASURE_RECENCY),
        ("U27-A", "U27", lay.MEASURE_FIXED),
    ):
        a, t = tables[placebo], tables[treat]
        assert a.digest() == build(arm=placebo, arrays=arrays).digest(), (
            f"{placebo} differs on a second call"
        )
        n_checked = 0
        for pi in range(N_PRIMARIES):
            for di in range(len(ORIGIN_DAYS)):
                for k in range(lay.N_LEAD_DAYS):
                    pa, ta = a.ages[pi, di, k].astype(np.int64), t.ages[pi, di, k].astype(np.int64)
                    pm, tm = a.measure[pi, di, k], t.measure[pi, di, k]
                    pg, tg = pa[pa >= 0], ta[ta >= 0]
                    assert len(pg) == len(set(pg.tolist()))
                    if keep is not None:
                        assert (
                            np.sort(tg[tm[: len(tg)] == keep]) == np.sort(pg[pm[: len(pg)] == keep])
                        ).all(), f"{placebo}: the kept rows differ from the treatment's"
                    replaced_p = [
                        x for x, m in zip(pg, pm[: len(pg)], strict=True) if m == lay.MEASURE_RANDOM
                    ]
                    assert not set(replaced_p) & set(tg.tolist()), (
                        f"{placebo}: a replaced row is a row of the treatment"
                    )
                    n_checked += 1
        lines.append(
            f"{placebo}: kept rows equal the treatment's, no replaced row is a treatment row, no repeat, repeatable ({n_checked} samples)"
        )
    return lines


def check_class_counts(*, arrays: ret.PoolArrays) -> list[str]:
    """KR4: each placebo's count of rows per age class equals its treatment's, where no class is
    exhausted, and its rows are farther in weather than the treatment's.
    """  # noqa: D205
    lines = []
    for treatment, placebo in (("R", "R-A"), ("U", "U-A"), ("U27", "U27-A")):
        r, ra = build(arm=treatment, arrays=arrays), build(arm=placebo, arrays=arrays)
        same = total = 0
        for idx in np.ndindex(r.ages.shape[:3]):
            t = r.ages[idx].astype(np.int64)
            p = ra.ages[idx].astype(np.int64)
            total += 1
            if (p >= 0).sum() == (t >= 0).sum():
                ct = np.bincount(lay.age_class(ages=t[t >= 0]), minlength=11)
                cp = np.bincount(lay.age_class(ages=p[p >= 0]), minlength=11)
                same += int((ct == cp).all())
        assert same >= 0.9 * total, (
            f"{placebo}: only {same} of {total} samples match the treatment's class counts"
        )
        assert r.distance is not None and ra.distance is not None
        near = r.distance[r.measure == lay.MEASURE_WEATHER]
        far = ra.distance[ra.measure == lay.MEASURE_RANDOM]
        near, far = near[np.isfinite(near)], far[np.isfinite(far)]
        assert far.mean() > near.mean(), (
            f"{placebo}'s random rows are not farther than {treatment}'s weather rows"
        )
        lines.append(
            f"{placebo}: class counts equal {treatment}'s in {same} of {total} samples (the rest passed a shortfall to an older class); mean weather distance of the replaced rows {far.mean():.2f} against {near.mean():.2f} for {treatment}'s weather rows"
        )
    return lines


def check_distance(*, arrays: ret.PoolArrays) -> str:
    """The distance of a hand-built case, and the tie rule."""
    rng = np.random.default_rng(3)
    pool_w = rng.normal(size=(5, 48, 7))
    pool_dt = rng.integers(0, 7, size=(5, 48))
    target = ret.TargetRow(
        weather=rng.normal(size=(48, 7)),
        filled=np.ones(48, bool),
        day_type=rng.integers(0, 7, size=48),
    )
    got = ret.weather_distance(target=target, pool_weather=pool_w, pool_day_type=pool_dt)
    for b in range(5):
        expect = (
            np.sqrt(((pool_w[b] - target.weather) ** 2).sum())
            + (pool_dt[b] != target.day_type).mean()
        )
        assert abs(got[b] - expect) < 1e-12
    half = replace(target, filled=np.arange(48) < 24)
    got_half = ret.weather_distance(target=half, pool_weather=pool_w, pool_day_type=pool_dt)
    expect = (
        np.sqrt(((pool_w[0, :24] - target.weather[:24]) ** 2).sum() * 2.0)
        + (pool_dt[0] != target.day_type).mean()
    )
    assert abs(got_half[0] - expect) < 1e-12
    ages = np.array([5, 3, 9, 4])
    dist = np.array([1.0, 1.0 + 5e-10, 0.5, 1.0])
    pick = ret.pick_nearest(distance=dist, ages=ages, allowed=np.ones(4, bool), n=3)
    assert pick.tolist() == [9, 3, 4], pick
    return "weather distance equals the written formula; scaling over unfilled half-hours; near-ties go to the younger block"


def check_no_weather_and_poison(*, arrays: ret.PoolArrays, power: np.ndarray) -> list[str]:
    """A target row with no weather takes the youngest blocks, and the tables ignore power after the
    origin (KR2 (a) on the tables).
    """  # noqa: D205
    lines = []
    empty = build(arm="R", arrays=arrays, provider=provider_for(arrays=arrays, filled_share=0.0))
    assert empty.no_weather.all()
    base = build(arm="Z", arrays=arrays)
    assert base.no_weather.sum() == 0
    for arm in ("R", "U", "U27"):
        build(arm=arm, arrays=arrays)
        for value in (1e6, np.nan):
            bad_power = power.copy()
            for day in ORIGIN_DAYS:
                c = int(lay.contract_idx_of_day(day=int(day)))
                bad_power[:, c - ret.NEWEST_VISIBLE_BEFORE_CONTRACT + 1 :] = value
            # readings newer than the 27 h rule differ per origin, so test one origin at a time
            for day in ORIGIN_DAYS[:2]:
                c = int(lay.contract_idx_of_day(day=int(day)))
                p2 = power.copy()
                p2[:, c - ret.NEWEST_VISIBLE_BEFORE_CONTRACT + 1 :] = value
                finite = np.isfinite(p2).astype(np.int64)
                a2 = replace(
                    arrays,
                    finite_cumsum=np.concatenate(
                        [np.zeros((N_PRIMARIES, 1), np.int64), finite.cumsum(axis=1)], axis=1
                    ),
                )
                one = build(arm=arm, arrays=a2, days=np.array([day]))
                ref = build(arm=arm, arrays=arrays, days=np.array([day]))
                assert (one.ages == ref.ages).all(), (
                    f"{arm}: the table read power newer than the 27 h rule ({value})"
                )
        lines.append(
            f"{arm}: power newer than origin minus 27 h set to 1e6 and to NaN leaves the table unchanged"
        )
    # negative control: a pool that admits the newest block changes the table
    c = int(lay.contract_idx_of_day(day=int(ORIGIN_DAYS[0])))
    ages = np.array([0, 1, 2])
    el = ret.eligible_ages(arrays=arrays, primary=1, contract_idx=c, ages=ages)
    assert not el[0], "the block of age 0 must not be eligible"
    lines.append("negative control: the blocks of age 0 and 1 are never eligible")
    return lines


def check_ux(*, arrays: ret.PoolArrays) -> str:
    """Arm `UX` equals `U` outside the window, and in the window takes calendar rows of an earlier winter."""
    u = build(arm="U", arrays=arrays)
    ux = build(arm="UX", arrays=arrays)
    outside = 0
    for idx in np.ndindex(u.ages.shape[:3]):
        day = int(ORIGIN_DAYS[idx[1]]) + idx[2]
        if not bool(ret._CHRISTMAS_CACHE[1][day]):
            assert (u.ages[idx] == ux.ages[idx]).all(), "UX differs from U outside the window"
            outside += 1
    days = np.arange(
        ret.day_number_of_date(d=date(2020, 12, 22)), ret.day_number_of_date(d=date(2020, 12, 24))
    )
    w_u = build(arm="U", arrays=arrays, days=days)
    w_ux = build(arm="UX", arrays=arrays, days=days)
    calendar_rows = int((w_ux.measure == lay.MEASURE_CALENDAR).sum())
    assert calendar_rows > 0, "UX took no calendar row in the window"
    assert not (w_u.ages == w_ux.ages).all(), "UX equals U inside the window"
    for pi, di, k in np.ndindex(w_ux.ages.shape[:3]):
        got = w_ux.ages[pi, di, k][w_ux.measure[pi, di, k] == lay.MEASURE_CALENDAR].astype(np.int64)
        start = days[di] - got
        assert ret._CHRISTMAS_CACHE[1][start].all(), (
            "a calendar row starts outside 20 December to 5 January"
        )
        assert (ret._CHRISTMAS_CACHE[2][start] != ret._CHRISTMAS_CACHE[2][days[di] + k]).all(), (
            "a calendar row is not from an earlier winter"
        )
    return f"UX equals U on {outside} samples outside the window and takes {calendar_rows} calendar rows of earlier winters inside it"


def main() -> int:
    """Run every check."""
    arrays, power = synthetic_arrays()
    tables = {arm: build(arm=arm, arrays=arrays) for arm in ALL_ARMS}
    lines = check_tables(tables=tables, arrays=arrays)
    lines += check_placebos(tables=tables, arrays=arrays)
    lines += check_class_counts(arrays=arrays)
    lines.append(check_distance(arrays=arrays))
    lines += check_no_weather_and_poison(arrays=arrays, power=power)
    lines.append(check_ux(arrays=arrays))
    h1 = {arm: tab.digest() for arm, tab in tables.items()}
    h2 = {arm: build(arm=arm, arrays=arrays).digest() for arm in ALL_ARMS}
    assert h1 == h2, "a table is not reproducible"
    lines.append(f"every table is reproducible: {h1}")
    print("\n".join(lines))
    print("e00718_check_retrieval: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
