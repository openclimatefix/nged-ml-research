# ruff: noqa: PT018, D103
"""Experiment 713: CPU checks of the layout (KA7 and KA9). Exit code 0 means pass.

KA7: the grid indices and per-token day types of 1,000 random samples equal an independent
calendar computation with `datetime` and `zoneinfo`, and the masked columns of the age-2 row are
exactly 43 to 47. KA9: no placebo row is younger than 16 days, the draw is identical on a second
call, and no placebo row repeats inside a sample.
"""

import sys
from datetime import UTC, datetime, timedelta

import e00713_layout as lay
import numpy as np

GRID_START = datetime(2019, 9, 1, tzinfo=UTC)
GRID_START_US = int(GRID_START.timestamp() * 1e6)
N_GRID = 48 * 3000
N_SAMPLES = 1_000


def main() -> int:
    holidays = lay.england_bank_holidays(years=range(2019, 2029))
    table = lay.day_types_for_grid(grid_start_us=GRID_START_US, n_grid=N_GRID, holidays=holidays)
    rng = np.random.default_rng(0)

    # the age-2 row is masked exactly at columns 43 to 47
    assert lay.masked_columns_of_newest_row().tolist() == [43, 44, 45, 46, 47]
    one = np.array([1000])
    hist = lay.history_grid_index(contract_idx=one)
    vis = lay.visible_by_time(contract_idx=one, grid_index=hist)
    assert vis[0, 0].sum() == 43 and vis[0, 1:].all(), "ages 3 to 15 must be fully visible"
    assert hist[0, 0, 0] == 1000 - 96 and hist[0, -1, 0] == 1000 - 15 * 48
    top = lay.target_grid_index(contract_idx=one, lead_day=np.array([0]))
    assert top[0, 0] == 1000 and top[0, -1] == 1047
    last = lay.target_grid_index(contract_idx=one, lead_day=np.array([13]))
    assert last[0, -1] == 1000 + 672 - 1, "14 target rows tile 672 half-hours"

    day = rng.integers(60, 2900, N_SAMPLES)
    day_start_us = GRID_START_US + day * lay.DAY_US
    c_idx = lay.contract_index(day_start_us=day_start_us, grid_start_us=GRID_START_US)
    lead = rng.integers(0, 14, N_SAMPLES)
    hist = lay.history_grid_index(contract_idx=c_idx)
    top = lay.target_grid_index(contract_idx=c_idx, lead_day=lead)
    for i in range(N_SAMPLES):
        c = datetime.fromtimestamp(day_start_us[i] / 1e6, UTC) + timedelta(hours=12)
        assert c.hour == 12
        for a_slot, age in enumerate(range(2, 16)):
            for j in (0, 17, 25, 47):
                stamp = c - timedelta(hours=24 * age) + timedelta(minutes=30 * j)
                want = int((stamp - GRID_START).total_seconds() // 1800)
                assert hist[i, a_slot, j] == want, (i, age, j)
                ref = lay.day_type_reference(
                    grid_index=want, grid_start_us=GRID_START_US, holidays=holidays
                )
                assert table[want] == ref, (i, age, j, int(table[want]), ref)
        for j in (0, 30, 47):
            stamp = c + timedelta(hours=24 * int(lead[i])) + timedelta(minutes=30 * j)
            want = int((stamp - GRID_START).total_seconds() // 1800)
            assert top[i, j] == want
            ref = lay.day_type_reference(
                grid_index=want, grid_start_us=GRID_START_US, holidays=holidays
            )
            assert table[want] == ref
    # the whole-table check around both clock changes of 2023 and a bank holiday
    for stamp, expect in (
        (datetime(2023, 12, 25, 12, 0, tzinfo=UTC), lay.HOLIDAY_DAY_TYPE),
        (datetime(2023, 3, 26, 0, 30, tzinfo=UTC), 6),
        (datetime(2023, 10, 29, 12, 0, tzinfo=UTC), 6),
        (datetime(2023, 6, 29, 23, 30, tzinfo=UTC), 4),
        (datetime(2023, 6, 30, 0, 0, tzinfo=UTC), 4),
        (datetime(2023, 6, 30, 23, 30, tzinfo=UTC), 5),
    ):
        g = int((stamp - GRID_START).total_seconds() // 1800)
        assert table[g] == lay.day_type_reference(
            grid_index=g, grid_start_us=GRID_START_US, holidays=holidays
        ), stamp
        assert table[g] == expect, (stamp, int(table[g]), expect)
    # visibility: a poisoned-index test of the 27 h rule
    assert (lay.newest_visible_index(contract_idx=one) == 946).all()

    # KA9: placebo draw
    for primary, origin_day in ((0, 3000), (7, 3123), (15, 4000)):
        a = lay.placebo_ages(primary=primary, origin_day=origin_day)
        b = lay.placebo_ages(primary=primary, origin_day=origin_day)
        assert np.array_equal(a, b), "the draw must repeat"
        assert len(a) == 14 and len(set(a.tolist())) == 14, "no repeat inside a sample"
        assert a.min() >= 16 and a.max() <= 379, a
        assert (np.diff(a) > 0).all()
    other = lay.placebo_ages(primary=1, origin_day=3000)
    assert not np.array_equal(other, lay.placebo_ages(primary=0, origin_day=3000))
    avail = np.zeros(364, dtype=bool)
    avail[:5] = True
    few = lay.placebo_ages(primary=0, origin_day=3000, available=avail)
    assert len(few) == 5 and few.max() <= 20
    print("KA7 and KA9 passed:", N_SAMPLES, "samples")
    return 0


if __name__ == "__main__":
    sys.exit(main())
