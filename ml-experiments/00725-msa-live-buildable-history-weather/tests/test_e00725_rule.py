# ruff: noqa: E501
"""Tests of experiment 725's rule table, stopping-pass record, and cache assembly on tiny synthetic data.

Nothing here reads the data folder or experiment 713's modules: the tests that need experiment 713's
frozen checkout skip when `e00713_arms` is not importable.
"""

from datetime import UTC, date, datetime, timedelta

import e00725_common as cm
import e00725_rule as rule
import e00725_sources as src
import numpy as np
import pytest

CONTRACTS = (
    datetime(2025, 11, 12, 12, tzinfo=UTC),
    datetime(2025, 10, 28, 12, tzinfo=UTC),
    datetime(2026, 3, 31, 12, tzinfo=UTC),
    datetime(2026, 6, 20, 12, tzinfo=UTC),
)


def test_changed_value_count_is_3763() -> None:
    """The live rule changes 3,715 ENS values and 48 day-summary values."""
    changed = rule.changed_cells(rule="L")
    assert int(changed.sum()) == rule.CHANGED_VALUES_PER_SAMPLE
    assert int((rule.cell_sources(rule="L") == rule.SRC_ENS).sum()) == 3715


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("name", rule.RULES)
def test_grid_table_equals_datetime_table(contract: datetime, name: str) -> None:
    """The index arithmetic and the `datetime` arithmetic give the same source for every cell (check KL2)."""
    grid = rule.cell_sources(rule=name)  # ty: ignore[invalid-argument-type]
    reference = rule.reference_table(contract=contract, rule=name)  # ty: ignore[invalid-argument-type]
    assert np.array_equal(grid, reference)


def test_cams_boundary_at_age_three() -> None:
    """At age 3 the CAMS source stamp 00:00 UTC on D - 2 is in hand, and the half-hour before it too."""
    table = rule.cell_sources(rule="L")
    slot = 3 - rule.FIRST_AGE
    # age 3 stamps run from 12:00 UTC on D - 3; the source stamp of column j is at or before 00:00 UTC on D - 2
    # (12 hours after the row starts), so columns up to 24 read CAMS and later columns read ENS.
    assert table[slot, 24, 0] == rule.SRC_CAMS
    assert table[slot, 25, 0] == rule.SRC_ENS
    assert table[slot, 26, 0] == rule.SRC_ENS


def test_era5_first_age_is_seven() -> None:
    """ERA5 columns read the ENS value up to age 6 and the analysis from age 7."""
    table = rule.cell_sources(rule="L")
    for age in range(rule.FIRST_AGE, rule.FIRST_AGE + rule.N_ROWS):
        want = rule.SRC_ERA5 if age >= 7 else rule.SRC_ENS
        assert (table[age - rule.FIRST_AGE][:, 1:15] == want).all(), age


def test_rule_z_is_the_analysis_and_rule_e_is_ens_everywhere() -> None:
    """`Z` reads the analysis in every cell, and `E` reads ENS in columns 0 to 18 at every age."""
    z = rule.cell_sources(rule="Z")
    assert not (z == rule.SRC_ENS).any()
    assert not (z == rule.SRC_DAY_D3).any()
    e = rule.cell_sources(rule="E")
    assert (e[..., : rule.N_ENS_COLUMNS] == rule.SRC_ENS).all()


def test_day_summary_reads_d_minus_three_for_young_rows() -> None:
    """The summary read is the earlier of the day before the stamp day and D - 3."""
    day = np.array([100, 98, 97])
    live = rule.summary_day_read(token_day=day, contract_day=100, live=True)
    assert live.tolist() == [97, 97, 96]
    assert rule.summary_day_read(token_day=day, contract_day=100, live=False).tolist() == [
        99,
        97,
        96,
    ]


def test_stopping_record_uses_the_first_best_pass_and_the_steps_formula() -> None:
    """The best pass is the first pass within 1e-5 of the best loss, and steps are best pass times ceil(pairs / 256)."""
    first = {
        "passes": 6,
        "best_val": 0.9,
        "cap_hit": False,
        "val_history": [1.0, 0.95, 0.9, 0.95, 0.9, 0.91],
    }
    entry = {"stage": 3, "pairs": 513, "first": first, "rerun": None}
    rec = cm.stopping_record(entry=entry)
    assert rec["best_pass"] == 3
    assert rec["steps_at_best_pass"] == 3 * 3
    assert rec["val_pass_1"] == 1.0
    assert not rec["rerun_ran"]
    rerun = {"passes": 8, "best_val": 0.8, "cap_hit": False, "val_history": [1.0, 0.8]}
    entry2 = {**entry, "first": {**first, "cap_hit": True}, "rerun": rerun}
    rec2 = cm.stopping_record(entry=entry2)
    assert rec2["best_pass"] == 2
    assert rec2["rerun_ran"]
    assert rec2["cap_hit_first_run"]


class FakeLookup:
    """A B1 lookup that returns the origin index and valid index as two of the twelve values."""

    def get(
        self, *, primary: np.ndarray, origin_idx: np.ndarray, valid_idx: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return values that encode the key, found only where the valid index is even."""
        values = np.zeros((len(valid_idx), 14))
        values[:, 0] = origin_idx
        values[:, 1] = valid_idx
        return values, valid_idx % 2 == 0


def test_assemble_raw_and_standardise_cache() -> None:
    """The cache holds the ft weather and the looked-up B1 values, and standardising zeroes the missing cells."""
    n_primaries, g0, n_grid = 2, 6, 8
    cols = np.arange(n_primaries * g0 * 7, dtype=np.float64).reshape(n_primaries, g0, 7)
    origin_us = np.zeros((n_primaries, g0), dtype=np.int64)
    step = src.STEP_US
    origin_us[:, 1:5] = 10 * step
    raw, origin_idx, found = src.assemble_raw(
        cols=cols, origin_us=origin_us, grid_start_us=0, b1_lookup=FakeLookup(), n_grid=n_grid
    )
    assert raw.shape == (n_primaries, n_grid, 19)
    assert origin_idx[0, 0] == -1
    assert origin_idx[0, 1] == 10
    assert found[0].tolist() == [False, False, True, False, True, False, False, False]
    assert raw[0, 2, 7] == 10
    assert raw[0, 2, 8] == 2
    assert np.isnan(raw[0, 3, 7])
    cache = src.EnsCache(raw=raw, origin_idx=origin_idx, found=found)
    z = src.standardise_cache(
        cache=cache, weather_mean=np.zeros(7), weather_std=np.ones(7) * 2.0,
        b1_mean=np.zeros(14), b1_std=np.ones(14),
    )  # fmt: skip
    assert z.dtype == np.float16
    assert (z[0, 3] == 0).all()
    assert z[0, 2, 0] == np.float16(np.clip(cols[0, 2, 0] / 2.0, -5.0, 5.0))
    assert z[0, 2, 7] == np.float16(10)


def test_fallback_count_reference_counts_missing_ens_cells() -> None:
    """With no cell found, a sample has one fallback cell for each ENS cell of the table."""
    pytest.importorskip("e00713_arms")
    import e00725_poison as pz

    n_grid = rule.STEPS_PER_DAY * 130
    found = np.zeros((2, n_grid), dtype=bool)
    c = rule.STEPS_PER_DAY * 120 + 24
    count = pz.fallback_count_reference(
        found=found, rule_name="L", primary=np.array([0]), c_idx=np.array([c])
    )
    assert count == int((rule.cell_sources(rule="L") == rule.SRC_ENS).sum())
    found[:] = True
    assert (
        pz.fallback_count_reference(
            found=found, rule_name="L", primary=np.array([0]), c_idx=np.array([c])
        )
        == 0
    )


def test_clock_change_dates_are_in_the_history() -> None:
    """The check that picks origins near the two clock changes finds a grid index for each of them."""
    import e00725_checks as ck

    for change in ck.CLOCK_CHANGES:
        assert isinstance(change, date)
        idx = ck.contract_index(day=change + timedelta(days=2))
        assert ck.contract_datetime(c_idx=idx).date() == change + timedelta(days=2)
