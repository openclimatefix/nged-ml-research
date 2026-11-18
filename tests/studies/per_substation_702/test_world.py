# ruff: noqa: ANN001, ANN202, E501, PT018
"""Tests of the one-Primary world of experiment 702 (isolation by construction, shifts equal across arms)."""

from types import SimpleNamespace

import e00702_world as world
import numpy as np
import polars as pl
import pytest

N = 16
GRID = 3000


def _series() -> pl.DataFrame:
    sites = [f"D{i}" for i in (2, 3, 4, 5, 8, 9, 10, 11, 12, 14, 15, 16, 18, 19, 21, 22)]
    return pl.DataFrame(
        {
            "site": [*sites, "B1", "G1", "B2", "G2"],
            "time_series_id": list(range(20)),
            "substation_type": ["Primary"] * 16 + ["BSP", "GSP", "BSP", "GSP"],
        }
    )


def test_one_site_world_holds_one_primary_and_reports_its_pooled_index():
    """After the patch, `series_table()` returns the one Primary, and its index is its place among the 16."""
    common = SimpleNamespace(series_table=_series)
    index = world.restrict_to_primary(common=common, label="D15")
    assert index == 10
    assert common.series_table()["site"].to_list() == ["D15"]
    assert common.series_table()["time_series_id"].to_list() == [10]


def test_a_label_that_is_not_a_primary_is_refused():
    """A BSP label, or a label that does not exist, cannot be a one-site world."""
    for label in ("B1", "D99"):
        with pytest.raises(AssertionError):
            world.restrict_to_primary(common=SimpleNamespace(series_table=_series), label=label)


bank = pytest.importorskip("e00581_bank")


def _pooled(*, seed: int = 202511):
    rng = np.random.default_rng(0)
    clean = rng.normal(size=(N, GRID))
    clean[3, :200] = np.nan  # a short history
    first, last = 100, GRID - 50
    return clean, first, last, seed


@pytest.mark.parametrize("primary", [0, 7, 15])
def test_shifts_of_one_primary_equal_its_rows_in_the_pooled_call(primary):
    """The one-row call with the fold's span gives Primary P the shifts it has in the global arm."""
    clean, first, last, seed = _pooled()
    pooled, n_pooled = bank.synthetic_clean(clean=clean, first=first, last=last, seed=seed)
    trick = world.schedule_trick(
        original=bank.synthetic_clean, index=primary, n_primaries=N, first=first, last=last
    )
    own, n_own = trick(
        clean=clean[primary : primary + 1], first=first + 5, last=last - 5, seed=seed
    )
    np.testing.assert_array_equal(own[0], pooled[primary])
    assert n_own == n_pooled  # the call draws the same events: the other rows are NaN, not absent


def test_shifts_of_one_primary_do_not_depend_on_the_other_primaries_values():
    """Replacing every finite value of the other 15 rows with random numbers leaves Primary P's shifts unchanged."""
    clean, first, last, seed = _pooled()
    other = clean.copy()
    mask = np.arange(N) != 6
    other[mask] = np.where(
        np.isfinite(other[mask]), np.random.default_rng(9).normal(size=(15, GRID)) * 50, np.nan
    )
    a, _ = bank.synthetic_clean(clean=clean, first=first, last=last, seed=seed)
    b, _ = bank.synthetic_clean(clean=other, first=first, last=last, seed=seed)
    np.testing.assert_array_equal(a[6] - clean[6], b[6] - other[6])
    trick = world.schedule_trick(
        original=bank.synthetic_clean, index=6, n_primaries=N, first=first, last=last
    )
    own, _ = trick(clean=clean[6:7], first=first, last=last, seed=seed)
    np.testing.assert_array_equal(own[0] - clean[6], a[6] - clean[6])


def test_a_span_wider_than_the_folds_is_refused():
    """A Primary whose own span leaves the fold's span would draw another schedule, so the call stops."""
    clean, first, last, seed = _pooled()
    trick = world.schedule_trick(
        original=bank.synthetic_clean, index=0, n_primaries=N, first=first, last=last
    )
    with pytest.raises(AssertionError):
        trick(clean=clean[:1], first=first - 1, last=last, seed=seed)


def test_the_stored_cache_is_sliced_and_never_written(tmp_path):
    """The one-row table is row P of the cache, and the cache folder is unchanged after the call."""
    cols = np.random.default_rng(0).normal(size=(N, 40, 28)).astype(np.float32)
    origin = np.arange(N * 40, dtype=np.int64).reshape(N, 40)
    np.save(tmp_path / "expected_ft_cols.npy", cols)
    np.save(tmp_path / "expected_ft_origin.npy", origin)
    before = sorted(p.name for p in tmp_path.iterdir())
    got_cols, got_origin = world.sliced_raw_table(index=5)(source="ft", cache=tmp_path)
    np.testing.assert_array_equal(got_cols, cols[5:6])
    np.testing.assert_array_equal(got_origin, origin[5:6])
    assert sorted(p.name for p in tmp_path.iterdir()) == before
    with pytest.raises(AssertionError):
        world.sliced_raw_table(index=5)(source="pre", cache=tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == before


def _cleaned_fixture(*, tmp_path, label: str):
    """Return stand-in modules and a flagged-key file for `one_site_cleaned_arrays`."""
    series = _series().filter(pl.col("site") == label)
    power = np.arange(GRID, dtype=float) + 1.0
    arrays = {label: SimpleNamespace(power=power)}
    fd = SimpleNamespace(_series_and_arrays=lambda: (series, arrays))
    grid = pl.datetime_range(
        pl.datetime(2019, 9, 1, time_zone="UTC"),
        pl.datetime(2019, 12, 1, time_zone="UTC"),
        "30m",
        eager=True,
    )

    def grid_index(*, times):
        """Return half-hour indices of a series, or one index for a single timestamp."""
        if isinstance(times, pl.Series):
            return ((times - grid[0]).dt.total_minutes() // 30).to_numpy()
        return int((times - grid[0]).total_seconds() // 1800)

    common = SimpleNamespace(grid_index=grid_index)
    stale = SimpleNamespace(
        SHORT_HISTORY_LABELS=("D15", "D18"),
        SHORT_HISTORY_CUT=grid[100],
    )
    own_id = series["time_series_id"][0]
    flagged = pl.DataFrame(
        {"time_series_id": [own_id, own_id, 999], "valid_time": [grid[10], grid[11], grid[12]]}
    )
    path = tmp_path / "flagged.parquet"
    flagged.write_parquet(path)
    return SimpleNamespace(stale=stale), fd, common, path, power


def test_one_site_cleaned_power_drops_only_its_own_flagged_half_hours(tmp_path):
    """The half-hours flagged for the Primary are NaN; a flag of another series changes nothing."""
    prep422, fd, common, path, power = _cleaned_fixture(tmp_path=tmp_path, label="D5")
    out = world.one_site_cleaned_arrays(prep422=prep422, fd=fd, common=common, flagged_keys=path)
    assert list(out) == ["D5"]
    assert np.isnan(out["D5"][[10, 11]]).all()
    assert np.isfinite(out["D5"][12])
    assert np.isfinite(out["D5"][:10]).all()
    assert np.array_equal(power, np.arange(GRID) + 1.0), "the source array must not be modified"


def test_short_history_cut_applies_to_d15_and_d18_only(tmp_path):
    """D15 loses the half-hours before the cut; D5 keeps them."""
    prep422, fd, common, path, _ = _cleaned_fixture(tmp_path=tmp_path, label="D15")
    out = world.one_site_cleaned_arrays(prep422=prep422, fd=fd, common=common, flagged_keys=path)
    assert np.isnan(out["D15"][:100]).all() and np.isfinite(out["D15"][100:]).sum() > 0
    prep422, fd, common, path, _ = _cleaned_fixture(tmp_path=tmp_path, label="D5")
    out = world.one_site_cleaned_arrays(prep422=prep422, fd=fd, common=common, flagged_keys=path)
    assert np.isfinite(out["D5"][:10]).all()


lags = pytest.importorskip("e00572_lags")


def test_lag_statistics_of_one_primary_ignore_the_other_primaries_power():
    """`lag_stats` on one Primary's rows is unchanged when the other 15 rows of the power array are randomised."""
    rng = np.random.default_rng(3)
    n_rows = 4000
    primary = np.full(n_rows, 4)
    origin_idx = rng.integers(2000, 2400, n_rows)
    target_idx = origin_idx + rng.integers(14, 400, n_rows)
    rows = SimpleNamespace(
        is_val=rng.random(n_rows) < 0.1,
        target_idx=target_idx,
        origin_idx=origin_idx,
        primary=primary,
    )
    clean = rng.normal(size=(N, 3000))
    scrambled = clean.copy()
    keep = np.arange(N) != 4
    scrambled[keep] = np.where(
        np.isfinite(scrambled[keep]), rng.normal(size=(15, 3000)) * 7, np.nan
    )
    a = lags.lag_stats(rows=rows, clean=clean, cap_steps=lags.CAP_24H)
    b = lags.lag_stats(rows=rows, clean=scrambled, cap_steps=lags.CAP_24H)
    np.testing.assert_array_equal(a, b)
    one = lags.lag_stats(
        rows=SimpleNamespace(**{**rows.__dict__, "primary": np.zeros(n_rows, dtype=int)}),
        clean=clean[4:5],
        cap_steps=lags.CAP_24H,
    )
    np.testing.assert_array_equal(a, one)


def test_a_pooled_lag_statistic_does_depend_on_the_other_primaries():
    """The negative control: pooled rows over all 16 Primaries change when other rows are randomised."""
    rng = np.random.default_rng(4)
    n_rows = 4000
    primary = rng.integers(0, N, n_rows)
    origin_idx = rng.integers(2000, 2400, n_rows)
    rows = SimpleNamespace(
        is_val=np.zeros(n_rows, bool),
        target_idx=origin_idx + 100,
        origin_idx=origin_idx,
        primary=primary,
    )
    clean = rng.normal(size=(N, 3000))
    scrambled = clean.copy()
    scrambled[1:] = rng.normal(size=(15, 3000)) * 7
    assert not np.array_equal(
        lags.lag_stats(rows=rows, clean=clean, cap_steps=lags.CAP_24H),
        lags.lag_stats(rows=rows, clean=scrambled, cap_steps=lags.CAP_24H),
    )


LABELS_16 = (
    "D2",
    "D3",
    "D4",
    "D5",
    "D8",
    "D9",
    "D10",
    "D11",
    "D12",
    "D14",
    "D15",
    "D16",
    "D18",
    "D19",
    "D21",
    "D22",
)


def test_path_naming_another_primary_is_found_by_whole_token():
    """`D22` is a token of its own, so a path of D2 does not match D22 and a path of D22 does not match D2."""
    names = world.other_primary_in_path
    assert names(path="/x/summary/D5.npz", own="D5", labels=LABELS_16) is None
    assert names(path="/x/summary/D9.npz", own="D5", labels=LABELS_16) == "D9"
    assert names(path="/x/trailing/D22.parquet", own="D2", labels=LABELS_16) == "D22"
    assert names(path="/x/trailing/D2.parquet", own="D22", labels=LABELS_16) == "D2"
    assert (
        names(path="/x/00702-run/arms/split1/2026-05/D5/work/m.pt", own="D5", labels=LABELS_16)
        is None
    )
    assert names(path="/x/P0_D50_2026-05_5w_sp1/bank.parquet", own="D5", labels=LABELS_16) is None


def test_guarded_readers_refuse_another_primarys_file_and_log_the_rest():
    """The one-Primary process can open its own files and shared files, and raises on a sibling's file."""
    seen = []
    fake = SimpleNamespace(load=lambda path, **kw: seen.append(path) or "data")
    opened = world.guard_readers(own="D5", labels=LABELS_16, modules={"np.load": (fake, "load")})
    assert fake.load("/x/summary/D5.npz") == "data"
    assert fake.load("/x/shared/flagged_keys.parquet", mmap_mode="r") == "data"
    with pytest.raises(PermissionError, match="D9"):
        fake.load("/x/summary/D9.npz")
    assert seen == ["/x/summary/D5.npz", "/x/shared/flagged_keys.parquet"]
    assert opened == seen


def test_guard_reads_the_path_from_the_file_keyword():
    """`pl.scan_parquet(source=...)` and `np.load(file=...)` are guarded too."""
    fake = SimpleNamespace(read=lambda **kw: "data")
    world.guard_readers(own="D5", labels=LABELS_16, modules={"read": (fake, "read")})
    with pytest.raises(PermissionError):
        fake.read(source="/x/D8.parquet")
