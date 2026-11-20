# ruff: noqa: ANN001, PERF401
"""Tests of the cloud summary, its checks, and the era step tests of experiment 711."""

from datetime import UTC, date, datetime, timedelta
from typing import Any

import e00711_cloud as cloud
import e00711_common as common
import numpy as np
import polars as pl
import pytest

RUN = datetime(2026, 1, 14, tzinfo=UTC)
DTYPE = pl.Datetime("us", "UTC")


def test_cache_run_calendar_matches_the_plan():
    rows_from_2021 = common.cache_runs(first=common.ROW_FIRST_RUN)
    backfill = [r for r in rows_from_2021 if r <= common.BACKFILL_LAST_RUN]
    recent = [r for r in rows_from_2021 if r >= common.RECENT_FIRST_RUN]
    assert len(backfill) == 985
    assert len(recent) == 222
    assert all(not common.is_missing_run(run=r) for r in rows_from_2021)
    assert datetime(2024, 1, 3, tzinfo=UTC) not in rows_from_2021
    assert common.is_missing_run(run=datetime(2024, 4, 1, tzinfo=UTC))
    assert common.is_missing_run(run=datetime(2025, 11, 20, tzinfo=UTC))
    assert not common.is_missing_run(run=datetime(2025, 11, 21, tzinfo=UTC))
    assert not common.is_missing_run(run=datetime(2024, 3, 31, tzinfo=UTC))


def test_pinned_schema_holds_the_cloud_column():
    common.assert_cloud_in_schema()


def _members(*, values: list[float]) -> pl.DataFrame:
    n = len(values)
    origin = common.origin_of(run=RUN)
    return pl.DataFrame(
        {
            "time_series_id": [1] * n,
            "ensemble_member": list(range(n)),
            "nwp_init_time": [RUN] * n,
            "power_fcst_init_time": [origin] * n,
            "valid_time": [origin + timedelta(minutes=30)] * n,
            "cloud": values,
        }
    ).cast(dict.fromkeys(("nwp_init_time", "power_fcst_init_time", "valid_time"), DTYPE))


def test_cloud_stats_of_a_split_ensemble_separates_it_from_a_half_cloudy_one():
    split = cloud.cloud_stats(half_hourly=_members(values=[0.0] * 25 + [100.0] * 25))
    half = cloud.cloud_stats(half_hourly=_members(values=[50.0] * 50))
    assert split["cloud_mean"].item() == pytest.approx(0.5)
    assert half["cloud_mean"].item() == pytest.approx(0.5)
    assert split["cloud_spread"].item() == pytest.approx(0.5)  # population standard deviation
    assert half["cloud_spread"].item() == pytest.approx(0.0)
    assert split["cloud_clear_share"].item() == pytest.approx(0.5)
    assert split["cloud_overcast_share"].item() == pytest.approx(0.5)
    assert half["cloud_clear_share"].item() == 0.0
    assert half["cloud_overcast_share"].item() == 0.0


def test_cloud_stats_thresholds_are_inclusive_at_20_and_80_percent():
    stats = cloud.cloud_stats(half_hourly=_members(values=[20.0, 80.0, 20.1, 79.9]))
    assert stats["cloud_clear_share"].item() == pytest.approx(0.25)
    assert stats["cloud_overcast_share"].item() == pytest.approx(0.25)
    assert stats["n_members"].item() == 4


def _cache_table(*, n_series: int = 2, hours: float = 351.0) -> pl.DataFrame:
    origin = common.origin_of(run=RUN)
    steps = cloud.VALID_TIMES_PER_RUN
    rows = []
    for series in range(1, n_series + 1):
        for i in range(1, steps + 1):
            rows.append(
                {
                    "time_series_id": series,
                    "nwp_init_time": RUN,
                    "power_fcst_init_time": origin,
                    "valid_time": origin + timedelta(minutes=30 * i),
                    "cloud_mean": 0.5,
                    "cloud_spread": 0.1,
                    "cloud_clear_share": 0.2,
                    "cloud_overcast_share": 0.3,
                }
            )
    return pl.DataFrame(rows).cast(
        dict.fromkeys(("nwp_init_time", "power_fcst_init_time", "valid_time"), DTYPE)
    )


def test_cache_table_passes_and_is_float32():
    out = cloud.validate_cache_table(table=_cache_table(), run=RUN, n_series=2)
    assert out.schema["cloud_mean"] == pl.Float32


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (pl.col("cloud_mean") + 0.6, "outside its range"),
        (pl.lit(None, dtype=pl.Float64), "null or not finite"),
    ],
)
def test_cache_table_rejects_bad_values(change, message):
    table = _cache_table().with_columns(cloud_mean=change)
    with pytest.raises(ValueError, match=message):
        cloud.validate_cache_table(table=table, run=RUN, n_series=2)


def test_cache_table_rejects_overlapping_shares_a_missing_series_and_a_short_run():
    table = _cache_table().with_columns(cloud_clear_share=pl.lit(0.8))
    with pytest.raises(ValueError, match="outside its range"):
        cloud.validate_cache_table(table=table, run=RUN, n_series=2)
    with pytest.raises(ValueError, match="height other than"):
        cloud.validate_cache_table(table=_cache_table(), run=RUN, n_series=3)
    short = _cache_table().filter(
        pl.col("valid_time") < pl.col("power_fcst_init_time") + timedelta(hours=300)
    )
    with pytest.raises(ValueError, match="longest lead"):
        cloud.validate_cache_table(table=short, run=RUN, n_series=2)


def test_run_table_rejects_a_valid_time_with_fewer_than_51_members_or_a_null():
    table = _cache_table().with_columns(n_members=pl.lit(51), n_null=pl.lit(0))
    cloud.validate_run_table(table=table, run=RUN, n_series=2)
    with pytest.raises(ValueError, match="fewer than 51"):
        cloud.validate_run_table(
            table=table.with_columns(n_members=pl.lit(50)), run=RUN, n_series=2
        )
    with pytest.raises(ValueError, match="null cloud value"):
        cloud.validate_run_table(table=table.with_columns(n_null=pl.lit(1)), run=RUN, n_series=2)


def _stats(*, mutate=None) -> pl.DataFrame:
    """Per-run table statistics that satisfy the plan's conditions."""
    runs = [
        r
        for r in common.runs_between(
            first=common.CACHE_FIRST_RUN, last=common.TABLE_REQUIRED_LAST_RUN
        )
        if r != common.UPSTREAM_GAP_RUN
    ]
    recent_runs = [r for r in runs if common.RECENT_FIRST_RUN <= r <= common.SCORED_LAST_RUN]
    per_recent = common.EXPECTED_RECENT_ROWS_PER_RUN
    rows: list[dict[str, Any]] = []
    for r in runs:
        in_gap = common.GAP_FIRST_RUN <= r <= common.GAP_LAST_RUN
        n_rows = per_recent if r in recent_runs else 1000
        rows.append(
            {
                "init_time": r,
                "n_rows": n_rows,
                "n_members": 51,
                "n_null": n_rows if in_gap else 0,
                "cloud_min": None if in_gap else 0.0,
                "cloud_max": None if in_gap else 100.0,
            }
        )
    frame = pl.DataFrame(rows, schema_overrides={"cloud_min": pl.Float64, "cloud_max": pl.Float64})
    frame = frame.cast({"init_time": DTYPE})
    return mutate(frame) if mutate else frame


def test_presence_conditions_pass_on_the_expected_table():
    assert cloud.presence_problems(stats=_stats()) == []


@pytest.mark.parametrize(
    ("mutate", "fragment"),
    [
        (lambda f: f.filter(pl.col("init_time") != datetime(2026, 3, 1, tzinfo=UTC)), "absent"),
        (
            lambda f: f.with_columns(
                n_members=pl.when(pl.col("init_time") == datetime(2026, 3, 1, tzinfo=UTC))
                .then(50)
                .otherwise(51)
            ),
            "50 members",
        ),
        (
            lambda f: f.with_columns(
                n_null=pl.when(pl.col("init_time") == datetime(2026, 3, 1, tzinfo=UTC))
                .then(3)
                .otherwise(pl.col("n_null"))
            ),
            "null cloud values outside the gap",
        ),
        (
            lambda f: f.with_columns(
                cloud_max=pl.when(pl.col("init_time") == datetime(2023, 1, 1, tzinfo=UTC))
                .then(101.0)
                .otherwise(pl.col("cloud_max"))
            ),
            "above 100",
        ),
        (
            lambda f: f.with_columns(
                n_rows=pl.when(pl.col("init_time") == datetime(2026, 3, 1, tzinfo=UTC))
                .then(pl.col("n_rows") + 1)
                .otherwise(pl.col("n_rows"))
            ),
            "do not hold 7,243,785",
        ),
    ],
)
def test_presence_conditions_report_each_violation(mutate, fragment):
    problems = cloud.presence_problems(stats=_stats(mutate=mutate))
    assert any(fragment in p for p in problems), problems


def test_presence_conditions_check_the_optional_run_of_2026_11_19_for_nulls():
    def add_run(frame: pl.DataFrame) -> pl.DataFrame:
        extra = frame.tail(1).with_columns(
            init_time=pl.lit(common.TABLE_OPTIONAL_RUN).cast(DTYPE),
            n_null=pl.lit(2, dtype=pl.Int64),
        )
        return pl.concat([frame, extra])

    problems = cloud.presence_problems(stats=_stats(mutate=add_run))
    assert any("2026-11-19" in p and "outside the gap" in p for p in problems), problems


def test_presence_conditions_reject_a_null_the_day_after_the_gap():
    def move_null(frame: pl.DataFrame) -> pl.DataFrame:
        return frame.with_columns(
            n_null=pl.when(pl.col("init_time") == common.RECENT_FIRST_RUN)
            .then(5)
            .otherwise(pl.col("n_null"))
        )

    problems = cloud.presence_problems(stats=_stats(mutate=move_null))
    assert any("2025-11-21" in p and "outside the gap" in p for p in problems), problems


def test_presence_conditions_report_cloud_stored_as_a_fraction_and_nan():
    def fraction(frame: pl.DataFrame) -> pl.DataFrame:
        return frame.with_columns(
            cloud_max=pl.when(pl.col("init_time") == datetime(2023, 1, 1, tzinfo=UTC))
            .then(1.0)
            .otherwise(pl.col("cloud_max")),
            n_nan=pl.when(pl.col("init_time") == datetime(2026, 3, 1, tzinfo=UTC))
            .then(4)
            .otherwise(0),
        )

    problems = cloud.presence_problems(stats=_stats(mutate=fraction))
    assert any("2023-01-01" in p and "fraction" in p for p in problems), problems
    assert any("2026-03-01" in p and "NaN" in p for p in problems), problems


def test_run_table_rejects_cloud_that_looks_like_a_fraction():
    table = _cache_table().with_columns(
        cloud_mean=pl.col("cloud_mean") * 0.01, n_members=pl.lit(51), n_null=pl.lit(0)
    )
    with pytest.raises(ValueError, match="fraction"):
        cloud.validate_run_table(table=table, run=RUN, n_series=2)


def test_cache_table_rejects_duplicate_keys_with_the_right_height():
    table = _cache_table()
    duplicated = pl.concat([table.head(table.height - 1), table.head(1)])
    with pytest.raises(ValueError, match="duplicate keys"):
        cloud.validate_cache_table(table=duplicated, run=RUN, n_series=2)


def test_stuck_series_and_identical_runs_are_reported():
    table = _cache_table()
    assert len(cloud.stuck_problems(table=table, run=RUN)) == 2  # constant fixture: both stuck
    varied = table.with_columns(cloud_mean=pl.int_range(pl.len()).cast(pl.Float64) / 10_000.0)
    assert cloud.stuck_problems(table=varied, run=RUN) == []
    same = cloud.content_hash(table=varied)
    other = cloud.content_hash(table=varied.with_columns(cloud_mean=pl.col("cloud_mean") + 0.1))
    assert same == cloud.content_hash(table=varied.reverse())  # the hash is sorted by key
    assert cloud.duplicate_content_problems(hashes={"a": same, "b": other}) == []
    assert len(cloud.duplicate_content_problems(hashes={"a": same, "b": same})) == 1


def test_checks_gate_the_build(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "CHECK_DIR", tmp_path)
    assert len(common.checks_pass_at(version=7)) == 3  # no record yet
    for name in common.CHECK_NAMES:
        common.record_check(name=name, version=7, passed=True, details={})
    assert common.checks_pass_at(version=7) == []
    assert len(common.checks_pass_at(version=8)) == 3  # a pass at another version does not count
    common.record_check(name="era_check", version=7, passed=False, details={})
    assert common.checks_pass_at(version=7) == ["era_check: not a pass at Delta version 7"]


def test_build_refuses_without_recorded_passes_and_with_stale_files(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "CHECK_DIR", tmp_path / "checks")
    monkeypatch.setattr(common, "CLOUD_DIR", tmp_path / "ens_cloud")
    monkeypatch.setattr(common, "LINEAGE_PATH", tmp_path / "ens_cloud" / "lineage.json")
    monkeypatch.setattr(common, "require_one_polars_thread", lambda: None)
    common.atomic_write_text(text='{"delta_version": 7}', target=common.LINEAGE_PATH)
    with pytest.raises(SystemExit, match="build refused"):
        cloud.build(workers=1, limit=1, last_run=common.SCORED_LAST_RUN)
    with pytest.raises(SystemExit, match="2024-03-31 only"):
        cloud.build(workers=1, limit=1, last_run=common.SCORED_LAST_RUN, back_filled_only=True)
    for name in common.CHECK_NAMES:
        common.record_check(name=name, version=7, passed=True, details={})
    old = _cache_table().with_columns(delta_version=pl.lit(6, dtype=pl.Int32))
    old.write_parquet(common.CLOUD_DIR / "init=2026-01-14.parquet")
    with pytest.raises(SystemExit, match="another Delta version"):
        cloud.build(workers=1, limit=1, last_run=common.SCORED_LAST_RUN)
    assert cloud.stale_cache_files(version=7) == [common.CLOUD_DIR / "init=2026-01-14.parquet"]
    with pytest.raises(SystemExit, match="--drop-stale"):
        cloud.pin(version=7, force=True, drop_stale=False)
    assert (common.CLOUD_DIR / "init=2026-01-14.parquet").exists()


def test_one_polars_thread_is_enforced(monkeypatch):
    monkeypatch.setattr(pl, "thread_pool_size", lambda: 4)
    with pytest.raises(RuntimeError, match="POLARS_MAX_THREADS=1"):
        common.require_one_polars_thread()
    monkeypatch.setattr(pl, "thread_pool_size", lambda: 1)
    common.require_one_polars_thread()


def test_cross_check_fails_on_zero_overlap_and_low_correlation(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "CHECK_DIR", tmp_path / "checks")
    monkeypatch.setattr(common, "CLOUD_DIR", tmp_path / "ens_cloud")
    monkeypatch.setattr(cloud, "EXPERIMENT_561_DIR", tmp_path / "e561")
    (tmp_path / "e561").mkdir()
    recent = {"init=2026-01-14.parquet"}
    with pytest.raises(ValueError, match="none overlaps"):
        cloud.cross_check_561(present=recent)
    cloud.cross_check_561(present=set())  # no recent run cached: not applicable
    mine = _cache_table(n_series=1).with_columns(
        cloud_mean=pl.int_range(pl.len()).cast(pl.Float64) / 1000.0
    )
    common.CLOUD_DIR.mkdir()
    mine.write_parquet(common.CLOUD_DIR / "init=2026-01-14.parquet")
    theirs = mine.select(
        *common.KEYS,
        mean_total_cloud_cover=pl.lit(5.0) * pl.col("cloud_spread"),
        std_total_cloud_cover=pl.col("cloud_spread") * 100.0,
    )
    theirs = theirs.with_columns(
        mean_total_cloud_cover=(
            pl.col("valid_time").dt.minute() + pl.col("valid_time").dt.hour() * 7.0
        )
    )
    theirs.write_parquet(tmp_path / "e561" / "init=2026-01-14.parquet")
    with pytest.raises(ValueError, match="below"):
        cloud.cross_check_561(present=recent)


def _monthly(*, recent_shift: float = 0.0, resolution_shift: float = 0.0) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    month = date(2021, 8, 1)
    while month <= date(2026, 3, 1):
        for series in ("D1", "farm A"):
            seasonal = 3.0 * np.cos(2 * np.pi * (month.month - 1) / 12)
            noise = rng.normal(0, 0.5)
            shift = 0.0
            if month >= date(2025, 12, 1):
                shift += recent_shift
            if date(2023, 7, 1) <= month <= date(2024, 3, 1):
                shift += resolution_shift
            rows.append(
                {
                    "series": series,
                    "month": month,
                    "diff": seasonal + noise + shift,
                    "spread": 20.0 + noise + shift,
                }
            )
        month = date(month.year + (month.month == 12), month.month % 12 + 1, 1)
    return pl.DataFrame(rows)


def test_step_tests_pass_when_the_eras_agree():
    results = cloud.step_failures(monthly=_monthly())
    assert results.height == 2 * 2 * 2  # series x quantity x test
    assert not results["failed"].any()


def test_step_test_fails_on_a_level_shift_in_the_recent_winter():
    results = cloud.step_failures(monthly=_monthly(recent_shift=10.0))
    failed = results.filter(pl.col("failed"))
    assert set(failed["test"]) == {"winters"}
    assert set(failed["quantity"]) == {"diff", "spread"}


def test_step_test_fails_on_a_level_shift_at_the_resolution_change():
    results = cloud.step_failures(monthly=_monthly(resolution_shift=10.0))
    assert "resolution_change" in set(results.filter(pl.col("failed"))["test"])


def test_deseasonalised_scale_catches_a_step_as_large_as_the_seasonal_cycle():
    monthly = _monthly(recent_shift=3.5)  # seasonal amplitude is 3, noise sd 0.5
    results = cloud.step_failures(monthly=monthly)
    winters = results.filter(pl.col("test") == "winters", pl.col("quantity") == "diff")
    assert winters["failed"].all()
    assert (winters["scale"] < winters["scale_literal"]).all()
    assert (winters["sd"] > winters["sd_literal"]).all()
    assert (winters["sd_se"] > winters["sd"]).all()
    resolution = cloud.step_failures(monthly=_monthly(resolution_shift=3.0))
    shifted = resolution.filter(pl.col("test") == "resolution_change", pl.col("quantity") == "diff")
    assert shifted["failed"].any()
    assert (shifted["sd"] > shifted["sd_literal"]).all()
