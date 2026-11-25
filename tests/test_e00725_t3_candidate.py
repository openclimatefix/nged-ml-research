# ruff: noqa: E501
"""Tests of the `T3-C` arm of experiment 725 (synthetic data; no GPU and no stored data)."""

from datetime import UTC, datetime

import e00725_common as cm
import e00725_phases as ph
import e00725_t3_candidate as t3c
import numpy as np
import polars as pl
import pytest

DAY0 = 20_000
"""A day code (days since the epoch) for the first synthetic origin."""
N_DAYS = 8


def cams_by_day() -> dict[int, tuple[float, float]]:
    """Return a CAMS table with one distinct value per UTC day, except that day `DAY0 + 1` is missing."""
    return {
        DAY0 + k: (100.0 + k, 0.5 + 0.01 * k) for k in range(-4, N_DAYS) if DAY0 + k != DAY0 + 1
    }


def synthetic_source(*, cams: dict[int, tuple[float, float]]) -> pl.DataFrame:
    """Return a table of 3 rows per origin day whose CAMS columns hold the values of the day before the origin."""
    origin = np.repeat((DAY0 + np.arange(N_DAYS)) * t3c.US_PER_DAY + 9 * 3_600_000_000, 3)
    previous = t3c.day_columns(origin_us=origin, cams=cams, back_days=1)
    return pl.DataFrame(
        {
            "origin_us": origin,
            "valid_us": origin + 1_800_000_000 * np.tile([1, 2, 3], N_DAYS),
            "other": np.arange(len(origin), dtype=np.float32),
            t3c.CAMS_COLUMNS[0]: previous[:, 0].astype(np.float32),
            t3c.CAMS_COLUMNS[1]: previous[:, 1].astype(np.float32),
        }
    )


def test_day_columns_read_the_day_before_the_origin_minus_the_offset() -> None:
    """An origin on day d reads CAMS day d - back_days, and NaN where that day has no CAMS."""
    origin = np.array([DAY0 * t3c.US_PER_DAY + 1, (DAY0 + 4) * t3c.US_PER_DAY + 5])
    one = t3c.day_columns(origin_us=origin, cams=cams_by_day(), back_days=1)
    three = t3c.day_columns(origin_us=origin, cams=cams_by_day(), back_days=3)
    assert one[0].tolist() == [99.0, pytest.approx(0.49)]
    assert three[0].tolist() == [97.0, pytest.approx(0.47)]
    assert np.isnan(three[1]).all()


def test_patched_table_moves_only_the_scored_rows_and_only_the_two_columns() -> None:
    """Rows before the fold month and every other column are unchanged; later rows hold D - 3."""
    cams = cams_by_day()
    source = synthetic_source(cams=cams)
    start = (DAY0 + 5) * t3c.US_PER_DAY
    table = t3c.patched_table(source=source, cams=cams, start_us=start)
    origin = source["origin_us"].to_numpy()
    scored = origin >= start
    assert table.schema == source.schema
    assert table["other"].equals(source["other"])
    for name in t3c.CAMS_COLUMNS:
        assert np.array_equal(
            table[name].to_numpy()[~scored], source[name].to_numpy()[~scored], equal_nan=True
        )
    moved = t3c.day_columns(origin_us=origin, cams=cams, back_days=3)
    assert table[t3c.CAMS_COLUMNS[0]].to_numpy()[scored].tolist() == pytest.approx(
        moved[scored, 0].astype(np.float32).tolist()
    )
    assert not np.array_equal(
        table[t3c.CAMS_COLUMNS[0]].to_numpy()[scored],
        source[t3c.CAMS_COLUMNS[0]].to_numpy()[scored],
    )


def test_patched_table_refuses_a_stored_column_that_is_not_the_previous_day() -> None:
    """The reproduction check fails when the stored column is not CAMS day origin - 1."""
    cams = cams_by_day()
    source = synthetic_source(cams=cams).with_columns(pl.col(t3c.CAMS_COLUMNS[0]) + 1.0)
    with pytest.raises(AssertionError, match="differs from CAMS day origin - 1"):
        t3c.patched_table(source=source, cams=cams, start_us=0)


def test_month_start_and_output_path() -> None:
    """The fold month starts at 00:00 UTC on its first day, and the forecast sits in the Candidate's layout."""
    assert (
        t3c.month_start_us(month="2025-11")
        == int(datetime(2025, 11, 1, tzinfo=UTC).timestamp()) * 1_000_000
    )
    path = t3c.fit_file(month="2026-03", seed=1)
    assert path.parts[-5:] == ("P7_D50_2026-03", "work", "fits", "P5", "2026-03_d0_s1.parquet")
    assert cm.ROOT in path.parents


def test_phase_4_holds_the_candidate_jobs_for_every_fold_and_seed() -> None:
    """Phase 4 predicts `T3-C` on 8 folds and 2 seeds, one GPU job each, with the arm's forecast file as output."""
    jobs = [j for j in ph.phase_jobs(phase=4) if j.script == "e00725_t3_candidate.py"]
    assert len(jobs) == len(cm.ALL_FOLDS) * len(cm.TWO_SEEDS)
    assert all(j.gpu and j.kind == "predict" and j.args[0] == "T3-C" for j in jobs)
    assert {j.output for j in jobs} == {
        t3c.fit_file(month=m, seed=s) for m in cm.ALL_FOLDS for s in cm.TWO_SEEDS
    }
    assert len(ph.phase_jobs(phase=4)) == 3 * len(jobs)


def test_the_scorer_reads_the_candidate_arm_from_its_own_folder() -> None:
    """`T3-C` is a F8 arm whose file is the one that the script writes."""
    score = pytest.importorskip("e00725_score")
    assert "T3-C" in score.F8_ARMS
    assert score.arm_file(arm="T3-C", month="2025-11", seed=0) == t3c.fit_file(
        month="2025-11", seed=0
    )
    assert score.arm_file(arm="L", month="2025-11", seed=0).parts[-2] == "L"
