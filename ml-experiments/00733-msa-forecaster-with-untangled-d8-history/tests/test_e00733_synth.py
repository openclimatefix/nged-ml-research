# ruff: noqa: PT018
"""Tests of `e00733_synth`: the ensemble coverage result, the sample layout, visibility, and
leakage."""

from pathlib import Path

import e00733_channels as ch
import e00733_core as core
import e00733_synth as sy
import numpy as np
import pytest
from synth_fixture import make_segment


def _days(*dates: str) -> np.ndarray:
    return np.array(dates, dtype="datetime64[D]")


def test_nwp_run_days_reads_partition_names(tmp_path: Path) -> None:
    """The run days come from the `init_time=...T00%3A00` folder names; other folders are
    ignored."""
    folder = tmp_path / f"nwp_model_id={sy.ENS_MODEL_ID}"
    for name in (
        "init_time=2020-10-15T00%3A00%3A00.000000Z",
        "init_time=2020-10-16T00%3A00%3A00.000000Z",
        "init_time=2020-10-17T12%3A00%3A00.000000Z",
        "other",
    ):
        (folder / name).mkdir(parents=True)
    assert sy.nwp_run_days(nwp_root=tmp_path) == {
        np.datetime64("2020-10-15"),
        np.datetime64("2020-10-16"),
    }
    with pytest.raises(sy.SynthError):
        sy.nwp_run_days(nwp_root=tmp_path / "none")


def test_coverage_reports_gaps_as_data_and_never_raises() -> None:
    """A base date without a run is a `coverage_gap` status; segments with no covered date are left
    out."""
    runs = {np.datetime64("2020-10-15"), np.datetime64("2020-10-16")}
    res = sy.ens_coverage(
        base_dates=[
            _days("2020-10-15", "2020-10-16"),
            _days("2020-10-10", "2020-10-16"),
            _days("2020-01-01"),
        ],
        run_days=runs,
    )
    assert res.counts() == {
        "n_segments": 3,
        "n_left_out": 1,
        "n_fully_covered": 1,
        "n_base_dates": 5,
        "n_covered": 3,
        "n_gap_base_dates": 2,
    }
    assert res.statuses(segment=1) == [sy.STATUS_GAP, sy.STATUS_OK]
    assert sy.kept_segments(result=res) == [0, 1]
    assert res.left_out_share == pytest.approx(1 / 3)


def test_more_than_half_of_the_test_segments_left_out_stops_the_line() -> None:
    """Row 0 applies above one half; exactly half is allowed; an empty split also stops."""
    runs = {np.datetime64("2021-01-01")}
    ok = _days("2021-01-01")
    gap = _days("2020-01-01")
    sy.assert_enough_test_segments(result=sy.ens_coverage(base_dates=[ok, gap], run_days=runs))
    with pytest.raises(ch.StoppedError):
        sy.assert_enough_test_segments(
            result=sy.ens_coverage(base_dates=[ok, gap, gap], run_days=runs)
        )
    with pytest.raises(ch.StoppedError):
        sy.assert_enough_test_segments(result=sy.ens_coverage(base_dates=[], run_days=runs))


def test_history_and_target_slots_follow_the_layout() -> None:
    """History row age 2 starts at stamp c - 96; the target row of lead index 0 starts at the
    contract origin."""
    d = np.array([20])
    h = sy.history_slots(origin_day=d)
    c = 20 * 48 + 24
    assert h[0, 0, 0] == c - 96 - 1 and h[0, 13, 47] == c - 48 * 15 + 47 - 1
    t = sy.target_slots(origin_day=d, lead_index=np.array([0]))
    assert t[0, 0] == c - 1
    vis = sy.visible_slots(origin_day=d, slots=h)
    assert vis[0, 0, :43].all() and not vis[0, 0, 43:].any()
    assert vis[0, 1].all()


def test_samples_need_seven_history_rows_and_a_target_inside_the_segment() -> None:
    """A sample on day 5 has too few history rows.

    One near the end has a target row outside; a missing day drops rows.
    """
    seg = make_segment(n_days=40)
    idx = sy.enumerate_samples(segments=[seg])
    assert len(idx) > 0
    assert idx.origin_day.min() >= core.MIN_AGE + core.MIN_HISTORY_ROWS - 1
    rows = sy.history_row_exists(seg=seg, origin_day=idx.origin_day).sum(axis=1)
    assert (rows >= core.MIN_HISTORY_ROWS).all()
    t = sy.target_slots(origin_day=idx.origin_day, lead_index=idx.lead_index)
    assert t.max() < seg.n_days * 48 and t.min() >= 0
    gap = make_segment(n_days=40, missing_days=tuple(range(20, 30)))
    sample = sy.enumerate_samples(segments=[gap])
    t2 = sy.target_slots(origin_day=sample.origin_day, lead_index=sample.lead_index)
    assert gap.present[t2 // 48].all()
    assert not np.isin(sample.origin_day, range(20, 30)).any()


def test_an_origin_without_a_run_has_no_sample() -> None:
    """A base day whose run is missing (a coverage gap) is never an origin."""
    seg = make_segment(n_days=40)
    seg.covered[:] = False
    seg.covered[30] = True
    idx = sy.enumerate_samples(segments=[seg])
    assert set(idx.origin_day.tolist()) <= {30}


def test_gather_reads_nothing_after_the_visibility_limit_or_from_the_target_row() -> None:
    """Poisoning the reading, sign.

    And channels from the first invisible slot of an origin changes no history input.
    """
    seg = make_segment(n_days=40)
    idx = sy.enumerate_samples(segments=[seg])
    rows = np.flatnonzero((idx.origin_day == 25) & (idx.lead_index == 3))
    assert len(rows) == 1
    clean = sy.gather_batch(segments=[seg], index=idx, rows=rows, d8_index=0, target_std=1.0)
    limit = (
        25 * 48 + 24 - core.NEWEST_VISIBLE_BEFORE_CONTRACT
    )  # last visible stamp; its slot is limit - 1
    bad = make_segment(n_days=40)
    for name in ("reading_z", "x_neg", "solar_points", "u_ok"):
        a = getattr(seg, name)
        setattr(bad, name, a.copy())
    for name in ("reading_z", "x_neg", "solar_points"):
        getattr(bad, name)[limit:] = np.nan
    bad.u_raw = seg.u_raw.copy()
    bad.u_raw[limit:] = 1e6
    bad.hist_weather, bad.hist_datetime = seg.hist_weather, seg.hist_datetime
    bad.top_weather, bad.top_spread, bad.top_datetime = (
        seg.top_weather,
        seg.top_spread,
        seg.top_datetime,
    )
    bad.top_day_type, bad.day_type, bad.present, bad.covered = (
        seg.top_day_type,
        seg.day_type,
        seg.present,
        seg.covered,
    )
    got = sy.gather_batch(segments=[bad], index=idx, rows=rows, d8_index=0, target_std=1.0)
    for name in ("hist_power", "hist_visible", "hist_u", "hist_u_ok", "hist_weather", "row_exists"):
        assert np.array_equal(clean.arrays[name], got.arrays[name]), name
    assert not clean.arrays["hist_visible"][0, 0, 43:].any()
    assert np.isfinite(clean.arrays["hist_u"]).all()


def test_modes_absent_true_untangled() -> None:
    """Y0 reads no channel, YT the true sign and solar output, and YU the stored outputs."""
    seg = make_segment(n_days=40)
    idx = sy.enumerate_samples(segments=[seg])
    rows = np.arange(4)
    absent = sy.gather_batch(
        segments=[seg], index=idx, rows=rows, d8_index=0, target_std=2.0, mode="absent"
    )
    assert not absent.arrays["hist_u_ok"].any() and not absent.arrays["hist_u"].any()
    true = sy.gather_batch(
        segments=[seg], index=idx, rows=rows, d8_index=0, target_std=2.0, mode="true"
    )
    ok = true.arrays["hist_u_ok"]
    assert ok.any()
    assert set(np.unique(true.arrays["hist_u"][..., 0][ok]).tolist()) <= {0.0, 1.0}
    assert (true.arrays["hist_u"][..., 2] == 0).all()
    un = sy.gather_batch(segments=[seg], index=idx, rows=rows, d8_index=0, target_std=2.0)
    assert (un.arrays["hist_u"][..., 1] <= core.SOLAR_CLIP_MAX).all()
    assert (un.arrays["hist_u"][..., 0][un.arrays["hist_u_ok"]] > 0).any()
    with pytest.raises(sy.SynthError):
        sy.gather_batch(
            segments=[seg], index=idx, rows=rows, d8_index=0, target_std=2.0, mode="bad"
        )


def test_masks_hide_channels_but_never_the_reading() -> None:
    """The training masks switch off channel flags only."""
    seg = make_segment(n_days=40)
    idx = sy.enumerate_samples(segments=[seg])
    rows = np.arange(6)
    full = sy.gather_batch(segments=[seg], index=idx, rows=rows, d8_index=0, target_std=1.0)
    allm = np.ones(6, dtype=bool)
    masked = sy.gather_batch(
        segments=[seg], index=idx, rows=rows, d8_index=0, target_std=1.0, mask_history=(allm, allm)
    )
    assert not masked.arrays["hist_u_ok"][:, :4].any()
    assert (masked.arrays["hist_u"][~masked.arrays["hist_u_ok"]] == 0).all()
    assert np.array_equal(full.arrays["hist_power"], masked.arrays["hist_power"])


def test_forbidden_inputs_are_refused() -> None:
    """A world type, engine parameter, capacity, or truth array among the inputs fails the check."""
    sy.assert_no_world_leak(input_names=["hist_power", "hist_u", "top_weather"])
    with pytest.raises(sy.SynthError):
        sy.assert_no_world_leak(input_names=["hist_power", "world_type"])


def test_share_reverse_counts_scored_rows_only() -> None:
    """The share and count are over the scored lead days and finite signs."""
    seg = make_segment(n_days=40)
    idx = sy.enumerate_samples(segments=[seg])
    share, n = sy.share_reverse(index=idx, segments=[seg])
    assert 0.1 < share < 0.3 and n > 0


def test_summary_run_days_unions_two_folders(tmp_path: Path) -> None:
    """Run summaries in experiment 607's folder and 707's back-fill folder both count as covered;
    nothing is built."""
    a, b = tmp_path / "607", tmp_path / "707"
    a.mkdir()
    b.mkdir()
    (a / "init=2024-04-01.parquet").write_text("")
    (b / "init=2023-06-30.parquet").write_text("")
    (b / "README").write_text("")
    days = sy.summary_run_days(summary_dirs=[a, b, tmp_path / "missing"])
    assert days == {np.datetime64("2024-04-01"), np.datetime64("2023-06-30")}
    result = sy.ens_coverage(
        base_dates=[np.array(["2023-06-30", "2023-07-01"], dtype="datetime64[D]")], run_days=days
    )
    assert result.statuses(segment=0) == [sy.STATUS_OK, sy.STATUS_GAP]
    with pytest.raises(sy.SynthError):
        sy.summary_run_days(summary_dirs=[tmp_path / "missing"])


# --------------------------------------------------------------------------- origin-time weather


def test_every_synthetic_batch_asserts_the_origin_time_weather_and_prints_an_example() -> None:
    """A batch of covered origins passes, and its example row names the run read from the file."""
    seg = make_segment()
    times = sy.OriginTimes.empty(b=2)
    for n, d in enumerate((10, 11)):
        times.take(n=n, seg=seg, d=d, k=3)
    assert times.assert_origin_time_weather() == 2 * 48
    row = times.example()
    assert row is not None and row.endswith("pass") and "lead day 4" in row


@pytest.mark.parametrize(
    ("field", "offset_us"),
    [
        ("top_run_avail_us", 9 * 3_600_000_000 + 1_800_000_000),
        ("top_run_init_us", 1_800_000_000),
        ("top_other_valid_us", 9 * 3_600_000_000),
    ],
)
def test_a_synthetic_run_time_one_step_past_the_native_origin_fails(
    field: str, offset_us: int
) -> None:
    """A run available, initialised, or a column valid after the native origin fails the batch."""
    seg = make_segment()
    assert seg.top_run_init_us is not None
    getattr(seg, field)[10] = seg.top_run_init_us[10] + offset_us
    times = sy.OriginTimes.empty(b=1)
    times.take(n=0, seg=seg, d=10, k=0)
    with pytest.raises(sy.leak.CorrectnessFailureError):
        times.assert_origin_time_weather()


def test_a_segment_without_origin_time_record_cannot_be_gathered() -> None:
    """Missing run times raise instead of skipping the check."""
    seg = make_segment()
    seg.top_c_grid = None
    times = sy.OriginTimes.empty(b=1)
    with pytest.raises(sy.SynthError):
        times.take(n=0, seg=seg, d=10, k=0)
