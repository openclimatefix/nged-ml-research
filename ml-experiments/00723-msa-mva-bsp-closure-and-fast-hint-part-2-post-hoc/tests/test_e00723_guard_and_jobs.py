# ruff: noqa: E501, ANN001
"""The guard, the run wrapper, the job order, and the pure helpers of job W0, on synthetic inputs."""

import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import e00723_common as common
import e00723_core as core
import e00723_guard as guard
import e00723_w0 as w0
import e00723_w3 as w3
import e00723_world as world
import numpy as np
import polars as pl
import pytest

HOME_PREFIX = "/home/" + "jack"
FOLDER = Path(__file__).resolve().parent.parent


def test_outputs_go_under_experiment_723s_own_folder():
    assert guard.OUT_DIR.name == "00723-msa-mva-bsp-closure-and-fast-hint-part-2-post-hoc"
    assert guard.PRIVATE_DIR == guard.OUT_DIR / "PRIVATE"
    assert guard.OUT_DIR_722.name.startswith("00722-")
    assert FOLDER.name == guard.EXPERIMENT_FOLDER


def test_the_public_writers_refuse_dates_and_generator_identifiers(tmp_path):
    with pytest.raises(guard.AnonymisationError):
        guard.write_public_csv(
            frame=pl.DataFrame({"day": ["2025-03-04"]}), name="x.csv", out_dir=tmp_path
        )
    with pytest.raises(guard.AnonymisationError):
        guard.write_public_csv(
            frame=pl.DataFrame({"generator_id": [1]}), name="x.csv", out_dir=tmp_path
        )
    with pytest.raises(guard.AnonymisationError):
        guard.write_public_json(payload={"note": "2026-01-02"}, name="x.json", out_dir=tmp_path)
    with pytest.raises(guard.AnonymisationError):
        guard.say("on 2025-03-04 the farm read 3 MW")
    guard.write_public_csv(
        frame=pl.DataFrame({"farm": ["F1"], "n": [3]}), name="ok.csv", out_dir=tmp_path
    )
    assert (tmp_path / "ok.csv").is_file()
    assert not list(tmp_path.glob("*.tmp"))


def test_the_code_hash_covers_this_experiments_scripts_only(tmp_path):
    (tmp_path / "e00723_a.py").write_text("a")
    (tmp_path / "e00722_b.py").write_text("b")
    h1 = guard.code_hash(folder=tmp_path)
    (tmp_path / "e00722_b.py").write_text("changed")
    assert guard.code_hash(folder=tmp_path) == h1
    (tmp_path / "e00723_a.py").write_text("changed")
    assert guard.code_hash(folder=tmp_path) != h1


def test_the_run_log_holds_an_input_fingerprint(tmp_path):
    table = tmp_path / "table.parquet"
    table.write_bytes(b"abc")
    gate = guard.TimingGate()
    with guard.run_log(
        job="t",
        gate=gate,
        inputs=[table, tmp_path / "missing"],
        info={"x": 1, "attempt": 1},
        out_dir=tmp_path,
    ):
        pass
    import json

    log = json.loads((tmp_path / "run_log_t.json").read_text())
    assert len(log["input_fingerprints"]) == 1
    assert log["completed"] is True
    assert log["inputs_missing"]
    assert log["n_boot"] == 1000


def test_a_crashed_job_still_writes_its_run_log_with_completed_false(tmp_path):
    gate = guard.TimingGate()
    with (
        pytest.raises(RuntimeError),
        guard.run_log(job="c", gate=gate, inputs=[], info={"attempt": 1}, out_dir=tmp_path),
    ):
        raise RuntimeError
    import json

    assert json.loads((tmp_path / "run_log_c.json").read_text())["completed"] is False


def test_job_w3_refuses_unless_w2_is_supported(tmp_path):
    with pytest.raises(guard.JobRefusedError):
        common.require_w2_supported(out_dir=tmp_path)
    (tmp_path / common.W2_VERDICT_NAME).write_text('{"verdict": "partly supported"}')
    with pytest.raises(guard.JobRefusedError):
        common.require_w2_supported(out_dir=tmp_path)
    (tmp_path / common.W2_VERDICT_NAME).write_text('{"verdict": "supported"}')
    assert common.require_w2_supported(out_dir=tmp_path)["verdict"] == "supported"


def test_a_job_refuses_without_the_check_file_of_job_w0(tmp_path):
    with pytest.raises(guard.JobRefusedError):
        common.require_w0(out_dir=tmp_path)


def test_a_job_refuses_to_run_off_the_launch_wrapper(monkeypatch):
    for name in common.ENV_DEFAULTS:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(guard.JobRefusedError):
        common.assert_launch()


def test_the_run_wrapper_carries_every_slot_rule():
    text = (FOLDER / "run.sh").read_text()
    for needle in (
        '"--go"', "SLOT_CPU", "MemoryMax=12G", "MemorySwapMax=0", "OOMPolicy=continue",
        "OMP_NUM_THREADS=1", "POLARS_MAX_THREADS=1", 'CUDA_VISIBLE_DEVICES=""', ".venv/bin/python",
        "systemd-run --user --scope", "taskset -c", "-lt 25", "00722-msa-mva-bsp-closure-and-fast-hint",
    ):  # fmt: skip
        assert needle in text
    assert HOME_PREFIX not in text
    assert text.startswith("#!/usr/bin/env bash")


def test_no_script_names_a_home_path_or_a_dated_value():
    for path in FOLDER.glob("*.py"):
        text = path.read_text()
        assert HOME_PREFIX not in text, path.name
    for path in (FOLDER / "tests").glob("*.py"):
        assert HOME_PREFIX not in path.read_text(), path.name


def test_a_rerun_moves_the_earlier_outputs_into_an_attempt_folder(tmp_path):
    (tmp_path / "PRIVATE").mkdir()
    (tmp_path / "a.csv").write_text("1")
    (tmp_path / "PRIVATE" / "w.parquet").write_text("p")
    first = guard.archive_outputs(
        names=("a.csv", "b.csv"), private_names=("w.parquet",), out_dir=tmp_path
    )
    assert first == tmp_path / "attempt_1"
    assert first is not None
    assert (first / "a.csv").read_text() == "1"
    assert (first / "PRIVATE" / "w.parquet").read_text() == "p"
    assert not (tmp_path / "a.csv").exists()
    # Nothing to move: no folder is made.
    assert guard.archive_outputs(names=("a.csv",), out_dir=tmp_path) is None
    (tmp_path / "a.csv").write_text("2")
    second = guard.archive_outputs(names=("a.csv",), out_dir=tmp_path)
    assert second == tmp_path / "attempt_2"
    assert second is not None
    assert (first / "a.csv").read_text() == "1"
    assert (second / "a.csv").read_text() == "2"


def test_the_first_attempt_is_counted_and_a_rerun_after_any_end_is_refused_without_approval(
    tmp_path,
):
    (tmp_path / "PRIVATE").mkdir()
    names = ("a.csv", "run_log_w0.json")
    assert guard.begin_attempt(job="w0", names=names, out_dir=tmp_path, env={}) == 1
    marker = tmp_path / guard.attempt_marker_name(job="w0")
    assert json.loads(marker.read_text()) == {"approval": "", "attempt": 1, "job": "w0"}
    # The job crashed before it wrote any output: the marker alone refuses the next run.
    with pytest.raises(guard.JobRefusedError):
        guard.begin_attempt(job="w0", names=names, out_dir=tmp_path, env={})
    # Another job's marker is separate.
    assert guard.begin_attempt(job="w2", names=names, out_dir=tmp_path, env={}) == 1


@pytest.mark.parametrize("verdict", ["supported", "inconclusive", "not supported", "invalid"])
def test_a_verdict_of_any_kind_including_invalid_is_refused_a_rerun_without_approval(
    tmp_path, verdict
):
    names = ("v.json", "run_log_w2.json")
    guard.begin_attempt(job="w2", names=names, out_dir=tmp_path, env={})
    (tmp_path / "v.json").write_text(json.dumps({"verdict": verdict}))
    with pytest.raises(guard.JobRefusedError):
        guard.begin_attempt(job="w2", names=names, out_dir=tmp_path, env={})
    approved = {guard.RERUN_APPROVAL_ENV: "w2:2"}
    assert guard.begin_attempt(job="w2", names=names, out_dir=tmp_path, env=approved) == 2
    # The earlier outputs and the old marker moved into the job's folder for attempt 1.
    assert (tmp_path / "attempt_w2_1" / "v.json").is_file()
    assert (tmp_path / "attempt_w2_1" / guard.attempt_marker_name(job="w2")).is_file()
    assert not (tmp_path / "v.json").exists()
    marker = json.loads((tmp_path / guard.attempt_marker_name(job="w2")).read_text())
    assert marker["attempt"] == 2
    assert marker["approval"] == "w2:2"
    # The same approval does not permit the third attempt, and a wrong job or number refuses.
    for value in ("w2:2", "w2:3x", "w0:3", "approved by the manager", "3"):
        with pytest.raises(guard.JobRefusedError):
            guard.begin_attempt(
                job="w2", names=names, out_dir=tmp_path, env={guard.RERUN_APPROVAL_ENV: value}
            )
    assert (
        guard.begin_attempt(
            job="w2", names=names, out_dir=tmp_path, env={guard.RERUN_APPROVAL_ENV: "w2:3"}
        )
        == 3
    )


def test_attempt_folders_are_numbered_per_job(tmp_path):
    names = ("o.csv",)
    for job in ("w0", "w2"):
        guard.begin_attempt(job=job, names=names, out_dir=tmp_path, env={})
        (tmp_path / "o.csv").write_text(job)
        guard.begin_attempt(
            job=job, names=names, out_dir=tmp_path, env={guard.RERUN_APPROVAL_ENV: f"{job}:2"}
        )
    assert (tmp_path / "attempt_w0_1" / "o.csv").read_text() == "w0"
    assert (tmp_path / "attempt_w2_1" / "o.csv").read_text() == "w2"


def test_a_crash_then_an_unapproved_retry_leaves_the_crash_log_untouched(tmp_path):
    gate = guard.TimingGate()
    names = ("run_log_w0.json",)

    def attempt(*, env: dict[str, str]) -> None:
        info: dict = {}
        inputs: list = []
        with guard.run_log(job="w0", gate=gate, inputs=inputs, info=info, out_dir=tmp_path):
            info["attempt"] = guard.begin_attempt(job="w0", names=names, out_dir=tmp_path, env=env)
            info["read"] = "something"
            raise RuntimeError("crash")

    with pytest.raises(RuntimeError, match="crash"):
        attempt(env={})
    log = tmp_path / "run_log_w0.json"
    before = log.read_bytes()
    assert json.loads(before)["attempt"] == 1
    assert json.loads(before)["completed"] is False
    with pytest.raises(guard.JobRefusedError):
        attempt(env={})
    assert log.read_bytes() == before  # the refusal did not overwrite the crash log
    refused = list(tmp_path.glob("run_log_w0_refused_*.json"))
    assert len(refused) == 1
    assert json.loads(refused[0].read_text())["refused"] is True
    # A start that fails the launch check (no attempt began) is logged the same way.
    with (
        pytest.raises(guard.JobRefusedError),
        guard.run_log(job="w0", gate=gate, inputs=[], info={}, out_dir=tmp_path),
    ):
        raise guard.JobRefusedError("launch")
    assert log.read_bytes() == before


def test_the_known_answer_tests_are_run_and_read_by_row_1(tmp_path):
    (tmp_path / "test_ok.py").write_text("def test_a():\n    assert 1 + 1 == 2\n")
    ok = guard.known_answer_tests(tests_dir=tmp_path)
    assert ok["passed"] is True
    assert ok["n_tests"] == 1
    assert len(ok["code_hash"]) == 64
    (tmp_path / "test_bad.py").write_text("def test_b():\n    assert False\n")
    bad = guard.known_answer_tests(tests_dir=tmp_path)
    assert bad["passed"] is False
    text = (FOLDER / "e00723_w2.py").read_text()
    assert 'tests["passed"]' in text  # job W2's correctness check reads the result


def test_job_w2_refuses_a_world_run_after_a_failed_sensor_audit_and_writes_both_counts():
    text = (FOLDER / "e00723_w2.py").read_text()
    assert 'arc_void = bool(w0["arc_term_void"])' in text
    assert "arc_term_void_flag=arc_void" in text
    for key in ("worlds_scored_equal_planned", "worlds_planned", "worlds_skipped"):
        assert key in text
    assert "void_by_construction" in text


def test_the_verdict_file_is_written_before_any_read_out_runs():
    text = (FOLDER / "e00723_w2.py").read_text()
    run = text[text.index("def run(") : text.index("def readouts(")]
    assert run.index("name=common.W2_VERDICT_NAME") < run.index("    readouts(")
    text3 = (FOLDER / "e00723_w3.py").read_text()
    assert text3.index("name=common.W3_VERDICT_NAME") < text3.index("# ---- Read-outs")
    # The two D15 fits run last in job W2's read-outs.
    closure = text[text.index("def closure_readouts(") :]
    assert closure.index("d15_fit_a_lit") > closure.index("d10_fold_ratio_import_hours")


def test_the_scripts_are_named_with_the_experiment_id():
    names = sorted(p.name for p in FOLDER.glob("*.py"))
    assert names
    assert all(re.fullmatch(r"e00723_[a-z0-9]+\.py", n) for n in names)


def test_sheet_summary_and_windows_from_sheets():
    sheet = pl.DataFrame(
        {
            "Substation Number": ["10", "10", "11", "10"],
            "Substation Name": ["a", "a", "b", "a"],
            "Start Time": [
                datetime(2025, 3, 1, 8, 0),
                datetime(2025, 4, 1, 9, 0),
                datetime(2025, 5, 1),
                None,
            ],
            "End Time": [
                datetime(2025, 3, 1, 12, 0),
                None,
                datetime(2025, 5, 2),
                datetime(2025, 6, 1),
            ],
        }
    )
    summary = w0.sheet_summary(sheets={"Sheet1": sheet})
    assert summary["n_rows"].to_list() == [4]
    assert "Start Time" in summary["columns"][0]
    wins = w0.windows_from_sheets(sheets={"Sheet1": sheet}, numbers={"D1": 10, "D2": 11, "D3": 99})
    assert len(wins["D1"]) == 2  # the row with no start is skipped
    assert wins["D3"] == []
    first, last = wins["D1"][0]
    assert last - first == 8  # 4 hours
    day_first, day_last = wins["D1"][1]
    assert day_last - day_first == 48  # no end time: the calendar day
    assert (
        world.utc_window_rows(start=datetime(2025, 3, 1, 8, tzinfo=UTC), end=None)[1]
        - world.row_of_instant(instant=datetime(2025, 3, 1, tzinfo=UTC))
        == 48
    )


def sheet_of(
    *, numbers, starts, ends=None, number_name="Substation Number", start_name="Start Time"
):
    cols = {number_name: numbers, "Substation Name": ["x"] * len(numbers), start_name: starts}
    if ends is not None:
        cols["End Time"] = ends
    return pl.DataFrame(cols)


def test_a_substation_number_stored_as_a_float_or_with_a_null_still_matches():
    start = [datetime(2025, 3, 1, 8, 0)] * 3
    for numbers in ([10.0, 11.0, None], ["10.0", "11", None], [10, 11, None]):
        sheet = sheet_of(numbers=numbers, starts=start)
        wins = w0.windows_from_sheets(sheets={"s": sheet}, numbers={"D1": 10, "D2": 11})
        assert len(wins["D1"]) == 1
        assert len(wins["D2"]) == 1


def test_a_switching_time_with_no_zone_is_read_as_europe_london():
    # 08:00 local on a summer day is 07:00 UTC; on a winter day it is 08:00 UTC.
    summer = sheet_of(
        numbers=[10], starts=[datetime(2025, 7, 1, 8, 0)], ends=[datetime(2025, 7, 1, 9, 0)]
    )
    winter = sheet_of(
        numbers=[10], starts=[datetime(2025, 1, 6, 8, 0)], ends=[datetime(2025, 1, 6, 9, 0)]
    )
    aware = sheet_of(
        numbers=[10],
        starts=[datetime(2025, 7, 1, 8, 0, tzinfo=UTC)],
        ends=[datetime(2025, 7, 1, 9, 0, tzinfo=UTC)],
    )
    aware = aware.with_columns(pl.col("Start Time", "End Time").dt.convert_time_zone("UTC"))
    numbers = {"D1": 10}
    s_row = w0.windows_from_sheets(sheets={"s": summer}, numbers=numbers)["D1"][0][0]
    w_row = w0.windows_from_sheets(sheets={"s": winter}, numbers=numbers)["D1"][0][0]
    a_row = w0.windows_from_sheets(sheets={"s": aware}, numbers=numbers)["D1"][0][0]
    midnight_s = world.row_of_instant(instant=datetime(2025, 7, 1, tzinfo=UTC))
    midnight_w = world.row_of_instant(instant=datetime(2025, 1, 6, tzinfo=UTC))
    assert s_row - midnight_s == 14  # 07:00 UTC
    assert w_row - midnight_w == 16  # 08:00 UTC
    assert a_row - midnight_s == 16  # an aware time is kept as UTC


def test_text_times_in_mixed_formats_are_all_parsed_and_the_unparsed_are_counted():
    sheet = sheet_of(
        numbers=[10, 10, 10, 10],
        starts=["2025-03-01 08:00:00", "01/04/2025 09:30", "2025-05-01", "not a time"],
    )
    wins, diag = w0.windows_with_diagnostics(sheets={"s": sheet}, numbers={"D1": 10})
    assert len(wins["D1"]) == 3
    assert diag[0]["n_rows"] == 4
    assert diag[0]["n_rows_start_parsed"] == 3
    assert diag[0]["n_rows_start_not_parsed"] == 1
    assert diag[0]["n_rows_matched"] == 3


def test_the_column_search_uses_whole_words_and_skips_transfer_columns():
    sheet = pl.DataFrame(
        {
            "Substation Number": [10],
            "Transferred from": [datetime(2025, 1, 1)],
            "Start Time": [datetime(2025, 3, 1, 8)],
            "Attended": [datetime(2025, 2, 1)],
            "Restored Time": [datetime(2025, 3, 1, 12)],
        }
    )
    _, diag = w0.windows_with_diagnostics(sheets={"s": sheet}, numbers={"D1": 10})
    assert diag[0]["start_column"] == "Start Time"
    assert diag[0]["end_column"] == "Restored Time"


def _utc_row(*, year, month, day, hour=0) -> int:
    """Return the grid row of `hour` hours after 00:00 UTC of the day (hour may be negative or 24)."""
    return world.row_of_instant(
        instant=datetime(year, month, day, tzinfo=UTC) + timedelta(hours=hour)
    )


def test_a_date_only_start_with_no_end_is_that_local_calendar_day_in_summer_and_winter():
    # Summer: local midnight is 23:00 UTC the day before, so the window is 23:00 to 23:00 UTC.
    summer = sheet_of(numbers=[10], starts=[datetime(2025, 6, 10)])
    first, last = w0.windows_from_sheets(sheets={"s": summer}, numbers={"D1": 10})["D1"][0]
    assert first == _utc_row(year=2025, month=6, day=9, hour=23)
    assert last == _utc_row(year=2025, month=6, day=10, hour=23)
    # Winter: local midnight is 00:00 UTC.
    winter = sheet_of(numbers=[10], starts=[datetime(2025, 1, 15)])
    first, last = w0.windows_from_sheets(sheets={"s": winter}, numbers={"D1": 10})["D1"][0]
    assert first == _utc_row(year=2025, month=1, day=15)
    assert last == _utc_row(year=2025, month=1, day=16)
    # The same day as a text date and as a Date dtype.
    for starts in (["2025-06-10"], ["10/06/2025"], [datetime(2025, 6, 10).date()]):
        sheet = sheet_of(numbers=[10], starts=starts)
        got = w0.windows_from_sheets(sheets={"s": sheet}, numbers={"D1": 10})["D1"][0]
        assert got == (
            _utc_row(year=2025, month=6, day=9, hour=23),
            _utc_row(year=2025, month=6, day=10, hour=23),
        )


@pytest.mark.parametrize(
    ("year", "month", "day", "offset_hours"), [(2025, 7, 1, 1), (2025, 1, 6, 0)]
)
def test_a_date_only_end_is_local_24_00_so_a_same_day_start_never_ends_before_it_starts(
    year, month, day, offset_hours
):
    # A timed start at 09:00 local with a date-only end on the same day, and a date-only start
    # with a date-only end on the same day, in summer (offset 1) and in winter (offset 0).
    sheet = sheet_of(
        numbers=[10, 10, 10],
        starts=[
            datetime(year, month, day, 9, 0),
            datetime(year, month, day),
            datetime(year, month, day, 9, 0),
        ],
        ends=[
            datetime(year, month, day),
            datetime(year, month, day),
            datetime(year, month, day, 17, 0),
        ],
    )
    # `sheet` mixes a midnight end and a 17:00 end in one column, and `only_dates` holds midnights
    # alone; the per-row rule reads both the same way.
    only_dates = sheet_of(
        numbers=[10, 10],
        starts=[datetime(year, month, day, 9, 0), datetime(year, month, day, 9, 0)],
        ends=[datetime(year, month, day), datetime(year, month, day)],
    )
    wins, diag = w0.windows_with_diagnostics(sheets={"s": only_dates}, numbers={"D1": 10})
    assert diag[0]["n_rows_end_before_start"] == 0
    assert len(wins["D1"]) == 2
    first, last = wins["D1"][0]
    assert first == _utc_row(year=year, month=month, day=day, hour=9 - offset_hours)
    # Local 24:00 of the day is 23:00 UTC in summer and 00:00 UTC of the next day in winter.
    assert last == _utc_row(year=year, month=month, day=day, hour=24 - offset_hours)
    # A date-only start with a date-only end on the same day covers the whole local day.
    both = sheet_of(
        numbers=[10],
        starts=[datetime(year, month, day)],
        ends=[datetime(year, month, day)],
    )
    first, last = w0.windows_from_sheets(sheets={"s": both}, numbers={"D1": 10})["D1"][0]
    assert first == _utc_row(year=year, month=month, day=day, hour=0) - offset_hours * 2
    assert last == _utc_row(year=year, month=month, day=day, hour=24) - offset_hours * 2
    # A midnight end on a row whose end is not after a same-day start is local 24:00, whatever the
    # other cells of the column hold.
    _, diag_mixed = w0.windows_with_diagnostics(sheets={"s": sheet}, numbers={"D1": 10})
    assert diag_mixed[0]["n_rows_end_before_start"] == 0


def test_a_date_only_end_in_text_and_a_timed_start_on_the_same_day_is_read_as_the_end_of_that_day():
    sheet = sheet_of(
        numbers=[10, 10],
        starts=["2025-07-01 09:00", "01/07/2025"],
        ends=["2025-07-01", "01/07/2025"],
    )
    wins, diag = w0.windows_with_diagnostics(sheets={"s": sheet}, numbers={"D1": 10})
    assert diag[0]["n_rows_end_before_start"] == 0
    assert wins["D1"][0] == (
        _utc_row(year=2025, month=7, day=1, hour=8),
        _utc_row(year=2025, month=7, day=1, hour=23),
    )
    assert wins["D1"][1] == (
        _utc_row(year=2025, month=6, day=30, hour=23),
        _utc_row(year=2025, month=7, day=1, hour=23),
    )


def test_a_date_only_end_before_the_start_day_is_still_counted():
    sheet = sheet_of(numbers=[10], starts=[datetime(2025, 7, 3)], ends=[datetime(2025, 7, 1)])
    wins, diag = w0.windows_with_diagnostics(sheets={"s": sheet}, numbers={"D1": 10})
    assert wins["D1"] == []
    assert diag[0]["n_rows_end_before_start"] == 1


def test_a_window_that_ends_before_it_starts_is_counted_not_silently_dropped():
    sheet = sheet_of(
        numbers=[10, 10],
        starts=[datetime(2025, 3, 2, 8), datetime(2025, 3, 1, 8)],
        ends=[datetime(2025, 3, 1, 8), datetime(2025, 3, 1, 12)],
    )
    wins, diag = w0.windows_with_diagnostics(sheets={"s": sheet}, numbers={"D1": 10})
    assert len(wins["D1"]) == 1
    assert diag[0]["n_rows_end_before_start"] == 1


def test_a_sheet_with_the_columns_and_rows_but_no_match_is_an_input_defect():
    sheet = sheet_of(numbers=[999], starts=[datetime(2025, 3, 1, 8)])
    with pytest.raises(w0.SwitchingLogDefectError):
        w0.windows_from_sheets(sheets={"s": sheet}, numbers={"D1": 10})
    # A sheet with no substation column is no mask, not a defect.
    assert w0.windows_from_sheets(sheets={"s": pl.DataFrame({"x": [1]})}, numbers={"D1": 10}) == {
        "D1": []
    }


def test_a_workbook_with_no_usable_sheet_gives_no_windows():
    wins = w0.windows_from_sheets(sheets={"s": pl.DataFrame({"x": [1]})}, numbers={"D1": 10})
    assert wins == {"D1": []}


def test_d10_counts_recount_export_hours_with_the_base_stated():
    n = 4 * 48
    signed = np.full(n, 3.0)
    signed[100:110] = -1.0
    zenith = np.full(n, 40.0)  # sun at 50 degrees
    out = w0.d10_counts(signed=signed, zenith=zenith, start=0)
    assert out["n_export_hours"] == 5  # hour-ending rows 101, 103, ..., 109
    assert out["n_export_days"] == 1
    assert out["n_base_hours"] == 96
    assert "hour-ending" in str(out["base"])
    dark = w0.d10_counts(signed=signed, zenith=np.full(n, 95.0), start=0)
    assert dark["n_export_hours"] == 0
    after = w0.d10_counts(signed=signed, zenith=zenith, start=120)
    assert after["n_export_hours"] == 0


def test_control_populations_and_outcomes():
    table = pl.DataFrame(
        {
            "is_post_bounce": [False, False, True, True, True, True],
            "n_post_bounce_in_day": [2, 2, 1, 1, 1, 2],
            "n_farm": [10, 7, 10, 12, 9, 11],
            "c_farm": [0.1, 0.9, -0.5, 0.6, -0.4, -0.3],
        }
    )
    # D12's anchor stretches on every day; D3's post-bounce stretches on one-stretch days only.
    anchors = w3.control_population(table=table, which="anchor")
    assert anchors.height == 1
    post = w3.control_population(table=table, which="post_bounce")
    assert post.height == 3
    outcome, counts = w3.control_outcome(rows=post)
    assert outcome == "void"
    assert counts == {"n": 3, "n_export_evidence": 1, "allowance": 0}
    big = pl.DataFrame({"c_farm": [-0.5] * 99 + [0.9]})
    assert w3.control_outcome(rows=big)[0] == "pass"
    worse = pl.DataFrame({"c_farm": [-0.5] * 97 + [0.9] * 3})
    assert w3.control_outcome(rows=worse)[0] == "fail"


def test_shares_of_dead_band_and_evidence():
    rows = pl.DataFrame({"c_farm": [0.5, -0.5, 0.0, 0.9], "n_farm": [10, 10, 10, 5]})
    s = w3.shares_of(rows=rows, sensor="farm")
    assert s["share_export"] == pytest.approx(0.25)
    assert s["share_import"] == pytest.approx(0.25)
    assert s["share_dead_band"] == pytest.approx(0.5)
    assert s["share_export_of_evidence"] == pytest.approx(0.5)


def test_one_stretch_days_in_season():
    month = np.array([1, 4, 7, 10, 9])
    table = pl.DataFrame(
        {
            "day": [0, 1, 2, 3, 4, 4],
            "is_post_bounce": [True] * 6,
            "n_post_bounce_in_day": [1, 1, 1, 1, 1, 2],
        }
    )
    kept = w3.one_stretch_days(table=table, month_of_day=month)
    assert kept["day"].to_list() == [1, 2, 4]


def test_the_w0_power_table_columns_follow_the_plan_page():
    rng = np.random.default_rng(8)
    rows = world.hour_end_rows(n=2000)
    row = world.power_row(
        response_change=rng.normal(0, 2, 2000), regressor_change=rng.normal(0, 1, 2000), rows=rows
    )
    assert set(row) == {"n_rows", "sd_response", "sd_regressor", "predicted_se", "detection_limit"}
    assert row["detection_limit"] == pytest.approx(core.DETECTION_SE_MULT * row["predicted_se"])


def _pairs(*, qualifies, negative=None, powered=None) -> pl.DataFrame:
    n = len(qualifies)
    frame = pl.DataFrame(
        {
            "host": ["D5", "D9", "D5", "D10"][:n],
            "farm": ["F1", "F2", "F3", "F1"][:n],
            "qualifies": qualifies,
        }
    )
    if negative is not None:
        frame = frame.with_columns(farm_reads_negative=pl.Series(negative))
    if powered is not None:
        frame = frame.with_columns(
            **{f"powered_level_{lv}": pl.Series(powered) for lv in core.LEVELS},
            **{f"scale_finite_level_{lv}": pl.Series([True] * n) for lv in core.LEVELS},
        )
    return frame


def test_w0_counts_a_pair_table_with_no_qualifying_pair_without_crashing(tmp_path):
    # The table of a run where every farm fails the arc condition: no pair qualifies, so no row
    # carries the scale, sign, or powered columns.
    pairs = _pairs(qualifies=[False, False])
    counts = common.pair_counts(pairs=pairs)
    assert counts == {
        "n_qualifying_pairs": 0,
        "n_powered_pairs_level_2": 0,
        "n_powered_hosts": 0,
        "n_pairs_non_finite_injection_scale": 0,
        "n_pairs_farm_reads_negative": 0,
    }
    full = common.with_pair_columns(frame=pairs)
    for name in (*common.PAIR_FLAG_COLUMNS, *common.PAIR_MEASURED_COLUMNS):
        assert name in full.columns
    # The check file writes, and the pair CSV reads back with typed Boolean columns, so job W2's
    # row 2 can record the outcome.
    guard.write_public_json(payload=counts, name=common.W0_CHECKS_NAME, out_dir=tmp_path)
    guard.write_public_csv(frame=full, name=common.W0_PAIRS_NAME, out_dir=tmp_path)
    back = common.read_pairs(out_dir=tmp_path)
    assert back["qualifies"].dtype == pl.Boolean
    assert common.pair_counts(pairs=back) == counts
    # A table with no row at all, and none of the columns, also counts as zero.
    assert common.pair_counts(pairs=pl.DataFrame())["n_qualifying_pairs"] == 0
    empty_csv = pl.DataFrame(schema=dict.fromkeys(common.PAIR_FLAG_COLUMNS, pl.Boolean))
    empty_csv.write_csv(tmp_path / common.W0_PAIRS_NAME)
    assert common.read_pairs(out_dir=tmp_path).height == 0
    assert core.void_before_run(
        n_pairs_keeping_sensor=0,
        n_powered_pairs_level_2=0,
        n_powered_hosts=0,
        arc_term_void_flag=False,
    )


def test_a_pair_whose_farm_reads_negative_does_not_qualify_and_is_counted():
    pairs = _pairs(
        qualifies=[True, True, True, False],
        negative=[False, True, False, True],
        powered=[True, True, False, True],
    )
    kept, n_dropped = common.qualifying_pairs(pairs=pairs)
    assert kept["farm"].to_list() == ["F1", "F3"]  # the second pair reads negative
    assert n_dropped == 1  # the fourth pair did not qualify, so it is not counted
    counts = common.pair_counts(pairs=pairs)
    assert counts["n_qualifying_pairs"] == 2
    assert counts["n_powered_pairs_level_2"] == 1
    assert counts["n_pairs_farm_reads_negative"] == 1


def _workbook(*, tmp_path: Path, rows: list[tuple]) -> dict[str, pl.DataFrame]:
    """Write a workbook with openpyxl (every date becomes a datetime) and read it as job W0 does."""
    import openpyxl

    book = openpyxl.Workbook()
    sheet = book.active
    sheet.append(["Substation Number", "Substation Name", "Start Time", "End Time"])
    for number, start, end in rows:
        sheet.append([number, "x", start, end])
    path = tmp_path / "log.xlsx"
    book.save(path)
    return w0.read_workbook(path=path)


def test_a_workbook_with_a_text_cell_in_a_date_column_is_readable_and_the_cell_is_counted(tmp_path):
    sheets = _workbook(
        tmp_path=tmp_path,
        rows=[
            (10, datetime(2025, 6, 10, 9, 0), datetime(2025, 6, 10, 12, 0)),
            (10, datetime(2025, 6, 11, 9, 0), "ongoing"),
            (10, "not known", datetime(2025, 6, 12)),
        ],
    )
    wins, diag = w0.windows_with_diagnostics(sheets=sheets, numbers={"D1": 10})
    assert len(wins["D1"]) == 2
    assert diag[0]["n_rows_end_not_parsed"] == 1
    assert diag[0]["n_rows_start_not_parsed"] == 1
    # The text end is a missing end: the local calendar day of the start.
    assert wins["D1"][1] == (
        _utc_row(year=2025, month=6, day=10, hour=23),
        _utc_row(year=2025, month=6, day=11, hour=23),
    )


@pytest.mark.parametrize(("month", "day", "offset"), [(6, 10, 1), (1, 15, 0)])
def test_a_mixed_column_of_dates_and_times_is_read_per_row_in_summer_and_winter(
    tmp_path, month, day, offset
):
    d = lambda h=0, m=0, dd=day: datetime(2025, month, dd, h, m)  # noqa: E731
    sheets = _workbook(
        tmp_path=tmp_path,
        rows=[
            (10, d(9), d()),  # timed start, date-only end the same day: local 24:00
            (10, d(), d(17)),  # date-only start, timed end
            (10, d(), None),  # date-only start with no end: the local day
            (10, d(9, 30), None),  # timed start with no end: the local day too
            (10, d(), d()),  # the same date at both ends: the whole local day
            (10, d(22), d(dd=day + 1)),  # overnight: the end stays at 00:00 of the next day
        ],
    )
    wins, diag = w0.windows_with_diagnostics(sheets=sheets, numbers={"D1": 10})
    assert diag[0]["n_rows_end_before_start"] == 0
    local_midnight = -offset
    whole_day = (
        _utc_row(year=2025, month=month, day=day, hour=local_midnight),
        _utc_row(year=2025, month=month, day=day, hour=24 + local_midnight),
    )
    got = wins["D1"]
    assert got[0] == (
        _utc_row(year=2025, month=month, day=day, hour=9 - offset),
        whole_day[1],
    )
    assert got[1] == (
        whole_day[0],
        _utc_row(year=2025, month=month, day=day, hour=17 - offset),
    )
    assert got[2] == whole_day
    assert got[3] == whole_day
    assert got[4] == whole_day
    assert got[5] == (
        _utc_row(year=2025, month=month, day=day, hour=22 - offset),
        _utc_row(year=2025, month=month, day=day + 1, hour=local_midnight),
    )


def test_a_two_digit_year_is_an_error_and_a_window_outside_the_grid_is_counted():
    sheet = sheet_of(
        numbers=[10, 10, 10],
        starts=["10/06/25", "10/06/1999", "01/01/2001"],
        ends=["10/06/25", "11/06/1999", "02/01/2001"],
    )
    wins, diag = w0.windows_with_diagnostics(sheets={"s": sheet}, numbers={"D1": 10})
    assert diag[0]["n_rows_time_implausible"] == 2  # a two-digit year reads as year 25
    assert diag[0]["n_rows_start_parsed"] == 1
    assert diag[0]["n_rows_start_not_parsed"] == 0
    assert diag[0]["n_windows_outside_grid"] == 1  # the window of 2001 lies before the grid
    assert len(wins["D1"]) == 1
