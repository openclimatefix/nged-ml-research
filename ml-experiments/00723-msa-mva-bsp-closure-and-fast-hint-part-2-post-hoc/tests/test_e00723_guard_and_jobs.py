# ruff: noqa: E501, ANN001
"""The guard, the run wrapper, the job order, and the pure helpers of job W0, on synthetic inputs."""

import re
from datetime import UTC, datetime
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
        job="t", gate=gate, inputs=[table, tmp_path / "missing"], info={"x": 1}, out_dir=tmp_path
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
        guard.run_log(job="c", gate=gate, inputs=[], info={}, out_dir=tmp_path),
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
            "is_post_bounce": [False, False, True, True, True],
            "n_farm": [10, 7, 10, 12, 9],
            "c_farm": [0.1, 0.9, -0.5, 0.6, -0.4],
        }
    )
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
