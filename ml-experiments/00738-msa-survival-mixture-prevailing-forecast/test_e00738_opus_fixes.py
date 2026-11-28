"""Tests of the fixes that the second (Opus) code review of experiment 738 asked for."""
# ruff: noqa: D103, E402, PT018, ANN001

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

for _k in ("NGED_S3_BUCKET_URL", "NGED_S3_BUCKET_ACCESS_KEY", "NGED_S3_BUCKET_SECRET"):
    os.environ.setdefault(_k, "dummy")
os.environ.setdefault("E00738_OUT", str(Path(tempfile.gettempdir()) / "scratch" / "e738_tests"))
os.environ.setdefault("E00738_SCRATCH", "1")
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
for _d in HERE.parent.glob("*"):
    if _d.is_dir() and _d.name[:5].isdigit() and int(_d.name[:5]) in {
        11, 87, 377, 378, 381, 391, 397, 400, 402, 404, 407, 410, 418, 422, 425, 428, 435, 437, 448,
        454, 470, 510, 511, 513, 515, 521, 545, 558, 559, 560, 562, 565, 568, 569, 572, 574, 578,
        581, 632, 667, 668, 709, 647, 650, 652, 654, 713, 726,
    }:  # fmt: skip
        sys.path.append(str(_d))

import e00738_common as common
import e00738_gate as gate
import e00738_members as members
import e00738_panel as panel
import e00738_phases as phases
import e00738_predict as pred
import e00738_report as report
import e00738_run as run
import e00738_score as score
import e00738_survival as surv
import e00738_world as world
import e00738_worldcheck as wc
import polars as pl

# --- A1: the detection threshold is tuned on real event-free rows, and a failure stops the run ---


def _quiet_rows(*, n: int, jump_share: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return history rows with unit noise and a level jump of 5 noise units in a share of them."""
    rng = np.random.default_rng(seed)
    power = rng.normal(0.0, 0.05, (n, 14, 48))
    jump = rng.random(n) < jump_share
    power[jump, :4, :] += 5.0 * 0.05 * 3.0
    return power, np.ones_like(power, dtype=bool)


def test_tune_a1_z_returns_the_smallest_threshold_that_meets_the_bar() -> None:
    power, visible = _quiet_rows(n=3000, jump_share=0.0)
    z, share = surv.tune_a1_z(hist_power=power, visible=visible, noise=0.05 / np.sqrt(48))
    assert z is not None and share <= common.A1_FALSE_ALARM_MAX
    smaller = [g for g in common.A1_Z_GRID if g < z]
    if smaller:
        age = surv.age_rule_a1(
            hist_power=power, visible=visible, noise=0.05 / np.sqrt(48), z=smaller[-1]
        )
        assert (age > 0).mean() > common.A1_FALSE_ALARM_MAX


def test_tune_a1_z_reports_failure_when_no_threshold_on_the_grid_meets_the_bar() -> None:
    power, visible = _quiet_rows(n=2000, jump_share=1.0)
    z, share = surv.tune_a1_z(hist_power=power, visible=visible, noise=0.001)
    assert z is None and share > common.A1_FALSE_ALARM_MAX


def test_a1_threshold_is_read_from_phase_zero_and_refused_when_missing(
    monkeypatch, tmp_path
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    monkeypatch.setattr(common, "REAL_OUT", real)
    folder = tmp_path / "data"
    monkeypatch.setattr(common, "OUT", folder)
    monkeypatch.setattr(common, "is_scratch", lambda out=None: False)
    with pytest.raises(RuntimeError, match="threshold"):
        surv.a1_threshold()
    common.atomic_json(obj={"z": 4.5}, path=folder / common.A1_THRESHOLD_FILE)
    assert surv.a1_threshold() == 4.5


# --- A2: the empirical curve ------------------------------------------------------------------


def test_the_empirical_curve_is_a_step_function_of_the_durations() -> None:
    curve = surv.EmpiricalCurve.of(durations_days=np.array([4.0, 2.0, 2.0, 10.0]))
    np.testing.assert_allclose(
        curve.sf(np.array([0.0, 2.0, 3.0, 4.0, 10.0])), [1, 0.5, 0.5, 0.25, 0]
    )
    back = surv.curve_from_record(record=surv.curve_record(curve=curve))
    assert back == curve
    p = surv.ratio(curve=curve, age_days=np.array([3.0]), lead_days=np.array([2.0]))
    np.testing.assert_allclose(p, [0.5])


def test_the_empirical_curve_matches_the_generator_where_the_weibull_does_not() -> None:
    rng = np.random.default_rng(0)
    drawn = np.where(
        rng.random(20000) < 0.5, rng.integers(3, 15, 20000), rng.integers(28, 113, 20000)
    ).astype(float)
    empirical = surv.EmpiricalCurve.of(durations_days=drawn)
    assert surv.curve_error_ok(curve=empirical)
    assert max(surv.curve_error(curve=surv.fit_curve(durations_days=drawn)).values()) > 0.05


# --- A4: nothing on or after 2026-07-01 -------------------------------------------------------


def test_holdout_start_is_2026_07_01_utc() -> None:
    assert (
        int(time.mktime(time.strptime("2026-07-01", "%Y-%m-%d")) - time.timezone) * 1_000_000
        == common.HOLDOUT_START_US
    )


def test_the_forecast_valid_time_filter_drops_exactly_the_holdout_rows() -> None:
    valid = np.array(
        [common.HOLDOUT_START_US - 1, common.HOLDOUT_START_US, common.HOLDOUT_START_US + 5]
    )
    assert panel.before_holdout(valid_us=valid).tolist() == [True, False, False]


def test_scoring_refuses_a_valid_time_in_the_holdout() -> None:
    limit = (common.HOLDOUT_START_US - common.GRID_START_US) // common.STEP_US
    score.assert_before_holdout(valid_idx=np.array([limit - 1]))
    with pytest.raises(AssertionError):
        score.assert_before_holdout(valid_idx=np.array([limit]))


def test_the_phase_zero_world_check_reads_the_files_and_finds_the_holdout_fold(tmp_path) -> None:
    for month in common.FOLDS:
        wc.write_worlds(month=month, out=tmp_path)
    found = wc.check_worlds(months=common.FOLDS, out=tmp_path)
    result = wc.verdict(found=found)
    assert result == {"off": {}, "faults": []}
    assert found["dropped_rows"][wc.LAST_HOLDOUT_FOLD] > 0
    assert all(n == 0 for m, n in found["dropped_rows"].items() if m != wc.LAST_HOLDOUT_FOLD)
    assert found["latest_shift_end_idx"] <= found["holdout_idx"]


def test_the_world_check_names_a_shift_that_reaches_the_holdout_and_a_missing_filter() -> None:
    found = {
        "shares": dict(common.EXPECTED_SHARES),
        "dropped_rows": dict.fromkeys(common.FOLDS, 0),
        "latest_shift_end_idx": 10,
        "holdout_idx": 5,
    }
    faults = wc.verdict(found=found)["faults"]
    assert any("reaches" in f for f in faults) and any(wc.LAST_HOLDOUT_FOLD in f for f in faults)


# --- A5: strata from the drawn end ------------------------------------------------------------


def test_scoring_strata_use_the_drawn_end_and_never_clip() -> None:
    shift = world.Shift(
        primary=0, kind="constant", onset_idx=100, ramp_half_hours=1, end_idx=150,
        drawn_days=10.0, depth=0.1, sign=1, family="train",
    )  # fmt: skip
    onset, end, clipped = score.shift_arrays(shifts=[shift])
    assert end[0] == 100 + 480 and not clipped[0]
    label = world.strata(
        origin_idx=np.array([120]), valid_idx=np.array([200]), onset_idx=onset, end_idx=end,
        clipped=clipped,
    )  # fmt: skip
    assert label[0] == "middle"


# --- A6: the gate's bars follow M1's own follow share -----------------------------------------


def test_the_gate_bars_scale_with_the_share_that_m1_already_follows() -> None:
    zero = gate.follow_bars(m1_follow=0.0)
    assert zero["lp_follow_bar"] == pytest.approx(0.75) and zero[
        "depth_ratio_bar"
    ] == pytest.approx(0.7)
    half = gate.follow_bars(m1_follow=0.5)
    assert half["lp_follow_bar"] == pytest.approx(0.375)
    assert gate.follow_bars(m1_follow=1.0)["remaining"] == pytest.approx(0.2)
    ok = {"midday": 1.0, "midnight": 1.0}
    assert gate.evaluate(oracle=ok, lp_follow=0.4, depth_ratio=0.36, m1_follow=0.5)["passed"]
    rec = gate.evaluate(oracle=ok, lp_follow=0.4, depth_ratio=0.36, m1_follow=0.0)
    assert not rec["passed"] and "M1 follows 0.000" in rec["derivation"]


# --- A7: S* by unscaled CRPS ------------------------------------------------------------------


def test_the_screen_chooses_s_star_with_the_unscaled_metric() -> None:
    import inspect

    assert inspect.signature(report.screen_in_force).parameters["metric"].default == "crps"
    assert "in_force_unscaled" in inspect.getsource(run.stage_screen_score)


# --- B1: recipe hashes -------------------------------------------------------------------------


def test_a_file_with_no_or_another_recipe_hash_is_refused(tmp_path) -> None:
    code = tmp_path / "code"
    code.mkdir()
    for name in phases.FORECAST_MODULES:
        (code / f"e00738_{name}.py").write_text(f"# {name}\n")
    data = tmp_path / "x.parquet"
    data.write_bytes(b"x")
    with pytest.raises(RuntimeError, match="recipe"):
        phases.require_recipe(path=data, modules=phases.FIT_MODULES, folder=code)
    phases.write_recipe(path=data, modules=phases.FIT_MODULES, folder=code)
    phases.require_recipe(path=data, modules=phases.FIT_MODULES, folder=code)
    (code / "e00738_fit.py").write_text("# changed\n")
    with pytest.raises(RuntimeError, match="recipe"):
        phases.require_recipe(path=data, modules=phases.FIT_MODULES, folder=code)
    # a module outside the fitting recipe does not invalidate the file
    (code / "e00738_fit.py").write_text("# fit\n")
    (code / "e00738_report.py").write_text("# changed\n")
    phases.require_recipe(path=data, modules=phases.FIT_MODULES, folder=code)


def test_require_all_recipes_walks_states_and_forecasts(tmp_path) -> None:
    state = tmp_path / "networks" / "LP" / "2025-11_s0.pt"
    state.parent.mkdir(parents=True)
    state.write_bytes(b"x")
    forecast = tmp_path / "predictions" / "LP" / "2025-11_s0" / "real.parquet"
    link = phases.state_digest(arm="LP", stem="2025-11_s0", out=tmp_path)
    pred.write_parquet_atomic(frame=pl.DataFrame({"a": [1]}), path=forecast, link=link)
    with pytest.raises(RuntimeError):  # the state has no recipe file
        phases.require_all_recipes(out=tmp_path)
    phases.write_recipe(path=state, modules=phases.FIT_MODULES)
    assert phases.require_all_recipes(out=tmp_path) == 2
    phases.recipe_path(path=forecast).write_text("0" * 64 + "\n")
    with pytest.raises(RuntimeError):
        phases.require_all_recipes(out=tmp_path)


def test_a_forecast_written_by_other_code_is_neither_reused_nor_overwritten(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(common, "OUT", tmp_path)
    path = pred.panel_path(arm="LP", month="2025-11", seed=0, name="real")
    pred.write_parquet_atomic(frame=pl.DataFrame({"a": [1]}), path=path)
    # a head arm's forecast without its diagnostics is incomplete
    assert not pred.panel_done(arm="LP", month="2025-11", seed=0, name="real")
    pred.write_parquet_atomic(
        frame=pl.DataFrame({"a": [1]}), path=pred.diag_path(path=path, name="real")
    )
    assert pred.panel_done(arm="LP", month="2025-11", seed=0, name="real")
    phases.recipe_path(path=path).write_text("0" * 64 + "\n")
    with pytest.raises(RuntimeError):
        pred.panel_done(arm="LP", month="2025-11", seed=0, name="real")


# --- B3: the gate's remedy resumes after a crash ----------------------------------------------


def test_the_first_attempt_moves_aside_once_and_a_rerun_keeps_the_remedy_fits(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(common, "OUT", tmp_path)
    for m in common.SCREEN_FOLDS:
        for arm in common.HEAD_ARMS:
            for path in common.fit_paths(arm=arm, month=m, seed=0).values():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("first")
            folder = tmp_path / "predictions" / arm / f"{m}_s0"
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "real.parquet").write_text("first")
    assert run.archive_first_attempt()
    state = common.fit_paths(arm="LP", month=common.SCREEN_FOLDS[0], seed=0)["state"]
    assert not state.exists()
    assert (tmp_path / "gate_first_attempt" / "networks" / "LP" / state.name).read_text() == "first"
    state.write_text("remedy")  # a remedy fit finishes, then the stage crashes
    assert not run.archive_first_attempt()
    assert state.read_text() == "remedy"


def test_remedy_complete_needs_every_remedy_task_done(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(common, "OUT", tmp_path)
    (tmp_path / gate.REMEDY_FLAG).write_text("x")
    assert not gate.remedy_complete()


# --- B5: waits and workers --------------------------------------------------------------------


def test_a_wait_is_printed_and_does_not_count_against_the_stop_line(
    monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.setattr(common, "OUT", tmp_path)
    state = {"n": 0}

    def busy() -> bool:
        state["n"] += 1
        return state["n"] <= 2

    assert phases.wait_for_gpu(stage="s", task="t", busy=busy, sleep=lambda _s: None) == 2
    assert "waiting for the GPU" in capsys.readouterr().out
    log = tmp_path / "time_log.csv"
    log.write_text(
        "s,wait_t,gpu,2026-01-01T00:00:00,2026-01-01T05:00:00\n"
        "s,fit,gpu,2026-01-01T05:00:00,2026-01-01T06:00:00\n"
    )
    assert phases.spent_hours(device="gpu", log=log) == pytest.approx(1.0)
    assert phases.waited_hours(device="gpu", log=log) == pytest.approx(5.0)


def test_a_gpu_stage_refuses_more_than_one_worker() -> None:
    with pytest.raises(SystemExit) as exc:
        run.main(argv=["screen-fits", "--of", "2", "--device", "cpu"])
    assert exc.value.code == common.EXIT_BAD_CALL


# --- C3: smoke hours enter the real ledger ----------------------------------------------------


def test_smoke_ledger_lines_are_booked_once_in_the_real_ledger(monkeypatch, tmp_path) -> None:
    real, scratch = tmp_path / "real", tmp_path / "scratch"
    real.mkdir()
    scratch.mkdir()
    monkeypatch.setattr(common, "REAL_OUT", real)
    monkeypatch.setattr(common, "OUT", scratch)
    (scratch / "time_log.csv").write_text(
        "smoke,a,gpu,2026-01-01T00:00:00,2026-01-01T00:06:00\nother,b,cpu,2026-01-01T00:00:00,2026-01-01T01:00:00\n"
    )
    assert run.book_smoke_ledger() == 1
    assert run.book_smoke_ledger() == 0
    assert phases.spent_hours(device="gpu", log=real / "time_log.csv") == pytest.approx(0.1)


# --- Members: partial blocks at the holdout cut -----------------------------------------------


def test_the_member_check_leaves_out_partial_blocks() -> None:
    n = members.STEPS_PER_ROW
    frame = pl.DataFrame(
        {
            "time_series_id": [1] * (2 * n - 5),
            "power_fcst_init_time": [0] * (2 * n - 5),
            "lead_day": [1] * n + [2] * (n - 5),
        }
    )
    assert members.complete_blocks(frame=frame).height == n


# --- C1 and C2: the report ---------------------------------------------------------------------


def test_the_scenario_slices_are_the_ones_the_sums_hold() -> None:
    assert report.SCENARIOS["inside_events"] == ("in_force",)
    assert {sl for v in report.SCENARIOS.values() for sl in v} <= {
        "all", "in_force", "event_free", "ended_24h", "ended_1_7",
    }  # fmt: skip


def test_phase_zero_readings_read_the_curve_and_a1_files(tmp_path) -> None:
    common.atomic_json(
        obj={"worst": 0.01, "error_ok": True}, path=tmp_path / "curves" / "2025-11.json"
    )
    common.atomic_json(obj={"tuned_z": 5.0}, path=tmp_path / "a1" / "2025-11.json")
    common.atomic_json(obj={"z": 5.0}, path=tmp_path / "a1" / "threshold.json")
    found = report.phase0_readings(out=tmp_path)
    assert found["curves"]["2025-11"]["worst"] == 0.01
    assert list(found["a1"]) == ["2025-11"]
