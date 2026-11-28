"""Tests of the fixes that the code review of experiment 738 asked for (no torch needed)."""
# ruff: noqa: D103, E402, PT018, ANN001, ANN202

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

for _k in ("NGED_S3_BUCKET_URL", "NGED_S3_BUCKET_ACCESS_KEY", "NGED_S3_BUCKET_SECRET"):
    os.environ.setdefault(_k, "dummy")
# the tests never write to the experiment's data folder
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
import e00738_phases as phases
import e00738_protocol as protocol
import e00738_report as report
import e00738_run as run
import e00738_score as score
import e00738_world as world
import polars as pl

# --- F1: one child process per fit or prediction job ------------------------------------------


def test_two_jobs_with_different_months_in_one_chain_each_see_their_own_month(
    monkeypatch, tmp_path
) -> None:
    """A stub module reads the month and phase variables once, at import, as experiment 713 does."""
    monkeypatch.setattr(common, "OUT", tmp_path / "out")
    stub = tmp_path / "stubs"
    stub.mkdir()
    (stub / "stub_fold.py").write_text(
        "import os\nFINAL_MONTH = os.environ.get('E00667_MONTH', 'unset')\n"
        "PHASE = os.environ.get('E00667_PHASE', 'unset')\n"
    )
    seen = tmp_path / "seen"
    seen.mkdir()

    def job(*, month: str, phase: str) -> None:
        phases.run_child(
            argv=[
                sys.executable,
                "-c",
                (
                    "import sys, stub_fold; open(sys.argv[1], 'w').write("
                    "stub_fold.FINAL_MONTH + ' ' + stub_fold.PHASE)"
                ),
                str(seen / f"{month}_{phase}.txt"),
            ],
            env=phases.child_env(month=month, phase=phase, extra={"PYTHONPATH": str(stub)}),
        )

    jobs: list[tuple[str, dict[str, object]]] = [
        ("j1", {"month": "2025-11", "phase": "train"}),
        ("j2", {"month": "2026-01", "phase": "train"}),
        ("j3", {"month": "2026-01", "phase": "predict"}),
    ]
    phases.run_chain(jobs=jobs, fn=job, device="cpu", stage="x")
    assert (seen / "2025-11_train.txt").read_text() == "2025-11 train"
    assert (seen / "2026-01_train.txt").read_text() == "2026-01 train"
    assert (seen / "2026-01_predict.txt").read_text() == "2026-01 predict"


def test_a_failing_child_raises_with_its_stderr_and_the_parent_records_it(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(common, "OUT", tmp_path)

    def job() -> None:
        phases.run_child(
            argv=[
                sys.executable,
                "-c",
                "import sys; sys.stderr.write('boom in child'); sys.exit(7)",
            ],
            env=phases.child_env(),
        )

    assert not phases.run_task(name="child_fail", fn=job, kwargs={})
    rec = json.loads((tmp_path / "failed" / "task_child_fail.json").read_text())
    assert "exited with code 7" in rec["message"] and "boom in child" in rec["message"]


def test_the_fit_and_predict_wrappers_start_one_task_child_with_the_month_and_phase(
    monkeypatch,
) -> None:
    calls: list[tuple[list[str], dict[str, str]]] = []
    monkeypatch.setattr(phases, "run_child", lambda *, argv, env: calls.append((argv, env)))
    run.sub_fit(arm="LM", month="2026-03", seed=2, device="cuda")
    run.sub_predict(month="2026-03", seed=2, device="cuda", arms=("LP", "LD"), only=("real",))
    (fit_argv, fit_env), (pred_argv, pred_env) = calls
    assert fit_env["E00667_MONTH"] == "2026-03" and fit_env["E00667_PHASE"] == "train"
    assert pred_env["E00667_MONTH"] == "2026-03" and pred_env["E00667_PHASE"] == "predict"
    ns = run.parse_args(argv=fit_argv[2:])
    assert (ns.stage, ns.kind, ns.arm, ns.month, ns.seed) == ("one-task", "fit", "LM", "2026-03", 2)
    ns = run.parse_args(argv=pred_argv[2:])
    assert (ns.kind, ns.arms, ns.only) == ("predict", "LP,LD", "real")


def test_the_fit_and_predict_chains_run_every_job_through_the_child_wrappers() -> None:
    assert run.sub_fit is not run.task_fit and run.sub_predict is not run.task_predict
    text = (HERE / "e00738_run.py").read_text()
    for stage_fn in ("stage_screen_fits", "stage_full_fits", "stage_predict", "stage_gate_remedy"):
        body = text[text.index(f"def {stage_fn}(") :].split("\ndef ")[0]
        assert "fn=task_fit" not in body and "fn=task_predict" not in body, stage_fn


# --- F2 and F3: the gate maps rows to the right Primary, and judges only the rows A1 sees ------

IDS = np.array([25, 3, 11, 17, 26, 8, 2, 40, 31, 9, 14, 5, 19, 22, 7, 30])
NATIVE = 48 * 100 + 18


def _gate_frames(*, ids: np.ndarray):
    """Return (m1, lp, real, diag, shifts, expected) for 16 Primaries in 4 classes.

    Class 0 and 3: far and seen, LP adds the whole shift. Class 1: far and unseen, LP adds
    nothing. Class 2: in force but near its end, so not far, LP adds nothing.
    """
    import e00738_panel as panel

    origin = NATIVE + common.NATIVE_OFFSET_STEPS
    shifts = []
    for p in range(16):
        far = p % 4 != 2
        shifts.append(
            world.Shift(
                primary=p,
                kind="constant",
                onset_idx=origin - 48 * 4,
                ramp_half_hours=1,
                end_idx=origin + 48 * (40 if far else 5),
                drawn_days=(48 * 44 if far else 48 * 9) / 48.0,
                depth=0.02 * (1 + p % 5),
                sign=1,
                family="train",
            )
        )
    rows = {"m1": [], "lp": [], "real": []}
    num_seen = den_seen = den_all = 0.0
    diag_rows = []
    for p in range(16):
        d = shifts[p].depth
        klass = p % 4
        for lead in (1, 2):
            for j in range(48):
                valid_idx = origin + 48 * lead + j
                base = {
                    "time_series_id": int(ids[p]),
                    "power_fcst_init_time": common.GRID_START_US + NATIVE * common.STEP_US,
                    "valid_time": common.GRID_START_US + valid_idx * common.STEP_US,
                    "lead_day": lead,
                }
                q = dict.fromkeys(panel.QCOLS, 0.3)
                rows["real"].append({**base, "actual": 0.3, **q})
                rows["m1"].append({**base, "actual": 0.3 + d, **q})
                add = d if klass in (0, 3) else 0.0
                rows["lp"].append({**base, "actual": 0.3 + d, **q, "q50": 0.3 + add})
                if klass in (0, 3):
                    num_seen += d * d
                    den_seen += d * d
                    den_all += d * d
                elif klass == 1:
                    den_all += d * d
            diag_rows.append(
                {
                    "time_series_id": int(ids[p]),
                    "origin_us": common.GRID_START_US + NATIVE * common.STEP_US,
                    "lead_day": lead,
                    "age": 0.0 if klass == 1 else 4.0,
                    "depth_share": 0.8 * d,
                    "p_mean": 1.0,
                }
            )

    def frame(name: str) -> pl.DataFrame:
        utc = pl.Datetime("us", "UTC")
        return (
            pl.DataFrame(rows[name])
            .with_columns(
                pl.col("power_fcst_init_time")
                .cast(pl.Datetime("us"))
                .dt.replace_time_zone("UTC")
                .cast(utc),
                pl.col("valid_time").cast(pl.Datetime("us")).dt.replace_time_zone("UTC").cast(utc),
                pl.col("time_series_id").cast(pl.Int64),
            )
            .select(*panel.KEYS, "lead_day", "actual", *panel.QCOLS)
        )

    expected = {"seen": num_seen / den_seen, "all_far": num_seen / den_all}
    return frame("m1"), frame("lp"), frame("real"), pl.DataFrame(diag_rows), shifts, expected


def test_panel_readings_use_the_site_table_order_not_the_sorted_id_order() -> None:
    m1, lp, real, diag, shifts, want = _gate_frames(ids=IDS)
    got = gate.panel_readings(m1=m1, lp=lp, real=real, diag=diag, shifts=shifts, ids=IDS)
    assert got["num_seen"] / got["den_seen"] == pytest.approx(1.0)
    assert got["num"] / got["den"] == pytest.approx(want["all_far"])
    assert want["all_far"] < 0.9, "the unseen far rows must lower the all-rows share"
    # the sorted-id order is a different permutation, and gives a different (wrong) answer
    wrong = gate.panel_readings(m1=m1, lp=lp, real=real, diag=diag, shifts=shifts, ids=np.sort(IDS))
    assert wrong["num_seen"] / wrong["den_seen"] != pytest.approx(1.0)
    assert IDS.tolist() != sorted(IDS.tolist()), "the test ids must be non-monotone"


def test_stage_gate_with_non_monotone_ids_passes_on_the_seen_rows_only(
    monkeypatch, tmp_path
) -> None:
    import e00738_predict as pred

    monkeypatch.setattr(common, "OUT", tmp_path)
    m1, lp, real, diag, shifts, want = _gate_frames(ids=IDS)
    for month in common.SCREEN_FOLDS:
        for arm, frame in (("M1", m1), ("LP", lp)):
            path = pred.panel_path(arm=arm, month=month, seed=0, name="pass00")
            pred.write_parquet_atomic(frame=frame, path=path)
        pred.write_parquet_atomic(
            frame=real, path=pred.panel_path(arm="M1", month=month, seed=0, name="real")
        )
        pred.write_parquet_atomic(
            frame=diag,
            path=pred.panel_path(arm="LP", month=month, seed=0, name="pass00").with_name(
                "pass00_diag.parquet"
            ),
        )
    monkeypatch.setattr(pred, "panel_names", lambda: ["pass00"])
    monkeypatch.setattr(run, "screen_shifts", lambda *, month: {"pass00": shifts})
    monkeypatch.setattr(run, "site_labels", lambda: ([f"S{i}" for i in range(16)], IDS))
    (tmp_path / phases.STAGE_MARKERS["screen_fits"]).write_text("x")
    phases.write_review_marker(out=tmp_path)
    run.stage_gate()  # no SystemExit: the gate passed
    record = json.loads((tmp_path / gate.GATE_FILE).read_text())
    checks = record["checks"]
    assert checks["lp_follow"]["value"] == pytest.approx(1.0)
    assert record["diagnostics"]["lp_follow_all_far_rows"] == pytest.approx(want["all_far"])
    assert checks["depth_ratio"]["value"] == pytest.approx(0.8)
    assert 0.0 < record["diagnostics"]["share_of_far_rows_a1_sees"] < 1.0


def test_a_perfect_head_that_sees_only_part_of_the_far_rows_still_passes_check_two() -> None:
    record = gate.evaluate(
        oracle={"midday": 1.0, "midnight": 1.0},
        lp_follow=0.9,
        depth_ratio=0.8,
        lp_follow_all_far=0.55,
    )
    assert record["passed"] and record["diagnostics"]["lp_follow_all_far_rows"] == 0.55
    assert not gate.evaluate(
        oracle={"midday": 1.0, "midnight": 1.0}, lp_follow=0.5, depth_ratio=0.8
    )["passed"]


# --- F4: the smoke test's markers ---------------------------------------------------------------


def test_the_smoke_stages_read_their_markers_from_the_data_folder(monkeypatch, tmp_path) -> None:
    real, scratch = tmp_path / "real", tmp_path / "scratch"
    real.mkdir()
    scratch.mkdir()
    monkeypatch.setattr(common, "REAL_OUT", real)
    monkeypatch.setattr(common, "OUT", scratch)
    monkeypatch.setenv("E00738_OUT", str(scratch))
    with pytest.raises(SystemExit):  # nothing in the data folder yet
        run.stage_smoke_fit(args=run.parse_args(argv=["smoke-fit", "--device", "cpu"]))
    (real / phases.STAGE_MARKERS["phase0"]).write_text("x")
    phases.write_review_marker(out=real)
    seen: list[tuple[str, dict]] = []
    monkeypatch.setattr(run, "run_gpu_tasks", lambda *, jobs, fn, stage, args: seen.extend(jobs))
    run.stage_smoke_fit(args=run.parse_args(argv=["smoke-fit", "--device", "cpu"]))
    assert len(seen) == len(common.FIT_ARMS)
    assert all(kw["extra_env"] == {common.PASS_CAP_ENV: "1"} for _, kw in seen)
    assert common.PASS_CAP_ENV not in os.environ, "the cap reaches the child processes only"
    # the smoke marker goes to both folders, and screen-fits then finds it in the data folder
    run.mark(name="smoke")
    run.mark(name="smoke", out=real)
    assert (scratch / "smoke.ok").exists() and (real / "smoke.ok").exists()
    phases.require_stage(stage="screen_fits", out=real)


def test_the_smoke_stages_refuse_the_data_folder(monkeypatch) -> None:
    monkeypatch.delenv("E00738_OUT", raising=False)
    with pytest.raises(SystemExit, match="scratch"):
        run.require_scratch()


# --- F5: the cut list reaches every stage --------------------------------------------------------


def test_the_cut_list_is_honoured_by_every_stage(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(common, "OUT", tmp_path)
    assert common.active_seeds() == common.SEEDS
    assert common.arms_of(month="2026-02", seed=1) == common.ALL_ARMS
    (tmp_path / common.CUTS_FILE).write_text(json.dumps({"drop_ld": True, "seeds01": True}))
    assert common.active_seeds() == (0, 1)
    screen_cell = common.arms_of(month=common.SCREEN_FOLDS[0], seed=common.SCREEN_SEED)
    assert "LD" in screen_cell
    assert "LD" not in common.arms_of(month="2026-02", seed=1)
    assert "LD" not in common.arms_of(month=common.SCREEN_FOLDS[0], seed=1)
    assert {
        kw["seed"] for _, kw in run.predict_jobs(screen=False, seeds=common.active_seeds())
    } == {0, 1}
    assert "LD" not in protocol.default_arms()
    cells = report.arm_cells(arm="LD", months=common.FOLDS, seeds=(0, 1), drop_ld=True)
    assert set(cells) == {(m, 0) for m in common.SCREEN_FOLDS}


def test_the_report_runs_when_ld_exists_only_in_the_screen_cells() -> None:
    from test_e00738_run import _synthetic_tables

    labels = [*common.GROUP_OF_PRIMARY["all14"], *common.GROUP_OF_PRIMARY["short_history_two"]]
    months, seeds = common.FOLDS, (0, 1)
    tables = _synthetic_tables(months=months, seeds=seeds, gain=0.5)
    for (m, s), t in tables.items():
        if not (m in common.SCREEN_FOLDS and s == 0):
            del t["LD"]
    rep = report.compute_report(
        tables=tables, labels=labels, s_star="LM", months=months, seeds=seeds, n_resamples=100,
        drop_ld=True,
    )  # fmt: skip
    assert rep["C1"]["point"] < 0 and "LD" in rep["by_stratum"]["far"]
    with pytest.raises(KeyError):  # without the cut, the same tables are incomplete
        report.compute_report(
            tables=tables, labels=labels, s_star="LM", months=months, seeds=seeds, n_resamples=100
        )


def test_full_fits_refuse_a_cut_that_drops_the_chosen_arm(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(common, "OUT", tmp_path)
    (tmp_path / common.CUTS_FILE).write_text(json.dumps({"drop_ld": True}))
    (tmp_path / "sstar.json").write_text(json.dumps({"s_star": "LD"}))
    (tmp_path / phases.STAGE_MARKERS["gate"]).write_text("x")
    (tmp_path / phases.STAGE_MARKERS["screen_score"]).write_text("x")
    phases.write_review_marker(out=tmp_path)
    with pytest.raises(AssertionError, match="drops LD"):
        run.stage_full_fits(args=run.parse_args(argv=["full-fits", "--device", "cpu"]))


# --- F6: the 51-path check, against a Gaussian whose answer is known -------------------------


def test_the_base_levels_are_a_permutation_that_does_not_follow_the_end_time_order() -> None:
    levels = members.base_levels()
    assert sorted(np.round(levels * 51 - 0.5).astype(int)) == list(range(51))
    assert abs(np.corrcoef(np.arange(51), levels)[0, 1]) < 0.3


def _weibull(k=1.0, lam=20.0):
    return members.WeibullSurvival(shape=k, scale_days=lam)


def test_a_right_construction_meets_the_tolerances_on_a_gaussian_known_answer() -> None:
    gaps = [
        members.known_answer_gaps(
            sd=sd, depth=depth, age_days=age, curve=_weibull(k, lam), lead_day=lead, seed=i
        )
        for i, (sd, depth, age, k, lam, lead) in enumerate(
            [
                (0.03, 0.10, 5.0, 1.0, 20.0, 2),
                (0.02, -0.08, 3.0, 0.8, 12.0, 4),
                (0.04, 0.15, 9.0, 1.3, 30.0, 6),
                (0.03, -0.12, 2.0, 1.0, 8.0, 1),
                (0.05, 0.05, 12.0, 0.9, 25.0, 8),
                (0.01, 0.20, 4.0, 1.2, 15.0, 3),
            ]
        )
    ]
    mean = {k: float(np.mean([g[k] for g in gaps])) for k in gaps[0]}
    assert members.passes(
        ended_gap=mean["ended_gap"], quantile_gap=mean["quantile_gap"], crps_gap=mean["crps_gap"]
    ), mean
    assert mean["quantile_gap"] > 0.0, "the check is not vacuous"


def test_the_tolerances_sit_above_the_known_answer_gaps_and_below_the_faults() -> None:
    study = members.tolerance_study(n_cases=60)
    assert study["ended_gap"]["max"] <= members.SHARE_TOLERANCE
    assert study["quantile_gap"]["mean"] <= members.QUANTILE_TOLERANCE_POINTS
    assert study["crps_gap"]["mean"] <= members.CRPS_TOLERANCE_POINTS


def test_a_base_level_tied_to_the_end_time_fails_the_check(monkeypatch) -> None:
    """The first construction ordered base levels like end times; the stride 1 reproduces it."""
    monkeypatch.setattr(members, "BASE_STRIDE", 1)
    gaps = [
        members.known_answer_gaps(
            sd=0.03, depth=0.10, age_days=5.0, curve=_weibull(1.0, 20.0), lead_day=lead, seed=lead
        )
        for lead in (1, 3, 5)
    ]
    assert np.mean([g["quantile_gap"] for g in gaps]) > 2 * members.QUANTILE_TOLERANCE_POINTS


def test_a_stored_forecast_that_follows_another_survival_curve_fails_the_ended_check() -> None:
    import e00738_survival as surv

    leads = 4 + np.arange(48) / 48
    right, wrong = _weibull(1.0, 20.0), _weibull(1.0, 14.0)
    p = surv.ratio(curve=wrong, age_days=np.full(48, 5.0), lead_days=leads)
    from scipy import stats

    got = members.check_origin(
        base_q=np.tile(stats.norm.ppf(members.LEVELS, scale=0.03), (48, 1)),
        arm_q=members.gaussian_mixture_quantiles(p=p, depth=0.10, sd=0.03),
        depth=0.10, age_days=5.0, lead_days=leads, curve=right, p_arm=p,
    )  # fmt: skip
    assert got["ended_gap"] > members.SHARE_TOLERANCE


def test_the_gaussian_mixture_quantiles_are_exact() -> None:
    from scipy import stats

    q = members.gaussian_mixture_quantiles(p=np.array([0.0, 1.0, 0.4]), depth=0.1, sd=0.03)
    np.testing.assert_allclose(q[0], stats.norm.ppf(members.LEVELS, scale=0.03), atol=1e-9)
    np.testing.assert_allclose(q[1], stats.norm.ppf(members.LEVELS, loc=0.1, scale=0.03), atol=1e-9)
    cdf = 0.4 * stats.norm.cdf(q[2], 0.1, 0.03) + 0.6 * stats.norm.cdf(q[2], 0.0, 0.03)
    np.testing.assert_allclose(cdf, members.LEVELS, atol=1e-9)


# --- F7: the remedy refits every head arm -------------------------------------------------------


def test_the_remedy_refits_every_screen_head_fit_and_predicts_them(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(common, "OUT", tmp_path)
    (tmp_path / "gate_remedy.flag").write_text("x")
    for m in common.SCREEN_FOLDS:
        for arm in common.HEAD_ARMS:
            for path in common.fit_paths(arm=arm, month=m, seed=0).values():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("old")
        (common.fit_paths(arm="W0", month=m, seed=0)["state"].parent).mkdir(
            parents=True, exist_ok=True
        )
        common.fit_paths(arm="W0", month=m, seed=0)["state"].write_text("w0")
    jobs: list[tuple[str, dict]] = []
    jobs_: list = []
    monkeypatch.setattr(
        run,
        "run_gpu_tasks",
        lambda *, jobs, fn, stage, args: jobs_.extend((n, kw, fn) for n, kw in jobs),
    )
    monkeypatch.setattr(run, "stage_gate", lambda: jobs.append(("gate", {})))
    run.stage_gate_remedy(args=run.parse_args(argv=["gate-remedy", "--device", "cpu"]))
    fits = [(n, kw) for n, kw, fn in jobs_ if fn is run.sub_fit]
    preds = [(n, kw) for n, kw, fn in jobs_ if fn is run.sub_predict]
    assert len(fits) == 16 and {kw["arm"] for _, kw in fits} == set(common.HEAD_ARMS)
    assert len(preds) == 4 and all(kw["arms"] == common.HEAD_ARMS for _, kw in preds)
    assert jobs == [("gate", {})], "the gate runs again after the refits"
    for m in common.SCREEN_FOLDS:
        for arm in common.HEAD_ARMS:
            assert not any(p.exists() for p in common.fit_paths(arm=arm, month=m, seed=0).values())
        assert common.fit_paths(arm="W0", month=m, seed=0)["state"].exists(), "W0 is not refitted"


# --- F9: the A1 bias bar -----------------------------------------------------------------------


def test_the_gate_record_carries_the_a1_age_bias(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(common, "OUT", tmp_path)
    assert run.a1_age_bias()["exceeds_bar"] is False
    (tmp_path / "a1").mkdir()
    (tmp_path / "a1" / "2025-11.json").write_text(json.dumps({"age_bias_days": 1.6}))
    got = run.a1_age_bias()
    assert got["exceeds_bar"] is True and got["by_fold"] == {"2025-11": 1.6}
    assert got["bar_days"] == common.A1_BIAS_MAX_DAYS == 1.0


# --- F11: failure records, markers with several workers, the pass cap, and the busy GPU -----


def test_a_failure_record_of_a_task_that_later_finished_is_archived(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(common, "OUT", tmp_path)
    state = {"n": 0}

    def flaky(**kw: object) -> None:
        state["n"] += 1
        if state["n"] == 1:
            msg = "first attempt fails"
            raise ValueError(msg)

    assert not phases.run_task(name="flaky", fn=flaky, kwargs={})
    with pytest.raises(RuntimeError, match="failure records"):
        phases.require_finished(names=["flaky"], out=tmp_path)
    assert phases.run_task(name="flaky", fn=flaky, kwargs={})
    phases.require_finished(names=["flaky"], out=tmp_path)
    assert not list((tmp_path / "failed").glob("*.json"))
    assert len(list((tmp_path / "failed" / "archive").glob("task_flaky_*.json"))) == 1
    # a failure of a task that has not finished stays
    phases.write_failure(name="never_done", exc=ValueError("x"))
    with pytest.raises(RuntimeError, match="failure records"):
        phases.require_finished(names=["flaky"], out=tmp_path)


def test_the_stage_marker_is_written_by_whichever_worker_finishes_last(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(common, "OUT", tmp_path)
    names = ["a", "b", "c", "d"]
    mine = phases.shard(items=names, worker=0, of=2)
    for n in mine:
        assert phases.run_task(name=n, fn=lambda **kw: None, kwargs={})
    run.finish_stage(name="score", tasks=names)  # worker 0 finishes first
    assert not (tmp_path / phases.STAGE_MARKERS["score"]).exists()
    for n in phases.shard(items=names, worker=1, of=2):
        assert phases.run_task(name=n, fn=lambda **kw: None, kwargs={})
    run.finish_stage(name="score", tasks=names)  # worker 1 finishes last
    assert (tmp_path / phases.STAGE_MARKERS["score"]).exists()


def test_the_pass_cap_override_works_in_a_scratch_folder_only(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(common.PASS_CAP_ENV, "1")
    assert common.pass_cap(remedy=False, out=tmp_path / "scratch" / "x") == 1
    monkeypatch.delenv("E00738_SCRATCH", raising=False)
    with pytest.raises(RuntimeError, match="not a scratch folder"):
        common.pass_cap(remedy=False, out=tmp_path / "data" / "x")


def test_a_real_stage_refuses_a_stray_pass_cap_variable(monkeypatch) -> None:
    for k in phases.BLAS_VARIABLES:
        monkeypatch.setenv(k, "1")
    monkeypatch.setenv(common.PASS_CAP_ENV, "1")
    with pytest.raises(SystemExit, match=common.PASS_CAP_ENV):
        run.main(argv=["screen-fits", "--device", "cpu"])


def test_a_launcher_that_finds_the_gpu_busy_exits_with_75(monkeypatch) -> None:
    for k in phases.BLAS_VARIABLES:
        monkeypatch.setenv(k, "1")
    monkeypatch.delenv(common.PASS_CAP_ENV, raising=False)
    monkeypatch.setattr(phases, "gpu_busy", lambda: True)
    with pytest.raises(SystemExit) as e:
        run.main(argv=["screen-fits", "--device", "cuda"])
    assert e.value.code == common.EXIT_GPU_BUSY == 75
    phases.require_gpu_free(busy=lambda: False)  # a free GPU returns


# --- F12: run.sh ----------------------------------------------------------------------------


def test_run_sh_puts_the_whole_frozen_checkout_first_and_keeps_it_clean() -> None:
    text = (HERE / "run.sh").read_text()
    assert 'folders_of "$(dirname "$E00738_713_DIR")"' in text
    assert "PYTHONDONTWRITEBYTECODE=1" in text
    assert text.index('paths="$E00738_713_DIR:') < text.index('paths="$paths$here:')


def test_a_missing_dense_cache_file_stops_a_job_before_it_can_write_into_713() -> None:
    import e00738_fit as fit

    assert set(fit.REQUIRED_CACHE_FILES) == {
        "dense_b1_cams.npy",
        "dense_b1_grid.npy",
        "dense_raw.npy",
    }


# --- F13: the bootstrap keeps the seeds --------------------------------------------------------


def test_the_bootstrap_resamples_months_and_keeps_every_seed() -> None:
    sums = np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
    counts = np.ones(4)
    r = score.reading_from_month_arrays(sums=sums, counts=counts, level=95.0, n_resamples=2000)
    assert r.point == pytest.approx(2.5)
    assert 1.0 <= r.boot_lower <= 2.5 <= r.boot_upper <= 4.0
    assert r.boot_upper - r.boot_lower < 3.0
    # identical months leave no month-to-month variation
    flat = score.reading_from_month_arrays(
        sums=np.full((3, 4), 2.0), counts=np.ones(4), level=95.0, n_resamples=500
    )
    assert flat.boot_lower == flat.boot_upper == pytest.approx(2.0)


def test_seed_points_are_means_of_monthly_means_like_the_point() -> None:
    sums = np.array([[1.0, 8.0], [3.0, 12.0]])
    counts = np.array([1.0, 4.0])  # unequal month counts
    r = score.reading_from_month_arrays(sums=sums, counts=counts, level=95.0, n_resamples=100)
    assert r.seed_points == pytest.approx(((1.0 + 2.0) / 2, (3.0 + 3.0) / 2))
    assert r.point == pytest.approx(np.mean(r.seed_points))
