"""Tests of the job list, the fit loop, the continuation, the budget, and the verdict report. No
data, no GPU."""

import json
from pathlib import Path

import e00721_core as core
import e00721_fit as fit
import e00721_report as report
import numpy as np
import pytest


class _CurveFit:
    """A trainable whose validation loss follows a stored curve, to test the loop without a
    network."""

    def __init__(self, curve: list[float]) -> None:
        self.curve = curve
        self.i = 0

    def train_pass(self) -> None:
        self.i += 1

    def validate(self) -> float:
        return self.curve[min(self.i - 1, len(self.curve) - 1)]

    def get_state(self) -> dict:
        return {"pass": self.i}

    def set_state(self, state: dict) -> None:
        self.i = state["pass"]


def _falling_after_stall() -> list[float]:
    rng = np.random.default_rng(0)
    descent = 1.0 - 0.02 * np.arange(1, 21)
    stalled = 0.61 + np.abs(rng.normal(0, 0.004, 12))
    falling = 0.6 - 0.01 * np.arange(1, 120)
    return [*descent, *stalled, *falling]


def test_patience_stops_and_returns_the_best_state() -> None:
    curve = [1.0, 0.9, 0.8] + [0.85] * 20
    hist, best_pass, state = fit.fit_patience(fit=_CurveFit(curve), max_passes=100, patience=8)
    assert best_pass == 2
    assert len(hist) == 11
    assert state == {"pass": 3}


def test_cap_stops_a_fit_that_keeps_improving() -> None:
    hist, best_pass, _ = fit.fit_patience(
        fit=_CurveFit(list(1.0 - 0.01 * np.arange(200))), max_passes=60, patience=8
    )
    assert len(hist) == 60
    assert best_pass == 59


def test_continuation_flags_and_keeps_the_patience_8_state_for_scoring() -> None:
    result = fit.run_fit_with_flag(fit=_CurveFit(_falling_after_stall()), continuation=True)
    assert result.continued
    assert result.flag is not None
    assert result.flag["flagged"]
    assert result.best_pass_p8 == 19
    assert result.best_state_p8 == {"pass": 20}
    assert result.best_state_p30 != result.best_state_p8
    assert result.stop_pass_p8 < len(result.val_history) - 1
    assert len(result.val_history) <= core.CONTINUATION_CAP_PASSES


def test_the_continuation_never_exceeds_150_passes_in_total() -> None:
    result = fit.run_fit_with_flag(
        fit=_CurveFit(list(1.0 - 0.001 * np.arange(400))), continuation=True
    )
    assert len(result.val_history) == core.CONTINUATION_CAP_PASSES


def test_a_capped_stage_goes_on_to_120_in_one_run_and_is_not_trained_twice() -> None:
    trainable = _CurveFit(list(1.0 - 0.001 * np.arange(400)))
    result = fit.run_fit_with_flag(fit=trainable, continuation=False)
    assert result.rerun_of_cap
    assert len(result.val_history) == core.REAL_FOLD_CAP_RERUN
    assert trainable.i == core.REAL_FOLD_CAP_RERUN  # each pass trained once


def test_no_continuation_means_no_flag() -> None:
    result = fit.run_fit_with_flag(fit=_CurveFit(_falling_after_stall()), continuation=False)
    assert not result.continued
    assert result.flag is None


def test_the_floor_is_tallied_from_stored_curve_files_across_processes(tmp_path: Path) -> None:
    folder = fit.data_root(data_dir=tmp_path) / "fits"
    folder.mkdir(parents=True)
    for i, hours in enumerate((0.3, 0.35)):
        body = {"hours": 1.0, "continuation_hours": hours}
        (folder / f"job{i}.curve.json").write_text(json.dumps(body))
    budget = fit.Budget.from_files(data_dir=tmp_path)
    assert budget.spent_hours == pytest.approx(2.0)
    assert budget.continuation_hours == pytest.approx(0.65)
    assert budget.floor_reached()
    with pytest.raises(fit.ContinuationFloorReachedError):
        budget.check_before_continued_fit()
    assert not fit.Budget(continuation_hours=0.59).floor_reached()


def test_the_cap_blocks_a_fit_that_would_pass_14_hours() -> None:
    assert fit.Budget(spent_hours=13.0).may_start_fit(expected_hours=0.5)
    assert not fit.Budget(spent_hours=13.9).may_start_fit(expected_hours=0.5)


def test_waves_hold_the_page_job_counts() -> None:
    assert len(fit.wave_jobs(wave=1)) == 5
    assert len(fit.wave_jobs(wave=3)) == 20
    assert len(fit.wave_jobs(wave=2)) == 19
    assert len(fit.wave_jobs(wave=4)) == 20
    assert sum(len(fit.wave_jobs(wave=w)) for w in (1, 2, 3, 4)) == 64
    keys = [j.key for w in (1, 2, 3, 4) for j in fit.wave_jobs(wave=w)]
    assert len(keys) == len(set(keys))
    with pytest.raises(ValueError, match="wave 5"):
        fit.wave_jobs(wave=5)


def test_every_job_has_its_own_file_names(tmp_path: Path) -> None:
    jobs = [j for w in (1, 2, 3, 4) for j in fit.wave_jobs(wave=w)]
    paths = [fit.job_paths(data_dir=tmp_path, job=j)["done"] for j in jobs]
    assert len(set(paths)) == len(paths)
    assert all(p.is_relative_to(tmp_path) for p in paths)


def test_atomic_write_leaves_no_temporary_file(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b.json"
    fit.atomic_write_json(path=target, body={"x": 1})
    assert json.loads(target.read_text()) == {"x": 1}
    assert [p.name for p in target.parent.iterdir()] == ["b.json"]


def test_resource_rules() -> None:
    assert fit.resource_problems(load=3.0, free_gb=60.0) == []
    assert len(fit.resource_problems(load=30.0, free_gb=10.0)) == 2
    assert fit.resource_problems(load=float("nan"), free_gb=60.0)


def test_report_applies_the_verdict_table(tmp_path: Path) -> None:
    summary = {
        "g1": {"point": -0.6, "upper": -0.2}, "g2": {"point": -0.5, "upper": -0.1},
        "g3_checks": [
            {"reading": "plain", "scope": "pooled", "point": 0.02, "lower": -0.1, "limit": 0.10},
            {"reading": "open", "scope": "D5", "point": 0.05, "lower": -0.1, "limit": 0.20},
        ],
        "flags": [{"evaluated": True, "flagged": False}],
    }  # fmt: skip
    text = report.format_report(summary=summary, mde=0.404)
    assert text.splitlines()[0] == "verdict row 5: supported"
    summary["flags"] = [{"evaluated": False, "flagged": True}]
    summary["g2"] = {"point": -0.1, "upper": 0.1}
    assert (
        report.format_report(summary=summary, mde=0.404).splitlines()[0]
        == "verdict row 7: inconclusive"
    )
    mde_path = tmp_path / "mde.json"
    core.write_mde_file(path=mde_path, seed_difference_points=0.1)
    summary_path = tmp_path / "s.json"
    summary_path.write_text(json.dumps(summary))
    report.main(summary_path=summary_path, mde_path=mde_path)
