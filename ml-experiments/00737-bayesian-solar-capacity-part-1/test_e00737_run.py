"""Tests of the scoring, the verdict mapping, the run guards, and the attribute check."""
# ruff: noqa: D103, E402

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

for _k in ("NGED_S3_BUCKET_URL", "NGED_S3_BUCKET_ACCESS_KEY", "NGED_S3_BUCKET_SECRET"):
    os.environ.setdefault(_k, "dummy")
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
for _n in ("39", "40", "653", "660"):
    for _d in HERE.parent.glob(f"{_n.zfill(5)}-*"):
        sys.path.append(str(_d))
import json

import e00737_common as common
import e00737_gates as gates
import e00737_phases as phases
import e00737_score as score
import e00737_worlds as worlds
import polars as pl
from e00660_generator import PowerPanel, apply_events


def test_path_error_recovered_share_and_hits() -> None:
    truth = np.array([2.0, 4.0, 6.0, 8.0])
    estimate = np.array([3.0, 4.0, 5.0, np.nan])
    scored = np.array([True, True, True, True])
    # The NaN quarter is left out of both means: mean truth 4, mean absolute gap 2/3.
    assert score.path_error(truth=truth, estimate=estimate, scored=scored) == pytest.approx(
        (2 / 3) / 4.0
    )
    assert score.recovered_share(truth=truth, estimate=estimate, scored=scored) == pytest.approx(
        12.0 / 12.0
    )
    sd = np.array([1.0, 1.0, 0.5, 1.0])
    hits, cells = score.interval_hits(truth=truth, estimate=estimate, sd=sd, scored=scored)
    # 1.28 sd covers a gap of 1.0 at sd 1.0 (1 <= 1.28), but not at sd 0.5 (1 > 0.64).
    assert (hits, cells) == (2, 3)
    assert np.isnan(score.path_error(truth=truth, estimate=estimate, scored=~scored))
    assert np.isnan(score.recovered_share(truth=truth * 0, estimate=truth, scored=scored))


def test_exact_sign_p_known_answers() -> None:
    assert score.exact_sign_p(positives=6, n=6) == pytest.approx(1 / 64)
    assert score.exact_sign_p(positives=5, n=6) == pytest.approx(7 / 64)
    assert score.exact_sign_p(positives=0, n=6) == pytest.approx(1.0)


def test_sign_test_counts_at_the_threshold() -> None:
    positives, p, counts = score.sign_test_counts(medians=np.array([1.0] * 6))
    assert (positives, counts) == (6, True)
    assert p == pytest.approx(0.015625)
    _, _, counts = score.sign_test_counts(medians=np.array([1.0] * 5 + [-1.0]))
    assert counts is False  # 5 of 6 gives p = 0.109


def test_c1_status() -> None:
    assert score.c1_status(medians=np.full(6, 0.2)) == "pass"
    assert score.c1_status(medians=np.full(6, 0.05)) == "fail"  # counts, but under 0.10
    assert score.c1_status(medians=np.full(6, -0.2)) == "reversed"
    assert score.c1_status(medians=np.array([0.3, 0.3, 0.3, 0.3, 0.3, -0.1])) == "fail"


def test_c2_state() -> None:
    assert score.c2_state(medians=np.full(6, -0.1)) == "R"
    assert score.c2_state(medians=np.full(6, 0.08)) == "M"
    assert score.c2_state(medians=np.full(6, 0.02)) == "S"
    assert score.c2_state(medians=np.array([0.2, 0.2, 0.2, 0.2, -0.1, -0.1])) == "N"


def test_count_bar_and_bars_state() -> None:
    assert score.count_bar(passes=5, unreadable=0, need=5) is True
    assert score.count_bar(passes=2, unreadable=1, need=5) is False
    assert score.count_bar(passes=3, unreadable=2, need=5) is None
    assert score.bars_state(bars={"B1": True, "B2": True}) == "pass"
    assert score.bars_state(bars={"B1": True, "B2": None}) == "unreadable"
    assert score.bars_state(bars={"B1": False, "B2": None}) == "fail"


def _row(**changes: object) -> tuple[int, str]:
    inputs: dict = {
        "gates_ok": True,
        "lookahead": False,
        "c1": "pass",
        "c2": "M",
        "bars": "pass",
        "c2_ci_upper": 0.2,
        "n_converged_primaries": 6,
    }
    inputs.update(changes)
    return score.verdict_row(**inputs)


def test_verdict_mapping_covers_every_row() -> None:
    assert _row() == (3, "supported")
    assert _row(bars="fail") == (4, "partly supported")
    assert _row(c2="S") == (5, "small gain")
    assert _row(c2="S", bars="fail") == (6, "not supported")
    assert _row(c2="N", c2_ci_upper=0.01) == (6, "not supported")
    assert _row(c2="N", c2_ci_upper=0.2) == (7, "inconclusive")
    assert _row(c2="R") == (2, "refuted")
    assert _row(c1="reversed") == (2, "refuted")
    assert _row(c1="fail") == (7, "inconclusive")
    assert _row(n_converged_primaries=4) == (7, "inconclusive")
    assert _row(bars="unreadable") == (7, "inconclusive")
    assert _row(gates_ok=False) == (1, "invalid")
    assert _row(lookahead=True, c2="R") == (1, "invalid")


def test_median_of_medians_interval_brackets_the_median() -> None:
    rng = np.random.default_rng(0)
    values = [rng.normal(0.2, 0.05, size=20) for _ in range(6)]
    lo, hi = score.median_of_medians_interval(values=values, draws=500, seed=1)
    assert 0.1 < lo < 0.2 < hi < 0.3


def test_attributes_taken_from_imported_modules_exist() -> None:
    assert gates.missing_attributes() == []


def test_forward_model_known_answers_pass() -> None:
    assert gates.known_answers()["passed"] is True


def test_transfer_addition_equals_apply_events() -> None:
    length = common.GRID_LENGTH
    rng = np.random.default_rng(0)
    power = rng.normal(5.0, 1.0, size=length)
    power[100:110] = np.nan
    windows = [(50, 50 + 96), (1000, 1000 + 96)]
    panel = PowerPanel(
        power={"X": power},
        is_mva={"X": False},
        noise={"X": 1.0},
        neighbours={"X": []},
        origin=common.GRID_START,
    )
    events = pl.DataFrame(
        [
            {
                "site": "X",
                "start": s,
                "end": e,
                "delta": common.TRANSFER_MW,
                "event_class": "step",
                "shape": "flat",
                "shape_source": None,
                "prop_share": 0.0,
                "ramp_in_hh": 0,
                "ramp_out_hh": 0,
                "drift": 0.0,
                "mid_factor": 1.0,
                "staged_hh": 0,
                "residual_frac": 0.0,
                "period_hh": 0.0,
                "phase": 0.0,
            }
            for s, e in windows
        ],
        schema_overrides={"shape_source": pl.String},
    )
    moved = apply_events(panel=panel, events=events)["X"]
    direct = power + worlds.transfer_addition(windows=windows, power=power)
    assert np.allclose(moved, direct, equal_nan=True)


@pytest.fixture
def scratch_out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    out = tmp_path / "scratch"
    out.mkdir()
    monkeypatch.setattr(phases, "OUT", out)
    monkeypatch.setattr(phases, "CODE_SHA", out / "code.sha256")
    monkeypatch.setattr(common, "OUT", out)
    return out


def test_code_hash_changes_with_a_module_and_with_run_scripts_not_tests(tmp_path: Path) -> None:
    (tmp_path / "e00737_a.py").write_text("x = 1\n")
    (tmp_path / "test_e00737_a.py").write_text("y = 1\n")
    first = phases.code_hash(folder=tmp_path)
    (tmp_path / "test_e00737_a.py").write_text("y = 2\n")
    assert phases.code_hash(folder=tmp_path) == first
    (tmp_path / "e00737_a.py").write_text("x = 2\n")
    second = phases.code_hash(folder=tmp_path)
    assert second != first
    (tmp_path / "run.sh").write_text("echo 1\n")
    assert phases.code_hash(folder=tmp_path) != second


def test_review_and_freeze_guards(scratch_out: Path) -> None:
    with pytest.raises(SystemExit, match="missing"):
        phases.require_review(marker=phases.MARKER_REVIEW)
    (scratch_out / phases.MARKER_REVIEW).write_text("0000 old\n")
    with pytest.raises(SystemExit, match="different code"):
        phases.require_review(marker=phases.MARKER_REVIEW)
    (scratch_out / phases.MARKER_REVIEW).write_text(phases.code_hash() + " now\n")
    phases.require_review(marker=phases.MARKER_REVIEW)
    with pytest.raises(SystemExit, match=r"code\.sha256"):
        phases.require_code_frozen()
    (scratch_out / "code.sha256").write_text(phases.code_hash() + "\n")
    phases.require_code_frozen()


def test_gates_guard_needs_every_gate_and_the_current_hash(scratch_out: Path) -> None:
    record: dict[str, object] = {
        g: {"passed": True} for g in (*phases.GATES_REQUIRED, "attributes")
    }
    record |= {"known_answers": {"passed": True}, "quarters": {"passed": True}}
    record["code_hash"] = phases.code_hash()
    path = scratch_out / "gates.json"
    path.write_text(json.dumps(record))
    phases.require_gates()
    record["G3"] = {"passed": False}
    path.write_text(json.dumps(record))
    with pytest.raises(SystemExit, match="G3"):
        phases.require_gates()
    record["G3"] = {"passed": True}
    record["code_hash"] = "stale"
    path.write_text(json.dumps(record))
    with pytest.raises(SystemExit, match="different code"):
        phases.require_gates()


def test_budget_stop_and_extension_marker(scratch_out: Path) -> None:
    phases.require_budget()
    (scratch_out / "time_log.csv").write_text(
        "fit,a,cpu,2026-01-01T00:00:00,2026-01-02T03:00:00\n"
    )  # 27 hours
    assert phases.spent_cpu_hours() == pytest.approx(27.0)
    with pytest.raises(SystemExit, match="CPU-hours"):
        phases.require_budget()
    (scratch_out / phases.BUDGET_MARKER).write_text("ok\n")
    phases.require_budget()


def test_projection_over_the_stop_refuses(scratch_out: Path) -> None:
    (scratch_out / "unit_timing.json").write_text(
        json.dumps({"code_hash": phases.code_hash(), "projected_total_cpu_hours": 40.0})
    )
    with pytest.raises(SystemExit, match="projects"):
        phases.require_budget(check_projection=True)


def test_done_marker_and_failure_record(scratch_out: Path) -> None:
    calls: list[int] = []

    def good() -> None:
        calls.append(1)

    def bad() -> None:
        raise RuntimeError("boom")

    assert phases.run_task(name="t1", fn=good) is True
    marker = (scratch_out / "done" / "t1.ok").read_text()
    assert marker.startswith(f"code_hash={phases.code_hash()} ")
    assert "peak_rss_gb=" in marker
    assert phases.run_task(name="t1", fn=good) is True
    assert calls == [1]  # the second call skipped
    assert phases.run_task(name="t2", fn=bad) is False
    failure = json.loads((scratch_out / "failed" / "t2.json").read_text())
    assert failure["message"] == "boom"
    assert "RuntimeError" in failure["traceback"]
    assert not (scratch_out / "done" / "t2.ok").exists()
    with pytest.raises(SystemExit) as info:
        phases.run_shard(names_and_kwargs=[("t3", {})], fn=bad)
    assert info.value.code == common.EXIT_FAILED


def test_shard_partitions_the_items() -> None:
    items = list(range(10))
    parts = [phases.shard(items=items, worker=w, of=3) for w in range(3)]
    assert sorted(x for p in parts for x in p) == items


def test_blas_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in phases.BLAS_VARIABLES:
        monkeypatch.setenv(name, "1")
    phases.assert_blas_pinned()
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    with pytest.raises(SystemExit):
        phases.assert_blas_pinned()


def test_scratch_redirect_needs_a_guard(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("E00737_OUT", str(tmp_path / "elsewhere"))
    monkeypatch.delenv("E00737_SCRATCH", raising=False)
    with pytest.raises(RuntimeError, match="scratch"):
        common._output_folder()
    monkeypatch.setenv("E00737_SCRATCH", "1")
    assert common._output_folder() == tmp_path / "elsewhere"
    monkeypatch.delenv("E00737_SCRATCH")
    monkeypatch.setenv("E00737_OUT", str(tmp_path / "scratch" / "x"))
    assert common._output_folder() == tmp_path / "scratch" / "x"
