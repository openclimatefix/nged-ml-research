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
    # Row 6 is tested before row 7's unreadable bars and unconverged fits.
    assert _row(c2="N", c2_ci_upper=0.01, bars="unreadable") == (6, "not supported")
    assert _row(c2="N", c2_ci_upper=0.01, n_converged_primaries=3) == (6, "not supported")
    assert _row(c2="N", c2_ci_upper=0.01, c1="fail") == (7, "inconclusive")
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
    with pytest.raises(RuntimeError, match="CPU-hours"):
        phases.require_budget()
    (scratch_out / phases.BUDGET_MARKER).write_text("ok\n")
    phases.require_budget()


def test_projection_over_the_stop_refuses(scratch_out: Path) -> None:
    (scratch_out / "unit_timing.json").write_text(
        json.dumps({"code_hash": phases.code_hash(), "projected_total_cpu_hours": 40.0})
    )
    with pytest.raises(RuntimeError, match="projects"):
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


def test_scoring_needs_every_done_marker_of_the_current_code_and_no_failures(
    scratch_out: Path,
) -> None:
    phases.require_finished(names=[])
    phases.run_task(name="a", fn=lambda: None)
    with pytest.raises(RuntimeError, match="1 of 2"):
        phases.require_finished(names=["a", "b"])
    phases.run_task(name="b", fn=lambda: None)
    phases.require_finished(names=["a", "b"])
    (scratch_out / "done" / "b.ok").write_text("code_hash=stale 2026 peak_rss_gb=1\n")
    with pytest.raises(RuntimeError, match="1 of 2"):
        phases.require_finished(names=["a", "b"])
    phases.run_task(name="b", fn=lambda: None)
    phases.require_finished(names=["a", "b"])
    (scratch_out / "failed").mkdir(exist_ok=True)
    (scratch_out / "failed" / "old.json").write_text("{}")
    with pytest.raises(RuntimeError, match="1 failure records"):
        phases.require_finished(names=["a", "b"])


def test_g1_collect_refuses_a_gates_record_of_other_code(
    scratch_out: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import e00737_run as run

    monkeypatch.setattr(run, "OUT", scratch_out)
    monkeypatch.setattr(phases, "OUT", scratch_out)
    with pytest.raises(SystemExit, match="does not exist"):
        run.stage_g1_collect()
    (scratch_out / "gates.json").write_text(json.dumps({"code_hash": "stale"}))
    with pytest.raises(SystemExit, match="different code"):
        run.stage_g1_collect()
    assert json.loads((scratch_out / "gates.json").read_text()) == {"code_hash": "stale"}


def test_worst_case_timing_scales_the_fit_times_to_30_iterations() -> None:
    import e00737_run as run

    timing = {"load": 10.0, "control": 100.0, "world": 200.0, "fit_a": 20.0}
    worst = run.worst_case_timing(timing=timing, iterations=10)
    assert worst == {"load": 10.0, "control": 300.0, "world": 600.0, "fit_a": 60.0}
    assert run.worst_case_timing(timing=timing, iterations=30) == timing


def test_ridge_draws_use_seeds_that_differ_from_every_other_seed() -> None:
    seeds = set(range(common.RIDGE_SEED, common.RIDGE_SEED + common.RIDGE_DRAWS))
    assert (min(seeds), max(seeds)) == (737200, 737229)
    g1_seeds = {common.G1_SEED + d + h for d in range(common.G1_DRAWS) for h in (0, 100_000)}
    assert seeds.isdisjoint(g1_seeds)
    others: set[int] = set(g1_seeds)
    for name in dir(common):
        value = getattr(common, name)
        if name.endswith(("_SEED", "_SEEDS")) and name not in ("RIDGE_SEED", "G1_SEED"):
            others |= set(value) if isinstance(value, tuple) else {value}
    assert {*common.TUNING_SEEDS, *common.MAIN_SEEDS, *common.MATCHED_SEEDS} <= others
    assert common.SPECIAL_SEED in others
    assert seeds.isdisjoint(others)
    assert 1.0 in common.RIDGE_SCALE_GRID
    assert common.RIDGE_SCALE_GRID[0] == max(common.RIDGE_SCALE_GRID)  # a tie goes to the first


def test_tuning_gates_leave_out_g1_and_g1b_but_the_bulk_needs_them(scratch_out: Path) -> None:
    record: dict[str, object] = {g: {"passed": True} for g in ("G2", "G3", "G4", "attributes")}
    record |= {"known_answers": {"passed": True}, "quarters": {"passed": True}}
    record["code_hash"] = phases.code_hash()
    (scratch_out / "gates.json").write_text(json.dumps(record))
    phases.require_gates(required=phases.GATES_BEFORE_TUNING)
    with pytest.raises(SystemExit, match="G1"):
        phases.require_gates()


def test_tuned_record_holds_both_values_and_every_fit_task_carries_them(scratch_out: Path) -> None:
    import e00737_run as run

    path = scratch_out / "tuned.json"
    path.write_text(
        json.dumps({"growth_sd": 0.6, "ridge_scale": 0.01, "code_hash": phases.code_hash()})
    )
    with pytest.raises(SystemExit, match=r"tuned\.sha256"):
        phases.require_tuned()
    digest = scratch_out / phases.TUNED_SHA_NAME
    digest.write_text(f"{common.sha256_of(path=path)}  tuned.json\n")
    tuned = phases.require_tuned()
    assert tuned == common.Tuned(growth_sd=0.6, ridge_scale=0.01)
    path.write_text(
        json.dumps({"growth_sd": 0.15, "ridge_scale": 0.01, "code_hash": phases.code_hash()})
    )
    with pytest.raises(SystemExit, match=r"tuned\.sha256"):
        phases.require_tuned()
    path.write_text(json.dumps({"growth_sd": 0.6, "code_hash": "stale"}))
    with pytest.raises(SystemExit, match="different code"):
        phases.require_tuned()
    path.write_text(
        json.dumps({"growth_sd": 0.6, "ridge_scale": 0.01, "code_hash": phases.code_hash()})
    )
    tasks = run.fit_tasks(primaries=[common.PRIMARIES[0]], store=scratch_out, tuned=tuned)
    assert tasks
    for name, kwargs in tasks:
        assert kwargs["growth_sd"] == 0.6, name
        assert kwargs["ridge_scale"] == 0.01, name


def test_tune_collect_picks_the_ridge_scale_first_and_then_the_growth_sd(
    scratch_out: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import e00737_run as run

    monkeypatch.setattr(run, "OUT", scratch_out)

    def finish_ridge_tasks() -> None:
        for d in range(common.RIDGE_DRAWS):
            phases.run_task(name=f"tune_ridge_{d:03d}", fn=lambda: None)

    folder = scratch_out / "tuning" / "ridge"
    folder.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for d in range(common.RIDGE_DRAWS):
        # Scale 1 gives the largest error and the smallest scale the lowest, in every draw.
        errors = np.array([0.30, 0.10, 0.05]) + rng.uniform(0.0, 0.01, size=3)
        np.savez(folder / f"draw_{d:03d}.npz", error=errors)
    with pytest.raises(RuntimeError, match="30 of 30"):
        run.tune_ridge_collect()
    finish_ridge_tasks()
    record = run.tune_ridge_collect()
    assert record["ridge_scale"] == 0.01
    assert record["seeds"] == [737200, 737229]
    truth = np.linspace(0.0, 4.0, common.N_QUARTERS)
    scored = np.ones(common.N_QUARTERS, dtype=bool)
    growth_error = [0.3, 0.2, 0.05]  # the largest growth sd is the best
    for site in common.PRIMARIES:
        folder = scratch_out / "tuning" / site
        folder.mkdir(parents=True)
        np.savez(folder / "growth_control.npz", cap=np.zeros((3, common.N_QUARTERS)), scored=scored)
        for spec in worlds.tuning_specs(primary=site):
            # An estimate of truth x (1 - e) has a path error of exactly e.
            cap = np.array([truth * (1 - e) for e in growth_error])
            np.savez(folder / f"growth_{spec.name}.npz", cap=cap, truth=truth)
    with pytest.raises(RuntimeError, match="126 of 126"):
        run.tune_collect()
    for name, _ in run.tune_tasks(
        primaries=list(common.PRIMARIES),
        store=scratch_out,
        part="growth",
        ridge_scale=0.01,
        worker=0,
        of=1,
    ):
        phases.run_task(name=name, fn=lambda: None)
    tuned = run.tune_collect()
    assert tuned["ridge_scale"] == 0.01
    assert tuned["growth_sd"] == 0.60
    assert tuned["median_path_error"] == pytest.approx(growth_error)
    assert json.loads((scratch_out / "tuned.json").read_text())["ridge_scale"] == 0.01
    digest = (scratch_out / phases.TUNED_SHA_NAME).read_text().split()[0]
    assert digest == common.sha256_of(path=scratch_out / "tuned.json")
    assert phases.require_tuned() == common.Tuned(growth_sd=0.60, ridge_scale=0.01)


def test_ridge_tie_goes_to_the_larger_scale(
    scratch_out: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import e00737_run as run

    monkeypatch.setattr(run, "OUT", scratch_out)
    folder = scratch_out / "tuning" / "ridge"
    folder.mkdir(parents=True)
    for d in range(common.RIDGE_DRAWS):
        np.savez(folder / f"draw_{d:03d}.npz", error=np.array([0.2, 0.1, 0.1]))
        phases.run_task(name=f"tune_ridge_{d:03d}", fn=lambda: None)
    assert run.tune_ridge_collect()["ridge_scale"] == 0.1


def test_ridge_tasks_are_sharded_by_draw_and_growth_tasks_by_primary(scratch_out: Path) -> None:
    import e00737_run as run

    shards = [
        run.tune_tasks(
            primaries=list(common.PRIMARIES[:1]),
            store=scratch_out,
            part="ridge",
            ridge_scale=None,
            worker=w,
            of=3,
        )
        for w in range(3)
    ]
    names = sorted(name for shard in shards for name, _ in shard)
    assert len(names) == common.RIDGE_DRAWS == len(set(names))
    growth = run.tune_tasks(
        primaries=list(common.PRIMARIES[:1]),
        store=scratch_out,
        part="growth",
        ridge_scale=0.1,
        worker=0,
        of=1,
    )
    assert len(growth) == 1 + 20
    assert all(kwargs["ridge_scale"] == 0.1 for _, kwargs in growth)


def test_tuning_makes_468_fits_and_the_projection_counts_them() -> None:
    import e00737_run as run

    # 30 ridge draws at 3 ridge scales, then 120 tuning worlds and 6 real records at 3 growth sds.
    fits = common.RIDGE_DRAWS * len(common.RIDGE_SCALE_GRID) + (
        common.EXPECTED_TUNING_WORLDS + len(common.PRIMARIES)
    ) * len(common.GROWTH_SD_GRID)
    assert fits == 468
    growth = run.tune_tasks(
        primaries=list(common.PRIMARIES),
        store=Path("x"),
        part="growth",
        ridge_scale=0.1,
        worker=0,
        of=1,
    )
    assert len(growth) == len(common.PRIMARIES) * (1 + 20) == 126
    timing = {"load": 0.0, "control": 0.0, "world": 0.0, "fit_a": 3600.0}
    hours = run.project_cpu_hours(timing=timing)
    key = next(k for k in hours if k.startswith("tune"))
    assert hours[key] == pytest.approx(468.0)


def test_the_tuned_stages_refuse_when_ridge_json_is_missing_or_stale(scratch_out: Path) -> None:
    with pytest.raises(SystemExit, match=r"ridge\.json"):
        phases.require_ridge_scale()
    with pytest.raises(SystemExit, match=r"ridge\.json"):
        phases.read_record(name="ridge.json")
    (scratch_out / "ridge.json").write_text(json.dumps({"ridge_scale": 0.1, "code_hash": "stale"}))
    with pytest.raises(SystemExit, match="different code"):
        phases.require_ridge_scale()
    (scratch_out / "ridge.json").write_text(
        json.dumps({"ridge_scale": 0.1, "code_hash": phases.code_hash()})
    )
    assert phases.require_ridge_scale() == 0.1


def test_g1_records_must_hold_the_current_tuned_values_and_code() -> None:
    tuned = common.Tuned(growth_sd=0.3, ridge_scale=0.1)
    good = {"growth_sd": 0.3, "ridge_scale": 0.1, "code_hash": phases.code_hash()}
    gates.check_tuned_record(record=good, tuned=tuned, what="a draw")
    for bad in ({"growth_sd": 0.6}, {"ridge_scale": 0.01}, {"code_hash": "stale"}):
        with pytest.raises(RuntimeError, match="other tuned values or other code"):
            gates.check_tuned_record(record={**good, **bad}, tuned=tuned, what="a draw")


def test_task_keyword_arguments_do_not_collide_with_run_task(tmp_path: Path) -> None:
    import inspect

    import e00737_run as run

    reserved = set(inspect.signature(phases.run_task).parameters) - {"kwargs"}
    tasks = [
        *run.tune_tasks(
            primaries=["D5"], store=tmp_path, part="growth", ridge_scale=0.1, worker=0, of=1
        ),
        *run.tune_tasks(
            primaries=["D5"], store=tmp_path, part="ridge", ridge_scale=None, worker=0, of=1
        ),
        *run.fit_tasks(primaries=["D5"], store=tmp_path, tuned=common.Tuned(0.3, 0.1)),
    ]
    assert tasks
    for task_name, kwargs in tasks:
        assert not set(kwargs) & reserved, task_name
