# ruff: noqa: ANN202, PERF401, PLW0108
"""Tests of the scorer's contrasts, stages, and gate marker on synthetic losses (no torch)."""

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import e00711_arms as arms
import e00711_gate as gate
import e00711_jobs as jobs
import e00711_score as score
import numpy as np
import polars as pl
import pytest

from studies import bootstrap, experiment_store

PASSED_CHECKS = {
    "D1p minus DF the six": True,
    "D1p minus DF the eight": True,
    "P1p minus PF": True,
}
MONTHS = ("2025-12", "2026-01", "2026-02", "2026-03")


def _losses(
    *, seeds: tuple[int, ...], shift: float, months: tuple[str, ...] = MONTHS
) -> pl.DataFrame:
    """Return per-seed losses of one Primary demand arm on 3 runs a month."""
    rng = np.random.default_rng(3)
    rows = []
    for month in months:
        year, number = (int(p) for p in month.split("-"))
        for day in (1, 2, 3):
            origin = datetime(year, number, day, 9, tzinfo=UTC)
            for step in range(1, 21):
                base = float(rng.normal(1.0, 0.1))
                for seed in seeds:
                    rows.append(
                        {
                            "time_series_id": 1,
                            "power_fcst_init_time": origin,
                            "valid_time": datetime(
                                year, number, day, 10 + step // 2, 30 * (step % 2), tzinfo=UTC
                            ),
                            "seed": seed,
                            "crps": base + shift + 0.01 * seed,
                            "mae_median": base,
                            "label": "D5",
                            "fold": month,
                            "lead_hours": float(step),
                            "regime": "sunny",
                        }
                    )
    return pl.DataFrame(rows).with_columns(
        pl.col("seed").cast(pl.Int16),
        pl.col("crps").cast(pl.Float32),
        pl.col("mae_median").cast(pl.Float32),
        pl.col("lead_hours").cast(pl.Float32),
        pl.col("label").cast(pl.Categorical),
        pl.col("fold").cast(pl.Categorical),
        pl.col("regime").cast(pl.Categorical),
    )


def _arm_losses(*, seeds: tuple[int, ...], shift: float, months: tuple[str, ...] = MONTHS):
    frame = _losses(seeds=seeds, shift=shift, months=months)
    return score.ArmLosses(
        losses=frame,
        averaged=frame.filter(pl.col("seed") == seeds[0]).with_columns(
            seed=pl.lit(-1).cast(pl.Int16)
        ),
        coverage={},
        cells=frozenset((m, s) for m in months for s in seeds),
    )


def _source(**held: score.ArmLosses | None) -> score.LossSource:
    return score.LossSource(load=lambda name: held.get(name), keep=3)


def test_contrast_reads_the_shared_seeds_and_the_difference():
    source = _source(
        real=_arm_losses(seeds=(0, 1), shift=-0.5), fake=_arm_losses(seeds=(0, 1, 2), shift=0.0)
    )
    reading = score.read_contrast(
        source=source,
        treatment="real",
        reference="fake",
        label="t",
        scope="s",
        where=None,
        level=99.0,
        n_resamples=300,
    )
    assert reading.seeds == (0, 1)
    assert reading.difference == pytest.approx(-0.5, abs=1e-4)
    assert reading.upper < 0
    assert reading.n_months == 4
    assert reading.label_rows == {"D5": 240}  # 3 runs x 20 half-hours x 4 months, one seed


def test_month_totals_interval_equals_the_row_level_bootstrap():
    real, fake = _losses(seeds=(0, 1), shift=-0.3), _losses(seeds=(0, 1), shift=0.0)
    fake = fake.with_columns(
        crps=pl.col("crps")
        + pl.Series(np.random.default_rng(0).normal(0, 0.1, fake.height)).cast(pl.Float32)
    )
    paired = score.paired_month_sums(a=real, b=fake)
    ours = score.month_total_interval(paired=paired, level=95.0, n_resamples=500)
    frame = experiment_store.losses_for_bootstrap(arm_losses={"real": real, "fake": fake})
    theirs = bootstrap.bootstrap_difference_at_level(
        losses=frame, treatment="real", reference="fake", metric="crps", level=95.0, n_resamples=500
    )
    assert ours == pytest.approx(theirs, abs=1e-6)


def test_rows_that_differ_make_the_reading_incomplete():
    short = _arm_losses(seeds=(0,), shift=0.0)
    short.losses = short.losses.head(short.losses.height - 1)
    source = _source(real=_arm_losses(seeds=(0,), shift=0.0), fake=short)
    with pytest.raises(ValueError, match="same rows"):
        score.read_contrast(
            source=source,
            treatment="real",
            reference="fake",
            label="t",
            scope="s",
            where=None,
            level=99.0,
            n_resamples=50,
        )


def test_decisive_interval_is_the_wider_not_the_union():
    boot, t_int = (-0.10, 0.02), (-0.01, 0.12)
    assert score.wider(boot=boot, t_int=t_int) == t_int
    assert score.wider(boot=boot, t_int=None) == boot
    with pytest.raises(ValueError, match="no interval"):
        score.wider(boot=None, t_int=None)


def test_t_interval_needs_two_months():
    one = _losses(seeds=(0,), shift=0.0, months=("2025-12",))
    paired = score.paired_month_sums(a=one, b=one)
    assert score.t_interval_of_months(paired=paired, level=95.0) is None
    two = _losses(seeds=(0,), shift=0.0, months=MONTHS)
    interval = score.t_interval_of_months(paired=score.paired_month_sums(a=two, b=two), level=95.0)
    assert interval == pytest.approx((0.0, 0.0))


def test_minimum_detectable_effect_needs_both_halves_of_the_seeds():
    full = _arm_losses(seeds=(0, 1, 2, 3, 4, 5), shift=0.0)
    source = _source(p=full, short=_arm_losses(seeds=(0, 1, 2), shift=0.0))
    mde = score.minimum_detectable_effect(
        source=source, arm="p", target="primary", where=None, level=99.0, n_resamples=300
    )
    assert mde is not None
    assert mde >= 0
    assert (
        score.minimum_detectable_effect(
            source=source, arm="short", target="primary", where=None, level=99.0, n_resamples=300
        )
        is None
    )


# --- stages, completeness, and the loader -----------------------------------------------------


def test_stage_a_has_132_jobs_and_a_short_wall_clock():
    stage_a = jobs.plan_jobs(stage="A")
    assert len(stage_a) == jobs.STAGE_A_COUNT == 132
    counts = {}
    for job in stage_a:
        counts[job.arm] = counts.get(job.arm, 0) + 1
    assert counts == {"D0n": 6, "DF": 21, "D1": 21, "D1p": 21, "PF": 21, "P1": 21, "P1p": 21}
    assert all(j.seed in jobs.STAGE_A_SEEDS for j in stage_a)
    low, high = jobs.wall_hours(jobs=stage_a)
    assert 2.0 < low < high < 4.0
    stage_b = jobs.plan_jobs(stage="B")
    assert len(stage_b) == jobs.STAGE_B_COUNT == 946 - 132 - 150
    assert not {j.set_name for j in stage_a} & {j.set_name for j in stage_b}


def test_stage_b1_is_the_150_reduced_stage_b_jobs_on_seeds_0_to_2():
    stage_b1 = jobs.plan_jobs(stage="B1")
    assert len(stage_b1) == jobs.STAGE_B1_COUNT == 150
    counts = {}
    for job in stage_b1:
        counts[job.arm] = counts.get(job.arm, 0) + 1
    assert counts == {"R0": 36, "R1": 36, "R1p": 36, "D0g": 21, "D1g": 21}
    assert all(j.seed in (0, 1, 2) for j in stage_b1)
    names = {j.set_name for j in stage_b1}
    assert not names & {j.set_name for j in jobs.plan_jobs(stage="A")}
    assert not names & {j.set_name for j in jobs.plan_jobs(stage="B")}
    assert len(names) == 150
    # Seeds 3 to 5 of R1 and R1p are not run: they stay in Stage B.
    assert {j.seed for j in jobs.plan_jobs(stage="B") if j.arm == "R1"} >= {3, 4, 5}
    assert jobs.expected_cells(arm="R1", stage="B1") == {
        (m, s) for m in arms.REPLICATE_MONTHS for s in (0, 1, 2)
    }
    assert jobs.expected_cells(arm="D1", stage="B1") == set()


def test_stage_b1_specs_read_c3_and_g1_at_the_planned_level_and_nothing_else():
    specs = score.planned_specs(stage="B1")
    assert [s.key for s in specs] == ["C3 the five", "C3 the eight", "G1 the six", "G1 the eight"]
    assert {(s.treatment, s.reference) for s in specs} == {("R1", "R1p"), ("D1g", "D0g")}
    assert all(s.level == score.PLANNED_LEVEL for s in specs)
    assert score.check_specs(stage="B1") == []
    bounds = score.b1_bound_specs()
    assert {(b.treatment, b.reference) for b in bounds} == {("R1p", "R0"), ("R1", "R0")}
    assert all(b.level == score.CHECK_LEVEL for b in bounds)


def test_stage_b1_verdicts_apply_the_decision_rule_to_each_reading(
    capsys: pytest.CaptureFixture[str],
):
    def reading(key: str, difference: float, half: float) -> score.Reading:
        return score.Reading(
            key=key,
            label=key,
            scope="s",
            difference=difference,
            lower=difference - half,
            upper=difference + half,
            boot=None,
            t_interval=None,
            averaged=None,
            n_half_hours=10,
            n_months=7,
            seeds=(0, 1, 2),
            level=99.44,
            label_rows={},
        )

    out = score.verdicts_b1(
        readings={
            "C3 the five": reading("C3 the five", -0.1, 0.05),
            "C3 the eight": reading("C3 the eight", 0.0, 0.05),
            "G1 the six": reading("G1 the six", 0.2, 0.05),
        }
    )
    assert out == {
        "C3 the five": "supported",
        "C3 the eight": "inconclusive",
        "G1 the six": "refuted",
    }
    assert "G1 the eight: verdict withheld" in capsys.readouterr().out


def test_every_scorer_run_saves_its_output_under_work_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setattr(score.common, "WORK_DIR", tmp_path)
    when = datetime(2026, 11, 20, 8, 30, 5, tzinfo=UTC)
    path = score.scores_path(name="score-B1", now=when)
    assert path == tmp_path / "work" / "scores" / "score-B1_20261120T083005Z.txt"
    saved = score.run_saved(name="score-B1", action=lambda: print("a reading"))
    assert saved.parent == tmp_path / "work" / "scores"
    assert saved.name.startswith("score-B1_")
    assert saved.read_text() == "a reading\n"
    assert "a reading" in capsys.readouterr().out


def test_a_reading_needs_every_planned_cell_of_both_arms():
    held = _arm_losses(seeds=(0, 1, 2), shift=0.0, months=arms.CLOUD_SCORED_MONTHS)
    source = _source(
        D1=held, D1p=_arm_losses(seeds=(0, 1), shift=0.0, months=arms.CLOUD_SCORED_MONTHS), DF=None
    )
    spec = score.planned_specs(stage="A")[0]
    assert spec.key == "C1 the six"
    outcome = score.run_spec(source=source, spec=spec, stage="A", n_resamples=50)
    assert isinstance(outcome, score.Incomplete)
    assert "D1p lacks" in outcome.reason
    absent = score.run_spec(source=_source(), spec=spec, stage="A", n_resamples=50)
    assert isinstance(absent, score.Incomplete)
    assert "no predictions" in absent.reason


def test_stage_a_specs_are_c1_and_s1_and_the_checks_are_never_pooled():
    keys = [s.key for s in score.planned_specs(stage="A")]
    assert keys == ["C1 the six", "C1 the eight", "S1"]
    assert len(score.planned_specs(stage="B")) == 9
    checks = score.check_specs(stage="B")
    assert len(checks) == score.N_CHECKS == 8
    assert all("the six" in c.key or "the eight" in c.key or c.key.startswith("P") for c in checks)
    assert pytest.approx(100 * (1 - 0.05 / 8)) == score.CHECK_LEVEL


def test_loader_failure_of_one_arm_does_not_stop_the_others():
    def load(name: str) -> score.ArmLosses | None:
        if name == "P0":
            msg = "no stored PLP row covers 2025-12"
            raise ValueError(msg)
        if name == "D0":
            raise FileNotFoundError
        return _arm_losses(seeds=(0,), shift=0.0)

    source = score.LossSource(load=load, keep=2)
    assert source.get("P0") is None
    assert "no stored PLP row" in source.errors["P0"]
    assert source.get("D0") is None
    assert source.get("D1") is not None
    reasons = score.missing_for(source=source, arm="P0", months=MONTHS, stage="B")
    assert "no stored PLP row" in reasons[0]


def test_loader_keeps_only_a_few_arms_in_memory():
    calls = []

    def load(name: str) -> score.ArmLosses | None:
        calls.append(name)
        return _arm_losses(seeds=(0,), shift=0.0, months=MONTHS[:1])

    source = score.LossSource(load=load, keep=2)
    for name in ("a", "b", "c", "a"):
        source.get(name)
    assert calls == ["a", "b", "c", "a"]


def test_stage_a_verdicts_use_only_c1_and_s1(capsys: pytest.CaptureFixture[str]):
    def reading(key: str, lower: float, upper: float) -> score.Reading:
        return score.Reading(
            key=key,
            label=key,
            scope="s",
            difference=(lower + upper) / 2,
            lower=lower,
            upper=upper,
            boot=None,
            t_interval=None,
            averaged=None,
            n_half_hours=10,
            n_months=7,
            seeds=(0, 1, 2),
            level=99.44,
            label_rows={"D5": 10},
        )

    readings = {
        "C1 the six": reading("C1 the six", -0.2, -0.1),
        "C1 the eight": reading("C1 the eight", -0.01, 0.01),
        "S1": reading("S1", -0.2, -0.05),
    }
    out = score.verdicts(readings=readings, controls={}, stage="A", checks=PASSED_CHECKS)
    assert out["the six"] == "supported"
    assert out["the eight"] == "not supported"
    assert out["solar"] == "supported"
    assert out["front matter"] == "partly supported"
    assert "Stage A verdict" in capsys.readouterr().out
    withheld = score.verdicts(
        readings={**readings, "S1": score.Incomplete(key="S1", reason="x")},
        controls={},
        stage="A",
        checks=PASSED_CHECKS,
    )
    assert "front matter" not in withheld
    full = score.verdicts(readings=readings, controls={}, stage="B", checks=PASSED_CHECKS)
    assert full == {}


# --- the gate marker and the manifest check ---------------------------------------------------


def test_fits_refuse_without_both_gates_but_the_gate_arm_and_measuring_fit_run(tmp_path: Path):
    marker = tmp_path / "gate.json"
    gate.refuse_without_gate(arm="D0n", measure=False, path=marker)
    gate.refuse_without_gate(arm="D1", measure=True, path=marker)
    with pytest.raises(ValueError, match="fit gate at 2026-01: not run"):
        gate.refuse_without_gate(arm="D1", measure=False, path=marker)
    for kind in gate.GATE_KINDS:
        for month in arms.GATE_MONTHS:
            gate.record(
                kind=kind,
                month=month,
                passed=True,
                detail="ok",
                commit=gate.head_commit(),
                path=marker,
            )
    gate.refuse_without_gate(arm="D1", measure=False, path=marker)
    gate.record(
        kind="predict",
        month=arms.GATE_MONTHS[0],
        passed=False,
        detail="1e-2",
        commit=gate.head_commit(),
        path=marker,
    )
    with pytest.raises(ValueError, match="failed"):
        gate.refuse_without_gate(arm="P1", measure=False, path=marker)
    assert set(json.loads(marker.read_text())) == {"fit", "predict"}


def test_existing_set_is_refused_when_dirty_or_from_another_branch(tmp_path: Path):
    folder = tmp_path / "d1-2026-06-s0"
    folder.mkdir()
    manifest = folder / experiment_store.MANIFEST_NAME
    head = gate.head_commit()
    manifest.write_text(json.dumps({"commit": head, "dirty": False}))
    gate.check_existing_set(folder=folder, head=head)
    manifest.write_text(json.dumps({"commit": head, "dirty": True}))
    with pytest.raises(ValueError, match="uncommitted"):
        gate.check_existing_set(folder=folder, head=head)
    manifest.write_text(json.dumps({"commit": "0" * 40, "dirty": False}))
    with pytest.raises(ValueError, match="not an ancestor"):
        gate.check_existing_set(folder=folder, head=head)
    usable, rejected = gate.usable_sets(names=[folder.name, "missing"], folder=tmp_path)
    assert usable == set()
    assert len(rejected) == 1


def test_a_gate_result_from_another_commit_does_not_open_the_fits(tmp_path: Path):
    marker = tmp_path / "gate.json"
    for kind in gate.GATE_KINDS:
        for month in arms.GATE_MONTHS:
            gate.record(
                kind=kind, month=month, passed=True, detail="ok", commit="0" * 40, path=marker
            )
    with pytest.raises(ValueError, match="code has changed"):
        gate.refuse_without_gate(arm="D1", measure=False, path=marker)
    assert gate.commit_is_current(commit=gate.head_commit())
    assert gate.commit_is_current(commit=gate.FLOOR_COMMIT) in (True, False)


def test_two_gate_commands_at_once_keep_both_results(tmp_path: Path):
    from concurrent.futures import ThreadPoolExecutor

    marker = tmp_path / "gate.json"
    months = [f"2026-{m:02d}" for m in range(1, 13)]
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(
            pool.map(
                lambda m: gate.record(
                    kind="predict", month=m, passed=True, detail=m, commit="x", path=marker
                ),
                months,
            )
        )
    assert set(gate.read_marker(path=marker)["predict"]) == set(months)


def test_a_set_older_than_the_floor_commit_is_refused(tmp_path: Path):
    folder = tmp_path / "d1-2026-06-s0"
    folder.mkdir()
    head = gate.head_commit()
    parent = subprocess.run(
        ["git", "rev-parse", f"{gate.FLOOR_COMMIT}~1"], check=True, capture_output=True, text=True
    ).stdout.strip()
    (folder / experiment_store.MANIFEST_NAME).write_text(
        json.dumps({"commit": parent, "dirty": False})
    )
    with pytest.raises(ValueError, match="older than the floor"):
        gate.check_existing_set(folder=folder, head=head)
    (folder / experiment_store.MANIFEST_NAME).write_text(
        json.dumps({"commit": gate.FLOOR_COMMIT, "dirty": False})
    )
    gate.check_existing_set(folder=folder, head=head)


def test_the_scorer_does_not_read_a_set_that_the_manifest_check_rejects(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(score.common, "WORK_DIR", tmp_path)
    folder = tmp_path / "predictions" / arms.set_name(arm="D1", month="2026-06", seed=0)
    folder.mkdir(parents=True)
    (folder / experiment_store.MANIFEST_NAME).write_text(
        json.dumps({"commit": gate.head_commit(), "dirty": True})
    )
    score.REJECTED_SETS.clear()
    assert not score.available(arm="D1", month="2026-06", seed=0)
    assert "uncommitted" in score.REJECTED_SETS["d1-2026-06-s0"]
    assert not score.available(arm="D1", month="2026-05", seed=0)


def test_the_interim_minimum_detectable_effect_needs_only_seeds_0_and_1():
    source = _source(
        p=_arm_losses(seeds=(0, 1, 2), shift=0.0), one=_arm_losses(seeds=(0,), shift=0.0)
    )
    mde = score.minimum_detectable_effect(
        source=source, arm="p", target="interim", where=None, level=99.0, n_resamples=200
    )
    assert mde is not None
    assert mde >= 0
    assert (
        score.minimum_detectable_effect(
            source=source, arm="one", target="interim", where=None, level=99.0, n_resamples=200
        )
        is None
    )


def test_a_failed_or_incomplete_placebo_check_overrides_c1_and_s1():
    def reading(key: str) -> score.Reading:
        return score.Reading(
            key=key,
            label=key,
            scope="s",
            difference=-0.2,
            lower=-0.3,
            upper=-0.1,
            boot=None,
            t_interval=None,
            averaged=None,
            n_half_hours=10,
            n_months=7,
            seeds=(0, 1, 2),
            level=99.44,
            label_rows={"D5": 10},
        )

    readings = {k: reading(k) for k in ("C1 the six", "C1 the eight", "S1")}
    failed = {**PASSED_CHECKS, "D1p minus DF the six": False, "P1p minus PF": False}
    out = score.verdicts(readings=readings, controls={}, stage="A", checks=failed)
    assert out["the six"] == "inconclusive"
    assert out["the eight"] == "supported"
    assert out["solar"] == "inconclusive"
    incomplete = {**PASSED_CHECKS, "D1p minus DF the eight": None}
    out = score.verdicts(readings=readings, controls={}, stage="A", checks=incomplete)
    assert "the eight" not in out
    assert "front matter" not in out
    assert "front matter" not in score.verdicts(readings=readings, controls={}, stage="A")


def test_bounds_are_the_real_arm_against_no_cloud():
    keys = [b.key for b in score.bound_specs()]
    assert keys == ["D1 minus DF the six", "D1 minus DF the eight", "P1 minus PF"]


def _solar_arm(*, shift_by_month: dict[str, float]) -> score.ArmLosses:
    months = arms.SOLAR_MONTHS
    frame = _losses(seeds=(0, 1), shift=0.0, months=months).with_columns(
        label=pl.lit("A").cast(pl.Categorical)
    )
    shifts = pl.col("fold").cast(pl.String).replace_strict(shift_by_month, return_dtype=pl.Float32)
    frame = frame.with_columns(crps=pl.col("crps") + shifts)
    return score.ArmLosses(
        losses=frame,
        averaged=frame.filter(pl.col("seed") == 0).with_columns(seed=pl.lit(-1).cast(pl.Int16)),
        coverage={},
        cells=frozenset((m, s) for m in months for s in (0, 1)),
    )


def test_exploratory_per_fold_solar_rows_use_their_own_month_and_carry_no_verdict(
    capsys: pytest.CaptureFixture[str],
):
    months = arms.SOLAR_MONTHS
    shifts = {m: 0.1 * (i + 1) for i, m in enumerate(months)}
    source = _source(
        P1=_solar_arm(shift_by_month=shifts),
        P1p=_solar_arm(shift_by_month=dict.fromkeys(months, 0.0)),
    )
    score.exploratory(source=source)
    lines = [
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("P1 minus P1p") and "unavailable" not in line
    ]
    folds = [line for line in lines if ", fold " in line]
    assert len(folds) == len(months)
    for line, month, shift in zip(folds, months, shifts.values(), strict=True):
        assert f"fold {month}" in line
        assert f"{shift:+.3f}" in line
        assert "months 1" in line
        assert "range of 2 seeds" in line
    total = [line for line in lines if ", fold " not in line and "months 7" in line]
    assert total, "an all-fold row must still read seven months"
    for line in lines:
        assert line.endswith("-> exploratory")
        for word in ("supported", "refuted", "inconclusive", "small gain"):
            assert word not in line
