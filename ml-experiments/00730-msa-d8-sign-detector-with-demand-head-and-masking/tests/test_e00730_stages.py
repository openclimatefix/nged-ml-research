"""Tests of the stage drivers on a tiny synthetic data folder (no real file, no torch)."""

import dataclasses
import json
from pathlib import Path
from typing import Any

import e00729_banks as banks729
import e00729_baselines as base
import e00729_score as score729
import e00730_banks as banks
import e00730_core as core
import e00730_stages as stages
import numpy as np
import pytest
from fixture730 import (
    make_root,
    open_gallery,
    with_unique_keys,
    write_choices,
    write_predictions,
    write_real,
)

ALL_STAGES = (
    "gallery", "pretrain", "timing", "screen", "styles", "later", "refit729", "score", "real",
    "report", "figures", "series", "components", "choose", "margin",
)  # fmt: skip


def fit_invented_baselines(rows: score729.Rows) -> tuple:
    """Fit the climatology and the regression on the rows' own reading (invented)."""
    return (
        base.Climatology.fit(demand=rows.features.r, f=rows.features),
        base.Regression.fit(demand=rows.features.r, f=rows.features),
    )


def lines() -> list[str]:
    return []


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple:
    return make_root(tmp_path, monkeypatch)


def test_every_stage_refuses_before_its_gates(tmp_path: Path) -> None:
    root = tmp_path / "empty"
    root.mkdir()
    for stage in ALL_STAGES:
        with pytest.raises(RuntimeError, match="marker"):
            banks.guard_run(root=root, stage=stage)


def test_fit_stages_need_the_gallery_check_and_then_the_timing_gate(world: tuple) -> None:
    root, _, _ = world
    calls: list[tuple] = []

    def run_one(arm: str, arch: str, seed: int) -> dict:
        calls.append((arm, arch, seed))
        banks.write_json(
            path=stages.fit_dir(root=root)
            / f"{stages.job_name(arm=arm, arch=arch, seed=seed)}.json",
            payload={},
        )
        return {}

    with pytest.raises(RuntimeError, match="panel_check"):
        stages.stage_screen(root=root, run_one=run_one, log=lambda _: None)
    open_gallery(root)
    with pytest.raises(RuntimeError, match="timing"):
        stages.stage_screen(root=root, run_one=run_one, log=lambda _: None)
    assert not calls


def test_timing_gate_stops_the_line_and_opens_the_fits(world: tuple) -> None:
    root, _, _ = world
    open_gallery(root)
    slow = dict.fromkeys(core.ARCHITECTURES, 60.0)
    with pytest.raises(banks.StoppedError):
        stages.stage_timing(
            root=root,
            seconds_per_epoch=slow,
            peak_memory_gb=5.0,
            manager_approved=False,
            log=lambda _: None,
        )
    assert not stages.timing_passed(root=root)
    fast = dict.fromkeys(core.ARCHITECTURES, 0.5)
    stages.stage_timing(
        root=root,
        seconds_per_epoch=fast,
        peak_memory_gb=5.0,
        manager_approved=False,
        log=lambda _: None,
    )
    assert stages.timing_passed(root=root)
    done: list[tuple] = []

    def run_one(arm: str, arch: str, seed: int) -> dict:
        done.append((arm, arch, seed))
        banks.write_json(
            path=stages.fit_dir(root=root)
            / f"{stages.job_name(arm=arm, arch=arch, seed=seed)}.json",
            payload={},
        )
        return {}

    counts = stages.stage_screen(root=root, run_one=run_one, log=lambda _: None)
    assert counts == {"run": 24, "skipped": 0, "failed": 0}
    assert {a for a, _, _ in done} == set(core.SCREEN_ARMS)
    again = stages.stage_screen(root=root, run_one=run_one, log=lambda _: None)
    assert again["skipped"] == 24


def test_memory_over_the_cap_stops_even_with_approval(world: tuple) -> None:
    root, _, _ = world
    open_gallery(root)
    with pytest.raises(banks.StoppedError):
        stages.stage_timing(
            root=root,
            seconds_per_epoch=dict.fromkeys(core.ARCHITECTURES, 0.5),
            peak_memory_gb=15.0,
            manager_approved=True,
            log=lambda _: None,
        )


def test_a_failed_fit_is_recorded_and_the_run_goes_on(world: tuple) -> None:
    root, _, _ = world

    def run_one(arm: str, arch: str, seed: int) -> dict:
        if seed == 0:
            msg = "boom"
            raise ValueError(msg)
        banks.write_json(
            path=stages.fit_dir(root=root)
            / f"{stages.job_name(arm=arm, arch=arch, seed=seed)}.json",
            payload={},
        )
        return {}

    counts = stages.run_jobs(
        root=root,
        jobs=[("S", "bilstm", 0), ("S", "bilstm", 1)],
        run_one=run_one,
        log=lambda _: None,
    )
    assert counts == {"run": 1, "skipped": 0, "failed": 1}
    assert (stages.fit_dir(root=root) / "S_bilstm_s0.failed.json").exists()
    assert "boom" not in (stages.fit_dir(root=root) / "S_bilstm_s0.failed.json").read_text()


def test_job_lists_follow_the_plan() -> None:
    assert len(core.screen_jobs()) == 24
    assert len(core.style_jobs(arch="bilstm")) == 12
    assert len(core.later_jobs(arch="bilstm")) == 12
    assert len(core.margin_fit_jobs(arch="bilstm")) == 2
    assert len(core.refit_jobs()) == 9
    assert core.N_FITS == 24 + 2 + 12 + 12 + 9
    assert stages.arm_alias(arm="SD*", style="W") == "SD-W"
    assert stages.arm_alias(arm="S", style="W") == "S"


def test_gallery_stage_writes_published_and_private_panels_apart(world: tuple) -> None:
    root, _, _ = world
    results = stages.stage_gallery(root=root, shares_real=None, log=lambda _: None)
    assert set(results) <= set(core.PANEL_REQUIREMENTS)
    public = root / "figures" / "gallery"
    private = root / "PRIVATE" / "gallery"
    assert not (public / "G3_private.svg").exists()
    assert all(p.suffix == ".svg" for p in public.glob("*"))
    assert (private / "auto_check.json").exists()
    assert not (private / "panel_check.json").exists()
    for svg in public.glob("*.svg"):
        text = svg.read_text()
        assert "<svg" in text
        assert "2022-" not in text


def test_panel_check_needs_both_the_automatic_check_and_the_design_lead(world: tuple) -> None:
    root, _, _ = world
    stages.stage_gallery(root=root, shares_real=None, log=lambda _: None)
    auto = json.loads((root / "PRIVATE" / "gallery" / "auto_check.json").read_text())["panels"]
    record = stages.record_panel_check(
        root=root, design_lead_confirms=dict.fromkeys(core.PANEL_REQUIREMENTS, True)
    )
    assert record["panels"] == {k: bool(auto.get(k)) for k in core.PANEL_REQUIREMENTS}
    none = stages.record_panel_check(root=root, design_lead_confirms={})
    assert not any(none["panels"].values())
    assert not banks.gallery_passed(root=root)


def test_score_real_report_figures_and_series_on_invented_predictions(
    world: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, holidays, _ = world
    write_choices(root)
    test = banks729.load_bank(path=banks.bank_path(root=root, kind="test"))
    rows = score729.rows_from_segments(segments=test.segments, holidays=holidays)
    write_predictions(root, rows)
    unique = with_unique_keys(rows)
    real_rows = dict.fromkeys(("r1", "k1", "k2", "eras"), unique)
    era = np.array(["net", "solar"])[np.arange(rows.n) % 2]
    baselines = fit_invented_baselines(rows)
    real = stages.stage_real(
        root=root, real_rows=real_rows, era_of_row=era, baselines=baselines, log=lambda _: None
    )
    assert set(real) == {"h2", "k1", "k2", "r2", "r3"}
    assert real["k1"]["status"] in {"pass", "fail", "void"}
    assert {"brier_b1", "log_loss_b1", "demand_mae", "demand_mae_climatology"} <= set(real["k1"])
    assert real["r2"]["share_demand_above_by_era_and_hour"]
    verdict = stages.stage_score(
        root=root, holidays=holidays, n_resamples=200, baselines=baselines, log=lambda _: None
    )
    signed = verdict["signed"]
    assert {"R1_crps", "R1_log_loss", "R2", "R3", "R4", "R5", "R6", "T1", "figure9"} <= set(signed)
    assert set(signed["T1"]) == {"S", "SD*", "D-sign", "A", "D"}
    assert set(signed["figure9"]) == {"P0f", "B1", "A", "D", "D-sign"}
    assert isinstance(signed["R4"]["pass"], bool)
    assert set(verdict["figure2"]) >= {"P0", "S", "SD*"}
    assert set(verdict["figure5"]) == {"S", "SD*"}
    assert 0 <= verdict["row"] <= 8
    assert set(verdict["contrasts"]) <= set(core.CONTRASTS)
    assert "R1" in verdict["report_only"]
    assert set(verdict["causal"]["brier_by_k"]) == {str(k) for k in core.CAUSAL_K} or set(
        verdict["causal"]["brier_by_k"]
    ) == set(core.CAUSAL_K)
    assert set(verdict["shifted_mix"]) == {"S", "SD*"}
    text = stages.stage_report(root=root)
    assert "verdict row" in text
    assert "2022-" not in text
    import e00730_figures as figures

    seen: dict[str, float] = {}
    draw = figures.causal_readout_chart

    def spy(**kwargs: Any) -> Any:
        seen["p0"] = kwargs["p0"]
        return draw(**kwargs)

    monkeypatch.setattr(figures, "causal_readout_chart", spy)
    written = stages.stage_figures(root=root, log=lambda _: None)
    assert written
    assert seen["p0"] == core.read_limits_file(path=root / "limits.json").s  # P0's Brier score
    assert seen["p0"] > 0
    table, windows = stages.series_inputs(root=root, holidays=holidays, real_rows=real_rows)
    assert table.real.any()  # ty: ignore[unresolved-attribute]
    assert len(windows) == len(table.world)  # ty: ignore[unresolved-attribute]
    paths = stages.stage_series(root=root, table=table, windows=windows, log=lambda _: None)
    assert paths
    assert {"S14", "S15"} <= {p.stem for p in paths}
    assert {"brier_by_subset", "signed_crps", "harm_detection", "real_eras"} <= {
        p.stem for p in written
    }
    assert not any(p.parent.name == "S1" for p in paths)
    assert all(p.stem in {f"S{i}" for i in range(1, 17)} for p in paths)


def test_score_refuses_before_h2_is_read(world: tuple) -> None:
    root, holidays, _ = world
    write_choices(root)
    test = banks729.load_bank(path=banks.bank_path(root=root, kind="test"))
    rows = score729.rows_from_segments(segments=test.segments, holidays=holidays)
    write_predictions(root, rows)
    with pytest.raises(RuntimeError, match="real"):
        stages.stage_score(root=root, holidays=holidays, n_resamples=100, log=lambda _: None)
    assert stages.stage_names().index("real") < stages.stage_names().index("score")


def test_a_failed_fit_makes_the_verdict_row_two(world: tuple) -> None:
    root, holidays, _ = world
    write_choices(root)
    test = banks729.load_bank(path=banks.bank_path(root=root, kind="test"))
    rows = score729.rows_from_segments(segments=test.segments, holidays=holidays)
    write_predictions(root, rows)
    write_real(root)
    banks.write_json(
        path=stages.fit_dir(root=root) / "S_bilstm_s0.failed.json", payload={"error": "X"}
    )
    verdict = stages.stage_score(root=root, holidays=holidays, n_resamples=100, log=lambda _: None)
    assert verdict["row"] == 2


def test_the_refit_stage_runs_the_nine_refits_of_729s_arms(world: tuple) -> None:
    root, _, _ = world
    open_gallery(root)
    banks.write_json(path=root / "scores" / "timing.json", payload={"pass": True})
    seen: list[tuple] = []

    def run_one(arm: str, arch: str, seed: int) -> dict:
        seen.append((arm, arch, seed))
        banks.write_json(
            path=stages.fit_dir(root=root)
            / f"{stages.job_name(arm=arm, arch=arch, seed=seed)}.json",
            payload={},
        )
        return {}

    counts = stages.stage_refit729(root=root, run_one=run_one, log=lambda _: None)
    assert counts["run"] == 9
    assert {a for a, _, _ in seen} == {"A", "D", "D-sign"}
    assert {ar for _, ar, _ in seen} == {"bilstm"}


def take_rows(rows: score729.Rows, keep: np.ndarray) -> score729.Rows:
    """Return the rows where `keep` is true, with every array and the baselines' features cut."""
    cut = {
        f.name: getattr(rows, f.name)[keep]
        for f in dataclasses.fields(rows)
        if isinstance(getattr(rows, f.name), np.ndarray)
    }
    features = dataclasses.replace(
        rows.features,
        **{
            f.name: getattr(rows.features, f.name)[keep]
            for f in dataclasses.fields(rows.features)
            if isinstance(getattr(rows.features, f.name), np.ndarray)
        },
    )
    return dataclasses.replace(rows, features=features, **cut)


def test_real_stage_aligns_a_k1_that_lacks_the_half_hours_where_the_record_has_no_row(
    world: tuple,
) -> None:
    root, holidays, _ = world
    write_choices(root)
    test = banks729.load_bank(path=banks.bank_path(root=root, kind="test"))
    rows = score729.rows_from_segments(segments=test.segments, holidays=holidays)
    write_predictions(root, rows)
    unique = with_unique_keys(rows)
    keep = np.ones(rows.n, dtype=bool)
    keep[[3, 4, 100, rows.n - 1]] = False  # half-hours where the record has no row
    k1 = take_rows(unique, keep)
    arm = f"SD-{json.loads((root / 'style.json').read_text())['chosen']}"
    arch = stages.chosen_architecture(root=root)
    for seed in core.ARCH_SEEDS:
        folder = root / "predictions" / stages.job_name(arm=arm, arch=arch, seed=seed)
        np.savez(
            folder / "real_k1.npz",
            p=np.full(k1.n, 0.01),
            demand_mu=np.full(k1.n, 50.0),
        )
    real_rows = {"r1": unique, "k1": k1, "k2": unique, "eras": unique}
    era = np.array(["net", "solar"])[np.arange(rows.n) % 2]
    real = stages.stage_real(
        root=root,
        real_rows=real_rows,
        era_of_row=era,
        baselines=fit_invented_baselines(rows),
        log=lambda _: None,
    )
    assert "demand_mae" in real["k1"]
    assert real["k2"]["n"] == rows.n  # K2's harm check keeps every half-hour
    assert not (root / "scores" / "invalid.json").exists()
