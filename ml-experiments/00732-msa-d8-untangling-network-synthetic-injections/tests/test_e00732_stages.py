# ruff: noqa: E501
"""Tests of the guard, the banks check, the limits, and the overlays stage on an invented copy of
experiment 730's folder.
"""

import json
import stat
from typing import Any

import e00732_core as core
import e00732_leakage as leak
import e00732_stages as stages
import fake730
import numpy as np
import pytest
from world732 import seed_neighbour_files


@pytest.fixture
def folders(tmp_path: Any, monkeypatch: Any):
    root730 = tmp_path / "730"
    world = fake730.build(root=root730)
    fake730.patch_constants(monkeypatch, root=root730, world=world)
    root = tmp_path / "732"
    root.mkdir()
    seed_neighbour_files(root=root, base=world["base"], arrays=world["neighbours"])
    return root, root730, world


def test_banks_stage_verifies_records_once_and_marks_the_run(folders: Any) -> None:
    root, root730, world = folders
    record = stages.stage_banks(
        root=root, root730=root730, holidays=world["cal"].holidays, log=print
    )
    assert record["n_rows"] == world["n"]
    assert (root / "run_marker.json").exists()
    again = stages.stage_banks(
        root=root, root730=root730, holidays=world["cal"].holidays, log=print
    )
    assert again == record


def test_banks_stage_stops_when_a_stored_file_changes_after_the_first_record(folders: Any) -> None:
    root, root730, world = folders
    stages.stage_banks(root=root, root730=root730, holidays=world["cal"].holidays, log=print)
    path = root730 / "predictions" / "SD-R_unet_s1" / "test.npz"  # the canary reads arm S only
    with np.load(path) as z:
        arrays = {k: z[k].copy() for k in z.files}
    arrays["p"][0] = 0.5 if arrays["p"][0] != 0.5 else 0.4
    np.savez(path, **arrays)
    with pytest.raises(stages.StoppedError, match="differ from their recorded digests"):
        stages.stage_banks(root=root, root730=root730, holidays=world["cal"].holidays, log=print)


def test_banks_stage_fails_the_canary_and_the_row_count(folders: Any, monkeypatch: Any) -> None:
    root, root730, world = folders
    monkeypatch.setattr(core, "BRIER_CANARY", core.BRIER_CANARY + 0.01)
    with pytest.raises(leak.CorrectnessFailureError, match="canary"):
        stages.stage_banks(root=root, root730=root730, holidays=world["cal"].holidays, log=print)
    monkeypatch.setattr(core, "BRIER_CANARY", core.BRIER_CANARY - 0.01)
    monkeypatch.setattr(core, "N_TEST_ROWS", world["n"] + 1)
    with pytest.raises(leak.CorrectnessFailureError, match="row count"):
        stages.stage_banks(root=root, root730=root730, holidays=world["cal"].holidays, log=print)


def test_banks_stage_stops_on_a_changed_bank_digest(folders: Any) -> None:
    root, root730, world = folders
    summary = json.loads((root730 / "banks" / "summary.json").read_text())
    summary["digests"]["test"] = "0" * 64
    (root730 / "banks" / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(stages.StoppedError, match="digest"):
        stages.stage_banks(root=root, root730=root730, holidays=world["cal"].holidays, log=print)


def test_the_stored_arrays_must_match_the_row_count(folders: Any) -> None:
    root, root730, world = folders
    path = root730 / "predictions" / "SD-R_unet_s1" / "test.npz"
    np.savez(path, p=np.zeros(world["n"] - 1, np.float32))
    with pytest.raises(leak.CorrectnessFailureError, match="row count"):
        stages.stage_banks(root=root, root730=root730, holidays=world["cal"].holidays, log=print)


def test_limits_are_written_once_and_a_wrong_pooled_limit_stops(
    folders: Any, monkeypatch: Any
) -> None:
    root, root730, world = folders
    stages.stage_banks(root=root, root730=root730, holidays=world["cal"].holidays, log=print)
    rec = stages.stage_limits(
        root=root,
        root730=root730,
        holidays=world["cal"].holidays,
        log=print,
        base=world["base"],
        arrays=world["neighbours"],
    )
    assert set(rec["bands"]) == set(core.BANDS)
    assert rec["n_rows"] == world["n"]
    assert (
        stages.stage_limits(
            root=root,
            root730=root730,
            holidays=world["cal"].holidays,
            log=print,
            base=world["base"],
            arrays=world["neighbours"],
        )
        == rec
    )
    (root / "limits.json").unlink()
    monkeypatch.setattr(core, "POOLED_LIMIT_730", 0.5)
    with pytest.raises(stages.StoppedError, match="pooled limit"):
        stages.stage_limits(
            root=root,
            root730=root730,
            holidays=world["cal"].holidays,
            log=print,
            base=world["base"],
            arrays=world["neighbours"],
        )


def test_guard_orders_the_stages_and_a_recorded_failure_ends_the_run(folders: Any) -> None:
    root, root730, world = folders
    with pytest.raises(RuntimeError, match="run `banks` first"):
        stages.guard_run(root=root, root730=root730, stage="limits")
    stages.stage_banks(root=root, root730=root730, holidays=world["cal"].holidays, log=print)
    stages.guard_run(root=root, root730=root730, stage="limits")
    for stage in ("timing", "fit", "chain"):
        with pytest.raises(RuntimeError, match=r"limits\.json"):
            stages.guard_run(root=root, root730=root730, stage=stage)
    (root / "limits.json").write_text("{}")
    with pytest.raises(RuntimeError, match="overlays"):
        stages.guard_run(root=root, root730=root730, stage="fit")
    (root / "overlays").mkdir()
    (root / "overlays" / "summary.json").write_text(json.dumps({"pass": True}))
    with pytest.raises(RuntimeError, match="tensor"):
        stages.guard_run(root=root, root730=root730, stage="fit")
    (root / "scores").mkdir()
    (root / "scores" / "tensor_checks.json").write_text(json.dumps({"pass": True}))
    with pytest.raises(RuntimeError, match="panel"):
        stages.guard_run(root=root, root730=root730, stage="fit")
    gallery = root / "PRIVATE" / "gallery"
    gallery.mkdir(parents=True)
    ok = {
        "banks_digest": stages.banks730.banks_digest_id(root=root730),
        "panels": dict.fromkeys(core.PANEL_REQUIREMENTS, True),
    }
    (gallery / "panel_check.json").write_text(json.dumps(ok))
    stages.guard_run(root=root, root730=root730, stage="timing")
    with pytest.raises(RuntimeError, match="timing"):
        stages.guard_run(root=root, root730=root730, stage="fit")
    ok["panels"]["G12_raised_sky"] = False
    (gallery / "panel_check.json").write_text(json.dumps(ok))
    with pytest.raises(RuntimeError, match="panel"):
        stages.guard_run(root=root, root730=root730, stage="timing")
    ok["panels"]["G12_raised_sky"] = True
    ok["banks_digest"] = "stale"
    (gallery / "panel_check.json").write_text(json.dumps(ok))
    with pytest.raises(RuntimeError, match="panel"):
        stages.guard_run(root=root, root730=root730, stage="timing")
    ok["banks_digest"] = stages.banks730.banks_digest_id(root=root730)
    (gallery / "panel_check.json").write_text(json.dumps(ok))
    stages.record_invalid(root=root, rule="a check")
    with pytest.raises(RuntimeError, match="correctness"):
        stages.guard_run(root=root, root730=root730, stage="timing")


def banks_and_limits(*, root: Any, root730: Any, world: Any) -> None:
    """Run the banks and limits stages, which every later gate needs."""
    stages.stage_banks(root=root, root730=root730, holidays=world["cal"].holidays, log=print)
    stages.stage_limits(
        root=root,
        root730=root730,
        holidays=world["cal"].holidays,
        log=print,
        base=world["base"],
        arrays=world["neighbours"],
    )


def test_overlays_stage_passes_on_the_invented_bank_and_writes_digests(folders: Any) -> None:
    root, root730, world = folders
    banks_and_limits(root=root, root730=root730, world=world)
    summary = stages.stage_overlays(
        root=root,
        root730=root730,
        holidays=world["cal"].holidays,
        log=print,
        train_seeds=(0, 1),
        neighbours=world["neighbours"],
    )
    assert summary["pass"]
    assert set(summary["banks"]) == {
        "train_s0",
        "train_s1",
        "validation",
        "test",
    }
    on_disk = json.loads((root / "overlays" / "summary.json").read_text())
    assert on_disk["digests"]["test"] == summary["banks"]["test"]["digest"]
    assert stages.overlays_passed(root=root)
    assert summary["banks"]["test"]["reachability"]["transfer_steps_5_points"]["pass"]
    assert all(v["n"] > 0 for v in summary["banks"]["test"]["transfer_checks"].values())
    summary2 = stages.stage_overlays(
        root=root,
        root730=root730,
        holidays=world["cal"].holidays,
        log=print,
        train_seeds=(0, 1),
        neighbours=world["neighbours"],
    )
    assert summary2["banks"]["test"]["digest"] == summary["banks"]["test"]["digest"]


def test_overlays_stage_stops_on_a_short_reachability_count(folders: Any, monkeypatch: Any) -> None:
    root, root730, world = folders
    banks_and_limits(root=root, root730=root730, world=world)
    monkeypatch.setattr(core, "MIN_TRANSFER_STEPS_TEST", 10**9)
    with pytest.raises(stages.StoppedError, match="reachability"):
        stages.stage_overlays(
            root=root,
            root730=root730,
            holidays=world["cal"].holidays,
            log=print,
            train_seeds=(0,),
            neighbours=world["neighbours"],
        )
    assert not stages.overlays_passed(root=root)


def test_overlays_stage_stops_when_a_training_month_lacks_sibling_pairs(folders: Any) -> None:
    root, root730, world = folders
    import e00729_banks as banks729

    no_solar = [s for s in world["segments"] if s.world in ("none", "engine")]
    bank = banks729.Bank(
        kind="train",
        seed=0,
        root=0,
        none_share={},
        segments=no_solar,
        digest=banks729.bank_digest(segments=no_solar),
    )
    banks729.save_bank(bank=bank, path=root730 / "banks" / "train_s0.pkl")
    summary = json.loads((root730 / "banks" / "summary.json").read_text())
    summary["digests"]["train_s0"] = bank.digest
    (root730 / "banks" / "summary.json").write_text(json.dumps(summary))
    banks_and_limits(root=root, root730=root730, world=world)
    with pytest.raises(stages.StoppedError, match="sibling pairs"):
        stages.stage_overlays(
            root=root,
            root730=root730,
            holidays=world["cal"].holidays,
            log=print,
            train_seeds=(0,),
            neighbours=world["neighbours"],
        )


def test_overlays_stage_stops_when_the_training_reverse_share_moves_by_more_than_ten_percent(
    folders: Any, monkeypatch: Any
) -> None:
    root, root730, world = folders
    banks_and_limits(root=root, root730=root730, world=world)
    monkeypatch.setattr(core, "TRAIN_REVERSE_SHARE_TOLERANCE", -1.0)
    with pytest.raises(stages.StoppedError, match="reverse flow"):
        stages.stage_overlays(
            root=root,
            root730=root730,
            holidays=world["cal"].holidays,
            log=print,
            train_seeds=(0,),
            neighbours=world["neighbours"],
        )


def test_experiment_730s_folder_is_never_written(folders: Any) -> None:
    root, root730, world = folders
    before = {p: p.stat().st_mtime_ns for p in root730.rglob("*") if p.is_file()}
    for p in root730.rglob("*"):
        p.chmod(p.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    try:
        stages.stage_banks(root=root, root730=root730, holidays=world["cal"].holidays, log=print)
        stages.stage_limits(
            root=root,
            root730=root730,
            holidays=world["cal"].holidays,
            log=print,
            base=world["base"],
            arrays=world["neighbours"],
        )
        stages.stage_overlays(
            root=root,
            root730=root730,
            holidays=world["cal"].holidays,
            log=print,
            train_seeds=(0,),
            neighbours=world["neighbours"],
        )
    finally:
        for p in root730.rglob("*"):
            p.chmod(p.stat().st_mode | stat.S_IWUSR)
    assert before == {p: p.stat().st_mtime_ns for p in root730.rglob("*") if p.is_file()}
    assert list(root730.iterdir())
    assert not any(root730 in q.parents for q in root.rglob("*"))


def test_guard_needs_the_neighbour_table_before_limits_and_fits(folders: Any) -> None:
    root, root730, world = folders
    stages.stage_banks(root=root, root730=root730, holidays=world["cal"].holidays, log=print)
    (root / "neighbours" / "check.json").write_text(json.dumps({"pass": False}))
    with pytest.raises(RuntimeError, match="neighbours"):
        stages.guard_run(root=root, root730=root730, stage="limits")
    (root / "neighbours" / "check.json").write_text(json.dumps({"pass": True}))
    (root / "limits.json").write_text("{}")
    (root / "neighbours" / "check.json").write_text(json.dumps({"pass": False}))
    with pytest.raises(RuntimeError, match="neighbour"):
        stages.guard_run(root=root, root730=root730, stage="fit")


def test_a_changed_code_version_allows_one_more_gate_attempt_and_no_more(
    folders: Any, monkeypatch: Any
) -> None:
    root, root730, world = folders
    stages.stage_banks(root=root, root730=root730, holidays=world["cal"].holidays, log=print)
    stages.begin_gate_attempt(root=root)
    stages.begin_gate_attempt(root=root)  # the same version is no new attempt
    assert stages.gate_attempts(root=root)["attempts"] == 1
    monkeypatch.setattr(stages, "CODE_VERSION", 2)
    with pytest.raises(RuntimeError, match="no run marker"):  # the marker holds version 1
        stages.guard_run(root=root, root730=root730, stage="limits")
    stages.guard_run(
        root=root, root730=root730, stage="banks"
    )  # a rebuild is allowed: no fit exists
    stages.stage_banks(root=root, root730=root730, holidays=world["cal"].holidays, log=print)
    stages.begin_gate_attempt(root=root)
    assert stages.gate_attempts(root=root)["attempts"] == 2
    monkeypatch.setattr(stages, "CODE_VERSION", 3)
    with pytest.raises(RuntimeError, match="one attempt"):
        stages.guard_run(root=root, root730=root730, stage="banks")
    with pytest.raises(stages.StoppedError, match="one attempt"):
        stages.begin_gate_attempt(root=root)


def test_the_banks_stage_refuses_a_changed_marker_once_a_fit_exists(
    folders: Any, monkeypatch: Any
) -> None:
    root, root730, world = folders
    stages.stage_banks(root=root, root730=root730, holidays=world["cal"].holidays, log=print)
    (root / "fits").mkdir()
    (root / "fits" / "x.json").write_text("{}")
    monkeypatch.setattr(stages, "CODE_VERSION", 2)
    with pytest.raises(RuntimeError, match="superseded_run"):
        stages.guard_run(root=root, root730=root730, stage="banks")


def _mirrored_world(*, reading: np.ndarray) -> dict[str, np.ndarray]:
    """Return neighbours whose half-hour changes mirror D8's, so the timestamp check holds."""
    mirror = 80.0 - np.cumsum(
        np.nan_to_num(np.diff(reading, axis=1, prepend=reading[:, :1])), axis=1
    )
    rng = np.random.default_rng(1)
    return {n: mirror + rng.normal(0, 0.1, mirror.shape) for n in ("D3", "D5", "D14")}


def test_the_neighbours_stage_writes_the_table_once_runs_its_gates_and_records_medians(
    tmp_path: Any, monkeypatch: Any
) -> None:
    root730, root = tmp_path / "730", tmp_path / "732"
    root.mkdir()
    world = fake730.build(root=root730)
    fake730.patch_constants(monkeypatch, root=root730, world=world)
    base = world["base"]
    stages.stage_banks(root=root, root730=root730, holidays=base.holidays, log=print)
    arrays = _mirrored_world(reading=np.where(np.isfinite(base.reading), base.reading, 50.0))
    calls = []
    record = stages.stage_neighbours(
        root=root, root730=root730, base=base, build=lambda: calls.append(1) or arrays, log=print
    )
    assert record["pass"]
    assert calls == [1]
    assert (root / "neighbours" / "table.parquet").exists()
    assert set(record["coverage_by_split"]) == {"training", "validation", "test"}
    assert set(json.loads((root / "neighbours" / "medians.json").read_text())) == {
        "D3",
        "D5",
        "D14",
    }
    again = stages.stage_neighbours(
        root=root, root730=root730, base=base, build=lambda: calls.append(1) or arrays, log=print
    )
    assert again["pass"]
    assert calls == [1]


def test_the_neighbours_stage_stops_on_low_coverage_and_on_a_shifted_timestamp(
    tmp_path: Any, monkeypatch: Any
) -> None:
    for fault in ("coverage", "shift"):
        root730, root = tmp_path / fault / "730", tmp_path / fault / "732"
        root.mkdir(parents=True)
        world = fake730.build(root=root730)
        fake730.patch_constants(monkeypatch, root=root730, world=world)
        base = world["base"]
        stages.stage_banks(root=root, root730=root730, holidays=base.holidays, log=print)
        reading = np.where(np.isfinite(base.reading), base.reading, 50.0)
        arrays = _mirrored_world(reading=reading)
        if fault == "coverage":
            arrays["D14"][:, ::2] = np.nan
        else:
            arrays = {n: np.roll(a, 1, axis=1) for n, a in arrays.items()}  # one half-hour late
        with pytest.raises(
            stages.StoppedError, match="coverage" if fault == "coverage" else "lag 0"
        ):
            stages.stage_neighbours(
                root=root, root730=root730, base=base, build=lambda arrays=arrays: arrays, log=print
            )
        assert json.loads((root / "neighbours" / "check.json").read_text())["pass"] is False
        assert not stages.neighbours_passed(root=root)


@pytest.fixture
def gated(folders: Any):
    """The folders after the banks and limits stages, with an Env on the invented calendar."""
    import e00729_arms as arms729
    import e00732_neighbours as nb

    root, root730, world = folders
    banks_and_limits(root=root, root730=root730, world=world)
    cal, base = world["cal"], world["base"]
    stats = arms729.input_stats(cal=cal, train_days=np.arange(len(cal.dates)))
    medians = {
        n: nb.training_median(
            folded=nb.fold(signed=world["neighbours"][n], floor=0.0), train_rows=base.train_rows
        )
        for n in nb.NEIGHBOURS
    }
    env = stages.Env(
        holidays=cal.holidays, stats=stats, neighbours=world["neighbours"], medians=medians
    )
    return root, root730, env


def test_the_tensor_stage_checks_three_arms_on_the_training_and_validation_banks(
    gated: Any,
) -> None:
    root, root730, env = gated
    summary = stages.stage_tensor_checks(root=root, root730=root730, env=env, log=print)
    assert summary["pass"]
    assert stages.tensor_passed(root=root)
    assert set(summary["checks"]) == {
        f"{arm}/{where}" for arm in ("U*", "U-N", "U-noBSP") for where in ("train", "validation")
    }
    # the fixed hidden spans of the validation batches give the hidden check cases
    assert summary["checks"]["U*/validation"]["hidden_zero"]["n"] > 0
    assert summary["checks"]["U*/train"]["mirror_edges"]["n"] > 0
    assert "mirror_edges" not in summary["checks"]["U-N/train"]


def test_the_tensor_stage_stops_at_row_0_when_a_check_fails(gated: Any, monkeypatch: Any) -> None:
    root, root730, env = gated
    monkeypatch.setattr(stages.arms, "tensor_problems", lambda *, checks: ["mirror_edges"])
    with pytest.raises(stages.StoppedError, match="tensor check mirror_edges failed"):
        stages.stage_tensor_checks(root=root, root730=root730, env=env, log=print)
    assert not stages.tensor_passed(root=root)
    assert json.loads((root / "scores" / "tensor_checks.json").read_text())["pass"] is False


def test_the_tensor_stage_fails_a_silent_mirror_in_the_stage_itself(gated: Any) -> None:
    root, root730, env = gated
    silent = {
        n: np.full_like(v, np.nan) for n, v in env.neighbours.items()
    }  # no neighbour holds a reading, so no mirror can show
    env2 = stages.Env(
        holidays=env.holidays, stats=env.stats, neighbours=silent, medians=env.medians
    )
    with pytest.raises(stages.StoppedError):
        stages.stage_tensor_checks(root=root, root730=root730, env=env2, log=print)


def test_the_gallery_stage_writes_published_and_private_panels_and_the_check_file(
    gated: Any, folders: Any
) -> None:
    import e00732_gallery as gallery

    root, root730, env = gated
    world = folders[2]
    stages.stage_overlays(
        root=root, root730=root730, holidays=world["cal"].holidays, log=print, train_seeds=(0, 1),
        neighbours=world["neighbours"],
    )  # fmt: skip
    results = gallery.stage_gallery(
        root=root, root730=root730, env=env, base=world["base"], log=print
    )
    assert set(results) == set(core.PANEL_REQUIREMENTS)
    private = gallery.gallery_private_dir(root=root)
    assert list(private.glob("gallery_g17*.svg"))
    assert not list((root / "figures" / "gallery").glob("*g17*"))
    assert list((root / "figures" / "gallery").glob("gallery_g10*.svg"))
    auto = json.loads((private / gallery.AUTO_FILE).read_text())
    assert set(auto["panels"]) == set(core.PANEL_REQUIREMENTS)
    confirms = dict.fromkeys(core.PANEL_REQUIREMENTS, True)
    record = gallery.record_panel_check(root=root, root730=root730, confirms=confirms)
    assert record["panels"] == auto["panels"]  # the reading cannot turn a failed check into a pass
    assert not gallery.record_panel_check(
        root=root, root730=root730, confirms=dict.fromkeys(confirms, False)
    )["all_passed"]


def _write_stub_predictions(*, root: Any, truth_rows: int) -> None:
    """Write seed folders for U* with invented outputs of the right row count (the plain, transfer,
    injection, and raised sets).
    """
    rng = np.random.default_rng(0)
    spec = core.ARMS["U*"]
    for s in core.SEEDS:
        folder = root / "predictions" / core.job_name(arm="U*", arch=spec.arch, seed=s)
        folder.mkdir(parents=True)
        base = {
            "p": rng.uniform(0, 0.02, truth_rows), "pv": rng.uniform(0, 5, truth_rows),
            "step_p": rng.uniform(0, 0.1, truth_rows), "size": rng.normal(0, 1, truth_rows),
        }  # fmt: skip
        for name in (
            "test",
            "test_transfer",
            "test_raised",
            *(f"test_inj_{a:g}" for a in core.TEST_INJECTION_POINTS),
        ):
            np.savez(folder / f"{name}.npz", **{k: v.astype(np.float32) for k, v in base.items()})  # ty: ignore[invalid-argument-type]


def test_the_score_stage_reads_stored_arms_and_leaves_a_missing_arm_void(
    gated: Any, folders: Any
) -> None:
    import e00732_report as report

    root, root730, env = gated
    world = folders[2]
    stages.stage_overlays(
        root=root, root730=root730, holidays=world["cal"].holidays, log=print, train_seeds=(0, 1),
        neighbours=world["neighbours"],
    )  # fmt: skip
    _write_stub_predictions(root=root, truth_rows=world["n"])
    record = report.stage_score(root=root, root730=root730, env=env, base=world["base"], log=print)
    assert record["n_rows"] == world["n"]
    assert "U*" in record["arms_present"]
    assert "B-PV0" in record["arms_present"]  # the sky-only rule is fitted on the validation bank
    assert "B-PV1" not in record["arms_present"]  # without the context, B-PV1 stays void
    c = record["readings"]["counted"]
    assert c["C1"]["void"] is False  # U* and the stored S are both present
    assert c["C6"]["void"] is True  # S+ was not fitted: a missing arm is void, never zero
    verdict = report.stage_report(
        root=root, root730=root730, env=env, base=world["base"], log=print
    )
    # H2 was not read (no real-day predictions) and S+ is missing: the run is "failed" (row 2), never a silent row 6
    assert verdict["row"] == 2
    assert "H2" in str(verdict["reason"])


def test_the_series_and_figures_stages_draw_windows_and_the_contrast_chart(
    gated: Any, folders: Any
) -> None:
    import e00732_report as report

    root, root730, env = gated
    world = folders[2]
    stages.stage_overlays(
        root=root, root730=root730, holidays=world["cal"].holidays, log=print, train_seeds=(0, 1),
        neighbours=world["neighbours"],
    )  # fmt: skip
    _write_stub_predictions(root=root, truth_rows=world["n"])
    report.stage_score(root=root, root730=root730, env=env, base=world["base"], log=print)
    result = report.stage_series(root=root, root730=root730, env=env, base=world["base"], log=print)
    assert result["drawn"]
    assert list((root / "figures" / "series").glob("series_W*.svg"))
    pairs = [n for n in ("W9", "W10", "W11") if n in result["drawn"]]
    assert pairs or {"W9", "W10", "W11"} <= set(result["no_window"])
    assert {"W24", "W25"} <= set(result["no_window"])  # no real window without K3
    assert not next(iter((root / "figures" / "series").glob("*.svg"))).read_text().count("2022-")
    report.stage_figures(root=root, root730=root730, env=env, base=world["base"], log=print)
    assert (root / "figures" / "contrasts.svg").exists()


def test_the_train_tensor_check_has_a_bsp_present_injected_pair(gated: Any) -> None:
    root, root730, env = gated
    summary = stages.stage_tensor_checks(root=root, root730=root730, env=env, log=print)
    assert summary["checks"]["U*/train"]["injected_bsp_on_delta"]["n"] > 0
    assert summary["checks"]["U*/validation"]["injected_bsp_on_delta"]["n"] > 0


def test_bsp_present_picks_hold_only_windows_with_a_meter_and_solar_output(gated: Any) -> None:
    import e00730_banks as banks730
    import e00732_arms as arms
    import e00732_overlays as ov
    from e00729_banks import load_bank

    _, root730, env = gated
    bank = load_bank(path=banks730.bank_path(root=root730, kind="train", seed=0))
    over = ov.overlay_segments(
        segments=bank.segments, kind="train", seed=0, neighbours=env.neighbours
    )
    prepared = arms.prepare(
        overlays=over, holidays=env.holidays, stats=env.stats, medians=env.medians
    )
    picks = stages.bsp_present_solar_picks(prepared=prepared, limit=50)
    assert picks
    for seg_i, day in picks:
        p = prepared[seg_i]
        assert p.world in ov.SOLAR_WORLDS
        assert np.isfinite(p.bsp[day * 48 : day * 48 + 48]).all()
    for p in prepared:
        p.bsp[:] = np.nan
    assert stages.bsp_present_solar_picks(prepared=prepared, limit=50) == []


def test_a_tensor_check_without_a_case_is_a_problem() -> None:
    import e00732_arms as arms
    import e00732_overlays as ov

    empty = ov.Share(value=0.0, n=0) if hasattr(ov, "Share") else None
    assert empty is not None
    assert arms.tensor_problems(checks={"injected_bsp_on_delta": empty}) == [
        "injected_bsp_on_delta"
    ]


def _real_rows(*, n_days: int, first_day: int = 0) -> Any:
    import e00732_report as report

    hh = np.tile(np.arange(48), n_days)
    day = np.repeat(np.arange(first_day, first_day + n_days), 48)
    cosz = np.tile(np.clip(np.sin(np.linspace(0, np.pi, 48)), 0, 1), n_days)
    csi = np.repeat(np.linspace(0.1, 1.0, n_days), 48)
    return report.RealRows(cosz=cosz, csi=csi, hh=hh, day=day)


def test_real_day_checks_read_p2_p3_p4_and_k3_and_leave_missing_inputs_void() -> None:
    import e00732_real as real
    import e00732_report as report

    n_days = 30
    eras = _real_rows(n_days=n_days)
    r1 = _real_rows(n_days=4, first_day=20)
    k1 = _real_rows(n_days=4, first_day=40)
    # the solar output follows the sky (P3), is zero before the farm exists (P2), and ignores the engine (P4)
    pv_eras = eras.csi * eras.cosz * 10.0
    day_eras = np.array(["solar"] * 60)
    running = np.zeros(len(k1.hh), dtype=bool)
    running[::7] = True
    dates = np.datetime64("2022-01-01") + np.arange(60)
    outputs = {
        "r1": {"pv": np.zeros(len(r1.hh)), "u_step": np.zeros(len(r1.hh)), "un_step": np.zeros(len(r1.hh))},
        "eras": {"pv": pv_eras},
        "k1": {"pv": np.ones(len(k1.hh))},
    }  # fmt: skip
    got = report.real_day_checks(
        r1=r1, eras=eras, k1=k1, outputs=outputs, day_eras=day_eras, running_k1=running, edges=[],
        closure_days=np.zeros(60, dtype=bool), dates=dates,
    )  # fmt: skip
    assert got["p2"]["pass"] is True
    assert got["p3"]["pass"] is True
    assert got["p4"]["pass"] is True
    assert set(got["k3"]) == {"U*", "U-N"}
    assert got["k3"]["U*"]["hit"]["void"] is True  # no logged edges: void, never zero
    assert got["k3"]["U*"]["false"]["flagged_per_day"] == 0.0
    # a solar output that ignores the sky fails P3, and a missing set leaves its check None
    flat = {**outputs, "eras": {"pv": np.full(len(eras.hh), 3.0)}}
    bad = report.real_day_checks(
        r1=r1, eras=eras, k1=None, outputs=flat, day_eras=day_eras, running_k1=None, edges=[],
        closure_days=None, dates=dates,
    )  # fmt: skip
    assert bad["p3"]["pass"] is False
    assert bad["p4"] is None
    assert bad["k3"] is None
    assert isinstance(real.Edge, type)


def test_the_series_stage_fails_loudly_without_fitted_outputs(gated: Any, folders: Any) -> None:
    import e00732_report as report

    root, root730, env = gated
    world = folders[2]
    stages.stage_overlays(
        root=root, root730=root730, holidays=world["cal"].holidays, log=print, train_seeds=(0, 1),
        neighbours=world["neighbours"],
    )  # fmt: skip
    _write_stub_predictions(root=root, truth_rows=world["n"])
    report.stage_score(root=root, root730=root730, env=env, base=world["base"], log=print)
    import shutil

    for folder in (root / "predictions").iterdir():
        shutil.rmtree(folder)
    with pytest.raises(report.FiguresError, match="no empty plot"):
        report.stage_series(root=root, root730=root730, env=env, base=world["base"], log=print)
    assert not (root / "figures" / "series").exists() or not list(
        (root / "figures" / "series").glob("*.svg")
    )


def test_the_figures_stage_fails_loudly_without_u_star_and_draws_the_real_checks(
    gated: Any, folders: Any
) -> None:
    import e00732_figures as figures
    import e00732_report as report

    root, root730, env = gated
    world = folders[2]
    stages.stage_overlays(
        root=root, root730=root730, holidays=world["cal"].holidays, log=print, train_seeds=(0, 1),
        neighbours=world["neighbours"],
    )  # fmt: skip
    _write_stub_predictions(root=root, truth_rows=world["n"])
    record = report.stage_score(root=root, root730=root730, env=env, base=world["base"], log=print)
    stages.write_json(
        path=root / report.REAL_FILE,
        payload={"p1": {"all": {"pass": True}}, "p2": {"pass": False}, "p3": None, "p4": None, "k3": None, "h2": None},
    )  # fmt: skip
    written = report.stage_figures(
        root=root, root730=root730, env=env, base=world["base"], log=print
    )
    assert {"contrasts", "bsp_input", "real_checks"} <= set(written)
    for name in (
        "fig01_brier_by_group", "fig03_harm", "fig04_reliability", "fig05_solar_error",
        "fig06_recovery", "fig07_solar_checks", "fig08_transfer_steps", "fig09_scenarios",
    ):  # fmt: skip
        assert (root / "figures" / f"{name}.svg").stat().st_size > 1000, name
    assert (
        figures.real_check_rows(real={"p1": {"all": {"pass": True}}})["word"].to_list()[0] == "pass"
    )
    with pytest.raises(figures.HistoryError, match="empty"):
        figures.real_check_rows(real={})
    record["arms_present"].remove("U*")
    stages.write_json(path=root / report.READINGS_FILE, payload=record)
    with pytest.raises(report.FiguresError, match="no empty plot"):
        report.stage_figures(root=root, root730=root730, env=env, base=world["base"], log=print)


def test_arm_scores_read_the_bsp_hidden_set_for_r21(folders: Any) -> None:
    import e00732_report as report

    root, _, world = folders
    _write_stub_predictions(root=root, truth_rows=world["n"])
    assert report.arm_scores(root=root, arm="U*", n_rows=world["n"]).p_nobsp is None
    for s in core.SEEDS:
        folder = root / "predictions" / core.job_name(arm="U*", arch=core.ARMS["U*"].arch, seed=s)
        np.savez(folder / "test_nobsp.npz", p=np.full(world["n"], 0.01, dtype=np.float32))
    got = report.arm_scores(root=root, arm="U*", n_rows=world["n"])
    assert got.p_nobsp is not None
    assert got.p_nobsp.shape == (world["n"],)


def _run_overlays(*, folders: Any) -> dict:
    root, root730, world = folders
    banks_and_limits(root=root, root730=root730, world=world)
    return stages.stage_overlays(
        root=root, root730=root730, holidays=world["cal"].holidays, log=print, train_seeds=(0,),
        neighbours=world["neighbours"],
    )  # fmt: skip


def test_overlays_stage_records_the_unreliable_share_per_bank(folders: Any) -> None:
    summary = _run_overlays(folders=folders)
    for name in ("train_s0", "validation", "test"):
        rec = summary["banks"][name]["unreliable"]
        assert rec["all_half_hours_by_mask"] == rec["all_half_hours_by_loop"]
        assert 0.0 <= rec["share"] <= core.MAX_UNRELIABLE_SHARE
        assert summary["banks"][name]["unreliable_digest"]
    assert any(k.endswith("/unreliable") for k in json.loads(
        (folders[0] / "overlays" / "summary.json").read_text()
    )["digests"])  # fmt: skip


def test_overlays_stage_stops_when_more_than_five_percent_are_unreliable(
    folders: Any, monkeypatch: Any
) -> None:
    import e00732_overlays as ov

    monkeypatch.setattr(ov, "unreliable_counts", lambda *, segments: (10, 100))
    with pytest.raises(stages.StoppedError, match="unreliable solar targets"):
        _run_overlays(folders=folders)


def test_overlays_stage_stops_on_a_silent_mask(folders: Any, monkeypatch: Any) -> None:
    import e00732_overlays as ov

    monkeypatch.setattr(
        ov, "unreliable_by_loop", lambda *, seg: 4
    )  # a gap exists, the mask finds none
    with pytest.raises(stages.StoppedError, match="independent count"):
        _run_overlays(folders=folders)


def test_every_result_figure_fails_loudly_without_any_fitted_arm(gated: Any, folders: Any) -> None:
    import e00732_report as report
    import e00732_result_figures as rf

    root, root730, env = gated
    world = folders[2]
    truth_, _, _ = report.build_truth(root730=root730, env=env, noise_d8=1.0)
    limits = core.read_limits_file(path=root / "limits.json")
    assert truth_.rows.n == world["n"]
    for draw in (
        rf.fig1_brier_by_group, rf.fig4_reliability, rf.fig5_solar_error, rf.fig6_recovery,
        rf.fig7_solar_checks, rf.fig8_transfer_steps, rf.fig9_scenarios,
    ):  # fmt: skip
        with pytest.raises((rf.FiguresError, ValueError)):
            draw(truth=truth_, arms={})
    with pytest.raises(rf.FiguresError):
        rf.fig3_harm(truth=truth_, arms={}, limits=limits)
