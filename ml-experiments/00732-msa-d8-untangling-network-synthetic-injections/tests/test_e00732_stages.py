"""Tests of the guard, the banks check, the limits, and the overlays stage on an invented copy of experiment 730's folder."""

import json
import os
import stat

import e00732_core as core
import e00732_leakage as leak
import e00732_stages as stages
import fake730
import numpy as np
import pytest
from world732 import seed_neighbour_files


@pytest.fixture
def folders(tmp_path, monkeypatch):
    root730 = tmp_path / "730"
    world = fake730.build(root=root730)
    fake730.patch_constants(monkeypatch, root=root730, world=world)
    root = tmp_path / "732"
    root.mkdir()
    seed_neighbour_files(root=root, base=world["base"], arrays=world["neighbours"])
    return root, root730, world


def test_banks_stage_verifies_records_once_and_marks_the_run(folders) -> None:
    root, root730, world = folders
    record = stages.stage_banks(
        root=root, root730=root730, holidays=world["cal"].holidays, log=print
    )
    assert record["n_rows"] == world["n"] and (root / "run_marker.json").exists()
    again = stages.stage_banks(
        root=root, root730=root730, holidays=world["cal"].holidays, log=print
    )
    assert again == record


def test_banks_stage_stops_when_a_stored_file_changes_after_the_first_record(folders) -> None:
    root, root730, world = folders
    stages.stage_banks(root=root, root730=root730, holidays=world["cal"].holidays, log=print)
    path = root730 / "predictions" / "SD-R_unet_s1" / "test.npz"  # the canary reads arm S only
    with np.load(path) as z:
        arrays = {k: z[k].copy() for k in z.files}
    arrays["p"][0] = 0.5 if arrays["p"][0] != 0.5 else 0.4
    np.savez(path, **arrays)
    with pytest.raises(stages.StoppedError, match="differ from their recorded digests"):
        stages.stage_banks(root=root, root730=root730, holidays=world["cal"].holidays, log=print)


def test_banks_stage_fails_the_canary_and_the_row_count(folders, monkeypatch) -> None:
    root, root730, world = folders
    monkeypatch.setattr(core, "BRIER_CANARY", core.BRIER_CANARY + 0.01)
    with pytest.raises(leak.CorrectnessFailureError, match="canary"):
        stages.stage_banks(root=root, root730=root730, holidays=world["cal"].holidays, log=print)
    monkeypatch.setattr(core, "BRIER_CANARY", core.BRIER_CANARY - 0.01)
    monkeypatch.setattr(core, "N_TEST_ROWS", world["n"] + 1)
    with pytest.raises(leak.CorrectnessFailureError, match="row count"):
        stages.stage_banks(root=root, root730=root730, holidays=world["cal"].holidays, log=print)


def test_banks_stage_stops_on_a_changed_bank_digest(folders) -> None:
    root, root730, world = folders
    summary = json.loads((root730 / "banks" / "summary.json").read_text())
    summary["digests"]["test"] = "0" * 64
    (root730 / "banks" / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(stages.StoppedError, match="digest"):
        stages.stage_banks(root=root, root730=root730, holidays=world["cal"].holidays, log=print)


def test_the_stored_arrays_must_match_the_row_count(folders) -> None:
    root, root730, world = folders
    path = root730 / "predictions" / "SD-R_unet_s1" / "test.npz"
    np.savez(path, p=np.zeros(world["n"] - 1, np.float32))
    with pytest.raises(leak.CorrectnessFailureError, match="row count"):
        stages.stage_banks(root=root, root730=root730, holidays=world["cal"].holidays, log=print)


def test_limits_are_written_once_and_a_wrong_pooled_limit_stops(folders, monkeypatch) -> None:
    root, root730, world = folders
    stages.stage_banks(root=root, root730=root730, holidays=world["cal"].holidays, log=print)
    rec = stages.stage_limits(root=root, root730=root730, holidays=world["cal"].holidays, log=print, base=world["base"], arrays=world["neighbours"])
    assert set(rec["bands"]) == set(core.BANDS) and rec["n_rows"] == world["n"]
    assert (
        stages.stage_limits(root=root, root730=root730, holidays=world["cal"].holidays, log=print, base=world["base"], arrays=world["neighbours"])
        == rec
    )
    (root / "limits.json").unlink()
    monkeypatch.setattr(core, "POOLED_LIMIT_730", 0.5)
    with pytest.raises(stages.StoppedError, match="pooled limit"):
        stages.stage_limits(root=root, root730=root730, holidays=world["cal"].holidays, log=print, base=world["base"], arrays=world["neighbours"])


def test_guard_orders_the_stages_and_a_recorded_failure_ends_the_run(folders) -> None:
    root, root730, world = folders
    with pytest.raises(RuntimeError, match="run `banks` first"):
        stages.guard_run(root=root, root730=root730, stage="limits")
    stages.stage_banks(root=root, root730=root730, holidays=world["cal"].holidays, log=print)
    stages.guard_run(root=root, root730=root730, stage="limits")
    for stage in ("timing", "fit", "chain"):
        with pytest.raises(RuntimeError, match="limits.json"):
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


def banks_and_limits(*, root, root730, world) -> None:
    """Run the banks and limits stages, which every later gate needs."""
    stages.stage_banks(root=root, root730=root730, holidays=world["cal"].holidays, log=print)
    stages.stage_limits(
        root=root, root730=root730, holidays=world["cal"].holidays, log=print,
        base=world["base"], arrays=world["neighbours"],
    )


def test_overlays_stage_passes_on_the_invented_bank_and_writes_digests(folders) -> None:
    root, root730, world = folders
    banks_and_limits(root=root, root730=root730, world=world)
    summary = stages.stage_overlays(
        root=root, root730=root730, holidays=world["cal"].holidays, log=print, train_seeds=(0, 1), neighbours=world["neighbours"],
    )
    assert summary["pass"] and set(summary["banks"]) == {
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
        root=root, root730=root730, holidays=world["cal"].holidays, log=print, train_seeds=(0, 1), neighbours=world["neighbours"],
    )
    assert summary2["banks"]["test"]["digest"] == summary["banks"]["test"]["digest"]


def test_overlays_stage_stops_on_a_short_reachability_count(folders, monkeypatch) -> None:
    root, root730, world = folders
    banks_and_limits(root=root, root730=root730, world=world)
    monkeypatch.setattr(core, "MIN_TRANSFER_STEPS_TEST", 10**9)
    with pytest.raises(stages.StoppedError, match="reachability"):
        stages.stage_overlays(
            root=root, root730=root730, holidays=world["cal"].holidays, log=print, train_seeds=(0,), neighbours=world["neighbours"],
        )
    assert not stages.overlays_passed(root=root)


def test_overlays_stage_stops_when_a_training_month_lacks_sibling_pairs(folders) -> None:
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
            root=root, root730=root730, holidays=world["cal"].holidays, log=print, train_seeds=(0,), neighbours=world["neighbours"],
        )


def test_overlays_stage_stops_when_the_training_reverse_share_moves_by_more_than_ten_percent(
    folders, monkeypatch
) -> None:
    root, root730, world = folders
    banks_and_limits(root=root, root730=root730, world=world)
    monkeypatch.setattr(core, "TRAIN_REVERSE_SHARE_TOLERANCE", -1.0)
    with pytest.raises(stages.StoppedError, match="reverse flow"):
        stages.stage_overlays(
            root=root, root730=root730, holidays=world["cal"].holidays, log=print, train_seeds=(0,), neighbours=world["neighbours"],
        )


def test_experiment_730s_folder_is_never_written(folders) -> None:
    root, root730, world = folders
    before = {p: p.stat().st_mtime_ns for p in root730.rglob("*") if p.is_file()}
    for p in root730.rglob("*"):
        p.chmod(p.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    try:
        stages.stage_banks(root=root, root730=root730, holidays=world["cal"].holidays, log=print)
        stages.stage_limits(root=root, root730=root730, holidays=world["cal"].holidays, log=print, base=world["base"], arrays=world["neighbours"])
        stages.stage_overlays(
            root=root, root730=root730, holidays=world["cal"].holidays, log=print, train_seeds=(0,), neighbours=world["neighbours"],
        )
    finally:
        for p in root730.rglob("*"):
            p.chmod(p.stat().st_mode | stat.S_IWUSR)
    assert before == {p: p.stat().st_mtime_ns for p in root730.rglob("*") if p.is_file()}
    assert os.listdir(root730) and not any(root730 in q.parents for q in root.rglob("*"))


def test_guard_needs_the_neighbour_table_before_limits_and_fits(folders) -> None:
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


def test_a_changed_code_version_allows_one_more_gate_attempt_and_no_more(folders, monkeypatch) -> None:
    root, root730, world = folders
    stages.stage_banks(root=root, root730=root730, holidays=world["cal"].holidays, log=print)
    stages.begin_gate_attempt(root=root)
    stages.begin_gate_attempt(root=root)  # the same version is no new attempt
    assert stages.gate_attempts(root=root)["attempts"] == 1
    monkeypatch.setattr(stages, "CODE_VERSION", 2)
    with pytest.raises(RuntimeError, match="no run marker"):  # the marker holds version 1
        stages.guard_run(root=root, root730=root730, stage="limits")
    stages.guard_run(root=root, root730=root730, stage="banks")  # a rebuild is allowed: no fit exists
    stages.stage_banks(root=root, root730=root730, holidays=world["cal"].holidays, log=print)
    stages.begin_gate_attempt(root=root)
    assert stages.gate_attempts(root=root)["attempts"] == 2
    monkeypatch.setattr(stages, "CODE_VERSION", 3)
    with pytest.raises(RuntimeError, match="one attempt"):
        stages.guard_run(root=root, root730=root730, stage="banks")
    with pytest.raises(stages.StoppedError, match="one attempt"):
        stages.begin_gate_attempt(root=root)


def test_the_banks_stage_refuses_a_changed_marker_once_a_fit_exists(folders, monkeypatch) -> None:
    root, root730, world = folders
    stages.stage_banks(root=root, root730=root730, holidays=world["cal"].holidays, log=print)
    (root / "fits").mkdir()
    (root / "fits" / "x.json").write_text("{}")
    monkeypatch.setattr(stages, "CODE_VERSION", 2)
    with pytest.raises(RuntimeError, match="superseded_run"):
        stages.guard_run(root=root, root730=root730, stage="banks")


def _mirrored_world(*, reading: np.ndarray) -> dict[str, np.ndarray]:
    """Return neighbours whose half-hour changes mirror D8's, so the timestamp check holds."""
    mirror = 80.0 - np.cumsum(np.nan_to_num(np.diff(reading, axis=1, prepend=reading[:, :1])), axis=1)
    rng = np.random.default_rng(1)
    return {n: mirror + rng.normal(0, 0.1, mirror.shape) for n in ("D3", "D5", "D14")}


def test_the_neighbours_stage_writes_the_table_once_runs_its_gates_and_records_medians(tmp_path, monkeypatch) -> None:
    root730, root = tmp_path / "730", tmp_path / "732"
    root.mkdir()
    world = fake730.build(root=root730)
    fake730.patch_constants(monkeypatch, root=root730, world=world)
    base = world["base"]
    stages.stage_banks(root=root, root730=root730, holidays=base.holidays, log=print)
    arrays = _mirrored_world(reading=np.where(np.isfinite(base.reading), base.reading, 50.0))
    calls = []
    record = stages.stage_neighbours(root=root, root730=root730, base=base, build=lambda: calls.append(1) or arrays, log=print)
    assert record["pass"] and calls == [1] and (root / "neighbours" / "table.parquet").exists()
    assert set(record["coverage_by_split"]) == {"training", "validation", "test"}
    assert set(json.loads((root / "neighbours" / "medians.json").read_text())) == {"D3", "D5", "D14"}
    again = stages.stage_neighbours(root=root, root730=root730, base=base, build=lambda: calls.append(1) or arrays, log=print)
    assert again["pass"] and calls == [1]  # the stored table is read, not rebuilt


def test_the_neighbours_stage_stops_on_low_coverage_and_on_a_shifted_timestamp(tmp_path, monkeypatch) -> None:
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
        with pytest.raises(stages.StoppedError, match="coverage" if fault == "coverage" else "lag 0"):
            stages.stage_neighbours(root=root, root730=root730, base=base, build=lambda: arrays, log=print)
        assert json.loads((root / "neighbours" / "check.json").read_text())["pass"] is False
        assert not stages.neighbours_passed(root=root)


@pytest.fixture
def gated(folders):
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
    env = stages.Env(holidays=cal.holidays, stats=stats, neighbours=world["neighbours"], medians=medians)
    return root, root730, env


def test_the_tensor_stage_checks_three_arms_on_the_training_and_validation_banks(gated) -> None:
    root, root730, env = gated
    summary = stages.stage_tensor_checks(root=root, root730=root730, env=env, log=print)
    assert summary["pass"] and stages.tensor_passed(root=root)
    assert set(summary["checks"]) == {
        f"{arm}/{where}" for arm in ("U*", "U-N", "U-noBSP") for where in ("train", "validation")
    }
    # the fixed hidden spans of the validation batches give the hidden check cases
    assert summary["checks"]["U*/validation"]["hidden_zero"]["n"] > 0
    assert summary["checks"]["U*/train"]["mirror_edges"]["n"] > 0
    assert "mirror_edges" not in summary["checks"]["U-N/train"]


def test_the_tensor_stage_stops_at_row_0_when_a_check_fails(gated, monkeypatch) -> None:
    root, root730, env = gated
    monkeypatch.setattr(stages.arms, "tensor_problems", lambda *, checks: ["mirror_edges"])
    with pytest.raises(stages.StoppedError, match="tensor check mirror_edges failed"):
        stages.stage_tensor_checks(root=root, root730=root730, env=env, log=print)
    assert not stages.tensor_passed(root=root)
    assert json.loads((root / "scores" / "tensor_checks.json").read_text())["pass"] is False


def test_the_tensor_stage_fails_a_silent_mirror_in_the_stage_itself(gated) -> None:
    root, root730, env = gated
    silent = {n: np.full_like(v, np.nan) for n, v in env.neighbours.items()}  # no neighbour holds a reading, so no mirror can show
    env2 = stages.Env(holidays=env.holidays, stats=env.stats, neighbours=silent, medians=env.medians)
    with pytest.raises(stages.StoppedError):
        stages.stage_tensor_checks(root=root, root730=root730, env=env2, log=print)
