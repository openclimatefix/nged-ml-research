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


@pytest.fixture
def folders(tmp_path, monkeypatch):
    root730 = tmp_path / "730"
    world = fake730.build(root=root730)
    fake730.patch_constants(monkeypatch, root=root730, world=world)
    root = tmp_path / "732"
    root.mkdir()
    return root, root730, world


def test_banks_stage_verifies_records_once_and_marks_the_run(folders) -> None:
    root, root730, world = folders
    record = stages.stage_banks(root=root, root730=root730, holidays=world["cal"].holidays, log=print)
    assert record["n_rows"] == world["n"] and (root / "run_marker.json").exists()
    again = stages.stage_banks(root=root, root730=root730, holidays=world["cal"].holidays, log=print)
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
    with pytest.raises(leak.CorrectnessFailureError, match="scored half-hours"):
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
    with pytest.raises(leak.CorrectnessFailureError, match="rows"):
        stages.stage_banks(root=root, root730=root730, holidays=world["cal"].holidays, log=print)


def test_limits_are_written_once_and_a_wrong_pooled_limit_stops(folders, monkeypatch) -> None:
    root, root730, world = folders
    stages.stage_banks(root=root, root730=root730, holidays=world["cal"].holidays, log=print)
    rec = stages.stage_limits(root=root, root730=root730, holidays=world["cal"].holidays, log=print)
    assert set(rec["bands"]) == set(core.BANDS) and rec["n_rows"] == world["n"]
    assert stages.stage_limits(root=root, root730=root730, holidays=world["cal"].holidays, log=print) == rec
    (root / "limits.json").unlink()
    monkeypatch.setattr(core, "POOLED_LIMIT_730", 0.5)
    with pytest.raises(stages.StoppedError, match="pooled limit"):
        stages.stage_limits(root=root, root730=root730, holidays=world["cal"].holidays, log=print)


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
    with pytest.raises(RuntimeError, match="panel"):
        stages.guard_run(root=root, root730=root730, stage="fit")
    gallery = root / "PRIVATE" / "gallery"
    gallery.mkdir(parents=True)
    ok = {"banks_digest": stages.banks730.banks_digest_id(root=root730), "panels": dict.fromkeys(core.PANEL_REQUIREMENTS, True)}
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


def test_overlays_stage_passes_on_the_invented_bank_and_writes_digests(folders) -> None:
    root, root730, world = folders
    stages.stage_banks(root=root, root730=root730, holidays=world["cal"].holidays, log=print)
    summary = stages.stage_overlays(root=root, root730=root730, holidays=world["cal"].holidays, log=print, train_seeds=(0, 1))
    assert summary["pass"] and set(summary["banks"]) == {"train_s0", "train_s1", "validation", "test"}
    on_disk = json.loads((root / "overlays" / "summary.json").read_text())
    assert on_disk["digests"]["test"] == summary["banks"]["test"]["digest"]
    assert stages.overlays_passed(root=root)
    assert summary["banks"]["test"]["reachability"]["transfer_steps_5_points"]["pass"]
    assert all(v["n"] > 0 for v in summary["banks"]["test"]["transfer_checks"].values())
    summary2 = stages.stage_overlays(root=root, root730=root730, holidays=world["cal"].holidays, log=print, train_seeds=(0, 1))
    assert summary2["banks"]["test"]["digest"] == summary["banks"]["test"]["digest"]


def test_overlays_stage_stops_on_a_short_reachability_count(folders, monkeypatch) -> None:
    root, root730, world = folders
    stages.stage_banks(root=root, root730=root730, holidays=world["cal"].holidays, log=print)
    monkeypatch.setattr(core, "MIN_TRANSFER_STEPS_TEST", 10**9)
    with pytest.raises(stages.StoppedError, match="reachability"):
        stages.stage_overlays(root=root, root730=root730, holidays=world["cal"].holidays, log=print, train_seeds=(0,))
    assert not stages.overlays_passed(root=root)


def test_overlays_stage_stops_when_a_training_month_lacks_sibling_pairs(folders) -> None:
    root, root730, world = folders
    import e00729_banks as banks729

    no_solar = [s for s in world["segments"] if s.world in ("none", "engine")]
    bank = banks729.Bank(kind="train", seed=0, root=0, none_share={}, segments=no_solar, digest=banks729.bank_digest(segments=no_solar))
    banks729.save_bank(bank=bank, path=root730 / "banks" / "train_s0.pkl")
    summary = json.loads((root730 / "banks" / "summary.json").read_text())
    summary["digests"]["train_s0"] = bank.digest
    (root730 / "banks" / "summary.json").write_text(json.dumps(summary))
    stages.stage_banks(root=root, root730=root730, holidays=world["cal"].holidays, log=print)
    with pytest.raises(stages.StoppedError, match="sibling pairs"):
        stages.stage_overlays(root=root, root730=root730, holidays=world["cal"].holidays, log=print, train_seeds=(0,))


def test_overlays_stage_stops_when_the_training_reverse_share_moves_by_more_than_ten_percent(folders, monkeypatch) -> None:
    root, root730, world = folders
    stages.stage_banks(root=root, root730=root730, holidays=world["cal"].holidays, log=print)
    monkeypatch.setattr(core, "TRAIN_REVERSE_SHARE_TOLERANCE", -1.0)
    with pytest.raises(stages.StoppedError, match="reverse flow"):
        stages.stage_overlays(root=root, root730=root730, holidays=world["cal"].holidays, log=print, train_seeds=(0,))


def test_experiment_730s_folder_is_never_written(folders) -> None:
    root, root730, world = folders
    before = {p: p.stat().st_mtime_ns for p in root730.rglob("*") if p.is_file()}
    for p in root730.rglob("*"):
        p.chmod(p.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    try:
        stages.stage_banks(root=root, root730=root730, holidays=world["cal"].holidays, log=print)
        stages.stage_limits(root=root, root730=root730, holidays=world["cal"].holidays, log=print)
        stages.stage_overlays(root=root, root730=root730, holidays=world["cal"].holidays, log=print, train_seeds=(0,))
    finally:
        for p in root730.rglob("*"):
            p.chmod(p.stat().st_mode | stat.S_IWUSR)
    assert before == {p: p.stat().st_mtime_ns for p in root730.rglob("*") if p.is_file()}
    assert os.listdir(root730) and not any(root730 in q.parents for q in root.rglob("*"))
