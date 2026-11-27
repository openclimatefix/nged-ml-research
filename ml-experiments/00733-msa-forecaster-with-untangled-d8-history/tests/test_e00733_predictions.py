# ruff: noqa: PT018, E501
"""Tests of `e00733_predictions` and `e00733_scorestage`: loading a fitted network, the synthetic
archive, a prediction that is written once, and the score files."""

import json
from pathlib import Path

import e00733_core as core
import e00733_score as sc
import e00733_scorestage as stage
import e00733_scoring as scoring
import e00733_scoring_synth as synth
import e00733_store as store
import e00733_synth as sy
import e00733_world as world
import numpy as np
import pytest
from synth_fixture import make_segment

pytest.importorskip("torch")
import e00733_adapters as adapters
import e00733_fit as fit
import e00733_model as model
import e00733_predictions as pr
import torch
from e00733_baselines import fit_clim


def _save_net(*, root: Path, arm: str, seed: int = 0) -> None:
    torch.manual_seed(seed)
    mode = fit.synthetic_mode(arm=arm)
    net = model.UntangledNet(seed=seed, path=mode != "absent", sign_head=True, clim_logit=-3.0)
    fit.save_state(
        state=net.state_dict(), path=fit.network_path(root=root, arm=arm, fold=None, seed=seed)
    )


def test_load_net_refuses_a_missing_fit(tmp_path: Path) -> None:
    """A job without a state file raises instead of predicting from random weights."""
    with pytest.raises(store.PredictionError, match="fit it before predicting"):
        pr.load_net(root=tmp_path, arm="YU", fold=None, seed=0, device="cpu")
    with pytest.raises(store.PredictionError):
        pr.load_net(root=tmp_path, arm="FU", fold="2025-11", seed=0, device="cpu")


def test_synthetic_prediction_is_written_once_and_read_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The archive of a synthetic fit holds one row per sample, passes the stored-output check, and is not rewritten."""
    segments = [make_segment(seed=1)]
    monkeypatch.setattr(world, "read_world", lambda **_k: (segments, {}))
    monkeypatch.setattr(world, "read_meta", lambda **_k: {"constants": {"mean": 0.0, "std": 1.0}})
    monkeypatch.setattr(adapters, "d8_index", lambda: 0)
    _save_net(root=tmp_path, arm="YU")
    path = pr.predict_synthetic_arm(
        root=tmp_path, arm="YU", seed=0, device="cpu", rows="validation"
    )
    assert path == synth.archive_path(root=tmp_path, arm="YU", seed=0, rows="validation")
    got = synth.read_arm(root=tmp_path, arm="YU", seeds=(0,), rows="validation")
    assert got is not None and got["p_neg"].shape[1] == 48 and got["p_neg"].min() >= 0.0
    stamp = path.stat().st_mtime_ns
    assert pr.predict_job(
        root=tmp_path, arm="YU", fold=None, seed=0, device="cpu", rows="validation"
    ) == [path]
    assert path.stat().st_mtime_ns == stamp
    with pytest.raises(store.PredictionError):
        pr.predict_job(root=tmp_path, arm="FU", fold=None, seed=0, device="cpu")
    with pytest.raises(store.PredictionError):
        pr.predict_fold(root=tmp_path, arm="M*", fold="2025-11", seed=0, device="cpu")
    with pytest.raises(store.PredictionError):
        pr.predict_fold(
            root=tmp_path, arm="F0r", fold="2025-11", seed=0, device="cpu", variants=("masked",)
        )


def test_run_score_writes_the_files_and_marks_validation_interim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With stubbed readings the score writes verdict.json and readings.json with R8 to R12 beside them."""
    segments = [make_segment(seed=1)]
    reading = scoring.Readings(
        contrasts=dict.fromkeys(stage.REAL_NAMES), plans=dict.fromkeys(stage.REAL_NAMES), guards_inputs=None,
        r8={"status": "void"}, r9={"status": "void"}, r10={"status": "void"},
    )  # fmt: skip
    monkeypatch.setattr(scoring, "real_readings", lambda **_k: reading)
    syn = {
        "contrasts": dict.fromkeys(("C3", "C4", "R5", "R6", "R7")),
        "sign_inputs": None,
        "log_loss": {},
        "r11": {"status": "void"},
        "n_scored": 0,
    }
    monkeypatch.setattr(synth, "synthetic_readings", lambda **_k: syn)
    verdict = stage.run_score(root=tmp_path, rows="validation")
    assert verdict["interim"] is True and verdict["rows"] == "validation"
    readings = json.loads((tmp_path / "scores" / "readings.json").read_text())
    assert {"R8", "R9", "R10", "R11", "plans", "mae_of_median"} <= set(readings)
    with pytest.raises(sc.ScoreError):
        stage.run_score(root=tmp_path, rows="test")
    assert (
        segments
        and core.EXPERIMENT_ID == "00733"
        and sy.STEPS == 48
        and np.isfinite(fit_clim(segments=segments).overall)
    )
