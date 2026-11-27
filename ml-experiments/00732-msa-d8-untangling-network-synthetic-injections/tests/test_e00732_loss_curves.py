# ruff: noqa: E501
"""Tests of the loss-curve figure on a stored-history fixture shaped like experiment 730's
`fits/S_unet_s0.json` (no torch).
"""

import json
from typing import Any

import e00732_core as core
import e00732_figures as figures
import numpy as np
import pytest

SHAPE_730 = {  # the keys of experiment 730's fit record, with this experiment's two history dictionaries added
    "arm": "S", "arch": "unet", "seed": 0, "epochs": 117, "best_epoch": 92, "best_val_brier": 0.0098, "n_weights": 1065857,
    "seconds": 49.7, "seconds_per_epoch": 0.42, "peak_memory_gb": 0.21,
}  # fmt: skip


def record(
    *,
    arm: str = "U*",
    epochs: int = 40,
    best: int = 30,
    terms: tuple[str, ...] = ("sign", "solar", "step", "total"),
) -> dict:
    rng = np.random.default_rng(0)
    history = list(
        0.04
        - 0.001 * np.minimum(np.arange(epochs), best - 1)
        + 0.0005 * np.maximum(np.arange(epochs) - (best - 1), 0)
    )
    hist = {
        k: [float(x) for x in np.exp(-np.arange(epochs) / 20) + 0.01 * rng.random(epochs)]
        for k in terms
    }
    return {**SHAPE_730, "arm": arm, "arch": core.ARMS[arm].arch, "epochs": epochs, "best_epoch": best, "history": history,
            "train_terms": hist, "val_terms": {k: [v * 1.1 for v in vs] for k, vs in hist.items()}}  # fmt: skip


def test_the_marker_is_the_stored_best_epoch_and_the_rows_cover_every_term_and_split() -> None:
    r = record()
    rows = figures.curve_rows(record=r)
    assert rows.height == 2 * 4 * 40
    assert set(rows["split"]) == {"train", "validation"}
    assert figures.best_epoch_of(record=r) == r["best_epoch"] == 30
    chart = figures.loss_panel(record=r)
    spec = chart.to_dict()
    rule = next(layer for layer in spec["layer"] if layer["mark"]["type"] == "rule")
    data = spec["datasets"][rule["data"]["name"]]
    assert [d["epoch"] for d in data] == [30]


def test_a_best_epoch_that_does_not_match_the_history_fails_loudly() -> None:
    r = record()
    r["best_epoch"] = 31
    with pytest.raises(figures.HistoryError, match="best_epoch differs"):
        figures.best_epoch_of(record=r)
    r["best_epoch"] = 99
    with pytest.raises(figures.HistoryError, match="outside"):
        figures.curve_rows(record=r)


def test_an_empty_or_missing_history_fails_loudly() -> None:
    r = record()
    for key in ("train_terms", "val_terms"):
        broken = {**r, key: {}}
        with pytest.raises(figures.HistoryError, match="no per-epoch loss history"):
            figures.curve_rows(record=broken)
    legacy = {
        k: v for k, v in r.items() if k not in ("train_terms", "val_terms")
    }  # a record in experiment 730's shape
    with pytest.raises(figures.HistoryError, match="no per-epoch loss history"):
        figures.loss_panel(record=legacy)
    with pytest.raises(figures.HistoryError, match="validation history"):
        figures.best_epoch_of(record={**r, "history": []})
    with pytest.raises(figures.HistoryError, match="no fit record"):
        figures.loss_curves_chart(records=[])


def test_a_short_nonfinite_or_mismatched_curve_fails() -> None:
    r = record()
    r["train_terms"]["sign"] = r["train_terms"]["sign"][:-1]
    with pytest.raises(figures.HistoryError, match="finite values"):
        figures.curve_rows(record=r)
    r = record()
    r["val_terms"]["solar"][3] = float("nan")
    with pytest.raises(figures.HistoryError, match="finite values"):
        figures.curve_rows(record=r)
    r = record()
    del r["val_terms"]["step"]
    with pytest.raises(figures.HistoryError, match="different loss terms"):
        figures.curve_rows(record=r)


def test_the_stage_reads_stored_records_only_and_names_the_seed_in_the_title(tmp_path: Any) -> None:
    fits = tmp_path / "fits"
    fits.mkdir()
    for arm, spec in core.ARMS.items():
        terms = (
            ("sign", "total")
            if arm == "S+"
            else ("sign", "solar", "step", "size", "injection", "night", "monotone", "total")
        )
        (fits / f"{core.job_name(arm=arm, arch=spec.arch, seed=0)}.json").write_text(
            json.dumps(record(arm=arm, terms=terms))
        )
    logged = []
    written = figures.stage_loss_curves(root=tmp_path, log=logged.append)
    svg = written[0].read_text()
    assert written[0].name == "loss_curves.svg"
    assert "<svg" in svg
    assert "seed 0" in svg
    assert "stored model at epoch 30" in svg
    assert len(logged) == 1
    (fits / f"{core.job_name(arm='U-SG', arch='unet', seed=0)}.json").unlink()
    with pytest.raises(FileNotFoundError, match=r"U-SG|USG"):
        figures.stage_loss_curves(root=tmp_path, log=logged.append)


def test_checks_fail_when_an_arm_is_missing() -> None:
    checks = figures.loss_curve_checks(records=[record(arm="U*")])
    assert not checks["every_arm_drawn"]
