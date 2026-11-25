"""Known-answer tests of the Brier score, the harm and detection checks, and the contrasts."""

from types import SimpleNamespace

import e00730_core as core
import e00730_score as score
import numpy as np
import pytest


def rows(x: np.ndarray, r: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(x=x, r=r, n=x.size)


def test_brier_and_log_loss_known_answers() -> None:
    x = np.array([-1.0, -1.0, 1.0, 1.0])
    p = np.array([1.0, 0.0, 0.0, 0.5])
    assert score.row_brier(p=p, x=x).tolist() == [0.0, 1.0, 0.0, 0.25]
    ll = score.row_log_loss(p=np.array([0.5, 0.5]), x=np.array([-1.0, 1.0]))
    assert ll == pytest.approx(np.log(2.0))
    assert np.isfinite(score.row_log_loss(p=np.array([0.0]), x=np.array([-1.0]))).all()


def test_zeros_are_penalised_by_the_brier_score_on_reverse_flow() -> None:
    x = np.array([-5.0] * 10 + [5.0] * 90)
    assert score.mean_brier(p=np.zeros(100), x=x) == pytest.approx(0.1)


def test_harm_check_reads_only_decidable_import_rows() -> None:
    mvA = core.DECIDABLE_READING_POINTS
    r = np.array([2 * mvA, 2 * mvA, 0.1 * mvA, 2 * mvA])
    x = np.array([5.0, 5.0, 5.0, -5.0])
    p = np.array([0.0, 0.2, 0.9, 0.9])
    out = score.harm_check(rows=rows(x, r), p=p, limit=0.15)
    assert out["n"] == 2
    assert out["mean_p"] == pytest.approx(0.1)
    assert out["share_above_half"] == 0.0
    assert out["pass"]
    assert not score.harm_check(rows=rows(x, r), p=p, limit=0.05)["pass"]


def test_detection_bar_is_the_mean_probability_on_decidable_reverse_rows() -> None:
    mvA = core.DECIDABLE_READING_POINTS
    r = np.full(4, 2 * mvA)
    x = np.array([-5.0, -5.0, -5.0, 5.0])
    out = score.detection(rows=rows(x, r), p=np.array([0.9, 0.8, 0.7, 0.1]))
    assert out["d"] == pytest.approx(0.8)
    assert out["pass"]
    assert not score.detection(rows=rows(x, r), p=np.array([0.5, 0.5, 0.5, 0.1]))["pass"]


def test_a_perfectly_calibrated_forecast_passes_the_calibration_bar() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(0.0, 1.0, 40_000)
    x = np.where(rng.random(p.size) < p, -10.0, 10.0)
    ok = score.calibration(rows=rows(x, np.ones_like(x)), p=p, bar=0.01)
    assert ok["pass"]
    bad = score.calibration(rows=rows(x, np.ones_like(x)), p=np.clip(p * 0.2, 0, 1), bar=0.01)
    assert not bad["pass"]


def test_causal_rows_and_visible_position() -> None:
    hh = np.arange(48)
    assert score.causal_keep(hh=hh, k=12).sum() == 36
    assert score.causal_keep(hh=hh, k=48).all()
    assert score.causal_last_visible(k=0, middle_start=24) == 23
    assert score.causal_last_visible(k=6, middle_start=24) == 29


def test_demand_error_splits_hidden_and_visible() -> None:
    mu = np.array([[1.0, 2.0, 3.0, 4.0]])
    base = np.array([[1.0, 1.0, 1.0, 1.0]])
    valid = np.array([[True, True, True, False]])
    hidden = np.array([[False, True, True, True]])
    out = score.demand_mae(mu=mu, base=base, valid=valid, hidden=hidden)
    assert out == {"hidden": 1.5, "visible": 0.0, "all": 1.0}


def test_the_counted_contrasts_are_those_of_the_plan() -> None:
    assert set(core.CONTRASTS) == {f"C{i}" for i in range(1, 8)}
    assert core.CONTRASTS["C1"] == ("SD*", "S")
    assert core.CONTRASTS["C7"] == ("SD*-C", "SD*")
