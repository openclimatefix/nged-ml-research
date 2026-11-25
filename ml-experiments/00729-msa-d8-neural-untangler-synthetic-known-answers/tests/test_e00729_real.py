"""Tests of the real-data checks K1, K2, R2, R3 on synthetic arrays."""

import e00729_core as core
import e00729_real as real
import numpy as np
import pytest


def test_f1_counts_each_kind_of_error() -> None:
    truth = np.array([True, True, False, False])
    assert real.f1_score(truth=truth, flag=truth) == 1.0
    assert real.f1_score(truth=truth, flag=~truth) == 0.0
    assert real.f1_score(truth=truth, flag=np.array([True, False, True, False])) == pytest.approx(
        0.5
    )
    assert real.f1_score(truth=~truth & False, flag=~truth & False) == 0.0


def k1(*, head: np.ndarray, record: np.ndarray, m2: np.ndarray | None = None) -> dict:
    running = record > core.K1_RUNNING_POINTS
    return real.k1_engine_head(
        head_median=head, record=record, running=running, m2_flag=running if m2 is None else m2
    )


def record_of(n_run: int) -> np.ndarray:
    return np.r_[np.full(n_run, 30.0), np.zeros(1000)]


def test_k1_passes_a_good_head_and_fails_each_bar() -> None:
    rec = record_of(core.K1_MIN_RUNNING + 50)
    assert k1(head=rec.copy(), record=rec)["status"] == "pass"
    flat = k1(head=np.zeros_like(rec), record=rec)
    assert flat["status"] == "fail"
    assert not flat["bar_mae"]
    bad_f1 = k1(head=rec.copy(), record=rec, m2=rec > 10.0)
    assert bad_f1["status"] == "pass"
    # F1 bar alone: the head's error is small enough, but it flags fewer half-hours than M2 does
    head = rec.copy()
    head[: len(rec) // 20] = 9.0
    alone = k1(head=head, record=rec)
    assert alone["bar_mae"]
    assert not alone["bar_f1"]
    assert alone["status"] == "fail"


def test_k1_is_void_below_the_minimum_running_half_hours() -> None:
    rec = record_of(core.K1_MIN_RUNNING - 1)
    assert k1(head=rec.copy(), record=rec)["status"] == "void"
    assert k1(head=np.zeros_like(rec), record=rec)["status"] == "void"
    exact = record_of(core.K1_MIN_RUNNING)
    assert k1(head=exact.copy(), record=exact)["status"] == "pass"


def test_k2_fails_on_a_sign_or_an_engine_alarm_and_passes_otherwise() -> None:
    n = 10_000
    ok = real.k2_placebo(p_neg=np.zeros(n), head_median=np.zeros(n))
    assert ok["status"] == "pass"
    sign = np.zeros(n)
    sign[: int(core.K2_SIGN_SHARE * n) + 5] = 0.9
    assert real.k2_placebo(p_neg=sign, head_median=np.zeros(n))["status"] == "fail"
    eng = np.zeros(n)
    eng[: int(core.K2_ENGINE_SHARE * n) + 5] = 20.0
    assert real.k2_placebo(p_neg=np.zeros(n), head_median=eng)["status"] == "fail"
    edge = np.zeros(n)
    edge[: int(core.K2_ENGINE_SHARE * n)] = 20.0
    assert real.k2_placebo(p_neg=np.zeros(n), head_median=edge)["status"] == "pass"


def test_refold_share_counts_medians_within_the_tolerance() -> None:
    reading = np.full(4, 20.0)
    median = np.array([20.0, 21.9, 22.5, 10.0])
    assert real.refold_share(median=median, reading=reading) == pytest.approx(0.5, abs=0.3)
    far = np.full(4, 40.0)
    assert real.refold_share(median=far, reading=reading) == 0.0


def test_group_summaries_split_by_label() -> None:
    groups = np.array(["a", "a", "b", "b"])
    out = real.r3_sign_by_group(p_neg=np.array([0.9, 0.1, 0.1, 0.1]), groups=groups)
    assert out["a"] == pytest.approx(0.5)
    assert out["b"] == 0.0
    r2 = real.r2_by_group(
        median=np.array([20.0, 40.0, 20.0, 20.0]), reading=np.full(4, 20.0), groups=groups
    )
    assert r2["a"] == pytest.approx(0.5)
    assert r2["b"] == 1.0


def test_gaussian_mixture_median_is_the_middle_of_the_mixture() -> None:
    mu = np.array([[5.0, 5.0], [-5.0, 5.0], [5.0, 5.0]])
    med = real.gaussian_mixture_median(mu=mu, sigma=np.ones_like(mu))
    assert med.shape == (2,)
    assert med[0] == pytest.approx(4.3255, abs=0.01)  # solves 2/3 Phi(m-5) + 1/3 = 0.5
    assert med[1] == pytest.approx(5.0, abs=1e-6)
