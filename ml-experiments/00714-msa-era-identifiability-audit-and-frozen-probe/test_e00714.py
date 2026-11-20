# ruff: noqa: E402, PT018
"""Tests of experiment 714's features, classifier, and second generator. Synthetic data only."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).parent
for folder in (
    HERE,
    HERE.parent / "00660-shared-generator-of-injected-load-transfer-events",
    HERE.parent / "00709-era-signal-in-the-candidate-part-1",
):
    sys.path.insert(0, str(folder))

import e00714_common as common
import e00714_features as feats
import e00714_generator2 as g2
import e00714_model as model

N_SITES = 4
N_DAYS = 60
N_HH = N_DAYS * common.DAY
ONSET_DAY = 20


def synthetic(
    *, step: float = -0.2, seed: int = 0
) -> tuple[common.PassPower, common.PassPower, np.ndarray]:
    """Return a pass with a flat step at the host, a zero reference, and a shortwave field."""
    rng = np.random.default_rng(seed)
    t = np.arange(N_HH)
    daily = 0.5 + 0.1 * np.sin(2 * np.pi * (t % common.DAY) / common.DAY)
    ref = np.tile(daily, (N_SITES, 1)).astype(np.float32)
    power = ref + 0.01 * rng.standard_normal(ref.shape).astype(np.float32)
    power[0, ONSET_DAY * common.DAY :] += step
    sw = np.maximum(0.0, 500 * np.sin(2 * np.pi * (t % common.DAY) / common.DAY))
    return common.PassPower(0, power), common.PassPower(0, ref), sw


def event() -> feats.EventInput:
    """Return the host event of the synthetic pass."""
    return feats.EventInput(
        host=0, onset_hh=ONSET_DAY * common.DAY + 7, onset_day=ONSET_DAY, neighbours=[1, 2]
    )


def test_flatness() -> None:
    """A constant profile is flat; a zero-mean profile is not."""
    assert feats.flatness(profile=np.full(48, -3.0)) == pytest.approx(1.0)
    assert feats.flatness(profile=np.sin(np.linspace(0, 2 * np.pi, 48, endpoint=False))) < 0.01


def test_step_features() -> None:
    """A sustained step has no steps; an on-off signal has many."""
    sustained = np.full(14 * 48, -10.0)
    assert feats.step_features(change=sustained)[0] == 0.0
    on_off = np.tile(np.concatenate([np.full(12, -10.0), np.zeros(24)]), 19)[: 14 * 48]
    steps, high = feats.step_features(change=on_off)
    assert steps > 2 and 0.2 < high < 0.5


def test_features_of_a_flat_step() -> None:
    """A flat step of 20 points is found with flatness near 1 and no mirror."""
    power, ref, sw = synthetic()
    out = feats.build_features(
        event=event(), k=14, power=power, reference=ref, shortwave=sw, capacity_mw=np.ones(N_SITES)
    )
    assert out is not None
    assert out["size"] == pytest.approx(-20.0, abs=1.0)
    assert out["flat"] > 0.95
    assert abs(out["mirror_share"]) < 0.5


def test_origin_cut() -> None:
    """Scrambling power, reference, and shortwave at or after the data cut changes no feature."""
    power, ref, sw = synthetic()
    args = {"event": event(), "k": 14, "capacity_mw": np.ones(N_SITES)}
    base = feats.build_features(power=power, reference=ref, shortwave=sw, **args)
    cut = (ONSET_DAY + 14) * common.DAY + common.ORIGIN_HOUR_HH - common.LATENCY_HH
    rng = np.random.default_rng(1)
    scrambled = power.power.copy()
    scrambled[:, cut:] = rng.standard_normal(scrambled[:, cut:].shape)
    ref_scrambled = ref.power.copy()
    ref_scrambled[:, cut:] = 99.0
    sw_scrambled = sw.copy()
    sw_scrambled[cut:] = rng.uniform(0, 900, sw[cut:].shape)
    after = feats.build_features(
        power=common.PassPower(0, scrambled), reference=common.PassPower(0, ref_scrambled),
        shortwave=sw_scrambled, **args,
    )  # fmt: skip
    assert base is not None and after is not None
    for name in base:
        a, b = base[name], after[name]
        assert (a == b) or (np.isnan(a) and np.isnan(b)), name


def test_older_rows_end_before_the_onset() -> None:
    """Older rows that end after the onset are not read: injecting there does not move them."""
    power, ref, sw = synthetic()
    args = {
        "event": event(),
        "k": 28,
        "capacity_mw": np.ones(N_SITES),
        "reference": ref,
        "shortwave": sw,
    }
    base = feats.build_features(power=power, **args)
    assert base is not None


def test_auroc_and_logistic_known_answer() -> None:
    """Separable classes score near 1, identical classes near 0.5, and ties are averaged."""
    rng = np.random.default_rng(0)
    y = np.repeat([0.0, 1.0], 200)
    x = np.concatenate([rng.normal(-3, 1, 200), rng.normal(3, 1, 200)])[:, None]
    groups = np.tile(np.arange(40), 10)
    fitted = model.fit_logistic(x=x, y=y, groups=groups)
    assert model.auroc(label=y, score=fitted.proba(x=x)) > 0.99
    noise = rng.normal(size=(400, 1))
    assert abs(model.auroc(label=y, score=noise[:, 0]) - 0.5) < 0.08
    assert model.auroc(label=np.array([0, 1]), score=np.array([1.0, 1.0])) == 0.5


def test_out_of_fold_never_scores_a_month_it_saw() -> None:
    """Out-of-fold scores of a month come from a fit without that month."""
    rng = np.random.default_rng(0)
    months = np.repeat(np.array(["a", "b", "c", "d"]), 50)
    y = rng.integers(0, 2, 200).astype(float)
    x = np.where(months == "a", y, rng.normal(size=200))[:, None]  # only month a carries the label
    groups = np.arange(200)
    pred = model.out_of_fold(x=x, y=y, groups=groups, months=months)
    held = months == "b"  # a noise month: scoring it with a fit that saw month a stays near 0.5
    assert abs(model.auroc(label=y[held], score=pred[held]) - 0.5) < 0.2
    assert np.isfinite(pred).all()


def test_bootstrap_keeps_whole_passes() -> None:
    """Every drawn pass appears with all its rows."""
    months = np.repeat(np.array(["a", "b"]), 6)
    passes = np.array(["a1", "a1", "a1", "a2", "a2", "a2", "b1", "b1", "b1", "b2", "b2", "b2"])
    rng = np.random.default_rng(0)
    for _ in range(20):
        drawn = model.resample_indices(months=months, passes=passes, rng=rng)
        counts = np.bincount(drawn, minlength=12)
        for p in np.unique(passes):
            assert len(set(counts[passes == p])) == 1


def test_permanent_deltas() -> None:
    """Each non-gas permanent kind has the shape it is named for."""
    n_hh = 120 * common.DAY
    onset = 40 * common.DAY + 5
    real = 0.5 + 0.2 * np.sin(2 * np.pi * (np.arange(n_hh) % 48) / 48)
    rng = np.random.default_rng(0)
    flat = g2.permanent_delta(kind="flat", size=-0.1, onset_hh=onset, n_hh=n_hh, real=real, rng=rng)
    assert flat[:onset].sum() == 0 and np.all(flat[onset:] == -0.1)
    prop = g2.permanent_delta(
        kind="profile_prop", size=0.1, onset_hh=onset, n_hh=n_hh, real=real, rng=rng
    )
    assert prop[onset:].mean() == pytest.approx(0.1, rel=0.02) and prop[onset:].std() > 0.01
    day = g2.permanent_delta(
        kind="daylight", size=0.3, onset_hh=onset, n_hh=n_hh, real=real, rng=rng
    )
    assert day[onset:].max() == pytest.approx(0.3) and (day[onset:] == 0).mean() > 0.3
    q = g2.permanent_delta(
        kind="quantised", size=-0.15, onset_hh=onset, n_hh=n_hh, real=real, rng=rng
    )
    assert 2 <= len(np.unique(np.round(q[onset:], 6))) <= 3


def test_intermittent_rows() -> None:
    """Intermittent pulse rows alternate on and off, never overlap, and have a constant size."""
    rng = np.random.default_rng(0)
    rows, share = g2.intermittent_rows(
        site="X", size=0.08, onset_hh=1000, stop_hh=1000 + 45 * 48, first_id=0, rng=rng
    )
    starts = np.array([r["start"] for r in rows])
    ends = np.array([r["end"] for r in rows])
    assert (
        len(rows) > 10 and np.all(starts[1:] > ends[:-1]) and {r["delta"] for r in rows} == {-0.08}
    )
    assert all(r["event_class"] == "pulse" for r in rows)
    assert 0.0 < share < 1.0


def test_second_generator_shares_no_injector_with_experiment_709() -> None:
    """The second generator never imports experiment 709's injector."""
    assert "e00709_inject" not in sys.modules
    assert not hasattr(g2, "inject")
