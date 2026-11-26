"""Tests of the arm specifications, the loss keeps, and the causal cut on a synthetic segment."""

import e00729_arms as arms729
import e00729_core as core729
import e00729_sim as sim
import e00730_arms as arms
import e00730_core as core
import e00730_leakage as leak
import numpy as np
import pytest
from world730 import make_calendar

STATS = arms729.InputStats(10.0, 3.0, 5.0, 2.0)


@pytest.fixture(scope="module")
def prepared_world() -> dict:
    cal = make_calendar()
    segment = sim.simulate_segment(
        cal=cal,
        month="2022-03",
        use=np.ones(len(cal.dates), dtype=bool),
        world="both",
        rng=np.random.default_rng(1),
    )
    out = {}
    for name, spec in arms.arm_specs(style="K").items():
        if spec.inputs == "enc":
            continue
        out[name] = arms.prepare_segments(
            segments=[segment], spec=spec, holidays=cal.holidays, stats=STATS
        )
    return out


def test_arms_of_the_plan_and_their_outputs() -> None:
    specs = arms.arm_specs(style="W")
    assert set(specs) == set(core.ARMS_NEW)
    assert specs["S"].n_outputs == 1
    assert specs["SD*"].n_outputs == 3
    assert specs["SD*"].style == "W"
    assert specs["SD*-C"].causal
    assert specs["SD*-lambda"].lam == core.LAMBDA_REPORT
    assert specs["SD-0"].style == "0"


def test_channel_counts() -> None:
    specs = arms.arm_specs(style="K")
    assert arms.n_channels(spec=specs["SD*"]) == arms.n_channels(spec=specs["S"])
    assert arms.n_channels(spec=specs["SD*-C"]) == arms.n_channels(spec=specs["SD*"]) + 1
    assert arms.n_channels(spec=specs["S-A"]) == 2
    enc = arms.n_channels(spec=specs["SDe-PT"])
    assert enc > core.ENCODER_INPUT_COLUMNS


def test_demand_head_is_never_an_input(prepared_world: dict) -> None:
    spec = arms.arm_specs(style="K")["SD*"]
    p = prepared_world["SD*"][0]
    day = int(p.days[3])
    w, t, _ = arms.build_one(prepared=p, dense_day=day, spec=spec)
    leak.assert_inputs_not_targets(windows=w[None], targets=t[None], spec=spec)
    assert t.shape == (6, 48)


def test_hidden_half_hours_keep_the_demand_loss_and_lose_the_sign_loss(
    prepared_world: dict,
) -> None:
    spec = arms.arm_specs(style="K")["SD*"]
    p = prepared_world["SD*"][0]
    day = int(p.days[3])
    hidden = np.zeros(96, dtype=bool)
    hidden[core729.MIDDLE_START + 10 : core729.MIDDLE_START + 20] = True
    w, t, keeps = arms.build_one(prepared=p, dense_day=day, spec=spec, hidden=hidden)
    assert (w[0, hidden] == 0).all()
    assert (w[1, hidden] == 0).all()
    assert (keeps[0, 10:20] == 0).all()
    assert (keeps[1, 10:20] == t[1, 10:20]).all()
    leak.assert_keeps_follow_hidden(
        hidden=hidden[None, core729.MIDDLE_START : core729.MIDDLE_START + 48],
        keeps=keeps[None],
        valid=t[None, 1] > 0.5,
    )


def test_an_arm_without_the_demand_output_has_no_demand_loss(prepared_world: dict) -> None:
    spec = arms.arm_specs(style="K")["S"]
    p = prepared_world["S"][0]
    _, _, keeps = arms.build_one(prepared=p, dense_day=int(p.days[3]), spec=spec)
    assert (keeps[1] == 0).all()


def test_causal_window_ignores_the_future(prepared_world: dict) -> None:
    spec = arms.arm_specs(style="K")["SD*-C"]
    p = prepared_world["SD*-C"][0]
    day = int(p.days[3])
    cut = core729.MIDDLE_START + 12
    w1, _, keeps = arms.build_one(prepared=p, dense_day=day, spec=spec, cut_pos=cut)
    p2 = prepared_world["SD*-C"][0]
    stack = p2.stack.copy()
    start = day * 48 - core729.MIDDLE_START
    stack[:, start + cut + 4 :] += 5.0
    scrambled = arms.PreparedSegment(**{**p2.__dict__, "stack": stack})
    w2, _, _ = arms.build_one(prepared=scrambled, dense_day=day, spec=spec, cut_pos=cut)
    names = arms.stack_names(spec=spec)
    observed = [i for i, n in enumerate(names) if n in core_observed()]
    assert (w1[observed, cut + 1 :] == 0).all() or np.allclose(
        w1[observed, cut + 4 :], w2[observed, cut + 4 :]
    )
    assert np.allclose(w1[:, : cut + 1][observed], w2[:, : cut + 1][observed])
    assert (keeps[0, 13:] == 0).all()
    assert w1[names.index(arms.FUTURE_ROW), cut + 1 :].min() == 1.0
    assert w1[names.index(arms.FUTURE_ROW), : cut + 1].max() == 0.0


def core_observed() -> set[str]:
    import e00730_masks as masks

    return set(masks.OBSERVED_ROWS)


def test_training_masks_half_of_the_windows(prepared_world: dict) -> None:
    spec = arms.arm_specs(style="W")["SD*"]
    rng = np.random.default_rng(0)
    picks = [(0, int(prepared_world["SD*"][0].days[2]))] * 400
    x, _, keeps = arms.make_batch(
        prepared=prepared_world["SD*"], picks=picks, spec=spec, rng=rng, training=True
    )
    masked = (keeps[:, 0] == 0).all(axis=1)
    assert 0.35 < masked.mean() < 0.65
    x_eval, _, keeps_eval = arms.make_batch(
        prepared=prepared_world["SD*"], picks=picks[:5], spec=spec, rng=rng, training=False
    )
    assert (keeps_eval[:, 0] > 0).any()
    assert x.shape[1] == arms.n_channels(spec=spec)
    assert x_eval.shape[2] == 96
