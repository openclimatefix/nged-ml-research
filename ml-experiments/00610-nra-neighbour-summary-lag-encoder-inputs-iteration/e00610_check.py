# ruff: noqa: E501, D103, PT018, ANN001, ANN201
"""Experiment 606: CPU checks of the extra columns and the network (no data-folder rows needed).

1. A known-answer test on a toy world of 4 Primaries (one with two neighbours, one with none): a shift
   of known size on Primary 0, mirrored onto its neighbours, reaches the gap, the weighted averages,
   the neighbour summaries, and the own-plus-neighbours sum (which cancels for a pure transfer).
2. A weighted average reads no time after the lag's own time.
3. The expected-missing flag is 1 only where the weather run's origin is absent.
4. Each input-removing arm and each extra-column arm loads A1's step-2 state dict, and with the
   zero-initialised new columns the arm without an input removed reproduces A1's output exactly.

Usage: python e00610_check.py [STEP2_FILE]
"""

import sys
from pathlib import Path

import e00591_model as aux  # ty: ignore[unresolved-import]
import e00598_signals as s598  # ty: ignore[unresolved-import]
import e00610_model as model  # ty: ignore[unresolved-import]
import e00610_signals as sig  # ty: ignore[unresolved-import]
import numpy as np
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

STEP2 = (
    ML_EXPERIMENTS_DATA_DIR
    / "00591-shared-lag-encoder-auxiliary-shift-depth-loss/chain_shared/work/step2/P5/2025-12_s0.pt"
)
ALL_GROUPS = ("gap", "sustain", "ew", "exp", "nb", "bn")


def toy_world(*, n_days: int = 30, shift: float = 0.2) -> dict:
    """Return 4 Primaries: 0 has neighbours 1 and 2, 1 and 2 have neighbour 0, 3 has none."""
    n_g = n_days * sig.PER_DAY
    t = np.arange(n_g)
    wave = (0.5 + 0.1 * np.sin(2 * np.pi * t / sig.PER_DAY)).astype(np.float32)
    median = np.tile(wave, (4, 1))
    prim, org, tgt, med = [], [], [], []
    for p in range(4):
        for d in range(n_days):
            o = d * sig.PER_DAY + s598.ORIGIN_OFFSET
            for lead in range(1, 49):
                if o + lead < n_g:
                    prim.append(p)
                    org.append(o)
                    tgt.append(o + lead)
                    med.append(median[p, o + lead])
    table = s598.build_lag_free_table(
        prim=np.array(prim), origin_idx=np.array(org), target_idx=np.array(tgt),
        median=np.array(med, np.float32), n_primaries=4, n_grid=n_g,
    )  # fmt: skip
    cap = np.array([100.0, 50.0, 80.0, 60.0])
    on = 12 * sig.PER_DAY
    shift_arr = np.zeros_like(median)
    shift_arr[0, on:] = shift
    power_lag = median + shift_arr
    nb = s598.Neighbours(of={0: [1, 2], 1: [0], 2: [0], 3: []}, capacity=cap)
    # the transfer leaves Primary 0 and lands wholly on neighbour 1
    mirror = s598.mirrored_power(base=power_lag, shift=shift_arr, choice={0: 1}, capacity=cap)
    weather_origin = np.ones((4, n_g), np.int64)
    weather_origin[2, :] = 0
    return {
        "n_g": n_g, "table": table, "power": median, "power_lag": power_lag, "power_sig": mirror,
        "nb": nb, "cap": cap, "shift": shift, "on": on, "weather_origin": weather_origin,
    }  # fmt: skip


def block_of(*, w: dict, prim, day_lag, week_lag, power_lag=None, power_sig=None, miss=None):
    n = len(prim)
    miss = np.zeros(n, bool) if miss is None else miss
    o = 25 * sig.PER_DAY + s598.ORIGIN_OFFSET
    return sig.signal_columns(
        prim=np.asarray(prim), origin_idx=np.full(n, o), day_idx=np.asarray(day_lag),
        week_idx=np.asarray(week_lag), day_missing=miss, week_missing=miss,
        power_lag=w["power_lag"] if power_lag is None else power_lag,
        power_sig=w["power_sig"] if power_sig is None else power_sig, table=w["table"], nb=w["nb"],
        weather_origin=w["weather_origin"], columns=list(range(sig.N_ALL)),
    )  # fmt: skip


def check_known_answer() -> None:
    w = toy_world()
    o = 25 * sig.PER_DAY + s598.ORIGIN_OFFSET
    prim = [0, 1, 2, 3]
    day_lag = np.full(4, o - 48 - 5)  # inside the shift, 13 days after it began
    week_lag = np.full(4, o - 7 * sig.PER_DAY)  # inside the shift (day 18 of 12+)
    block = block_of(w=w, prim=prim, day_lag=day_lag, week_lag=week_lag)
    c, sh, sc = sig.COL, w["shift"], sig.SCALE
    np.testing.assert_allclose(block[0, c["gap_day"]], sh * sc, atol=1e-4)
    # a gap that has been `sh` for about 13 days: the short spans are close to `sh`, the long span lags it
    ew = [block[0, c[f"ew{h}_day"]] / sc for h in (1, 3, 7)]
    assert ew[0] > 0.99 * sh and ew[0] >= ew[1] > ew[2] > 0.5 * sh, ew
    # the transfer lands on neighbour 1 in the neighbour power only: its own lag is not shifted
    np.testing.assert_allclose(block[1, c["gap_day"]], 0.0, atol=1e-3)
    # Primary 1 sees Primary 0 (gap +shift, scaled to 1's capacity) in its neighbour summary
    np.testing.assert_allclose(
        block[1, c["nb_sum_gap_day"]], sh * w["cap"][0] / w["cap"][1] * sc, atol=1e-3
    )
    # Primary 0 sees neighbours 1 (the mirrored shift, scaled back to 0's capacity: -shift) and 2 (zero)
    np.testing.assert_allclose(block[0, c["nb_sum_gap_day"]], -sh * sc, atol=1e-3)
    np.testing.assert_allclose(block[0, c["nb_mean_gap_day"]], -sh * sc / 2, atol=1e-3)
    np.testing.assert_allclose(block[0, c["nb_min_gap_day"]], -sh * sc, atol=1e-3)
    np.testing.assert_allclose(block[0, c["nb_max_gap_day"]], 0.0, atol=1e-3)
    # a pure transfer: own plus the neighbours' sum cancels, at the gap and at every span
    for q in sig.QUANTITIES:
        np.testing.assert_allclose(block[0, c[f"own_plus_nb_{q}"]], 0.0, atol=1e-3)
    assert block[0, c["nb_count"]] == 2 and block[0, c["nb_none"]] == 0
    # the single best opposite-sign neighbour is neighbour 1
    np.testing.assert_allclose(block[0, c["bn_gap_day"]], -sh * sc, atol=1e-3)
    np.testing.assert_allclose(block[0, c["own_plus_bn_gap_day"]], 0.0, atol=1e-3)
    # Primary 3 has no neighbour: zeros plus the flag
    assert block[3, c["nb_none"]] == 1 and block[3, c["bn_none"]] == 1
    assert (block[3, [c[n] for n in sig.NB_NAMES if n.startswith("nb_sum")]] == 0).all()
    print("known-answer test passed: gap, weighted averages, neighbour summaries, own-plus-sum")


def check_ew_is_causal() -> None:
    w = toy_world()
    o = 25 * sig.PER_DAY + s598.ORIGIN_OFFSET
    lag = o - 48 - 5
    base = block_of(w=w, prim=[0], day_lag=[lag], week_lag=[lag])
    later = w["power_lag"].copy()
    later[:, lag + 1 :] += 5.0  # change everything after the lag's own time
    changed = block_of(w=w, prim=[0], day_lag=[lag], week_lag=[lag], power_lag=later)
    own = [sig.COL[n] for n in sig.OWN_NAMES if not n.startswith("sustain")]
    assert np.array_equal(base[0, own], changed[0, own]), "an own column read a later time"
    print("causality test passed: no own column reads a time after the lag's own time")


def check_expected_missing() -> None:
    w = toy_world()
    o = 25 * sig.PER_DAY + s598.ORIGIN_OFFSET
    block = block_of(w=w, prim=[0, 2], day_lag=[o - 60, o - 60], week_lag=[o - 400, o - 400])
    c = sig.COL
    assert block[0, c["exp_missing_day"]] == 0 and block[1, c["exp_missing_day"]] == 1
    assert block[1, c["exp_missing_week"]] == 1
    print("expected-missing flag test passed")


def check_masking() -> None:
    w = toy_world()
    o = 25 * sig.PER_DAY + s598.ORIGIN_OFFSET
    block = block_of(w=w, prim=[0, 2], day_lag=[o - 60, o - 60], week_lag=[o - 400, o - 400])
    cols = list(range(sig.N_ALL))
    sig.mask_lag_columns(
        block=block, columns=cols, day_masked=np.ones(2, bool), week_masked=np.ones(2, bool)
    )
    for j, name in enumerate(sig.ALL_NAMES):
        if name.startswith("exp_missing"):
            continue
        if name in {"nb_none", "bn_none"}:
            assert (block[:, j] == 1).all(), name
            continue
        assert (block[:, j] == 0).all(), name
    assert block[1, sig.COL["exp_missing_day"]] == 1, "masking cleared the expected-missing flag"
    print("masking test passed")


def check_young_lag_rejected() -> None:
    w = toy_world()
    o = 25 * sig.PER_DAY + s598.ORIGIN_OFFSET
    try:
        block_of(w=w, prim=[0], day_lag=[o - 10], week_lag=[o - 400])
    except AssertionError:
        print("a lag under 24 h old is rejected")
        return
    raise SystemExit("a lag younger than 24 h was accepted")


def check_model_is_a1_at_init(*, path: Path) -> None:
    state = torch.load(path, map_location="cpu")
    n_primaries = state["embedding.weight"].shape[0]
    shapes = {
        "n_primaries": n_primaries, "n_knots": state["knots_old"].shape[1] + 1,
        "n_b1": state["weather.0.weight"].shape[1] - 7 - 8,
    }  # fmt: skip
    import e00545_model as m545  # ty: ignore[unresolved-import]

    shapes["n_b1"] = state["weather.0.weight"].shape[1] - 7 - m545.EMBEDDING
    torch.manual_seed(0)
    a1 = aux.AuxLagMdn(stage="ft", kind="shared", **shapes)
    a1.load_state_dict(state)
    n_grid, rows = 400, 64
    table = torch.randn(n_primaries, n_grid, 28)
    a1.table = table
    label_col = a1.label_col
    x = torch.randn(rows, label_col)
    x[:, 43] = torch.randint(0, n_grid, (rows,)).float()
    x[:, 44] = torch.randint(0, n_grid, (rows,)).float()
    prim = torch.randint(0, n_primaries, (rows,))
    a1.eval()
    with torch.no_grad():
        ref = a1(x, prim)
    for n_extra in (4, 6, 20, 55):
        net = model.PermLagMdn(stage="ft", inputs=model.InputSet(n_extra=n_extra), **shapes)
        net.load_state_dict(state)
        net.table = table
        net.eval()
        with torch.no_grad():
            got = net(torch.cat([x, torch.randn(rows, n_extra) * 5], dim=1), prim)
        assert torch.allclose(ref, got, atol=1e-6), (n_extra, float((ref - got).abs().max()))
        with torch.no_grad():
            net.lag[0][0].weight[:, -1] = 1.0
            moved = net(torch.cat([x, torch.randn(rows, n_extra) * 5], dim=1), prim)
        assert not torch.allclose(ref, moved, atol=1e-4), "the extra columns are not wired in"
    # each removed input changes the output of the loaded A1 weights, and a kept input does not
    for name, inputs in (
        ("no age", model.InputSet(use_age=False)),
        ("no id", model.InputSet(use_id=False)),
        ("no expected", model.InputSet(use_expected=False)),
    ):
        net = model.PermLagMdn(stage="ft", inputs=inputs, **shapes)
        net.load_state_dict(state)
        net.table = table
        net.eval()
        with torch.no_grad():
            got = net(x, prim)
        assert not torch.allclose(ref, got, atol=1e-4), f"{name} left the output unchanged"
    same = model.PermLagMdn(stage="ft", inputs=model.InputSet(), **shapes)
    same.load_state_dict(state)
    same.table = table
    same.eval()
    with torch.no_grad():
        assert torch.allclose(ref, same(x, prim), atol=1e-6)
    print(
        "network loads A1's weights: extra columns start inert; each removed input changes the output"
    )


if __name__ == "__main__":
    check_known_answer()
    check_ew_is_causal()
    check_expected_missing()
    check_masking()
    check_young_lag_rejected()
    check_model_is_a1_at_init(path=Path(sys.argv[1]) if len(sys.argv) > 1 else STEP2)
    print("ALL CHECKS PASSED")
