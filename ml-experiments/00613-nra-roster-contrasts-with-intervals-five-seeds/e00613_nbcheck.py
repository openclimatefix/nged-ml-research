# ruff: noqa: E501, D103, PT018, ANN001, ANN201, PLR0915
"""Experiment 613 (copy of 610): CPU checks of the extra columns and the network (no data-folder rows needed).

1. A known-answer test on a toy world of 4 Primaries (one with two neighbours, one with none): a shift
   of known size on Primary 0, mirrored onto its neighbours, reaches the gap, the exponentially
   weighted and box averages, the neighbour summaries, the opposite-sign share, and the
   own-plus-neighbours sum (which cancels for a pure transfer).
2. Every average reads no time after the lag's own time, and the box average equals a direct mean.
3. The expected-missing flag is 1 only where the weather run's origin is absent.
4. Dropping signals sets the "no information" value and leaves the expected-missing flags alone.
5. Each network variant (extra columns, bottleneck, label head) loads A1's step-2 state dict and
   reproduces A1's output exactly at initialisation; the label head's loss is finite and trains.

Usage: python e00613_nbcheck.py [STEP2_FILE]
"""

import sys
from pathlib import Path

import e00591_model as aux  # ty: ignore[unresolved-import]
import e00598_signals as s598  # ty: ignore[unresolved-import]
import e00613_nbmodel as model  # ty: ignore[unresolved-import]
import e00613_nbsignals as sig  # ty: ignore[unresolved-import]
import numpy as np
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

STEP2 = (
    ML_EXPERIMENTS_DATA_DIR
    / "00591-shared-lag-encoder-auxiliary-shift-depth-loss/chain_shared/work/step2/P5/2025-12_s0.pt"
)


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
    # box means: the shift began about 12 days before the lag, so 1, 3, 7 days are full and 14 days is about 12/14
    bx = [block[0, c[f"bx{d}_day"]] / sc for d in (1, 3, 7, 14)]
    np.testing.assert_allclose(bx[:3], [sh] * 3, atol=1e-3)
    assert 0.8 * sh < bx[3] < 0.95 * sh, bx
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
    # Primary 0: neighbour 1 has the opposite sign (-shift, over the 5% threshold), neighbour 2 is zero
    np.testing.assert_allclose(block[0, c["nbopp_gap_day"]], 0.5 * sc, atol=1e-3)
    # Primary 1: its only neighbour (0) has gap +shift scaled to 1's capacity and Primary 1's own gap is 0
    assert block[1, c["nbopp_gap_day"]] == 0
    # Primary 3 has no neighbour: zeros plus the flag
    assert block[3, c["nb_none"]] == 1
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
    assert any(n.startswith("bx28") for n in sig.OWN_NAMES)
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


def check_box_average() -> None:
    rng = np.random.default_rng(1)
    res = rng.normal(size=(2, 5 * sig.PER_DAY)).astype(np.float32)
    res[0, 30:60] = np.nan
    got = sig.box_average(res=res, days=2)
    i = 4 * sig.PER_DAY
    np.testing.assert_allclose(
        got[0, i], np.nanmean(res[0, i - 2 * sig.PER_DAY + 1 : i + 1]), rtol=1e-4
    )
    assert np.isnan(got[0, 30]) or np.isfinite(got[0, 30])  # early rows may be NaN
    print("box-average test passed")


def check_drop_signals() -> None:
    w = toy_world()
    o = 25 * sig.PER_DAY + s598.ORIGIN_OFFSET
    cols = sig.columns_of(groups=("gap", "own:ew1,bx3", "exp", "nb:gap,ew1"))
    n = 6
    block = sig.signal_columns(
        prim=np.zeros(n, int), origin_idx=np.full(n, o), day_idx=np.full(n, o - 60),
        week_idx=np.full(n, o - 400), day_missing=np.zeros(n, bool), week_missing=np.zeros(n, bool),
        power_lag=w["power_lag"], power_sig=w["power_sig"], table=w["table"], nb=w["nb"],
        weather_origin=np.zeros((4, w["n_g"]), np.int64), columns=cols,
    )  # fmt: skip
    rows = np.array([True, False, True, False, False, False])
    before = block.copy()
    sig.drop_signals(block=block, columns=cols, rows=rows)
    names = [sig.ALL_NAMES[c] for c in cols]
    for j, name in enumerate(names):
        if name.startswith("exp_missing"):
            assert np.array_equal(block[:, j], before[:, j]), name
            continue
        want = 1.0 if (name.endswith("_missing") or name == "nb_none") else 0.0
        assert (block[rows, j] == want).all(), name
        assert np.array_equal(block[~rows, j], before[~rows, j]), name
    print("signal-dropout test passed: dropped rows hold the no-information value")


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
        if name == "nb_none":
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
        for variant in (
            model.InputSet(n_extra=n_extra),
            model.InputSet(n_extra=n_extra, bottleneck=True),
            model.InputSet(n_extra=n_extra, label_head=True),
        ):
            net = model.PermLagMdn(stage="ft", inputs=variant, **shapes)
            net.load_state_dict(state)
            net.table = table
            net.eval()
            with torch.no_grad():
                got = net(torch.cat([x, torch.randn(rows, n_extra) * 5], dim=1), prim)
            assert torch.allclose(ref, got, atol=1e-6), (variant, float((ref - got).abs().max()))
            with torch.no_grad():
                net.lag[0][0].weight[:, -1] = 1.0
                moved = net(torch.cat([x, torch.randn(rows, n_extra) * 5], dim=1), prim)
            assert not torch.allclose(ref, moved, atol=1e-4), "the extra columns are not wired in"
    # the label head: labels in training mode give a finite loss that trains, and are ignored in eval mode
    n_extra = 6
    net = model.PermLagMdn(
        stage="ft", inputs=model.InputSet(n_extra=n_extra, label_head=True), **shapes
    )
    net.load_state_dict(state)
    net.table = table
    xe = torch.cat([x, torch.randn(rows, n_extra)], dim=1)
    nan = float("nan")
    labels = torch.tensor([[1.0, 0.2, 0.0, 0.0]] * 32 + [[0.0, 0.0, nan, nan]] * 32)
    net.train()
    out = net(torch.cat([xe, labels], dim=1), prim)
    loss = model.STASH.pop("shift")
    assert loss.shape == (rows,) and torch.isfinite(loss).all()
    (out.mean() + loss.mean()).backward()
    assert (
        net.shift_head[0].weight.grad is not None and net.shift_head[0].weight.grad.abs().sum() > 0
    )
    net.eval()
    with torch.no_grad():
        out_eval = net(torch.cat([xe, labels], dim=1), prim)
        assert "shift" not in model.STASH
        assert torch.allclose(out_eval, net(xe, prim)), "labels reached the forecast"
        probs = net.shift_probability(x=xe, primary=prim)
    assert probs.shape == (rows, 2) and ((probs >= 0) & (probs <= 1)).all()
    print(
        "network loads A1's weights: every variant starts inert; the label head trains and its labels never reach a forecast"
    )


if __name__ == "__main__":
    check_known_answer()
    check_ew_is_causal()
    check_box_average()
    check_drop_signals()
    check_expected_missing()
    check_masking()
    check_young_lag_rejected()
    check_model_is_a1_at_init(path=Path(sys.argv[1]) if len(sys.argv) > 1 else STEP2)
    print("ALL CHECKS PASSED")
