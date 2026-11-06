# ruff: noqa: E501, D103, PT018
"""Experiment 598: CPU checks of the signal build and the network (no data folder rows needed).

1. A known-answer test: a shift of known size, added to the lag power, to the fresh power, and (mirrored)
   to a neighbour, reaches each signal column with the expected value, and no signal reads a time at
   or after the row's origin.
2. The network loads an A1 step-2 state dict, and with zero-initialised new columns reproduces A1's
   output exactly for any signal values.
3. The mirrored-neighbour construction conserves megawatts.

Usage: python e00598_check.py [STEP2_FILE]
"""

import sys
from pathlib import Path

import e00591_model as aux  # ty: ignore[unresolved-import]
import e00598_model as model  # ty: ignore[unresolved-import]
import e00598_signals as sig  # ty: ignore[unresolved-import]
import numpy as np
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

STEP2 = (
    ML_EXPERIMENTS_DATA_DIR
    / "00591-shared-lag-encoder-auxiliary-shift-depth-loss/chain_shared/work/step2/P5/2025-12_s0.pt"
)


def toy_world(*, n_days: int = 20, shift: float = 0.2) -> dict:
    """Return a tiny world: 3 Primaries, `T` = a daily wave, power = T + a shift on Primary 0 from day 6 on."""
    n_g = n_days * sig.PER_DAY
    t = np.arange(n_g)
    wave = (0.5 + 0.1 * np.sin(2 * np.pi * t / sig.PER_DAY)).astype(np.float32)
    median = np.tile(wave, (3, 1))
    # forecast origins every day at 09:00; each covers the next 48 half-hours
    prim, org, tgt, med = [], [], [], []
    for p in range(3):
        for d in range(n_days):
            o = d * sig.PER_DAY + sig.ORIGIN_OFFSET
            for lead in range(1, 49):
                if o + lead < n_g:
                    prim.append(p)
                    org.append(o)
                    tgt.append(o + lead)
                    med.append(median[p, o + lead])
    table = sig.build_lag_free_table(
        prim=np.array(prim), origin_idx=np.array(org), target_idx=np.array(tgt),
        median=np.array(med, np.float32), n_primaries=3, n_grid=n_g,
    )  # fmt: skip
    cap = np.array([100.0, 50.0, 80.0])
    power = median.copy()
    on = 6 * sig.PER_DAY
    shift_arr = np.zeros_like(power)
    shift_arr[0, on:] = shift
    power_lag = power + shift_arr
    nb = sig.Neighbours(of={0: [1], 1: [0], 2: []}, capacity=cap)
    mirror = sig.mirrored_power(base=power_lag, shift=shift_arr, choice={0: 1}, capacity=cap)
    return {
        "n_g": n_g, "table": table, "power": power, "power_lag": power_lag, "power_sig": mirror,
        "nb": nb, "cap": cap, "shift": shift, "on": on,
    }  # fmt: skip


def check_known_answer() -> None:
    w = toy_world()
    d = 10  # origin day
    o = d * sig.PER_DAY + sig.ORIGIN_OFFSET
    n = 4
    prim = np.array([0, 0, 1, 2])
    origin = np.full(n, o)
    day_lag = np.array([o + 10 - 2 * 48 + 0, o + 10 - 2 * 48, o + 10 - 2 * 48, o + 10 - 2 * 48])
    day_lag = np.minimum(day_lag, o - 48)
    week_lag = np.full(n, 3 * sig.PER_DAY + 5)  # before the shift starts on day 6
    miss = np.zeros(n, bool)
    widths = np.zeros((n, 2), np.float32)
    block = sig.signal_block(
        prim=prim, origin_idx=origin, day_idx=day_lag, week_idx=week_lag, day_missing=miss,
        week_missing=miss, power_lag=w["power_lag"], power_sig=w["power_sig"], table=w["table"],
        widths=widths, nb=w["nb"],
    )  # fmt: skip
    col = sig.COL
    shift, scale = w["shift"], sig.SCALE
    # Primary 0, day lag inside the shifted span: disagreement = shift; the lag is 2+ days into the shift
    assert day_lag[0] >= w["on"], (day_lag[0], w["on"])
    np.testing.assert_allclose(block[0, col["dis_day"]], shift * scale, atol=1e-4)
    # week lag before the shift: zero disagreement
    np.testing.assert_allclose(block[0, col["dis_week"]], 0.0, atol=1e-5)
    # fresh window [o-48, o-8) is wholly inside the shift (it starts on day 6)
    np.testing.assert_allclose(block[0, col["fresh"]], shift * scale, atol=1e-4)
    assert block[0, col["fresh_missing"]] == 0
    # sustain: the day lag's 3 earlier days: the lag sits ~day 8, so the earlier days (7, 6, 5) are shifted for 2 of 3
    assert 0 < block[0, col["sustain_day"]] <= 1
    # Primary 1 is Primary 0's neighbour: its own fresh reading holds the mirrored shift, its neighbour signal is Primary 0's
    mirrored = -shift * w["cap"][0] / w["cap"][1]
    np.testing.assert_allclose(block[2, col["fresh"]], mirrored * scale, atol=1e-3)
    np.testing.assert_allclose(
        block[2, col["nb_fresh"]], -shift * w["cap"][0] / w["cap"][1] * -1 * scale * -1, atol=1e-3
    )
    # Primary 0 sees Primary 1's fresh reading (the mirrored shift), mirrored back to Primary 0's capacity: +shift
    np.testing.assert_allclose(block[0, col["nb_fresh"]], shift * scale, atol=1e-3)
    assert block[0, col["nb_present"]] == 1 and block[3, col["nb_present"]] == 0
    assert block[3, col["nb_fresh"]] == 0
    # an absent T at the day lag sets the missing flag and zeroes the signal; a present T leaves it unset
    assert block[0, col["dis_day_missing"]] == 0 and block[0, col["nb_fresh_missing"]] == 0
    holes = sig.LagFreeTable(median=w["table"].median.copy(), origin_idx=w["table"].origin_idx)
    holes.median[:, :] = np.nan
    gap = sig.signal_block(
        prim=prim, origin_idx=origin, day_idx=day_lag, week_idx=week_lag, day_missing=miss,
        week_missing=miss, power_lag=w["power_lag"], power_sig=w["power_sig"], table=holes,
        widths=widths, nb=w["nb"],
    )  # fmt: skip
    live = ~miss
    assert (gap[live, col["dis_day_missing"]] == 1).all() and (gap[live, col["dis_day"]] == 0).all()
    assert (gap[live, col["dis_week_missing"]] == 1).all()
    assert gap[0, col["nb_fresh_missing"]] == 1 and gap[3, col["nb_fresh_missing"]] == 0
    assert gap[0, col["nb_dis_day_missing"]] == 1
    # masked lag zeroes the lag-derived columns
    sig.mask_lag_columns(block=block, day_masked=np.ones(n, bool), week_masked=np.ones(n, bool))
    assert (
        block[
            :,
            [
                col["dis_day"],
                col["dis_week"],
                col["sustain_day"],
                col["sustain_week"],
                col["nb_dis_day"],
            ],
        ]
        == 0
    ).all()
    # a lag at the cap age (24 h) is allowed; one younger must fail
    try:
        sig.signal_block(
            prim=prim[:1], origin_idx=origin[:1], day_idx=origin[:1] - 10, week_idx=week_lag[:1],
            day_missing=miss[:1], week_missing=miss[:1], power_lag=w["power_lag"],
            power_sig=w["power_sig"], table=w["table"], widths=widths[:1], nb=w["nb"],
        )  # fmt: skip
    except AssertionError:
        pass
    else:
        raise SystemExit("a lag younger than 24 h was accepted")
    print(
        "known-answer test passed: shift", shift, "reaches dis, fresh, sustain, neighbour columns"
    )


def check_table_no_lookahead() -> None:
    w = toy_world()
    n = w["table"].check_no_lookahead()
    assert n > 0
    # T at a grid index comes from the freshest origin before it
    t = w["table"]
    i = 8 * sig.PER_DAY + 30
    assert t.origin_idx[0, i] == 8 * sig.PER_DAY + sig.ORIGIN_OFFSET
    assert (
        t.origin_idx[0, 8 * sig.PER_DAY + sig.ORIGIN_OFFSET] == 7 * sig.PER_DAY + sig.ORIGIN_OFFSET
    )
    print("T takes the freshest earlier origin and never a later one:", n, "entries")


def check_mirror_conserves_megawatts() -> None:
    w = toy_world()
    base = w["power_lag"]
    out = w["power_sig"]
    delta_mw = (out - base) * w["cap"][:, None]
    own_mw = (base - w["power"]) * w["cap"][:, None]
    np.testing.assert_allclose(delta_mw.sum(axis=0) + own_mw.sum(axis=0), 0.0, atol=1e-3)
    print("mirrored shifts conserve megawatts")


def check_noise_rule() -> None:
    rng = np.random.default_rng(0)
    n = 20_000
    prim = rng.integers(0, 2, n)
    tr = np.zeros((n, sig.N_SIGNALS), np.float32)
    te = np.zeros((n, sig.N_SIGNALS), np.float32)
    for name in sig.NOISED:
        tr[:, sig.COL[name]] = rng.standard_normal(n) * np.where(prim == 0, 0.5, 1.0)
        te[:, sig.COL[name]] = rng.standard_normal(n)
    sd = sig.train_test_sd(
        blocks={"train": tr, "test": te}, prim={"train": prim, "test": prim}, n_primaries=2
    )
    nz = sig.noise_sd(sd=sd)
    assert (nz[:, 0] > 0.7).all() and (nz[:, 1] == 0).all(), nz  # Primary 0 is under-dispersed
    sig.add_training_noise(block=tr, prim=prim, sd=nz, seed=0)
    after = sig.train_test_sd(
        blocks={"train": tr, "test": te}, prim={"train": prim, "test": prim}, n_primaries=2
    )
    np.testing.assert_allclose(after["train"][:, 0], after["test"][:, 0], rtol=0.05)
    print(
        "training noise restores the under-dispersed Primary's spread; the matched Primary gets none"
    )


def check_placebo_keeps_marginals() -> None:
    rng = np.random.default_rng(1)
    src = np.stack(
        [
            rng.integers(0, 2, 5_000),
            rng.standard_normal(5_000) * 3,
            np.clip(rng.exponential(1, 5_000), 0, 2),
        ],
        axis=1,
    ).astype(np.float32)
    out = sig.placebo_block(source=src, n=8_000, seed=0)
    assert set(np.unique(out[:, 0])) <= {0.0, 1.0}
    assert out[:, 2].max() <= 2.0
    np.testing.assert_allclose(out.std(axis=0), src.std(axis=0), rtol=0.06)
    print("placebo block keeps binary columns binary, clipped columns clipped, and the variances")


def check_model_is_a1_at_init(*, path: Path) -> None:
    state = torch.load(path, map_location="cpu")
    shapes = model.shapes_from_state(state=state)
    n_sig = sig.N_SIGNALS
    torch.manual_seed(0)
    a1 = aux.AuxLagMdn(stage="ft", kind="shared", **shapes)
    a1.load_state_dict(state)
    s = model.SignalLagMdn(stage="ft", n_sig=n_sig, **shapes)
    s.load_state_dict(state)
    n_grid, rows = 400, 64
    table = torch.randn(shapes["n_primaries"], n_grid, 28)
    a1.table, s.table = table, table
    label_col = a1.label_col
    x = torch.randn(rows, label_col)
    x[:, 43] = torch.randint(0, n_grid, (rows,)).float()
    x[:, 44] = torch.randint(0, n_grid, (rows,)).float()
    prim = torch.randint(0, shapes["n_primaries"], (rows,))
    a1.eval()
    s.eval()
    with torch.no_grad():
        ref = a1(x, prim)
        got = s(torch.cat([x, torch.randn(rows, n_sig) * 5], dim=1), prim)
        again = s(torch.cat([x, torch.zeros(rows, n_sig)], dim=1), prim)
    assert torch.equal(ref, got) or torch.allclose(ref, got, atol=1e-6), float(
        (ref - got).abs().max()
    )
    assert torch.allclose(ref, again, atol=1e-6)
    # the new columns are the only difference in the parameter set, and they are zero
    extra = s.lag[0][0].weight[:, -s.extra :]
    assert float(extra.detach().abs().max()) == 0.0
    # a trainable change in a new column moves the output (the columns are wired in)
    with torch.no_grad():
        s.lag[0][0].weight[:, -s.extra] = 1.0
        moved = s(torch.cat([x, torch.randn(rows, n_sig) * 5], dim=1), prim)
    assert not torch.allclose(ref, moved, atol=1e-4)
    print(
        "network loads A1's step-2 weights and gives A1's output exactly (max abs diff",
        float((ref - got).abs().max()),
        ")",
    )


if __name__ == "__main__":
    check_known_answer()
    check_table_no_lookahead()
    check_mirror_conserves_megawatts()
    check_noise_rule()
    check_placebo_keeps_marginals()
    check_model_is_a1_at_init(path=Path(sys.argv[1]) if len(sys.argv) > 1 else STEP2)
    print("ALL CHECKS PASSED")
