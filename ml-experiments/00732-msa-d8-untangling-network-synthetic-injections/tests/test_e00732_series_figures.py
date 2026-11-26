"""Test that the window picker never draws one window under two rules."""

import e00732_series_figures as sf
import numpy as np


def table(n: int) -> sf.WindowTable:
    """Return a table of `n` synthetic windows, every one a transfer window of every kind."""
    f = np.zeros(n)
    t = np.ones(n, dtype=bool)
    return sf.WindowTable(
        world=np.array(["both"] * n), engine_run=t, solar_out=t, source=np.array(["S2"] * n),
        depth=np.arange(n, dtype=float) + 1.0, near_zero=np.full(n, 10.0), evening=t,
        mean_p=f + 0.5,
        max_p=f + 0.5, gain_vs_s=np.arange(n, dtype=float), solar_error=f + 1.0, pv_max=f + 1.0,
        adds=t, removes=t, in_solar=t, other_bsp=t, solar_shape=t, mirrored=np.arange(n) % 2 == 0,
        step_gain_vs_un=np.arange(n, dtype=float), missed_noise_units=f + 5.0, real=~t,
    )  # fmt: skip


def test_no_transfer_window_repeats_when_windows_are_few() -> None:
    """With fewer distinct windows than slots, later rules get None rather than a repeat."""
    plan = sf.plan_windows(t=table(8))
    transfer = [
        plan[k] for k in ("W14", "W15", "W16", "W17", "W18", "W19", "W20", "W21", "W22", "W23")
    ]
    drawn = [i for i in transfer if i is not None]
    assert len(drawn) == len(set(drawn))
    assert None in transfer


def test_every_transfer_rule_is_filled_when_windows_are_many() -> None:
    """With enough windows, each of the ten transfer rules gets its own window."""
    plan = sf.plan_windows(t=table(60))
    transfer = [plan[f"W{k}"] for k in range(14, 24)]
    assert None not in transfer
    assert len(set(transfer)) == 10
