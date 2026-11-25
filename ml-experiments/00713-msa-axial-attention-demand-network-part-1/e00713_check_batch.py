# ruff: noqa: E501, PT018
"""Experiment 713: CPU checks of batch assembly on random arrays (WP4, KA5, KA6). Exit code 0 means pass.

The arrays are random, so the check needs no data and no fit. It checks the cheap per-batch
assertions, the placebo rows, the block-dropout masks, check KA5 (poisoned copies leave every input and
every model output unchanged, and the negative control with a cap of 0 changes them), and the KA6
canary (a target row from the next day's ensemble run raises). It skips when `torch` is missing.
"""

import importlib.util
import sys

if importlib.util.find_spec("torch") is None:
    print("SKIP: torch is not installed")
    sys.exit(0)

import e00713_batch as bt
import e00713_layout as lay
import numpy as np
import torch

N_GRID = 48 * 420
FIRST_DAY = 5


def fake_source() -> bt.Source:
    """Return a source of random arrays with a power gap and an ensemble table."""
    g = torch.Generator().manual_seed(0)
    power = torch.randn(8, N_GRID, generator=g)
    power[:, 1000:1100] = float("nan")
    ens_days = N_GRID // 48 - FIRST_DAY
    ens = bt.EnsTensors(
        weather=torch.randn(8, ens_days, 672, 21, generator=g).half(),
        spread=torch.rand(8, ens_days, 672, 7, generator=g).half(),
        filled=torch.ones(8, ens_days, 672, dtype=torch.bool),
        first_day=FIRST_DAY,
        run_us=(
            lay.GRID_START_US
            + ((FIRST_DAY + torch.arange(ens_days)) * 48 + 24 - 6) * lay.STEP_US
            - lay.ENSEMBLE_LEAD_US
        )[None]
        .expand(8, -1)
        .clone(),
    )
    return bt.Source(
        power_z=power, dense_weather=torch.randn(8, N_GRID, 21, generator=g).half(),
        dense_datetime=torch.randn(8, N_GRID, 21, generator=g).half(),
        day_type=torch.randint(0, 8, (N_GRID,), generator=g), ens=ens,
    )  # fmt: skip


def main() -> int:
    """Run the checks."""
    src = fake_source()
    rng = np.random.default_rng(0)
    n = 32
    c = torch.from_numpy((rng.integers(390, N_GRID // 48 - 16, n) * 48 + 24).astype(np.int64))
    prim = torch.from_numpy(rng.integers(0, 8, n))
    lead = torch.from_numpy(rng.integers(0, 14, n))
    for ensemble in (False, True):
        batch = src.assemble(primary=prim, c_idx=c, lead_day=lead, ensemble=ensemble)
        inp = batch.inputs
        assert inp.top_weather.shape == (n, 48, 21) and inp.hist_weather.shape == (n, 14, 48, 21)
        assert inp.hist_power.shape == (n, 14, 48) and inp.hist_visible.shape == (n, 14, 48)
        assert not bool(inp.hist_visible[:, 0, 43:].any()), (
            "columns 43 to 47 of the age-2 row must be masked"
        )
        assert bool(inp.hist_visible[:, 1:, :].any()), "older rows must be visible"
        print(
            bt.poison_check(
                source=src,
                batch=batch,
                ages=None,
                age_exists=None,
                ensemble=ensemble,
                lead_day=lead,
            )
        )
    # block dropout
    drop_all = torch.zeros(n, dtype=torch.bool)
    drop_all[:8] = True
    newest = torch.zeros(n, dtype=torch.bool)
    newest[8:16] = True
    d = src.assemble(
        primary=prim,
        c_idx=c,
        lead_day=lead,
        ensemble=True,
        mask_all_power=drop_all,
        mask_newest_row=newest,
    )
    assert not bool(d.inputs.hist_visible[:8].any()) and not bool(
        d.inputs.hist_visible[8:16, 0].any()
    )
    assert bool(d.inputs.hist_visible[8:16, 1:].any())
    # placebo rows: ages 16 or more, a true past day with its own weather
    ages = torch.from_numpy(
        np.stack(
            [
                lay.placebo_ages(primary=int(p), origin_day=int(cc) // 48)
                for p, cc in zip(prim, c, strict=True)
            ]
        )
    )
    p = src.assemble(
        primary=prim,
        c_idx=c,
        lead_day=lead,
        ensemble=True,
        ages=ages,
        age_exists=torch.ones(n, 14, dtype=torch.bool),
    )
    assert bool((p.hist_idx[:, :, 0] <= (c - 48 * 16)[:, None]).all()), (
        "a placebo row is younger than 16 days"
    )
    # the target is NaN beyond the stage's cut and where the power gap is
    cut = int(c[0]) + 10
    t = src.assemble(primary=prim, c_idx=c, lead_day=lead, ensemble=True, target_end=cut)
    assert bool(torch.isnan(t.target[t.target_idx >= cut]).all())
    # KA6 canary
    print("KA6 canary raised:", bt.ka6_canary(source=src, primary=prim, c_idx=c, lead_day=lead))
    print("batch checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
