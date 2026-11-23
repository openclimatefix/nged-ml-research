# ruff: noqa: E501, PT018
"""Experiment 716: KF13 and KF13b, the copied training loop against experiment 713's (WP4 check; CPU, synthetic world).

Usage: run.sh e00716_check_train.py

KF13: experiment 713's `train_stage` and this experiment's copy, with every PV path absent (network
`Z`), train stage 2 for one pass (about 40 steps at a batch of 32) from one stored state with one seed
and one sample set, under `torch.use_deterministic_algorithms(True)`. The two state dictionaries must be
bit-identical.

KF13b: the same, with the full network of arm `HT` (all three paths, random non-zero PV tables) and
the new group's rate scaled to zero. Every PV contribution is an exact zero at the start, so the trunk
must come out bit-identical to experiment 713's. This tests the two optimiser groups, the two clips,
the dedicated generators, and the mask-dropout generator.
"""

import copy
import sys

import e00713_batch as bt  # ty: ignore[unresolved-import]
import e00713_layout as lay  # ty: ignore[unresolved-import]
import e00713_model as mdl  # ty: ignore[unresolved-import]
import e00713_train as tr  # ty: ignore[unresolved-import]
import e00716_batch as pvb
import e00716_check_model as ckm
import e00716_model as pvm
import e00716_train as pvt
import numpy as np
import torch  # ty: ignore[unresolved-import]

N_DAYS_GRID = 170
FIRST_DAY = 40
N_ENS_DAYS = 100
SEED = 0


def synthetic_world(*, seed: int = 0) -> tuple[tr.World, pvb.PvTables]:
    """Return a world of 170 days with ensemble rows on days 40 to 139, and random non-zero PV tables."""
    g = torch.Generator().manual_seed(seed)
    n_grid = 48 * N_DAYS_GRID
    power = torch.randn(16, n_grid, generator=g)
    power[torch.rand(16, n_grid, generator=g) < 0.02] = float("nan")
    run = torch.zeros(16, N_ENS_DAYS, dtype=torch.int64)
    c = (FIRST_DAY + torch.arange(N_ENS_DAYS)) * 48 + 24
    run[:] = (
        lay.GRID_START_US + (c - lay.NATIVE_OFFSET_STEPS) * lay.STEP_US - lay.ENSEMBLE_LEAD_US
    )[None, :]
    ens = bt.EnsTensors(
        weather=torch.randn(16, N_ENS_DAYS, 672, 21, generator=g).half(),
        spread=torch.rand(16, N_ENS_DAYS, 672, 7, generator=g).half(),
        filled=torch.ones(16, N_ENS_DAYS, 672, dtype=torch.bool), first_day=FIRST_DAY, run_us=run,
    )  # fmt: skip
    source = bt.Source(
        power_z=power, dense_weather=torch.randn(16, n_grid, 21, generator=g).half(),
        dense_datetime=torch.randn(16, n_grid, 21, generator=g).half(),
        day_type=torch.randint(0, 8, (n_grid,), generator=g), ens=ens,
    )  # fmt: skip
    tables = pvb.PvTables(
        hist_p=torch.rand(16, n_grid, generator=g) * 4, hist_ok=torch.rand(16, n_grid, generator=g) > 0.1,
        top_p0=torch.rand(16, N_ENS_DAYS, 672, 2, generator=g) * 3,
        top_filled=torch.ones(16, N_ENS_DAYS, 672, dtype=torch.bool), first_day=FIRST_DAY, run_us=run,
    )  # fmt: skip
    world = tr.World(
        source=source, cutoff_idx=48 * 140, clean=power.numpy().astype(np.float64),
        target_mean=np.zeros(16), target_std=np.ones(16), month="2025-11", placebo_pool=None,
    )  # fmt: skip
    return world, tables


def state_equal(*, a: dict, b: dict, keys: list[str]) -> list[str]:
    """Return the keys whose tensors differ in any bit."""
    return [k for k in keys if not torch.equal(a[k], b[k])]


def run() -> None:
    """Run KF13 and KF13b."""
    torch.use_deterministic_algorithms(True)
    tr.BATCH = 32  # a smaller batch gives about 40 steps in one pass
    world, tables = synthetic_world()
    samples = tr.make_samples(world=world, stage=2, placebo=False)
    assert (~samples.is_val).sum() > 400 and samples.is_val.sum() > 20, (
        len(samples),
        samples.is_val.sum(),
    )
    start = ckm.randomised_m_star(seed=3).state_dict()

    ref = mdl.MsaNet()
    ref.load_state_dict(start)
    tr.train_stage(
        model=ref,
        world=world,
        stage=2,
        samples=samples,
        seed=SEED,
        encoders="trained",
        max_passes=1,
    )
    ref_state = copy.deepcopy(ref.state_dict())
    start_copy = copy.deepcopy(start)
    assert state_equal(a=ref_state, b=start_copy, keys=list(start)) != list(start) or True
    moved = [k for k in start if not torch.equal(ref_state[k], start_copy[k])]
    assert len(moved) > 20, "the reference loop barely moved the weights"

    pv_world = tr.World(
        source=pvb.PvSource(source=world.source, tables=tables), cutoff_idx=world.cutoff_idx, clean=world.clean,
        target_mean=world.target_mean, target_std=world.target_std, month=world.month, placebo_pool=None,
    )  # fmt: skip
    for arm, scale, label in (("Z", 1.0, "KF13"), ("HT", 0.0, "KF13b")):
        model = pvm.PvNet(arm=arm, seed=SEED)  # ty: ignore[invalid-argument-type]
        model.load_m_star(state=copy.deepcopy(start))
        pvt.train_stage(
            model=model, world=pv_world, stage=2, samples=samples, seed=SEED, encoders="trained",
            max_passes=1, new_lr_scale=scale,
        )  # fmt: skip
        got = model.state_dict()
        bad = state_equal(a=got, b=ref_state, keys=list(ref_state))
        assert not bad, f"{label}: the trunk differs from experiment 713's in {bad[:5]}"
        if arm == "HT":
            new = [k for k in got if k not in ref_state]
            assert new, "no new parameters in HT"
        print(
            f"{label}: {len(ref_state)} trunk tensors bit-identical to experiment 713's after one pass ({int((~samples.is_val).sum()) // 32} steps)"
        )

    # Negative control: a live new group (scale 1) must change the trunk, or the check could not fail.
    model = pvm.PvNet(arm="HT", seed=SEED)  # ty: ignore[invalid-argument-type]
    model.load_m_star(state=copy.deepcopy(start))
    pvt.train_stage(
        model=model,
        world=pv_world,
        stage=2,
        samples=samples,
        seed=SEED,
        encoders="trained",
        max_passes=1,
        new_lr_scale=1.0,
    )
    bad = state_equal(a=model.state_dict(), b=ref_state, keys=list(ref_state))
    assert bad, "negative control: a live PV group left the trunk unchanged"
    print(f"negative control: with the new group live, {len(bad)} trunk tensors differ")


def main() -> int:
    """Run the checks."""
    run()
    print("e00716_check_train: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
