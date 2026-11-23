# ruff: noqa: E501
"""Experiment 718, WP5a: check KR6 (the copied loop equals experiment 713's, and stage 3's shifts) on the CPU.

KR6 runs experiment 713's `train_stage` and F9's `train_stage` with the rows of ages 2 to 15 (arm `Z`)
from the same seed and samples for one pass of stage 1 under `torch.use_deterministic_algorithms`,
and requires bit-identical shared parameters. It then compares the stage 3 history power of F9's
`synthetic_source` with experiment 713's, which must be bit-identical on every row that `M*` read, and
checks the half-batch normalisation of the memory rule. Usage: e00718_check_train.py
"""

import sys

import e00713_model as mdl  # ty: ignore[unresolved-import]
import e00713_train as tr  # ty: ignore[unresolved-import]
import e00718_check_batch as cb
import e00718_model as f9
import e00718_retrieval as ret
import e00718_tables as tab
import e00718_train as ftr
import numpy as np
import torch  # ty: ignore[unresolved-import]

torch.set_num_threads(1)
SMALL_BATCH = 32
N_DAYS_SAMPLES = 80
FIRST_DAY = 30


def synthetic_world(*, source, month: str = "2025-11") -> tr.World:  # noqa: ANN001
    """Return a `World` on the synthetic source with a random clean array."""
    rng = np.random.default_rng(5)
    n_grid = source.n_grid
    clean = rng.normal(size=(16, n_grid)) * 0.2 + 0.5
    clean[rng.random(clean.shape) < 0.02] = np.nan
    return tr.World(
        source=source, cutoff_idx=n_grid - 1, clean=clean, target_mean=np.full(16, 0.5), target_std=np.full(16, 0.2),
        month=month, placebo_pool=None,
    )  # fmt: skip


def stage1_samples(*, world: tr.World) -> tr.SampleSet:
    """Return all (Primary, origin day) pairs of 80 days, every tenth held out."""
    days = np.arange(FIRST_DAY, FIRST_DAY + N_DAYS_SAMPLES)
    prim, dd = np.meshgrid(np.arange(16), days, indexing="ij")
    prim, c = prim.ravel(), dd.ravel() * 48 + 24
    return tr.SampleSet(primary=prim, c_idx=c, is_val=(np.arange(len(prim)) % 10 == 0))


def check_loop_equals_713() -> list[str]:
    """KR6 (a): one pass of stage 1 gives bit-identical shared parameters."""
    torch.use_deterministic_algorithms(True)
    old = tr.BATCH
    tr.BATCH = SMALL_BATCH
    try:
        source = cb.synthetic_source()
        world = synthetic_world(source=source)
        samples = stage1_samples(world=world)
        arrays = tab.pool_arrays_from_source(source=source, with_calendar=False)
        provider = tab.era5_provider(arrays=arrays)
        z_table = ret.build_table(
            arm="Z",
            arrays=arrays,
            primaries=np.arange(16),
            days=np.arange(FIRST_DAY, FIRST_DAY + N_DAYS_SAMPLES),
            target_provider=provider,
        )
        torch.manual_seed(0)
        base = mdl.MsaNet()
        torch.manual_seed(0)
        net = f9.F9Net(seed=0)
        tr.train_stage(
            model=base,
            world=world,
            stage=1,
            samples=samples,
            seed=0,
            encoders="random",
            max_passes=1,
        )
        ftr.train_stage(
            model=net,
            world=world,
            tables={"era5": z_table, "ens": z_table},
            arm="Z",
            stage=1,
            samples=samples,
            seed=0,
            encoders="random",
            max_passes=1,
        )
    finally:
        tr.BATCH = old
    mine, theirs = net.state_dict(), base.state_dict()
    n_equal = 0
    for key, value in theirs.items():
        if mine[key].shape == value.shape:
            assert torch.equal(mine[key], value), f"{key} differs after one pass"
        elif key in {"age_emb.weight", "beta_age"}:
            assert torch.equal(mine[key][:14], value), key
        else:
            assert torch.equal(mine[key][:, :15], value), key
        n_equal += value.numel()
    steps = int(np.ceil((~samples.is_val).sum() / SMALL_BATCH))
    assert n_equal == f9.SHARED_PARAMETERS
    moved = float(
        (base.age_emb.weight - mdl.MsaNet().age_emb.weight).abs().max()
    )  # a different seed's init, so a positive control that the loop trained
    assert moved > 0
    return [
        f"KR6 (a): {steps} optimiser steps of stage 1 (batch {SMALL_BATCH}, CPU, deterministic algorithms): all {n_equal:,} shared parameters are bit-identical to experiment 713's loop"
    ]


def check_stage3_power() -> list[str]:
    """KR6 (b): stage 3's history power equals experiment 713's wherever `M*` read it, and gains events over the older rows."""
    source = cb.synthetic_source()
    world = synthetic_world(source=source)
    days = np.arange(cb.FIRST_ORIGIN_DAY, cb.FIRST_ORIGIN_DAY + cb.N_ORIGIN_DAYS)
    prim, dd = np.meshgrid(np.arange(16), days, indexing="ij")
    samples = tr.SampleSet(
        primary=prim.ravel(), c_idx=dd.ravel() * 48 + 24, is_val=np.zeros(prim.size, bool)
    )
    ref = tr.synthetic_source(world=world, samples=samples).power_in
    lines = []
    for arm in ("U", "UX"):
        mine = ftr.synthetic_source(world=world, samples=samples, tables={}, arm=arm).power_in
        first = int(samples.c_idx.min()) - tr.SYNTHETIC_FIRST_MARGIN
        assert torch.equal(
            mine[:, first:].nan_to_num(nan=1e9),
            ref[:, first:].nan_to_num(nan=1e9),
        ), f"{arm}: the shifts of experiment 713 changed on rows that M* read"
        older = not torch.equal(mine[:, :first].nan_to_num(), ref[:, :first].nan_to_num())
        assert older, f"{arm}: the second schedule added no event over the older rows"
        lo = 0 if arm == "UX" else max(0, int(samples.c_idx.min()) - 48 * ftr.SECOND_SCHEDULE_DAYS)
        assert torch.equal(mine[:, :lo].nan_to_num(), ref[:, :lo].nan_to_num()), (
            f"{arm}: an event lies before the second schedule's range"
        )
        lines.append(
            f"KR6 (b): {arm}: stage 3 history power is bit-identical to experiment 713's from index {first}, and differs only on [{lo}, {first}) below it"
        )
    return lines


def check_half_batches() -> str:
    """The half-batch loss, divided by the weight sum of the full batch, gives the full batch's gradient (and the naive average does not)."""
    torch.manual_seed(1)
    net = f9.F9Net(seed=1).eval()
    inp = __import__("e00718_check_model").random_inputs(batch=8, n_history=14, seed=2)
    target = torch.randn(8, 48)
    w = torch.rand(8, 48) * (torch.rand(8, 48) < 0.5)
    w[:4] *= 3.0  # unequal weight sums between the halves

    def grads(*, rows: slice, norm: float) -> list[torch.Tensor]:
        net.zero_grad()
        sub = f9.F9Inputs(
            **{k: (v[rows] if v is not None else None) for k, v in inp.__dict__.items()}
        )
        nll = mdl.per_token_nll(out=net(sub).out, target=target[rows])
        ((w[rows] * nll).sum() / norm).backward()
        return [p.grad.clone() for p in net.parameters() if p.grad is not None]

    full = grads(rows=slice(0, 8), norm=float(w.sum()))
    a, b = (
        grads(rows=slice(0, 4), norm=float(w.sum())),
        grads(rows=slice(4, 8), norm=float(w.sum())),
    )
    err = max(float((f - (x + y)).abs().max()) for f, x, y in zip(full, a, b, strict=True))
    assert err <= 1e-6, err
    na, nb = (
        grads(rows=slice(0, 4), norm=float(w[:4].sum())),
        grads(rows=slice(4, 8), norm=float(w[4:].sum())),
    )
    naive = max(
        float((f - 0.5 * (x + y)).abs().max()) for f, x, y in zip(full, na, nb, strict=True)
    )
    assert naive > 1e-4, (
        "the naive average of half-batch losses equals the full-batch loss on this draw"
    )
    return f"half-batch gradients normalised by the full weight sum equal the full batch to {err:.1e}; the average of separately normalised halves is off by {naive:.1e}"


def check_accumulate_runs() -> str:
    """A pass with `accumulate=2` runs and trains."""
    torch.use_deterministic_algorithms(True)
    old = tr.BATCH
    tr.BATCH = SMALL_BATCH
    try:
        source = cb.synthetic_source()
        world = synthetic_world(source=source)
        samples = stage1_samples(world=world)
        arrays = tab.pool_arrays_from_source(source=source, with_calendar=False)
        table = ret.build_table(
            arm="Z",
            arrays=arrays,
            primaries=np.arange(16),
            days=np.arange(FIRST_DAY, FIRST_DAY + N_DAYS_SAMPLES),
            target_provider=tab.era5_provider(arrays=arrays),
        )
        torch.manual_seed(0)
        net = f9.F9Net(seed=0)
        before = net.state_dict()["w_q.weight"].clone()
        log = ftr.train_stage(
            model=net,
            world=world,
            tables={"era5": table, "ens": table},
            arm="Z",
            stage=1,
            samples=samples,
            seed=0,
            encoders="random",
            max_passes=1,
            accumulate=2,
        )
    finally:
        tr.BATCH = old
    assert not torch.equal(before, net.state_dict()["w_q.weight"]), "accumulate=2 did not train"
    return f"a pass with two half-batches per step ran and trained (validation loss {log.best_val:.3f})"


def main() -> int:
    """Run every check."""
    lines = (
        check_loop_equals_713()
        + check_stage3_power()
        + [check_half_batches(), check_accumulate_runs()]
    )
    print("\n".join(lines))
    print("e00718_check_train: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
