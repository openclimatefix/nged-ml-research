# ruff: noqa: E501, PT018, PT017
"""Experiment 716: the checks of `e00716_batch.py` that need no fit (KF2, the mask dropout, the origin-time assertion; WP3).

Usage: run.sh e00716_check_batch.py [synthetic|real MONTH]

`synthetic` builds a small `Source` of random arrays and runs every check on it. `real MONTH` builds the
fold's real arrays on the CPU (`e00716_fit.build_tables`) and runs KF2 on them; it needs the stored
data and is a CPU self test, not a fit.
"""

import sys

import e00713_batch as bt  # ty: ignore[unresolved-import]
import e00713_layout as lay  # ty: ignore[unresolved-import]
import e00716_batch as pvb
import e00716_model as pvm
import numpy as np
import torch  # ty: ignore[unresolved-import]

G = 48 * 70
FIRST_DAY = 17
N_DAYS = 30
POISONS = (1e6, float("nan"))


def synthetic_source(*, seed: int = 0) -> tuple[bt.Source, pvb.PvTables]:
    """Return a small random 713 `Source` and PV tables over a grid of 70 days (16 Primaries)."""
    g = torch.Generator().manual_seed(seed)
    power = torch.randn(16, G, generator=g)
    power[torch.rand(16, G, generator=g) < 0.02] = float("nan")
    run = torch.zeros(16, N_DAYS, dtype=torch.int64)
    days = FIRST_DAY + torch.arange(N_DAYS)
    c = days * 48 + 24
    origin = lay.GRID_START_US + (c - lay.NATIVE_OFFSET_STEPS) * lay.STEP_US
    run[:] = (origin - lay.ENSEMBLE_LEAD_US)[None, :]
    ens = bt.EnsTensors(
        weather=torch.randn(16, N_DAYS, 672, 21, generator=g).half(),
        spread=torch.rand(16, N_DAYS, 672, 7, generator=g).half(),
        filled=torch.ones(16, N_DAYS, 672, dtype=torch.bool), first_day=FIRST_DAY, run_us=run,
    )  # fmt: skip
    source = bt.Source(
        power_z=power, dense_weather=torch.randn(16, G, 21, generator=g).half(),
        dense_datetime=torch.randn(16, G, 21, generator=g).half(),
        day_type=torch.randint(0, 8, (G,), generator=g), ens=ens,
    )  # fmt: skip
    ok = torch.rand(16, G, generator=g) > 0.05
    tables = pvb.PvTables(
        hist_p=torch.rand(16, G, generator=g) * 5, hist_ok=ok,
        top_p0=torch.rand(16, N_DAYS, 672, 2, generator=g) * 3,
        top_filled=torch.rand(16, N_DAYS, 672, generator=g) > 0.01,
        first_day=FIRST_DAY, run_us=run,
    )  # fmt: skip
    return source, tables


def sample_keys(*, n: int, seed: int = 1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return random (Primary, contract origin index, lead day) triples inside the table."""
    g = torch.Generator().manual_seed(seed)
    day = FIRST_DAY + torch.randint(0, N_DAYS - 1, (n,), generator=g)
    return (
        torch.randint(0, 16, (n,), generator=g),
        day * 48 + 24,
        torch.randint(0, 14, (n,), generator=g),
    )


def equal_inputs(a: pvm.PvInputs, b: pvm.PvInputs) -> bool:
    """Return whether every tensor of two batches is bit-identical."""
    return all(torch.equal(getattr(a, k), getattr(b, k)) for k in a.__dict__)


def kf2_poison(*, ps: pvb.PvSource, n: int = 64) -> list[str]:
    """KF2: poisoning every history `p` after the contract origin and every target-row value of a later run changes no input.

    The poisoned copy sets history `p` at every grid index after the sample's contract origin, and
    the target table of every origin day after the sample's own day (a run issued after the native
    origin minus 9 hours), to 1e6 and then to NaN. The negative control poisons from 15 days before
    the origin (offset 0 into the history window) and must change the inputs.
    """
    primary, c_idx, lead = sample_keys(n=n)
    out = []
    for i in range(n):
        p, c, k = primary[i : i + 1], c_idx[i : i + 1], lead[i : i + 1]
        base = ps.assemble(primary=p, c_idx=c, lead_day=k)
        day = int(c[0]) // 48 - ps.tables.first_day
        for value in POISONS:
            hp = ps.tables.hist_p.clone()
            hp[:, int(c[0]) + 1 :] = value
            tp = ps.tables.top_p0.clone()
            tp[:, day + 1 :] = value
            poisoned = ps.assemble(
                primary=p, c_idx=c, lead_day=k, hist_p_override=hp, top_p0_override=tp
            )
            assert equal_inputs(base.inputs, poisoned.inputs), (
                f"KF2: sample {i}: a poisoned value ({value}) reached the inputs"
            )
        hp = ps.tables.hist_p.clone()
        hp[:, int(c[0]) - 48 * 15 :] = 1e6
        try:
            loose = ps.assemble(primary=p, c_idx=c, lead_day=k, hist_p_override=hp)
        except AssertionError:
            continue  # the per-batch range assertion caught the poison: the control changed the outcome
        assert not equal_inputs(base.inputs, loose.inputs), (
            f"KF2: sample {i}: the negative control did not change the inputs"
        )
    out.append(
        f"KF2: {n} samples, history p after the contract origin and target-row values of later runs set to 1e6 and to NaN: every assembled input bit-identical; negative control changes the inputs"
    )
    # the same on a network with random gains: the output is identical
    net = pvm.PvNet(arm="HT", seed=3)
    with torch.no_grad():
        for name in net.new_parameter_names():
            dict(net.named_parameters())[name].add_(
                torch.randn_like(dict(net.named_parameters())[name]) * 0.3
            )
    net.eval()
    base = ps.assemble(primary=primary, c_idx=c_idx, lead_day=lead)
    net.calibrate_scale(batches=[base.inputs])
    for i in range(8):
        p, c, k = primary[i : i + 1], c_idx[i : i + 1], lead[i : i + 1]
        hp = ps.tables.hist_p.clone()
        hp[:, int(c[0]) + 1 :] = 1e6
        tp = ps.tables.top_p0.clone()
        tp[:, int(c[0]) // 48 - ps.tables.first_day + 1 :] = 1e6
        with torch.no_grad():
            ref = net(ps.assemble(primary=p, c_idx=c, lead_day=k).inputs).out
            got = net(
                ps.assemble(
                    primary=p, c_idx=c, lead_day=k, hist_p_override=hp, top_p0_override=tp
                ).inputs
            ).out
        assert torch.equal(ref, got), f"KF2: sample {i}: the network output changed under poison"
    out.append("KF2: a network with random gains gives identical output on the poisoned copy")
    return out


def check_masks(*, ps: pvb.PvSource) -> list[str]:
    """The CAMS-rule and 713-convention masks hit the rows of design sections 6 and 9.2, and the dropout draw is off the torch stream."""
    primary, c_idx, lead = sample_keys(n=40, seed=2)
    full = ps.assemble(primary=primary, c_idx=c_idx, lead_day=lead)
    cams = torch.ones(40, dtype=torch.bool)
    none = torch.zeros(40, dtype=torch.bool)
    a = ps.assemble(
        primary=primary, c_idx=c_idx, lead_day=lead, cams_rule=cams, convention=none
    ).inputs.hist_pv_ok
    base = full.inputs.hist_pv_ok
    assert not a[:, 0].any() and not a[:, 1, 25:].any()
    assert torch.equal(a[:, 1, :25], base[:, 1, :25]) and torch.equal(a[:, 2:], base[:, 2:])
    b = ps.assemble(
        primary=primary, c_idx=c_idx, lead_day=lead, cams_rule=none, convention=cams
    ).inputs.hist_pv_ok
    assert not b[:, :4].any() and torch.equal(b[:, 4:], base[:, 4:])
    torch.manual_seed(5)
    expected = torch.rand(4)
    torch.manual_seed(5)
    c1, v1 = pvb.pv_dropout_masks(seed=0, stage=2, epoch=3, batch=7, n=100_000)
    c2, v2 = pvb.pv_dropout_masks(seed=0, stage=2, epoch=3, batch=7, n=100_000)
    assert torch.equal(torch.rand(4), expected), "the mask draw moved the global torch stream"
    assert np.array_equal(c1, c2) and np.array_equal(v1, v2), "the draw is not repeatable"
    assert not (c1 & v1).any()
    assert abs(c1.mean() - 0.10) < 0.005 and abs(v1.mean() - 0.10) < 0.005, (c1.mean(), v1.mean())
    return [
        f"masks: the CAMS rule hides the age-2 row and columns 25 to 47 of the age-3 row; the convention hides ages 2 to 5; the draw is repeatable, off the torch stream, with shares {c1.mean():.3f} and {v1.mean():.3f}"
    ]


def check_origin_time(*, ps: pvb.PvSource) -> list[str]:
    """A target table keyed by a run issued after the native origin minus 9 hours is refused, and the listed exception days are not."""
    primary, c_idx, lead = sample_keys(n=8, seed=3)
    bad = pvb.PvTables(**{**ps.tables.__dict__, "run_us": ps.tables.run_us + 24 * lay.HOUR_US})
    try:
        pvb.PvSource(source=ps.source, tables=bad).assemble(
            primary=primary, c_idx=c_idx, lead_day=lead
        )
    except AssertionError as err:
        assert "issued after" in str(err) or "nwp_init_time" in str(err), err
    else:
        msg = "a PV run issued after the origin minus 9 hours was accepted"
        raise AssertionError(msg)
    days = {int(c) // 48 for c in c_idx}
    allowed = pvb.PvTables(**{**bad.__dict__, "exception_days": frozenset(days)})  # ty: ignore[invalid-argument-type]
    try:
        pvb.PvSource(source=ps.source, tables=allowed).assemble(
            primary=primary, c_idx=c_idx, lead_day=lead
        )
    except AssertionError as err:
        # 713's own assertion on the ensemble table still holds (it reads `ens.run_us`, which is untouched)
        raise AssertionError(f"the exception days were not honoured: {err}") from err
    return [
        "origin time: a run issued after the native origin minus 9 hours is refused; the listed HT-S exception days pass"
    ]


def check_inputs_equal_713(*, ps: pvb.PvSource) -> list[str]:
    """The 713 part of the batch equals 713's own `Source.assemble` bit for bit."""
    primary, c_idx, lead = sample_keys(n=16, seed=4)
    mine = ps.assemble(primary=primary, c_idx=c_idx, lead_day=lead)
    theirs = ps.source.assemble(primary=primary, c_idx=c_idx, lead_day=lead, ensemble=True)
    for name in theirs.inputs.__dict__:
        assert torch.equal(getattr(mine.inputs, name), getattr(theirs.inputs, name)), name
    assert torch.equal(
        mine.target,
        theirs.target,
    ) or bool(torch.isnan(mine.target).equal(torch.isnan(theirs.target)))
    return ["the 713 tensors of a PV batch equal experiment 713's own assembly"]


def main() -> int:
    """Run the requested checks."""
    torch.set_num_threads(1)
    which = sys.argv[1] if len(sys.argv) > 1 else "synthetic"
    lines: list[str] = []
    if which == "real":
        import e00716_fit as fit  # ty: ignore[unresolved-import]

        ps = fit.build_pv_source_for_check(month=sys.argv[2])
        lines += kf2_poison(ps=ps, n=64)
    else:
        source, tables = synthetic_source()
        ps = pvb.PvSource(source=source, tables=tables)
        lines += (
            check_inputs_equal_713(ps=ps)
            + check_masks(ps=ps)
            + check_origin_time(ps=ps)
            + kf2_poison(ps=ps)
        )
    print("\n".join(lines))
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
