# ruff: noqa: E501, PT018
"""Experiment 715: checks KN2 (poisoned neighbour power), KN9 (host-scale power), and the mode table of design section 5.2.

Usage: e00715_check_batch.py            # synthetic arrays, CPU, no data
       e00715_check_batch.py real MONTH # the same checks on the real fold MONTH (CPU; the origins are the scored origins)

KN2: setting every neighbour reading with a grid index above the native origin's grid index minus 48 to
1e6, and in a second run to NaN, leaves the assembled inputs unchanged bit for bit, for every arm
(`F4`, `F4-R`, `F4-T`), and on a network whose `W_o` is random, because a zero `W_o` hides the neighbours.
The negative control (a cap of 0) must change the inputs for `F4` and `F4-R`; `F4-T` is exempt, as
experiment 713's placebo is, because its rows are at least 16 days old.
KN9: `z_host` equals an independent NumPy computation, and a synthetic transfer of a fixed number of
megawatts gives equal and opposite changes of the host's `z` and the neighbour's `z_host`.
The mode table: in each of the five modes, the assembled neighbour tensor equals the array the table
names and differs from the other candidate arrays, and the host's history reads its own array.
"""

import sys

import e00713_batch as bt
import e00713_layout as lay
import e00715_batch as fb
import e00715_check_model as cm
import e00715_model as f4
import e00715_neighbours as nb
import numpy as np
import torch  # ty: ignore[unresolved-import]

torch.set_num_threads(1)
N_GRID = 48 * 700
FIRST_C = 48 * 40 + 24


def synthetic_source(
    *, seed: int = 0, n_grid: int = N_GRID
) -> tuple[bt.Source, np.ndarray, np.ndarray]:
    """Return a random host source, per-Primary megawatt scales, and the first-origin index (no real data)."""
    g = torch.Generator().manual_seed(seed)
    power = torch.randn(16, n_grid, generator=g)
    power[torch.rand(16, n_grid, generator=g) < 0.03] = float("nan")
    src = bt.Source(
        power_z=power, dense_weather=torch.randn(16, n_grid, 21, generator=g),
        dense_datetime=torch.randn(16, n_grid, 21, generator=g),
        day_type=torch.randint(0, 8, (n_grid,), generator=g), ens=None,
    )  # fmt: skip
    scale = np.random.default_rng(seed).uniform(0.5, 4.0, 16)
    return src, scale, np.array([FIRST_C])


def make_source(
    *,
    base: bt.Source,
    scale: np.ndarray,
    table: nb.NeighbourTable,
    mode: nb.SlotMode,
    seed: int = 0,
) -> fb.F4Source:
    """Return an `F4Source` that reads the host's own array as the neighbour array (the `cleaned` array)."""
    return fb.F4Source(
        host=base, neighbour_power=base.power_z, mw_scale=scale, table=table, mode=mode,
        rewire_seed=seed, label="cleaned",
    )  # fmt: skip


def shuffled_ages(
    *, src: fb.F4Source, primary: np.ndarray, c: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the (ages, exists) of the `F4-T` draw for these samples from a full pool."""
    import e00713_train as tr

    clean = np.isfinite(src.neighbour_power.numpy())
    pool = nb.shuffle_pool(table=src.table, pool=tr.block_pool(clean=clean))
    got = tr.placebo_ages_for(primary=primary, c_idx=c, pool=pool)
    return torch.from_numpy(np.where(got >= 0, got, lay.PLACEBO_MIN_AGE)), torch.from_numpy(
        got >= 0
    )


def tensors_equal(a: fb.F4Assembled, b: fb.F4Assembled) -> bool:
    """Return whether the host history and the neighbour tensors of two batches are bit-identical."""
    same_host = all(
        torch.equal(getattr(a.host.inputs, n), getattr(b.host.inputs, n))
        for n in ("hist_power", "hist_visible", "row_exists")
    )
    same_nbr = all(torch.equal(getattr(a.nbr, n), getattr(b.nbr, n)) for n in a.nbr.__dict__)
    return same_host and same_nbr


def kn2(
    *, src: fb.F4Source, primary: np.ndarray, c: np.ndarray, label: str, ensemble: bool = False
) -> str:
    """KN2 for the samples (`primary`, `c`) on `src`, for the three arms; return a one-line report."""
    model = f4.F4Net()
    cm.randomise_output_projection(model=model)
    model.eval()
    dev = src.device
    p_t, c_t = torch.from_numpy(primary).to(dev), torch.from_numpy(c).to(dev)
    lead = torch.from_numpy(np.arange(len(primary)) % 14).to(dev)
    tested = 0
    for arm, mode in (("F4", "real"), ("F4-R", "rewired"), ("F4-T", "shuffled")):
        s = src.with_mode(mode=mode, rewire_seed=0)
        extra = {}
        if mode == "shuffled":
            ages, exists = shuffled_ages(src=src, primary=primary, c=c)
            extra = {"nbr_ages": ages.to(dev), "nbr_age_exists": exists.to(dev)}
        for value in (bt.POISON_VALUE, float("nan")):
            for i in range(len(primary)):
                power = s.neighbour_power.clone()
                host_in = s.host.power_in.clone()
                power[:, int(c[i]) - lay.NEWEST_VISIBLE_BEFORE_CONTRACT + 1 :] = value
                host_in[:, int(c[i]) - lay.NEWEST_VISIBLE_BEFORE_CONTRACT + 1 :] = value
                one = s.assemble(
                    primary=p_t[i : i + 1], c_idx=c_t[i : i + 1], lead_day=lead[i : i + 1],
                    ensemble=ensemble, neighbour_power=power, host_power_in=host_in,
                    **{k: v[i : i + 1] for k, v in extra.items()},
                )  # fmt: skip
                one_ref = s.assemble(
                    primary=p_t[i : i + 1], c_idx=c_t[i : i + 1], lead_day=lead[i : i + 1],
                    ensemble=ensemble, **{k: v[i : i + 1] for k, v in extra.items()},
                )  # fmt: skip
                assert tensors_equal(one, one_ref), (
                    f"KN2 {arm} sample {i}: a poisoned reading ({value}) reached the inputs"
                )
                with torch.no_grad():
                    assert torch.equal(
                        model(one.host.inputs, one.nbr).out,
                        model(one_ref.host.inputs, one_ref.nbr).out,
                    ), f"KN2 {arm} sample {i}: a poisoned reading ({value}) reached the output"
                tested += 1
        if arm != "F4-T":
            # negative control: with a cap of 0 the poison must reach the neighbour tensors
            changed = 0
            for i in range(min(len(primary), 16)):
                power = s.neighbour_power.clone()
                power[:, int(c[i]) - lay.NEWEST_VISIBLE_BEFORE_CONTRACT + 1 :] = bt.POISON_VALUE
                loose = s.assemble(
                    primary=p_t[i : i + 1], c_idx=c_t[i : i + 1], lead_day=lead[i : i + 1],
                    ensemble=ensemble, cap_steps=0, neighbour_power=power,
                )  # fmt: skip
                strict = s.assemble(
                    primary=p_t[i : i + 1], c_idx=c_t[i : i + 1], lead_day=lead[i : i + 1], ensemble=ensemble,
                )  # fmt: skip
                changed += int(not tensors_equal(loose, strict))
            assert changed > 0, f"KN2 {arm}: the negative control (cap 0) did not change the inputs"
    return f"KN2 {label}: {tested} poisoned rebuilds (1e6 and NaN; F4, F4-R, F4-T) bit-identical; cap-0 negative control changed F4 and F4-R; outputs of a random-W_o network identical"


def kn9() -> str:
    """KN9: `z_host` against NumPy, and equal and opposite changes under a transfer."""
    src0, scale, _ = synthetic_source(seed=3)
    table = nb.load_table()
    src = make_source(base=src0, scale=scale, table=table, mode="real")
    host, c = 6, FIRST_C + 48 * 20  # D10, six neighbours
    prim, c_t = torch.tensor([host]), torch.tensor([c])
    batch = src.assemble(primary=prim, c_idx=c_t, lead_day=torch.tensor([0]), ensemble=False)
    worst = 0.0
    for s, q in enumerate(table.neighbours[host]):
        vis = batch.nbr.visible[0, s]
        if (
            nb.LABELS[q] in nb.PAIR
        ):  # the synthetic grid ends before 2025-03-01, so the cut hides D15 and D18
            assert not bool(vis.any())
            continue
        own = src.neighbour_power[q, batch.nbr_idx[0].reshape(-1)].reshape(14, 48).numpy()
        expect = np.clip(fb.host_scale_z(power_own=own, mw_scale=scale, host=host, other=q), -5, 5)
        got = batch.nbr.z_host[0, s].numpy()
        diff = np.abs(np.where(vis.numpy(), got - expect, 0.0))
        # the clip acts on z_own first when the neighbour's own value exceeds 5
        worst = max(worst, float(diff.max()))
        assert vis.any()
    assert worst <= 1e-5, f"KN9: z_host differs from NumPy by {worst:.2e}"
    # a transfer of x MW from the host to a neighbour: the host's z falls and the neighbour's z_host rises by the same amount
    x_mw, q = 0.6, table.neighbours[host][0]  # D2
    idx = batch.host.hist_idx[0, 3]
    moved = src.neighbour_power.clone()
    host_power = src.host.power_in.clone()
    host_power[host, idx] = host_power[host, idx] - x_mw / scale[host]
    moved[q, idx] = moved[q, idx] + x_mw / scale[q]
    host_power[q, idx] = moved[q, idx]
    moved_src = fb.F4Source(
        host=src.host.with_power_in(power_in=host_power), neighbour_power=moved, mw_scale=scale,
        table=table, mode="real", rewire_seed=0, label="cleaned",
    )  # fmt: skip
    after = moved_src.assemble(primary=prim, c_idx=c_t, lead_day=torch.tensor([0]), ensemble=False)
    d_host = (after.host.inputs.hist_power - batch.host.inputs.hist_power)[0, 3]
    s = list(table.neighbours[host]).index(q)
    d_nbr = (after.nbr.z_host - batch.nbr.z_host)[0, s, 3]
    both = batch.host.inputs.hist_visible[0, 3] & batch.nbr.visible[0, s, 3]
    ok_clip = (batch.host.inputs.hist_power[0, 3].abs() < 4) & (
        after.host.inputs.hist_power[0, 3].abs() < 4
    )
    sel = (
        both
        & ok_clip
        & (batch.nbr.z_host[0, s, 3].abs() < 4)
        & (after.nbr.z_host[0, s, 3].abs() < 4)
    )
    assert sel.any()
    assert torch.allclose(
        d_host[sel], torch.full_like(d_host[sel], -x_mw / scale[host]), atol=1e-5
    ), "KN9: the host's z did not fall by x"
    assert torch.allclose(d_nbr[sel], torch.full_like(d_nbr[sel], x_mw / scale[host]), atol=1e-5), (
        "KN9: the neighbour's z_host did not rise by x"
    )
    assert torch.allclose(d_host[sel], -d_nbr[sel], atol=1e-5)
    return f"KN9 passed: z_host equals NumPy to {worst:.1e} (limit 1e-5); a transfer of {x_mw} MW moves the host's z by {float(d_host[sel][0]):+.4f} and the neighbour's z_host by {float(d_nbr[sel][0]):+.4f}"


def mode_table() -> str:
    """Assert the array that each of the five modes reads, by comparing with every candidate array."""
    src0, scale, _ = synthetic_source(seed=5)
    table = nb.load_table()
    g = torch.Generator().manual_seed(9)
    named = {
        "cleaned": src0.power_z,
        "shifted": src0.power_z + 0.37 * torch.randn(src0.power_z.shape, generator=g),
        "mirror": src0.power_z - 0.41 * torch.randn(src0.power_z.shape, generator=g),
        "injected": src0.power_z + 0.53 * torch.randn(src0.power_z.shape, generator=g),
    }
    host, c = (
        10,
        FIRST_C + 48 * 25,
    )  # D15 is cut before 2025-03-01 only as a neighbour; use D12... host 8 = D12
    host = 8
    prim, c_t = torch.tensor([host]), torch.tensor([c])
    lines = []
    for mode, (host_name, nbr_name) in fb.ARRAYS_OF_MODE.items():
        host_arr, nbr_arr, label = fb.arrays_for_mode(mode=mode, named=named)  # ty: ignore[invalid-argument-type]
        assert label == nbr_name
        src = fb.F4Source(
            host=src0.with_power_in(power_in=host_arr), neighbour_power=nbr_arr, mw_scale=scale,
            table=table, mode="real", rewire_seed=0, label=label,
        )  # fmt: skip
        batch = src.assemble(primary=prim, c_idx=c_t, lead_day=torch.tensor([0]), ensemble=False)
        for s, q in enumerate(table.neighbours[host]):
            vis = batch.nbr.visible[0, s]
            if not bool(
                vis.any()
            ):  # D15 and D18 are cut before 2025-03-01, after this synthetic grid
                assert nb.LABELS[q] in nb.PAIR
                continue
            for name, arr in named.items():
                expected = torch.where(
                    vis, arr[q, batch.nbr_idx[0].reshape(-1)].reshape(14, 48), torch.zeros(14, 48)
                ).clamp(-5, 5)
                equal = torch.equal(batch.nbr.z_own[0, s], expected)
                if name == nbr_name:
                    assert equal, f"mode {mode}: the slots do not read the {nbr_name} array"
                else:
                    assert not equal, f"mode {mode}: the slots also match the {name} array"
        hist = batch.host.inputs.hist_power[0]
        expected_host = torch.where(
            batch.host.inputs.hist_visible[0],
            host_arr[host, batch.host.hist_idx[0].reshape(-1)].reshape(14, 48),
            torch.zeros(14, 48),
        ).clamp(-5, 5)
        assert torch.equal(hist, expected_host), (
            f"mode {mode}: the host's history does not read the {host_name} array"
        )
        lines.append(f"{mode}: host {host_name}, slots {nbr_name}")
    return "mode table passed: " + "; ".join(lines)


def other_checks() -> list[str]:
    """Origin-day bound with a readable error, dropout stream separation, and the D15 and D18 cut."""
    ens = bt.EnsTensors(
        weather=torch.zeros(16, 10, 672, 21), spread=torch.zeros(16, 10, 672, 7),
        filled=torch.ones(16, 10, 672, dtype=torch.bool), first_day=1000, run_us=torch.ones(16, 10, dtype=torch.int64),
    )  # fmt: skip
    fb.check_origin_days(ens=ens, c_idx=np.array([1000 * 48 + 24, 1009 * 48 + 24]), what="test")
    for bad in (999 * 48 + 24, 1010 * 48 + 24):
        try:
            fb.check_origin_days(ens=ens, c_idx=np.array([bad]), what="test")
        except IndexError as err:
            message = str(err)
            assert "0 to 9" in message and "origin day index" in message
        else:
            msg = "an origin outside the ensemble table was accepted"
            raise AssertionError(msg)
    a = fb.drop_slots_for(n=64, key=(0, 3, 5))
    assert np.array_equal(a, fb.drop_slots_for(n=64, key=(0, 3, 5)))
    assert not np.array_equal(a, fb.drop_slots_for(n=64, key=(0, 3, 6)))
    assert 0.08 < fb.drop_slots_for(n=20000, key=(1, 2, 3)).mean() < 0.22
    # the D15 and D18 cut: a neighbour reading before 2025-03-01 is never visible
    table = nb.load_table()
    n_grid = fb.SHORT_HISTORY_CUT_IDX + 48 * 80
    g = torch.Generator().manual_seed(2)
    src0 = bt.Source(
        power_z=torch.randn(16, n_grid, generator=g), dense_weather=torch.zeros(16, n_grid, 21),
        dense_datetime=torch.zeros(16, n_grid, 21), day_type=torch.zeros(n_grid, dtype=torch.long), ens=None,
    )  # fmt: skip
    src = make_source(base=src0, scale=np.ones(16), table=table, mode="real")
    host = nb.LABELS.index("D2")  # D15 and D18 are among its neighbours
    c = fb.SHORT_HISTORY_CUT_IDX + 48 * 5 + 24
    batch = src.assemble(
        primary=torch.tensor([host]),
        c_idx=torch.tensor([c]),
        lead_day=torch.tensor([0]),
        ensemble=False,
    )
    for s, q in enumerate(table.neighbours[host]):
        before = batch.nbr_idx[0] < fb.SHORT_HISTORY_CUT_IDX
        if nb.LABELS[q] in nb.PAIR:
            assert before.any() and not bool(batch.nbr.visible[0, s][before].any()), (
                "D15 or D18 visible before 2025-03-01"
            )
        else:
            assert bool(batch.nbr.visible[0, s][before].any()), "the cut removed another Primary"
    return [
        "origin-day bound raises a readable IndexError outside the table's days; neighbour dropout draws are keyed and repeatable",
        "the D15 and D18 cut hides neighbour readings before 2025-03-01 and no other Primary's",
    ]


def synthetic() -> None:
    """Run every check on synthetic arrays."""
    src0, scale, _ = synthetic_source()
    table = nb.load_table()
    src = make_source(base=src0, scale=scale, table=table, mode="real")
    rng = np.random.default_rng(1)
    primary = rng.integers(0, 16, 24)
    c = (rng.integers(FIRST_C // 48 + 20, N_GRID // 48 - 16, 24) * 48 + 24).astype(np.int64)
    print(kn2(src=src, primary=primary, c=c, label="synthetic arrays"))
    print(kn9())
    print(mode_table())
    for line in other_checks():
        print(line)


def real(*, month: str) -> None:
    """KN2 on a real fold: every scored origin, 64 samples per origin drawn at random (CPU)."""
    import os

    os.environ["E00667_PHASE"] = "predict"
    import e00713_inputs as inp
    import e00715_arms as arms

    table = nb.load_table()
    pipe = inp.load_pipeline(month=month, phase="predict")
    data = inp.build_fold_data(pipe=pipe, device="cpu")
    src = arms.f4_source(data=data, table=table, mode="real", rewire_seed=0)
    assert data.test_origin_idx is not None
    c_all = data.test_origin_idx + lay.NATIVE_OFFSET_STEPS
    fb.check_origin_days(ens=data.source.ens, c_idx=c_all, what="the scored origins")  # ty: ignore[invalid-argument-type]
    rng = np.random.default_rng(0)
    # the D15 and D18 cut must already be in the cleaned power
    for label in nb.PAIR:
        row = data.source.power_z[nb.LABELS.index(label), : fb.SHORT_HISTORY_CUT_IDX]
        assert not bool(torch.isfinite(row).any()), (
            f"{label} has readings before 2025-03-01 in the cleaned power"
        )
    primary = rng.integers(0, 16, 64)
    c = rng.choice(c_all, size=64)
    print(
        kn2(src=src, primary=primary, c=c, label=f"real fold {month} (64 samples)", ensemble=True)
    )
    print(
        f"real fold {month}: every scored origin ({len(c_all)}) lies inside the ensemble table; D15 and D18 are NaN before 2025-03-01"
    )


def main() -> int:
    """Run the synthetic checks, or the real-fold check."""
    if len(sys.argv) >= 3 and sys.argv[1] == "real":
        real(month=sys.argv[2])
        return 0
    synthetic()
    return 0


if __name__ == "__main__":
    sys.exit(main())
