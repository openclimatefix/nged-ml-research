# ruff: noqa: E501, PT018, PLR0915
"""Experiment 715: checks KN1, KN4, and KN5 and the parameter counts of design section 3.8, on the CPU.

Usage: e00715_check_model.py

No data and no fit: every check runs on random tensors. Each assertion can fail. KN1: an `F4Net`
loaded with a state of `M*` gives the output of `M*` exactly, for real, rewired, and shuffled slots,
in evaluation and in training mode; an untrained `F4Net` equals an untrained `MsaNet` of the same
seed; every output and every gradient is finite on a batch with empty slots, slots removed by
neighbour dropout, and slots with no visible row. KN4: permuting the filled slots leaves the output
unchanged on a network with random `W_o`. KN5: on a network with random `W_o`, changing a masked
token changes no output, a host with no visible neighbour gives the output of every slot empty, one
sample does not change when another changes, and every gradient is finite.
"""

import sys

import e00713_model as mdl  # ty: ignore[unresolved-import]
import e00715_model as f4
import torch  # ty: ignore[unresolved-import]

torch.set_num_threads(1)
B = 6


def random_inputs(*, b: int = B, seed: int = 0) -> mdl.MsaInputs:
    """Return a random host batch with the shapes of `MsaInputs`; some history rows are absent."""
    g = torch.Generator().manual_seed(seed)
    rn = lambda *shape: torch.randn(*shape, generator=g)  # noqa: E731
    row_exists = torch.rand(b, 14, generator=g) > 0.2
    row_exists[:, 0] = True
    visible = (torch.rand(b, 14, 48, generator=g) > 0.1) & row_exists[:, :, None]
    return mdl.MsaInputs(
        primary=torch.randint(0, 16, (b,), generator=g),
        lead_day=torch.randint(0, 14, (b,), generator=g),
        top_weather=rn(b, 48, 21), top_spread=rn(b, 48, 7).abs(), top_datetime=rn(b, 48, 21),
        hist_weather=rn(b, 14, 48, 21), hist_datetime=rn(b, 14, 48, 21),
        hist_power=rn(b, 14, 48) * visible, hist_visible=visible, row_exists=row_exists,
        day_type_top=torch.randint(0, 8, (b, 48), generator=g),
        day_type_hist=torch.randint(0, 8, (b, 14, 48), generator=g),
    )  # fmt: skip


def random_neighbours(
    *, b: int = B, seed: int = 1, empty_pattern: bool = True
) -> f4.NeighbourInputs:
    """Return random neighbour tensors.

    With `empty_pattern`, sample 0 has no filled slot, sample 1 has a slot with no visible token,
    sample 2 has one filled slot, and the others have a random number of filled slots.
    """
    g = torch.Generator().manual_seed(seed)
    n_filled = torch.randint(1, 7, (b,), generator=g)
    exists = torch.arange(6)[None, :] < n_filled[:, None]
    visible = torch.rand(b, 6, 14, 48, generator=g) > 0.15
    if empty_pattern:
        exists[0] = False
        visible[1, 0] = False  # a filled slot with no visible token at all
        n_filled[2] = 1
        exists[2] = torch.arange(6) < 1
        visible[3, 1, :7] = False  # whole rows missing
    visible = visible & exists[:, :, None, None]
    z_own = torch.randn(b, 6, 14, 48, generator=g) * visible
    z_host = torch.randn(b, 6, 14, 48, generator=g) * visible
    return f4.NeighbourInputs(
        slot_index=torch.where(exists, torch.randint(0, 16, (b, 6), generator=g), torch.zeros(b, 6, dtype=torch.long)),
        slot_exists=exists, slot_relation=torch.randint(0, 8, (b, 6), generator=g),
        z_own=z_own, z_host=z_host, visible=visible,
    )  # fmt: skip


def randomise_output_projection(*, model: f4.F4Net, seed: int = 5) -> None:
    """Give both `W_o` a random value, so that the neighbours reach the output."""
    g = torch.Generator().manual_seed(seed)
    for cross in model.cross:
        cross.w_o.weight.data = torch.randn(cross.w_o.weight.shape, generator=g) * 0.2
        cross.null_key.data = torch.randn(cross.null_key.shape, generator=g) * 0.5


def mstar_pair(
    *, seed: int = 3, switches: mdl.ArmSwitches = mdl.ARM_SWITCHES["M"]
) -> tuple[mdl.MsaNet, f4.F4Net]:
    """Return a `MsaNet` with random (trained-looking) weights and an `F4Net` loaded with its state."""
    torch.manual_seed(seed)
    m = mdl.MsaNet(switches=switches)
    g = torch.Generator().manual_seed(seed + 100)
    for p in m.parameters():  # a trained-looking state: no zero projections
        p.data = p.data + torch.randn(p.shape, generator=g) * 0.05
    f = f4.F4Net(switches=switches)
    f.load_mstar_state(m.state_dict())
    return m, f


def kn1() -> None:
    """KN1: zero initialisation, an untrained F4 equal to an untrained M*, finite outputs and gradients."""
    inp = random_inputs()
    m, f = mstar_pair()
    m.eval()
    f.eval()
    ref = m(inp).out
    for label, seed in (("real", 11), ("rewired", 12), ("shuffled", 13)):
        nbr = random_neighbours(seed=seed)
        out = f(inp, nbr).out
        assert torch.equal(out, ref), (
            f"{label}: F4 with a state of M* differs from M* by {(out - ref).abs().max():.3e}"
        )
    # training mode: the global random stream must be consumed identically
    m.train()
    f.train()
    f.seed_neighbour_path(key=1)
    torch.manual_seed(77)
    ref_t = m(inp).out
    torch.manual_seed(77)
    out_t = f(inp, random_neighbours(seed=14)).out
    assert torch.equal(out_t, ref_t), (
        f"training mode: the neighbour path changed the trunk's dropout masks ({(out_t - ref_t).abs().max():.3e})"
    )
    # negative control: a network with a random W_o must differ
    randomise_output_projection(model=f)
    f.eval()
    m.eval()
    moved = (f(inp, random_neighbours(seed=11)).out - m(inp).out).abs().max()
    assert moved > 1e-4, "the negative control failed: a random W_o left the output unchanged"
    # an untrained F4 equals an untrained M* of the same seed
    for switches in (mdl.ARM_SWITCHES["M"], mdl.ARM_SWITCHES["E"]):
        torch.manual_seed(9)
        a = mdl.MsaNet(switches=switches)
        torch.manual_seed(9)
        b = f4.F4Net(switches=switches)
        sa, sb = a.state_dict(), b.state_dict()
        assert all(torch.equal(sa[k], sb[k]) for k in sa), (
            "an untrained F4 differs from an untrained M*"
        )
        assert set(sb) - set(sa) == b.new_state_keys()
        a.eval()
        b.eval()
        assert torch.equal(a(inp).out, b(inp, random_neighbours(seed=2)).out)
    # finite outputs and gradients with empty slots, dropped slots, and all-masked slots
    _, f = mstar_pair()
    randomise_output_projection(model=f)
    f.train()
    f.seed_neighbour_path(key=2)
    nbr = random_neighbours(seed=15)
    dropped = nbr.slot_exists.clone()
    dropped[:, 1] = False  # neighbour dropout of slot 1 (masked whole)
    nbr.visible = nbr.visible & dropped[:, :, None, None]
    nbr.slot_exists = dropped
    out = f(inp, nbr).out
    assert torch.isfinite(out).all(), "a non-finite output with empty, dropped, or all-masked slots"
    loss = mdl.per_token_nll(out=out, target=torch.zeros(B, 48)).mean()
    loss.backward()
    bad = [
        n
        for n, p in f.named_parameters()
        if p.grad is not None and not torch.isfinite(p.grad).all()
    ]
    assert not bad, f"non-finite gradients: {bad}"
    assert f.cross[0].w_o.weight.grad is not None and f.cross[0].w_o.weight.grad.abs().sum() > 0
    # a batch in which every slot of every sample is all-masked
    f.zero_grad()
    allmasked = random_neighbours(seed=16)
    allmasked.visible = torch.zeros_like(allmasked.visible)
    allmasked.slot_exists = torch.ones_like(allmasked.slot_exists)
    out = f(inp, allmasked).out
    mdl.per_token_nll(out=out, target=torch.zeros(B, 48)).mean().backward()
    assert torch.isfinite(out).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in f.parameters())
    # negative control of the NaN guard: a softmax over keys that are all masked is NaN, and a zero
    # W_o does not remove it (0 * NaN is NaN)
    soft = torch.softmax(torch.full((2, 4), float("-inf")), dim=-1)
    assert torch.isnan(soft).all(), "the all-masked softmax did not produce NaN"
    assert torch.isnan(soft @ torch.zeros(4, 3) * 0.0).all(), "a zero projection removed the NaN"
    att = mdl.Attention()  # experiment 713's Attention: whether its fused kernel returns NaN or zero is torch-version dependent
    all_masked = att(torch.randn(2, 5, 64), key_ok=torch.zeros(2, 5, dtype=torch.bool))
    print(
        "  experiment 713's Attention on all-masked keys returns NaN:",
        bool(torch.isnan(all_masked).any()),
    )
    print(
        "KN1 passed: F4 with a state of M* equals M* exactly (eval and train mode; real, rewired, and shuffled slots);"
    )
    print(
        "  an untrained F4 equals an untrained M* (arms M and E); finite outputs and gradients on empty, dropped, and all-masked slots;"
    )
    print(
        f"  negative controls: a random W_o moves the output by {float(moved.detach()):.3f}; an unguarded softmax gives NaN"
    )


def kn4() -> None:
    """KN4: permuting the filled slots leaves the output unchanged (random W_o)."""
    inp = random_inputs()
    _, f = mstar_pair()
    randomise_output_projection(model=f)
    f.eval()
    nbr = random_neighbours(seed=21, empty_pattern=False)
    ref = f(inp, nbr).out
    worst = 0.0
    for seed in range(5):
        perm = torch.randperm(6, generator=torch.Generator().manual_seed(seed))
        permuted = f4.NeighbourInputs(
            slot_index=nbr.slot_index[:, perm], slot_exists=nbr.slot_exists[:, perm],
            slot_relation=nbr.slot_relation[:, perm], z_own=nbr.z_own[:, perm],
            z_host=nbr.z_host[:, perm], visible=nbr.visible[:, perm],
        )  # fmt: skip
        worst = max(worst, float((f(inp, permuted).out - ref).abs().max()))
    assert worst <= 1e-5, f"KN4: a permutation of the slots changed the output by {worst:.2e}"
    # negative control: changing which Primary a slot holds does change the output
    other = nbr.slot_index.clone()
    other[:, 0] = (other[:, 0] + 3) % 16
    changed = float(
        (f(inp, f4.NeighbourInputs(**{**nbr.__dict__, "slot_index": other})).out - ref).abs().max()
    )
    assert changed > 1e-6, "KN4's negative control failed"
    print(
        f"KN4 passed: largest change under a permutation of the slots {worst:.1e} (limit 1e-5); changing a slot's Primary moves the output by {changed:.1e}"
    )


def kn5() -> None:
    """KN5: masked tokens, a host with no visible neighbour, independence of samples, finite gradients."""
    inp = random_inputs()
    _, f = mstar_pair()
    randomise_output_projection(model=f)
    f.eval()
    nbr = random_neighbours(seed=31)
    ref = f(inp, nbr).out
    # (1) the value of a masked neighbour token changes no output
    poisoned = f4.NeighbourInputs(**nbr.__dict__)
    masked = ~nbr.visible
    poisoned.z_own = torch.where(masked, torch.full_like(nbr.z_own, 1e6), nbr.z_own)
    poisoned.z_host = torch.where(masked, torch.full_like(nbr.z_host, -1e6), nbr.z_host)
    out = f(inp, poisoned).out
    assert torch.equal(out, ref), (
        f"KN5: a masked token changed the output by {(out - ref).abs().max():.2e}"
    )
    # (2) a host with no visible neighbour equals the same network with every slot empty
    novis = f4.NeighbourInputs(**nbr.__dict__)
    novis.visible = torch.zeros_like(nbr.visible)
    empty = f4.NeighbourInputs(**nbr.__dict__)
    empty.slot_exists = torch.zeros_like(nbr.slot_exists)
    empty.visible = torch.zeros_like(nbr.visible)
    assert torch.equal(f(inp, novis).out, f(inp, empty).out), (
        "KN5: no visible neighbour differs from every slot empty"
    )
    assert torch.equal(f(inp, empty).out, f(inp, None).out), (
        "KN5: every slot empty differs from the skipped path"
    )
    # sample 0 of the batch has no filled slot at all, so it equals the output without the path
    assert torch.equal(ref[0], f(inp, None).out[0]), (
        "KN5: a host with no neighbour differs from the skipped path"
    )
    # negative control: with visible neighbours the output differs from the skipped path
    assert (ref[3] - f(inp, None).out[3]).abs().max() > 1e-4, "KN5's negative control failed"
    # (3) a sample's output does not change when another sample changes
    other = f4.NeighbourInputs(**nbr.__dict__)
    other.z_own = nbr.z_own.clone()
    other.z_own[5] = other.z_own[5] + 3.0
    out2 = f(inp, other).out
    drift = float((out2[:5] - ref[:5]).abs().max())
    assert drift <= 1e-6, f"KN5: sample independence broken ({drift:.2e})"
    assert (out2[5] - ref[5]).abs().max() > 1e-6, "KN5's sample-change control failed"
    # (4) every gradient is finite
    f.train()
    f.seed_neighbour_path(key=4)
    f.zero_grad()
    mdl.per_token_nll(out=f(inp, nbr).out, target=torch.zeros(B, 48)).mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in f.parameters()), (
        "KN5: a non-finite gradient"
    )
    print(
        f"KN5 passed: masked tokens exact; no visible neighbour equals every slot empty (exact); sample independence {drift:.1e} (limit 1e-6); gradients finite"
    )


def counts() -> None:
    """Assert every count of design section 3.8."""
    f = f4.F4Net()
    table = f4.parameter_table(model=f)
    assert table["neighbour tokens"] == 2880, table
    assert table["neighbour encoder"] == 8600, table
    assert table["final LayerNorm"] == 64, table
    assert table["two across-substation attentions"] == 25_344, table
    assert table["new"] == f4.NEW_PARAMETERS == table["new_by_prefix"] == 36_888, table
    assert table["total"] == f4.F4_PARAMETERS == 165_543, table
    assert mdl.parameter_count(model=mdl.MsaNet()) == f4.M_STAR_PARAMETERS == 128_655
    assert (
        f4.parameter_table(model=f4.F4Net(switches=mdl.ARM_SWITCHES["E"]))["total"]
        == f4.F4E_PARAMETERS
        == 168_103
    )
    one = f.cross[0]
    assert sum(p.numel() for p in one.parameters()) == 12_672
    assert one.w_o.bias is None and float(one.w_o.weight.detach().abs().sum()) == 0.0
    m, f2 = mstar_pair()  # loading a state of MsaNet leaves only the new keys missing
    assert f2.new_state_keys() and set(f2.state_dict()) - set(m.state_dict()) == f2.new_state_keys()
    print(
        "counts passed:",
        {k: v for k, v in table.items() if k != "new_by_prefix"},
        "; with the convolution stem 168,103",
    )


def main() -> int:
    """Run every check."""
    counts()
    kn1()
    kn4()
    kn5()
    return 0


if __name__ == "__main__":
    sys.exit(main())
