# ruff: noqa: E501, PT018
"""Experiment 718, WP3: checks KR1 (nesting), KR5 (rows as a set), and KR7 (counts and buckets) of
the network.

Every check runs on the CPU with random inputs. Usage: e00718_check_model.py [--stored-state
PATH] `--stored-state` loads a stored stage 3 state of experiment 713's `M*` and checks that
every key fits.
"""  # noqa: D205

import argparse
import sys

import e00713_model as mdl  # ty: ignore[unresolved-import]
import e00718_layout as lay
import e00718_model as f9
import numpy as np
import torch  # ty: ignore[unresolved-import]

torch.set_num_threads(1)


def random_inputs(
    *, batch: int, n_history: int, seed: int, buckets: np.ndarray | None = None
) -> f9.F9Inputs:
    """Return random inputs of `n_history` rows with distinct random age buckets (or the given buckets)."""
    g = torch.Generator().manual_seed(seed)
    rnd = lambda *shape: torch.randn(*shape, generator=g)  # noqa: E731
    if buckets is None:
        rng = np.random.default_rng(seed)
        buckets = np.stack(
            [
                rng.choice(lay.N_BUCKETS, size=n_history, replace=n_history > lay.N_BUCKETS)
                for _ in range(batch)
            ]
        )
    visible = torch.rand(batch, n_history, 48, generator=g) < 0.9
    exists = torch.rand(batch, n_history, generator=g) < 0.95
    exists[:, 0] = True
    visible &= exists[:, :, None]
    power = rnd(batch, n_history, 48) * visible
    return f9.F9Inputs(
        primary=torch.randint(0, 16, (batch,), generator=g),  # ty: ignore[unknown-argument]
        lead_day=torch.randint(0, 14, (batch,), generator=g),  # ty: ignore[unknown-argument]
        top_weather=rnd(batch, 48, 21), top_spread=rnd(batch, 48, 7).abs(),  # ty: ignore[unknown-argument]
        top_datetime=rnd(batch, 48, 21), hist_weather=rnd(batch, n_history, 48, 21),  # ty: ignore[unknown-argument]
        hist_datetime=rnd(batch, n_history, 48, 21), hist_power=power, hist_visible=visible,  # ty: ignore[unknown-argument]
        row_exists=exists, day_type_top=torch.randint(0, 8, (batch, 48), generator=g),  # ty: ignore[unknown-argument]
        day_type_hist=torch.randint(0, 8, (batch, n_history, 48), generator=g),  # ty: ignore[unknown-argument]
        age_bucket=torch.as_tensor(buckets, dtype=torch.long),
    )  # fmt: skip


def as_713(*, inp: f9.F9Inputs) -> mdl.MsaInputs:
    """Return experiment 713's inputs from the same tensors."""
    fields = {k: v for k, v in inp.__dict__.items() if k != "age_bucket"}
    return mdl.MsaInputs(**fields)


def bucket_reference(*, age: int) -> int:
    """Return the bucket of an age with a plain loop, written from design section 3.2."""
    if 2 <= age <= 15:
        return age - 2
    ranges = [
        (16, 22),
        (23, 29),
        (30, 43),
        (44, 63),
        (64, 91),
        (92, 182),
        (183, 343),
        (344, 386),
        (387, 751),
        (752, 10**9),
    ]
    for i, (lo, hi) in enumerate(ranges):
        if lo <= age <= hi:
            return 14 + i
    raise ValueError(age)


def perturb(*, model: torch.nn.Module, seed: int) -> None:
    """Give every parameter that starts at zero a random value, so that no layer hides a difference."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            if float(p.abs().max()) == 0.0:
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)


def check_counts() -> list[str]:
    """KR7: the parameter counts and tokens per sample."""
    torch.manual_seed(0)
    base = mdl.MsaNet()
    torch.manual_seed(0)
    net = f9.F9Net(seed=0)
    n_base, n_new = mdl.parameter_count(model=base), mdl.parameter_count(model=net)
    assert n_base == f9.SHARED_PARAMETERS, n_base
    assert n_new - n_base == f9.NEW_PARAMETERS == 730, n_new - n_base
    assert net.age_emb.weight.shape == (24, 64) and net.beta_age.shape == (24,)
    for block in net.blocks:
        assert block.across.key_bias.shape == (4, 25)
    assert f9.tokens_per_sample(n_history=14) == 720
    assert f9.tokens_per_sample(n_history=27) == 1344
    ages = np.arange(2, 800)
    got = lay.age_bucket(ages=ages)
    ref = np.array([bucket_reference(age=int(a)) for a in ages])
    assert (got == ref).all(), "the bucket function differs from the loop written from section 3.2"
    classes = lay.age_class(ages=ages)
    assert (classes[ages < 16] == 0).all() and classes.max() == 10
    return [
        f"KR7: parameters {n_base:,} -> {n_new:,} (+{n_new - n_base}: 640 age table, 10 beta_age, 80 key bias)",
        "KR7: 720 tokens for 14 history rows, 1,344 for 27; the bucket function equals a loop for ages 2 to 799",
    ]


def check_fresh_equals_713() -> list[str]:
    """KR1 (a): a fresh F9 network equals a fresh `MsaNet` of the same seed on every shared parameter."""
    lines = []
    for seed in (0, 1, 7):
        torch.manual_seed(seed)
        base = mdl.MsaNet()
        torch.manual_seed(seed)
        net = f9.F9Net(seed=seed)
        mine, theirs = net.state_dict(), base.state_dict()
        assert set(mine) == set(theirs)
        for key, value in theirs.items():
            if mine[key].shape == value.shape:
                assert torch.equal(mine[key], value), f"{key} differs (seed {seed})"
            elif key == "age_emb.weight":
                assert torch.equal(mine[key][:14], value)
            elif key == "beta_age":
                assert torch.equal(mine[key][:14], value) and (mine[key][14:] == 0).all()
            else:
                assert key.endswith("across.key_bias")
                assert torch.equal(mine[key][:, :15], value) and (mine[key][:, 15:] == 0).all()
        torch.manual_seed(seed + 100)  # another global state must not change the new rows
        again = f9.F9Net(seed=seed)
        assert torch.equal(again.age_emb.weight[14:], net.age_emb.weight[14:])
    lines.append(
        "KR1: a fresh F9 network equals a fresh MsaNet of seeds 0, 1, and 7 on every shared parameter; the 10 new age rows depend on the seed only"
    )
    assert not torch.equal(
        f9.F9Net(seed=0).age_emb.weight[14:], f9.F9Net(seed=1).age_emb.weight[14:]
    )
    return lines


def check_nesting() -> list[str]:
    """KR1 (b): with rows of ages 2 to 15, a loaded state of `MsaNet` gives experiment 713's output."""
    torch.manual_seed(3)
    base = mdl.MsaNet()
    perturb(model=base, seed=4)
    net = f9.F9Net(seed=3)
    net.load_mstar_state(base.state_dict())
    base.eval()
    net.eval()
    batch = 6
    buckets = np.tile(np.arange(14), (batch, 1))
    inp = random_inputs(batch=batch, n_history=14, seed=5, buckets=buckets)
    with torch.no_grad():
        a, b = base(as_713(inp=inp)), net(inp)
    err = float((a.out - b.out).abs().max())
    assert err <= 1e-6, err
    assert float((a.alpha - b.alpha).abs().max()) <= 1e-6
    # a negative control: a different bucket order must move the output
    shuffled = random_inputs(batch=batch, n_history=14, seed=5, buckets=buckets[:, ::-1].copy())
    with torch.no_grad():
        moved = float((net(shuffled).out - b.out).abs().max())
    assert moved > 1e-3, "the bucket order did not matter, so the tables are not read"
    return [
        f"KR1: loaded with a perturbed MsaNet state, F9 equals MsaNet to {err:.1e}; reversing the buckets moves the output by {moved:.2f}"
    ]


def check_untrained_readout() -> list[str]:
    """KR1 (c): the untrained read-out is the day-type-weighted mean of the visible history rows."""
    lines = []
    for n_history in (14, 27):
        torch.manual_seed(1)
        net = f9.F9Net(seed=1)
        net.eval()
        inp = random_inputs(batch=5, n_history=n_history, seed=6)
        with torch.no_grad():
            out = net(inp)
        beta = net.beta_day.detach().numpy().astype(np.float64)
        dt_top = inp.day_type_top.numpy()
        dt_hist = inp.day_type_hist.numpy()
        vis = (inp.hist_visible & inp.row_exists[:, :, None]).numpy()
        z = inp.hist_power.clamp(-5, 5).numpy().astype(np.float64)
        expect = np.zeros((5, 48))
        for b in range(5):
            for j in range(48):
                rows = np.flatnonzero(vis[b, :, j])
                if len(rows) == 0:
                    continue
                logit = beta[dt_top[b, j], dt_hist[b, rows, j]]
                w = np.exp(logit - logit.max())
                expect[b, j] = (w / w.sum() * z[b, rows, j]).sum()
        err = float(np.abs(out.base.numpy() - expect).max())
        assert err <= 1e-5, err
        median = mdl.standardised_quantiles(out=out.out)[..., 5]
        assert float((median - out.base).abs().max()) <= 1e-5, (
            "the untrained median is not the read-out"
        )
        lines.append(
            f"KR1: untrained median and read-out equal the day-type-weighted mean of {n_history} rows to {err:.1e}"
        )
    return lines


def check_row_set() -> list[str]:
    """KR5: permuting the history rows together with their buckets and masks leaves every output unchanged."""
    lines = []
    for n_history in (14, 27):
        torch.manual_seed(2)
        net = f9.F9Net(seed=2)
        perturb(model=net, seed=8)
        net.eval()
        inp = random_inputs(batch=4, n_history=n_history, seed=9)
        perm = torch.randperm(n_history, generator=torch.Generator().manual_seed(10))
        moved = f9.F9Inputs(
            **{
                k: (v[:, perm] if k in {"hist_weather", "hist_datetime", "hist_power", "hist_visible", "row_exists", "day_type_hist", "age_bucket"} else v)
                for k, v in inp.__dict__.items()
            }
        )  # fmt: skip
        with torch.no_grad():
            a, b = net(inp), net(moved)
        err = float((a.out - b.out).abs().max())
        assert err <= 1e-6, err
        assert float((a.alpha[:, perm] - b.alpha).abs().max()) <= 1e-6
        lines.append(
            f"KR5: permuting {n_history} history rows with their buckets changes the output by {err:.1e}"
        )
        # negative control: permuting the rows but not the buckets must change the output
        wrong = f9.F9Inputs(**{**moved.__dict__, "age_bucket": inp.age_bucket})
        with torch.no_grad():
            diff = float((net(wrong).out - a.out).abs().max())
        assert diff > 1e-4, "permuting rows without their buckets changed nothing"
    return lines


def check_gradients() -> str:
    """The new age rows, `beta_age` entries, and key-bias entries receive gradient when a row uses them."""
    torch.manual_seed(4)
    net = f9.F9Net(seed=4)
    perturb(model=net, seed=11)
    net.train()
    buckets = np.tile(np.array([0, 5, 13, 14, 18, 22, 23]), (3, 1))
    inp = random_inputs(batch=3, n_history=7, seed=12, buckets=buckets)
    out = net(inp)
    out.out.square().mean().backward()
    for name, grad in (
        ("age_emb", net.age_emb.weight.grad),
        ("beta_age", net.beta_age.grad),
        ("key_bias", net.blocks[0].across.key_bias.grad),
    ):
        assert grad is not None
        new = grad[14:] if name != "key_bias" else grad[:, 15:]
        assert float(new.abs().sum()) > 0, f"{name}: the new entries got no gradient"
    unused = net.age_emb.weight.grad[[b for b in range(24) if b not in {0, 5, 13, 14, 18, 22, 23}]]
    assert float(unused.abs().sum()) == 0, "an unused bucket received gradient"
    return "gradients reach the new age rows, beta_age entries, and key-bias entries, and no unused bucket"


def check_stored(*, path: str) -> str:
    """Every key of a stored `M*` state fits `load_mstar_state`."""
    state = torch.load(path, map_location="cpu")
    net = f9.F9Net(seed=0)
    net.load_mstar_state(state)
    return f"the stored state {path} loads into F9Net with every key matched"


def main() -> int:
    """Run every check."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--stored-state", default=None)
    args = parser.parse_args()
    lines = (
        check_counts()
        + check_fresh_equals_713()
        + check_nesting()
        + check_untrained_readout()
        + check_row_set()
    )
    lines.append(check_gradients())
    if args.stored_state:
        lines.append(check_stored(path=args.stored_state))
    print("\n".join(lines))
    print("e00718_check_model: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
