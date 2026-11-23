# ruff: noqa: E501, B023, ISC004, E731
"""Experiment 716: the checks of `e00716_model.py` that need no data (KF1, KF8, the parameter counts; WP2).

Usage: run.sh e00716_check_model.py

All on random tensors and the CPU: the counts of design section 4.3, the loading of a state of `M*`,
KF1 (an `M*` state gives the output of `M*` bit for bit, for real, shifted, and other-Primary PV
inputs and every `ok` pattern), and `check_pv_gradients` (KF8).
"""

import sys

import e00713_model as mdl  # ty: ignore[unresolved-import]
import e00716_arms as arms
import e00716_model as pvm
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

B = 6


def random_inputs(*, seed: int, b: int = B) -> pvm.PvInputs:
    """Return a random batch with every field of experiment 713's inputs and the four PV tensors."""
    g = torch.Generator().manual_seed(seed)
    rnd = lambda *shape: torch.randn(*shape, generator=g)
    visible = torch.rand(b, 14, 48, generator=g) > 0.3
    return pvm.PvInputs(
        primary=torch.randint(0, 16, (b,), generator=g),  # ty: ignore[unknown-argument]
        lead_day=torch.randint(0, 14, (b,), generator=g),  # ty: ignore[unknown-argument]
        top_weather=rnd(b, 48, 21), top_spread=rnd(b, 48, 7).abs(), top_datetime=rnd(b, 48, 21),  # ty: ignore[unknown-argument]
        hist_weather=rnd(b, 14, 48, 21), hist_datetime=rnd(b, 14, 48, 21),  # ty: ignore[unknown-argument]
        hist_power=rnd(b, 14, 48) * visible, hist_visible=visible,  # ty: ignore[unknown-argument]
        row_exists=torch.ones(b, 14, dtype=torch.bool),  # ty: ignore[unknown-argument]
        day_type_top=torch.randint(0, 8, (b, 48), generator=g),  # ty: ignore[unknown-argument]
        day_type_hist=torch.randint(0, 8, (b, 14, 48), generator=g),  # ty: ignore[unknown-argument]
        hist_pv=(torch.rand(b, 14, 48, generator=g) * 4), hist_pv_ok=torch.rand(b, 14, 48, generator=g) > 0.2,
        top_pv=torch.rand(b, 48, 2, generator=g) * 3, top_pv_ok=torch.rand(b, 48, generator=g) > 0.1,
    )  # fmt: skip


def randomised_m_star(*, seed: int) -> mdl.MsaNet:
    """Return an `MsaNet` whose every parameter is random, so that no zero layer hides a difference."""
    torch.manual_seed(seed)
    net = mdl.MsaNet()
    with torch.no_grad():
        for p in net.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    return net.eval()


def check_counts() -> list[str]:
    """Return the parameter counts of every arm and assert section 4.3's numbers."""
    base = mdl.parameter_count(model=mdl.MsaNet())
    assert base == 129_999, base
    lines = [f"M*: {base:,} parameters"]
    expected_new = {"Z": 0, "H": 2560, "HT": 4800, "T": 2592}
    for arm, new in expected_new.items():
        net = pvm.PvNet(arm=arm)  # ty: ignore[invalid-argument-type]
        total = mdl.parameter_count(model=net)
        assert pvm.new_parameter_count(model=net) == new == total - base, (arm, total - base)
        lines.append(
            f"{arm}: {new:,} new parameters, {total:,} in all ({100 * new / base:.1f}% more)"
        )
    assert base + 4800 == 134_799
    return lines


def check_loading() -> list[str]:
    """A state of `M*` loads into every arm with only `pv_*` keys missing; a stray key is refused."""
    star = randomised_m_star(seed=1)
    state = star.state_dict()
    for arm in ("Z", "H", "HT", "T"):
        net = pvm.PvNet(arm=arm)  # ty: ignore[invalid-argument-type]
        net.load_m_star(state=state)
        assert all(torch.equal(v, net.state_dict()[k]) for k, v in state.items())
    net = pvm.PvNet(arm="H")
    bad = dict(state)
    bad["stray.weight"] = torch.zeros(1)
    try:
        net.load_m_star(state=bad)
    except AssertionError:
        pass
    else:
        msg = "a stray key was accepted"
        raise AssertionError(msg)
    short = {k: v for k, v in state.items() if k != "tok_emb.weight"}
    try:
        net.load_m_star(state=short)
    except AssertionError:
        return [
            "a state of M* loads into every arm with only the pv_* keys missing; a stray or a missing key is refused"
        ]
    msg = "a state with a missing M* key was accepted"
    raise AssertionError(msg)


def check_zero_start() -> list[str]:
    """KF1: an `M*` state gives the output of `M*` bit for bit for real, shifted, and other-Primary PV inputs and every `ok` pattern."""
    star = randomised_m_star(seed=2)
    reference = {}
    inputs = random_inputs(seed=3)
    with torch.no_grad():
        ref = star(inputs).out
    cases = {
        "real": inputs,
        "shifted": inputs.with_pv(hist_pv=inputs.hist_pv.roll(shifts=7, dims=2)),
        "other primary": inputs.with_pv(
            hist_pv=inputs.hist_pv.flip(0), top_pv=inputs.top_pv.flip(0)
        ),
        "all ok": inputs.with_pv(
            hist_pv_ok=torch.ones_like(inputs.hist_pv_ok),
            top_pv_ok=torch.ones_like(inputs.top_pv_ok),
        ),
        "none ok": inputs.with_pv(
            hist_pv_ok=torch.zeros_like(inputs.hist_pv_ok),
            top_pv_ok=torch.zeros_like(inputs.top_pv_ok),
        ),
        "huge": inputs.with_pv(
            hist_pv=torch.full_like(inputs.hist_pv, 1e6), top_pv=torch.full_like(inputs.top_pv, 1e6)
        ),
    }
    del reference
    for arm in ("H", "HT", "T", "Z"):
        net = pvm.PvNet(arm=arm)  # ty: ignore[invalid-argument-type]
        net.load_m_star(state=star.state_dict())
        net.eval()
        if net.pv.any_path:
            net.calibrate_scale(batches=[inputs])
        for name, case in cases.items():
            with torch.no_grad():
                out = net(case)
            assert torch.isfinite(out.out).all(), (arm, name)
            assert torch.equal(out.out, ref), (
                f"KF1: arm {arm}, case {name}: the output differs from M*"
            )
    # negative control: a nonzero gain must change the output
    net = pvm.PvNet(arm="HT")
    net.load_m_star(state=star.state_dict())
    net.eval()
    net.calibrate_scale(batches=[inputs])
    with torch.no_grad():
        for p in net.new_parameter_names():
            dict(net.named_parameters())[p].add_(0.05)
        moved = net(inputs).out
    assert not torch.equal(moved, ref), (
        "negative control: nonzero PV parameters left the output unchanged"
    )
    # a later random draw is not disturbed by building a network
    torch.manual_seed(11)
    mdl.MsaNet()
    a = torch.rand(3)
    torch.manual_seed(11)
    pvm.PvNet(arm="HT", seed=5)
    assert torch.equal(a, torch.rand(3)), "the new modules moved the global random stream"
    return [
        "KF1: for arms Z, H, HT, and T an M* state gives the output of M* bit for bit (six PV input cases, including 1e6); a nonzero gain moves it; the new modules leave the global torch stream where building M* leaves it"
    ]


def check_same_start() -> list[str]:
    """Arms `H` and `HT` of one seed start from the same `W1_h` and `W2_h`; another seed differs."""
    h, ht = pvm.PvNet(arm="H", seed=4), pvm.PvNet(arm="HT", seed=4)
    for name in ("pv_hist_w1.weight", "pv_hist_w1.bias", "pv_hist_w2.weight"):
        assert torch.equal(h.state_dict()[name], ht.state_dict()[name]), name
    other = pvm.PvNet(arm="H", seed=5)
    assert not torch.equal(
        h.state_dict()["pv_hist_w2.weight"], other.state_dict()["pv_hist_w2.weight"]
    )
    return [
        "arms H and HT of one seed start from the same W1_h and W2_h; a second seed starts elsewhere"
    ]


def check_pv_gradients() -> list[str]:
    """KF8: every zero layer has a non-zero gradient at step 0 and W1 and W2 of both token paths from step 1.

    The gain an arm leaves unused is exempt by name (`c` in `H`, `a` in `T`).
    """
    star = randomised_m_star(seed=6)
    inputs = random_inputs(seed=7)
    target = torch.randn(B, 48, generator=torch.Generator().manual_seed(8))
    lines = []
    for arm in ("H", "HT", "T"):
        net = pvm.PvNet(arm=arm)  # ty: ignore[invalid-argument-type]
        net.load_m_star(state=star.state_dict())
        net.eval()
        net.calibrate_scale(batches=[inputs])
        exempt = set(net.unused_gain_names())
        params = dict(net.named_parameters())
        zero_layers = [
            n
            for n in net.new_parameter_names()
            if n in {"pv_hist_gamma", "pv_top_gamma", "pv_a", "pv_c", "pv_head_in.weight"}
        ]
        w_layers = [n for n in net.new_parameter_names() if n not in zero_layers]

        def grads() -> dict[str, torch.Tensor]:
            net.zero_grad()
            loss = mdl.per_token_nll(out=net(inputs).out, target=target).mean()
            loss.backward()
            return {
                n: (p.grad.clone() if p.grad is not None else torch.zeros_like(p))
                for n, p in params.items()
                if n in net.new_parameter_names()
            }

        g0 = grads()
        for n in zero_layers:
            if n in exempt:
                assert float(g0[n].abs().sum()) == 0.0, f"{arm}: the unused gain {n} has a gradient"
                continue
            assert float(g0[n].abs().sum()) > 0.0, (
                f"KF8: {arm}: the zero layer {n} has no gradient at step 0"
            )
        with torch.no_grad():  # one step of the zero layers, to leave the zero point
            for n in zero_layers:
                if n not in exempt:
                    params[n].add_(-0.05 * torch.sign(g0[n]))
        g1 = grads()
        for n in w_layers:
            assert float(g1[n].abs().sum()) > 0.0, f"KF8: {arm}: {n} has no gradient at step 1"
        lines.append(
            f"KF8 {arm}: zero layers {sorted(set(zero_layers) - exempt)} have a gradient at step 0; {sorted(w_layers)} from step 1; exempt by name: {sorted(exempt)}"
        )
    return lines


def check_cosine_table() -> list[str]:
    """Print the median cosine similarity between the token vectors of the path at p = 0, 0.5, 1, 2, 3, 5 (KF8)."""
    net = pvm.PvNet(arm="H", seed=0)
    ps = torch.tensor([0.0, 0.5, 1.0, 2.0, 3.0, 5.0])
    feats = torch.stack([ps, torch.ones_like(ps)], dim=-1)
    with torch.no_grad():
        y = net.pv_hist_w2(torch.nn.functional.gelu(net.pv_hist_w1(feats)))
        cos = torch.nn.functional.cosine_similarity(y[:-1], y[1:], dim=-1)
        lengths = y.norm(dim=-1)
    return [
        f"KF8 token path at p = {ps.tolist()}: cosine of neighbouring vectors {[round(float(c), 3) for c in cos]}; "
        f"lengths {[round(float(n), 3) for n in lengths]} (a constant divisor keeps the length; a LayerNorm would not)"
    ]  # fmt: skip


def check_calibration_generator() -> list[str]:
    """Regression for review blocker B1: `calibrate_scale` must accept a one-shot generator for `HT` (both buffers)."""
    inp = random_inputs(seed=1, b=4)
    out = []
    for arm in ("H", "T", "HT"):
        net = pvm.PvNet(arm=arm, seed=0)
        scale = net.calibrate_scale(
            batches=(x for x in [inp, inp])  # ty: ignore[invalid-argument-type]
        )  # a generator, as `fit()` passes
        want = {"H": {"hist"}, "T": {"top"}, "HT": {"hist", "top"}}[arm]
        assert set(scale) == want, (arm, scale)
        listed = pvm.PvNet(arm=arm, seed=0).calibrate_scale(batches=[inp, inp])
        assert scale == listed, (arm, scale, listed)
        out.append(f"B1 {arm}: calibrated {scale} from a generator, equal to the list result")
    return out


def main() -> int:
    """Run every check and print the lines."""
    torch.set_num_threads(1)
    lines = (
        check_counts()
        + check_loading()
        + check_same_start()
        + check_zero_start()
        + check_pv_gradients()
        + check_cosine_table()
        + check_calibration_generator()
    )
    print("\n".join(lines))
    print("PASS")
    if "--record" in sys.argv:
        for name in ("KF1", "KF8"):
            arms.write_wiring_record(data_dir=ML_EXPERIMENTS_DATA_DIR, name=name, passed=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
