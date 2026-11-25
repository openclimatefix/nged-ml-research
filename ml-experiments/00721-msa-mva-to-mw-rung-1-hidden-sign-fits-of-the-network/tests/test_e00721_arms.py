# ruff: noqa: E501
"""Known-answer tests of the sign head, the signed-expected history, and the folded loss (CPU,
`torch` needed)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import e00721_paths  # noqa: E402

e00721_paths.pin_threads()
arms = pytest.importorskip("e00721_arms")
mdl = pytest.importorskip("e00713_model")
pvm = pytest.importorskip("e00716_model")


def _inputs(*, b: int = 2, seed: int = 0, u: float = 0.8) -> arms.SgInputs:
    g = torch.Generator().manual_seed(seed)

    def n(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=g)

    return arms.SgInputs(
        primary=torch.tensor([0, 3][:b]), lead_day=torch.tensor([1, 5][:b]),
        top_weather=n(b, 48, 21), top_spread=n(b, 48, 7).abs(), top_datetime=n(b, 48, 21),
        hist_weather=n(b, 14, 48, 21), hist_datetime=n(b, 14, 48, 21), hist_power=n(b, 14, 48).abs(),
        hist_visible=torch.ones(b, 14, 48, dtype=torch.bool), row_exists=torch.ones(b, 14, dtype=torch.bool),
        day_type_top=torch.zeros(b, 48, dtype=torch.long), day_type_hist=torch.zeros(b, 14, 48, dtype=torch.long),
        hist_pv=n(b, 14, 48).abs(), hist_pv_ok=torch.ones(b, 14, 48, dtype=torch.bool),
        top_pv=n(b, 48, 2).abs(), top_pv_ok=torch.ones(b, 48, dtype=torch.bool),
        hist_u=torch.full((b, 14, 48), u),
    )  # fmt: skip


def test_signed_expected_history_known_answers() -> None:
    z = torch.full((1, 14, 48), 0.5)
    shift = torch.tensor([1.0])
    same, var0 = arms.signed_expected_history(z=z, rho=torch.zeros_like(z), shift=shift)
    assert torch.allclose(same, z)
    assert torch.allclose(var0, torch.zeros_like(z))
    flipped, var1 = arms.signed_expected_history(z=z, rho=torch.ones_like(z), shift=shift)
    assert torch.allclose(flipped, torch.full_like(z, -2.5))  # -(0.5 + 1) - 1
    assert torch.allclose(var1, torch.zeros_like(z))
    half, var_half = arms.signed_expected_history(z=z, rho=torch.full_like(z, 0.5), shift=shift)
    assert torch.allclose(half, torch.full_like(z, -1.0))  # 0 - shift
    assert torch.allclose(var_half, torch.full_like(z, 1.5**2))  # 4 * 0.25 * v^2 with v = 1.5


def test_a_closed_gate_gives_the_output_of_the_network_without_the_sign_head() -> None:
    inp = _inputs()
    torch.manual_seed(0)
    plain = pvm.PvNet(arm="HT", seed=0).eval()
    torch.manual_seed(0)
    sg = arms.SgNet(seed=0, sign_head=True).eval()
    sg.set_gate(gate=np.zeros(16))
    with torch.no_grad():
        want = plain(inp).out
        got = sg(inp).out
    assert torch.allclose(want, got, atol=1e-5)


def test_the_head_starts_near_zero_and_gets_a_gradient_when_the_gate_is_open() -> None:
    inp = _inputs()
    torch.manual_seed(0)
    sg = arms.SgNet(seed=0, sign_head=True).train()
    sg.set_gate(gate=np.ones(16))
    out = sg(inp)
    assert sg.last_rho is not None
    assert float(sg.last_rho.max()) == pytest.approx(1 / (1 + np.exp(4.0)), rel=1e-4)
    loss = out.out.square().mean()
    loss.backward()
    assert sg.pv_sign_lin.bias.grad is not None
    assert float(sg.pv_sign_lin.bias.grad.abs()) > 0.0


def test_u_below_the_floor_closes_the_row_gate() -> None:
    torch.manual_seed(0)
    sg = arms.SgNet(seed=0).eval()
    sg.set_gate(gate=np.ones(16))
    with torch.no_grad():
        sg(_inputs(u=0.1))
    assert sg.last_rho is not None
    assert float(sg.last_rho.abs().max()) == 0.0


def test_sg0_has_rho_zero() -> None:
    torch.manual_seed(0)
    sg = arms.SgNet(seed=0, sign_head=False).eval()
    with torch.no_grad():
        sg(_inputs())
    assert sg.last_rho is not None
    assert float(sg.last_rho.abs().max()) == 0.0


def _single_component_out(*, mean: float, log_std: float, b: int = 1) -> torch.Tensor:
    k = mdl.m559.COMPONENTS
    out = torch.zeros(b, 48, 3 * k)
    out[..., 1:k] = -20.0  # almost all the weight on component 0
    out[..., k : 2 * k] = mean
    out[..., 2 * k :] = log_std
    return out


def test_folded_density_integrates_to_one() -> None:
    out = _single_component_out(mean=0.3, log_std=float(np.log(0.5)))[:, :1]
    q = torch.tensor([0.05])
    x = torch.linspace(0.002, 6.0, 20001, dtype=torch.float64)
    r = torch.sqrt(x**2 + 0.05**2).float()
    nll = arms.folded_nll(
        out=out.repeat(1, len(r), 1), reading_z=r[None, :], shift=torch.zeros(1), q_v=q
    )
    dens = torch.exp(-nll[0].double()) * (x / torch.sqrt(x**2 + 0.05**2).double())
    area = float(torch.trapezoid(dens, x))
    assert area == pytest.approx(1.0, abs=0.01)


def test_folded_quantiles_match_a_monte_carlo_of_the_mixture() -> None:
    out = _single_component_out(mean=-0.2, log_std=float(np.log(0.4)))[:, :1]
    got = arms.folded_quantiles(out=out, shift=torch.zeros(1), q_v=torch.tensor([0.1]))[
        0, 0
    ].numpy()
    rng = np.random.default_rng(0)
    x = rng.normal(-0.2, 0.4, 400_000)
    want = np.quantile(np.sqrt(x**2 + 0.1**2), arms.mdl.base_models.LEVELS)
    np.testing.assert_allclose(got, want, atol=0.01)
    assert (np.diff(got) >= 0).all()


def test_folded_nll_is_finite_and_differentiable_with_censored_readings() -> None:
    out = _single_component_out(mean=0.0, log_std=0.0, b=2).clone().requires_grad_(True)
    reading = torch.full((2, 48), 0.1)
    reading[:, :10] = 0.05  # at the floor: censored
    nll = arms.folded_nll(
        out=out, reading_z=reading, shift=torch.zeros(2), q_v=torch.full((2,), 0.05)
    )
    assert torch.isfinite(nll).all()
    nll.mean().backward()
    assert out.grad is not None
    assert torch.isfinite(out.grad).all()


def test_blank_sky_zeroes_exactly_the_sky_columns() -> None:
    inp = _inputs()
    blank = arms.blank_sky(inp=inp)
    sky = list(arms.SKY_WEATHER_COLUMNS)
    assert float(blank.hist_weather[..., sky].abs().sum()) == 0.0
    rest = [c for c in range(21) if c not in sky]
    assert torch.equal(blank.hist_weather[..., rest], inp.hist_weather[..., rest])
    assert float(blank.top_datetime[..., :3].abs().sum()) == 0.0


def test_arm_table_matches_the_page() -> None:
    assert set(arms.ARM_SPECS) == {"P0", "S", "SG-0", "SG-A", "SG-A-S", "SG-A-open"}
    assert arms.ARM_SPECS["SG-A-S"].shift_days == 28
    assert arms.ARM_SPECS["SG-A-open"].force_gate_open
    assert not arms.ARM_SPECS["SG-0"].sign_head
    assert arms.ARM_SPECS["S"].readings == "signed"
    assert "SG-A-S" not in {a for a, s in arms.ARM_SPECS.items() if "W0" in s.worlds}
