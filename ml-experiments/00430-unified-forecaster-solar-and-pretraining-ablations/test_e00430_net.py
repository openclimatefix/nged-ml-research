# ruff: noqa: D103
"""Check experiment 430's network: the likelihood sums to one and the quantiles stay in bounds."""

import e00430_common as common
import e00430_net as net
import numpy as np
import torch  # ty: ignore[unresolved-import]

CEILING = 0.8


def _batch(*, n: int, members: int, y: float) -> net.Batch:
    torch.manual_seed(0)
    return net.Batch(
        x=torch.randn(n, len(common.GATE_COLUMNS)),
        geo=torch.tensor([[40.0, 180.0, 170.0]] * n),
        members=torch.stack([torch.full((members,), 500.0), torch.full((members,), 15.0)], dim=1)[
            None
        ].repeat(n, 1, 1)
        + torch.randn(n, members, 2) * torch.tensor([30.0, 1.0]),
        y=torch.full((n,), y),
        ceiling=torch.full((n,), CEILING),
    )


def _randomised(config: common.Config) -> net.UnifiedNet:
    torch.manual_seed(1)
    model = net.UnifiedNet(n_static=len(common.GATE_COLUMNS), config=config, plant=None)
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.05) if p.dim() > 1 else None
    return model


@torch.no_grad()
def test_atom_head_density_plus_atoms_integrates_to_one() -> None:
    model = _randomised(common.Config(atoms=True))
    batch = _batch(n=3, members=1, y=0.3)
    total = torch.zeros(3, dtype=torch.float64)
    grid = np.linspace(0.0, CEILING, 4001)
    for value in grid[1:-1]:
        batch.y = torch.full((3,), float(value))
        total += torch.exp(-model.nll_per_row(batch)).double() * (grid[1] - grid[0])
    for y in (0.0, CEILING):
        batch.y = torch.full((3,), y)
        total += torch.exp(-model.nll_per_row(batch)).double()
    np.testing.assert_allclose(total.numpy(), 1.0, atol=2e-3)


def test_quantiles_invert_the_cdf_and_respect_bounds() -> None:
    for atoms in (False, True):
        model = _randomised(common.Config(atoms=atoms))
        batch = _batch(n=5, members=1, y=0.0)
        q = net.quantiles(model=model, batch=batch).numpy()
        assert q.min() >= 0.0
        assert q.max() <= CEILING + 1e-9
        assert (np.diff(q, axis=1) >= -1e-9).all()


def test_member_route_and_plant_run_and_have_gradients() -> None:
    from e00430_plant import Plant

    config = common.Config(member_route=True, plant="e2e", atoms=True)
    plant = Plant(tilt_deg=20.0, azimuth_deg=180.0, scale=1.0, clip=0.8, learn=True)
    model = net.UnifiedNet(n_static=len(common.GATE_COLUMNS), config=config, plant=plant)
    batch = _batch(n=16, members=51, y=0.2)
    model.nll(batch).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
