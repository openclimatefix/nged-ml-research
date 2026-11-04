# ruff: noqa: D103, E402
"""Check experiment 516's convolutional encoder: shapes, the position it keeps, and wiring."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
sys.path[:0] = [
    str(Path(__file__).parent),
    str(
        Path(__file__).parents[1]
        / "00397-plain-and-attention-mixture-density-network-for-primary-demand"
    ),
]

import e00397_models as base  # ty: ignore[unresolved-import]
import e00516_cnn as cnn  # ty: ignore[unresolved-import]
import torch  # ty: ignore[unresolved-import]


def test_encoder_returns_a_small_vector() -> None:
    encoder = cnn.WindowEncoder()
    out = encoder(torch.randn(5, cnn.N_VARIABLES, cnn.N_POSITIONS))
    assert out.shape == (5, cnn.OUT_DIM)


def test_encoder_rejects_the_wrong_window_shape() -> None:
    with pytest.raises(ValueError, match="window must be"):
        cnn.WindowEncoder()(torch.randn(2, cnn.N_VARIABLES, 47))


def test_encoder_keeps_the_position_of_a_change() -> None:
    """A step early in the window and the same step late in it give different vectors."""
    torch.manual_seed(0)
    encoder = cnn.WindowEncoder().eval()
    early = torch.zeros(1, cnn.N_VARIABLES, cnn.N_POSITIONS)
    late = torch.zeros_like(early)
    early[:, 2, :10] = 1.0
    late[:, 2, -10:] = 1.0
    assert not torch.allclose(encoder(early), encoder(late), atol=1e-4)


def test_windowed_network_wires_encoder_into_the_plain_network() -> None:
    torch.manual_seed(0)
    n_columns = 37
    mdn = base.PlainMdn(n_inputs=n_columns + cnn.OUT_DIM, n_primaries=16)
    for parameter in mdn.parameters():  # the head starts at zero, which blocks every gradient
        torch.nn.init.normal_(parameter, std=0.1)
    model = cnn.WindowedMdn(mdn=mdn, encoder=cnn.WindowEncoder())
    out = model(
        torch.randn(4, n_columns),
        torch.randn(4, cnn.N_VARIABLES, cnn.N_POSITIONS),
        torch.randint(0, 16, (4,)),
    )
    assert out.shape == (4, 9)
    out.sum().backward()
    assert model.encoder.project.weight.grad is not None
    assert model.encoder.project.weight.grad.abs().sum() > 0
