"""The edge-state network of experiment 658: temporal encoder, graph layer, two heads."""

from __future__ import annotations

from dataclasses import dataclass

import torch  # ty: ignore[unresolved-import]
from e00658_data import CLASS_NONE, CLASS_SHIFT, N_CHANNELS
from torch import nn  # ty: ignore[unresolved-import]
from torch.nn import functional  # ty: ignore[unresolved-import]


@dataclass
class Out:
    """Network outputs for a batch of windows.

    Shapes: `cls` (batch, primaries, time, 4), `sign` and `amp` (batch, primaries, time), `edge`
    (batch, edges, time, 3) with state 1 meaning the first Primary of the edge donates.
    """

    cls: torch.Tensor
    sign: torch.Tensor
    amp: torch.Tensor
    edge: torch.Tensor


class EdgeStateNet(nn.Module):
    """Bidirectional recurrent encoder shared by all Primaries, a neighbour-mean layer, heads."""

    def __init__(self, *, hidden: int = 48, dropout: float = 0.1) -> None:
        """Build the layers.

        Args:
            hidden: Hidden units per direction of each recurrent layer.
            dropout: Dropout probability between the recurrent layers.
        """
        super().__init__()
        d = 2 * hidden
        self.inp = nn.Sequential(nn.Linear(N_CHANNELS, 32), nn.GELU())
        self.gru1 = nn.GRU(
            32, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout
        )
        self.gru2 = nn.GRU(2 * d, hidden, batch_first=True, bidirectional=True)
        self.prim = nn.Linear(d, 6)
        self.edge_sym = nn.Sequential(nn.Linear(2 * d, 32), nn.GELU(), nn.Linear(32, 1))
        self.edge_dir = nn.Sequential(nn.Linear(2 * d, 32), nn.GELU(), nn.Linear(32, 1))
        self.drop = nn.Dropout(dropout)

    def forward(self, *, x: torch.Tensor, nbr: torch.Tensor, edges: torch.Tensor) -> Out:
        """Run the network.

        Args:
            x: Features, shape (batch, primaries, time, N_CHANNELS).
            nbr: Row-normalised neighbour matrix, shape (primaries, primaries).
            edges: Integer pairs of Primary indices, shape (edges, 2).

        Returns:
            The class, sign, amplitude, and edge-state outputs.
        """
        b, p, t, _ = x.shape
        h = self.inp(x).reshape(b * p, t, -1)
        h, _ = self.gru1(h)
        h = h.reshape(b, p, t, -1)
        pooled = torch.einsum("pq,bqtc->bptc", nbr, h)
        g, _ = self.gru2(torch.cat([h, pooled], dim=-1).reshape(b * p, t, -1))
        g = self.drop(g.reshape(b, p, t, -1))
        head = self.prim(g)
        hi, hj = g[:, edges[:, 0]], g[:, edges[:, 1]]
        ij, ji = torch.cat([hi, hj], dim=-1), torch.cat([hj, hi], dim=-1)
        sym = (self.edge_sym(ij) + self.edge_sym(ji)).squeeze(-1)
        direction = (self.edge_dir(ij) - self.edge_dir(ji)).squeeze(-1)
        edge = torch.stack([torch.zeros_like(sym), sym + direction, sym - direction], dim=-1)
        return Out(cls=head[..., :4], sign=head[..., 4], amp=head[..., 5], edge=edge)


def loss_fn(
    *,
    out: Out,
    cls: torch.Tensor,
    weight: torch.Tensor,
    amp: torch.Tensor,
    edge_state: torch.Tensor,
    edge_weight: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Weighted cross-entropy for class and edge state, plus sign, amplitude, and wobble terms.

    Args:
        out: Network outputs.
        cls: Class target, shape (batch, primaries, time).
        weight: Class weight (1 where an event was injected or labelled none, 0.1 elsewhere).
        amp: Injected amplitude in noise units.
        edge_state: Edge target (batch, edges, time).
        edge_weight: Edge weight.

    Returns:
        The total loss and its parts as floats.
    """
    class_w = torch.tensor([1.0, 2.0, 1.0, 1.0], device=cls.device)
    ce = functional.cross_entropy(
        out.cls.permute(0, 3, 1, 2), cls.long(), weight=class_w, reduction="none"
    )
    l_class = (ce * weight).sum() / weight.sum().clamp(min=1.0)
    event = (cls != CLASS_NONE).float() * (weight > 0.5).float()
    sign_t = (amp > 0).float()
    l_sign = (
        functional.binary_cross_entropy_with_logits(out.sign, sign_t, reduction="none") * event
    ).sum()
    l_sign = l_sign / event.sum().clamp(min=1.0)
    l_amp = (
        functional.huber_loss(out.amp, amp.clamp(-30, 30), reduction="none", delta=2.0) * event
    ).sum()
    l_amp = l_amp / event.sum().clamp(min=1.0)
    ce_e = functional.cross_entropy(
        out.edge.permute(0, 3, 1, 2), edge_state.long(), reduction="none"
    )
    l_edge = (ce_e * edge_weight).sum() / edge_weight.sum().clamp(min=1.0)
    p_shift = torch.softmax(out.cls, dim=-1)[..., CLASS_SHIFT]
    l_wobble = (p_shift[..., 1:] - p_shift[..., :-1]).abs().mean()
    total = l_class + 0.5 * l_sign + 0.1 * l_amp + l_edge + 0.1 * l_wobble
    return total, {
        "class": float(l_class),
        "sign": float(l_sign),
        "amp": float(l_amp),
        "edge": float(l_edge),
        "wobble": float(l_wobble),
    }
