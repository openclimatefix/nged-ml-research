"""The donor-finding network of experiment 672: experiment 658's encoder plus a donor head.

Per Primary the network outputs the class (none, shift, ramp, routine), the sign and size of a
shift, and the **donor logit**: the chance that the Primary is the donor of a transfer that has a
metered recipient. The link head reads the two Primaries' encodings *and their donor logits*, so
the recipient state of a link is conditioned on whether the network thinks either end is a donor.
With a zero neighbour matrix and no links the network is the no-edge arm (G-0).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch  # ty: ignore[unresolved-import]
from e00658_data import CLASS_NONE, CLASS_SHIFT, N_CHANNELS
from torch import nn  # ty: ignore[unresolved-import]
from torch.nn import functional  # ty: ignore[unresolved-import]

DONOR_POS_WEIGHT: float = 5.0
"""Weight of a donor block in the donor loss (about 2% of blocks are donor blocks)."""


@dataclass
class Out:
    """Network outputs for a batch of windows.

    Shapes: `cls` (batch, primaries, time, 4), `sign`, `amp`, `donor` (batch, primaries, time),
    `edge` (batch, links, time, 3) with state 1 meaning the first Primary of the link donates.
    """

    cls: torch.Tensor
    sign: torch.Tensor
    amp: torch.Tensor
    donor: torch.Tensor
    edge: torch.Tensor


class DonorNet(nn.Module):
    """Recurrent encoder shared by all Primaries, a neighbour-mean layer, and four heads."""

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
        self.prim = nn.Linear(d, 7)
        self.edge_sym = nn.Sequential(nn.Linear(2 * d + 2, 32), nn.GELU(), nn.Linear(32, 1))
        self.edge_dir = nn.Sequential(nn.Linear(2 * d + 2, 32), nn.GELU(), nn.Linear(32, 1))
        self.drop = nn.Dropout(dropout)

    def forward(self, *, x: torch.Tensor, nbr: torch.Tensor, edges: torch.Tensor) -> Out:
        """Run the network.

        Args:
            x: Features, shape (batch, primaries, time, N_CHANNELS).
            nbr: Row-normalised neighbour matrix, shape (primaries, primaries).
            edges: Integer pairs of Primary indices, shape (links, 2).

        Returns:
            The class, sign, size, donor, and link-state outputs.
        """
        b, p, t, _ = x.shape
        h = self.inp(x).reshape(b * p, t, -1)
        h, _ = self.gru1(h)
        h = h.reshape(b, p, t, -1)
        pooled = torch.einsum("pq,bqtc->bptc", nbr, h)
        g, _ = self.gru2(torch.cat([h, pooled], dim=-1).reshape(b * p, t, -1))
        g = self.drop(g.reshape(b, p, t, -1))
        head = self.prim(g)
        donor = head[..., 6]
        if len(edges) == 0:
            edge = torch.zeros(b, 0, t, 3, device=x.device)
        else:
            hi, hj = g[:, edges[:, 0]], g[:, edges[:, 1]]
            di, dj = donor[:, edges[:, 0]].unsqueeze(-1), donor[:, edges[:, 1]].unsqueeze(-1)
            ij = torch.cat([hi, hj, di, dj], dim=-1)
            ji = torch.cat([hj, hi, dj, di], dim=-1)
            sym = (self.edge_sym(ij) + self.edge_sym(ji)).squeeze(-1)
            direction = (self.edge_dir(ij) - self.edge_dir(ji)).squeeze(-1)
            edge = torch.stack([torch.zeros_like(sym), sym + direction, sym - direction], dim=-1)
        return Out(cls=head[..., :4], sign=head[..., 4], amp=head[..., 5], donor=donor, edge=edge)


def loss_fn(  # noqa: PLR0913
    *,
    out: Out,
    cls: torch.Tensor,
    weight: torch.Tensor,
    amp: torch.Tensor,
    donor: torch.Tensor,
    edge_state: torch.Tensor,
    edge_weight: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Experiment 658's loss plus a weighted cross-entropy on the donor flag.

    Args:
        out: Network outputs.
        cls: Class target, shape (batch, primaries, time).
        weight: Class weight (1 where an event was injected, 0.1 elsewhere).
        amp: Injected amplitude in noise units.
        donor: Donor flag target.
        edge_state: Link target (batch, links, time).
        edge_weight: Link weight.

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
    ).sum() / event.sum().clamp(min=1.0)
    l_amp = (
        functional.huber_loss(out.amp, amp.clamp(-30, 30), reduction="none", delta=2.0) * event
    ).sum() / event.sum().clamp(min=1.0)
    bce = functional.binary_cross_entropy_with_logits(out.donor, donor, reduction="none")
    d_w = weight * (1.0 + (DONOR_POS_WEIGHT - 1.0) * donor)
    l_donor = (bce * d_w).sum() / weight.sum().clamp(min=1.0)
    if out.edge.shape[1]:
        ce_e = functional.cross_entropy(
            out.edge.permute(0, 3, 1, 2), edge_state.long(), reduction="none"
        )
        l_edge = (ce_e * edge_weight).sum() / edge_weight.sum().clamp(min=1.0)
    else:
        l_edge = torch.zeros((), device=cls.device)
    p_shift = torch.softmax(out.cls, dim=-1)[..., CLASS_SHIFT]
    l_wobble = (p_shift[..., 1:] - p_shift[..., :-1]).abs().mean()
    total = l_class + 0.5 * l_sign + 0.1 * l_amp + l_edge + l_donor + 0.1 * l_wobble
    return total, {
        "class": float(l_class),
        "sign": float(l_sign),
        "amp": float(l_amp),
        "donor": float(l_donor),
        "edge": float(l_edge),
        "wobble": float(l_wobble),
    }
