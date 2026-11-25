# ruff: noqa: E501, PT018, PLR0912
"""Experiment 715: `F4Net`, the axial-attention demand network with a third axis of neighbour slots.

`F4Net` subclasses experiment 713's `MsaNet` and keeps every one of its module names, so a stored
state of `M*` loads without renaming. It builds its new modules after every module of `MsaNet`
(and after the convolution stem of arm `E`), so an untrained `F4Net` draws the same initial weights
as an untrained `MsaNet` of the same seed.

A sample holds up to six neighbour slots of 14 rows by 48 half-hours at ages 2 to 15. Neighbour
tokens are width 32. A small neighbour encoder (across-day attention and a transition, applied once)
turns the tokens into `n_out`. In each of the two axial blocks, between the within-day and the
across-day attention, every host history token attends over the six `n_out` tokens of its own age
and half-hour and over one learned null key. The output projection `W_o` of that attention starts at
zero and has no bias, so the neighbour path adds exactly zero until it has been trained, and adds
exactly zero to a host token with no visible neighbour at every point of training.

**The NaN guard.** Experiment 713's `Attention` returns NaN for a row whose keys are all masked. The
neighbour path never meets that case, by three rules: every neighbour token is a key for itself in
the neighbour encoder; `n_out` is multiplied by `nbr_visible` (a finite zero) before `W_k` and `W_v`;
and the across-substation attention always holds the null key.

**The random streams.** The neighbour path draws its dropout masks from `F4Net.nbr_generator`, a
`torch.Generator` of its own, and never from the global stream. The trunk's dropout masks are then the
masks that `M*` drew.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final

import e00713_model as mdl  # ty: ignore[unresolved-import]
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]
from torch.utils.checkpoint import checkpoint  # ty: ignore[unresolved-import]

N_SLOTS: Final = 6
D_N: Final = 32
"""The width of a neighbour token."""
N_RELATIONS: Final = 8
N_NBR_HEADS: Final = 4
CROSS_HEADS: Final = 4
CROSS_HEAD_WIDTH: Final = mdl.WIDTH // CROSS_HEADS
NBR_ENCODER_HEADS: Final = 4
NBR_DROPOUT: Final = 0.1
"""Dropout inside the neighbour path (the same rate as the trunk's)."""
M_STAR_PARAMETERS: Final = 129_999
NEW_PARAMETERS: Final = 36_888
F4_PARAMETERS: Final = M_STAR_PARAMETERS + NEW_PARAMETERS
F4E_PARAMETERS: Final = 169_447
NEW_PREFIXES: Final = (
    "nbr_emb.", "rel_emb.", "age_emb_n.", "col_emb_n.", "u_own", "u_host", "b_nbr", "m_nbr",
    "nbr_encoder.", "nbr_ln_out.", "cross.",
)  # fmt: skip
"""The state-dictionary prefixes of the parameters that `M*` does not have."""


@dataclass
class NeighbourInputs:
    """The neighbour tensors of one batch of `B` samples, all on the device.

    Attributes:
        slot_index: (B, 6) long, the Primary in each slot; 0 where the slot is empty (a placeholder
            that the masks make harmless).
        slot_exists: (B, 6) bool.
        slot_relation: (B, 6) long, `4 * same_bsp + 2 * host_is_MW + neighbour_is_MW`.
        z_own: (B, 6, 14, 48) float, the neighbour's standardised power, 0 where not visible.
        z_host: (B, 6, 14, 48) float, the same deviation on the host's scale, 0 where not visible.
        visible: (B, 6, 14, 48) bool, `nbr_visible`.
    """

    slot_index: torch.Tensor
    slot_exists: torch.Tensor
    slot_relation: torch.Tensor
    z_own: torch.Tensor
    z_host: torch.Tensor
    visible: torch.Tensor

    def to(self, device: str | torch.device) -> NeighbourInputs:
        """Return the batch on `device`."""
        return NeighbourInputs(**{k: v.to(device) for k, v in self.__dict__.items()})

    @property
    def row_exists(self) -> torch.Tensor:
        """Return `nbr_row_exists`, (B, 6, 14): any token of the row is visible."""
        return self.visible.any(dim=-1)


@dataclass
class F4Output(mdl.MsaOutput):
    """`MsaOutput` plus the across-substation attention weights of each block (only with `return_attention`)."""

    nbr_attention: list[torch.Tensor] | None = None
    """Per block, (B, 14, 48, heads, 7): weights over the six slots and the null key (last index)."""
    nbr_encoder_attention: torch.Tensor | None = None


def drop(x: torch.Tensor, *, p: float, training: bool, gen: torch.Generator | None) -> torch.Tensor:
    """Dropout with an explicit mask drawn from `gen`, never from the global random stream."""
    if not training or p == 0.0:
        return x
    assert gen is not None, "the neighbour path needs `F4Net.nbr_generator` in training"
    keep = torch.rand(x.shape, generator=gen, device=x.device) >= p
    return x * keep.to(x.dtype) / (1.0 - p)


class NeighbourEncoder(nn.Module):
    """Across-day attention at width 32 over the 14 tokens of one slot and one column, then a transition.

    The block is pre-LayerNorm and residual. A row with no visible token is excluded as a key for
    the other rows, and every token keeps itself as a key, so no softmax has an empty key set.
    """

    def __init__(self) -> None:
        """Build the block: 4 heads of 8, a zero key-row bias per head, and a 32 to 64 to 32 transition."""
        super().__init__()
        self.ln_attn = nn.LayerNorm(D_N)
        self.qkv = nn.Linear(D_N, 3 * D_N)
        self.out = nn.Linear(D_N, D_N)
        self.key_bias = nn.Parameter(torch.zeros(NBR_ENCODER_HEADS, mdl.N_HISTORY))
        self.ln_ff = nn.LayerNorm(D_N)
        self.transition = nn.Sequential(nn.Linear(D_N, 2 * D_N), nn.GELU(), nn.Linear(2 * D_N, D_N))
        self.record: dict[str, torch.Tensor] | None = None

    def forward(
        self, tokens: torch.Tensor, *, row_exists: torch.Tensor, training: bool,
        gen: torch.Generator | None,
    ) -> torch.Tensor:  # fmt: skip
        """Return the encoded tokens: `tokens` (B, 6, 14, 48, 32) and `row_exists` (B, 6, 14)."""
        b, s, a, c, w = tokens.shape
        x = tokens.permute(0, 1, 3, 2, 4).reshape(b * s * c, a, w)
        ok = row_exists[:, :, None, :].expand(b, s, c, a).reshape(b * s * c, a)
        eye = torch.eye(a, dtype=torch.bool, device=x.device)
        key_ok = ok[:, None, :] | eye[None]  # (N, a, a): every token keeps itself as a key
        heads = NBR_ENCODER_HEADS
        q, k, v = self.qkv(self.ln_attn(x)).view(-1, a, 3, heads, w // heads).permute(2, 0, 3, 1, 4)
        logits = q @ k.transpose(-1, -2) / float(w // heads) ** 0.5
        logits = logits + self.key_bias[None, :, None, :]
        logits = logits.masked_fill(~key_ok[:, None, :, :], float("-inf"))
        weights = torch.softmax(logits, dim=-1)
        if self.record is not None:
            self.record["weights"] = weights.detach()
        weights = drop(weights, p=NBR_DROPOUT, training=training, gen=gen)
        y = (weights @ v).transpose(1, 2).reshape(-1, a, w)
        x = x + drop(self.out(y), p=NBR_DROPOUT, training=training, gen=gen)
        x = x + drop(self.transition(self.ln_ff(x)), p=NBR_DROPOUT, training=training, gen=gen)
        return x.view(b, s, c, a, w).permute(0, 1, 3, 2, 4)


class CrossAttention(nn.Module):
    """Across-substation attention: each host history token attends over six neighbour tokens and a null key.

    `W_o` has no bias and starts at zero. A host token whose neighbour tokens are all masked puts all
    its weight on the null key, whose value is zero, and so receives exactly zero.
    """

    def __init__(self) -> None:
        """Build the LayerNorm, the three projections, the output projection, and the null key."""
        super().__init__()
        self.ln = nn.LayerNorm(mdl.WIDTH)
        self.w_q = nn.Linear(mdl.WIDTH, mdl.WIDTH)
        self.w_k = nn.Linear(D_N, mdl.WIDTH)
        self.w_v = nn.Linear(D_N, mdl.WIDTH)
        self.w_o = nn.Linear(mdl.WIDTH, mdl.WIDTH, bias=False)
        self.null_key = nn.Parameter(torch.zeros(mdl.WIDTH))
        nn.init.zeros_(self.w_o.weight)
        self.record: dict[str, torch.Tensor] | None = None

    def forward(
        self, x: torch.Tensor, *, n_out: torch.Tensor, visible: torch.Tensor, training: bool,
        gen: torch.Generator | None,
    ) -> torch.Tensor:  # fmt: skip
        """Return the update of the host history tokens.

        Args:
            x: (B, 14, 48, 64) host history tokens.
            n_out: (B, 6, 14, 48, 32) encoded neighbour tokens, already zero where not visible.
            visible: (B, 6, 14, 48) bool.
            training: Whether dropout is on.
            gen: The neighbour path's generator.

        Returns:
            The (B, 14, 48, 64) update, which the caller adds to `x`.
        """
        b, a, c, w = x.shape
        h, d = CROSS_HEADS, CROSS_HEAD_WIDTH
        q = self.w_q(self.ln(x)).view(b, a, c, h, d)
        k = self.w_k(n_out).view(b, N_SLOTS, a, c, h, d)
        v = self.w_v(n_out).view(b, N_SLOTS, a, c, h, d)
        scale = float(d) ** 0.5
        logit_nbr = torch.einsum("bajhd,bsajhd->bajhs", q, k) / scale
        logit_null = (q * self.null_key.view(h, d)).sum(dim=-1, keepdim=True) / scale
        ok = visible.permute(0, 2, 3, 1)[:, :, :, None, :]  # (B, 14, 48, 1, 6)
        logit_nbr = logit_nbr.masked_fill(~ok, float("-inf"))
        weights = torch.softmax(torch.cat([logit_nbr, logit_null], dim=-1), dim=-1)  # null key last
        if self.record is not None:
            self.record["weights"] = weights.detach()
        weights = drop(weights, p=NBR_DROPOUT, training=training, gen=gen)
        y = torch.einsum("bajhs,bsajhd->bajhd", weights[..., :N_SLOTS], v)  # the null value is zero
        return drop(self.w_o(y.reshape(b, a, c, w)), p=NBR_DROPOUT, training=training, gen=gen)


class F4Net(mdl.MsaNet):
    """The axial-attention demand network with neighbour rows (experiment 715)."""

    def __init__(self, *, switches: mdl.ArmSwitches = mdl.ARM_SWITCHES["M"]) -> None:
        """Build `MsaNet` first, then the neighbour modules (36,888 parameters)."""
        super().__init__(switches=switches)
        assert switches.within_day and switches.across_day and not switches.top_row_only
        self.checkpoint_neighbours = False
        """Recompute the neighbour path's activations in the backward pass instead of storing them."""
        self.nbr_generator: torch.Generator | None = None
        """The neighbour path's own generator; set it with `seed_neighbour_path` before training."""
        self.nbr_emb = nn.Embedding(mdl.N_PRIMARIES, D_N)
        self.rel_emb = nn.Embedding(N_RELATIONS, D_N)
        self.age_emb_n = nn.Embedding(mdl.N_HISTORY, D_N)
        self.col_emb_n = nn.Embedding(mdl.N_COLS, D_N)
        for emb in (self.nbr_emb, self.rel_emb, self.age_emb_n, self.col_emb_n):
            nn.init.normal_(emb.weight, std=0.02)
        self.u_own = nn.Parameter(torch.randn(D_N) * 0.02)
        self.u_host = nn.Parameter(torch.randn(D_N) * 0.02)
        self.b_nbr = nn.Parameter(torch.randn(D_N) * 0.02)
        self.m_nbr = nn.Parameter(torch.randn(D_N) * 0.02)
        self.nbr_encoder = NeighbourEncoder()
        self.nbr_ln_out = nn.LayerNorm(D_N)
        self.cross = nn.ModuleList(CrossAttention() for _ in range(mdl.BLOCKS))

    # ------------------------------------------------------------------ parameters and state

    def neighbour_parameters(self) -> list[nn.Parameter]:
        """Return the parameters that `M*` does not have, in `state_dict` order."""
        return [p for name, p in self.named_parameters() if name.startswith(NEW_PREFIXES)]

    def new_state_keys(self) -> set[str]:
        """Return the state-dictionary keys that `M*` does not have."""
        return {k for k in self.state_dict() if k.startswith(NEW_PREFIXES)}

    def load_mstar_state(self, state: dict[str, torch.Tensor]) -> None:
        """Load a stored state of `MsaNet`; only the new keys may be missing, and nothing may be unexpected."""
        result = self.load_state_dict(state, strict=False)
        assert not result.unexpected_keys, result.unexpected_keys
        assert set(result.missing_keys) == self.new_state_keys(), sorted(
            set(result.missing_keys) ^ self.new_state_keys()
        )

    def seed_neighbour_path(self, *, key: int) -> None:
        """Seed the neighbour path's generator (on the device of the parameters) with `key`."""
        device = self.m_nbr.device
        if self.nbr_generator is None:
            self.nbr_generator = torch.Generator(device=device)
        self.nbr_generator.manual_seed(int(key))

    # ------------------------------------------------------------------ forward

    def _run_path(self, fn: Callable[..., Any], *args: torch.Tensor) -> Any:
        """Call `fn(*args)`, with gradient checkpointing when `checkpoint_neighbours` is set in training.

        `torch.utils.checkpoint` restores only the default random generators, so a recompute would draw
        other dropout masks from `nbr_generator` than the forward pass did. This wrapper sets the
        generator to its state before the forward call at the start of a recompute, and after the
        recompute puts back the state it found, so the masks, the gradients, and the
        later draws equal those of the unchecked path.
        """
        if not (self.checkpoint_neighbours and self.training and torch.is_grad_enabled()):
            return fn(*args)
        gen = self.nbr_generator
        assert gen is not None, "the neighbour path needs `F4Net.nbr_generator` in training"
        start = gen.get_state()
        first = [True]

        def inner(*a: torch.Tensor) -> Any:
            if first[0]:
                first[0] = False
                gen.set_state(start)  # a no-op: the state is the one the call began with
                return fn(*a)
            kept = gen.get_state()  # the state after every forward draw
            gen.set_state(start)
            try:
                return fn(*a)
            finally:  # the recompute may stop early by raising, so restore in every case
                gen.set_state(kept)

        return checkpoint(inner, *args, use_reentrant=False)

    def _neighbour_tokens(self, nbr: NeighbourInputs) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `n_out` (B, 6, 14, 48, 32), zero where not visible, and the visibility mask."""
        vis = nbr.visible & nbr.slot_exists[:, :, None, None]
        emb = (
            self.nbr_emb(nbr.slot_index.clamp_min(0))[:, :, None, None, :]
            + self.rel_emb(nbr.slot_relation)[:, :, None, None, :]
            + self.age_emb_n.weight[None, None, :, None, :]
            + self.col_emb_n.weight[None, None, None, :, :]
        )
        z_own = nbr.z_own.clamp(-mdl.POWER_CLIP, mdl.POWER_CLIP)
        z_host = nbr.z_host.clamp(-mdl.POWER_CLIP, mdl.POWER_CLIP)
        power = torch.where(
            vis[..., None],
            z_own[..., None] * self.u_own + z_host[..., None] * self.u_host + self.b_nbr,
            self.m_nbr.expand_as(emb),
        )
        tokens = emb + power
        row_exists = vis.any(dim=-1)
        encoded = self.nbr_encoder(
            tokens, row_exists=row_exists, training=self.training, gen=self.nbr_generator
        )
        n_out = self.nbr_ln_out(encoded) * vis[..., None].to(encoded.dtype)
        return n_out, vis

    def forward(  # ty: ignore[invalid-method-override]
        self,
        inp: mdl.MsaInputs,
        nbr: NeighbourInputs | None = None,
        *,
        return_attention: bool = False,
    ) -> F4Output:
        """Return the raw mixture outputs of the 48 target tokens of each sample.

        With `nbr` None, the neighbour path is skipped, which gives the output of every slot empty
        (the null key alone adds exactly zero), so the output equals that of `MsaNet` with the same
        trunk weights. With `return_attention` the output also holds the across-substation weights.
        """
        x, row_exists = self._tokens(inp)
        n_out = vis = None
        if return_attention:
            self.nbr_encoder.record = {}  # the encoder runs once, before the blocks
        if nbr is not None:
            n_out, vis = self._run_path(
                lambda *f: self._neighbour_tokens(NeighbourInputs(*f)),
                nbr.slot_index, nbr.slot_exists, nbr.slot_relation, nbr.z_own, nbr.z_host, nbr.visible,
            )  # fmt: skip
        recorded: list[tuple[str, mdl.Attention]] = []
        if return_attention:
            for i, block in enumerate(self.blocks):
                for kind in ("within", "across"):
                    module = getattr(block, kind)
                    module.record = {}
                    recorded.append((f"{kind}_{i}", module))
            for cross in self.cross:
                cross.record = {}
        for i, block in enumerate(self.blocks):
            b, r, c, w = x.shape
            # the three steps of `AxialBlock.forward`, with the across-substation attention between the first two
            y = block.within(block.ln_within(x).reshape(b * r, c, w))
            x = x + block.drop(y.view(b, r, c, w))
            if n_out is not None:
                assert vis is not None
                delta = self._run_path(
                    lambda x_, n_, v_, cross=self.cross[i]: cross(
                        x_, n_out=n_, visible=v_, training=self.training, gen=self.nbr_generator
                    ),
                    x[:, 1:], n_out, vis,
                )  # fmt: skip
                x = torch.cat([x[:, :1], x[:, 1:] + delta], dim=1)
            cols = block.ln_across(x).permute(0, 2, 1, 3).reshape(b * c, r, w)
            ok = row_exists[:, None, :].expand(b, c, r).reshape(b * c, r)
            y = block.across(cols, key_ok=ok).view(b, c, r, w).permute(0, 2, 1, 3)
            x = x + block.drop(y)
            x = x + block.drop(block.transition(block.ln_transition(x)))
        h0, base, spread, no_history, alpha = self._readout(x, inp)
        out = self._mixture_out(
            h0=h0, base=base, spread=spread, no_history=no_history, lead_day=inp.lead_day
        )
        attention = nbr_attention = enc_attention = None
        if return_attention:
            attention = {}
            nbr_attention = []
            batch = len(inp.primary)
            for name, module in recorded:
                w_ = module.record.pop("weights")  # ty: ignore[unresolved-attribute]
                module.record = None
                if name.startswith("within"):
                    attention[name] = w_.view(batch, -1, *w_.shape[1:])
                else:
                    attention[name] = w_[:, :, 0, :].view(
                        batch, mdl.N_COLS, *w_.shape[1:2], w_.shape[-1]
                    )
            for cross in self.cross:
                if "weights" in cross.record:  # ty: ignore[unsupported-operator]
                    nbr_attention.append(cross.record.pop("weights"))  # ty: ignore[unresolved-attribute]
                cross.record = None
            enc_attention = self.nbr_encoder.record.pop("weights", None)  # ty: ignore[unresolved-attribute]
            self.nbr_encoder.record = None
        return F4Output(
            out=out, base=base, spread=spread, no_history=no_history, alpha=alpha,  # ty: ignore[unknown-argument]
            attention=attention, target_vector=h0.detach() if return_attention else None,  # ty: ignore[unknown-argument]
            nbr_attention=nbr_attention, nbr_encoder_attention=enc_attention,
        )  # fmt: skip


def parameter_table(*, model: F4Net) -> dict[str, int]:
    """Return the parameter counts of design section 3.8, from the built module."""
    count = lambda *mods: sum(p.numel() for m in mods for p in m.parameters())  # noqa: E731
    tokens = (
        model.nbr_emb.weight.numel() + model.rel_emb.weight.numel() + model.age_emb_n.weight.numel()
        + model.col_emb_n.weight.numel() + model.u_own.numel() + model.u_host.numel()
        + model.b_nbr.numel() + model.m_nbr.numel()
    )  # fmt: skip
    table = {
        "neighbour tokens": tokens,
        "neighbour encoder": count(model.nbr_encoder),
        "final LayerNorm": count(model.nbr_ln_out),
        "two across-substation attentions": count(model.cross),
    }
    table["new"] = sum(table.values())
    table["new_by_prefix"] = sum(p.numel() for p in model.neighbour_parameters())
    table["total"] = mdl.parameter_count(model=model)
    return table
