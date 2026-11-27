# ruff: noqa: E501, PT018
"""Experiment 733: the axial-attention forecaster with the untangled history path and the sign head.

`UntangledNet` subclasses experiment 713's `MsaNet`, keeps every module name, and builds the new
modules last, so a state dictionary of M* loads into it with only the new keys (`u_*` and `sign_*`)
missing. Every new path starts at exactly zero effect, so a network that loads M*'s state and reads
every channel as absent gives M*'s output bit for bit, whatever the new weights hold, because the
history path's output is gated by the per-token flag.

Three new parts:

1. The history token path `γ ⊙ (W2 GELU(W1 [u·ok, v·ok, w·ok, ok]) / r)`, added to each history token
   before the first axial block (`W1`: 4 to 32 with bias, `W2`: 32 to 64 without bias, `γ`: 64 gains
   starting at zero, `r`: a stored constant that `calibrate_scale` sets from the arm's training tokens).
2. The read-out path: the read-out weights `α` give the weighted means of `u`, `v`, and `w` and the
   cover, and a zero-initialised linear map adds them to the mixture head's first layer.
3. The sign head (synthetic arms only): a layer of 32 units on `[h0, ū, cover, one-hot lead day]`
   whose output starts at the persistence of the history's untangled P(X<0).

A channel that the rule drops is zero in every token, and the path keeps its shape.
"""

import math
from dataclasses import dataclass
from typing import Final

import e00733_core as core
import e00733_paths as paths  # noqa: F401

# isort: split
import e00713_model as mdl  # ty: ignore[unresolved-import]
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]
from torch.nn import functional  # ty: ignore[unresolved-import]

PATH_PREFIX: Final = "u_"
SIGN_PREFIX: Final = "sign_"
NEW_PREFIXES: Final = (PATH_PREFIX, SIGN_PREFIX)
GAIN_NAMES: Final = ("u_gamma",)
"""The gain gets its own learning rate and no weight decay."""
READOUT_FEATURES: Final = core.N_CHANNELS + 1
"""`[ubar, vbar, wbar, cover]`."""
LOGIT_EPS: Final = 1e-4
"""P(X<0) is clipped to this distance from 0 and 1 before its logit."""


@dataclass
class UntInputs(mdl.MsaInputs):
    """713's batch plus the untangled channels of the 14 history rows.

    Attributes:
        hist_u: `(B, 14, 48, 3)` the channels `[u, v, w]` in the units of the forecaster's input,
            zero where `hist_u_ok` is false.
        hist_u_ok: `(B, 14, 48)` bool, the presence flag.
    """

    hist_u: torch.Tensor
    hist_u_ok: torch.Tensor

    def to(self, device: str | torch.device) -> UntInputs:
        """Return the batch on `device`."""
        return UntInputs(**{k: v.to(device) for k, v in self.__dict__.items()})

    def with_channels(self, *, hist_u: torch.Tensor, hist_u_ok: torch.Tensor) -> UntInputs:
        """Return the batch with other channels (the placebo, the mask of R4, a poisoned copy)."""
        values = dict(self.__dict__)
        values.update(hist_u=hist_u, hist_u_ok=hist_u_ok)
        return UntInputs(**values)

    def absent(self) -> UntInputs:
        """Return the batch with every channel absent (zeros and a flag that is off)."""
        return self.with_channels(
            hist_u=torch.zeros_like(self.hist_u), hist_u_ok=torch.zeros_like(self.hist_u_ok)
        )


@dataclass
class UntOutput(mdl.MsaOutput):
    """713's output plus the logit of P(X<0) for each target half-hour (the sign head, or None)."""

    sign_logit: torch.Tensor | None = None  # (B, 48)


def logit(*, p: torch.Tensor) -> torch.Tensor:
    """Return the logit of `p` clipped to `LOGIT_EPS` from 0 and 1."""
    p = p.clamp(LOGIT_EPS, 1.0 - LOGIT_EPS)
    return torch.log(p) - torch.log1p(-p)


def init_like_default(*, layer: nn.Linear, generator: torch.Generator) -> None:
    """Draw `layer`'s weights and bias from PyTorch's default initialisation, with `generator`."""
    bound = 1.0 / math.sqrt(layer.in_features)
    with torch.no_grad():
        layer.weight.uniform_(-bound, bound, generator=generator)
        if layer.bias is not None:
            layer.bias.uniform_(-bound, bound, generator=generator)


class UntangledNet(mdl.MsaNet):
    """The axial-attention network with the untangled history path, and optionally the sign head."""

    def __init__(
        self,
        *,
        seed: int = 0,
        switches: mdl.ArmSwitches = mdl.ARM_SWITCHES["M"],
        path: bool = True,
        sign_head: bool = False,
        clim_logit: float = 0.0,
    ) -> None:
        """Build M*'s modules, then the new modules from a generator seeded by `seed`.

        Args:
            seed: Seeds the random partner of each zero layer.
            switches: 713's arm switches.
            path: Whether to build the history path and the read-out path.
            sign_head: Whether to build the sign head.
            clim_logit: The logit of the training banks' share of reverse flow, the head's start at
                a column with no history.
        """
        super().__init__(switches=switches)
        self.has_path = path
        self.has_sign_head = sign_head
        self.u_w1 = self.u_w2 = self.u_gamma = self.u_head_in = None
        self.sign_hidden = self.sign_out = None
        generator = torch.Generator().manual_seed(seed)
        with torch.random.fork_rng(devices=[]):
            if path:
                self.u_w1 = nn.Linear(core.HISTORY_INPUT_WIDTH, core.HISTORY_HIDDEN_WIDTH)
                self.u_w2 = nn.Linear(core.HISTORY_HIDDEN_WIDTH, mdl.WIDTH, bias=False)
                init_like_default(layer=self.u_w1, generator=generator)
                init_like_default(layer=self.u_w2, generator=generator)
                self.u_gamma = nn.Parameter(torch.zeros(mdl.WIDTH))
                self.register_buffer("u_r", torch.ones(()))
                self.u_head_in = nn.Linear(READOUT_FEATURES, mdl.WIDTH, bias=False)
                nn.init.zeros_(self.u_head_in.weight)
            if sign_head:
                self.sign_hidden = nn.Linear(
                    mdl.WIDTH + core.SIGN_HEAD_EXTRA_INPUTS, core.SIGN_HEAD_HIDDEN
                )
                init_like_default(layer=self.sign_hidden, generator=generator)
                self.sign_out = nn.Linear(core.SIGN_HEAD_HIDDEN, 1)
                nn.init.zeros_(self.sign_out.weight)
                nn.init.constant_(self.sign_out.bias, float(clim_logit))
                self.register_buffer("sign_clim", torch.tensor(float(clim_logit)))

    # ------------------------------------------------------------------ state

    def new_parameter_names(self) -> list[str]:
        """Return the names of the new trainable parameters."""
        return [n for n, _ in self.named_parameters() if n.startswith(NEW_PREFIXES)]

    def load_m_star(self, *, state: dict[str, torch.Tensor]) -> None:
        """Load a state of M*: only the new keys may be missing, and none may be unexpected."""
        result = self.load_state_dict(state, strict=False)
        assert not result.unexpected_keys, result.unexpected_keys
        missing = [k for k in result.missing_keys if not k.startswith(NEW_PREFIXES)]
        assert not missing, f"keys of M* missing from the state: {missing}"

    # ------------------------------------------------------------------ the history path

    def raw_path_output(self, *, inp: UntInputs) -> torch.Tensor:
        """Return `W2 GELU(W1 [u·ok, v·ok, w·ok, ok])` before the divisor and the gain (sets `r`)."""
        assert self.u_w1 is not None and self.u_w2 is not None
        okf = inp.hist_u_ok.to(inp.hist_u.dtype)
        features = torch.cat([inp.hist_u * okf[..., None], okf[..., None]], dim=-1)
        return self.u_w2(functional.gelu(self.u_w1(features)))

    def hist_path(self, inp: UntInputs) -> torch.Tensor:
        """Return the history token path output `(B, 14, 48, 64)`.

        The output is gated by the per-token flag, so a token whose channel is absent receives
        exactly zero, whatever the weights and the gain hold.
        """
        assert self.u_gamma is not None
        okf = inp.hist_u_ok.to(inp.hist_u.dtype)[..., None]
        return okf * (self.u_gamma * (self.raw_path_output(inp=inp) / self.u_r))

    @torch.no_grad()
    def calibrate_scale(self, *, batches: list[UntInputs]) -> float:
        """Set `r` to the root mean square of the path before the gain over every token with the flag on.

        Args:
            batches: The arm's own training batches (one pass).

        Returns:
            The root mean square.

        Raises:
            ValueError: When no token has its flag on (the divisor would be empty).
        """
        total, count = 0.0, 0
        for batch in batches:
            picked = self.raw_path_output(inp=batch)[batch.hist_u_ok]
            total += float((picked.double() ** 2).sum())
            count += picked.numel()
        if count == 0:
            msg = "no history token has its flag on, so the scale of the path cannot be set"
            raise ValueError(msg)
        rms = math.sqrt(total / count)
        self.u_r.fill_(rms)
        return rms

    def _tokens(self, inp: UntInputs) -> tuple[torch.Tensor, torch.Tensor]:  # ty: ignore[invalid-method-override]
        """Return 713's tokens with the path added to history rows 1 to 14."""
        x, row_exists = super()._tokens(inp)
        if self.u_w1 is not None and not self.switches.top_row_only:
            x = torch.cat([x[:, :1], x[:, 1:] + self.hist_path(inp)], dim=1)
        return x, row_exists

    # ------------------------------------------------------------------ the read-out and the heads

    @staticmethod
    def readout_means(*, inp: UntInputs, alpha: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the weighted means `[ubar, vbar, wbar]` `(B, 48, 3)` and the cover `(B, 48)`.

        The weights are `α` times the flag, so a token without a value takes no weight.
        """
        okf = inp.hist_u_ok.to(alpha.dtype)
        weight = alpha * okf
        cover = weight.sum(dim=1)
        means = torch.einsum("brj,brjc->bjc", weight, inp.hist_u) / cover.clamp_min(1e-6)[..., None]
        return means, cover

    def _mixture_untangled(
        self,
        *,
        h0: torch.Tensor,
        base: torch.Tensor,
        spread: torch.Tensor,
        no_history: torch.Tensor,
        lead_day: torch.Tensor,
        means: torch.Tensor,
        cover: torch.Tensor,
    ) -> torch.Tensor:
        """Return the mixture outputs with the read-out features added to the head's first layer."""
        assert self.u_head_in is not None
        lead = functional.one_hot(lead_day, mdl.N_LEAD_DAYS).to(h0.dtype)[:, None, :]
        x = torch.cat(
            [
                h0,
                base[..., None],
                spread[..., None],
                no_history[..., None],
                lead.expand(-1, mdl.N_COLS, -1),
            ],
            dim=-1,
        )
        feats = torch.cat([means, cover[..., None]], dim=-1)
        hidden = self.head[0](x) + self.u_head_in(feats)
        out = self.head[1:](hidden)
        k = mdl.m559.COMPONENTS
        return torch.cat(
            [out[..., :k], out[..., k : 2 * k] + base[..., None], out[..., 2 * k :]], dim=-1
        )

    def sign_logit_of(
        self, *, h0: torch.Tensor, lead_day: torch.Tensor, ubar: torch.Tensor, cover: torch.Tensor
    ) -> torch.Tensor:
        """Return the logit of P(X<0) for each target half-hour `(B, 48)`.

        The logit is `b + cover (ū − b) + f(h0, ū, cover, lead day)`, where `b` is the logit of the
        training banks' share of reverse flow, `ū` the logit of the attention-weighted mean of the
        history's P(X<0), and `f` ends in a layer whose weights start at zero. At the start the head
        is the persistence of the history's P(X<0) where the history covers the column, and the
        climatology where it does not.
        """
        assert self.sign_hidden is not None and self.sign_out is not None
        u_logit = logit(p=ubar)
        lead = functional.one_hot(lead_day, mdl.N_LEAD_DAYS).to(h0.dtype)[:, None, :]
        feats = torch.cat(
            [h0, u_logit[..., None], cover[..., None], lead.expand(-1, mdl.N_COLS, -1)], dim=-1
        )
        learned = self.sign_out(functional.gelu(self.sign_hidden(feats)))[..., 0]
        skip = cover * (u_logit - self.sign_clim)
        return learned + skip

    def forward(self, inp: UntInputs, *, return_attention: bool = False) -> UntOutput:  # ty: ignore[invalid-method-override]  # noqa: PLR0912
        """Return the raw mixture outputs; a network with no new part runs 713's forward unchanged."""
        if not self.has_path and not self.has_sign_head:
            base_out = super().forward(inp, return_attention=return_attention)
            return UntOutput(**base_out.__dict__)
        x, row_exists = self._tokens(inp)
        attention_modules = []
        if return_attention:
            for i, block in enumerate(self.blocks):
                for kind in ("within", "across"):
                    module = getattr(block, kind)
                    if module is not None:
                        module.record = {}
                        attention_modules.append((f"{kind}_{i}", module))
        for block in self.blocks:
            x = block(x, row_exists=row_exists)
        h0, base, spread, no_history, alpha = self._readout(x, inp)
        if self.has_path:
            means, cover = self.readout_means(inp=inp, alpha=alpha)
            out = self._mixture_untangled(
                h0=h0, base=base, spread=spread, no_history=no_history, lead_day=inp.lead_day,
                means=means, cover=cover,
            )  # fmt: skip
        else:  # the control arm of the synthetic world: the sign head and no untangled path
            means = torch.zeros(
                *cover_shape(base), core.N_CHANNELS, device=base.device, dtype=base.dtype
            )
            cover = torch.zeros_like(base)
            out = self._mixture_out(
                h0=h0, base=base, spread=spread, no_history=no_history, lead_day=inp.lead_day
            )
        sign = None
        if self.has_sign_head:
            sign = self.sign_logit_of(
                h0=h0, lead_day=inp.lead_day, ubar=means[..., core.P_NEG], cover=cover
            )
        attention = None
        if return_attention:
            attention = {}
            b = len(inp.primary)
            for name, module in attention_modules:
                w = module.record.pop("weights")
                module.record = None
                if name.startswith("within"):
                    attention[name] = w.view(b, -1, *w.shape[1:])
                else:
                    attention[name] = w[:, :, 0, :].view(b, mdl.N_COLS, *w.shape[1:2], w.shape[-1])
        return UntOutput(
            out=out, base=base, spread=spread, no_history=no_history, alpha=alpha,  # ty: ignore[unknown-argument]
            attention=attention, target_vector=h0.detach() if return_attention else None,  # ty: ignore[unknown-argument]
            sign_logit=sign,
        )  # fmt: skip


def cover_shape(base: torch.Tensor) -> tuple[int, ...]:
    """Return the `(B, 48)` shape of a per-column tensor."""
    return tuple(base.shape)


def new_parameter_count(*, model: UntangledNet) -> int:
    """Return the number of new parameters (every `u_*` and `sign_*` parameter)."""
    return sum(p.numel() for n, p in model.named_parameters() if n.startswith(NEW_PREFIXES))


def sign_loss(*, logit_: torch.Tensor, truth: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Return the weighted binary cross-entropy of the sign head.

    Args:
        logit_: `(B, 48)` logits.
        truth: `(B, 48)` indicator of X < 0 (NaN where unknown).
        weight: `(B, 48)` weights (zero where the target is unknown).

    Returns:
        The mean over the weighted tokens.
    """
    ok = torch.isfinite(truth) & (weight > 0)
    t = torch.nan_to_num(truth)
    bce = functional.binary_cross_entropy_with_logits(logit_, t, reduction="none")
    w = weight * ok
    return (w * bce).sum() / w.sum().clamp_min(1e-9)
