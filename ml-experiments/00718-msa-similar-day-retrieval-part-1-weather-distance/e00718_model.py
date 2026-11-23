# ruff: noqa: E501, PT018, PLR1714
"""Experiment 718: the axial-attention demand network with age-bucketed tables (design section 3).

`F9Net` subclasses experiment 713's `MsaNet`. Experiment 713 indexes three parameter tables by the
slot of a history row: `age_emb` (added to every token of a row), `beta_age` (added to the read-out
logit), and the across-day `key_bias` (one entry per key row and head). Here the three tables are
indexed by the age bucket of the row (`e00718_layout.age_bucket`), so a row keeps its meaning wherever
it sits in the tensor, and the network reads the history rows as a set. The history may hold 14 or
27 rows.

The first 14 buckets (ages 2 to 15) are experiment 713's slots. The 14-row tables are built inside
`MsaNet.__init__`, in experiment 713's random-number order, and extended afterwards: the 10 new
rows of `age_emb` come from a generator of their own that is seeded by the network seed, and the new
`beta_age` and key-bias entries start at zero. A fresh F9 network of seed `s` therefore equals a fresh
`MsaNet` of seed `s` on every shared parameter, and every F9 arm of one seed starts from the same
values. An F9 network with rows of ages 2 to 15 is experiment 713's network exactly.
"""

import copy
from dataclasses import dataclass
from typing import Final

import e00713_model as mdl  # ty: ignore[unresolved-import]
import e00718_layout as lay
import numpy as np
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]
from torch.nn import functional  # ty: ignore[unresolved-import]

AGE_INIT_STD: Final = 0.02
SHARED_PARAMETERS: Final = 129_999
"""The parameters of experiment 713's `M*`, all of which an F9 network shares."""
NEW_PARAMETERS: Final = 730
"""640 in the age table, 10 in `beta_age`, and 80 in the two key-bias tables."""
TARGET_KEY: Final = 0
"""The key-bias row of the target row; the key row of bucket `b` is `b + 1`."""


@dataclass
class F9Inputs(mdl.MsaInputs):
    """One batch: experiment 713's inputs with any number of history rows, plus the age buckets."""

    age_bucket: torch.Tensor | None = None  # (B, H) long, 0 to 23

    def to(self, device: str | torch.device) -> F9Inputs:
        """Return the batch on `device`."""
        return F9Inputs(**{k: v.to(device) for k, v in self.__dict__.items() if v is not None})

    @property
    def n_history(self) -> int:
        """Return the number of history rows."""
        return self.hist_power.shape[1]


class F9Attention(mdl.Attention):
    """Across-day attention whose key bias is gathered by the age bucket of each key row."""

    @classmethod
    def upgrade(cls, att: mdl.Attention) -> F9Attention:
        """Return a copy of `att` whose key bias has 25 entries per head (the first 15 are kept)."""
        assert att.key_bias is not None
        new = copy.deepcopy(att)
        new.__class__ = cls
        old = att.key_bias.detach()
        assert old.shape == (mdl.HEADS, mdl.N_ROWS)
        new.key_bias = nn.Parameter(
            torch.cat([old, torch.zeros(mdl.HEADS, lay.N_KEY_ROWS - mdl.N_ROWS)], dim=1)
        )
        return new  # ty: ignore[invalid-return-type]

    def forward(  # ty: ignore[invalid-method-override]
        self,
        x: torch.Tensor,
        *,
        key_ok: torch.Tensor | None = None,
        key_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attend over the length-L axis of `x` (N, L, 64); `key_ids` (N, L) selects each key's bias."""
        assert key_ids is not None and self.key_bias is not None
        n, length, _ = x.shape
        q, k, v = (
            self.qkv(x).view(n, length, 3, mdl.HEADS, mdl.WIDTH // mdl.HEADS).permute(2, 0, 3, 1, 4)
        )
        bias = torch.zeros(n, mdl.HEADS, 1, length, device=x.device, dtype=x.dtype)
        bias = bias + self.key_bias[:, key_ids].permute(1, 0, 2)[:, :, None, :]
        if key_ok is not None:
            bias = bias.masked_fill(~key_ok[:, None, None, :], float("-inf"))
        if self.record is not None:
            with torch.no_grad():
                logits = q @ k.transpose(-1, -2) / float(np.sqrt(mdl.WIDTH // mdl.HEADS)) + bias
                self.record["weights"] = torch.softmax(logits, dim=-1)
        y = functional.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, dropout_p=mdl.DROPOUT if self.training else 0.0
        )
        return self.out(y.transpose(1, 2).reshape(n, length, mdl.WIDTH))


class F9Block(mdl.AxialBlock):
    """An axial block whose across-day attention reads the bucket-indexed key bias."""

    @classmethod
    def upgrade(cls, block: mdl.AxialBlock) -> F9Block:
        """Return a copy of `block` with the bucketed across-day attention."""
        new = copy.deepcopy(block)
        new.__class__ = cls
        if new.across is not None:
            new.across = F9Attention.upgrade(block.across)  # ty: ignore[invalid-argument-type]
        return new  # ty: ignore[invalid-return-type]

    def forward(  # ty: ignore[invalid-method-override]
        self, x: torch.Tensor, *, row_exists: torch.Tensor, key_ids: torch.Tensor
    ) -> torch.Tensor:
        """Return `x` (B, R, 48, 64) after the block; `row_exists` is (B, R), `key_ids` (B, R)."""
        b, r, c, w = x.shape
        if self.within is not None and self.ln_within is not None:
            y = self.within(self.ln_within(x).reshape(b * r, c, w))
            x = x + self.drop(y.view(b, r, c, w))
        if self.across is not None and self.ln_across is not None:
            cols = self.ln_across(x).permute(0, 2, 1, 3).reshape(b * c, r, w)
            ok = row_exists[:, None, :].expand(b, c, r).reshape(b * c, r)
            ids = key_ids[:, None, :].expand(b, c, r).reshape(b * c, r)
            y = self.across(cols, key_ok=ok, key_ids=ids).view(b, c, r, w).permute(0, 2, 1, 3)  # ty: ignore[unknown-argument]
            x = x + self.drop(y)
        return x + self.drop(self.transition(self.ln_transition(x)))


class F9Net(mdl.MsaNet):
    """The axial-attention network with the age table, `beta_age`, and the key bias indexed by age bucket."""

    def __init__(self, *, seed: int) -> None:
        """Build the network of experiment 713's arm `M`, then extend the three tables.

        The caller seeds the global generator (`torch.manual_seed(seed)`) before this call, as
        experiment 713's `new_model` does. `seed` seeds the generator of the 10 new age rows.
        """
        super().__init__(switches=mdl.ARM_SWITCHES["M"])
        old_age = self.age_emb.weight.detach().clone()
        generator = torch.Generator().manual_seed(seed)
        extra = (
            torch.randn(lay.N_BUCKETS - mdl.N_HISTORY, mdl.WIDTH, generator=generator)
            * AGE_INIT_STD
        )
        self.age_emb = nn.Embedding(lay.N_BUCKETS, mdl.WIDTH)
        with torch.no_grad():
            self.age_emb.weight.copy_(torch.cat([old_age, extra]))
        old_beta = self.beta_age.detach().clone()
        self.beta_age = nn.Parameter(
            torch.cat([old_beta, torch.zeros(lay.N_BUCKETS - mdl.N_HISTORY)])
        )
        self.blocks = nn.ModuleList(F9Block.upgrade(b) for b in self.blocks)

    # ------------------------------------------------------------------ state of experiment 713

    def load_mstar_state(self, state: dict[str, torch.Tensor]) -> None:
        """Load a state of experiment 713's `MsaNet` (arm `M`): the 14-row tables fill the first buckets.

        The shared parameters are copied, `age_emb` and `beta_age` fill their first 14 rows, and each
        key bias fills its first 15 entries. The new rows and entries keep their values.
        """
        mine = self.state_dict()
        for key, value in state.items():
            assert key in mine, f"{key} is not a parameter of F9Net"
            if mine[key].shape == value.shape:
                mine[key].copy_(value)
            elif key == "age_emb.weight" or key == "beta_age":
                mine[key][: mdl.N_HISTORY].copy_(value)
            elif key.endswith("across.key_bias"):
                mine[key][:, : mdl.N_ROWS].copy_(value)
            else:
                msg = f"{key}: shape {tuple(value.shape)} does not fit {tuple(mine[key].shape)}"
                raise AssertionError(msg)
        missing = set(mine) - set(state)
        assert not missing, f"the state of experiment 713 lacks {sorted(missing)}"

    # ------------------------------------------------------------------ forward

    def _tokens_f9(self, inp: F9Inputs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the tokens (B, 1 + H, 48, 64), the (B, 1 + H) row-exists mask, and the key ids."""
        assert inp.age_bucket is not None
        b, h = len(inp.primary), inp.n_history
        emb = self.embedding(inp.primary)
        weather = torch.cat([inp.top_weather[:, None], inp.hist_weather], dim=1)
        datetime_ = torch.cat([inp.top_datetime[:, None], inp.hist_datetime], dim=1)
        e = emb[:, None, None, :].expand(b, 1 + h, mdl.N_COLS, emb.shape[-1])
        w_in = torch.cat([weather[..., : mdl.N_WEATHER], e, weather[..., mdl.N_WEATHER :]], dim=-1)
        e_w = functional.gelu(self.weather(w_in))
        e_d = functional.gelu(self.datetime(torch.cat([datetime_, e], dim=-1)))
        x = self.proj_w(e_w) + self.proj_d(e_d)
        x = x + self.tok_emb(inp.primary)[:, None, None, :] + self.col_emb.weight[None, None]
        top = (
            self.lead_emb(inp.lead_day)[:, None, :]
            + self.proj_s(inp.top_spread)
            + self.m_target[None, None, :]
        )
        z = inp.hist_power.clamp(-mdl.POWER_CLIP, mdl.POWER_CLIP)
        power = torch.where(
            inp.hist_visible[..., None],
            z[..., None] * self.power_u + self.power_b,
            self.m_hist.expand_as(x[:, 1:]),
        )
        hist = x[:, 1:] + self.age_emb(inp.age_bucket)[:, :, None, :] + power
        row_exists = torch.ones(b, 1 + h, dtype=torch.bool, device=x.device)
        row_exists[:, 1:] = inp.row_exists
        key_ids = torch.cat(
            [torch.full((b, 1), TARGET_KEY, dtype=torch.long, device=x.device), inp.age_bucket + 1],
            dim=1,
        )
        return torch.cat([x[:, :1] + top[:, None], hist], dim=1), row_exists, key_ids

    def _readout_f9(self, x: torch.Tensor, inp: F9Inputs) -> tuple[torch.Tensor, ...]:
        """Return `h0`, `base`, `spread`, `no_history`, and `alpha` with `beta_age` read by bucket."""
        assert inp.age_bucket is not None
        h = self.ln_readout(x)
        h0 = h[:, 0]
        lead = functional.one_hot(inp.lead_day, mdl.N_LEAD_DAYS).to(h0.dtype)
        q = self.w_q(h0) + self.w_lead(lead)[:, None, :]
        k = self.w_k(h[:, 1:])
        logit = torch.einsum("bjd,brjd->brj", q, k) / float(np.sqrt(mdl.READOUT_WIDTH))
        day = self.beta_day[inp.day_type_top[:, None, :], inp.day_type_hist]
        logit = logit + day + self.beta_age[inp.age_bucket][:, :, None]
        visible = inp.hist_visible & inp.row_exists[:, :, None]
        any_visible = visible.any(dim=1)
        logit = logit.masked_fill(~visible, -1e9)
        alpha = torch.softmax(logit, dim=1) * any_visible[:, None, :]
        z = inp.hist_power.clamp(-mdl.POWER_CLIP, mdl.POWER_CLIP)
        base = (alpha * z).sum(dim=1)
        var = (alpha * (z - base[:, None, :]) ** 2).sum(dim=1)
        spread = torch.where(any_visible, torch.sqrt(var + 1e-6), torch.zeros_like(var))
        return h0, base, spread, (~any_visible).to(x.dtype), alpha

    def forward(self, inp: F9Inputs, *, return_attention: bool = False) -> mdl.MsaOutput:  # ty: ignore[invalid-method-override]
        """Return the raw mixture outputs of the 48 target tokens of each sample.

        With `return_attention` the output also holds the attention weights of every block and the
        64-number vector that enters the head; the flag only records.
        """
        x, row_exists, key_ids = self._tokens_f9(inp)
        attention_modules: list[tuple[str, mdl.Attention]] = []
        if return_attention:
            for i, block in enumerate(self.blocks):
                for kind in ("within", "across"):
                    module = getattr(block, kind)
                    if module is not None:
                        module.record = {}
                        attention_modules.append((f"{kind}_{i}", module))
        for block in self.blocks:
            x = block(x, row_exists=row_exists, key_ids=key_ids)
        h0, base, spread, no_history, alpha = self._readout_f9(x, inp)
        out = self._mixture_out(
            h0=h0, base=base, spread=spread, no_history=no_history, lead_day=inp.lead_day
        )
        attention = None
        if return_attention:
            attention = {}
            b = len(inp.primary)
            for name, module in attention_modules:
                assert module.record is not None
                w = module.record.pop("weights")
                module.record = None
                if name.startswith("within"):
                    attention[name] = w.view(b, -1, *w.shape[1:])
                else:
                    attention[name] = w[:, :, 0, :].view(b, mdl.N_COLS, *w.shape[1:2], w.shape[-1])
        return mdl.MsaOutput(
            out=out, base=base, spread=spread, no_history=no_history, alpha=alpha,
            attention=attention, target_vector=h0.detach() if return_attention else None,
        )  # fmt: skip


def tokens_per_sample(*, n_history: int) -> int:
    """Return the tokens of one sample: the target row and the history rows of 48 half-hours."""
    return (1 + n_history) * mdl.N_COLS
