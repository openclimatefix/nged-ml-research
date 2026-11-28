"""The level head on a frozen experiment 713 network, and the four ways of carrying the level term.

`LevelNet` subclasses `MsaNet`. Its base is the stored stage 3 state of `M1`, frozen, so with the
level term at zero every head arm gives `M1`'s forecast. The head reads only quantities that
`Source.assemble` has already limited to the visibility limit (the 14 history rows, the target row's
weather and calendar through the frozen trunk, the lead day), and the fold's noise unit and survival
curve, which are fitted on the training window.

Arms (`mode`):

- `LP`: the level term is added to the three mixture means at every lead (full persistence). Its
  depth reads only the history rows and the Primary, so the level term is the same at every lead.
- `LD`: the level term times the survival ratio p(a, h) is added to the three means (a decay).
- `LM`: a mixture of six components: the base three with weight 1 - p, and a shifted copy with
  weight p, with p from the fixed fitted curve.
- `LR`: as `LM`, with p read from 10 learned non-negative daily hazards.

W0 is the plain `MsaNet`, trained by `e00738_train`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import e00397_models as base_models  # ty: ignore[unresolved-import]
import e00559_model as m559  # ty: ignore[unresolved-import]
import e00713_model as mdl  # ty: ignore[unresolved-import]
import e00738_common as common
import e00738_survival as surv
import numpy as np
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]

N_FEATURES: Final[int] = mdl.WIDTH + 2 + mdl.N_LEAD_DAYS + 2 * mdl.N_HISTORY
"""Pooled target summary (64), read-out mean and spread (2), one-hot lead day (14), 14 row means,
and 14 presence flags."""
N_FEATURES_LP: Final[int] = 2 * mdl.N_HISTORY + mdl.N_PRIMARIES
"""`LP` reads only the history rows (14 row means, 14 presence flags) and the Primary's one-hot, so
its depth cannot change with the lead day or with the target row's weather."""
P_EPS: Final[float] = 1e-6
UNSEEN_LOGIT: Final[float] = -30.0
"""The shifted branch's log weight where A1 sees no shift, so that the arm equals `M1` there."""


@dataclass
class LevelOutput:
    """One forward pass of a level arm.

    Attributes:
        out: The raw mixture outputs (B, 48, 3 * K), K = 3 (LP, LD) or 6 (LM, LR).
        components: K.
        depth: The signed depth in standardised units after the tanh limit and the A1 gate, (B,).
        p: The survival ratio at each target token (B, 48); 1 for LP, 0 where A1 sees no shift.
        age: The age in days that A1 gives (B,); 0 where it sees no shift.
        hazards: The 10 daily hazards (B, 10) for LR, else None.
        features: The head's input (B, 108).
    """

    out: torch.Tensor
    components: int
    depth: torch.Tensor
    p: torch.Tensor
    age: torch.Tensor
    hazards: torch.Tensor | None
    features: torch.Tensor


def lead_days_of(*, lead_day: torch.Tensor, device: str | torch.device) -> torch.Tensor:
    """Return the lead in days after the contract origin of each of the 48 target tokens (B, 48).

    Args:
        lead_day: The zero-based lead day of each sample (B,).
        device: The device of the result.

    Returns:
        `(48 * lead_day + column) / 48`.
    """
    col = torch.arange(mdl.N_COLS, device=device, dtype=torch.float32)
    return (lead_day.to(torch.float32)[:, None] * mdl.N_COLS + col[None, :]) / float(mdl.N_COLS)


def torch_cumulative_hazard(*, hazards: torch.Tensor, t_days: torch.Tensor) -> torch.Tensor:
    """Return H(t) for daily hazards (B, 10) and times (B, 48), in torch."""
    n = hazards.shape[-1]
    days = torch.arange(n, device=hazards.device, dtype=t_days.dtype)
    share = torch.clamp(t_days[..., None] - days, 0.0, 1.0)
    tail = torch.clamp(t_days - n, min=0.0) * hazards[:, -1:]
    return (share * hazards[:, None, :]).sum(-1) + tail


class LevelNet(mdl.MsaNet):
    """`MsaNet` with a level head on a frozen base."""

    def __init__(
        self,
        *,
        mode: str,
        target_std: np.ndarray,
        noise: float,
        curve: surv.SurvivalFunction | None,
        a1_z: float = common.A1_Z,
    ) -> None:
        """Build the head.

        Args:
            mode: `LP`, `LD`, `LM`, or `LR`.
            target_std: The standard deviation of each Primary's power over capacity (16,).
            noise: The noise unit of A1, from the fold's training window.
            curve: The empirical survival curve of the training worlds (needed by `LD` and `LM`).
            a1_z: A1's detection threshold in noise units, as Phase 0 tuned it.
        """
        super().__init__()
        assert mode in common.HEAD_ARMS, mode
        assert mode not in {"LD", "LM"} or curve is not None, "LD and LM need the fitted curve"
        self.mode = mode
        self.noise = float(noise)
        self.a1_z = float(a1_z)
        self.curve = curve
        self.register_buffer(
            "depth_limit",
            torch.from_numpy(
                (common.LEVEL_DEPTH_LIMIT / np.asarray(target_std)).astype(np.float32)
            ),
        )
        n_out = 1 + (common.N_HAZARDS if mode == "LR" else 0)
        self.level_head = nn.Sequential(
            nn.Linear(N_FEATURES_LP if mode == "LP" else N_FEATURES, common.HIDDEN),
            nn.ReLU(),
            nn.Linear(common.HIDDEN, common.HIDDEN),
            nn.ReLU(),
            nn.Linear(common.HIDDEN, n_out),
        )
        # zero effect at the start: the depth output is zero, so every head arm equals the base
        nn.init.zeros_(self.level_head[-1].weight)  # ty: ignore[invalid-argument-type]
        nn.init.zeros_(self.level_head[-1].bias)  # ty: ignore[invalid-argument-type]
        self.base_frozen = False

    def level_parameters(self) -> list[nn.Parameter]:
        """Return the parameters of the level head, the only ones that train."""
        return list(self.level_head.parameters())

    def freeze_base(self) -> None:
        """Freeze every parameter except the level head, and keep the base in evaluation mode."""
        keep = {id(p) for p in self.level_parameters()}
        for p in self.parameters():
            p.requires_grad_(id(p) in keep)
        self.base_frozen = True
        self.train(self.training)

    def train(self, mode: bool = True) -> LevelNet:
        """Set the mode; a frozen base stays in evaluation mode (no dropout)."""
        super().train(mode)
        if self.base_frozen:
            for name, module in self.named_children():
                if name != "level_head":
                    module.eval()
        return self

    def load_base(self, *, state: dict[str, torch.Tensor]) -> None:
        """Load a stored stage 3 `M1` state into the base, exactly.

        Args:
            state: The state dict of `MsaNet`, which holds every key of the base and none of the
            head.
        """
        result = self.load_state_dict(state, strict=False)
        assert not result.unexpected_keys, result.unexpected_keys
        assert all(k.startswith(("level_head", "depth_limit")) for k in result.missing_keys), (
            result.missing_keys
        )

    # ------------------------------------------------------------------ features

    def a1_age(self, *, inp: mdl.MsaInputs) -> torch.Tensor:
        """Return A1's age in days (B,), 0 where no shift is seen. It reads history rows only."""
        power = inp.hist_power.detach().cpu().numpy()
        visible = inp.hist_visible.detach().cpu().numpy()
        age = surv.age_rule_a1(hist_power=power, visible=visible, noise=self.noise, z=self.a1_z)
        return torch.from_numpy(age.astype(np.float32)).to(inp.hist_power.device)

    def features_of(
        self, *, inp: mdl.MsaInputs, h0: torch.Tensor, base: torch.Tensor, spread: torch.Tensor
    ) -> torch.Tensor:
        """Return the head's input: (B, 108), or (B, 44) for `LP` (history only)."""
        n = inp.hist_visible.sum(dim=-1)
        present = n > 0
        total = torch.where(inp.hist_visible, inp.hist_power, torch.zeros_like(inp.hist_power)).sum(
            -1
        )
        row_mean = torch.where(present, total / n.clamp_min(1), torch.zeros_like(total))
        if self.mode == "LP":
            primary = nn.functional.one_hot(inp.primary, mdl.N_PRIMARIES).to(h0.dtype)
            return torch.cat([row_mean, present.to(h0.dtype), primary], dim=-1)
        lead = nn.functional.one_hot(inp.lead_day, mdl.N_LEAD_DAYS).to(h0.dtype)
        return torch.cat(
            [
                h0.mean(dim=1),
                base.mean(dim=1, keepdim=True),
                spread.mean(dim=1, keepdim=True),
                lead,
                row_mean,
                present.to(h0.dtype),
            ],
            dim=-1,
        )

    # ------------------------------------------------------------------ forward

    def base_forward(
        self, inp: mdl.MsaInputs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return `h0`, `base`, `spread`, and the base's raw mixture outputs, with no gradient."""
        with torch.no_grad():
            x, row_exists = self._tokens(inp)
            for block in self.blocks:
                x = block(x, row_exists=row_exists)
            h0, base, spread, no_history, _ = self._readout(x, inp)
            out = self._mixture_out(
                h0=h0, base=base, spread=spread, no_history=no_history, lead_day=inp.lead_day
            )
        return h0, base, spread, out

    def survival_ratio(
        self, *, age: torch.Tensor, lead_days: torch.Tensor, hazards: torch.Tensor | None
    ) -> torch.Tensor:
        """Return p(a, h) (B, 48): the fitted curve, learned hazards, or 1 for persistence."""
        if self.mode == "LP":
            return torch.ones_like(lead_days)
        if self.mode == "LR":
            assert hazards is not None
            a = age[:, None].expand_as(lead_days)
            s_a = torch.exp(-torch_cumulative_hazard(hazards=hazards, t_days=a))
            s_ah = torch.exp(-torch_cumulative_hazard(hazards=hazards, t_days=a + lead_days))
            return torch.clamp(s_ah / torch.clamp(s_a, min=common.SURVIVAL_FLOOR), 0.0, 1.0)
        assert self.curve is not None
        p = surv.ratio(
            curve=self.curve,
            age_days=age.detach().cpu().numpy()[:, None],
            lead_days=lead_days.detach().cpu().numpy(),
        )
        return torch.from_numpy(p.astype(np.float32)).to(lead_days.device)

    def level_forward(self, inp: mdl.MsaInputs) -> LevelOutput:
        """Return the mixture of one batch with the level term carried by this arm's rule."""
        h0, base, spread, out0 = self.base_forward(inp)
        feats = self.features_of(inp=inp, h0=h0, base=base, spread=spread)
        raw = self.level_head(feats)
        age = self.a1_age(inp=inp)
        seen = (age > 0).to(raw.dtype)
        limit = self.depth_limit[inp.primary]
        depth = limit * torch.tanh(raw[:, 0]) * seen
        hazards = None
        if self.mode == "LR":
            hazards = nn.functional.softplus(raw[:, 1:] - common.HAZARD_OFFSET)
        lead_days = lead_days_of(lead_day=inp.lead_day, device=raw.device)
        p = self.survival_ratio(age=age, lead_days=lead_days, hazards=hazards)
        p = p * seen[:, None]
        k = m559.COMPONENTS
        logit, mean, log_std = out0[..., :k], out0[..., k : 2 * k], out0[..., 2 * k :]
        if self.mode in {"LP", "LD"}:
            shift = depth[:, None, None] * (p[..., None] if self.mode == "LD" else 1.0)
            out = torch.cat([logit, mean + shift, log_std], dim=-1)
            return LevelOutput(out, k, depth, p, age, hazards, feats)
        pc = p.clamp(P_EPS, 1.0 - P_EPS)
        log_keep = torch.log1p(-pc)[..., None]
        log_move = torch.where(
            seen[:, None, None] > 0,
            torch.log(pc)[..., None],
            torch.full_like(log_keep, UNSEEN_LOGIT),
        )
        out = torch.cat(
            [
                logit + log_keep,
                logit + log_move,
                mean,
                mean + depth[:, None, None],
                log_std,
                log_std,
            ],
            dim=-1,
        )
        return LevelOutput(out, 2 * k, depth, p, age, hazards, feats)


def nll_of(*, out: torch.Tensor, target: torch.Tensor, components: int) -> torch.Tensor:
    """Return the mixture negative log-likelihood of every target token (B, 48)."""
    return base_models.mixture_nll(out=out, z=target, components=components)


def quantiles_of(*, out: torch.Tensor, components: int) -> torch.Tensor:
    """Return the 11 quantiles (B, 48, 11) of a K-component mixture, in standardised units."""
    shape = out.shape[:-1]
    log_w, mean, std = base_models.mixture_parts(
        out=out.reshape(-1, out.shape[-1]), components=components
    )
    return base_models.mixture_quantiles(log_w=log_w, mean=mean, std=std).view(
        *shape, len(base_models.LEVELS)
    )


def mixture_weights(*, out: torch.Tensor, components: int) -> torch.Tensor:
    """Return the mixture weights (B, 48, K), which sum to 1 over the last axis."""
    return torch.softmax(out[..., :components], dim=-1)
