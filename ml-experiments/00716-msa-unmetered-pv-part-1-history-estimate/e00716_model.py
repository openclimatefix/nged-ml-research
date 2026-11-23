# ruff: noqa: E501, PT018, PERF403
"""Experiment 716: `PvNet`, the axial-attention network with the unmetered PV paths (design section 4, WP2).

`PvNet` subclasses experiment 713's `MsaNet`, keeps every module name, and builds the new modules last,
so a state dictionary of `M*` loads into it with only the new keys (`pv_*`) missing. Every new path
starts at exactly zero (a zero gain or a zero layer) with a random or already trained partner, so an
`M*` state gives the output of `M*` bit for bit for any finite PV input.

Three paths, each enabled by the arm:

1. History token path (arms `H`, `HT`, and their placebos): `γ_h ⊙ (W2_h GELU(W1_h [p·ok, ok]) / r_h)`
   added to each history token before the first axial block. `r_h` is a stored constant (a buffer
   set once by `calibrate_scale`), not a per-token LayerNorm.
2. Target-row token path (arms `HT`, `HT-S`, `T`): the same form on `[p0_mean·ok, p0_sd·ok, ok]`.
3. Read-out and head path (every PV arm): `pbar` and `cover` from the read-out weights `α`, entering
   the mixture means as `a[primary] × pbar × cover + c[primary] × p0_mean` and the head's first
   layer through `pv_head_in`.
"""

import math
from dataclasses import dataclass
from typing import Final, Literal

import e00713_model as mdl  # ty: ignore[unresolved-import]
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]
from torch.nn import functional  # ty: ignore[unresolved-import]

PATH_HIDDEN: Final = 32
HEAD_FEATURES: Final = 5
"""`[pbar, cover, p0_mean, p0_sd, top_ok]`."""
NEW_PREFIX: Final = "pv_"
PvArm = Literal["Z", "H", "HT", "T"]
"""The network of each arm: `Z` has no PV path, `H` the history path, `HT` both token paths, `T` the target path.

The placebos `H-S`, `H-X`, and `HT-S` use the network of their treatment: `H`, `H`, and `HT`.
"""


@dataclass(frozen=True)
class PvSwitches:
    """Which new paths an arm builds.

    Attributes:
        history: The history token path.
        target: The target-row token path.
    """

    history: bool
    target: bool

    @property
    def any_path(self) -> bool:
        """Return whether the arm builds any new parameter."""
        return self.history or self.target


PV_SWITCHES: Final[dict[str, PvSwitches]] = {
    "Z": PvSwitches(history=False, target=False),
    "H": PvSwitches(history=True, target=False),
    "HT": PvSwitches(history=True, target=True),
    "T": PvSwitches(history=False, target=True),
}


@dataclass
class PvInputs(mdl.MsaInputs):
    """713's batch plus the four PV tensors (section 4.1)."""

    hist_pv: torch.Tensor  # (B, 14, 48) `p` at each history token's own half-hour; 0 where absent
    hist_pv_ok: torch.Tensor  # (B, 14, 48) bool
    top_pv: torch.Tensor  # (B, 48, 2) `p0_mean` and `p0_sd` of the target row
    top_pv_ok: torch.Tensor  # (B, 48) bool

    def to(self, device: str | torch.device) -> PvInputs:
        """Return the batch on `device`."""
        return PvInputs(**{k: v.to(device) for k, v in self.__dict__.items()})

    def with_power(self, *, power: torch.Tensor, visible: torch.Tensor) -> PvInputs:
        """Return the batch with another history power and visibility, keeping the PV tensors."""
        values = dict(self.__dict__)
        values.update(hist_power=power, hist_visible=visible)
        return PvInputs(**values)

    def with_pv(
        self, *, hist_pv: torch.Tensor | None = None, hist_pv_ok: torch.Tensor | None = None,
        top_pv: torch.Tensor | None = None, top_pv_ok: torch.Tensor | None = None,
    ) -> PvInputs:  # fmt: skip
        """Return the batch with some of the PV tensors replaced."""
        values = dict(self.__dict__)
        for name, value in (
            ("hist_pv", hist_pv),
            ("hist_pv_ok", hist_pv_ok),
            ("top_pv", top_pv),
            ("top_pv_ok", top_pv_ok),
        ):
            if value is not None:
                values[name] = value
        return PvInputs(**values)


def init_like_default(*, layer: nn.Linear, generator: torch.Generator) -> None:
    """Draw `layer`'s weights and bias from PyTorch's default initialisation, with `generator`.

    The default is the uniform distribution on `±1/sqrt(fan_in)` for both the weight and the bias.
    """
    bound = 1.0 / math.sqrt(layer.in_features)
    with torch.no_grad():
        layer.weight.uniform_(-bound, bound, generator=generator)
        if layer.bias is not None:
            layer.bias.uniform_(-bound, bound, generator=generator)


class PvNet(mdl.MsaNet):
    """The axial-attention network with the unmetered PV paths."""

    def __init__(
        self, *, arm: PvArm = "H", seed: int = 0, switches: mdl.ArmSwitches = mdl.ARM_SWITCHES["M"]
    ) -> None:
        """Build `M*`'s modules, then the PV modules of `arm` from a generator seeded by `seed`.

        History modules are drawn first, so `H`, `H-S`, `HT`, and `HT-S` of one seed start from the same
        `W1_h` and `W2_h`. The global torch stream is saved and restored around the new modules.
        """
        super().__init__(switches=switches)
        self.arm: PvArm = arm
        self.pv = PV_SWITCHES[arm]
        self.pv_hist_w1 = self.pv_hist_w2 = self.pv_top_w1 = self.pv_top_w2 = None
        self.pv_hist_gamma = self.pv_top_gamma = None
        self.pv_a = self.pv_c = self.pv_head_in = None
        if not self.pv.any_path:
            return
        generator = torch.Generator().manual_seed(seed)
        with torch.random.fork_rng(devices=[]):
            if self.pv.history:
                self.pv_hist_w1 = nn.Linear(2, PATH_HIDDEN)
                self.pv_hist_w2 = nn.Linear(PATH_HIDDEN, mdl.WIDTH, bias=False)
                init_like_default(layer=self.pv_hist_w1, generator=generator)
                init_like_default(layer=self.pv_hist_w2, generator=generator)
                self.pv_hist_gamma = nn.Parameter(torch.zeros(mdl.WIDTH))
                self.register_buffer("pv_hist_r", torch.ones(()))
            if self.pv.target:
                self.pv_top_w1 = nn.Linear(3, PATH_HIDDEN)
                self.pv_top_w2 = nn.Linear(PATH_HIDDEN, mdl.WIDTH, bias=False)
                init_like_default(layer=self.pv_top_w1, generator=generator)
                init_like_default(layer=self.pv_top_w2, generator=generator)
                self.pv_top_gamma = nn.Parameter(torch.zeros(mdl.WIDTH))
                self.register_buffer("pv_top_r", torch.ones(()))
            self.pv_a = nn.Parameter(torch.zeros(mdl.N_PRIMARIES))
            self.pv_c = nn.Parameter(torch.zeros(mdl.N_PRIMARIES))
            self.pv_head_in = nn.Linear(HEAD_FEATURES, mdl.WIDTH, bias=False)
            nn.init.zeros_(self.pv_head_in.weight)

    # ------------------------------------------------------------------ state

    def new_parameter_names(self) -> list[str]:
        """Return the names of the new trainable parameters (every `pv_*` parameter)."""
        return [n for n, _ in self.named_parameters() if n.startswith(NEW_PREFIX)]

    def unused_gain_names(self) -> list[str]:
        """Return the gains the arm builds and never uses: `c` where `p0_mean` is zero, `a` in `T`."""
        names = []
        if not self.pv.target and self.pv.any_path:
            names.append("pv_c")
        if not self.pv.history and self.pv.any_path:
            names.append("pv_a")
        return names

    def load_m_star(self, *, state: dict[str, torch.Tensor]) -> None:
        """Load a state of `M*` (or of an arm without PV paths): only the `pv_*` keys may be missing."""
        result = self.load_state_dict(state, strict=False)
        assert not result.unexpected_keys, result.unexpected_keys
        missing = [k for k in result.missing_keys if not k.startswith(NEW_PREFIX)]
        assert not missing, f"keys of M* missing from the state: {missing}"

    # ------------------------------------------------------------------ the token paths

    def _path(
        self, *, w1: nn.Linear, w2: nn.Linear, gamma: torch.Tensor, r: torch.Tensor,
        values: torch.Tensor, ok: torch.Tensor,
    ) -> torch.Tensor:  # fmt: skip
        """Return `γ ⊙ (W2 GELU(W1 [values·ok, ok]) / r)` for values (..., n) and ok (...) of a path."""
        okf = ok.to(values.dtype)
        features = torch.cat([values * okf[..., None], okf[..., None]], dim=-1)
        return gamma * (w2(functional.gelu(w1(features))) / r)

    def _tokens(self, inp: PvInputs) -> tuple[torch.Tensor, torch.Tensor]:  # ty: ignore[invalid-method-override]
        """Return 713's tokens with the history path added to rows 1 to 14 and the target path to row 0."""
        x, row_exists = super()._tokens(inp)
        if self.pv_hist_w1 is not None and not self.switches.top_row_only:
            hist = self.pv_hist_path(inp)
            x = torch.cat([x[:, :1], x[:, 1:] + hist], dim=1)
        if self.pv_top_w1 is not None:
            top = self.pv_top_path(inp)
            x = torch.cat([x[:, :1] + top[:, None], x[:, 1:]], dim=1)
        return x, row_exists

    def pv_hist_path(self, inp: PvInputs) -> torch.Tensor:
        """Return the history token path output (B, 14, 48, 64)."""
        assert (
            self.pv_hist_w1 is not None
            and self.pv_hist_w2 is not None
            and self.pv_hist_gamma is not None
        )
        return self._path(
            w1=self.pv_hist_w1, w2=self.pv_hist_w2, gamma=self.pv_hist_gamma, r=self.pv_hist_r,
            values=inp.hist_pv[..., None], ok=inp.hist_pv_ok,
        )  # fmt: skip

    def pv_top_path(self, inp: PvInputs) -> torch.Tensor:
        """Return the target-row token path output (B, 48, 64)."""
        assert (
            self.pv_top_w1 is not None
            and self.pv_top_w2 is not None
            and self.pv_top_gamma is not None
        )
        return self._path(
            w1=self.pv_top_w1, w2=self.pv_top_w2, gamma=self.pv_top_gamma, r=self.pv_top_r,
            values=inp.top_pv, ok=inp.top_pv_ok,
        )  # fmt: skip

    def raw_path_output(self, *, which: Literal["hist", "top"], inp: PvInputs) -> torch.Tensor:
        """Return `W2 GELU(W1 [...])` before the divisor and the gain (used to set `r`)."""
        if which == "hist":
            assert self.pv_hist_w1 is not None and self.pv_hist_w2 is not None
            w1, w2, values, ok = (
                self.pv_hist_w1,
                self.pv_hist_w2,
                inp.hist_pv[..., None],
                inp.hist_pv_ok,
            )
        else:
            assert self.pv_top_w1 is not None and self.pv_top_w2 is not None
            w1, w2, values, ok = self.pv_top_w1, self.pv_top_w2, inp.top_pv, inp.top_pv_ok
        okf = ok.to(values.dtype)
        features = torch.cat([values * okf[..., None], okf[..., None]], dim=-1)
        return w2(functional.gelu(w1(features)))

    @torch.no_grad()
    def calibrate_scale(self, *, batches: list[PvInputs]) -> dict[str, float]:
        """Set the stored divisors `r_h` and `r_t` from the arm's own training samples, once, after the random build.

        `r` is the root mean square of `W2 GELU(W1 [...])` over every element of every token with
        `ok` true, so the divided path has about unit variance per element. It is never trained.
        """
        out = {}
        for which, enabled in (
            ("hist", self.pv_hist_w1 is not None),
            ("top", self.pv_top_w1 is not None),
        ):
            if not enabled:
                continue
            buffer = self.pv_hist_r if which == "hist" else self.pv_top_r
            total, count = 0.0, 0
            for batch in batches:
                y = self.raw_path_output(which=which, inp=batch)  # ty: ignore[invalid-argument-type]
                ok = batch.hist_pv_ok if which == "hist" else batch.top_pv_ok
                picked = y[ok]
                total += float((picked.double() ** 2).sum())
                count += picked.numel()
            assert count > 0, f"no token with ok true to calibrate r_{which}"
            rms = math.sqrt(total / count)
            buffer.fill_(rms)
            out[which] = rms
        return out

    # ------------------------------------------------------------------ the read-out and the head

    def _without_unused_parts(self, inp: PvInputs) -> PvInputs:
        """Feed zeros for the part of the PV input that the arm does not use (`p0` in `H`, the history `p` in `T`)."""
        if not self.pv.target:
            inp = inp.with_pv(
                top_pv=torch.zeros_like(inp.top_pv), top_pv_ok=torch.zeros_like(inp.top_pv_ok)
            )
        if not self.pv.history:
            inp = inp.with_pv(
                hist_pv=torch.zeros_like(inp.hist_pv), hist_pv_ok=torch.zeros_like(inp.hist_pv_ok)
            )
        return inp

    def pv_readout(self, inp: PvInputs, alpha: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `pbar` and `cover` (B, 48): the PV inside `base`, and the weight of the rows that have it."""
        ok = inp.hist_pv_ok.to(alpha.dtype)
        weight = alpha * ok
        cover = weight.sum(dim=1)
        pbar = (weight * inp.hist_pv).sum(dim=1) / cover.clamp_min(1e-6)
        return pbar, cover

    def _mixture_pv(
        self, *, h0: torch.Tensor, base: torch.Tensor, spread: torch.Tensor, no_history: torch.Tensor,
        lead_day: torch.Tensor, primary: torch.Tensor, pbar: torch.Tensor, cover: torch.Tensor,
        top_pv: torch.Tensor, top_ok: torch.Tensor,
    ) -> torch.Tensor:  # fmt: skip
        """Return the mixture outputs with `a × pbar × cover + c × p0_mean` on the means and the head map added."""
        assert self.pv_head_in is not None and self.pv_a is not None and self.pv_c is not None
        lead = functional.one_hot(lead_day, mdl.N_LEAD_DAYS).to(h0.dtype)[:, None, :]
        okf = top_ok.to(h0.dtype)
        p0_mean, p0_sd = top_pv[..., 0] * okf, top_pv[..., 1] * okf
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
        feats = torch.stack([pbar, cover, p0_mean, p0_sd, okf], dim=-1)
        head = self.head
        hidden = head[0](x) + self.pv_head_in(feats)
        out = head[1:](hidden)
        k = mdl.m559.COMPONENTS
        shift = self.pv_a[primary][:, None] * pbar * cover + self.pv_c[primary][:, None] * p0_mean
        return torch.cat(
            [
                out[..., :k],
                out[..., k : 2 * k] + base[..., None] + shift[..., None],
                out[..., 2 * k :],
            ],
            dim=-1,
        )

    def forward(self, inp: PvInputs, *, return_attention: bool = False) -> mdl.MsaOutput:  # ty: ignore[invalid-method-override]
        """Return the raw mixture outputs; an arm without a PV path runs 713's forward unchanged."""
        if not self.pv.any_path:
            return super().forward(inp, return_attention=return_attention)
        inp = self._without_unused_parts(inp)
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
        pbar, cover = self.pv_readout(inp, alpha)
        out = self._mixture_pv(
            h0=h0, base=base, spread=spread, no_history=no_history, lead_day=inp.lead_day,
            primary=inp.primary, pbar=pbar, cover=cover, top_pv=inp.top_pv, top_ok=inp.top_pv_ok,
        )  # fmt: skip
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
        return mdl.MsaOutput(
            out=out, base=base, spread=spread, no_history=no_history, alpha=alpha,
            attention=attention, target_vector=h0.detach() if return_attention else None,
        )  # fmt: skip


def new_parameter_count(*, model: PvNet) -> int:
    """Return the number of new parameters (every `pv_*` parameter, the unused gain included)."""
    return sum(p.numel() for n, p in model.named_parameters() if n.startswith(NEW_PREFIX))
